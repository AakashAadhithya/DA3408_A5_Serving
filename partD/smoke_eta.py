"""
smoke_eta.py -- AIOps Module 4, Part D: release gate. Decides if a freshly started ETA server may take traffic.
Usage:
  python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1 [--max-p95-ms 80]   # Part A
  python partD/smoke_eta.py --url http://127.0.0.1:8100/v1/models/eta:predict --expect-version v2 --beta  # Part C
  python partD/smoke_eta.py --url http://127.0.0.1:5102/invocations --expect-version v2           # Part B (MLflow)
--url may be the server base URL, the full :predict URL, or an MLflow /invocations URL (format auto-detected).
Exit code 0 = every check passed (SKIPs are allowed), 1 = at least one check failed (or the server is down).
Checks: health, version, output sanity + batch consistency, 3 model-sanity monotonicity checks
(distance, traffic, stops), invalid input -> 422, p95 latency budget, request-id echo.
"""
import argparse, json, math, sys, time

import requests

GOOD = {"distance_km": 20.0, "n_stops": 5, "hour": 13, "weekday": 2, "traffic_index": 0.5, "rain": 0, "weight_kg": 8.0}
BASES = [GOOD,
         {"distance_km": 20.0, "n_stops": 12, "hour": 8, "weekday": 4, "traffic_index": 0.3, "rain": 1, "weight_kg": 25.0},
         {"distance_km": 20.0, "n_stops": 3, "hour": 18, "weekday": 0, "traffic_index": 0.7, "rain": 0, "weight_kg": 3.0},
         {"distance_km": 20.0, "n_stops": 8, "hour": 22, "weekday": 6, "traffic_index": 0.2, "rain": 1, "weight_kg": 12.0}]
MAX_PLAUSIBLE_MIN = 24 * 60                                   # an ETA beyond one day is a broken model


class Client:
    def __init__(self, url, customer_id, beta, timeout):
        url = url.rstrip("/")
        if url.endswith("/invocations"):
            self.fmt, self.predict_url, self.base = "mlflow", url, url[: -len("/invocations")]
        elif url.endswith(":predict"):
            self.fmt, self.predict_url, self.base = "rest", url, url.split("/v1/")[0]
        else:
            self.fmt, self.predict_url, self.base = "rest", url + "/v1/models/eta:predict", url
        self.cid, self.beta, self.timeout = customer_id, beta, timeout
        self.s = requests.Session()

    def predict(self, rows, headers=None):
        """-> (status, predictions | None, model_version | None, response, ms)"""
        h = dict(headers or {})
        if self.beta:
            h["X-Beta"] = "1"
        if self.fmt == "mlflow":
            body = {"dataframe_records": rows}
        else:                                                 # Part A ignores customer_id; Part C needs it
            body = {"customer_id": self.cid, "instances": rows}
        t = time.perf_counter()
        r = self.s.post(self.predict_url, json=body, headers=h, timeout=self.timeout)
        ms = (time.perf_counter() - t) * 1000
        preds = ver = None
        if r.status_code == 200:
            j = r.json()
            if self.fmt == "mlflow":
                preds = [p["eta_minutes"] for p in j["predictions"]]
                ver = j["predictions"][0].get("model_version") if j["predictions"] else None
            else:
                preds, ver = j.get("predictions"), j.get("model_version")
        return r.status_code, preds, ver, r, ms


def run_checks(c, expect, max_p95, n_latency):
    results = []

    def record(name, ok, detail):
        results.append((name, "PASS" if ok is True else ("SKIP" if ok is None else "FAIL"), detail))

    def guarded(name, fn):
        try:
            ok, detail = fn()
        except Exception as e:                                # a crash inside a check is a failed check
            ok, detail = False, f"error: {type(e).__name__}: {e}"
        record(name, ok, detail)

    # 1. health / readiness
    def health():
        path = "/ping" if c.fmt == "mlflow" else "/healthz"
        r = c.s.get(c.base + path, timeout=c.timeout)
        try:
            info = r.json() if r.text.strip() else {}
        except ValueError:                                    # MLflow /ping: JSON content-type, empty body
            info = {}
        info = info if isinstance(info, dict) else {}
        hv = info.get("model_version")
        ok = r.status_code == 200 and (hv is None or hv == expect)
        return ok, f"GET {path} -> {r.status_code} {json.dumps(info)[:80] if info else ''}".strip()
    guarded("health endpoint", health)

    # 2. served version
    def version():
        st, preds, ver, _, _ = c.predict([GOOD])
        return st == 200 and ver == expect, f"status {st}, model_version={ver!r}, expected {expect!r}"
    guarded("expected model version", version)

    # 3. output sanity + batch consistency (same rows alone and in a batch -> same numbers, same order)
    def output():
        rows = [{**b, "distance_km": d} for b in BASES for d in (3.0, 12.0, 40.0, 90.0)]
        st, preds, _, _, _ = c.predict(rows)
        if st != 200 or preds is None:
            return False, f"batch status {st}"
        finite = all(isinstance(p, (int, float)) and math.isfinite(p) and 0 < p < MAX_PLAUSIBLE_MIN for p in preds)
        singles = [c.predict([r])[1][0] for r in rows[:4]]
        same = singles == preds[:4]
        return (len(preds) == len(rows) and finite and same,
                f"{len(preds)}/{len(rows)} predictions, all finite in (0, {MAX_PLAUSIBLE_MIN}) min: {finite}; "
                f"batch == single-row for first 4: {same}")
    guarded("output sanity + batch consistency", output)

    # 4-6. model sanity: monotone in the features that can only make a delivery slower
    def monotone(feature, values, fixed=None):
        def chk():
            bad, shown = [], []
            for b in BASES:
                rows = [{**b, **(fixed or {}), feature: v} for v in values]
                st, preds, _, _, _ = c.predict(rows)
                if st != 200:
                    return False, f"status {st}"
                shown.append(preds)
                if any(later < earlier for earlier, later in zip(preds, preds[1:])) or preds[-1] <= preds[0]:
                    bad.append(preds)
            return not bad, f"{feature} {values} -> {shown[0]} (and {len(BASES) - 1} more contexts); violations: {len(bad)}"
        return chk
    guarded("sanity: longer distance never shortens ETA", monotone("distance_km", [5.0, 25.0, 60.0]))
    guarded("sanity: heavier traffic never shortens ETA", monotone("traffic_index", [0.1, 0.9], {"distance_km": 30.0}))
    guarded("sanity: more stops never shortens ETA", monotone("n_stops", [2, 12]))

    # 7. contract: invalid input must be rejected with 422 (MLflow's pyfunc server only checks the schema -> 400)
    def invalid():
        if c.fmt == "mlflow":
            st = c.predict([{k: v for k, v in GOOD.items() if k != "traffic_index"}])[0]
            return st in (400, 422), f"missing column -> {st} (MLflow schema errors are 400; it has no range checks)"
        st1 = c.predict([{**GOOD, "hour": 25}])[0]
        st2 = c.predict([{**GOOD, "distance_km": 200.0, "n_stops": 30}])[0]
        st3 = c.predict([])[0]
        return (st1, st2, st3) == (422, 422, 422), f"hour=25 -> {st1}, infeasible 200km/30 stops -> {st2}, empty batch -> {st3}"
    guarded("invalid input -> 422", invalid)

    # 8. latency budget
    def latency():
        for _ in range(5):
            c.predict([GOOD])                                 # warm-up
        lat = sorted(c.predict([BASES[i % len(BASES)]])[4] for i in range(n_latency))
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        return p95 <= max_p95, f"single-row p95 = {p95:.1f} ms over {n_latency} requests (budget {max_p95} ms)"
    guarded("p95 latency within budget", latency)

    # 9. traceability: request id echoed (only meaningful for the Part A service)
    def request_id():
        st, _, _, r, _ = c.predict([GOOD], headers={"X-Request-ID": "smoke-rid-1"})
        if "x-request-id" not in r.headers:
            return None, "server does not emit X-Request-ID (not part of its contract) -> skipped"
        return r.headers["x-request-id"] == "smoke-rid-1", f"echoed {r.headers['x-request-id']!r}"
    guarded("request id echoed", request_id)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Release-gate smoke test for the SwiftParcel ETA service")
    ap.add_argument("--url", required=True)
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--max-p95-ms", type=float, default=80.0)
    ap.add_argument("--n-latency", type=int, default=50)
    ap.add_argument("--customer-id", default="smoke-test-customer")
    ap.add_argument("--beta", action="store_true", help="send X-Beta: 1 (Ray canary: force the candidate)")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--json-out")
    a = ap.parse_args(argv)

    c = Client(a.url, a.customer_id, a.beta, a.timeout)
    print(f"smoke test -> {c.predict_url}  (format={c.fmt}, expect={a.expect_version}, p95 budget={a.max_p95_ms} ms)")
    try:
        c.s.get(c.base + ("/ping" if c.fmt == "mlflow" else "/healthz"), timeout=a.timeout)
    except requests.RequestException as e:
        print(f"FAIL  server unreachable: {type(e).__name__}"); return 1
    results = run_checks(c, a.expect_version, a.max_p95_ms, a.n_latency)
    width = max(len(n) for n, _, _ in results)
    for name, status, detail in results:
        print(f"{status:4s}  {name:<{width}}  {detail}")
    failed = [n for n, s, _ in results if s == "FAIL"]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks not failing -> "
          f"{'RELEASE ALLOWED (exit 0)' if not failed else 'RELEASE BLOCKED (exit 1): ' + ', '.join(failed)}")
    if a.json_out:
        json.dump([{"check": n, "status": s, "detail": d} for n, s, d in results], open(a.json_out, "w"), indent=2)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
