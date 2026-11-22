"""
run_loadtest.py -- Part A3 driver: starts eta_app with v1, runs the given load_test.py and writes JSON results.
Run from the submission root (data/ must exist):   python partA/run_loadtest.py
Writes (into partA/):
  loadtest_results.json   concurrency sweep 1,2,4,8,16 x 4 s, batch 1, 1 uvicorn worker      (A3.1)
  loadtest_slo_probe.json  extra points at 24/32/48 clients to locate the p95 <= 80 ms boundary (A3.2)
  loadtest_batch.json     batch 1 vs batch 100 at concurrency 1 -> ms/row                    (A3.3)
  loadtest_change.json    same sweep after the capacity change (2 uvicorn worker processes)  (A3.4)
  loadtest_model_only.json  in-process model-only timing, to split compute from per-request overhead
"""
import json, os, sys, time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import load_test, m4util                                            # noqa: E402  (given generator + helpers)

PORT = int(os.getenv("PORT", "8000"))
URL = f"http://127.0.0.1:{PORT}/v1/models/eta:predict"
DATA = m4util.repo_path("data/eta_sample.csv")
CONC = [1, 2, 4, 8, 16]
DUR = float(os.getenv("DURATION", "4"))


def serve(workers):
    cmd = (f"{sys.executable} -m uvicorn --app-dir partA eta_app:app --host 127.0.0.1 --port {PORT} "
           f"--workers {workers} --log-level warning --no-access-log")
    p = m4util.start(cmd, log=str(HERE / f"uvicorn_w{workers}.log"), env={"ETA_MODEL": "data/eta_v1.joblib"})
    m4util.wait_ready(f"http://127.0.0.1:{PORT}/healthz")
    return p


def sweep(rows, conc, batch=1):
    load_test.run(URL, "rest", 2, 1.5, rows, batch)                     # warm-up (not recorded)
    return [load_test.run(URL, "rest", c, DUR, rows, batch) for c in conc]


def little(df):
    """Little's Law check: L = lambda * W  ->  implied concurrency = rps * mean-ish latency (p50 as proxy)."""
    df = df.copy()
    df["littles_L=rps*p50"] = (df.rps * df.p50_ms / 1000).round(2)
    return df


def model_only(rows):
    import joblib, numpy as np
    b = joblib.load(m4util.repo_path("data/eta_v1.joblib"))
    out = []
    for bs in (1, 100):
        X = pd.DataFrame(rows[:bs])[b["feature_columns"]]
        ts = []
        for _ in range(300):
            t = time.perf_counter(); b["model"].predict(X); ts.append((time.perf_counter() - t) * 1000)
        out.append({"batch": bs, "model_only_median_ms": round(float(np.median(ts)), 3)})
    return out


def main():
    rows = pd.read_csv(DATA).to_dict(orient="records")
    cols = ["concurrency", "requests", "rps", "p50_ms", "p95_ms", "p99_ms", "errors"]

    srv = serve(1)
    try:
        base = sweep(rows, CONC)
        json.dump(base, open(HERE / "loadtest_results.json", "w"), indent=2)
        print("== A3.1  v1, 1 worker, batch 1 ==\n", little(pd.DataFrame(base)[cols]).to_string(index=False))

        probe = sweep(rows, [24, 32, 48])                              # where does p95 cross the 80 ms SLO?
        json.dump(probe, open(HERE / "loadtest_slo_probe.json", "w"), indent=2)
        print("\n== A3.2  SLO probe beyond 16 clients ==\n", little(pd.DataFrame(probe)[cols]).to_string(index=False))

        batch = []
        for bs in (1, 100):
            r = sweep(rows, [1], bs)[0]
            r.update(batch=bs, ms_per_row=round(r["p50_ms"] / bs, 3), rows_per_s=round(r["rps"] * bs, 1))
            batch.append(r)
        json.dump(batch, open(HERE / "loadtest_batch.json", "w"), indent=2)
        print("\n== A3.3  batch 1 vs 100, concurrency 1 ==\n",
              pd.DataFrame(batch)[["batch", "requests", "rps", "p50_ms", "p95_ms", "ms_per_row", "rows_per_s"]]
              .to_string(index=False))
    finally:
        m4util.stop(srv)
    time.sleep(1)

    srv = serve(2)
    try:
        change = sweep(rows, CONC)
        json.dump(change, open(HERE / "loadtest_change.json", "w"), indent=2)
        print("\n== A3.4  v1, 2 uvicorn workers, batch 1 ==\n",
              little(pd.DataFrame(change)[cols]).to_string(index=False))
    finally:
        m4util.stop(srv)

    mo = model_only(rows)
    json.dump(mo, open(HERE / "loadtest_model_only.json", "w"), indent=2)
    print("\n== model-only (in-process) ==\n", pd.DataFrame(mo).to_string(index=False))
    print(f"\nmachine: {os.cpu_count()} logical CPUs")


if __name__ == "__main__":
    main()
