"""
verify_ray.py -- AIOps Module 4, Part C2: deploy eta_ray on a 2-CPU Ray and PROVE the sticky canary works.
Checks (each one an assertion + printed evidence), 1000 distinct customer ids, weight = 0.2:
  (a) share routed to the candidate is within +/-4 points of 0.2
  (b) 200 customers get the same arm on a second call
  (c) the same holds after deleting and redeploying the identical app (fresh router + replicas)
  (d) after raising the weight 0.2 -> 0.5, no customer who was on the candidate moves back to prod
  (e) the X-Beta: 1 header override works for 50 requests (prod customers, then back to prod without it)
  plus: fractional-CPU budget <= 2 CPUs and the candidate's autoscaling config, read back from Serve;
  optional (--autoscale-demo, on by default): load the candidate and watch it scale 1 -> up to 3 replicas.
Run from the submission root:   python partC/verify_ray.py        (writes partC/verify_output.txt)
"""
import argparse, collections, concurrent.futures as cf, json, os, sys, threading, time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eta_ray                                                           # noqa: E402

PORT = 8100
BASE = f"http://127.0.0.1:{PORT}"
URL = BASE + eta_ray.PREDICT_PATH
APP = "eta"
IDS = [f"cust-{i:04d}" for i in range(1000)]
LINES = []
_tls = threading.local()


def say(s=""):
    print(s, flush=True); LINES.append(str(s))


def check(name, cond, evidence):
    say(f"[{'PASS' if cond else 'FAIL'}] {name}: {evidence}")
    assert cond, f"{name} failed: {evidence}"


def sess():
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
    return _tls.s


def call(cid, row, beta=False):
    h = {"X-Beta": "1"} if beta else {}
    r = sess().post(URL, json={"customer_id": cid, "instances": [row]}, headers=h, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j["arm"], j["model_version"], j["served_by"]


def route_all(ids, rows, beta=False):
    """customer_id -> (arm, model_version, replica), 8 concurrent clients."""
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda i: call(ids[i], rows[i % len(rows)], beta), range(len(ids))))
    return dict(zip(ids, res))


def deploy(weight, extra_work_ms=0.0):
    from ray import serve
    t = time.time()
    serve.run(eta_ray.build(weight, extra_work_ms), name=APP, route_prefix="/")
    hz = requests.get(BASE + "/healthz", timeout=10).json()
    assert abs(hz["canary_weight"] - weight) < 1e-9, hz
    return round(time.time() - t, 1)


def replica_counts():
    from ray import serve
    st = serve.status().applications[APP].deployments
    return {n: sum(d.replica_states.values()) for n, d in st.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-autoscale-demo", action="store_true")
    a = ap.parse_args()

    import ray
    from ray import serve
    ray.init(address="local", num_cpus=2, include_dashboard=False, ignore_reinit_error=True, logging_level="ERROR",
             log_to_driver=False)
    serve.start(http_options={"host": "127.0.0.1", "port": PORT}, logging_config={"log_level": "WARNING"})
    rows = pd.read_csv(eta_ray.ROOT / "data" / "eta_sample.csv").to_dict(orient="records")
    say(f"# verify_ray.py  |  Ray {ray.__version__}  |  cluster CPUs = {ray.cluster_resources().get('CPU')}  "
        f"|  host logical CPUs = {os.cpu_count()}")

    try:
        # ---------------------------------------------------------------- deploy + configuration evidence
        say(f"deploy(weight=0.2) ready in {deploy(0.2)} s")
        st = serve.status().applications[APP].deployments
        say("deployments: " + json.dumps({n: dict(d.replica_states) for n, d in st.items()}, default=str))
        budget = eta_ray.ROUTER_CPUS + eta_ray.PROD_REPLICAS * eta_ray.MODEL_CPUS \
            + eta_ray.CAND_AUTOSCALING["max_replicas"] * eta_ray.MODEL_CPUS
        used = ray.cluster_resources()["CPU"] - ray.available_resources().get("CPU", 0.0)
        check("fractional CPU budget", budget <= 2.0 and used <= 2.0,
              f"worst case router {eta_ray.ROUTER_CPUS} + prod {eta_ray.PROD_REPLICAS}x{eta_ray.MODEL_CPUS} + "
              f"candidate max {eta_ray.CAND_AUTOSCALING['max_replicas']}x{eta_ray.MODEL_CPUS} = {budget} CPU; "
              f"reserved now = {used:.2f} of 2 CPU")
        say(f"candidate autoscaling_config: {eta_ray.CAND_AUTOSCALING}")
        say()

        # ---------------------------------------------------------------- (a) share
        first = route_all(IDS, rows)
        arms = {c: v[0] for c, v in first.items()}
        share = sum(a_ == "v2" for a_ in arms.values()) / len(IDS)
        versions_ok = all(v[1] == ("v2" if v[0] == "v2" else "v1") for v in first.values())
        agree = sum(arms[c] == eta_ray.choose_arm(c, 0.2) for c in IDS)
        check("(a) candidate share", abs(share - 0.2) <= 0.04 and versions_ok,
              f"{share:.3f} of 1000 customers on v2 (target 0.20 +/- 0.04); arm/model_version consistent: "
              f"{versions_ok}; matches local choose_arm() for {agree}/1000")
        reps = collections.defaultdict(set)
        for arm_, _, rep in first.values():
            reps[arm_].add(rep)
        say(f"    replicas that answered: v1 -> {len(reps['v1'])}, v2 -> {len(reps['v2'])}")

        # ---------------------------------------------------------------- (b) stickiness on a second call
        sub = IDS[:200]
        other_rows = rows[::-1]                                          # different payload, same customer
        second = route_all(sub, other_rows)
        same = sum(second[c][0] == arms[c] for c in sub)
        check("(b) sticky on 2nd call", same == 200,
              f"{same}/200 customers got the same arm (with a different delivery in the payload); "
              f"{sum(arms[c] == 'v2' for c in sub)} of them are on v2")

        # ---------------------------------------------------------------- (c) after redeploying identical app
        old_reps = {v[2] for v in first.values()}
        serve.delete(APP); time.sleep(2)
        say(f"    deleted app '{APP}', redeployed identical app in {deploy(0.2)} s")
        third = route_all(sub, rows)
        same = sum(third[c][0] == arms[c] for c in sub)
        new_reps = {v[2] for v in third.values()}
        check("(c) sticky after redeploy", same == 200,
              f"{same}/200 same arm; answering replicas are all new: {old_reps.isdisjoint(new_reps)} "
              f"({len(new_reps)} replica ids, none seen before)")

        # ---------------------------------------------------------------- (d) raise weight 0.2 -> 0.5
        say(f"    raised weight to 0.5 (in-place update) in {deploy(0.5)} s")
        fourth = route_all(IDS, rows)
        arms5 = {c: v[0] for c, v in fourth.items()}
        back = [c for c in IDS if arms[c] == "v2" and arms5[c] == "v1"]
        moved_up = sum(arms[c] == "v1" and arms5[c] == "v2" for c in IDS)
        share5 = sum(x == "v2" for x in arms5.values()) / len(IDS)
        check("(d) monotone weight increase", not back,
              f"{sum(x == 'v2' for x in arms.values())} customers were on v2 at 0.2 -> {len(back)} moved back to v1; "
              f"{moved_up} moved v1 -> v2; new share {share5:.3f}")

        # ---------------------------------------------------------------- (e) header override
        prod_customers = [c for c in IDS if arms5[c] == "v1"][:50]
        forced = route_all(prod_customers, rows, beta=True)
        n_forced = sum(v[0] == "v2" and v[1] == "v2" for v in forced.values())
        after = route_all(prod_customers, rows)
        n_back = sum(v[0] == "v1" for v in after.values())
        check("(e) X-Beta: 1 override", n_forced == 50 and n_back == 50,
              f"{n_forced}/50 prod customers served by v2 with X-Beta: 1; without the header {n_back}/50 are "
              f"back on v1 (override is per request, not sticky)")

        # ---------------------------------------------------------------- optional: autoscaling under load
        if not a.no_autoscale_demo:
            say()
            say("# autoscaling demo: weight=1.0, extra_work_ms=50 (simulated heavier model), 12 clients for 25 s")
            serve.delete(APP); time.sleep(2)
            deploy(1.0, extra_work_ms=50)
            timeline, stop = [], threading.Event()

            def poll():
                t0 = time.time()
                while not stop.is_set():
                    timeline.append((round(time.time() - t0), replica_counts().get("EtaModelV2", 0)))
                    time.sleep(1)

            th = threading.Thread(target=poll, daemon=True); th.start()
            end = time.time() + 25
            def hammer(k):
                n = 0
                while time.time() < end:
                    call(f"load-{k}-{n}", rows[n % len(rows)]); n += 1
                return n
            with cf.ThreadPoolExecutor(12) as ex:
                total = sum(ex.map(hammer, range(12)))
            time.sleep(3); stop.set(); th.join()
            peak = max(n for _, n in timeline)
            say(f"    {total} requests in 25 s; candidate replicas over time (t_s, n): "
                f"{timeline[::3]}")
            say(f"    [{'PASS' if peak > 1 else 'INFO'}] candidate scaled from 1 to a peak of {peak} replicas "
                f"(max_replicas={eta_ray.CAND_AUTOSCALING['max_replicas']}); scale-down follows after "
                f"downscale_delay_s={eta_ray.CAND_AUTOSCALING['downscale_delay_s']} s")
        say()
        say("ALL FIVE CHECKS PASSED")
    finally:
        (HERE / "verify_output.txt").write_text("\n".join(LINES) + "\n")
        serve.shutdown(); ray.shutdown()


if __name__ == "__main__":
    main()
