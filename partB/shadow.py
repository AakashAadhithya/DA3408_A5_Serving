"""
shadow.py -- AIOps Module 4, Part B3: shadow comparison of prod vs candidate served by `mlflow models serve`.
  prod      -> models:/eta-model@prod       on :5101
  candidate -> models:/eta-model@candidate  on :5102
Sends the SAME 200 single-row requests (data/eta_fresh.csv) to both, and reports the mean |diff| (minutes),
% of requests differing by > 10 min, and each server's p50/p95 latency. Then a short alias-resolution demo:
move `prod` to the candidate while :5101 runs (it keeps answering v1), restart it (now v2), and roll back.
Run from the submission root, after `python partB/eta_mlflow.py register`:   python partB/shadow.py
Shadow-test idea from the Task B solution in "Lecture 10 -- MLflow_Serving_Exercise.ipynb" (DA3408 repo).
"""
import os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import m4util                                                            # noqa: E402
import eta_mlflow                                                        # noqa: E402

PORTS = {"prod": 5101, "candidate": 5102}
N, DIFF_MIN = 200, 10.0
OUT = HERE / "shadow_output.txt"


def serve(alias, port):
    cmd = f'mlflow models serve -m "models:/{eta_mlflow.NAME}@{alias}" -p {port} -h 127.0.0.1 -w 1 --env-manager local'
    p = m4util.start(cmd, log=str(HERE / f"mlflow_serve_{port}.log"),
                     env={"MLFLOW_TRACKING_URI": eta_mlflow.tracking_uri(), "MLFLOW_DISABLE_TELEMETRY": "true"})
    m4util.wait_ready(f"http://127.0.0.1:{port}/ping", timeout=180)
    return p


def ask(sess, port, rec):
    t = time.perf_counter()
    r = sess.post(f"http://127.0.0.1:{port}/invocations", json={"dataframe_records": [rec]}, timeout=10)
    ms = (time.perf_counter() - t) * 1000
    r.raise_for_status()
    p = r.json()["predictions"][0]
    return p["eta_minutes"], p["model_version"], ms


def main():
    lines = []
    def say(s=""):
        print(s); lines.append(s)

    client = eta_mlflow._setup()
    fresh = pd.read_csv(m4util.repo_path("data/eta_fresh.csv")).head(N)
    recs = fresh[eta_mlflow.FEATURES].to_dict(orient="records")
    procs = {a: serve(a, port) for a, port in PORTS.items()}
    try:
        sess = requests.Session()
        for port in PORTS.values():                                      # warm-up, not measured
            for rec in recs[:10]:
                ask(sess, port, rec)
        res = {a: {"pred": [], "ms": [], "ver": set()} for a in PORTS}
        for i, rec in enumerate(recs):
            order = list(PORTS) if i % 2 == 0 else list(PORTS)[::-1]     # alternate who goes first (fairness)
            for a in order:
                y, v, ms = ask(sess, PORTS[a], rec)
                res[a]["pred"].append(y); res[a]["ms"].append(ms); res[a]["ver"].add(v)

        p, c = np.array(res["prod"]["pred"]), np.array(res["candidate"]["pred"])
        d = np.abs(p - c)
        truth = fresh["eta_minutes"].to_numpy()
        say(f"# Shadow comparison: {N} identical single-row requests from data/eta_fresh.csv")
        say(f"machine: {os.cpu_count()} logical CPUs; servers: mlflow models serve -w 1 --env-manager local")
        say(f"prod      :5101 -> models:/eta-model@prod      (registry v{client.get_model_version_by_alias('eta-model','prod').version}) answered as {sorted(res['prod']['ver'])}")
        say(f"candidate :5102 -> models:/eta-model@candidate (registry v{client.get_model_version_by_alias('eta-model','candidate').version}) answered as {sorted(res['candidate']['ver'])}")
        say()
        say(f"(i)   mean |prod - candidate|           = {d.mean():.2f} min   (median {np.median(d):.2f}, max {d.max():.1f})")
        say(f"(ii)  requests differing by > {DIFF_MIN:.0f} min    = {(d > DIFF_MIN).mean():.1%}  ({int((d > DIFF_MIN).sum())}/{N})")
        for a in PORTS:
            ms = np.array(res[a]["ms"])
            say(f"(iii) {a:9s} latency p50 / p95 / p99 = {np.percentile(ms, 50):.2f} / {np.percentile(ms, 95):.2f} / "
                f"{np.percentile(ms, 99):.2f} ms")
        say()
        say("bonus (ground truth is available for these rows, i.e. 'delayed labels' arrived):")
        say(f"      MAE prod = {np.abs(p - truth).mean():.2f} min, MAE candidate = {np.abs(c - truth).mean():.2f} min")
        say(f"      candidate closer to truth on {(np.abs(c - truth) < np.abs(p - truth)).mean():.0%} of requests")
        big = d > DIFF_MIN
        if big.any():
            sub = fresh[big]
            say(f"      the >{DIFF_MIN:.0f}-min disagreements: rush-hour share {sub.hour.isin(eta_mlflow.RUSH_HOURS).mean():.0%} "
                f"(vs {fresh.hour.isin(eta_mlflow.RUSH_HOURS).mean():.0%} overall), mean distance {sub.distance_km.mean():.1f} km "
                f"(vs {fresh.distance_km.mean():.1f} km)")

        # ---- alias resolution demo -----------------------------------------------------------------------
        say(); say("# Alias resolution demo (registry is restored afterwards)")
        prod_v = client.get_model_version_by_alias("eta-model", "prod").version
        cand_v = client.get_model_version_by_alias("eta-model", "candidate").version
        client.set_registered_model_alias("eta-model", "prod", cand_v)
        say(f"moved alias prod -> registry v{cand_v};  running :5101 still answers: {ask(sess, 5101, recs[0])[1]}")
        m4util.stop(procs["prod"]); procs["prod"] = serve("prod", 5101)
        say(f"after restarting :5101 (alias re-resolved at start-up):  {ask(sess, 5101, recs[0])[1]}")
        client.set_registered_model_alias("eta-model", "prod", prod_v)
        m4util.stop(procs["prod"]); procs["prod"] = serve("prod", 5101)
        say(f"rollback: alias prod -> registry v{prod_v}, restart:       {ask(sess, 5101, recs[0])[1]}")
    finally:
        m4util.stop_all()
    OUT.write_text("\n".join(lines) + "\n")
    print("written", OUT)


if __name__ == "__main__":
    main()
