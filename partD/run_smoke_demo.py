"""
run_smoke_demo.py -- runs smoke_eta.py against every server built in Parts A-C, plus two negative controls,
and writes partD/smoke_output.txt. Run from the submission root after Parts A-C are set up:
    python partD/run_smoke_demo.py            (needs data/ and the MLflow registry from partB register)
Scenarios
  1. Part A  v1                         --expect-version v1   -> must exit 0
  2. Part A  v1                         --expect-version v2   -> must exit 1 (wrong version deployed)
  3. Part A  BROKEN 'v2' (constant model, e.g. a training job that silently failed)   -> must exit 1
  4. Part B  mlflow models serve @candidate  --expect-version v2   -> must exit 0
  5. Part C  Ray Serve canary, X-Beta: 1     --expect-version v2   -> must exit 0
"""
import os, subprocess, sys, tempfile, time
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import m4util                                                            # noqa: E402

OUT = HERE / "smoke_output.txt"
PY = sys.executable
LOG = []


def smoke(title, args, expect_rc):
    cmd = [PY, str(HERE / "smoke_eta.py"), *args]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    ok = r.returncode == expect_rc
    block = (f"=== {title}\n$ python partD/smoke_eta.py {' '.join(args)}\n{r.stdout.strip()}\n"
             f"exit code {r.returncode} (expected {expect_rc}) -> {'as expected' if ok else 'UNEXPECTED'}\n")
    print(block); LOG.append(block)
    return ok


def uvicorn(port, model_path):
    p = m4util.start(f"{PY} -m uvicorn --app-dir partA eta_app:app --host 127.0.0.1 --port {port} --log-level warning",
                     log=str(HERE / f"uvicorn_{port}.log"), env={"ETA_MODEL": str(model_path)})
    m4util.wait_ready(f"http://127.0.0.1:{port}/healthz")
    return p


def main():
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        from sklearn.dummy import DummyRegressor
        import pandas as pd
        import eta_data
        tr = pd.read_csv(ROOT / "data" / "eta_train.csv")
        broken = DummyRegressor(strategy="mean").fit(tr[eta_data.FEATURES], tr.eta_minutes)
        bad_path = Path(tmp) / "eta_v2_broken.joblib"
        joblib.dump({"model": broken, "feature_columns": eta_data.FEATURES, "version": "v2"}, bad_path)

        a = uvicorn(8000, "data/eta_v1.joblib")
        try:
            results.append(smoke("1. Part A, v1 deployed, expecting v1", ["--url", "http://127.0.0.1:8000",
                                                                           "--expect-version", "v1"], 0))
            results.append(smoke("2. Part A, v1 deployed, expecting v2 (negative control)",
                                 ["--url", "http://127.0.0.1:8000/v1/models/eta:predict", "--expect-version", "v2"], 1))
        finally:
            m4util.stop(a)
        b = uvicorn(8001, bad_path)
        try:
            results.append(smoke("3. Part A, BROKEN constant-output 'v2' (negative control)",
                                 ["--url", "http://127.0.0.1:8001", "--expect-version", "v2"], 1))
        finally:
            m4util.stop(b)

    sys.path.insert(0, str(ROOT / "partB"))
    import eta_mlflow
    m = m4util.start(f'mlflow models serve -m "models:/eta-model@candidate" -p 5102 -h 127.0.0.1 -w 1 --env-manager local',
                     log=str(HERE / "mlflow_5102.log"), env={"MLFLOW_TRACKING_URI": eta_mlflow.tracking_uri()})
    try:
        m4util.wait_ready("http://127.0.0.1:5102/ping", timeout=180)
        results.append(smoke("4. Part B, mlflow models serve @candidate, expecting v2",
                             ["--url", "http://127.0.0.1:5102/invocations", "--expect-version", "v2"], 0))
    finally:
        m4util.stop(m)

    ray_script = (f"import sys,time; sys.path.insert(0, {str(ROOT / 'partC')!r}); import ray, eta_ray; from ray import serve; "
                  "ray.init(address='local', num_cpus=2, include_dashboard=False, logging_level='ERROR', log_to_driver=False); "
                  "serve.start(http_options={'host':'127.0.0.1','port':8100}); "
                  "serve.run(eta_ray.build(0.1), name='eta', route_prefix='/'); print('up', flush=True); "
                  "time.sleep(3600)")
    rp = m4util.start([PY, "-c", ray_script], log=str(HERE / "ray_8100.log"))
    try:
        m4util.wait_ready("http://127.0.0.1:8100/healthz", timeout=180)
        results.append(smoke("5. Part C, Ray Serve sticky canary with X-Beta: 1, expecting v2",
                             ["--url", "http://127.0.0.1:8100/v1/models/eta:predict", "--expect-version", "v2",
                              "--beta"], 0))
    finally:
        m4util.stop(rp)
        subprocess.run(["ray", "stop", "--force"], capture_output=True)

    summary = f"scenarios behaving as expected: {sum(results)}/{len(results)}"
    print(summary); LOG.append(summary)
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        OUT.write_text("\n".join(LOG) + "\n")
        print("written", OUT)
    sys.exit(rc)
