# Module 4 assignment — SwiftParcel ETA service (DA24B028)

Serve a delivery-time (ETA) regressor four ways: a hand-written REST API (A), MLflow registry + gate + shadow (B),
a Ray Serve sticky canary (C) and a release gate (D). All tables and written answers are in `REPORT.md`.

## Layout
```
Module4_DA24B028/
├── REPORT.md                  tables, written answers, machine, NOT-tested list
├── AI_DISCLOSURE.md           AI-tool disclosure required by the course code of conduct
├── README.md                  this file
├── requirements.txt           given (unchanged)
├── eta_data.py                given: data + model bundles -> data/
├── load_test.py               given: closed-loop load generator
├── m4util.py                  start / wait / stop helper (adapted from the lecture notebooks)
├── run_all.sh                 regenerate every output file in one go
├── partA/  eta_app.py  test_eta_app.py  run_loadtest.py
│           loadtest_results.json  loadtest_slo_probe.json  loadtest_batch.json  loadtest_change.json  loadtest_model_only.json
├── partB/  eta_mlflow.py  test_gate.py  shadow.py  MLproject  python_env.yaml
│           gate_output.txt  shadow_output.txt  registry_table.txt
├── partC/  eta_ray.py  verify_ray.py  serve_config.yaml  verify_output.txt
└── partD/  smoke_eta.py  run_smoke_demo.py  smoke_output.txt
            prometheus.yml  abort_rules.yml  abort_rules_test.yml
```

## Setup (from this folder; Python 3.10+, tested on 3.12)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python eta_data.py                                   # -> data/eta_{train,fresh}.csv, data/eta_v{1,2}.joblib
python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db       # relative sqlite paths are anchored at this folder
```
**Run every command from this folder** (the scripts also find `data/` if run from inside a part folder).

## Part A — REST service
```bash
pytest partA -q                                                      # 27 cases
ETA_MODEL=data/eta_v1.joblib uvicorn --app-dir partA eta_app:app --port 8000
curl -s -X POST localhost:8000/v1/models/eta:predict -H 'Content-Type: application/json' \
     -d '{"instances":[{"distance_km":12,"n_stops":3,"hour":8,"weekday":1,"traffic_index":0.5,"rain":0,"weight_kg":4}]}'
curl -s localhost:8000/metrics
python partA/run_loadtest.py      # A3: starts/stops its own servers on :8000, writes partA/loadtest_*.json (~2.5 min)
```

## Part B — MLflow
```bash
python partB/eta_mlflow.py register     # log v1+v2, register `eta-model`, aliases prod->v1, candidate->v2
python partB/eta_mlflow.py gate         # gate with defaults and with min_rel_gain=0.60 -> partB/gate_output.txt
pytest partB -q                         # 21 cases for decide()
python partB/shadow.py                  # serves @prod on :5101 and @candidate on :5102 -> partB/shadow_output.txt
mlflow ui --backend-store-uri sqlite:///mlflow.db          # optional: browse runs and the registry
# same steps as an MLflow Project (course style):
mlflow run partB -e gate -P min_rel_gain=0.60 --env-manager local --experiment-name eta-serving
```
Running `register` again creates versions 3 and 4 and moves both aliases to them (same behaviour as the lecture).

## Part C — Ray Serve sticky canary
```bash
python partC/verify_ray.py              # 2-CPU Ray, five assertions + autoscaling demo -> partC/verify_output.txt (~2 min)
python partC/eta_ray.py                 # or: run the app on :8100 and keep it up
cd partC && serve run serve_config.yaml # or: the declarative YAML deployment
curl -s -X POST localhost:8100/v1/models/eta:predict -H 'Content-Type: application/json' -H 'X-Beta: 1' \
     -d '{"customer_id":"cust-0001","instances":[{"distance_km":12,"n_stops":3,"hour":8,"weekday":1,"traffic_index":0.5,"rain":0,"weight_kg":4}]}'
```

## Part D — release gate
```bash
python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1 --max-p95-ms 80   # against a running server
python partD/run_smoke_demo.py          # all servers + 2 negative controls -> partD/smoke_output.txt (needs partB register)
promtool check config partD/prometheus.yml && promtool test rules partD/abort_rules_test.yml   # if Prometheus is installed
```

## Everything at once
```bash
bash run_all.sh                         # ~10 min on 2 cores; regenerates every output file listed above
```

## Interface contract (§2.1) → where it lives
| Part | Required | Here |
|---|---|---|
| A | `eta_app.py`: `app`, `ETA_MODEL` (default `data/eta_v1.joblib`), module-level `model` | `partA/eta_app.py` |
| A | `test_eta_app.py` | `partA/test_eta_app.py` |
| B | `register()`, `decide(prod, cand, **thresholds)`, `promotion_gate(**thresholds)`, `MLFLOW_TRACKING_URI` | `partB/eta_mlflow.py` |
| B | `test_gate.py` | `partB/test_gate.py` |
| C | `build(weight=0.1, extra_work_ms=0.0)`, `POST /v1/models/eta:predict` | `partC/eta_ray.py` |
| C | `verify_ray.py` | `partC/verify_ray.py` |
| D | `smoke_eta.py --url --expect-version --max-p95-ms` | `partD/smoke_eta.py` |

Ports used: 8000/8001 (Part A), 5101/5102 (MLflow), 8100 (Ray Serve), 9090 (Prometheus).
