#!/usr/bin/env bash
# run_all.sh -- regenerate every output of the Module 4 assignment from a clean checkout.
# Usage (from this folder, inside the venv):  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}"
export MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1

echo "== data";   python eta_data.py
python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"

echo "== Part A"; python -m pytest partA -q -p no:warnings
python partA/run_loadtest.py

echo "== Part B"; python -m pytest partB -q -p no:warnings
python partB/eta_mlflow.py register
python partB/eta_mlflow.py gate
python partB/shadow.py

echo "== Part C"; python partC/verify_ray.py

echo "== Part D"; python partD/run_smoke_demo.py
if command -v promtool >/dev/null 2>&1; then
  promtool check config partD/prometheus.yml && promtool test rules partD/abort_rules_test.yml
else
  echo "promtool not on PATH -> skipped Prometheus rule checks"
fi
echo "== done: outputs in partA/*.json, partB/*.txt, partC/verify_output.txt, partD/smoke_output.txt"
