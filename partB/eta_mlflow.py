"""
eta_mlflow.py -- AIOps Module 4, Part B: package both ETA bundles as an MLflow pyfunc, register them as
`eta-model` with aliases  prod -> v1,  candidate -> v2,  and gate promotion on fresh data.
Tracking store: MLFLOW_TRACKING_URI (default sqlite:///mlflow.db). A *relative* sqlite path is anchored at the
submission root so `python partB/eta_mlflow.py`, `mlflow run partB` and `mlflow models serve` all see one DB.

CLI (run from the submission root, after `python eta_data.py`):
    python partB/eta_mlflow.py register                    # log + register v1, v2; set aliases; print table
    python partB/eta_mlflow.py gate                        # gate with default thresholds and min_rel_gain=0.60
    python partB/eta_mlflow.py gate --min-rel-gain 0.1     # any threshold override
Pattern adapted from `mlflow_register.py` and the promotion-gate solution in
"Lecture 10 -- MLflow_Serving_Exercise.ipynb" (DA3408 repo); decide() with slices is new.
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parent.parent
NAME = "eta-model"
EXPERIMENT = "eta-serving"
FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"]
RUSH_HOURS = (7, 8, 9, 16, 17, 18, 19)
METRIC_KEYS = ("mae", "mae_rush", "mae_rain", "p95_ms")


# ----------------------------------------------------------------------------------------------- plumbing
def tracking_uri() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{(ROOT / uri[len('sqlite:///'):]).resolve()}"      # anchor relative DB at the root
    return uri


def _setup() -> MlflowClient:
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true"); os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return MlflowClient()


def _bundle_path(version: str) -> str:
    p = Path(f"data/eta_{version}.joblib")
    return str(p if p.exists() else ROOT / p)


# ----------------------------------------------------------------------------------------------- B1: pyfunc
class EtaPyfunc(mlflow.pyfunc.PythonModel):
    """Wraps a joblib bundle {model, feature_columns, version}; returns eta_minutes (1 dp) + model_version."""

    def load_context(self, context):
        b = joblib.load(context.artifacts["bundle"])
        self.model, self.cols, self.version = b["model"], list(b["feature_columns"]), b["version"]

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)[self.cols].astype("float64")       # column order from the bundle
        y = np.round(np.asarray(self.model.predict(X), dtype=float), 1)
        return pd.DataFrame({"eta_minutes": y, "model_version": [self.version] * len(y)})


def _pinned_requirements():
    import cloudpickle, sklearn
    return [f"mlflow=={mlflow.__version__}", f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}", f"joblib=={joblib.__version__}", f"cloudpickle=={cloudpickle.__version__}"]


def registry_table(client=None) -> pd.DataFrame:
    client = client or _setup()
    aliases = client.get_registered_model(NAME).aliases                  # {alias: version}
    rows = []
    for v in sorted(client.search_model_versions(f"name='{NAME}'"), key=lambda v: int(v.version)):
        tags = client.get_run(v.run_id).data.params if v.run_id else {}
        rows.append({"version": int(v.version), "bundle": tags.get("bundle_version", "?"),
                     "algorithm": tags.get("algorithm", "?"),
                     "aliases": ",".join(a for a, ver in aliases.items() if str(ver) == str(v.version)) or "-",
                     "run_id": (v.run_id or "")[:8]})
    return pd.DataFrame(rows)


def register():
    """Log v1 and v2 as pyfunc models (signature + input example + pinned pip requirements), register them
    as `eta-model`, and point aliases prod -> v1, candidate -> v2. Returns the registry table."""
    from mlflow.models import infer_signature
    from sklearn.metrics import mean_absolute_error

    client = _setup()
    train = pd.read_csv(ROOT / "data" / "eta_train.csv")
    Xs = train[FEATURES].astype("float64")                               # float schema: tolerant to JSON ints
    out_example = pd.DataFrame({"eta_minutes": [31.4], "model_version": ["v1"]})
    sig = infer_signature(Xs.head(5), out_example)                       # input/output schema = contract
    reqs = _pinned_requirements()
    new_versions = {}
    with mlflow.start_run(run_name="register-eta-model"):                # parent (resumes an `mlflow run`)
        for ver in ("v1", "v2"):
            path = _bundle_path(ver)
            b = joblib.load(path)
            mae = mean_absolute_error(train.eta_minutes, b["model"].predict(train[b["feature_columns"]]))
            with mlflow.start_run(run_name=f"eta-{ver}", nested=True):
                mlflow.log_params({"bundle_version": ver, "algorithm": type(b["model"]).__name__,
                                   "n_features": len(b["feature_columns"])})
                mlflow.log_metric("train_mae", mae)
                info = mlflow.pyfunc.log_model(
                    name="model", python_model=EtaPyfunc(), artifacts={"bundle": path},
                    code_paths=[__file__],                               # ship this module with the model
                    signature=sig, input_example=Xs.head(2), pip_requirements=reqs,
                    registered_model_name=NAME)
            new_versions[ver] = info.registered_model_version
            print(f"{ver}: train MAE {mae:.2f} -> {NAME} version {info.registered_model_version}")
    client.set_registered_model_alias(NAME, "prod", new_versions["v1"])
    client.set_registered_model_alias(NAME, "candidate", new_versions["v2"])
    client.update_registered_model(NAME, description="SwiftParcel delivery-time (minutes) regressor")
    return registry_table(client)


# ----------------------------------------------------------------------------------------------- B2: gate
def _at_least(value, threshold):
    return value >= threshold or math.isclose(value, threshold, rel_tol=1e-9, abs_tol=1e-12)


def _at_most(value, threshold):
    return value <= threshold or math.isclose(value, threshold, rel_tol=1e-9, abs_tol=1e-12)


def _rel_increase(prod_v, cand_v):
    """(cand - prod) / prod; a positive number is a regression (MAE got worse)."""
    if prod_v == 0:
        return 0.0 if cand_v == 0 else math.inf
    return (cand_v - prod_v) / prod_v


def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    """PURE promotion decision. prod/cand: dicts with mae, mae_rush, mae_rain, p95_ms.
    Promote iff ALL hold:
      * overall MAE improves by >= min_rel_gain, relative to prod:   (prod.mae - cand.mae) / prod.mae
      * cand single-row p95 latency <= max_p95_ms
      * no slice regresses by more than max_slice_regression, relative to prod's slice MAE
        (rush hours 7-9 & 16-19, and rain == 1):                     cand <= prod * (1 + max_slice_regression)
    Boundaries are inclusive ('at least' / 'not more than'), with a 1e-9 relative float tolerance."""
    for who, d in (("prod", prod), ("cand", cand)):
        missing = [k for k in METRIC_KEYS if k not in d]
        if missing:
            raise KeyError(f"{who} metrics missing {missing}")
        for k in METRIC_KEYS:
            v = float(d[k])
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{who}.{k} must be a finite non-negative number, got {d[k]!r}")
    p = {k: float(prod[k]) for k in METRIC_KEYS}
    c = {k: float(cand[k]) for k in METRIC_KEYS}

    gain = -_rel_increase(p["mae"], c["mae"])
    criteria = {
        "overall_mae_gain": {"prod": p["mae"], "cand": c["mae"], "value": gain,
                             "threshold": f">= {min_rel_gain}", "pass": _at_least(gain, min_rel_gain)},
        "candidate_p95_ms": {"value": c["p95_ms"], "threshold": f"<= {max_p95_ms}",
                             "pass": _at_most(c["p95_ms"], max_p95_ms)},
    }
    for key, label in (("mae_rush", "rush_hour_slice_regression"), ("mae_rain", "rain_slice_regression")):
        reg = _rel_increase(p[key], c[key])
        criteria[label] = {"prod": p[key], "cand": c[key], "value": reg, "threshold": f"<= {max_slice_regression}",
                           "pass": _at_most(reg, max_slice_regression)}
    for v in criteria.values():
        v["pass"] = bool(v["pass"])
        v["value"] = round(v["value"], 6) if math.isfinite(v["value"]) else v["value"]
    return {"criteria": criteria, "promote": all(v["pass"] for v in criteria.values())}


def _evaluate(alias, fresh, n_latency=200):
    m = mlflow.pyfunc.load_model(f"models:/{NAME}@{alias}")
    X, y = fresh[FEATURES].astype("float64"), fresh["eta_minutes"].to_numpy()
    out = m.predict(X)
    err = np.abs(out["eta_minutes"].to_numpy() - y)
    rush, rain = fresh.hour.isin(RUSH_HOURS).to_numpy(), (fresh.rain == 1).to_numpy()
    singles = [X.iloc[[i]] for i in range(n_latency)]
    for r in singles[:20]:                                               # warm-up, not measured
        m.predict(r)
    lat = []
    for r in singles:
        t = time.perf_counter(); m.predict(r); lat.append((time.perf_counter() - t) * 1000)
    mv = MlflowClient().get_model_version_by_alias(NAME, alias)
    return {"registry_version": int(mv.version), "model_version": str(out["model_version"].iloc[0]),
            "mae": round(float(err.mean()), 4), "mae_rush": round(float(err[rush].mean()), 4),
            "mae_rain": round(float(err[rain].mean()), 4), "p95_ms": round(float(np.percentile(lat, 95)), 4),
            "p50_ms": round(float(np.percentile(lat, 50)), 4), "n_rows": int(len(y)),
            "n_rush": int(rush.sum()), "n_rain": int(rain.sum())}


def promotion_gate(**thresholds):
    """Load prod and candidate from the registry, evaluate on FRESH data (make_data(2000, seed=99)),
    measure p95 over 200 single-row predictions, then call decide(prod, cand, **thresholds)."""
    sys.path.insert(0, str(ROOT))
    import eta_data
    _setup()
    fresh = eta_data.make_data(2000, seed=99)
    prod, cand = _evaluate("prod", fresh), _evaluate("candidate", fresh)
    result = decide({k: prod[k] for k in METRIC_KEYS}, {k: cand[k] for k in METRIC_KEYS}, **thresholds)
    return {"thresholds": thresholds or "defaults", "prod": prod, "candidate": cand, **result}


# ----------------------------------------------------------------------------------------------- CLI
def cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("cmd", choices=["register", "gate", "table"])
    ap.add_argument("--min-rel-gain", type=float); ap.add_argument("--max-p95-ms", type=float)
    ap.add_argument("--max-slice-regression", type=float)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "gate_output.txt"))
    a = ap.parse_args(argv)
    print("tracking URI:", tracking_uri())
    if a.cmd == "register":
        tbl = register()
        print(tbl.to_string(index=False))
        (Path(__file__).resolve().parent / "registry_table.txt").write_text(tbl.to_string(index=False) + "\n")
    elif a.cmd == "table":
        print(registry_table().to_string(index=False))
    else:
        overrides = {k: v for k, v in {"min_rel_gain": a.min_rel_gain, "max_p95_ms": a.max_p95_ms,
                                       "max_slice_regression": a.max_slice_regression}.items() if v is not None}
        runs = [overrides] if overrides else [{}, {"min_rel_gain": 0.60}]
        text = []
        for th in runs:
            res = promotion_gate(**th)
            block = f"### promotion_gate({', '.join(f'{k}={v}' for k, v in th.items()) or 'defaults'})\n" \
                    f"{json.dumps(res, indent=2)}\n"
            print(block); text.append(block)
            with mlflow.start_run(run_name="promotion-gate"):              # audit trail for every decision
                mlflow.log_params({f"th_{k}": v for k, v in th.items()} or {"th": "defaults"})
                mlflow.log_metrics({f"{arm}_{k}": res[arm][k] for arm in ("prod", "candidate") for k in METRIC_KEYS})
                mlflow.set_tag("promote", str(res["promote"]))
        if not overrides:
            Path(a.out).write_text("\n".join(text))
            print("written", a.out)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eta_mlflow                                                    # so the pyfunc pickles as eta_mlflow.EtaPyfunc
    eta_mlflow.cli()
