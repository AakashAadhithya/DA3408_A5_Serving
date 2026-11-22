"""
eta_ray.py -- AIOps Module 4, Part C: Ray Serve with a STICKY canary (CPU only, fits on 2 CPUs)
Architecture:  client --HTTP--> EtaRouter (FastAPI ingress, sticky split by customer_id)
                                   +--> EtaModelV1  prod       2 fixed replicas            x 0.25 CPU
                                   +--> EtaModelV2  candidate  autoscaling 1..3 replicas   x 0.25 CPU
CPU budget: router 0.25 + prod 2 x 0.25 + candidate (max) 3 x 0.25 = 1.50 CPU  <=  2 CPUs.
Route:  POST /v1/models/eta:predict   {"customer_id": "...", "instances": [{7 features}, ...]}
        header  X-Beta: 1  forces the candidate for that one request.
Run:    python partC/eta_ray.py                       (starts Ray(2 CPUs) + Serve on :8100 and blocks)
        cd partC && serve run serve_config.yaml      (declarative; weight set in the YAML args)
Structure (deployment class + ingress router + build function, fractional CPUs) follows `ray_serve_app.py`
in "Lecture 10 -- Ray_Serve_Exercise.ipynb" (DA3408 repo). The hashed, sticky split is new.
"""
import hashlib, os, time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ray import serve
from ray.serve.handle import DeploymentHandle

ROOT = Path(__file__).resolve().parent.parent
PREDICT_PATH = "/v1/models/eta:predict"
SALT = "swiftparcel-eta-canary-2026"          # one salt per experiment; changing it reshuffles everybody
ROUTER_CPUS, MODEL_CPUS, PROD_REPLICAS = 0.25, 0.25, 2
CAND_AUTOSCALING = dict(
    min_replicas=1, initial_replicas=1, max_replicas=3,  # worst case 1.5 of 2 CPUs: headroom for proxy/controller
    target_ongoing_requests=2,                           # ~2 queued requests/replica keeps p95 low
    upscale_delay_s=5, downscale_delay_s=60,             # scale up fast, down slowly (no flapping)
    metrics_interval_s=2, look_back_period_s=10)


# replicas must be able to `import eta_ray` however the driver was launched (script, -c, notebook, grader)
RUNTIME_ENV = {"env_vars": {"PYTHONPATH": os.pathsep.join(
    p for p in (str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")) if p)}}


def _bundle(version: str) -> str:
    p = Path(f"data/eta_{version}.joblib")
    return str((p if p.exists() else ROOT / p).resolve())


# ----------------------------------------------------------------------------------------------- bucketing
def bucket(customer_id: str, salt: str = SALT) -> float:
    """Stable hash of the id -> uniform float in [0, 1). Same id => same bucket in every process, on every
    machine, after every restart (unlike built-in hash(), which is salted per process)."""
    digest = hashlib.sha256(f"{salt}:{customer_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64


def choose_arm(customer_id: str, weight: float, beta: bool = False) -> str:
    """'v2' (candidate) for the customers whose bucket < weight, else 'v1'. Monotone in weight: raising the
    weight only ever moves customers v1 -> v2, never back."""
    return "v2" if beta or bucket(customer_id) < weight else "v1"


# ----------------------------------------------------------------------------------------------- model replicas
@serve.deployment(
    ray_actor_options={"num_cpus": MODEL_CPUS},
    max_ongoing_requests=16,                             # back-pressure per replica
    health_check_period_s=5,
    logging_config={"log_level": "WARNING"})
class EtaModel:
    def __init__(self, path: str, extra_work_ms: float = 0.0):
        # extra_work_ms simulates a heavier model (blocking sleep, costs no CPU) so scaling is visible on a laptop
        self.extra = float(extra_work_ms) / 1000.0
        b = joblib.load(path)                            # bundle loaded ONCE per replica
        self.model, self.cols, self.version = b["model"], list(b["feature_columns"]), b["version"]
        self.replica = serve.get_replica_context().replica_id.unique_id

    async def __call__(self, rows: List[Dict]):
        X = pd.DataFrame(rows)[self.cols]
        y = self.model.predict(X)
        if self.extra:
            time.sleep(self.extra)
        return {"predictions": [round(float(v), 1) for v in y], "model_version": self.version,
                "served_by": self.replica}

    async def check_health(self):                        # raise => Serve restarts the replica
        if self.model is None:
            raise RuntimeError("model missing")


# ----------------------------------------------------------------------------------------------- router (ingress)
class Delivery(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def _feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 together with n_stops > 20")
        return self


class CanaryRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=128)
    instances: List[Delivery] = Field(min_length=1, max_length=500)


api = FastAPI(title="SwiftParcel ETA - sticky canary")


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": ROUTER_CPUS}, logging_config={"log_level": "WARNING"})
@serve.ingress(api)
class EtaRouter:
    def __init__(self, prod: DeploymentHandle, cand: DeploymentHandle, weight: float):
        self.h = {"v1": prod, "v2": cand}
        self.weight = float(weight)

    @api.get("/healthz")
    def healthz(self):
        return {"status": "ok", "canary_weight": self.weight, "arms": {"v1": "prod", "v2": "candidate"}}

    @api.post(PREDICT_PATH)
    async def predict(self, req: CanaryRequest, x_beta: Optional[str] = Header(default=None, alias="X-Beta")):
        beta = (x_beta or "").strip().lower() in ("1", "true")
        arm = choose_arm(req.customer_id, self.weight, beta)
        try:
            out = await self.h[arm].remote([d.model_dump() for d in req.instances])
        except Exception:
            raise HTTPException(500, "inference failed")  # generic: no internals in the body
        return {**out, "arm": arm, "customer_id": req.customer_id}


# ----------------------------------------------------------------------------------------------- application
def build(weight: float = 0.1, extra_work_ms: float = 0.0):
    """Compose the Serve application: EtaRouter -> (EtaModelV1 x2 fixed, EtaModelV2 autoscaling 1..3)."""
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    model_opts = {"num_cpus": MODEL_CPUS, "runtime_env": RUNTIME_ENV}
    prod = EtaModel.options(name="EtaModelV1", num_replicas=PROD_REPLICAS,
                            ray_actor_options=model_opts).bind(_bundle("v1"), extra_work_ms)
    cand = EtaModel.options(name="EtaModelV2", num_replicas=None, autoscaling_config=CAND_AUTOSCALING,
                            ray_actor_options=model_opts).bind(_bundle("v2"), extra_work_ms)
    router = EtaRouter.options(ray_actor_options={"num_cpus": ROUTER_CPUS, "runtime_env": RUNTIME_ENV})
    return router.bind(prod, cand, weight)


def app_builder(args: Dict[str, str]):
    """Builder for serve_config.yaml (Serve passes the YAML `args:` mapping as one dict)."""
    return build(float(args.get("weight", 0.1)), float(args.get("extra_work_ms", 0.0)))


app = build(float(os.getenv("CANARY_WEIGHT", "0.1")))   # import path for `serve run eta_ray:app`


if __name__ == "__main__":
    import ray
    ray.init(address="local", num_cpus=int(os.getenv("RAY_CPUS", "2")), include_dashboard=False, ignore_reinit_error=True)
    serve.start(http_options={"host": "127.0.0.1", "port": int(os.getenv("SERVE_PORT", "8100"))})
    serve.run(app, name="eta", route_prefix="/")
    print("Serve is up on :8100  (Ctrl-C to stop)")
    while True:
        time.sleep(3600)
