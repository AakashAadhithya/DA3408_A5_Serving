"""
eta_app.py  --  AIOps Module 4, Part A: SwiftParcel delivery-time (ETA) model served over a REST API (FastAPI)
Run (from the submission root):
    ETA_MODEL=data/eta_v1.joblib uvicorn --app-dir partA eta_app:app --port 8000
Endpoints
  GET  /healthz                    liveness/readiness probe  {"status":"ok","model_version":...}
  GET  /v1/models/eta              metadata (version, features)
  POST /v1/models/eta:predict      {"instances":[{7 features}, ...]} -> {"predictions":[...],"model_version":...,"request_id":...}
  GET  /metrics                    Prometheus text exposition (counter + latency histogram)
Pattern (load-once bundle, pydantic contract, `def` endpoint in the thread pool, generic 500) follows
`rest_app.py` from "Lecture 10 -- REST_API_Serving_Exercise.ipynb" (DA3408 repo); metrics/request-id are new.
"""
import logging, os, threading, time, uuid
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

log = logging.getLogger("eta_app")
PREDICT_PATH = "/v1/models/eta:predict"


# ----------------------------------------------------------------------------------------------- model (loaded ONCE)
def _resolve(p: str) -> str:
    """ETA_MODEL may be relative to the cwd or to the submission root (one level above partA/)."""
    here = Path(__file__).resolve().parent
    for cand in (Path(p), here.parent / p, here / p):
        if cand.exists():
            return str(cand)
    return p                                            # let joblib raise a clear FileNotFoundError

MODEL_PATH = _resolve(os.getenv("ETA_MODEL", "data/eta_v1.joblib"))
bundle = joblib.load(MODEL_PATH)                        # loaded once at import, never per request
model = bundle["model"]                                 # module-level: tests monkeypatch this
FEATURES: List[str] = list(bundle["feature_columns"])
VERSION: str = bundle["version"]


# ----------------------------------------------------------------------------------------------- contract
class Delivery(BaseModel):
    """One delivery. Each Field rule is a business assumption written as code."""
    model_config = ConfigDict(allow_inf_nan=False)
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)                        # 0 or 1 (an int in [0, 1])
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def _feasible(self):                                 # cross-field rule
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 together with n_stops > 20")
        return self


class PredictRequest(BaseModel):
    instances: List[Delivery] = Field(min_length=1, max_length=500)


# ----------------------------------------------------------------------------------------------- metrics
class Metrics:
    """Minimal thread-safe Prometheus registry (no extra dependency). One process = one registry."""
    BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
    ETA_BUCKETS = (10, 20, 30, 45, 60, 90, 120, 180, 240)

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = {}                               # (path, status) -> count
        self.lat_buckets = [0] * len(self.BUCKETS)
        self.lat_sum, self.lat_count = 0.0, 0
        self.eta_buckets = [0] * len(self.ETA_BUCKETS)
        self.eta_sum, self.eta_count = 0.0, 0

    def observe_request(self, path, status, seconds):
        with self.lock:
            self.requests[(path, status)] = self.requests.get((path, status), 0) + 1
            for i, b in enumerate(self.BUCKETS):
                if seconds <= b:
                    self.lat_buckets[i] += 1
            self.lat_sum += seconds
            self.lat_count += 1

    def observe_predictions(self, values):
        with self.lock:
            for v in values:
                for i, b in enumerate(self.ETA_BUCKETS):
                    if v <= b:
                        self.eta_buckets[i] += 1
                self.eta_sum += v
                self.eta_count += 1

    @staticmethod
    def _hist(name, buckets, counts, total, n):
        out = [f'{name}_bucket{{le="{b}"}} {c}' for b, c in zip(buckets, counts)]
        out += [f'{name}_bucket{{le="+Inf"}} {n}', f"{name}_sum {total:.6f}", f"{name}_count {n}"]
        return out

    def render(self):
        with self.lock:
            lines = ["# HELP eta_requests_total Requests handled, by path and HTTP status (excludes /metrics).",
                     "# TYPE eta_requests_total counter"]
            lines += [f'eta_requests_total{{path="{p}",status="{s}"}} {c}'
                      for (p, s), c in sorted(self.requests.items())]
            lines += ["# HELP eta_request_latency_seconds End-to-end request latency inside the server.",
                      "# TYPE eta_request_latency_seconds histogram"]
            lines += self._hist("eta_request_latency_seconds", self.BUCKETS, self.lat_buckets,
                                self.lat_sum, self.lat_count)
            lines += ["# HELP eta_prediction_minutes Distribution of predicted ETAs (drift / canary analysis).",
                      "# TYPE eta_prediction_minutes histogram"]
            lines += self._hist("eta_prediction_minutes", self.ETA_BUCKETS, self.eta_buckets,
                                self.eta_sum, self.eta_count)
            lines += ["# HELP eta_model_info Model version currently loaded.", "# TYPE eta_model_info gauge",
                      f'eta_model_info{{model_version="{VERSION}"}} 1']
        return "\n".join(lines) + "\n"


METRICS = Metrics()
KNOWN_PATHS = {PREDICT_PATH, "/healthz", "/v1/models/eta"}


def _clean_request_id(raw):
    """Echo the client's id if it is a sane header value, otherwise mint a UUID4."""
    if raw:
        rid = raw.decode("latin-1").strip()
        if 0 < len(rid) <= 128 and rid.isprintable():
            return rid
    return str(uuid.uuid4())


class RequestContextMiddleware:
    """Pure-ASGI middleware: X-Request-ID on EVERY response, request metrics, last-resort generic 500.
    Pure ASGI (not @app.middleware) so it is cheap and also sees responses produced by exception handlers."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        rid = _clean_request_id(dict(scope.get("headers") or []).get(b"x-request-id"))
        scope.setdefault("state", {})["request_id"] = rid
        path = scope.get("path", "")
        t0, status, started = time.perf_counter(), [500], [False]

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                started[0], status[0] = True, message["status"]
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"]
                message = {**message, "headers": headers + [(b"x-request-id", rid.encode("latin-1"))]}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:                                # never leak a stack trace to the client
            log.exception("unhandled error request_id=%s path=%s", rid, path)
            if started[0]:
                raise
            await JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)(
                scope, receive, send_with_id)
        finally:
            if path != "/metrics":                       # scrapes are not traffic
                METRICS.observe_request(path if path in KNOWN_PATHS else "other", status[0],
                                        time.perf_counter() - t0)


# ----------------------------------------------------------------------------------------------- app
app = FastAPI(title="SwiftParcel ETA service", version=VERSION)
app.add_middleware(RequestContextMiddleware)


def _field(loc):
    parts = [p for p in loc if p != "body"]
    out = ""
    for p in parts:
        out += f"[{p}]" if isinstance(p, int) else (f".{p}" if out else str(p))
    return out or "body"


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    details = [{"field": _field(e.get("loc", ())), "msg": e.get("msg", "invalid")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "validation_failed", "details": details,
                                                  "request_id": request.state.request_id})


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": VERSION}


@app.get("/v1/models/eta")
def metadata():
    return {"name": "eta", "version": VERSION, "features": FEATURES, "max_batch": 500}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(METRICS.render(), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.post(PREDICT_PATH)
def predict(req: PredictRequest, request: Request):
    # plain `def` => runs in FastAPI's thread pool, so CPU-bound sklearn does not block the event loop
    rid = request.state.request_id
    try:
        X = pd.DataFrame(np.array([[getattr(d, f) for f in FEATURES] for d in req.instances], dtype=float),
                         columns=FEATURES)              # column order comes from the bundle, never guessed
        y = np.asarray(model.predict(X), dtype=float)
        if y.shape != (len(req.instances),) or not np.all(np.isfinite(y)):
            raise ValueError("model returned a malformed prediction vector")
        preds = [round(float(v), 1) for v in y]
    except Exception:
        log.exception("inference failed request_id=%s", rid)
        return JSONResponse({"error": "internal_error", "request_id": rid}, status_code=500)
    METRICS.observe_predictions(preds)
    return {"predictions": preds, "model_version": VERSION, "request_id": rid}
