"""test_eta_app.py -- pytest suite for Part A (run from the submission root:  pytest partA -q)."""
import re
import uuid

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import eta_app

URL = "/v1/models/eta:predict"
GOOD = {"distance_km": 12.5, "n_stops": 4, "hour": 8, "weekday": 2, "traffic_index": 0.6, "rain": 1, "weight_kg": 7.5}


@pytest.fixture()
def client():
    return TestClient(eta_app.app)


def rows(n):
    """n distinct, valid deliveries (distance grows with i so their ETAs differ)."""
    return [{**GOOD, "distance_km": round(2.0 + (7.3 * i) % 140, 2), "n_stops": 1 + i % 15, "hour": i % 24} for i in range(n)]


def local_predict(recs):
    X = pd.DataFrame(recs)[eta_app.FEATURES]
    return [round(float(v), 1) for v in eta_app.model.predict(X)]


def errors_for(resp):
    body = resp.json()
    assert resp.status_code == 422
    assert body["error"] == "validation_failed"
    assert isinstance(body["details"], list) and body["details"]
    assert all(set(d) >= {"field", "msg"} for d in body["details"])
    return body["details"]


# ------------------------------------------------------------------------------------------------- happy path
def test_valid_request_shape_and_values(client):
    r = client.post(URL, json={"instances": [GOOD]})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"predictions", "model_version", "request_id"}
    assert body["model_version"] == eta_app.VERSION
    assert body["predictions"] == local_predict([GOOD])                    # same number as the model, 1 dp
    assert body["predictions"][0] == round(body["predictions"][0], 1)


def test_batch_order_preserved(client):
    batch = rows(25)
    fwd = client.post(URL, json={"instances": batch}).json()["predictions"]
    rev = client.post(URL, json={"instances": batch[::-1]}).json()["predictions"]
    assert fwd == local_predict(batch)
    assert rev == fwd[::-1]
    assert len(set(fwd)) > 20                                              # rows really are distinguishable


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_version": eta_app.VERSION}


# ------------------------------------------------------------------------------------------------- invalid input
@pytest.mark.parametrize("field,value", [
    ("distance_km", 0.05), ("distance_km", 500.1), ("n_stops", 0), ("n_stops", 61), ("hour", 24),
    ("weekday", 7), ("traffic_index", 1.2), ("rain", 2), ("weight_kg", 0), ("weight_kg", 1000.5)])
def test_out_of_range_rejected(client, field, value):
    details = errors_for(client.post(URL, json={"instances": [{**GOOD, field: value}]}))
    assert any(d["field"] == f"instances[0].{field}" for d in details)


def test_missing_field_rejected(client):
    rec = {k: v for k, v in GOOD.items() if k != "traffic_index"}
    details = errors_for(client.post(URL, json={"instances": [GOOD, rec]}))
    assert details[0]["field"] == "instances[1].traffic_index"


def test_wrong_type_rejected(client):
    details = errors_for(client.post(URL, json={"instances": [{**GOOD, "n_stops": "many"}]}))
    assert details[0]["field"] == "instances[0].n_stops"


def test_fractional_integer_and_wrong_top_level_key_rejected(client):
    errors_for(client.post(URL, json={"instances": [{**GOOD, "rain": 0.5}]}))
    errors_for(client.post(URL, json={"rows": [GOOD]}))


def test_malformed_json_rejected(client):
    r = client.post(URL, content=b"{not json", headers={"Content-Type": "application/json"})
    errors_for(r)


def test_cross_field_rule_infeasible(client):
    r = client.post(URL, json={"instances": [GOOD, {**GOOD, "distance_km": 151, "n_stops": 21}]})
    details = errors_for(r)
    assert "infeasible" in r.text
    assert details[0]["field"] == "instances[1]"


def test_cross_field_rule_boundary_is_feasible(client):
    """Exactly 150 km / 20 stops is NOT infeasible (the rule is strict >)."""
    r1 = client.post(URL, json={"instances": [{**GOOD, "distance_km": 150, "n_stops": 60}]})
    r2 = client.post(URL, json={"instances": [{**GOOD, "distance_km": 500, "n_stops": 20}]})
    assert r1.status_code == 200 and r2.status_code == 200


# ------------------------------------------------------------------------------------------------- batch limits
def test_batch_lower_limit(client):
    details = errors_for(client.post(URL, json={"instances": []}))
    assert details[0]["field"] == "instances"
    assert client.post(URL, json={"instances": [GOOD]}).status_code == 200


def test_batch_upper_limit(client):
    ok = client.post(URL, json={"instances": rows(500)})
    assert ok.status_code == 200 and len(ok.json()["predictions"]) == 500
    details = errors_for(client.post(URL, json={"instances": rows(501)}))
    assert details[0]["field"] == "instances"


# ------------------------------------------------------------------------------------------------- request id
def test_request_id_echoed(client):
    r = client.post(URL, json={"instances": [GOOD]}, headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123" and r.json()["request_id"] == "abc-123"


def test_request_id_generated_when_absent(client):
    r1, r2 = client.post(URL, json={"instances": [GOOD]}), client.post(URL, json={"instances": [GOOD]})
    rid = r1.headers["x-request-id"]
    assert str(uuid.UUID(rid)) == rid                                      # a valid UUID
    assert r1.json()["request_id"] == rid and r2.headers["x-request-id"] != rid


def test_request_id_on_errors(client):
    echoed = client.post(URL, json={"instances": []}, headers={"X-Request-ID": "err-42"})
    assert echoed.status_code == 422
    assert echoed.headers["x-request-id"] == "err-42" and echoed.json()["request_id"] == "err-42"
    minted = client.post(URL, json={"instances": [{**GOOD, "hour": 99}]})
    assert minted.json()["request_id"] == minted.headers["x-request-id"]
    uuid.UUID(minted.headers["x-request-id"])
    assert "x-request-id" in client.get("/healthz").headers
    assert "x-request-id" in client.get("/no-such-route").headers


# ------------------------------------------------------------------------------------------------- metrics
def _counter(text, path, status):
    m = re.search(rf'^eta_requests_total\{{path="{re.escape(path)}",status="{status}"\}} (\d+)', text, re.M)
    return int(m.group(1)) if m else 0


def test_metrics_exposition(client):
    before = client.get("/metrics").text
    client.post(URL, json={"instances": [GOOD]})
    client.post(URL, json={"instances": []})
    client.get("/metrics")
    after = client.get("/metrics")
    text = after.text
    assert after.status_code == 200 and after.headers["content-type"].startswith("text/plain")
    assert _counter(text, URL, 200) == _counter(before, URL, 200) + 1
    assert _counter(text, URL, 422) == _counter(before, URL, 422) + 1
    assert _counter(text, "/metrics", 200) == 0                           # scrapes are not counted
    assert "# TYPE eta_request_latency_seconds histogram" in text
    buckets = re.findall(r'^eta_request_latency_seconds_bucket\{le="([^"]+)"\} (\d+)', text, re.M)
    assert len(buckets) >= 5 and buckets[-1][0] == "+Inf"
    counts = [int(c) for _, c in buckets]
    assert counts == sorted(counts)                                        # cumulative buckets
    n = int(re.search(r"^eta_request_latency_seconds_count (\d+)", text, re.M).group(1))
    assert n == counts[-1] and re.search(r"^eta_request_latency_seconds_sum [0-9.]+", text, re.M)


# ------------------------------------------------------------------------------------------------- generic 500
class Exploding:
    def predict(self, X):
        raise RuntimeError("secret internal detail: /opt/models/weights.bin corrupted")


def test_generic_500_when_model_raises(client, monkeypatch):
    monkeypatch.setattr(eta_app, "model", Exploding())
    r = client.post(URL, json={"instances": [GOOD]}, headers={"X-Request-ID": "boom-1"})
    assert r.status_code == 500
    assert r.json() == {"error": "internal_error", "request_id": "boom-1"}
    assert "secret" not in r.text and "Traceback" not in r.text and "RuntimeError" not in r.text
    assert r.headers["x-request-id"] == "boom-1"
    assert _counter(client.get("/metrics").text, URL, 500) >= 1


def test_generic_500_when_model_returns_nan(client, monkeypatch):
    class NaNModel:
        def predict(self, X):
            return np.full(len(X), np.nan)
    monkeypatch.setattr(eta_app, "model", NaNModel())
    r = client.post(URL, json={"instances": [GOOD, GOOD]})
    assert r.status_code == 500 and r.json()["error"] == "internal_error"
