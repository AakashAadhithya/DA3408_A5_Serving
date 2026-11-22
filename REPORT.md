# Module 4 Assignment — SwiftParcel ETA service, four ways
**Aakash Aadhithya · DA24B028 · DA3408 AI Operations**

## 0. Machine, environment, how the numbers were produced
| item | value |
|---|---|
| CPU | 2 logical CPUs, Intel Xeon @ 2.10 GHz (Linux VM, kernel 6.18) |
| RAM | 7 GB |
| Software | Python 3.12, FastAPI 0.165, uvicorn 0.54, scikit-learn 1.9.1, pandas 3.1, MLflow 3.18, Ray 2.60 |
| Load generator | the given `load_test.py`, **on the same 2 CPUs as the server** (so every capacity number below is a lower bound) |

Every number in this report is copied from a file the code writes (`partA/*.json`, `partB/gate_output.txt`, `partB/shadow_output.txt`, `partC/verify_output.txt`, `partD/smoke_output.txt`). `run_all.sh` regenerates all of them; see `README.md`. Data and models were regenerated with the given `eta_data.py` (fresh-data MAE: v1 11.95 min, v2 6.10 min). Run-to-run noise on this VM: a second full run from a clean checkout gave 8–16 % lower throughput at every load level, with the same shape and conclusions; model metrics (MAE, shadow differences) are deterministic.

**Reuse and citations.** Patterns adapted from the DA3408 lecture notebooks (github.com/sudarsun/DA3408): `rest_app.py` from *Lecture 10 — REST_API_Serving_Exercise* (load-once bundle, pydantic contract, `def` endpoint, generic 500) → `partA/eta_app.py`; `m4util.py` (process start/stop helpers, copied with small changes) and the `load_test.py` usage → `partA/run_loadtest.py`; `mlflow_register.py` and the promotion-gate / shadow-test solutions from *Lecture 10 — MLflow_Serving_Exercise* → `partB/eta_mlflow.py`, `partB/shadow.py`; `ray_serve_app.py` from *Lecture 10 — Ray_Serve_Exercise* (deployment + ingress router + build function, fractional CPUs) → `partC/eta_ray.py`; `MLproject`/`python_env.yaml` layout from *MLflow/Exercise2*; Prometheus YAML layout from `Monitoring/sample_prometheus.yml` and `sample_rules.yml`. AI-tool use is declared in `AI_DISCLOSURE.md`.

---
## Part A — REST service
**A1** `partA/eta_app.py` satisfies the seven requirements: pydantic ranges (`Field(ge/le/gt)`, `allow_inf_nan=False`), the cross-field rule as a `model_validator` (message contains *infeasible*, reported as field `instances[i]`), batch 1–500, a custom 422 handler producing `{"error":"validation_failed","details":[{"field","msg"}],"request_id"}`, and a **pure-ASGI middleware** that adds `X-Request-ID` to *every* response (including 404/422/500), records the Prometheus counter + latency histogram (skipping `/metrics`), and is the last-resort generic 500. The model is loaded once at import into the module-level `model`; inference failures (exception *or* non-finite output) return `{"error":"internal_error","request_id":…}` with no internals. An extra `eta_prediction_minutes` histogram feeds the canary abort rule A4 (Part D).

**A2** `partA/test_eta_app.py`: 18 test functions / **27 cases** (pass). They cover a valid request, order preservation (forward vs reversed batch and vs. the model called directly), ten out-of-range fields, missing field, wrong type, fractional int, wrong top-level key, malformed JSON, the infeasible rule and its strict boundary (150 km / 20 stops is feasible), both batch limits (0 and 501 rejected, 500 accepted), request-id echo / UUID generation / on errors, the metrics text (counter increments, `/metrics` not counted, cumulative buckets ending in `+Inf`, `_count` = `+Inf`), and a generic 500 with `monkeypatch` (exception, and NaN output).

**A3.1 Load test** — v1, 1 uvicorn worker, batch 1, 4 s per level (`loadtest_results.json`; 24–48 from `loadtest_slo_probe.json`):

| clients | requests | rps | p50 ms | p95 ms | p99 ms | errors | Little: rps×p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1372 | 343.0 | 2.8 | 3.5 | 4.4 | 0 | 0.96 |
| 2 | 1571 | 392.8 | 4.9 | 7.7 | 9.3 | 0 | 1.92 |
| 4 | 1760 | 440.0 | 8.7 | 14.9 | 17.9 | 0 | 3.83 |
| 8 | 1774 | 443.5 | 17.3 | 29.5 | 35.9 | 0 | 7.67 |
| 16 | 1903 | 475.8 | 33.1 | 51.6 | 63.5 | 0 | 15.75 |
| *24* | 1798 | 449.5 | 52.2 | **81.4** | 158.0 | 0 | 23.46 |
| *32* | 1841 | 460.2 | 67.1 | 112.3 | 163.9 | 0 | 30.88 |
| *48* | 1951 | 487.8 | 97.1 | 151.0 | 195.1 | 0 | 47.37 |

**A3.2 Saturation and capacity.** Throughput rises from 343 to 440 rps between 1 and 4 clients and then stays flat (440–488 rps) while latency doubles with every doubling of clients: the single worker's core is busy, so extra clients only queue. **Saturation ≈ 4 clients, X_max ≈ 460 ± 20 rps.** Little's Law (N = X·W) holds in every row (last column ≈ clients), so beyond saturation W ≈ N / X_max, i.e. each extra in-flight request adds about 1/460 s ≈ 2.2 ms. Above saturation, p95 ≈ 1.56 × p50 (51.6/33.1, 81.4/52.2). The SLO p95 ≤ 80 ms therefore needs p50 ≤ 80/1.56 ≈ 51 ms, so N ≤ 460 × 0.051 ≈ **23 in-flight requests**. The measurements agree: 16 clients pass (51.6 ms), 24 just fail (81.4 ms). **Capacity under the SLO ≈ 450–475 rps with at most ~20 concurrent requests** (I would admit up to 20 and shed or queue the rest). The SLO does not lower the *throughput* (the CPU saturates first); it caps the *queue depth*. I would plan at ~70 % of that (≈ 320 rps per worker) to leave headroom.

**A3.3 Batch 1 vs batch 100** (concurrency 1, `loadtest_batch.json`): batch 1 = 2.8 ms per request = **2.8 ms/row** (347 rows/s); batch 100 = 3.7 ms per request = **0.037 ms/row** (25 920 rows/s), **≈ 75× cheaper per row**. Each request pays a fixed ~2–3 ms (TCP + HTTP parse, JSON decode, ASGI/middleware, thread-pool hand-off, DataFrame build, response encode). The sklearn pipeline itself also costs about the same for 1 or 100 rows (0.67 ms vs 0.63 ms in-process, `loadtest_model_only.json`), because its cost is Python dispatch and input validation, not arithmetic. The true per-row cost (JSON + pydantic for 7 fields + the Ridge dot product) is only (3.7 − 2.8)/99 ≈ 9 µs. The price is latency: in a batch, the first row waits for the last.

**A3.4 Change: 2 uvicorn worker processes** (`--workers 2`, `loadtest_change.json`). Rationale: one Python process is held to about one core by the GIL, and the machine has two.

| clients | 1 worker rps / p95 | 2 workers rps / p95 | speed-up |
|---:|---:|---:|---:|
| 1 | 343.0 / 3.5 | 332.5 / 3.6 | 0.97× |
| 2 | 392.8 / 7.7 | 412.0 / 7.4 | 1.05× |
| 4 | 440.0 / 14.9 | 665.2 / 11.1 | 1.51× |
| 8 | 443.5 / 29.5 | 712.8 / 20.0 | 1.61× |
| 16 | 475.8 / 51.6 | 751.2 / 38.2 | 1.58× |

**It did not double: saturated throughput rose about 1.6×** (a second full run from a clean checkout gave 1.4–1.6×) and p95 at 8–16 clients fell by about 30 %. Below saturation (1–2 clients) nothing changes, as expected for a closed loop. The missing ~0.4× has a clear cause: the load generator (Python threads + `requests` + JSON) runs on the same two vCPUs and takes a large share of them, so the two workers never get two full cores. With the client on another machine I would expect close to 2×. Costs of the change: the model is held in memory twice, and with several workers each process has its own `/metrics` registry, so a scrape sees one worker at random (Prometheus multiprocess mode or one worker per container would fix this; not done).

**A3.5** The mean hides the tail: a few slow requests (queueing, GC, a cold cache) barely move the average, but at hundreds of requests per second they hit many users every minute, and a page that makes several calls almost always meets one of them. p95/p99 describe what the slowest 5 %/1 % of users actually see, so they are what SLOs and capacity are set on (our 24-client row has p50 52 ms but p99 158 ms).

---
## Part B — MLflow registry, gate, shadow
**B1** `eta_mlflow.py` wraps each bundle in `EtaPyfunc` (returns `eta_minutes` rounded to 1 dp + `model_version`). Each is logged with a signature (7 `double` inputs, which tolerates JSON ints as in the lecture; outputs `eta_minutes: double`, `model_version: string`), an input example, pinned `pip_requirements` (`mlflow==3.18.0, scikit-learn==1.9.1, pandas==3.1.1, numpy==2.5.5, joblib==1.6.0, cloudpickle==3.1.3`) and `code_paths` (so the server can import the class), into a SQLite store (`MLFLOW_TRACKING_URI`, default `sqlite:///mlflow.db`), registered as `eta-model`:

| registry version | alias | bundle | algorithm |
|---:|---|---|---|
| 1 | **prod** | v1 | Pipeline(StandardScaler, Ridge) |
| 2 | **candidate** | v2 | GradientBoostingRegressor |

**B2** `decide()` is pure (no I/O; validates inputs; inclusive bounds with 1e-9 float tolerance). Interpretation: the MAE gain and the slice regression are both **relative to prod** (gain = (prod−cand)/prod ≥ min_rel_gain; slice OK iff cand ≤ prod·(1+max_slice_regression)). `promotion_gate()` loads `@prod`/`@candidate`, evaluates on `make_data(2000, seed=99)` (571 rush-hour rows, 994 rain rows), and times 200 single-row `predict` calls after 20 warm-ups. `test_gate.py`: 15 test functions / **21 cases**, including just inside, exactly at, and just outside every threshold. Output (`gate_output.txt`):

| | prod (v1) | candidate (v2) | criterion | defaults | min_rel_gain = 0.60 |
|---|---:|---:|---|---|---|
| MAE | 11.955 | 6.105 | gain 0.489 ≥ θ | ✅ (θ 0.03) | ❌ (θ 0.60) |
| p95 single-row | 2.61 ms | 3.22 ms | ≤ 10 ms | ✅ | ✅ |
| MAE rush | 19.680 | 8.580 | −56 % ≤ +5 % | ✅ | ✅ |
| MAE rain | 11.623 | 6.285 | −46 % ≤ +5 % | ✅ | ✅ |
| **promote** | | | | **true** | **false** |

**B3 Shadow** (`shadow.py`; prod on :5101, candidate on :5102, `mlflow models serve -w 1 --env-manager local`; the same 200 single-row requests to both, alternating which server is asked first):

| (i) mean \|Δ\| | (ii) \|Δ\| > 10 min | (iii) p95 prod | (iii) p95 candidate |
|---:|---:|---:|---:|
| 9.48 min (median 7.10, max 51.7) | 35.0 % (70/200) | 8.41 ms | 8.32 ms |

I would promote v2, **through the sticky canary, not in one switch**. The gate passes with a large margin, latency is the same, and the large disagreements are concentrated where v1 is weakest: 51 % of them are rush-hour rows versus 28 % overall. On these 200 rows (labels known) v2's MAE is 6.11 min against 11.42, and v2 is closer to the truth on 66 % of them. But a third of customers would see their ETA move by more than 10 minutes, so the change must be ramped and watched (Parts C, D). **Moving an alias changes nothing in a running server**, because `mlflow models serve -m models:/eta-model@prod` resolves the alias to a version *once at start-up*, copies that version's artifacts and keeps the model in memory; it never reads the registry again. `shadow_output.txt` shows this: after moving `prod` → v2 the running server still answered `v1`, and only a restart gave `v2` (then rolled back). Two ways to pick up the new version without dropping requests: **(1) rolling update** — start new instances that resolve the alias at boot, add each to the load balancer only once its readiness probe passes (e.g. Kubernetes Deployment with `maxUnavailable: 0, maxSurge: 1`), and drain old instances gracefully (SIGTERM, finish in-flight requests). **(2) Blue-green** — bring up a complete "green" server on the new version, run the Part D smoke test on it, then switch the Service selector or load balancer in one step, keeping "blue" warm for an instant rollback. (Ray Serve's in-place redeploy in Part C is a third: replicas are replaced gradually.)

---
## Part C — Ray Serve sticky canary
**C1** `eta_ray.py`: `EtaModel` loads its bundle once per replica. Prod runs as `EtaModelV1` with 2 fixed replicas; the candidate runs as `EtaModelV2` with `autoscaling_config` min 1 / max 3, `target_ongoing_requests` 2, up/down delays 5/60 s. A FastAPI-ingress `EtaRouter` validates `{"customer_id", "instances"}` (same ranges + infeasible rule as Part A) and returns the model output plus `"arm"`. Bucketing: `SHA-256(salt + ":" + customer_id)` → the first 8 bytes / 2⁶⁴ give a uniform number in [0, 1), and the arm is `v2` iff that number < weight. This is per *customer*, deterministic, and monotone in the weight. `X-Beta: 1` forces v2 for that request only. Reservations: router 0.25 + prod 2×0.25 + candidate ≤ 3×0.25 = **1.5 CPU ≤ 2**. `build(weight=0.1, extra_work_ms=0.0)` returns the application; a `runtime_env` sets `PYTHONPATH` so replicas can import the module however the driver is launched. A declarative `serve_config.yaml` (Ray's YAML deploy path) is included and was deployed with `serve run`.

**C2** `verify_ray.py` (Ray with `num_cpus=2`, 1000 ids, weight 0.2), from `verify_output.txt`:

| check | evidence | result |
|---|---|---|
| (a) share | 0.208 on v2 (target 0.20 ± 0.04); 1000/1000 match `choose_arm()` | PASS |
| (b) 2nd call | 200/200 same arm, even with a different delivery in the payload | PASS |
| (c) redeploy | `serve.delete` + identical redeploy: 200/200 same arm; all answering replicas are new | PASS |
| (d) 0.2 → 0.5 | 208 were on v2 → 0 moved back; 305 moved v1 → v2; share 0.513 | PASS |
| (e) X-Beta | 50/50 prod customers served by v2 with the header; 50/50 back on v1 without it | PASS |
| extra | budget 1.5 CPU (1.00 reserved at rest); autoscaling demo (12 clients, 50 ms simulated work): candidate scaled 1 → 3 replicas within ~6 s (peak was 2 in a second run, since the load generator competes for the same 2 CPUs) | PASS |

**C3.1** Python's `hash()` of a `str` is **salted randomly per process** (PYTHONHASHSEED randomisation, on since 3.3). The same customer would get a different bucket after every router restart, and from each of several router replicas, which breaks stickiness and makes the split irreproducible. It is also not stable across Python versions or other languages, so another service could not recompute the arm. A cryptographic hash of `salt:id` is deterministic everywhere and uniform; changing the salt per experiment keeps experiments independent.

**C3.2** *Product:* a customer refreshing the tracking page, or placing several orders, must not see the promise jump between two models. In our own YAML test the same delivery was 42.7 min on v1 and 58.2 min on v2; flipping between such numbers erodes trust and triggers support calls. *Analysis:* the unit of the experiment is the customer, and outcomes such as complaints, cancellations and re-orders are measured per customer. Per-request randomisation exposes every customer to both arms, which contaminates and dilutes the comparison and breaks independence between arms. Sticky, monotone assignment also allows ramping 10 → 50 → 100 % with no customer crossing back (check d).

**C3.3** `target_ongoing_requests = 2`: a single-row prediction costs only ~1–3 ms, so two in flight per 0.25-CPU replica keeps queueing delay at a few ms, far below the 80 ms SLO, without scaling on noise. `max_replicas = 3` is set by the CPU budget (0.25 + 2×0.25 + 3×0.25 = 1.5 of 2 CPUs, leaving room for the proxy, controller and the OS), and `min_replicas = 1` avoids a ~2 s cold start for the canary arm. Upscale fast (5 s, 10 s look-back) to protect the SLO when the weight is raised or a rush-hour burst arrives; downscale slowly (60 s) so bursty courier traffic does not make replicas flap. In the demo the candidate reached 3 replicas about 6 s after load started.

---
## Part D — Release gate
**D1.1** `partD/smoke_eta.py --url … --expect-version … [--max-p95-ms 80]` auto-detects Part A / Part C (`…:predict` or a base URL) or MLflow (`…/invocations`). It runs nine checks: health; expected `model_version`; output sanity (finite, 0 < ETA < 1 day) plus batch == single-row; **three model-sanity checks** across 4 contexts (longer distance, heavier traffic, more stops never give a shorter ETA); **invalid input → 422** (hour = 25, the infeasible delivery, an empty batch); p95 over 50 single-row requests ≤ budget; and request-id echo (SKIP if the server has no such contract). Exit 0 iff nothing FAILs. `run_smoke_demo.py` → `smoke_output.txt`, **5/5 scenarios as expected**:

| server | expect | result |
|---|---|---|
| Part A, v1 | v1 | 9/9 PASS, p95 3.3 ms → **exit 0** |
| Part A, v1 | v2 (negative control) | version + health FAIL → **exit 1** |
| Part A, a broken constant-output "v2" (negative control) | v2 | all 3 sanity checks FAIL → **exit 1** |
| Part B, MLflow `@candidate` | v2 | 8 PASS + 1 SKIP → **exit 0** (MLflow answers 400, not 422, and has no range checks) |
| Part C, Ray + `X-Beta: 1` | v2 | **exit 0** |

**D1.2 Abort criteria for a 10 % canary of v2.** Each criterion compares v2 with v1 over the same window, so time-of-day effects cancel. They are implemented as Prometheus alerting rules (`partD/abort_rules.yml`, scraped via `partD/prometheus.yml`). They pass `promtool check`, have `promtool test rules` unit tests (each fires on a broken v2 and stays quiet on a healthy one), and were evaluated against two live Part A servers (all rules healthy, all inactive). Any firing alert ⇒ set the weight to 0.

| # | metric | abort threshold | window | guard |
|---|---|---|---|---|
| A1 | v2 5xx ratio of `/v1/models/eta:predict` | > 1 % | rate over 5 min, sustained 2 min | only when v2 ≥ 0.2 req/s |
| A2 | v2 p95 of `eta_request_latency_seconds` | > 80 ms (the SLO) | 5 min histogram, sustained 5 min | — |
| A3 | MAE of delivered orders (delayed labels), per slice all / rush / rain | v2 > 1.05 × v1 on any slice | 2 h, sustained 10 min | ≥ 200 delivered v2 orders in the slice |
| A4 | mean predicted ETA (`eta_prediction_minutes`) | \|v2 − v1\| / v1 > 10 % | 30 min, sustained 15 min | v1/v2 means agree within 0.4 % on fresh data; 30-min sampling error ≈ 1.6 % |

A1/A2 protect reliability and the SLO within minutes. A4 catches feature-pipeline or unit bugs (km vs miles) before any labels arrive. A3 is the slice-aware business check, the production twin of the B2 gate.

---
## What was NOT tested (honest list)
- Numbers are from a **2-vCPU Linux VM**, not a laptop, with the load generator on the same CPUs; absolute values will differ on other hardware.
- Python 3.12 only (3.10/3.11 not tried). Ray was single-node only.
- The rolling-update and blue-green procedures in B3 are described, not run (no Kubernetes here). `mlflow models build-docker` was not run.
- A3 needs a delayed-ground-truth exporter (`eta_delivered_abs_error_minutes_*`), which is **not implemented**; A3 is verified only with promtool's synthetic series. No alert was triggered by real traffic, and Alertmanager was not run.
- With `--workers 2`, `/metrics` is per process (not aggregated); not addressed.
- `MLproject` entry points: `gate` and `test` were run with `mlflow run`; `register` and `shadow` were run as plain scripts only.
- The autoscaling demo uses a simulated 50 ms blocking step (`extra_work_ms`), not a real heavier model.
