# Module 4 Assignment — SwiftParcel ETA service, four ways
**Aakash Aadhithya · DA24B028 · DA3408 AI Operations**

## 0. Machine, environment, how the numbers were produced
| item | value |
|---|---|
| CPU | 6 logical CPUs, ARM64 (aarch64): Ubuntu Linux VM (UTM) on a Mac |
| RAM |  FILL IN: output of `free -h` (Mem: total) |
| Software | Python 3.14, FastAPI 0.165, uvicorn 0.54, scikit-learn 1.9.1, pandas 3.1.1, MLflow 3.18, Ray 2.60 (Part C limits Ray to 2 CPUs, as required) |
| Load generator | the given `load_test.py`, **on the same VM as the server** (so every capacity number below is a lower bound) |

Every number in this report is copied from a file the code writes (`partA/*.json`, `partB/gate_output.txt`, `partB/shadow_output.txt`, `partC/verify_output.txt`, `partD/smoke_output.txt`). `run_all.sh` regenerates all of them; see `README.md`. Data and models were regenerated with the given `eta_data.py` (fresh-data MAE: v1 11.95 min, v2 6.10 min). Model metrics (MAE, gate verdicts, shadow differences) are deterministic. Latency and throughput are not: the same configuration (1 worker, 1 client, batch 1) gave 1,281 rps in the sweep and 1,012 rps in the batch test, so treat about ±20 % as run-to-run noise.

**Reuse and citations.** Patterns adapted from the DA3408 lecture notebooks (github.com/sudarsun/DA3408): `rest_app.py` from *Lecture 10 — REST_API_Serving_Exercise* (load-once bundle, pydantic contract, `def` endpoint, generic 500) → `partA/eta_app.py`; `m4util.py` (process start/stop helpers, copied with small changes) and the `load_test.py` usage → `partA/run_loadtest.py`; `mlflow_register.py` and the promotion-gate / shadow-test solutions from *Lecture 10 — MLflow_Serving_Exercise* → `partB/eta_mlflow.py`, `partB/shadow.py`; `ray_serve_app.py` from *Lecture 10 — Ray_Serve_Exercise* (deployment + ingress router + build function, fractional CPUs) → `partC/eta_ray.py`; `MLproject`/`python_env.yaml` layout from *MLflow/Exercise2*; Prometheus YAML layout from `Monitoring/sample_prometheus.yml` and `sample_rules.yml`. AI-tool use is declared in `AI_DISCLOSURE.md`.

---
## Part A — REST service
**A1** `partA/eta_app.py` satisfies the seven requirements: pydantic ranges (`Field(ge/le/gt)`, `allow_inf_nan=False`), the cross-field rule as a `model_validator` (message contains *infeasible*, reported as field `instances[i]`), batch 1–500, a custom 422 handler producing `{"error":"validation_failed","details":[{"field","msg"}],"request_id"}`, and a **pure-ASGI middleware** that adds `X-Request-ID` to *every* response (including 404/422/500), records the Prometheus counter + latency histogram (skipping `/metrics`), and is the last-resort generic 500. The model is loaded once at import into the module-level `model`; inference failures (exception *or* non-finite output) return `{"error":"internal_error","request_id":…}` with no internals. An extra `eta_prediction_minutes` histogram feeds the canary abort rule A4 (Part D).

**A2** `partA/test_eta_app.py`: 18 test functions / **27 cases** (pass). They cover a valid request, order preservation (forward vs reversed batch and vs. the model called directly), ten out-of-range fields, missing field, wrong type, fractional int, wrong top-level key, malformed JSON, the infeasible rule and its strict boundary (150 km / 20 stops is feasible), both batch limits (0 and 501 rejected, 500 accepted), request-id echo / UUID generation / on errors, the metrics text (counter increments, `/metrics` not counted, cumulative buckets ending in `+Inf`, `_count` = `+Inf`), and a generic 500 with `monkeypatch` (exception, and NaN output).

**A3.1 Load test** — v1, 1 uvicorn worker, batch 1, 4 s per level (`loadtest_results.json`; 24–48 from `loadtest_slo_probe.json`):

| clients | requests | rps | p50 ms | p95 ms | p99 ms | errors | Little: rps×p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5123 | 1280.8 | 0.8 | 0.9 | 0.9 | 0 | 1.02 |
| 2 | 7010 | **1752.5** | 1.1 | 1.2 | 1.4 | 0 | 1.93 |
| 4 | 5824 | 1456.0 | 2.6 | 4.4 | 5.6 | 0 | 3.79 |
| 8 | 5171 | 1292.8 | 5.9 | 10.6 | 13.5 | 0 | 7.63 |
| 16 | 5078 | 1269.5 | 12.1 | 21.3 | 26.7 | 0 | 15.36 |
| *24* | 4893 | 1223.2 | 19.0 | 31.9 | 39.4 | 0 | 23.24 |
| *32* | 4815 | 1203.8 | 25.8 | 41.0 | 51.1 | 0 | 31.06 |
| *48* | 4641 | 1160.2 | 41.1 | 58.6 | 71.0 | 0 | 47.68 |

**A3.2 Saturation and capacity.** Throughput peaks at **2 clients (1,752 rps)**. From 4 to 48 clients it sits on a slowly falling plateau of 1,160–1,460 rps while latency grows in proportion to the number of clients: the single worker process (one core, because of the GIL) is saturated, so extra clients only queue. **Saturation ≈ 2 clients; sustained throughput X ≈ 1,250 rps (peak 1,750).** Little's Law (N = X·W) holds in every row (last column ≈ clients), so beyond saturation W ≈ N / X, and each extra in-flight request adds about 1/1,250 s ≈ 0.8 ms. The plateau falls rather than staying flat. The likely cause is the client: `load_test.py` is one Python process whose threads share one GIL, so more client threads add contention on the generator side as well. **The SLO p95 ≤ 80 ms was never violated**: even at 48 clients p95 is 58.6 ms. Past saturation p95 ≈ 1.4–1.8 × p50 and p50 ≈ N/X, so p95 ≈ 1.45 · N / 1,160 s, which reaches 80 ms at N ≈ 60–65 in-flight requests. **Capacity under the SLO ≈ 1,200–1,250 rps with up to ~60 concurrent requests** (measured OK up to 48). On this machine the SLO is far from binding: one worker's CPU limits throughput long before latency matters. I would plan at ~70 % of the plateau (≈ 850 rps per worker) to leave headroom.

**A3.3 Batch 1 vs batch 100** (concurrency 1, `loadtest_batch.json`): batch 1 = 0.9 ms per request = **0.9 ms/row** (1,012 rows/s); batch 100 = 1.1 ms per request = **0.011 ms/row** (89,080 rows/s), **≈ 82× cheaper per row**. Each request pays a fixed ~0.7 ms (TCP + HTTP parse, JSON decode, ASGI/middleware, thread-pool hand-off, DataFrame build, response encode). The sklearn pipeline itself also costs the same for 1 or 100 rows (0.230 ms vs 0.232 ms in-process, `loadtest_model_only.json`), because its cost is Python dispatch and input validation, not arithmetic. The true per-row cost (JSON + pydantic for 7 fields + the Ridge dot product) is only (1.1 − 0.9)/99 ≈ 2 µs. The price is latency: in a batch, the first row waits for the last.

**A3.4 Change: 2 uvicorn worker processes** (`--workers 2`, `loadtest_change.json`). Rationale: one Python process is held to about one core by the GIL, and the VM has six.

| clients | 1 worker rps / p95 ms | 2 workers rps / p95 ms | speed-up |
|---:|---:|---:|---:|
| 1 | 1280.8 / 0.9 | 1231.2 / 1.0 | 0.96× |
| 2 | 1752.5 / 1.2 | 1626.0 / 1.6 | 0.93× |
| 4 | 1456.0 / 4.4 | 1949.0 / 3.7 | 1.34× |
| 8 | 1292.8 / 10.6 | 2205.5 / 6.6 | 1.71× |
| 16 | 1269.5 / 21.3 | 2248.2 / 12.4 | 1.77× |

**Under heavy load it nearly doubled; best-vs-best it did not.** At 8–16 clients throughput rose 1.7–1.8× and p95 almost halved (21.3 → 12.4 ms at 16 clients). Comparing the best point of each configuration (2,248 vs 1,752 rps), the gain is only **1.28×**. Below saturation (1–2 clients) extra workers cannot help in a closed loop. The server had spare cores (6 vCPUs), so what stopped a clean 2× is the client: `load_test.py` is a single Python process sharing one GIL, and at ~2,200 rps its own per-request work (`requests`, JSON) becomes the bottleneck. To measure the true ceiling I would run the generator as several processes or from another machine. Costs of the change: the model is held in memory twice, and each process has its own `/metrics` registry, so a scrape sees one worker at random (Prometheus multiprocess mode or one worker per container would fix this; not done).

**A3.5** The mean hides the tail: a few slow requests (queueing, GC, a cold cache) barely move the average, but at hundreds of requests per second they hit many users every minute, and a page that makes several calls almost always meets one of them. p95/p99 describe what the slowest 5 %/1 % of users actually see, so they are what SLOs and capacity are set on (at 48 clients p50 is 41 ms but p99 is 71 ms).

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
| MAE | 11.955 | 6.105 | gain 0.489 ≥ θ | yes (θ 0.03) | No (θ 0.60) |
| p95 single-row | 1.09 ms | 0.87 ms | ≤ 10 ms | yes | yes|
| MAE rush | 19.680 | 8.580 | −56 % ≤ +5 % | yes | yes |
| MAE rain | 11.623 | 6.285 | −46 % ≤ +5 % | yes | yes |
| **promote** | | | | **true** | **false** |

**B3 Shadow** (`shadow.py`; prod on :5101, candidate on :5102, `mlflow models serve -w 1 --env-manager local`; the same 200 single-row requests to both, alternating which server is asked first):

| (i) mean \|Δ\| | (ii) \|Δ\| > 10 min | (iii) p95 prod | (iii) p95 candidate |
|---:|---:|---:|---:|
| 9.48 min (median 7.10, max 51.7) | 35.0 % (70/200) | 3.04 ms | 3.04 ms |

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
| extra | budget 1.5 CPU (1.00 reserved at rest); autoscaling demo (12 clients, 50 ms simulated work): the candidate scaled up about 12 s after load started, peaking at 3 replicas in one run and 2 in a second run (the load generator shares the VM's CPUs) | PASS |

**C3.1** Python's `hash()` of a `str` is **salted randomly per process** (PYTHONHASHSEED randomisation, on since 3.3). The same customer would get a different bucket after every router restart, and from each of several router replicas, which breaks stickiness and makes the split irreproducible. It is also not stable across Python versions or other languages, so another service could not recompute the arm. A cryptographic hash of `salt:id` is deterministic everywhere and uniform; changing the salt per experiment keeps experiments independent.

**C3.2** *Product:* a customer refreshing the tracking page, or placing several orders, must not see the promise jump between two models. For example, the same delivery gets 42.7 min from v1 and 58.2 min from v2; flipping between such numbers erodes trust and triggers support calls. *Analysis:* the unit of the experiment is the customer, and outcomes such as complaints, cancellations and re-orders are measured per customer. Per-request randomisation exposes every customer to both arms, which contaminates and dilutes the comparison and breaks independence between arms. Sticky, monotone assignment also allows ramping 10 → 50 → 100 % with no customer crossing back (check d).

**C3.3** `target_ongoing_requests = 2`: a single-row prediction costs only ~1–3 ms, so two in flight per 0.25-CPU replica keeps queueing delay at a few ms, far below the 80 ms SLO, without scaling on noise. `max_replicas = 3` is set by the CPU budget (0.25 + 2×0.25 + 3×0.25 = 1.5 of 2 CPUs, leaving room for the proxy, controller and the OS), and `min_replicas = 1` avoids a ~2 s cold start for the canary arm. Upscale fast (5 s, 10 s look-back) to protect the SLO when the weight is raised or a rush-hour burst arrives; downscale slowly (60 s) so bursty courier traffic does not make replicas flap. In the demo the candidate scaled up about 12 s after load started (5 s upscale delay plus the 10 s look-back window filling up), peaking at 2–3 replicas.

---
## Part D — Release gate
**D1.1** `partD/smoke_eta.py --url … --expect-version … [--max-p95-ms 80]` auto-detects Part A / Part C (`…:predict` or a base URL) or MLflow (`…/invocations`). It runs nine checks: health; expected `model_version`; output sanity (finite, 0 < ETA < 1 day) plus batch == single-row; **three model-sanity checks** across 4 contexts (longer distance, heavier traffic, more stops never give a shorter ETA); **invalid input → 422** (hour = 25, the infeasible delivery, an empty batch); p95 over 50 single-row requests ≤ budget; and request-id echo (SKIP if the server has no such contract). Exit 0 iff nothing FAILs. `run_smoke_demo.py` → `smoke_output.txt`, **5/5 scenarios as expected**:

| server | expect | result |
|---|---|---|
| Part A, v1 | v1 | 9/9 PASS, p95 0.9 ms → **exit 0** |
| Part A, v1 | v2 (negative control) | version + health FAIL → **exit 1** |
| Part A, a broken constant-output "v2" (negative control) | v2 | all 3 sanity checks FAIL → **exit 1** |
| Part B, MLflow `@candidate` | v2 | 8 PASS + 1 SKIP, p95 2.0 ms → **exit 0** (MLflow answers 400, not 422, and has no range checks) |
| Part C, Ray + `X-Beta: 1` | v2 | 9/9 PASS, p95 10.2 ms → **exit 0** |

**D1.2 Abort criteria for a 10 % canary of v2.** Each criterion compares v2 with v1 over the same window, so time-of-day effects cancel. They are implemented as Prometheus alerting rules (`partD/abort_rules.yml`, scraped via `partD/prometheus.yml`). They pass `promtool check`, have `promtool test rules` unit tests (each fires on a broken v2 and stays quiet on a healthy one), and were evaluated against two live Part A servers (all rules healthy, all inactive). Any firing alert ⇒ set the weight to 0.

| # | metric | abort threshold | window | guard |
|---|---|---|---|---|
| A1 | v2 5xx ratio of `/v1/models/eta:predict` | > 1 % | rate over 5 min, sustained 2 min | only when v2 ≥ 0.2 req/s |
| A2 | v2 p95 of `eta_request_latency_seconds` | > 80 ms (the SLO) | 5 min histogram, sustained 5 min | — |
| A3 | MAE of delivered orders (delayed labels), per slice all / rush / rain | v2 > 1.05 × v1 on any slice | 2 h, sustained 10 min | ≥ 200 delivered v2 orders in the slice |
| A4 | mean predicted ETA (`eta_prediction_minutes`) | \|v2 − v1\| / v1 > 10 % | 30 min, sustained 15 min | v1/v2 means agree within 0.4 % on fresh data; 30-min sampling error ≈ 1.6 % |

A1/A2 protect reliability and the SLO within minutes. A4 catches feature-pipeline or unit bugs (km vs miles) before any labels arrive. A3 is the slice-aware business check, the production twin of the B2 gate.
