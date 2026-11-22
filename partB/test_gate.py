"""test_gate.py -- pytest for the PURE decide() function (synthetic numbers; no MLflow server, no data).
Run from the submission root:  pytest partB -q"""
import pytest

from eta_mlflow import decide

PROD = {"mae": 10.0, "mae_rush": 20.0, "mae_rain": 12.0, "p95_ms": 2.0}


def cand(**kw):
    """A candidate that comfortably passes every criterion, with selected fields overridden."""
    return {**{"mae": 8.0, "mae_rush": 18.0, "mae_rain": 11.0, "p95_ms": 3.0}, **kw}


def failing(res):
    return {k for k, v in res["criteria"].items() if not v["pass"]}


def test_clear_win_promotes_and_reports_every_criterion():
    res = decide(PROD, cand())
    assert res["promote"] is True
    assert set(res["criteria"]) == {"overall_mae_gain", "candidate_p95_ms",
                                    "rush_hour_slice_regression", "rain_slice_regression"}
    assert res["criteria"]["overall_mae_gain"]["value"] == pytest.approx(0.2)


# --- overall MAE gain: default min_rel_gain = 0.03 -> candidate MAE must be <= 9.70 when prod is 10.0
def test_mae_gain_exactly_at_threshold_promotes():
    assert decide(PROD, cand(mae=9.70))["promote"] is True                  # gain = 0.030 (inclusive)


def test_mae_gain_just_inside_promotes():
    assert decide(PROD, cand(mae=9.69))["promote"] is True                  # gain = 0.031


def test_mae_gain_just_outside_blocks():
    res = decide(PROD, cand(mae=9.71))                                      # gain = 0.029
    assert res["promote"] is False and failing(res) == {"overall_mae_gain"}


def test_worse_candidate_blocks_with_negative_gain():
    res = decide(PROD, cand(mae=10.5))
    assert res["promote"] is False and res["criteria"]["overall_mae_gain"]["value"] == pytest.approx(-0.05)


def test_custom_min_rel_gain_is_respected():
    assert decide(PROD, cand(mae=8.0), min_rel_gain=0.20)["promote"] is True
    assert decide(PROD, cand(mae=8.0), min_rel_gain=0.60)["promote"] is False


# --- latency: default max_p95_ms = 10.0
@pytest.mark.parametrize("p95,ok", [(9.99, True), (10.0, True), (10.01, False)])
def test_latency_boundary(p95, ok):
    res = decide(PROD, cand(p95_ms=p95))
    assert res["promote"] is ok
    assert ("candidate_p95_ms" in failing(res)) is (not ok)


def test_latency_uses_candidate_not_prod():
    slow_prod = {**PROD, "p95_ms": 50.0}
    assert decide(slow_prod, cand(p95_ms=3.0))["promote"] is True


# --- slices: default max_slice_regression = 0.05 (relative). prod rush 20.0 -> cand rush may be <= 21.0
@pytest.mark.parametrize("rush,ok", [(20.99, True), (21.0, True), (21.01, False)])
def test_rush_slice_boundary(rush, ok):
    res = decide(PROD, cand(mae_rush=rush))
    assert res["promote"] is ok
    assert ("rush_hour_slice_regression" in failing(res)) is (not ok)


@pytest.mark.parametrize("rain,ok", [(12.59, True), (12.6, True), (12.61, False)])   # prod rain 12.0 -> limit 12.6
def test_rain_slice_boundary(rain, ok):
    res = decide(PROD, cand(mae_rain=rain))
    assert res["promote"] is ok
    assert ("rain_slice_regression" in failing(res)) is (not ok)


def test_overall_win_hiding_a_slice_regression_is_blocked():
    """The whole point of slices: a big average gain must not hide a rush-hour regression."""
    res = decide(PROD, cand(mae=5.0, mae_rush=25.0))
    assert res["criteria"]["overall_mae_gain"]["pass"] is True
    assert res["promote"] is False and failing(res) == {"rush_hour_slice_regression"}


def test_multiple_failures_all_reported():
    res = decide(PROD, cand(mae=9.9, p95_ms=30.0, mae_rain=20.0))
    assert failing(res) == {"overall_mae_gain", "candidate_p95_ms", "rain_slice_regression"}


def test_slice_regression_threshold_override():
    assert decide(PROD, cand(mae_rush=23.0))["promote"] is False                  # +15 %
    assert decide(PROD, cand(mae_rush=23.0), max_slice_regression=0.20)["promote"] is True


def test_promote_is_a_real_bool_and_inputs_are_not_mutated():
    p, c = dict(PROD), cand()
    res = decide(p, c)
    assert type(res["promote"]) is bool and all(type(v["pass"]) is bool for v in res["criteria"].values())
    assert p == PROD and c == cand()


def test_missing_or_invalid_metric_raises():
    with pytest.raises(KeyError):
        decide(PROD, {"mae": 8.0, "mae_rush": 18.0, "p95_ms": 3.0})              # no mae_rain
    with pytest.raises(ValueError):
        decide(PROD, cand(mae=float("nan")))
