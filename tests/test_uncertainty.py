import numpy as np

from src.core.enums import TaskType
from src.core.schema import DataContract, ScenarioDefinition
from src.core.solver.context import SolveContext
from src.core.trace import SelectionTrace
from src.models.uncertainty import (
    PredictionInterval,
    UncertaintyEstimator,
    UncertaintyGate,
)


def _scenario():
    dc = DataContract(
        required_columns={"load": "float"},
        freq="H",
        min_samples=5,
        business_domain="load_forecast",
    )
    return ScenarioDefinition(
        task_type=TaskType.FORECASTING,
        business_domain="load_forecast",
        data_contract=dc,
        target_schema={"yhat": "float"},
        primary_metric="MAE",
        signature_features=["load"],
        signature={"load": 1.0},
        region="R",
    )


class NativeIntervalModel:
    def predict_interval(self, X, alpha=0.1):
        yhat = np.asarray([10.0, 12.0, 14.0])
        return yhat, yhat - 1.0, yhat + 2.0


class PointModel:
    def predict(self, X):
        return np.asarray([10.0, 20.0, 30.0])


def test_uncertainty_estimator_uses_native_predict_interval():
    interval = UncertaintyEstimator().estimate(
        NativeIntervalModel(),
        X=[0, 1, 2],
        model_id="native",
        alpha=0.2,
    )

    assert isinstance(interval, PredictionInterval)
    assert interval.model_id == "native"
    assert interval.method == "native"
    assert interval.alpha == 0.2
    assert np.allclose(interval.yhat, [10.0, 12.0, 14.0])
    assert np.allclose(interval.lower, [9.0, 11.0, 13.0])
    assert np.allclose(interval.upper, [12.0, 14.0, 16.0])
    assert interval.score > 0


def test_uncertainty_estimator_falls_back_to_residual_quantiles():
    residuals = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])

    interval = UncertaintyEstimator().estimate(
        PointModel(),
        X=[0, 1, 2],
        model_id="point",
        residuals=residuals,
        alpha=0.2,
    )

    lower_shift, upper_shift = np.quantile(residuals, [0.1, 0.9])
    yhat = np.asarray([10.0, 20.0, 30.0])
    assert interval.method == "residual_bootstrap"
    assert np.allclose(interval.lower, yhat + lower_shift)
    assert np.allclose(interval.upper, yhat + upper_shift)


def test_uncertainty_estimator_rejects_invalid_alpha():
    try:
        UncertaintyEstimator().estimate(PointModel(), X=[0], model_id="point", alpha=1.0)
    except ValueError as exc:
        assert "alpha" in str(exc)
    else:
        raise AssertionError("alpha must be validated")


def test_uncertainty_gate_can_bypass_and_reweight_result():
    ctx = SolveContext(
        scenario=_scenario(),
        available_features={"load"},
        model_cols=["safe", "risky"],
    )
    ctx.extras["result"] = {
        "models": ["safe", "risky"],
        "weights": {"safe": 0.4, "risky": 0.6},
        "strategy": "unit",
        "path_id": "p1",
    }
    ctx.uncertainty = {"safe": 0.1, "risky": 0.9}
    trace = SelectionTrace(scenario_id=ctx.scenario.scenario_id)

    UncertaintyGate(threshold=0.5).apply(ctx, trace)

    result = ctx.extras["result"]
    assert result["models"] == ["safe"]
    assert result["weights"] == {"safe": 1.0}
    assert "risky" in trace.candidates_rejected
    stage = next(s for s in trace.stages if s["stage"] == "UncertaintyEstimate")
    assert stage["metadata"]["bypass_applied"] is True
    assert stage["metadata"]["decision_authority"] == "uncertainty_bypass"


def test_uncertainty_gate_records_audit_stage_without_bypass():
    ctx = SolveContext(
        scenario=_scenario(),
        available_features={"load"},
        model_cols=["m1", "m2"],
    )
    ctx.extras["result"] = {
        "models": ["m1", "m2"],
        "weights": {"m1": 0.4, "m2": 0.6},
    }
    ctx.uncertainty = {"m1": 0.1, "m2": 0.2}
    trace = SelectionTrace(scenario_id=ctx.scenario.scenario_id)

    UncertaintyGate(threshold=0.5).apply(ctx, trace)

    assert ctx.extras["result"]["models"] == ["m1", "m2"]
    stage = next(s for s in trace.stages if s["stage"] == "UncertaintyEstimate")
    assert stage["metadata"]["bypass_applied"] is False
    assert stage["metadata"]["decision_authority"] == "audit_only"


def test_uncertainty_gate_rejects_negative_min_keep():
    try:
        UncertaintyGate(threshold=0.5, min_keep=-1)
    except ValueError as exc:
        assert "min_keep" in str(exc)
    else:
        raise AssertionError("min_keep must be validated")
