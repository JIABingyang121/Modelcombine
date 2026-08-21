import numpy as np
import pandas as pd

from src.core.enums import TaskType
from src.core.schema import DataContract, ScenarioDefinition
from src.core.solver.backends import ProtocolBBackend
from src.core.solver.context import SolveContext
from src.core.trace import SelectionTrace


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


def _prediction_frame(n=12):
    y = np.linspace(10.0, 21.0, n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
            "y": y,
            "m1": y + 0.1,
            "m2": y + np.linspace(0.2, 0.5, n),
        }
    )


def _raw_frame(n=12):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
            "temp": np.linspace(20.0, 30.0, n),
            "humidity": np.linspace(40.0, 60.0, n),
        }
    )


def test_protocol_b_backend_delegates_and_normalizes_result():
    ctx = SolveContext(
        scenario=_scenario(),
        available_features={"load", "temp", "humidity"},
        model_cols=["m1", "m2"],
        df_val=_prediction_frame(),
        df_test=_prediction_frame(),
        df_raw_val=_raw_frame(),
        df_raw_test=_raw_frame(),
        horizon=1,
        dataset_name="unit",
        base_model_cols=["m1", "m2"],
    )
    trace = SelectionTrace(scenario_id=ctx.scenario.scenario_id)

    result = ProtocolBBackend().combine(ctx, trace)

    assert set(result) >= {"models", "weights", "strategy", "path_id", "raw", "protocol"}
    assert result["models"]
    assert isinstance(result["weights"], dict)
    assert "val" in result["raw"]
    assert "test" in result["raw"]
    assert "protocol" in result["raw"]
    assert any(s["stage"] == "ProtocolBBackend" for s in trace.stages)


def test_protocol_b_backend_validates_required_payload():
    ctx = SolveContext(
        scenario=_scenario(),
        available_features={"load"},
        model_cols=["m1"],
        df_val=None,
        df_test=_prediction_frame(),
        horizon=1,
    )

    try:
        ProtocolBBackend().combine(ctx, SelectionTrace(scenario_id=ctx.scenario.scenario_id))
    except ValueError as exc:
        assert "df_val" in str(exc)
    else:
        raise AssertionError("ProtocolBBackend must validate required payload")
