from src.core.enums import TaskType
from src.core.schema import DataContract, ScenarioDefinition
from src.core.solver.backends import CombinatorBackend
from src.core.solver.context import SolveContext
from src.core.trace import SelectionTrace
from src.selector.combinator import PowerModelCombinator


def _scenario():
    dc = DataContract(
        required_columns={"load": "float"},
        freq="H",
        min_samples=50,
        business_domain="load_forecast",
    )
    return ScenarioDefinition(
        task_type=TaskType.FORECASTING,
        business_domain="load_forecast",
        data_contract=dc,
        target_schema={"yhat": "float"},
        primary_metric="MAE",
        signature_features=["load", "region_type"],
        signature={"load": 1.0, "region_type": 1.0},
        region="R",
    )


def _combinator():
    return PowerModelCombinator(enable_adaptive_weights=False, enable_resource_prediction=False)


def test_combinator_backend_matches_direct_combinator_output():
    scenario = _scenario()
    model_cols = ["m1", "m2"]
    constraints = {"max_latency": 500, "max_resource": 10.0}
    combinator = _combinator()
    direct = combinator.select_optimal_path(
        scenario_signature=dict(scenario.signature),
        available_models=model_cols,
        constraints=constraints,
        model_graph=None,
        similar_scenarios=[],
        actual_data_columns={"load", "region_type"},
    )
    ctx = SolveContext(
        scenario=scenario,
        available_features={"load", "region_type"},
        model_cols=model_cols,
        constraints=constraints,
        similar_scenarios=[],
    )

    wrapped = CombinatorBackend(combinator=combinator).combine(
        ctx,
        SelectionTrace(scenario_id=scenario.scenario_id),
    )

    assert wrapped["models"] == direct["models"]
    assert wrapped["weights"] == direct["weights"]
    assert wrapped["strategy"] == direct["strategy"]
    assert wrapped["path_id"] == direct["path_id"]
    assert wrapped["raw"] == direct


def test_combinator_backend_writes_trace_stage():
    scenario = _scenario()
    trace = SelectionTrace(scenario_id=scenario.scenario_id)
    ctx = SolveContext(
        scenario=scenario,
        available_features={"load", "region_type"},
        model_cols=["m1"],
        constraints={},
    )

    result = CombinatorBackend(combinator=_combinator()).combine(ctx, trace)

    assert result["models"] == ["m1"]
    stage = next(s for s in trace.stages if s["stage"] == "CombinatorBackend")
    assert stage["outputs"]["models"] == ["m1"]
