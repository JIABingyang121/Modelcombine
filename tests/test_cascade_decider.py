from src.core.enums import TaskType
from src.core.schema import DataContract, ScenarioDefinition
from src.core.solver.cascade import CascadeDecider
from src.core.solver.context import SolveContext
from src.core.trace import SelectionTrace
from src.graph.model_graph import ModelGraph
from src.graph.relations import add_cascade_relation


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
        signature_features=["load"],
        signature={"load": 1.0},
        region="R",
    )


def test_cascade_decider_replaces_high_uncertainty_model():
    graph = ModelGraph()
    add_cascade_relation(
        graph,
        "seasonal_naive",
        "lgbm_reg",
        cost_delta=20.0,
        uncertainty_gain=0.5,
        weight=0.9,
    )
    ctx = SolveContext(
        scenario=_scenario(),
        available_features={"load"},
        model_cols=["seasonal_naive", "lgbm_reg"],
    )
    ctx.extras["result"] = {
        "models": ["seasonal_naive"],
        "weights": {"seasonal_naive": 1.0},
    }
    ctx.uncertainty = {"seasonal_naive": 0.8, "lgbm_reg": 0.2}
    trace = SelectionTrace(scenario_id=ctx.scenario.scenario_id)

    CascadeDecider(graph=graph, uncertainty_threshold=0.5).apply(ctx, trace)

    assert ctx.extras["result"]["models"] == ["lgbm_reg"]
    assert ctx.extras["result"]["weights"] == {"lgbm_reg": 1.0}
    assert "seasonal_naive" in trace.candidates_rejected
    stage = next(s for s in trace.stages if s["stage"] == "CascadeDecide")
    assert stage["metadata"]["bypass_applied"] is True
    assert stage["outputs"]["cascades"] == {"seasonal_naive": "lgbm_reg"}


def test_cascade_decider_ignores_unavailable_target():
    graph = ModelGraph()
    add_cascade_relation(
        graph,
        "seasonal_naive",
        "lgbm_reg",
        cost_delta=20.0,
        uncertainty_gain=0.5,
        weight=0.9,
    )
    ctx = SolveContext(
        scenario=_scenario(),
        available_features={"load"},
        model_cols=["seasonal_naive"],
    )
    ctx.extras["result"] = {
        "models": ["seasonal_naive"],
        "weights": {"seasonal_naive": 1.0},
    }
    ctx.uncertainty = {"seasonal_naive": 0.8}
    trace = SelectionTrace(scenario_id=ctx.scenario.scenario_id)

    CascadeDecider(graph=graph, uncertainty_threshold=0.5).apply(ctx, trace)

    assert ctx.extras["result"]["models"] == ["seasonal_naive"]
    stage = next(s for s in trace.stages if s["stage"] == "CascadeDecide")
    assert stage["metadata"]["bypass_applied"] is False
    assert stage["outputs"]["cascades"] == {}


def test_cascade_decider_keeps_low_uncertainty_selection():
    graph = ModelGraph()
    add_cascade_relation(graph, "seasonal_naive", "lgbm_reg", 20.0, 0.5)
    ctx = SolveContext(
        scenario=_scenario(),
        available_features={"load"},
        model_cols=["seasonal_naive", "lgbm_reg"],
    )
    ctx.extras["result"] = {
        "models": ["seasonal_naive"],
        "weights": {"seasonal_naive": 1.0},
    }
    ctx.uncertainty = {"seasonal_naive": 0.1}
    trace = SelectionTrace(scenario_id=ctx.scenario.scenario_id)

    CascadeDecider(graph=graph, uncertainty_threshold=0.5).apply(ctx, trace)

    assert ctx.extras["result"]["models"] == ["seasonal_naive"]
    stage = next(s for s in trace.stages if s["stage"] == "CascadeDecide")
    assert stage["metadata"]["decision_authority"] == "audit_only"
