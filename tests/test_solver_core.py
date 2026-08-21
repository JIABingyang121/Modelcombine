from src.core.enums import ModelLifecycleStage, TaskType
from src.core.index import IndexManager
from src.core.schema import DataContract, ModelManifest, ScenarioDefinition
from src.core.solver.context import CombinationBackend, SolveContext
from src.core.solver.solver import LogicalModelSelectionSolver
from src.core.solver.stages import make_capability_match_stage, make_similar_scenario_stage
from src.core.trace import SelectionTrace
from src.graph.model_graph import ModelGraph
from src.graph.relations import add_substitute_relation


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


class FakeBackend(CombinationBackend):
    name = "fake"

    def combine(self, ctx, trace):
        trace.add_stage("Combine", outputs={"backend": self.name})
        weight = 1.0 / len(ctx.model_cols)
        return {
            "models": list(ctx.model_cols),
            "weights": {model: weight for model in ctx.model_cols},
            "strategy": "fake",
            "path_id": "fake_path",
        }


def test_solver_runs_prestages_then_backend():
    calls = []

    def stage_a(ctx, trace):
        calls.append("A")
        trace.add_stage("A")

    def stage_b(ctx, trace):
        calls.append("B")
        trace.add_stage("B")

    ctx = SolveContext(scenario=_scenario(), available_features={"load"}, model_cols=["m1", "m2"])
    solver = LogicalModelSelectionSolver(
        pre_stages=[stage_a, stage_b],
        backend=FakeBackend(),
        post_stages=[],
    )

    result, trace = solver.solve(ctx)

    assert calls == ["A", "B"]
    assert result["models"] == ["m1", "m2"]
    assert result["strategy"] == "fake"
    assert trace.scenario_id.startswith("R_")
    assert [s["stage"] for s in trace.stages[:2]] == ["A", "B"]
    assert any(s["stage"] == "Combine" for s in trace.stages)
    assert trace.final_selection == ["m1", "m2"]


def test_solver_saves_trace(tmp_path):
    ctx = SolveContext(scenario=_scenario(), available_features={"load"}, model_cols=["m1"])
    solver = LogicalModelSelectionSolver(pre_stages=[], backend=FakeBackend(), post_stages=[])

    result, trace = solver.solve(ctx, trace_path=str(tmp_path / "trace.json"))

    assert result["models"] == ["m1"]
    assert (tmp_path / "trace.json").exists()


def test_capability_match_stage_prunes_and_traces():
    manifests = {
        "m_ok": ModelManifest(
            "m_ok",
            [TaskType.FORECASTING],
            ["load_forecast"],
            {"features": ["load"]},
            {"yhat": "float"},
            {},
            ModelLifecycleStage.ACTIVE,
        ),
        "m_clf": ModelManifest(
            "m_clf",
            [TaskType.CLASSIFICATION],
            ["load_forecast"],
            {"features": ["load"]},
            {"yhat": "float"},
            {},
            ModelLifecycleStage.ACTIVE,
        ),
    }
    ctx = SolveContext(scenario=_scenario(), available_features={"load"}, model_cols=["m_ok", "m_clf"])
    trace = SelectionTrace(scenario_id="R_x")

    make_capability_match_stage(manifests)(ctx, trace)

    assert ctx.model_cols == ["m_ok"]
    assert "m_clf" in trace.candidates_rejected
    assert any(s["stage"] == "CapabilityMatch" for s in trace.stages)


def test_capability_match_stage_uses_all_eligible_when_model_cols_empty():
    manifests = {
        "m_ok": ModelManifest(
            "m_ok",
            [TaskType.FORECASTING],
            ["load_forecast"],
            {"features": ["load"]},
            {"yhat": "float"},
            {},
            ModelLifecycleStage.ACTIVE,
        ),
    }
    ctx = SolveContext(scenario=_scenario(), available_features={"load"}, model_cols=[])
    trace = SelectionTrace(scenario_id="R_x")

    make_capability_match_stage(manifests)(ctx, trace)

    assert ctx.model_cols == ["m_ok"]


def test_capability_match_stage_can_substitute_rejected_candidate():
    graph = ModelGraph()
    add_substitute_relation(graph, "m_missing", "m_ok", weight=0.9)
    manifests = {
        "m_missing": ModelManifest(
            "m_missing",
            [TaskType.FORECASTING],
            ["load_forecast"],
            {"features": ["load", "weather"]},
            {"yhat": "float"},
            {},
            ModelLifecycleStage.ACTIVE,
        ),
        "m_ok": ModelManifest(
            "m_ok",
            [TaskType.FORECASTING],
            ["load_forecast"],
            {"features": ["load"]},
            {"yhat": "float"},
            {},
            ModelLifecycleStage.ACTIVE,
        ),
    }
    ctx = SolveContext(
        scenario=_scenario(),
        available_features={"load"},
        model_cols=["m_missing"],
    )
    trace = SelectionTrace(scenario_id="R_x")

    make_capability_match_stage(
        manifests,
        allow_substitute=True,
        substitute_graph=graph,
    )(ctx, trace)

    assert ctx.model_cols == ["m_ok"]
    result = ctx.extras["capability_match"]
    assert result.substitutions == {"m_missing": "m_ok"}


def test_similar_scenario_stage_populates_ctx():
    manager = IndexManager.with_defaults(
        scenario_records=[
            {
                "scenario_id": "h1",
                "signature": {"load": 1.0},
                "business_domain": "load_forecast",
                "performance": {"score": 0.9},
            },
        ]
    )
    ctx = SolveContext(scenario=_scenario(), available_features={"load"}, model_cols=["m_ok"])
    trace = SelectionTrace(scenario_id="R_x")

    make_similar_scenario_stage(manager, top_k=3)(ctx, trace)

    assert len(ctx.similar) >= 1
    assert ctx.similar[0]["scenario_id"] == "h1"
    assert any(s["stage"] == "SimilarScenarioRetrieve" for s in trace.stages)
