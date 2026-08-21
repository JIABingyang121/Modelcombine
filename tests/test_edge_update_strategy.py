"""三策略对照实验前置钩子：图谱状态路径覆盖 + 边权更新策略开关。"""
import src.pipeline.main as pipeline_main
from src.graph.model_graph import ModelGraph
from src.pipeline.main import PowerPredictionPipeline
from src.selector.combinator import PowerModelCombinator


def _pipeline():
    p = PowerPredictionPipeline.__new__(PowerPredictionPipeline)
    p.model_combinator = PowerModelCombinator(
        enable_adaptive_weights=False, enable_resource_prediction=False
    )
    p.historical_scenarios = []
    p.enable_phase2 = False
    p.enable_phase3 = False
    p.config = {"assets": {"models": [], "relations": [], "selection_rules": []}}
    import os as _os
    p.history_path = _os.devnull
    return p


def test_graph_state_path_env_override_used_for_load(tmp_path, monkeypatch):
    custom = tmp_path / "arm_hawkes_graph.pkl"
    marker = ModelGraph()
    marker.add_scenario_node("MARKER_SCENARIO", {})
    marker.save_graph(str(custom))
    monkeypatch.setenv("MODELCOMBINE_GRAPH_STATE_PATH", str(custom))

    mg = _pipeline().build_model_graph()

    assert mg.G.has_node("MARKER_SCENARIO")


def test_graph_state_path_defaults_to_reports_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("MODELCOMBINE_GRAPH_STATE_PATH", raising=False)
    monkeypatch.setattr(pipeline_main, "PROJECT_ROOT", str(tmp_path))

    assert pipeline_main._graph_state_path() == str(tmp_path / "reports" / "graph_state.pkl")


def _feedback(pipeline, mg):
    pipeline.historical_scenarios = [("sid", {}, {})]
    pipeline.feedback_loop(
        {"by_region": {"R": {"RMSE": 1.0}}, "model_comparison": {"R": {"m1": {"RMSE": 1.0}}}},
        mg,
        scenario_id_map={"R": "sid"},
        path_id_map={"R": "p1"},
    )


def _graph_with_feedback_edges():
    mg = ModelGraph()
    mg.add_scenario_node("sid", {})
    mg.add_model_node("m1", {})
    mg.instantiate_path("p1", ["m1"], "single")
    mg.add_scenario_path_edge("sid", "p1", performance_score=0.8)
    mg.add_scenario_model_edge("sid", "m1", weight=0.8)
    return mg


def test_feedback_ema_runs_under_default_strategy(monkeypatch):
    monkeypatch.delenv("MODELCOMBINE_PIPELINE_EDGE_UPDATE_STRATEGY", raising=False)
    mg = _graph_with_feedback_edges()

    _feedback(_pipeline(), mg)

    assert mg.G["sid"]["p1"]["weight"] != 0.8  # EMA 更新发生


def test_feedback_ema_skipped_under_fixed_and_hawkes(monkeypatch):
    for strategy in ("fixed", "hawkes"):
        monkeypatch.setenv("MODELCOMBINE_PIPELINE_EDGE_UPDATE_STRATEGY", strategy)
        mg = _graph_with_feedback_edges()

        _feedback(_pipeline(), mg)

        assert mg.G["sid"]["p1"]["weight"] == 0.8, strategy
        assert mg.G["sid"]["m1"]["weight"] == 0.8, strategy


def test_strategy_hawkes_mounts_temporal_stage_without_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("MODELCOMBINE_PIPELINE_ENABLE_TEMPORAL_RELATIONS", raising=False)
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_EDGE_UPDATE_STRATEGY", "hawkes")
    mg = ModelGraph()
    mg.add_scenario_node("R_x", {})

    _, trace = _pipeline()._select_path_with_solver(
        scenario_signature={"load_mean": 100.0, "_scenario_id": "R_x"},
        region="R", scenario_id="R_x",
        available_models=["m1"], constraints={},
        model_graph=mg, similar_scenarios=[],
        actual_columns={"load"}, trace_path=tmp_path / "t.json",
    )

    assert "TemporalRelationUpdate" in [s["stage"] for s in trace.stages]


def test_strategy_fixed_suppresses_temporal_stage_even_with_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_ENABLE_TEMPORAL_RELATIONS", "1")
    monkeypatch.setenv("MODELCOMBINE_PIPELINE_EDGE_UPDATE_STRATEGY", "fixed")
    mg = ModelGraph()
    mg.add_scenario_node("R_x", {})

    _, trace = _pipeline()._select_path_with_solver(
        scenario_signature={"load_mean": 100.0, "_scenario_id": "R_x"},
        region="R", scenario_id="R_x",
        available_models=["m1"], constraints={},
        model_graph=mg, similar_scenarios=[],
        actual_columns={"load"}, trace_path=tmp_path / "t.json",
    )

    assert "TemporalRelationUpdate" not in [s["stage"] for s in trace.stages]
