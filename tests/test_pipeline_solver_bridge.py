from pathlib import Path

from src.pipeline.main import PowerPredictionPipeline
from src.selector.combinator import PowerModelCombinator


def _pipeline_with_combinator():
    pipeline = PowerPredictionPipeline.__new__(PowerPredictionPipeline)
    pipeline.model_combinator = PowerModelCombinator(
        enable_adaptive_weights=False,
        enable_resource_prediction=False,
    )
    pipeline.historical_scenarios = []
    return pipeline


def test_pipeline_solver_bridge_matches_direct_combinator(tmp_path):
    pipeline = _pipeline_with_combinator()
    scenario_signature = {"load_mean": 100.0, "region_type": 1.0, "_scenario_id": "R_x"}
    available_models = ["m1", "m2"]
    constraints = {"max_latency": 500, "max_resource": 10.0}
    actual_columns = {"timestamp", "region", "load", "region_type"}

    direct = pipeline.model_combinator.select_optimal_path(
        scenario_signature=dict(scenario_signature),
        available_models=list(available_models),
        constraints=dict(constraints),
        model_graph=None,
        similar_scenarios=[],
        actual_data_columns=set(actual_columns),
    )

    wrapped, trace = pipeline._select_path_with_solver(
        scenario_signature=dict(scenario_signature),
        region="R",
        scenario_id="R_x",
        available_models=list(available_models),
        constraints=dict(constraints),
        model_graph=None,
        similar_scenarios=[],
        actual_columns=set(actual_columns),
        trace_path=tmp_path / "trace.json",
    )

    assert wrapped == direct
    assert trace.final_selection == direct["models"]
    assert trace.final_weights == direct["weights"]
    assert [s["stage"] for s in trace.stages] == ["CapabilityMatch", "CombinatorBackend"]
    assert (tmp_path / "trace.json").exists()


def test_pipeline_solver_mounts_temporal_relation_stage_when_flag_on(tmp_path, monkeypatch):
    from src.graph.model_graph import ModelGraph

    monkeypatch.setenv("MODELCOMBINE_PIPELINE_ENABLE_TEMPORAL_RELATIONS", "1")
    pipeline = _pipeline_with_combinator()
    mg = ModelGraph()
    mg.add_scenario_node("R_x", {})

    _, trace = pipeline._select_path_with_solver(
        scenario_signature={"load_mean": 100.0, "_scenario_id": "R_x"},
        region="R",
        scenario_id="R_x",
        available_models=["m1", "m2"],
        constraints={"max_latency": 500, "max_resource": 10.0},
        model_graph=mg,
        similar_scenarios=[],
        actual_columns={"timestamp", "region", "load"},
        trace_path=tmp_path / "trace.json",
    )

    stage_names = [s["stage"] for s in trace.stages]
    assert "TemporalRelationUpdate" in stage_names


def test_pipeline_solver_skips_temporal_relation_stage_by_default(tmp_path, monkeypatch):
    from src.graph.model_graph import ModelGraph

    monkeypatch.delenv("MODELCOMBINE_PIPELINE_ENABLE_TEMPORAL_RELATIONS", raising=False)
    pipeline = _pipeline_with_combinator()
    mg = ModelGraph()
    mg.add_scenario_node("R_x", {})

    _, trace = pipeline._select_path_with_solver(
        scenario_signature={"load_mean": 100.0, "_scenario_id": "R_x"},
        region="R",
        scenario_id="R_x",
        available_models=["m1", "m2"],
        constraints={"max_latency": 500, "max_resource": 10.0},
        model_graph=mg,
        similar_scenarios=[],
        actual_columns={"timestamp", "region", "load"},
        trace_path=tmp_path / "trace.json",
    )

    assert "TemporalRelationUpdate" not in [s["stage"] for s in trace.stages]


def _solve_with_env(pipeline, mg, tmp_path):
    return pipeline._select_path_with_solver(
        scenario_signature={"load_mean": 100.0, "_scenario_id": "R_x"},
        region="R",
        scenario_id="R_x",
        available_models=["m1", "m2"],
        constraints={"max_latency": 500, "max_resource": 10.0},
        model_graph=mg,
        similar_scenarios=[],
        actual_columns={"timestamp", "region", "load"},
        trace_path=tmp_path / "trace.json",
    )


def test_pipeline_solver_mounts_cascade_decider_when_flag_on(tmp_path, monkeypatch):
    from src.graph.model_graph import ModelGraph

    monkeypatch.setenv("MODELCOMBINE_PIPELINE_ENABLE_CASCADE", "1")
    pipeline = _pipeline_with_combinator()
    mg = ModelGraph()
    mg.add_scenario_node("R_x", {})

    _, trace = _solve_with_env(pipeline, mg, tmp_path)

    assert "CascadeDecide" in [s["stage"] for s in trace.stages]


def test_pipeline_solver_passes_substitute_kwargs_when_flag_on(tmp_path, monkeypatch):
    import src.pipeline.main as pipeline_main
    from src.graph.model_graph import ModelGraph

    monkeypatch.setenv("MODELCOMBINE_PIPELINE_ENABLE_SUBSTITUTE", "1")
    captured = {}
    real_build = pipeline_main.build_solver

    def _capture(mode, **kwargs):
        captured.update(kwargs)
        return real_build(mode, **kwargs)

    monkeypatch.setattr(pipeline_main, "build_solver", _capture)
    pipeline = _pipeline_with_combinator()
    mg = ModelGraph()
    mg.add_scenario_node("R_x", {})

    _solve_with_env(pipeline, mg, tmp_path)

    assert captured.get("allow_substitute") is True
    assert captured.get("substitute_graph") is mg


def test_pipeline_solver_cascade_and_substitute_off_by_default(tmp_path, monkeypatch):
    from src.graph.model_graph import ModelGraph

    monkeypatch.delenv("MODELCOMBINE_PIPELINE_ENABLE_CASCADE", raising=False)
    monkeypatch.delenv("MODELCOMBINE_PIPELINE_ENABLE_SUBSTITUTE", raising=False)
    pipeline = _pipeline_with_combinator()
    mg = ModelGraph()
    mg.add_scenario_node("R_x", {})

    _, trace = _solve_with_env(pipeline, mg, tmp_path)

    assert "CascadeDecide" not in [s["stage"] for s in trace.stages]


def test_pipeline_solver_trace_path_uses_reports_traces():
    pipeline = _pipeline_with_combinator()

    path = pipeline._combinator_trace_path("scenario_123")

    assert isinstance(path, Path)
    assert path.name == "combinator_solver_scenario_123.json"
    assert path.parent.name == "traces"
