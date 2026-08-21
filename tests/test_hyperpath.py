from datetime import datetime, timezone

from src.core.trace import SelectionTrace
from src.graph.hyperpath import (
    add_scenario_hyperpath_edge,
    apply_hyperpath_temporal_update,
    get_hyperpaths_for_model,
    get_top_hyperpaths_for_scenario,
    instantiate_hyperpath,
    record_hyperpath_selection,
    update_hyperpath_metrics,
)
from src.graph.model_graph import ModelGraph
from src.graph.temporal_relations import HawkesRelationUpdater, RelationEvent


NOW = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)


def _graph_with_models(*model_ids):
    graph = ModelGraph()
    for model_id in model_ids:
        graph.add_model_node(model_id, {"task_type": "forecasting"})
    return graph


def test_instantiate_hyperpath_normalizes_order_insensitive_members():
    graph = _graph_with_models("catboost", "xgboost", "lgbm")

    path_a = instantiate_hyperpath(
        graph,
        models=["xgboost", "catboost", "lgbm"],
        strategy="weighted_mean",
        created_from="unit_test",
    )
    path_b = instantiate_hyperpath(
        graph,
        models=["lgbm", "xgboost", "catboost"],
        strategy="weighted_mean",
        created_from="unit_test",
    )

    assert path_a == path_b
    node = graph.G.nodes[path_a]
    assert node["node_type"] == "path"
    assert node["is_hyperedge"] is True
    assert node["members"] == ["catboost", "lgbm", "xgboost"]
    assert node["composition"] == ["catboost", "lgbm", "xgboost"]
    assert node["member_count"] == 3
    assert node["ordered"] is False
    assert node["created_from"] == "unit_test"
    assert graph.G["catboost"][path_a]["edge_type"] == "part_of"
    assert graph.G["catboost"][path_a]["order"] == 0


def test_order_sensitive_hyperpath_keeps_sequence_distinct():
    graph = _graph_with_models("scorecard", "lgbm")

    first = instantiate_hyperpath(
        graph,
        models=["scorecard", "lgbm"],
        strategy="stacking",
    )
    second = instantiate_hyperpath(
        graph,
        models=["lgbm", "scorecard"],
        strategy="stacking",
    )

    assert first != second
    assert graph.G.nodes[first]["ordered"] is True
    assert graph.G.nodes[first]["members"] == ["scorecard", "lgbm"]
    assert graph.G.nodes[second]["members"] == ["lgbm", "scorecard"]


def test_model_graph_instantiate_path_uses_hyperedge_representation():
    graph = _graph_with_models("xgboost", "lgbm")

    path_id = graph.instantiate_path(
        "legacy_path",
        ["xgboost", "lgbm"],
        "weighted_mean",
        created_from="legacy_api",
        evidence_ref="trace:legacy",
    )

    node = graph.G.nodes[path_id]
    assert node["is_hyperedge"] is True
    assert node["members"] == ["lgbm", "xgboost"]
    assert node["created_from"] == "legacy_api"
    assert node["evidence_refs"] == ["trace:legacy"]


def test_reinstantiating_hyperpath_preserves_existing_metrics():
    graph = _graph_with_models("xgboost", "lgbm")
    path_id = instantiate_hyperpath(
        graph,
        models=["xgboost", "lgbm"],
        strategy="weighted_mean",
        metrics={"mae": 1.1, "dynamic_strength": 0.7},
        evidence_ref="eval:first",
    )

    instantiate_hyperpath(
        graph,
        models=["lgbm", "xgboost"],
        strategy="weighted_mean",
        evidence_ref="trace:second",
    )

    node = graph.G.nodes[path_id]
    assert node["metrics"] == {"mae": 1.1, "dynamic_strength": 0.7}
    assert node["mae"] == 1.1
    assert node["dynamic_strength"] == 0.7
    assert node["evidence_refs"] == ["eval:first", "trace:second"]


def test_hyperpath_metrics_scenario_edges_queries_and_trace_refs():
    graph = _graph_with_models("catboost", "xgboost")
    graph.add_scenario_node("scenario_pjm_h1", {"signature": {}})
    path_id = instantiate_hyperpath(
        graph,
        models=["catboost", "xgboost"],
        strategy="weighted_mean",
        evidence_ref="trace:p1",
    )

    update_hyperpath_metrics(
        graph,
        path_id,
        {
            "mae": 1.2,
            "latency_ms": 15.5,
            "drift_level": "low",
            "uncertainty_score": 0.08,
            "dynamic_strength": 0.72,
        },
        evidence_ref="eval:p1",
    )
    add_scenario_hyperpath_edge(
        graph,
        "scenario_pjm_h1",
        path_id,
        relation_type="selected_for",
        weight=0.91,
        evidence_ref="trace:p1",
    )

    node = graph.G.nodes[path_id]
    assert node["mae"] == 1.2
    assert node["latency_ms"] == 15.5
    assert node["evidence_refs"] == ["trace:p1", "eval:p1"]
    assert graph.G["scenario_pjm_h1"][path_id]["evidence_refs"] == ["trace:p1"]
    assert get_top_hyperpaths_for_scenario(graph, "scenario_pjm_h1")[0]["path_id"] == path_id
    assert get_hyperpaths_for_model(graph, "catboost")[0]["path_id"] == path_id

    trace = SelectionTrace(
        scenario_id="scenario_pjm_h1",
        timestamp="2026-07-02T12:00:00Z",
    )
    record_hyperpath_selection(
        trace,
        path_id,
        score=0.91,
        evidence_ref="trace:p1",
    )
    stage = next(s for s in trace.stages if s["stage"] == "HyperPathSelect")
    assert stage["outputs"]["path_id"] == path_id
    assert trace.evidence_refs == ["trace:p1"]


def test_hyperpath_temporal_update_updates_path_and_member_edges():
    graph = _graph_with_models("catboost", "xgboost")
    graph.add_scenario_node("scenario_pjm_h1", {"signature": {}})
    path_id = instantiate_hyperpath(
        graph,
        models=["catboost", "xgboost"],
        strategy="weighted_mean",
    )
    add_scenario_hyperpath_edge(
        graph,
        "scenario_pjm_h1",
        path_id,
        relation_type="recommended_for",
        weight=0.5,
    )
    event = RelationEvent(
        source="scenario_pjm_h1",
        target=path_id,
        relation_type="recommended_for",
        timestamp=NOW,
        polarity="positive",
        magnitude=1.0,
        evidence_ref="trace:p1",
    )

    summary = apply_hyperpath_temporal_update(
        graph,
        [event],
        now=NOW,
        updater=HawkesRelationUpdater(base_strength=0.5, alpha=0.3, beta=0.0),
    )

    assert summary["path_updates"][0]["path_id"] == path_id
    assert graph.G["scenario_pjm_h1"][path_id]["dynamic_strength"] == 0.8
    assert graph.G.nodes[path_id]["dynamic_strength"] == 0.8
    assert graph.G.nodes[path_id]["event_evidence_refs"] == ["trace:p1"]
    assert graph.G["catboost"][path_id]["dynamic_strength"] == 0.65
    assert graph.G["xgboost"][path_id]["dynamic_strength"] == 0.65
