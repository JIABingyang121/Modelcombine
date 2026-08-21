from src.graph.model_graph import ModelGraph
from src.graph.relations import (
    add_cascade_relation,
    add_substitute_relation,
    get_cascade_options,
    get_relations_by_type,
    get_substitutes,
)


def test_substitute_relation_is_bidirectional_and_queryable():
    graph = ModelGraph()

    add_substitute_relation(
        graph,
        "lgbm_reg",
        "xgboost_reg",
        weight=0.83,
        reason="same task/domain/features",
    )

    assert get_substitutes(graph, "lgbm_reg") == ["xgboost_reg"]
    assert get_substitutes(graph, "xgboost_reg") == ["lgbm_reg"]
    relation = get_relations_by_type(graph, "lgbm_reg", "substitute")[0]
    assert relation["source"] == "lgbm_reg"
    assert relation["target"] == "xgboost_reg"
    assert relation["weight"] == 0.83
    assert relation["reason"] == "same task/domain/features"


def test_cascade_relation_is_directional_and_keeps_cost_metadata():
    graph = ModelGraph()

    add_cascade_relation(
        graph,
        "seasonal_naive",
        "lgbm_reg",
        cost_delta=25.0,
        uncertainty_gain=0.4,
        weight=0.9,
    )

    options = get_cascade_options(graph, "seasonal_naive")
    assert [item["target"] for item in options] == ["lgbm_reg"]
    assert options[0]["cost_delta"] == 25.0
    assert options[0]["uncertainty_gain"] == 0.4
    assert get_cascade_options(graph, "lgbm_reg") == []


def test_relation_query_filters_by_weight_and_sorts_descending():
    graph = ModelGraph()
    add_substitute_relation(graph, "m1", "m2", weight=0.2, bidirectional=False)
    add_substitute_relation(graph, "m1", "m3", weight=0.9, bidirectional=False)

    relations = get_relations_by_type(graph, "m1", "substitute", min_weight=0.5)

    assert [item["target"] for item in relations] == ["m3"]


def test_scenario_model_edge_is_direct_and_does_not_require_path_hyperedge():
    graph = ModelGraph()
    graph.add_scenario_node("scenario_1", {})
    graph.add_model_node("lgbm_reg", {})

    graph.add_scenario_model_edge("scenario_1", "lgbm_reg", weight=0.5)

    assert graph.G.has_edge("scenario_1", "lgbm_reg")
    assert graph.G["scenario_1"]["lgbm_reg"]["edge_type"] == "recommended_for"
    assert graph.G["scenario_1"]["lgbm_reg"]["weight"] == 0.5

    graph.update_edge_weight("scenario_1", "lgbm_reg", feedback_score=0.9, learning_rate=0.5)

    assert graph.G["scenario_1"]["lgbm_reg"]["weight"] == 0.7
