import json
from types import SimpleNamespace

from src.core.solver.backends import CombinatorBackend
from src.core.solver.context import SolveContext
from src.core.trace import SelectionTrace
from src.graph.model_graph import ModelGraph
from src.graph.path_reasoning import PathReasoningScorer, ReasoningPath


def _graph_with_history():
    mg = ModelGraph()
    mg.add_scenario_node("s1", {})
    mg.add_feature_node("f1")
    mg.add_model_node("m1", metadata={"input_constraints": {"features": ["f1"]}})
    mg.add_scenario_feature_edge("s1", "f1")
    mg.add_feature_model_edge("f1", "m1")
    path_id = mg.instantiate_path(
        "path::single_model::m1", ["m1"], "single_model", evidence_ref="ev::path"
    )
    mg.add_scenario_path_edge("s1", path_id, performance_score=0.8, evidence_ref="ev::edge")
    return mg, path_id


def test_scorer_returns_reasoning_path_with_hops_and_final_score():
    mg, path_id = _graph_with_history()

    rp = PathReasoningScorer().score(mg, "s1", path_id, available_features={"f1"})

    assert isinstance(rp, ReasoningPath)
    assert rp.scenario_id == "s1"
    assert rp.path_id == path_id
    assert 0.0 <= rp.final_score <= 1.0
    assert {"feature_coverage", "historical_performance", "relation_strength"} <= set(rp.hop_scores)
    assert rp.hop_scores["feature_coverage"] == 1.0
    assert "s1" in rp.nodes and "m1" in rp.nodes and path_id in rp.nodes
    assert "ev::edge" in rp.evidence_refs


def test_missing_features_lower_coverage_and_final_score():
    mg, path_id = _graph_with_history()
    scorer = PathReasoningScorer()

    full = scorer.score(mg, "s1", path_id, available_features={"f1"})
    empty = scorer.score(mg, "s1", path_id, available_features=set())

    assert empty.hop_scores["feature_coverage"] < full.hop_scores["feature_coverage"]
    assert empty.final_score < full.final_score


def test_reasoning_path_to_dict_is_json_serializable():
    mg, path_id = _graph_with_history()

    rp = PathReasoningScorer().score(mg, "s1", path_id, available_features={"f1"})
    payload = json.dumps(rp.to_dict())

    assert path_id in payload
    assert "hop_scores" in payload


def test_infer_optimal_path_by_reasoning_can_return_details():
    mg, _ = _graph_with_history()

    results = mg.infer_optimal_path_by_reasoning("s1", {"f1"}, return_details=True)

    assert results
    assert all(isinstance(item, ReasoningPath) for item in results)
    scores = [item.final_score for item in results]
    assert scores == sorted(scores, reverse=True)


class _StubCombinator:
    def select_optimal_path(self, **kwargs):
        return {
            "models": ["m1"],
            "weights": {"m1": 1.0},
            "strategy": "single",
            "path_id": "path::single_model::m1",
        }


def test_combinator_backend_writes_reasoning_paths_to_trace():
    mg, path_id = _graph_with_history()
    ctx = SolveContext(
        scenario=SimpleNamespace(signature={"load_mean": 1.0}, scenario_id="s1"),
        available_features={"f1"},
        model_cols=["m1"],
        model_graph=mg,
    )
    trace = SelectionTrace(scenario_id="s1")

    CombinatorBackend(combinator=_StubCombinator()).combine(ctx, trace)

    stage = [s for s in trace.stages if s["stage"] == "CombinatorBackend"][0]
    reasoning = stage["outputs"].get("reasoning_paths")
    assert reasoning, "CombinatorBackend should surface top reasoning paths when model_graph is present"
    assert reasoning[0]["path_id"]
    assert "final_score" in reasoning[0]
    assert "ev::edge" in trace.evidence_refs
