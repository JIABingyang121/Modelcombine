import json
import os
import subprocess
import sys
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


def test_reasoning_tie_break_is_stable_across_hash_seeds():
    """等分路径的首选不能随 Python 进程哈希种子改变。

    生产中的 Protocol B 默认使用 reasoning=hybrid；若多个冷启动路径同分，首条
    推理路径会参与后续模型合并。因此这里必须跨独立进程验证，而不只测单进程。
    当前实现遍历 set，seed=1/2 会分别产生不同首路径。
    """
    program = """
from src.graph.model_graph import ModelGraph
graph = ModelGraph()
graph.add_scenario_node('s', {})
graph.add_feature_node('f')
for model in ['arima', 'catboost_reg', 'lgbm_reg', 'xgboost_reg']:
    graph.add_model_node(model, {'input_constraints': {'features': ['f']}})
    graph.add_feature_model_edge('f', model)
graph.add_scenario_feature_edge('s', 'f')
print(graph.infer_optimal_path_by_reasoning('s', {'f'})[0][0])
"""
    project_root = os.path.dirname(os.path.dirname(__file__))
    chosen = []
    for seed in ('1', '2'):
        env = {**os.environ, 'PYTHONHASHSEED': seed, 'PYTHONPATH': project_root}
        output = subprocess.check_output(
            [sys.executable, '-c', program], text=True, env=env,
        ).strip()
        chosen.append(output)

    assert chosen[0] == chosen[1], (
        f"同分推理路径随 hash seed 改变：{chosen[0]} vs {chosen[1]}"
    )


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
