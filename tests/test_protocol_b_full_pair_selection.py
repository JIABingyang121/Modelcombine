"""Protocol B 必须以完整 validation 流水线比较允许的二模型候选。"""
from __future__ import annotations

from itertools import combinations

import pytest

import src.eval.kg.protocol_b as protocol_b
from src.graph.model_graph import ModelGraph
from tests.task83_fixtures import MODELS, make_protocol_b_frames


def _fixed_result(pair, score):
    return {
        "val": {"mae": score},
        "eligible_pair": True,
        "degenerate_reason": None,
        "effective_models": list(pair),
    }


def _flow(result):
    return result["val"]["weight_meta"]["protocol_b_selection_meta"]["selection_flow"]


def test_production_selects_lowest_full_validation_pair(monkeypatch):
    """有效 incumbent 也必须与其余允许 pair 比较，不能只在退化时替换。"""
    df_val, df_test, raw_val, raw_test = make_protocol_b_frames()
    scores = {
        frozenset(("m1", "m2")): 1.0,
        frozenset(("m1", "m3")): 3.0,
        frozenset(("m2", "m3")): 4.0,
    }
    calls = []

    def fixed_evaluator(*_args, selected_models, **_kwargs):
        pair = list(selected_models)
        calls.append(pair)
        return _fixed_result(pair, scores[frozenset(pair)])

    monkeypatch.setattr(protocol_b, "evaluate_fixed_protocol_b_candidate", fixed_evaluator)
    result = protocol_b.kg_combination_with_features(
        df_val, df_test, raw_val, raw_test, list(MODELS), 1,
        dataset_name="pjm", base_model_cols=list(MODELS),
    )

    flow = _flow(result)
    assert {frozenset(pair) for pair in calls} == {
        frozenset(pair) for pair in combinations(MODELS, 2)
    }
    assert flow["pair_eligibility"]["incumbent_pair"] == ["m3", "m2"]
    assert set(flow["selector_output"]) == {"m1", "m2"}
    assert set(flow["pair_eligibility"]["selected_pair"]) == {"m1", "m2"}
    assert flow["pair_eligibility"]["outcome"] == "replaced_by_full_validation"
    assert {
        frozenset(row["models"]): row["full_validation_mae"]
        for row in flow["pair_eligibility"]["evaluated_pairs"]
    } == {
        frozenset(("m1", "m2")): pytest.approx(1.0),
        frozenset(("m1", "m3")): pytest.approx(3.0),
        frozenset(("m2", "m3")): pytest.approx(4.0),
    }


def test_full_validation_selection_does_not_evaluate_explicit_conflict_pair(monkeypatch):
    """完整 pair 比较仍受既有显式 conflict 约束。"""
    df_val, df_test, raw_val, raw_test = make_protocol_b_frames()
    relation_graph = ModelGraph()
    for model in MODELS:
        relation_graph.add_model_node(model, {})
    relation_graph.add_relation("m1", "m2", "conflict", weight=1.0)
    relation_graph.add_relation("m2", "m1", "conflict", weight=1.0)
    calls = []

    def fixed_evaluator(*_args, selected_models, **_kwargs):
        pair = list(selected_models)
        calls.append(pair)
        score = 1.0 if set(pair) == {"m1", "m3"} else 2.0
        return _fixed_result(pair, score)

    monkeypatch.setattr(protocol_b, "evaluate_fixed_protocol_b_candidate", fixed_evaluator)
    result = protocol_b.kg_combination_with_features(
        df_val, df_test, raw_val, raw_test, list(MODELS), 1,
        dataset_name="pjm", base_model_cols=list(MODELS), relation_graph=relation_graph,
    )

    flow = _flow(result)
    assert all(set(pair) != {"m1", "m2"} for pair in calls)
    assert set(flow["selector_output"]) == {"m1", "m3"}
