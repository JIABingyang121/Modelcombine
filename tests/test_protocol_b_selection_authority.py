"""唯一候选选择权与 conflict 语义（Task 8.3 Task 3）。

`select_models_protocol_b` 是唯一改变 Protocol B 候选组合的模块：selector 输出后，
reasoning 与 conflict 阶段不得再静默改写模型集合。高误差相关只作连续惩罚，
只有外部图谱显式 `conflict` 边才构成硬约束。
"""
from __future__ import annotations

import pytest

from src.eval.kg import protocol_b as protocol_b_module
from src.eval.kg.conflict import check_conflict
from src.eval.kg.model_selection import select_models_protocol_b
from src.eval.kg.protocol_b import kg_combination_with_features
from src.eval.kg.reasoning_evidence import ReasoningEvidence
from src.graph.model_graph import ModelGraph
from tests.task83_fixtures import MODELS, make_protocol_b_frames


def _run(relation_graph=None, *, high_corr=False):
    df_val, df_test, raw_val, raw_test = make_protocol_b_frames(high_corr=high_corr)
    return kg_combination_with_features(
        df_val, df_test, raw_val, raw_test, MODELS, 1,
        dataset_name="pjm", base_model_cols=MODELS,
        relation_graph=relation_graph, relation_scenario_id="scenario_test",
    )


def test_selector_output_is_not_rewritten_before_fit(monkeypatch):
    monkeypatch.setattr(
        protocol_b_module,
        "build_reasoning_evidence",
        lambda **kwargs: ReasoningEvidence(
            contribution_by_model={"m1": 0.0, "m2": 0.0, "m3": 0.4},
            paths=[{"path_id": "path::m3", "evidence_refs": ["ev::history"]}],
            source="historical_evidence",
            cold_start_no_evidence=False,
        ),
    )
    result = _run()
    flow = result["val"]["weight_meta"]["protocol_b_selection_meta"]["selection_flow"]
    assert flow["selector_output"] == flow["final_selected_before_fit"]
    assert flow["post_selector_mutations"] == []


def test_high_error_correlation_is_soft_penalty_not_hard_exclusion():
    result = _run(high_corr=True)
    meta = result["val"]["weight_meta"]["protocol_b_selection_meta"]
    assert meta["pair_diagnostics"]["m1|m2"]["hard_conflict"] is False
    assert meta["pair_diagnostics"]["m1|m2"]["correlation_penalty"] > 0


def test_external_explicit_conflict_is_hard_constraint():
    """真实引擎：外部显式 conflict 边必须端到端过滤冲突对。

    两模型构造使候选自然想把 m1/m2 同时选入，conflict 过滤必须真实执行：
    selector_output 不得同时包含冲突对，constraint_decisions 必须留痕。
    """
    graph = ModelGraph()
    for model in ("m1", "m2"):
        graph.add_model_node(model, {})
    graph.add_relation("m1", "m2", "conflict", weight=1.0)
    df_val, df_test, raw_val, raw_test = make_protocol_b_frames(high_corr=False)
    result = kg_combination_with_features(
        df_val, df_test, raw_val, raw_test, ["m1", "m2"], 1,
        dataset_name="pjm", base_model_cols=["m1", "m2"],
        relation_graph=graph, relation_scenario_id="scenario_test",
    )
    meta = result["val"]["weight_meta"]["protocol_b_selection_meta"]
    flow = meta["selection_flow"]
    # B 的 selector_output 不得同时包含冲突对（不能只看可能经 guard 回退的最终输出）
    assert not {"m1", "m2"}.issubset(set(flow["selector_output"]))
    assert meta["pair_diagnostics"]["m1|m2"]["hard_conflict"] is True
    # 冲突过滤必须真实执行，出现在 constraint_decisions 里
    stages = [d["stage"] for d in flow["constraint_decisions"]]
    assert "explicit_conflict_filtering" in stages
    assert "minimum_not_met_due_to_explicit_conflict" in stages


def test_two_model_explicit_conflict_excludes_pair_from_selector_output():
    """两模型直接冲突：selector_output 不得同时包含冲突对，且允许少于最小数量。"""
    mg = ModelGraph()
    for m in ("m1", "m2"):
        mg.add_model_node(m, {})
    mg.add_scenario_node("scenario_h1", {})
    mg.add_relation("m1", "m2", "conflict", weight=1.0)
    maes = {"m1": 100.0, "m2": 101.0}
    selected, _scores, _bonus, meta = select_models_protocol_b(
        mg=mg,
        model_cols=["m1", "m2"],
        maes=maes,
        error_corrs={("m1", "m2"): 0.3},
        feat_model_corrs={},
        horizon=1,
        max_models=2,
    )
    assert not {"m1", "m2"}.issubset(set(selected)), f"冲突对被同时选择：{selected}"
    assert meta["pair_diagnostics"]["m1|m2"]["hard_conflict"] is True
    stages = [d["stage"] for d in meta["constraint_decisions"]]
    assert "minimum_not_met_due_to_explicit_conflict" in stages, (
        f"conflict 未优先于最小数量，constraint_decisions={meta['constraint_decisions']}"
    )


def test_base_model_injection_does_not_reintroduce_conflict():
    """base 注入不得重新引入与现有选择的显式冲突。"""
    mg = ModelGraph()
    for m in ("m_base", "m1", "m2"):
        mg.add_model_node(m, {})
    mg.add_scenario_node("scenario_h6", {})
    # m_base 与 m1 显式冲突，但 m_base 是唯一 base 模型，会被注入。
    mg.add_relation("m_base", "m1", "conflict", weight=1.0)
    maes = {"m_base": 105.0, "m1": 100.0, "m2": 101.0}
    error_corrs = {("m_base", "m1"): 0.2, ("m_base", "m2"): 0.2, ("m1", "m2"): 0.2}
    selected, _scores, _bonus, meta = select_models_protocol_b(
        mg=mg,
        model_cols=["m_base", "m1", "m2"],
        maes=maes,
        error_corrs=error_corrs,
        feat_model_corrs={},
        horizon=6,
        max_models=2,
        dataset_name="pjm",
        base_model_cols=["m_base"],
    )
    conflicts = []
    for i, x in enumerate(selected):
        for y in selected[i + 1:]:
            if check_conflict(mg, [x, y]):
                conflicts.append((x, y))
    assert conflicts == [], f"base 注入重新引入冲突：{selected} conflicts={conflicts}"
    assert "m_base" in selected, f"base 模型应仍被注入：{selected}"


def test_score_components_record_high_drift_factor_and_final_score():
    """高漂移扩展模型的 score_components 必须区分 total（乘系数前）与 final_score。

    total 是五项评分之和，final_score = total * high_drift_factor 才是真实用于
    候选排序的分数。只补审计字段，不改变评分行为。
    """
    mg = ModelGraph()
    for m in ("gating_network", "m_other"):
        mg.add_model_node(m, {})
    mg.add_scenario_node("scenario_h1", {})
    maes = {"gating_network": 100.0, "m_other": 101.0}
    error_corrs = {("gating_network", "m_other"): 0.3}
    _selected, scores, _bonus, meta = select_models_protocol_b(
        mg=mg,
        model_cols=["gating_network", "m_other"],
        maes=maes,
        error_corrs=error_corrs,
        feat_model_corrs={},
        horizon=1,
        max_models=2,
        drift_level="high",
    )
    comp = meta["score_components"]
    assert comp["gating_network"]["high_drift_factor"] == pytest.approx(0.8)
    assert comp["m_other"]["high_drift_factor"] == pytest.approx(1.0)
    assert comp["gating_network"]["final_score"] == pytest.approx(
        comp["gating_network"]["total"] * comp["gating_network"]["high_drift_factor"]
    )
    # final_score 必须等于真实候选排序分数（base_scores）
    for m in ("gating_network", "m_other"):
        assert comp[m]["final_score"] == pytest.approx(scores[m])
