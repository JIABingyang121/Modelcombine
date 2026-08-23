"""关系强度参与 Protocol B 候选评分（§11#7 的落点）。

**背景**：Hawkes/反馈会把关系强度写进图谱（`recommended_for` 边的
`weight` / `dynamic_strength`），但此前**没有任何决策路径读它**——Task 6C 的最小
因果实验已确认，边权只在历史 RMSE 恰好平局时才通过候选顺序间接影响旧
combinator，真实数据几乎不触发。§11#7 因此把"边权是否真正参与决策"列为待决策。

按既定方向，消费点做在 **Protocol B 侧**（旧 combinator 即将退役，不在其上加）。

**位置**：`select_models_protocol_b` 的 `base_scores[m]`。该处已经消费成对
`complementary` 边权（`comp_bonus`），缺的是**逐模型**的关系强度。

**公式**（保持简单、可解释）::

    score = base_norm + comp_bonus + feature_bonus_weight * feature_bonus
            + relation_strength_weight * (relation_strength[m] - NEUTRAL)

`relation_strength[m]` 取图中指向该模型的 `recommended_for` 边强度，缺省为中性值
0.5。因此**图中没有关系边时该项恒为 0，行为完全不变**——这保证既有实验证据不会
被动失效。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.kg.model_selection import select_models_protocol_b
from src.graph.model_graph import ModelGraph

SCENARIO = "scenario_pjm_h1"
M_A, M_B, M_C = "cand_a", "cand_b", "cand_c"
ALL = [M_A, M_B, M_C]


def _graph(**strengths: float) -> ModelGraph:
    """三个候选两两互补（权重对称），可选地各带一条 scenario->model 关系边。"""
    mg = ModelGraph()
    for m in ALL:
        mg.add_model_node(m, {})
    # 对称互补边：comp_bonus 对三者完全相同，排除该项造成的差异
    for i, m1 in enumerate(ALL):
        for m2 in ALL[i + 1:]:
            mg.add_relation(m1, m2, "complementary", weight=0.5)
            mg.add_relation(m2, m1, "complementary", weight=0.5)

    mg.add_scenario_node(SCENARIO, {})
    for m, strength in strengths.items():
        mg.add_scenario_model_edge(SCENARIO, m, weight=strength)
    return mg


def _near_tied_inputs():
    """三个候选 MAE 几乎相同，使关系强度成为唯一可区分因素。"""
    maes = {M_A: 100.0, M_B: 100.05, M_C: 100.10}
    error_corrs = {(M_A, M_B): 0.5, (M_A, M_C): 0.5, (M_B, M_C): 0.5}
    feat_model_corrs = {("f1", m): 0.30 for m in ALL}
    n = 240
    rng = np.random.default_rng(3)
    y = np.linspace(100, 120, n)
    df_val = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
        "y": y,
        M_A: y + rng.normal(0, 1.0, n),
        M_B: y + rng.normal(0, 1.0, n),
        M_C: y + rng.normal(0, 1.0, n),
    })
    return maes, error_corrs, feat_model_corrs, df_val


def _score(mg, max_models: int = 2):
    """走真实决策路径：max_models=2、三选二。

    注意不要用 max_models=1——那会掉进 `len(selected) < 2` 的兜底分支，
    该分支按 MAE 排序、完全绕开候选评分，测不到关系强度。
    """
    maes, error_corrs, feat_model_corrs, df_val = _near_tied_inputs()
    selected, scores, _bonus, meta = select_models_protocol_b(
        mg=mg,
        model_cols=list(ALL),
        maes=maes,
        error_corrs=error_corrs,
        feat_model_corrs=feat_model_corrs,
        horizon=1,
        df_val=df_val,
        max_models=max_models,
    )
    return selected, scores, meta


def test_without_relation_edges_scores_are_near_tied():
    """前提校验：没有关系边时两者几乎打平，后续断言才有意义。"""
    _selected, scores, _meta = _score(_graph())

    spread = max(scores.values()) - min(scores.values())
    assert spread < 1e-3, (
        f"构造的候选未打平：{scores}；关系强度实验需要近似平局作为基线"
    )


def test_relation_strength_changes_score_ranking():
    """关系强度必须改变候选得分与排序。"""
    _s1, scores_c_strong, _ = _score(_graph(**{M_C: 0.95}))
    _s2, scores_c_weak, _ = _score(_graph(**{M_C: 0.05}))

    # C 关系强时得分应最高；关系弱时应最低
    assert scores_c_strong[M_C] == max(scores_c_strong.values()), scores_c_strong
    assert scores_c_weak[M_C] == min(scores_c_weak.values()), scores_c_weak

    rank_strong = sorted(ALL, key=lambda m: -scores_c_strong[m])
    rank_weak = sorted(ALL, key=lambda m: -scores_c_weak[m])
    assert rank_strong != rank_weak, "关系强度未改变排序"
    assert rank_strong[0] == M_C


def test_relation_strength_changes_final_selection():
    """三选二时，只改一条关系强度必须改变最终被选中的模型集合。"""
    # 基线：C 的 MAE 最差，关系中性 -> 通常不入选
    sel_neutral, _, _ = _score(_graph())
    # 只把 C 的关系强度拉满 -> C 应进入最终选择
    sel_c_strong, _, _ = _score(_graph(**{M_C: 0.95}))

    assert set(sel_neutral) != set(sel_c_strong), (
        f"关系强度未改变最终选择：中性={sel_neutral} 强化后={sel_c_strong}"
    )
    assert M_C in sel_c_strong, f"关系强度拉满的候选未入选：{sel_c_strong}"


def test_absent_relation_edges_leave_scores_unchanged():
    """无关系边时该项必须恒为 0——否则会静默改变既有实验结果。"""
    _s1, baseline, _ = _score(_graph())
    # 只给中性强度 0.5，等价于"无信息"
    _s2, neutral, _ = _score(_graph(**{m: 0.5 for m in ALL}))

    for m in ALL:
        assert baseline[m] == pytest.approx(neutral[m], abs=1e-12), (
            f"中性关系强度改变了得分：{m} {baseline[m]} != {neutral[m]}"
        )


def test_selection_meta_records_relation_strength_terms():
    """评分项必须可审计：关系强度、权重与逐模型贡献都要能取到。"""
    _sel, _scores, meta = _score(_graph(**{M_A: 0.95, M_B: 0.05}))

    rel = meta.get("relation_strength")
    assert rel is not None, "selection meta 未记录 relation_strength"
    assert rel["weight"] > 0
    assert rel["neutral"] == 0.5
    assert rel["by_model"][M_A] == pytest.approx(0.95)
    assert rel["by_model"][M_B] == pytest.approx(0.05)
    # 逐模型贡献量（已乘权重、已减中性）
    assert rel["contribution"][M_A] > 0
    assert rel["contribution"][M_B] < 0


# --- SelectionTrace 留痕 -----------------------------------------------------


def test_backend_trace_records_relation_strength_scoring_and_ranking():
    """ProtocolBBackend 必须把关系强度、评分项、排序与最终选择写进 trace。

    否则"为什么这条候选排在前面"在事后无法回答——这正是本项目对可审计性的
    基本要求（关系强度更新必须能回溯到决策）。
    """
    from src.core.solver.backends import ProtocolBBackend
    from src.core.trace import SelectionTrace

    raw = {
        "val": {"mae": 1.0, "selected_models": [M_A], "weights": {M_A: 1.0}},
        "test": {
            "mae": 1.2,
            "selected_models": [M_A],
            "weights": {M_A: 1.0},
            "model_scores_b": {M_A: 1.90, M_B: 1.50, M_C: 1.70},
            "weight_meta": {
                "protocol_b_selection_meta": {
                    "relation_strength": {
                        "weight": 0.3,
                        "neutral": 0.5,
                        "by_model": {M_A: 0.95, M_B: 0.05, M_C: 0.50},
                        "contribution": {M_A: 0.135, M_B: -0.135, M_C: 0.0},
                        "edges_found": [M_A, M_B],
                    },
                },
            },
        },
        "protocol": "B_pred_features",
    }

    audit = ProtocolBBackend._relation_scoring_audit(raw)

    assert audit["relation_strength"]["by_model"][M_A] == pytest.approx(0.95)
    assert audit["relation_strength"]["contribution"][M_B] < 0
    # 排序必须按候选得分从高到低给出，便于直接回答"谁排在前面"
    assert audit["candidate_ranking"] == [M_A, M_C, M_B]
    assert audit["final_selection"] == [M_A]


def test_backend_trace_audit_tolerates_missing_relation_meta():
    """回退分支不带 selection meta 时不得抛错，只记为空。"""
    from src.core.solver.backends import ProtocolBBackend

    raw = {"test": {"selected_models": [M_A], "weights": {M_A: 1.0}}, "protocol": "B_fallback_to_A_guard"}

    audit = ProtocolBBackend._relation_scoring_audit(raw)

    assert audit["relation_strength"] is None
    assert audit["candidate_ranking"] == []
    assert audit["final_selection"] == [M_A]
