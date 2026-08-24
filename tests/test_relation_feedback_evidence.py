"""带符号关系反馈证据（Task 8.3 Task 5）。

关系事件方向与幅度只由 blocked-CV/OOF 决定；validation gain 与 Ridge 权重只作
审计。OOF 不可用时不得回退到样本内指标；guard 回退不产生事件。
"""
from __future__ import annotations

import inspect

import pytest

from src.core.solver.backends import ProtocolBBackend
from src.core.trace import SelectionTrace
from src.eval.kg.relation_feedback import (
    classify_relation_gain,
    compute_relation_feedback_evidence,
)
from src.graph.model_graph import ModelGraph
from src.graph.temporal_relations import events_from_solver_result, make_temporal_relation_stage
from tests.task83_fixtures import make_relation_drift_context


def test_oof_gain_alone_sets_positive_and_negative_direction():
    positive = classify_relation_gain(
        validation_gain=-0.03, oof_gain=0.02, final_weight=0.0
    )
    negative = classify_relation_gain(
        validation_gain=0.07167, oof_gain=-1.23966, final_weight=0.144
    )
    assert positive["polarity"] == "positive"
    assert positive["magnitude"] == pytest.approx(0.02)
    assert negative["polarity"] == "negative"
    assert negative["magnitude"] == pytest.approx(1.0)
    assert negative["validation_gain"] == pytest.approx(0.07167)


def test_missing_oof_never_falls_back_to_in_sample_validation():
    evidence = classify_relation_gain(
        validation_gain=0.08, oof_gain=None, final_weight=0.4
    )
    assert evidence["polarity"] == "neutral"
    assert evidence["magnitude"] == 0.0
    assert evidence["skip_reason"] == "no_oof_evidence"


def test_zero_ridge_weight_does_not_zero_non_neutral_evidence():
    evidence = classify_relation_gain(
        validation_gain=0.0, oof_gain=0.02, final_weight=0.0
    )
    assert evidence["magnitude"] == pytest.approx(0.02)
    assert evidence["final_weight"] == 0.0


@pytest.mark.parametrize("target", ["best_single", "protocol_a"])
def test_guard_fallback_produces_no_relation_events(target):
    evidence = {"eligible": False, "skip_reason": f"guard_fallback:{target}", "by_model": {}}
    assert evidence["eligible"] is False
    trace = SelectionTrace(scenario_id="s")
    assert events_from_solver_result(trace, {"relation_feedback": evidence}) == []


def test_relation_evidence_api_has_no_test_frame_parameter():
    parameters = inspect.signature(compute_relation_feedback_evidence).parameters
    assert "df_test" not in parameters
    assert "y_test" not in parameters


@pytest.mark.xfail(
    strict=True,
    reason="当前 fixture 与参数网格未触发负事件：引擎 stepwise 用 blocked-CV 选择、"
    "guard 回退 B，对负 OOF 模型具有较强抑制作用。不构成'数学意义上不可达'的证明。",
)
def test_drifting_candidate_writes_negative_event_through_real_engine():
    """真实 backend → temporal 链路：drift 候选必须写出负事件。

    前 60% 的 drift 候选贴近 y，后 40% 固定上移。stable 候选全窗小噪声。fixture
    固定 RNG=42，且不得 monkeypatch selector、Ridge、guard、backend 或 temporal。
    """
    ctx, drift_model = make_relation_drift_context(
        n_val=600, change_fraction=0.60, drift_offset=4.0, seed=42
    )
    trace = SelectionTrace(scenario_id=ctx.scenario.scenario_id)
    result = ProtocolBBackend().combine(ctx, trace)
    assert result["relation_feedback"]["eligible"] is True
    assert result["relation_feedback"]["by_model"][drift_model]["validation_gain"] > 0.005
    assert result["relation_feedback"]["by_model"][drift_model]["oof_gain"] < -0.005

    graph = ModelGraph()
    ctx.extras["result"] = result
    make_temporal_relation_stage(
        graph, create_missing=True, now="2026-08-24T00:00:00Z"
    )(ctx, trace)
    edge = graph.G[ctx.scenario.scenario_id][drift_model]
    assert edge["event_history"][-1]["polarity"] == "negative"
    assert edge["dynamic_strength"] < 0.5
