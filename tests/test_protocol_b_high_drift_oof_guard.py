"""high_drift_overfit_guard 的统一折外证据判据（Task 8.3 Task 11）。

现有规则把"样本内相对 A 改善过大"当作过拟合证据。AEMO NSW h=1 的证据显示，
该任务六条二模型候选的 `tail/full` 全在 1.090–1.093、都高于 h<=1 的 1.08 门槛，
tail 条件对候选完全不区分，复合判据于是退化成单条件——候选在验证集上越好越确定
被拒绝，唯一幸存的是最差的那条。

修正不动 5% 与 tail 阈值，只补一条判据：在**同一组 blocked-CV 折、同一评价范围**
上分别计算 A、B 的折外 MAE，折外证据支持 B 时不得以"样本内改善过大"为由回退；
折外不支持时维持保守回退。全程不读 test 标签。

该判据只在 ``len(df_val) >= 500`` 时可达，此时折必然存在且非退化，因此拟合异常
一律向上抛，不做不可达分支的兜底。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.kg.protocol_a import kg_combination_pred_only
from src.eval.kg.protocol_b import _unified_oof_mae, kg_combination_with_features
from tests.task83_fixtures import (
    HIGH_DRIFT_MODELS,
    make_high_drift_overfit_frames,
)


def _run(**kwargs):
    df_val, df_test, raw_val, raw_test = make_high_drift_overfit_frames(**kwargs)
    return kg_combination_with_features(
        df_val, df_test, raw_val, raw_test, list(HIGH_DRIFT_MODELS), 1,
        dataset_name="probe", base_model_cols=list(HIGH_DRIFT_MODELS),
    )


def _guard_config(result):
    return (result["val"].get("weight_meta") or {}).get("guard_config") or {}


def test_oof_evidence_supporting_b_blocks_the_in_sample_overfit_fallback():
    """折外证据支持 B 时，不得再以"样本内改善过大"为由回退。"""
    result = _run(tail_infl=1.3)
    guard = _guard_config(result)
    check = guard["high_drift_overfit_oof_check"]

    # 前提：这条判据确实被触发过（样本内改善 >5% 且 tail 条件不通过）
    assert check["evaluated"] is True
    assert check["rel_improve_b_vs_a"] > 0.05
    assert 1.08 < check["tail_full_ratio_b"] <= 1.15

    # 同折同范围：两侧覆盖样本数必须一致，否则数值不可比
    assert check["oof_coverage_a"] == check["oof_coverage_b"]
    assert check["oof_mae_b"] < check["oof_mae_a"]
    assert check["supports_b"] is True

    # 结果：不再因该判据回退
    assert result["protocol"] == "B_pred_features"
    reason = guard.get("final_fallback_reason") or ""
    assert "high_drift_overfit_guard" not in reason


def test_oof_check_is_not_run_when_the_overfit_condition_does_not_hold():
    """tail 条件本就通过时不触发该判据，也不该付出额外 CV 代价。"""
    result = _run(tail_infl=1.0)
    check = _guard_config(result).get("high_drift_overfit_oof_check")
    assert check is None or check["evaluated"] is False


def test_unified_oof_uses_identical_folds_for_both_sides():
    """同一协议下 A/B 两侧的折与评价范围必须一致，且不接触 test 帧。"""
    df_val, _df_test, raw_val, _raw_test = make_high_drift_overfit_frames()
    f = raw_val["wk"].values.astype(float)
    f = f - f.mean()
    inter = np.column_stack([df_val["m1"].values.astype(float) * f])

    mae_a, cov_a = _unified_oof_mae(
        df_val=df_val, models=["m1", "m3"], horizon=1, alpha=1.0, sample_weight=None,
    )
    mae_b, cov_b = _unified_oof_mae(
        df_val=df_val, models=["m1"], horizon=1, alpha=1.0, sample_weight=None,
        interaction_features=inter, interaction_alpha=1.0,
    )
    assert cov_a == cov_b > 0
    assert np.isfinite(mae_a) and np.isfinite(mae_b)
    # 该夹具的交互关系全窗有效，折外应确实更优
    assert mae_b < mae_a


def test_protocol_a_unchanged_on_high_drift_frames():
    """Protocol A 在该夹具上的选择与权重不受本次 guard 改动影响。"""
    df_val, df_test, _raw_val, _raw_test = make_high_drift_overfit_frames()
    result = kg_combination_pred_only(
        df_val, df_test, list(HIGH_DRIFT_MODELS), 1, 0.5, dataset_name="probe",
    )
    assert result["val"]["selected_models"] == ["m1", "m3"]
    assert result["val"]["mae"] == pytest.approx(19.188549481966263, rel=1e-9)
