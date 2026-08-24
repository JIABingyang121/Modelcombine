"""诊断专用固定二模型求值入口（Task 8.3 Task 4）。

固定二模型诊断必须真正绕过 selector 与最终 guard，不得通过调高候选分数诱导选择；
拟合、interaction 与 post-adjustment 复用生产实现。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.kg.protocol_b import evaluate_fixed_protocol_b_candidate
from tests.task83_fixtures import make_protocol_b_frames


def test_fixed_candidate_bypasses_selector_and_guard():
    df_val, df_test, raw_val, raw_test = make_protocol_b_frames()
    result = evaluate_fixed_protocol_b_candidate(
        df_val, df_test, raw_val, raw_test,
        selected_models=["m1", "m2"], horizon=6, dataset_name="pjm",
    )
    assert result["diagnostic_mode"] == "fixed_pair"
    assert result["requested_models"] == ["m1", "m2"]
    assert result["fallback_target"] is None
    assert "guard_would_fallback_to" in result
    # 真正绕过 selector：拟合的是固定 pair，而不是 selector 的自然选择
    assert set(result["val"]["selected_models"]) == {"m1", "m2"}
    assert result["eligible_pair"] is True
    assert result["degenerate_reason"] is None


def test_fixed_candidate_rejects_non_pair_input():
    df_val, df_test, raw_val, raw_test = make_protocol_b_frames()
    with pytest.raises(ValueError, match="exactly two"):
        evaluate_fixed_protocol_b_candidate(
            df_val, df_test, raw_val, raw_test,
            selected_models=["m1"], horizon=1, dataset_name="pjm",
        )


def test_degenerate_pair_is_marked_not_eligible():
    """一个模型近零权重时，effective_models 必须反映清理结果，eligible_pair=False。

    m_good 几乎等于 y，m_bad 是纯噪声，Ridge 会给 m_bad 近零权重。虽然生产清理
    在二模型下不触发（1 < len(nonzero) < 2 恒假），但诊断资格必须按清理后的
    zero_weight_cleanup["after"] 判定，不能把退化为单模型的候选当成有效二模型。
    """
    rng = np.random.default_rng(11)
    ts_val = pd.date_range("2026-01-01", periods=600, freq="h")
    ts_test = pd.date_range("2026-02-01", periods=180, freq="h")

    def make(ts):
        m = len(ts)
        y = 100.0 + 10.0 * np.sin(np.arange(m) * 2 * np.pi / 24)
        data = {"timestamp": ts, "y": y}
        data["m_good"] = y + rng.normal(0.0, 0.01, m)
        data["m_bad"] = rng.normal(0.0, 500.0, m)
        return pd.DataFrame(data)

    df_val, df_test = make(ts_val), make(ts_test)
    raw_val = pd.DataFrame({"timestamp": ts_val, "hour": ts_val.hour})
    raw_test = pd.DataFrame({"timestamp": ts_test, "hour": ts_test.hour})
    result = evaluate_fixed_protocol_b_candidate(
        df_val, df_test, raw_val, raw_test,
        selected_models=["m_good", "m_bad"], horizon=1, dataset_name="pjm",
    )
    assert result["diagnostic_mode"] == "fixed_pair"
    assert len(result["effective_models"]) != 2
    assert result["eligible_pair"] is False
    assert result["degenerate_reason"] == "zero_weight_cleanup_reduced_pair"
