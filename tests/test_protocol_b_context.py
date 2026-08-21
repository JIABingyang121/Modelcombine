"""`build_protocol_b_context` 契约测试（System A/B 合一 Task 1）。

该构造器是 System A（demo 入口）与 System B（真实数据实验脚本）未来共用的
唯一 Protocol B 上下文入口，因此契约必须显式而非依赖调用方自觉：

- 候选模型列与 timestamp 必须原样保留在 df_val/df_test 上（Protocol B 内部
  的 conflict.generate_stable_key 等环节依赖 timestamp）；
- available_features 必须汇总 val/test/raw_val/raw_test 四张表，并剔除
  y/timestamp 这两个非特征列；
- 场景签名必须稳定（同输入同 scenario_id，跨进程可复现）；
- 缺 y、缺 timestamp、缺候选列、val/test 为空时必须显式报错，不得静默产出
  退化上下文——旧实现对缺失 y 只是回退成空数组，会把问题推迟到 Protocol B
  内部才炸，难以定位。
"""
import numpy as np
import pandas as pd
import pytest

from src.core.solver.protocol_b_context import build_protocol_b_context


def _frame(periods: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=periods, freq="h"),
            "y": np.linspace(10.0, 10.0 + periods - 1, periods),
            "m1": np.linspace(10.1, 10.1 + periods - 1, periods),
            "m2": np.linspace(9.9, 9.9 + periods - 1, periods),
        }
    )


def _build(**overrides):
    kwargs = dict(
        dataset="pjm",
        horizon=1,
        df_val=_frame(),
        df_test=_frame(),
        df_raw_val=pd.DataFrame({"load": [1.0, 2.0], "temp": [5.0, 6.0]}),
        df_raw_test=pd.DataFrame({"load": [3.0, 4.0], "temp": [7.0, 8.0]}),
        model_cols=["m1", "m2"],
        base_model_cols=["m1"],
        feedback_store=None,
    )
    kwargs.update(overrides)
    return build_protocol_b_context(**kwargs)


def test_context_preserves_timestamp_and_candidate_columns():
    ctx = _build()

    for frame in (ctx.df_val, ctx.df_test):
        assert "timestamp" in frame.columns
        assert "m1" in frame.columns and "m2" in frame.columns

    assert ctx.model_cols == ["m1", "m2"]
    assert ctx.base_model_cols == ["m1"]
    assert ctx.dataset_name == "pjm"
    assert ctx.horizon == 1


def test_available_features_union_four_frames_without_y_and_timestamp():
    ctx = _build()

    assert {"m1", "m2", "load", "temp"} <= ctx.available_features
    assert "y" not in ctx.available_features
    assert "timestamp" not in ctx.available_features


def test_available_features_tolerates_missing_raw_frames():
    ctx = _build(df_raw_val=None, df_raw_test=None)

    assert {"m1", "m2"} <= ctx.available_features
    assert "load" not in ctx.available_features


def test_scenario_signature_is_stable_across_identical_inputs():
    first = _build()
    second = _build()

    assert first.scenario.signature == second.scenario.signature
    assert first.scenario.scenario_id == second.scenario.scenario_id
    assert first.scenario.business_domain == "load_forecast"
    assert first.scenario.primary_metric == "MAE"
    assert first.scenario.region == "pjm"


def test_scenario_signature_changes_when_horizon_changes():
    assert _build(horizon=1).scenario.scenario_id != _build(horizon=24).scenario.scenario_id


def test_missing_y_column_raises():
    bad = _frame().drop(columns=["y"])

    with pytest.raises(ValueError, match="y"):
        _build(df_val=bad)


def test_missing_timestamp_column_raises():
    bad = _frame().drop(columns=["timestamp"])

    with pytest.raises(ValueError, match="timestamp"):
        _build(df_test=bad)


def test_missing_candidate_column_raises():
    with pytest.raises(ValueError, match="m3"):
        _build(model_cols=["m1", "m3"])


def test_empty_model_cols_raises():
    with pytest.raises(ValueError, match="model_cols"):
        _build(model_cols=[])


@pytest.mark.parametrize("empty_side", ["df_val", "df_test"])
def test_empty_val_or_test_frame_raises(empty_side):
    with pytest.raises(ValueError, match="empty"):
        _build(**{empty_side: _frame().iloc[0:0]})
