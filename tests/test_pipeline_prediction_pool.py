"""无泄漏候选预测矩阵测试（System A/B 合一 Task 2）。

System A 现在是"先选模型再预测"，而 Protocol B 要求"先有候选预测矩阵再组合"。
本模块建立那个矩阵，核心风险是数据泄漏——validation 必须取自训练集时间尾部，
test 标签在任何一轮训练里都不可见。因此这里的断言不只看形状，还用 fake model
记录每次 fit 实际收到的标签，从调用记录上证明泄漏没有发生。
"""
import numpy as np
import pandas as pd
import pytest

from src.pipeline.prediction_pool import (
    RegionPredictionBundle,
    build_region_prediction_bundle,
    split_fit_validation,
)

REGION = "R1"


def _region_frame(periods: int = 240, start: str = "2026-01-01") -> pd.DataFrame:
    """构造带时间派生特征的单区域数据，列结构对齐真实 pipeline 产物。"""
    ts = pd.date_range(start, periods=periods, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "region": REGION,
            "region_type": "residential",
            "load": np.linspace(100.0, 200.0, periods) + rng.normal(0, 1.0, periods),
            "hour": ts.hour,
            "dow": ts.dayofweek,
            "is_weekend": ts.dayofweek.isin([5, 6]).astype(int),
            "day": ts.day,
            "month": ts.month,
            "is_holiday": np.zeros(periods, dtype=int),
            "lag_1": np.linspace(99.0, 199.0, periods),
        }
    )


# --- Step 1: 时间切分 ---------------------------------------------------------


def test_split_fit_validation_is_chronological_and_disjoint():
    train = _region_frame(periods=240)  # 10 天

    fit_df, val_df = split_fit_validation(train, validation_days=2)

    assert len(fit_df) > 0 and len(val_df) > 0
    assert fit_df["timestamp"].max() < val_df["timestamp"].min()
    assert len(fit_df) + len(val_df) == len(train)
    # 无重复行：两侧时间戳集合不相交
    assert set(fit_df["timestamp"]) & set(val_df["timestamp"]) == set()


def test_split_fit_validation_rejects_insufficient_history():
    train = _region_frame(periods=24)  # 只有 1 天

    with pytest.raises(ValueError, match="validation"):
        split_fit_validation(train, validation_days=5)


def test_split_fit_validation_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        split_fit_validation(_region_frame(periods=10).iloc[0:0], validation_days=1)


def test_split_fit_validation_rejects_non_positive_days():
    with pytest.raises(ValueError, match="validation_days"):
        split_fit_validation(_region_frame(), validation_days=0)


# --- Step 2: 预测矩阵 ---------------------------------------------------------


class _RecordingModel:
    """记录每次 fit 收到的标签，用于从调用记录上验证无泄漏。"""

    def __init__(self, calls, name, offset=0.0, fail_on_fit=False):
        self._calls = calls
        self._name = name
        self._offset = offset
        self._fail_on_fit = fail_on_fit

    def fit(self, X, y):
        if self._fail_on_fit:
            raise RuntimeError(f"{self._name} boom")
        self._calls.append(
            {"model": self._name, "n": len(y), "y": np.asarray(y, dtype=float).copy()}
        )
        return self

    def predict(self, X):
        return np.full(len(X), self._offset, dtype=float)


class _FakeRegistry:
    def __init__(self, failing=()):
        self.calls = []
        self._failing = set(failing)

    def create(self, key, **params):
        offsets = {"m1": 1.0, "m2": 2.0, "m3": 3.0}
        return _RecordingModel(
            self.calls,
            key,
            offset=offsets.get(key, 0.0),
            fail_on_fit=key in self._failing,
        )


def _build(**overrides):
    full = _region_frame(periods=240)
    train, test = full.iloc[:192], full.iloc[192:]
    kwargs = dict(
        region=REGION,
        train=train,
        test=test,
        candidate_models=["m1", "m2"],
        validation_days=2,
        registry=_FakeRegistry(),
    )
    kwargs.update(overrides)
    return build_region_prediction_bundle(**kwargs)


def test_bundle_frames_share_identical_candidate_columns():
    bundle = _build()

    assert isinstance(bundle, RegionPredictionBundle)
    assert bundle.model_cols == ["m1", "m2"]
    for frame in (bundle.df_val, bundle.df_test):
        assert "timestamp" in frame.columns
        assert "y" in frame.columns
        assert [c for c in frame.columns if c in bundle.model_cols] == bundle.model_cols
    assert list(bundle.df_val.columns) == list(bundle.df_test.columns)


def test_validation_predictions_come_from_fit_only_model():
    """第一轮只能用 fit 段训练，绝不能看到 val 或 test 标签。"""
    registry = _FakeRegistry()
    bundle = _build(registry=registry)

    val_labels = set(np.round(bundle.df_val["y"].values, 6))
    test_labels = set(np.round(bundle.df_test["y"].values, 6))

    first_round = [c for c in registry.calls if c["model"] == "m1"][0]
    seen = set(np.round(first_round["y"], 6))

    assert seen & val_labels == set(), "第一轮训练看到了 validation 标签"
    assert seen & test_labels == set(), "第一轮训练看到了 test 标签"


def test_test_predictions_use_full_train_but_never_test_labels():
    """第二轮在完整 train 上重训，样本数必须大于第一轮，且仍不含 test 标签。"""
    registry = _FakeRegistry()
    bundle = _build(registry=registry)

    m1_calls = [c for c in registry.calls if c["model"] == "m1"]
    assert len(m1_calls) == 2, "每个候选应训练两轮：fit->val 与 full train->test"

    first_round, second_round = m1_calls
    assert second_round["n"] > first_round["n"]

    test_labels = set(np.round(bundle.df_test["y"].values, 6))
    assert set(np.round(second_round["y"], 6)) & test_labels == set()


def test_raw_frames_keep_timestamp_and_time_derived_features():
    bundle = _build()

    for raw in (bundle.df_raw_val, bundle.df_raw_test):
        assert "timestamp" in raw.columns
        for feature in ("hour", "dow", "is_weekend", "day", "month", "is_holiday"):
            assert feature in raw.columns, f"时间派生特征 {feature} 在 raw frame 中丢失"
        # 不引入含糊的临时字段名
        assert "time" not in raw.columns


def test_failed_model_is_dropped_from_both_sides_with_reason():
    registry = _FakeRegistry(failing=["m2"])

    bundle = _build(candidate_models=["m1", "m2"], registry=registry)

    assert bundle.model_cols == ["m1"]
    assert "m2" not in bundle.df_val.columns
    assert "m2" not in bundle.df_test.columns
    assert "m2" in bundle.metadata["failed_models"]
    assert "boom" in bundle.metadata["failed_models"]["m2"]


def test_all_candidates_failing_raises():
    registry = _FakeRegistry(failing=["m1", "m2"])

    with pytest.raises(ValueError, match="no candidate"):
        _build(candidate_models=["m1", "m2"], registry=registry)


def test_base_model_cols_defaults_to_surviving_candidates():
    bundle = _build()

    assert bundle.base_model_cols == bundle.model_cols
    assert set(bundle.fitted_test_models) == set(bundle.model_cols)


def test_bundle_is_accepted_by_protocol_b_context_builder():
    """与 Task 1 的统一上下文构造器对接，确认契约校验直接通过。"""
    from src.core.solver import build_protocol_b_context

    bundle = _build()

    ctx = build_protocol_b_context(
        dataset=REGION,
        horizon=1,
        df_val=bundle.df_val,
        df_test=bundle.df_test,
        df_raw_val=bundle.df_raw_val,
        df_raw_test=bundle.df_raw_test,
        model_cols=bundle.model_cols,
        base_model_cols=bundle.base_model_cols,
        feedback_store=None,
    )

    assert ctx.model_cols == bundle.model_cols
    assert {"hour", "dow"} <= ctx.available_features
