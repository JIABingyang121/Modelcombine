"""Task 8.3 共用的确定性合成 Protocol B 帧构造器。

只被 Task 8.3 的测试 import，不进入生产路径。固定 RNG，保证断言可复现。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODELS = ["m1", "m2", "m3"]


def make_protocol_b_frames(high_corr: bool = False):
    rng = np.random.default_rng(83)
    ts_val = pd.date_range("2026-01-01", periods=600, freq="h")
    ts_test = pd.date_range("2026-02-01", periods=180, freq="h")

    def make(ts):
        y = 100.0 + 10.0 * np.sin(np.arange(len(ts)) * 2 * np.pi / 24)
        shared = rng.normal(0.0, 1.0, len(ts))
        data = {"timestamp": ts, "y": y}
        data["m1"] = y + shared
        # high_corr 让 m2 与 m1 高度相关（corr≈0.995，高于旧硬冲突阈值 0.9、
        # 低于同质去重阈值 0.999），而不是完全相同——完全相同会触发去重移除。
        data["m2"] = y + (
            shared + 0.1 * rng.normal(0.0, 1.0, len(ts))
            if high_corr
            else rng.normal(0.0, 1.01, len(ts))
        )
        data["m3"] = y + rng.normal(0.0, 1.02, len(ts))
        return pd.DataFrame(data)

    df_val, df_test = make(ts_val), make(ts_test)
    raw_val = pd.DataFrame({"timestamp": ts_val, "hour": ts_val.hour})
    raw_test = pd.DataFrame({"timestamp": ts_test, "hour": ts_test.hour})
    return df_val, df_test, raw_val, raw_test


def make_relation_drift_context(
    *,
    n_val: int = 600,
    change_fraction: float = 0.60,
    drift_offset: float = 4.0,
    seed: int = 42,
):
    """构造带确定性漂移候选的 Protocol B 求解上下文（Task 8.3 Task 5）。

    drift 候选前 change_fraction 贴近 y、后 (1-change_fraction) 固定上移 drift_offset；
    stable 候选全窗保持小噪声。返回 (ctx, "drift")，供真实 backend → temporal
    链路证明负事件可达。不 monkeypatch 任何决策/拟合阶段。
    """
    from src.core.solver import build_protocol_b_context

    rng = np.random.default_rng(seed)
    ts_val = pd.date_range("2026-01-01", periods=n_val, freq="h")
    ts_test = pd.date_range("2026-02-01", periods=180, freq="h")

    def build(ts, *, apply_drift):
        t = np.arange(len(ts), dtype=float)
        y = 20.0 + 0.03 * t + 2.0 * np.sin(2.0 * np.pi * t / 24.0)
        shared = rng.normal(0.0, 0.8, len(ts))
        stable = y + shared
        complementary = y - 0.55 * shared + rng.normal(0.0, 0.35, len(ts))
        drift = y + rng.normal(0.0, 0.15, len(ts))
        if apply_drift:
            drift[int(len(ts) * change_fraction):] += drift_offset
        frame = pd.DataFrame({
            "timestamp": ts, "y": y, "stable": stable,
            "complementary": complementary, "drift": drift,
        })
        raw = pd.DataFrame({"timestamp": ts, "hour": ts.hour})
        return frame, raw

    df_val, raw_val = build(ts_val, apply_drift=True)
    df_test, raw_test = build(ts_test, apply_drift=True)
    ctx = build_protocol_b_context(
        dataset="task83_relation_drift", horizon=1,
        df_val=df_val, df_test=df_test,
        df_raw_val=raw_val, df_raw_test=raw_test,
        model_cols=["stable", "complementary", "drift"],
        base_model_cols=["stable", "complementary", "drift"],
        feedback_store=None, return_predictions=True,
    )
    return ctx, "drift"
