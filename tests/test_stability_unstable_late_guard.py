"""`unstable_late` 硬删除准确候选的根因回归（合一计划 Task 6B）。

**根因（Task 5 + 6A 已取证）**：`_dedup_and_stability_filter` 在 drift=high 时，
只要 `_collect_stability_reasons` 返回任一原因就**硬删除**该候选。而
`unstable_late`（`late_ratio = 验证窗后段 MAE / 前段 MAE > 1.5`）只刻画窗口内
误差漂移，**不含任何准确度信息**。

实测后果（`result/ab_convergence/same_round_full.json`）：full 生产分割下
xgboost/lgbm/catboost 仅因该条被删，Protocol B test MAE 421.1973；只恢复这类
候选后 guard 自行选中 xgboost，MAE 降到 307.2811（-27.0%）。

本模块把该缺陷压缩成小数据上的确定性用例。修复方向必须是**判据语义**而非放宽
阈值：`unstable_late` 单独出现、且候选在准确度上明显优于 naive 时，不应硬删除。
"""
import numpy as np
import pandas as pd
import pytest

from src.eval.kg.stability import _dedup_and_stability_filter, _estimate_model_stability

PERIOD = 24
N = 240


def _frame(preds: dict) -> pd.DataFrame:
    """构造验证窗。

    关键：y 必须含**逐日水平漂移**。若只有干净日周期，naive(24) 几乎完美
    （y[t]≈y[t-24]），任何模型都打不过它，`rmse_ratio_vs_naive` 就永远 >1，
    无法复现"准确却被删"的条件——首版 fixture 正是栽在这里。
    """
    ts = pd.date_range("2026-01-01", periods=N, freq="h")
    rng = np.random.default_rng(0)
    days = np.arange(N) // PERIOD
    level = np.cumsum(rng.normal(0, 15.0, days.max() + 1))[days]  # 日间随机游走
    y = 100.0 + level + 20.0 * np.sin(np.arange(N) * 2 * np.pi / PERIOD) + rng.normal(0, 0.5, N)
    data = {"timestamp": ts, "y": y}
    for name, fn in preds.items():
        data[name] = fn(y, rng)
    return pd.DataFrame(data)


def _accurate_but_late_drifting(y, rng):
    """准确度远优于 naive，但误差集中在后段 -> late_ratio > 1.5。

    幅度需温和：跳得太狠会连带触发 unstable_cv，就不再是"仅 late 一条原因"。
    """
    err = np.empty(N)
    split = int(N * 0.7)
    err[:split] = rng.normal(0, 0.50, split)
    err[split:] = rng.normal(0, 1.05, N - split)
    return y + err


def _accurate_and_stable(y, rng):
    return y + rng.normal(0, 0.60, N)


def _inaccurate_and_late_drifting(y, rng):
    """既不准确（劣于 naive）又后段漂移 -> 两条原因都命中，必须仍被删除。"""
    err = np.empty(N)
    split = int(N * 0.7)
    err[:split] = rng.normal(0, 28.0, split)
    err[split:] = rng.normal(0, 58.0, N - split)
    return y + err


@pytest.fixture()
def frame():
    return _frame({
        "acc_late": _accurate_but_late_drifting,
        "acc_stable": _accurate_and_stable,
        "bad_late": _inaccurate_and_late_drifting,
    })


def _meta(frame):
    return _estimate_model_stability(frame, ["acc_late", "acc_stable", "bad_late"])


def test_fixture_reproduces_the_exact_input_condition(frame):
    """先证明构造数据确实命中根因条件，否则后面的断言没有意义。"""
    meta = _meta(frame)

    acc = meta["acc_late"]
    assert acc["late_ratio"] > 1.5, "acc_late 必须触发 unstable_late"
    assert acc["rmse_ratio_vs_naive"] < 1.0, "acc_late 必须显著优于 naive"
    assert acc["unstable_cv"] is False, "只保留 unstable_late 这一条原因"

    bad = meta["bad_late"]
    assert bad["late_ratio"] > 1.5
    assert bad["rmse_ratio_vs_naive"] > 1.05, "bad_late 必须同时劣于 naive"


def _run_filter(frame):
    model_cols = ["acc_late", "acc_stable", "bad_late"]
    y = frame["y"].values
    maes = {
        m: float(np.mean(np.abs(np.asarray(frame[m].values, dtype=float) - y)))
        for m in model_cols
    }
    return _dedup_and_stability_filter(
        df_val=frame,
        df_test=frame,
        model_cols=model_cols,
        maes=maes,
        horizon=1,
        drift_level_override="high",
        print_suffix="(test)",
    )


def test_accurate_model_is_not_hard_removed_for_late_ratio_alone(frame):
    """核心断言：只因 late_ratio 超阈、但准确度明显优于 naive 的候选不得被硬删除。"""
    ctx = _run_filter(frame)

    assert "acc_late" not in ctx["stability_removed"], (
        "准确度优于 naive 的候选仅因 late_ratio>1.5 被删除——这正是 "
        "same_round_full.json 中 MAE 421.20 vs 307.28 的根因"
    )
    assert "acc_late" in ctx["model_cols"]


def test_genuinely_bad_late_drifting_model_is_still_removed(frame):
    """修正不得削弱真实的移除：既漂移又劣于 naive 的候选必须仍被删掉。"""
    ctx = _run_filter(frame)

    assert "bad_late" in ctx["stability_removed"]
    assert "worse_than_naive" in ctx["stability_removed"]["bad_late"]


def test_stable_accurate_model_untouched(frame):
    ctx = _run_filter(frame)

    assert "acc_stable" not in ctx["stability_removed"]
    assert "acc_stable" in ctx["model_cols"]


def test_retention_decision_is_auditable(frame):
    """guard 判据变化必须留痕：保留原因与原始判据都要能在 meta 中查到。"""
    ctx = _run_filter(frame)
    meta = ctx["stability_meta"]["acc_late"]

    # 原始判据保持可见（未被抹掉）
    assert meta["unstable_late"] is True
    assert meta["late_ratio"] > 1.5
    # 新增：为什么最终没有删除
    assert meta.get("retained_despite_unstable_late") is True
    assert "rmse_ratio_vs_naive" in str(meta.get("retention_reason", ""))


def test_switch_off_reproduces_pre_fix_behaviour(frame, monkeypatch):
    """关掉开关必须精确复现修正前行为，供 Task 6B Step 3 做 A/B 对照。"""
    import src.eval.kg.stability as st

    monkeypatch.setattr(st, "MODEL_STABILITY_UNSTABLE_LATE_REQUIRES_CORROBORATION", False)
    ctx = _run_filter(frame)

    assert "acc_late" in ctx["stability_removed"]
    assert ctx["stability_removed"]["acc_late"] == "unstable_late"


def test_before_and_after_criteria_both_recorded(frame):
    """修改前判据必须与修改后结果并存，否则 guard 变化不可审计。"""
    ctx = _run_filter(frame)

    acc = ctx["stability_meta"]["acc_late"]
    assert acc["reasons_before_corroboration"] == ["unstable_late"]
    assert acc["retained_despite_unstable_late"] is True

    bad = ctx["stability_meta"]["bad_late"]
    assert "unstable_late" in bad["reasons_before_corroboration"]
    assert bad["retained_despite_unstable_late"] is False
