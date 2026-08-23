"""前向逐步选择不得让"低于求解器自身精度的差"决定选谁。

VIC h=24 在相同预测文件哈希、相同候选得分与相同候选排序下复现出两种结果
（`[catboost, lgbm]` MAE 347.376 与 `[catboost, arima]` MAE 518.062，三次重复
一次前者两次后者）。分歧发生在 `forward_stepwise_select` 内部。

放大机制在这里：候选评比用的是 `trial_mae < best_candidate_mae` 这种零容差
严格比较，而 `trial_mae` 来自 `fit_ridge_robust` 的 lbfgs 迭代解（`tol=1e-4`），
本身只保证有限精度。于是两个实质并列的候选之间 1e-9 量级的数值抖动，就能
决定最终进入组合的是哪个模型，并造成 49% 的 MAE 差异。

修复口径：**差异小于相对容差时视为并列，并列一律按调用方给定的候选顺序决定**。
这样噪声翻不动结果，而"顺序有意义"这个特性（Protocol B 的 base_score 排序）
仍然保留。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.eval.kg.model_selection as ms


@pytest.fixture
def df_val():
    n = 200
    rng = np.random.default_rng(0)
    y = 100 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 1.0, n)
    return pd.DataFrame({"y": y, "a": y + 1.0, "b": y + 2.0, "c": y + 3.0})


def _run(monkeypatch, df_val, table, order):
    """用写死的 MAE 表跑一次 stepwise，返回选中的模型。"""
    calls = {}

    def fake_fit(X_val, y_val, alpha, horizon, sample_weight=None):
        # 用列数 + 数值签名反解 trial：直接按传入的列内容匹配
        cols = tuple(
            col
            for col in ("a", "b", "c")
            if any(np.allclose(X_val[:, j], df_val[col].values) for j in range(X_val.shape[1]))
        )
        calls[cols] = calls.get(cols, 0) + 1
        return table[frozenset(cols)]

    monkeypatch.setattr(ms, "_fit_ridge_and_mae", fake_fit)
    selected, meta = ms.forward_stepwise_select(
        df_val=df_val,
        candidate_models=list(order),
        base_maes={"a": 1.0, "b": 2.0, "c": 3.0},
        max_models=3,
        horizon=1,
        min_improve_ratio=0.005,
        respect_candidate_order=True,
    )
    return selected, meta


BASE = {frozenset(("a",)): 100.0}


def test_subtolerance_noise_must_not_change_the_selection(monkeypatch, df_val):
    """b 与 c 实质并列（差 1e-9）时，噪声的符号不得改变最终选择。"""
    eps = 1e-9
    tbl_b_better = {**BASE, **{
        frozenset(("a", "b")): 90.0 - eps,
        frozenset(("a", "c")): 90.0 + eps,
        frozenset(("a", "b", "c")): 89.9,
    }}
    tbl_c_better = {**BASE, **{
        frozenset(("a", "b")): 90.0 + eps,
        frozenset(("a", "c")): 90.0 - eps,
        frozenset(("a", "b", "c")): 89.9,
    }}

    sel1, _ = _run(monkeypatch, df_val, tbl_b_better, ["a", "b", "c"])
    sel2, _ = _run(monkeypatch, df_val, tbl_c_better, ["a", "b", "c"])

    assert sel1 == sel2, (
        f"低于求解器精度的抖动改变了选择：{sel1} vs {sel2}——"
        "候选评比必须对小于容差的差异视为并列"
    )
    assert sel1[1] == "b", "并列时应按调用方给定的候选顺序取先者"


def test_tie_is_resolved_by_caller_order_in_both_directions(monkeypatch, df_val):
    """并列时顺序说了算：换个顺序，选择应随之改变（顺序仍然有意义）。"""
    tbl = {**BASE, **{
        frozenset(("a", "b")): 90.0,
        frozenset(("a", "c")): 90.0,
        frozenset(("a", "b", "c")): 89.9,
    }}

    sel_bc, _ = _run(monkeypatch, df_val, tbl, ["a", "b", "c"])
    sel_cb, _ = _run(monkeypatch, df_val, tbl, ["a", "c", "b"])

    assert sel_bc[1] == "b"
    assert sel_cb[1] == "c"


def test_real_improvement_above_tolerance_is_still_respected(monkeypatch, df_val):
    """容差不能宽到吃掉真实差异：c 明显更好时必须选 c，哪怕 b 排在前面。"""
    tbl = {**BASE, **{
        frozenset(("a", "b")): 90.0,
        frozenset(("a", "c")): 85.0,      # 明显更优（5.6%）
        frozenset(("a", "b", "c")): 84.9,
    }}

    sel, _ = _run(monkeypatch, df_val, tbl, ["a", "b", "c"])

    assert sel[1] == "c", "真实且显著的改善被容差吃掉了"


def test_candidate_evaluation_failures_are_recorded_not_silently_dropped(monkeypatch, df_val):
    """候选评估异常必须留痕：静默 continue 会让间歇性失败无法定位。"""

    def failing_fit(X_val, y_val, alpha, horizon, sample_weight=None):
        if X_val.shape[1] == 1:
            return 100.0
        raise RuntimeError("solver blew up")

    monkeypatch.setattr(ms, "_fit_ridge_and_mae", failing_fit)
    selected, meta = ms.forward_stepwise_select(
        df_val=df_val,
        candidate_models=["a", "b", "c"],
        base_maes={"a": 1.0, "b": 2.0, "c": 3.0},
        max_models=3,
        horizon=1,
        respect_candidate_order=True,
    )

    assert selected == ["a"]
    failures = meta.get("candidate_failures")
    assert failures, "候选评估失败没有任何留痕"
    assert any("solver blew up" in str(v) for v in failures.values())


def test_stepwise_records_all_trial_maes_and_tie_decisions(monkeypatch, df_val):
    """trace 必须说明每个候选为什么胜出、并列或落败。

    仅记录最终选中者无法判断 VIC h=24 的两次分叉是否落在并列容差内：
    两个运行可能各自只留下不同的 winner，而缺少另一个候选的 trial MAE。
    """
    eps = 1e-9
    table = {**BASE, **{
        frozenset(("a", "b")): 90.0,
        frozenset(("a", "c")): 90.0 + eps,
        frozenset(("a", "b", "c")): 89.9,
    }}

    selected, meta = _run(monkeypatch, df_val, table, ["a", "b", "c"])

    assert selected[:2] == ["a", "b"]
    first_step = meta["candidate_evaluations"][0]
    assert first_step["incumbent_models"] == ["a"]
    evaluations = {item["model"]: item for item in first_step["evaluations"]}
    assert evaluations["b"]["trial_mae"] == pytest.approx(90.0)
    assert evaluations["b"]["decision"] == "new_best"
    assert evaluations["c"]["trial_mae"] == pytest.approx(90.0 + eps)
    assert evaluations["c"]["decision"] == "tie_kept"
    assert evaluations["c"]["comparison_mae"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# 可诊断性：stepwise 轨迹必须能在 trace 里看到
# ---------------------------------------------------------------------------

def test_protocol_b_trace_exposes_stepwise_trajectory():
    """没有 stepwise 轨迹，再次分叉时无法定位到是哪一步选错。

    VIC h=24 这次分叉之所以只能靠推断，正是因为 Protocol B 的 stepwise 轨迹
    （每步选了谁、CV MAE 多少、相对改善多少）从未进入 trace——报告里只有
    Protocol A 的。修复后必须能直接对比两次运行的逐步轨迹。
    """
    import numpy as np
    import pandas as pd

    from src.core.solver import build_protocol_b_context
    from src.core.solver.backends import ProtocolBBackend
    from src.core.trace import SelectionTrace

    models = ["cand_a", "cand_b", "cand_c"]
    n = 600
    ts_val = pd.date_range("2026-01-01", periods=n, freq="h")
    ts_test = pd.date_range("2026-04-01", periods=n // 3, freq="h")
    rng = np.random.default_rng(3)

    def mk(ts):
        m = len(ts)
        y = 100 + 20 * np.sin(np.arange(m) * 2 * np.pi / 24) + rng.normal(0, 1.0, m)
        d = {"timestamp": ts, "y": y}
        for i, name in enumerate(models):
            d[name] = y + rng.normal(0, 1.0 + 0.02 * i, m)
        return pd.DataFrame(d)

    df_val, df_test = mk(ts_val), mk(ts_test)
    ctx = build_protocol_b_context(
        dataset="pjm",
        horizon=1,
        df_val=df_val,
        df_test=df_test,
        df_raw_val=pd.DataFrame({"timestamp": ts_val, "hour": ts_val.hour}),
        df_raw_test=pd.DataFrame({"timestamp": ts_test, "hour": ts_test.hour}),
        model_cols=list(models),
        base_model_cols=list(models),
        feedback_store=None,
    )
    trace = SelectionTrace(scenario_id=ctx.scenario.scenario_id)
    ProtocolBBackend().combine(ctx, trace)
    outputs = next(s for s in trace.stages if s["stage"] == "ProtocolBBackend")["outputs"]

    stepwise = outputs.get("stepwise")
    assert stepwise, "trace 未记录 Protocol B 的 stepwise 轨迹"
    assert stepwise.get("trace"), "stepwise 轨迹为空"
    assert "tie_rel_tol" in stepwise, "未记录并列容差，无法判断是否发生并列"
    assert "candidate_failures" in stepwise, "未记录候选评估失败"
