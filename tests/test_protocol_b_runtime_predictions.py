"""Protocol B 精确预测输出（System A/B 合一 Task 3.1）。

**为什么需要这个能力**：Protocol B 原先只返回指标，调用方只能用
`df_test[selected] @ weights` 重建预测。但该公式覆盖不了所有可达分支——交互
残差分支被接受、而其后的 post_adjustment 未被接受时，引擎最终的 `pred_test`
= 线性组合 + 交互残差，与重建值不同。此时若把重建值当作最终 `yhat` 写进
`predictions.csv`，就会出现"trace 和上报 MAE 都正确，但实际输出的是另一条
预测"的隐蔽错误。

因此引擎需要可选地把真实 `pred_val` / `pred_test` 交出来：
- 默认关闭，实验脚本的 JSON 结构与体积完全不变；
- 开启后每条分支（Protocol A 回退、最佳单模型回退、B_pred_features、交互
  分支）都必须返回与其上报 MAE 精确对应的预测。
"""
import json

import numpy as np
import pandas as pd
import pytest

import src.eval.kg.protocol_b as pb
from src.eval.kg.config import RUNTIME_PREDICTIONS_KEY

MODELS = ["m1", "m2", "m3"]


def _data(n_val=1500, n_test=300, seed_v=1, seed_t=2):
    tsv = pd.date_range("2026-01-01", periods=n_val, freq="h")
    tst = pd.date_range("2026-06-01", periods=n_test, freq="h")

    def mk(ts, n, seed):
        r = np.random.default_rng(seed)
        temp = np.linspace(5, 35, n) + r.normal(0, 1, n)
        y = 100 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24) + 1.5 * temp + r.normal(0, 1, n)
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "y": y,
                "m1": y - 0.9 * temp + r.normal(0, 2, n),
                "m2": y + 0.7 * temp + r.normal(0, 3, n),
                "m3": y + r.normal(0, 6, n),
            }
        )
        raw = pd.DataFrame({"timestamp": ts, "hour": ts.hour, "dow": ts.dayofweek, "temp": temp})
        return df, raw

    df_val, raw_val = mk(tsv, n_val, seed_v)
    df_test, raw_test = mk(tst, n_test, seed_t)
    return df_val, df_test, raw_val, raw_test


def _run(return_predictions=False, **kw):
    df_val, df_test, raw_val, raw_test = _data(**kw.pop("data_kw", {}))
    result = pb.kg_combination_with_features(
        df_val, df_test, raw_val, raw_test, MODELS, 1,
        dataset_name="task31", return_predictions=return_predictions, **kw
    )
    return result, df_val, df_test


def _mae(pred, y):
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(y)
    return float(np.mean(np.abs(pred[mask] - y[mask])))


def _linear_mae(result, df_test, split="test"):
    sel = result[split]["selected_models"]
    w = result[split]["weights"]
    pred = df_test[sel].values @ np.array([w[m] for m in sel], dtype=float)
    return _mae(pred, df_test["y"].values)


# --- 默认关闭：结构与体积不变（黄金测试） -----------------------------------


def test_default_off_returns_no_runtime_predictions_and_is_json_serializable():
    result, _, _ = _run()

    assert RUNTIME_PREDICTIONS_KEY not in result
    # B_pred_features 分支带 feedback_apply_meta 与 relation_feedback（Task 8.3 Task 5）；
    # 这里锁定"不因本改动多出键"（runtime predictions 不进结果）。
    assert set(result.keys()) == {"val", "test", "protocol", "feedback_apply_meta", "relation_feedback"}
    for split in ("val", "test"):
        assert RUNTIME_PREDICTIONS_KEY not in result[split]
    # 结构未被污染：默认路径仍可直接 JSON 序列化，体积不含预测数组
    json.dumps(result, default=str)


def test_default_off_matches_explicit_false():
    a, _, _ = _run(return_predictions=False)
    b, _, _ = _run()

    assert a["protocol"] == b["protocol"]
    assert a["test"]["selected_models"] == b["test"]["selected_models"]
    assert a["test"]["mae"] == pytest.approx(b["test"]["mae"], abs=1e-12)
    assert set(a["test"].keys()) == set(b["test"].keys())


# --- 开启后：每条分支的预测都必须与其上报 MAE 精确对应 -----------------------


def test_b_pred_features_predictions_match_reported_mae():
    result, df_val, df_test = _run(return_predictions=True)

    assert result["protocol"] == "B_pred_features"
    preds = result[RUNTIME_PREDICTIONS_KEY]
    assert len(preds["test"]) == len(df_test)
    assert len(preds["val"]) == len(df_val)
    assert _mae(preds["test"], df_test["y"].values) == pytest.approx(
        result["test"]["mae"], abs=1e-9
    )
    assert _mae(preds["val"], df_val["y"].values) == pytest.approx(result["val"]["mae"], abs=1e-9)


def test_interaction_applied_without_post_adjustment_predictions_still_exact(monkeypatch):
    """核心分支：交互已应用、post_adjustment 未应用 -> 线性重算不一致，但预测必须精确。

    通过放大 PROTOCOL_B_ADJUST_BONUS_SCALE 使 w_adj 退化，从而让 post_adjustment
    的 sanity check 拒绝（该常量在交互判定之后才被读取，故不影响交互是否被接受）。
    """
    monkeypatch.setattr(pb, "PROTOCOL_B_ADJUST_BONUS_SCALE", 50.0)

    result, _, df_test = _run(return_predictions=True)

    weight_meta = result["test"]["weight_meta"]
    assert weight_meta["interaction_branch"]["applied"] is True
    assert weight_meta["post_adjustment"]["applied"] is False
    assert result["protocol"] == "B_pred_features"

    reported = result["test"]["mae"]
    # 线性重算在该分支确实对不上——这正是本任务要解决的问题
    assert abs(_linear_mae(result, df_test) - reported) > 1e-8
    # 而引擎交出的真实预测必须精确对应上报 MAE
    preds = result[RUNTIME_PREDICTIONS_KEY]
    assert _mae(preds["test"], df_test["y"].values) == pytest.approx(reported, abs=1e-9)


def test_protocol_a_fallback_no_raw_returns_predictions():
    df_val, df_test, _, _ = _data()
    result = pb.kg_combination_with_features(
        df_val, df_test, None, None, MODELS, 1,
        dataset_name="task31", return_predictions=True,
    )

    assert result["protocol"] == "B_fallback_to_A_no_raw"
    preds = result[RUNTIME_PREDICTIONS_KEY]
    assert _mae(preds["test"], df_test["y"].values) == pytest.approx(
        result["test"]["mae"], abs=1e-9
    )


def test_protocol_a_fallback_no_features_returns_predictions():
    df_val, df_test, _, _ = _data()
    # raw 表只有被排除的列 -> 无可用特征 -> 回退 A
    raw_val = pd.DataFrame({"timestamp": df_val["timestamp"]})
    raw_test = pd.DataFrame({"timestamp": df_test["timestamp"]})
    result = pb.kg_combination_with_features(
        df_val, df_test, raw_val, raw_test, MODELS, 1,
        dataset_name="task31", return_predictions=True,
    )

    assert result["protocol"] == "B_fallback_to_A_no_features"
    preds = result[RUNTIME_PREDICTIONS_KEY]
    assert _mae(preds["test"], df_test["y"].values) == pytest.approx(
        result["test"]["mae"], abs=1e-9
    )


def _weak_data(n_val=400, n_test=120):
    """特征对残差无解释力 -> B 无法显著胜过 A -> 触发 guard 回退。"""
    tsv = pd.date_range("2026-01-01", periods=n_val, freq="h")
    tst = pd.date_range("2026-03-01", periods=n_test, freq="h")

    def mk(ts, n, seed):
        r = np.random.default_rng(seed)
        y = 100 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24) + r.normal(0, 2, n)
        df = pd.DataFrame({
            "timestamp": ts, "y": y,
            "m1": y + r.normal(0, 3, n),
            "m2": y + r.normal(0, 5, n),
            "m3": y + r.normal(0, 4, n),
        })
        raw = pd.DataFrame({"timestamp": ts, "hour": ts.hour, "dow": ts.dayofweek,
                            "temp": np.linspace(10, 20, n)})
        return df, raw

    dv, rv = mk(tsv, n_val, 1)
    dt, rt = mk(tst, n_test, 2)
    return dv, dt, rv, rt


def test_guard_fallback_returns_predictions_matching_reported_mae():
    """弱信号 -> 触发 guard 回退；无论回到 A 还是最优单模型都要精确。"""
    df_val, df_test, raw_val, raw_test = _weak_data()
    result = pb.kg_combination_with_features(
        df_val, df_test, raw_val, raw_test, MODELS, 1,
        dataset_name="task31_guard", return_predictions=True,
    )

    assert result["protocol"].startswith("B_fallback_to_")
    preds = result[RUNTIME_PREDICTIONS_KEY]
    assert _mae(preds["test"], df_test["y"].values) == pytest.approx(
        result["test"]["mae"], abs=1e-9
    )


def test_guard_fallback_preserves_interaction_candidate_audit_metadata():
    """外层 guard 回退后仍须解释候选 interaction 是否被评估、采用。"""
    df_val, df_test, raw_val, raw_test = _weak_data()
    result = pb.kg_combination_with_features(
        df_val, df_test, raw_val, raw_test, MODELS, 1,
        dataset_name="task31_guard_audit", return_predictions=True,
    )

    assert result["protocol"].startswith("B_fallback_to_")
    candidate = result["test"]["weight_meta"]["interaction_branch_candidate"]
    assert isinstance(candidate, dict)
    assert "applied" in candidate
    assert "enabled" in candidate


def test_best_single_fallback_predictions_equal_that_model_column(monkeypatch):
    """最佳单模型回退：guarded 是新构造的 dict，预测应逐点等于该模型列。

    用 complexity guard 强制回退到 best_single：把"相对最优单模型的最小相对提升"
    抬到 0.99，B 不可能达标，于是按配置默认目标 best_single 回退。
    """
    monkeypatch.setattr(pb, "PROTOCOL_B_COMPLEXITY_PENALTY_ENABLED", True)
    monkeypatch.setattr(pb, "PROTOCOL_B_COMPLEXITY_PENALTY_DATASETS", set())
    monkeypatch.setattr(pb, "PROTOCOL_B_COMPLEXITY_PENALTY_MIN_REL_IMPROVE", 0.99)
    monkeypatch.setattr(pb, "PROTOCOL_B_COMPLEXITY_PENALTY_MIN_REL_IMPROVE_FEW_MODELS", 0.99)
    monkeypatch.setattr(pb, "PROTOCOL_B_COMPLEXITY_PENALTY_FALLBACK", "best_single")

    df_val, df_test, raw_val, raw_test = _weak_data()
    result = pb.kg_combination_with_features(
        df_val, df_test, raw_val, raw_test, MODELS, 1,
        dataset_name="task31_bs", return_predictions=True,
    )

    assert result["protocol"] == "B_fallback_to_best_single_guard"
    preds = result[RUNTIME_PREDICTIONS_KEY]
    model = result["test"]["selected_models"][0]
    np.testing.assert_allclose(preds["test"], df_test[model].values, atol=1e-12)
    assert _mae(preds["test"], df_test["y"].values) == pytest.approx(
        result["test"]["mae"], abs=1e-9
    )
