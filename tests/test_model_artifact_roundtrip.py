"""真实模型与组合预测器的 pickle round-trip（SQLite 模型库 Task 2）。

- 基础模型：用项目真实 registry + configs/pipeline.yaml 参数构造并小数据拟合，
  保存 / 重新加载 / 比较预测。某个模型无法保存或加载时，让本测试失败并点名。
- 组合预测器：直接用 Protocol B 真实执行路径产出的
  ``CombinationPredictor``（含 interaction / post-adjustment 状态），
  重新加载后对同一 val/test 输入比较输出，容差 1e-8。
不使用只实现 ``predict()`` 的测试假类。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.eval.kg.protocol_b as pb
from src.eval.kg.config import RUNTIME_PREDICTIONS_KEY
from src.models.artifacts import load_artifact, save_artifact
from src.models.combination_predictor import COMBINATION_PREDICTOR_KEY, CombinationPredictor
from src.models.registry import model_registry
from src.utils.io import load_yaml

_PIPELINE_MODEL_PARAMS = {
    str(mid): dict(params)
    for mid, params in (load_yaml("configs/pipeline.yaml") or {}).get("models", {}).items()
    if isinstance(params, dict)
}


def _supervised_frame(n=220, seed=7):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="h")
    temp = np.linspace(4, 30, n) + rng.normal(0, 1.0, n)
    hour = ts.hour.to_numpy()
    load = (
        120.0
        + 18.0 * np.sin(np.arange(n) * 2 * np.pi / 24)
        + 1.4 * temp
        + rng.normal(0, 1.5, n)
    )
    X = pd.DataFrame({"hour": hour, "dow": ts.dayofweek.to_numpy(), "temp": temp}, index=ts)
    y = pd.Series(load, index=ts, name="target_h")
    return X, y


@pytest.mark.parametrize("model_id", sorted(_PIPELINE_MODEL_PARAMS))
def test_real_baseline_model_pickle_round_trips(model_id, tmp_path):
    params = dict(_PIPELINE_MODEL_PARAMS[model_id])
    if model_id == "arima":
        params.setdefault("freq", "h")
    model = model_registry.create(model_id, **params)

    X, y = _supervised_frame()
    X_fit, y_fit = X.iloc[:180], y.iloc[:180]
    X_eval = X.iloc[180:]
    model.fit(X_fit, y_fit)
    before = np.asarray(model.predict(X_eval), dtype=float)

    path = save_artifact(model, tmp_path / f"{model_id}.pkl")
    reloaded = load_artifact(path)
    after = np.asarray(reloaded.predict(X_eval), dtype=float)

    assert after.shape == before.shape
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-8)


# --------------------------------------------------------------------------- #
# Protocol B 组合预测器
# --------------------------------------------------------------------------- #


def _protocol_b_frames(n_val=1500, n_test=320, seed_v=1, seed_t=2):
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


def _fixed_run(members, monkeypatch=None, adjust_bonus_scale=None):
    if adjust_bonus_scale is not None:
        monkeypatch.setattr(pb, "PROTOCOL_B_ADJUST_BONUS_SCALE", adjust_bonus_scale)
    df_val, df_test, raw_val, raw_test = _protocol_b_frames()
    result = pb.kg_combination_with_features(
        df_val,
        df_test,
        raw_val,
        raw_test,
        ["m1", "m2", "m3"],
        1,
        dataset_name="task2_combo",
        base_model_cols=["m1", "m2", "m3"],
        return_predictions=True,
        _fixed_selected_models=list(members),
        _skip_final_guard=True,
        _return_combination_predictor=True,
    )
    return result, df_val, df_test, raw_val, raw_test


def _base_pred_map(df, members):
    return {m: df[m].to_numpy(dtype=float) for m in members}


def test_two_model_predictor_matches_engine_and_survives_pickle(tmp_path):
    result, df_val, df_test, raw_val, raw_test = _fixed_run(["m1", "m2"])
    predictor = result[COMBINATION_PREDICTOR_KEY]
    assert isinstance(predictor, CombinationPredictor)

    engine_test = np.asarray(result[RUNTIME_PREDICTIONS_KEY]["test"], dtype=float)
    engine_val = np.asarray(result[RUNTIME_PREDICTIONS_KEY]["val"], dtype=float)

    replay_test = predictor.predict(_base_pred_map(df_test, predictor.member_ids), raw_test)
    replay_val = predictor.predict(_base_pred_map(df_val, predictor.member_ids), raw_val)
    np.testing.assert_allclose(replay_test, engine_test, rtol=0, atol=1e-8)
    np.testing.assert_allclose(replay_val, engine_val, rtol=0, atol=1e-8)

    reloaded = load_artifact(save_artifact(predictor, tmp_path / "combo.pkl"))
    np.testing.assert_allclose(
        reloaded.predict(_base_pred_map(df_test, reloaded.member_ids), raw_test),
        engine_test,
        rtol=0,
        atol=1e-8,
    )


def test_three_model_predictor_matches_engine(tmp_path):
    result, _df_val, df_test, _raw_val, raw_test = _fixed_run(["m1", "m2", "m3"])
    predictor = result[COMBINATION_PREDICTOR_KEY]
    engine_test = np.asarray(result[RUNTIME_PREDICTIONS_KEY]["test"], dtype=float)

    reloaded = load_artifact(save_artifact(predictor, tmp_path / "combo3.pkl"))
    replay = reloaded.predict(_base_pred_map(df_test, reloaded.member_ids), raw_test)
    np.testing.assert_allclose(replay, engine_test, rtol=0, atol=1e-8)


def test_predictor_replays_interaction_branch_without_post_adjustment(tmp_path, monkeypatch):
    result, _df_val, df_test, _raw_val, raw_test = _fixed_run(
        ["m1", "m2"], monkeypatch=monkeypatch, adjust_bonus_scale=50.0
    )
    weight_meta = result["test"]["weight_meta"]
    assert weight_meta["interaction_branch"]["applied"] is True
    assert weight_meta["post_adjustment"]["applied"] is False

    predictor = result[COMBINATION_PREDICTOR_KEY]
    assert predictor.interaction is not None
    assert set(predictor.required_feature_names) == {"temp"}

    engine_test = np.asarray(result[RUNTIME_PREDICTIONS_KEY]["test"], dtype=float)
    reloaded = load_artifact(save_artifact(predictor, tmp_path / "combo_inter.pkl"))
    replay = reloaded.predict(_base_pred_map(df_test, reloaded.member_ids), raw_test)
    np.testing.assert_allclose(replay, engine_test, rtol=0, atol=1e-8)

    # 没有 raw_features 时带 interaction 的预测器必须显式报错，而不是给错结果
    with pytest.raises(ValueError):
        reloaded.predict(_base_pred_map(df_test, reloaded.member_ids), None)
