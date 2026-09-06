"""在线模型库预测入口 `run.py predict`（SQLite 模型库 Task 5）。

- 临时真实 SQLite + 真实序列化模型 / 组合器；
- 两个可比较场景，输入无未来 y；
- 选中预期关系、输出真实 yhat、trace 完整；
- 在线链路不出现候选选择器；
- 缺必要特征 / 无兼容场景时非零退出，且不回退旧引擎。
"""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.pipeline.library_prediction as library_prediction
from src.eval.kg.protocol_b import evaluate_fixed_protocol_b_combination
from src.models.artifacts import load_artifact, save_artifact
from src.models.registry import model_registry
from src.storage.model_store import ModelStore

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fit_lgbm(seed: int, feature_cols):
    rng = np.random.default_rng(seed)
    n = 240
    ts = pd.date_range("2025-01-01", periods=n, freq="h")
    temp = np.linspace(5, 30, n) + rng.normal(0, 1, n)
    X = pd.DataFrame({"hour": ts.hour, "dow": ts.dayofweek, "temp": temp})[feature_cols]
    y = 100 + 15 * np.sin(np.arange(n) * 2 * np.pi / 24) + 1.3 * temp + rng.normal(0, 1.5, n)
    model = model_registry.create("lgbm_reg", n_estimators=15, n_jobs=1, verbose=-1)
    model.fit(X, pd.Series(y))
    return model, X, pd.Series(y)


def _build_library(tmp_path: Path) -> dict:
    db = tmp_path / "lib.sqlite3"
    store = ModelStore(str(db))
    store.create_schema()
    artifacts = tmp_path / "artifacts"
    feature_cols = ["hour", "dow", "temp"]

    model_a, Xa, ya = _fit_lgbm(1, feature_cols)
    model_b, Xb, yb = _fit_lgbm(2, feature_cols)
    path_a = save_artifact(model_a, artifacts / "pjm__h24__mdl_a.pkl")
    path_b = save_artifact(model_b, artifacts / "pjm__h24__mdl_b.pkl")

    for mid, path in (("pjm__h24__mdl_a", path_a), ("pjm__h24__mdl_b", path_b)):
        store.add_model(
            model_id=mid,
            model_type=mid.split("__")[-1],
            task_type="load_forecast",
            artifact_path=str(path),
            required_features=feature_cols,
            model_params={"n_estimators": 15},
            lifecycle_stage="active",
        )

    # 真实组合预测器：由 Protocol B 固定集合入口在两模型预测上产出
    n = 200
    ts = pd.date_range("2025-06-01", periods=n, freq="h")
    rng = np.random.default_rng(9)
    y = 100 + 15 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {"timestamp": ts, "y": y, "mdl_a": y + rng.normal(0, 2, n), "mdl_b": y + rng.normal(0, 2, n)}
    )
    raw = pd.DataFrame({"timestamp": ts, "hour": ts.hour, "dow": ts.dayofweek, "temp": np.linspace(5, 30, n)})
    res = evaluate_fixed_protocol_b_combination(
        frame, frame, raw, raw,
        selected_models=["mdl_a", "mdl_b"], horizon=24,
        dataset_name="pjm", base_model_cols=["mdl_a", "mdl_b"],
        return_combination_predictor=True,
    )
    predictor = res["_combination_predictor"]
    combo_path = save_artifact(predictor, artifacts / "pjm__h24__combo.pkl")

    combo_id = store.add_combination(
        "protocol_b_combination",
        str(combo_path),
        [
            (f"pjm__h24__{predictor.member_ids[i]}", i, float(predictor.linear_weights[i]))
            for i in range(len(predictor.member_ids))
        ],
    )

    def _add_scenario(sid, region, sig, val_mae):
        store.add_scenario(
            scenario_id=sid, task_type="load_forecast", business_domain="power_load",
            region=region, horizon=24, forecast_steps=720, freq="h", signature=sig,
        )
        pid = store.add_data_profile(
            scenario_id=sid, data_ref=f"data/{region}", target_column="y",
            features=["hour", "dow", "temp"], sample_count=200,
            start_at="2025-01-01T00:00:00+00:00", end_at="2025-06-01T00:00:00+00:00",
            signature=sig,
        )
        return store.add_relation(sid, pid, combo_id, validation_mae=val_mae)

    rel_a = _add_scenario(
        "pjm_h24_A", "pjm",
        {"y_mean": 100.0, "y_std": 15.0, "y_cv": 0.15, "mean_load": 100.0, "cv_load": 0.15},
        5.0,
    )
    rel_b = _add_scenario(
        "aemo_vic_h24_B", "aemo_vic",
        {"y_mean": 520.0, "y_std": 260.0, "y_cv": 0.5, "mean_load": 520.0, "cv_load": 0.5},
        4.0,
    )
    store.close()
    return {
        "db": db, "combo_id": combo_id, "rel_a": rel_a, "rel_b": rel_b,
        "model_a": model_a, "model_b": model_b, "predictor": predictor,
        "feature_cols": feature_cols,
    }


def _future_features(tmp_path: Path, name="future.csv", drop=None):
    n = 720
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {"timestamp": ts, "hour": ts.hour, "dow": ts.dayofweek, "temp": np.linspace(0, 28, n)}
    )
    if drop:
        df = df.drop(columns=drop)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def _scenario_json(
    tmp_path: Path,
    *,
    horizon=24,
    region="pjm",
    signature=None,
    include_signature=True,
    name="scenario.json",
    forecast_steps=720,
):
    payload = {
        "task_type": "load_forecast",
        "business_domain": "power_load",
        "region": region,
        "horizon": horizon,
        "forecast_steps": forecast_steps,
        "freq": "h",
    }
    if include_signature:
        payload["signature"] = (
            signature
            if signature is not None
            else {"y_mean": 102.0, "y_std": 14.0, "y_cv": 0.14, "mean_load": 102.0, "cv_load": 0.14}
        )
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _empty_db(tmp_path: Path) -> Path:
    db = tmp_path / "lib.sqlite3"
    store = ModelStore(str(db))
    store.create_schema()
    store.close()
    return db


def _run_predict(tmp_path, db, scenario, features, output):
    return subprocess.run(
        [
            sys.executable, "run.py", "predict",
            "--database", str(db),
            "--scenario", str(scenario),
            "--features", str(features),
            "--output", str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_predict_selects_expected_relation_and_emits_real_yhat(tmp_path):
    lib = _build_library(tmp_path)
    features = _future_features(tmp_path)
    scenario = _scenario_json(tmp_path)
    output = tmp_path / "predictions.csv"

    proc = _run_predict(tmp_path, lib["db"], scenario, features, output)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    preds = pd.read_csv(output)
    assert list(preds.columns) == ["timestamp", "yhat"]
    assert len(preds) == 720
    assert preds["yhat"].notna().all()

    trace = json.loads((tmp_path / "predictions.trace.json").read_text())
    assert trace["scenario_id"] == "pjm_h24_A"
    assert trace["relation_id"] == lib["rel_a"]
    assert trace["combination_id"] == lib["combo_id"]
    assert trace["model_ids"] == ["pjm__h24__mdl_a", "pjm__h24__mdl_b"]
    assert trace["member_weights"] == {
        member_id: pytest.approx(float(weight), abs=1e-12)
        for member_id, weight in zip(
            trace["model_ids"], lib["predictor"].linear_weights
        )
    }
    assert 0.0 <= trace["data_similarity"] <= 1.0
    assert trace["artifact_paths"]
    assert trace["model_selection_source"] == "saved_relation"
    # 命中的是 V1 horizon=24 关系，基础预测器语义就是 24，不是恒为 1
    assert trace["base_horizon"] == 24
    assert trace["forecast_steps"] == 720
    assert trace["selector_invoked"] is False
    assert isinstance(trace["prediction_run_id"], int)

    # 真实 yhat：独立重算基础模型 + 组合器
    feat_df = pd.read_csv(features)
    X = feat_df[lib["feature_cols"]]
    base = {
        "mdl_a": np.asarray(lib["model_a"].predict(X), dtype=float),
        "mdl_b": np.asarray(lib["model_b"].predict(X), dtype=float),
    }
    expected = lib["predictor"].predict(base, feat_df)
    np.testing.assert_allclose(preds["yhat"].to_numpy(dtype=float), expected, rtol=0, atol=1e-8)

    # 使用计数在事务内 +1
    store = ModelStore(str(lib["db"]))
    assert store.get_relation(lib["rel_a"])["use_count"] == 1
    assert store.get_relation(lib["rel_b"])["use_count"] == 0
    store.close()


def test_online_path_never_invokes_candidate_selector():
    source = inspect.getsource(library_prediction)
    assert "select_models_protocol_b" not in source
    assert "kg_combination_with_features" not in source


def test_missing_required_feature_exits_nonzero(tmp_path):
    lib = _build_library(tmp_path)
    features = _future_features(tmp_path, drop=["temp"])
    scenario = _scenario_json(tmp_path)
    output = tmp_path / "predictions.csv"

    proc = _run_predict(tmp_path, lib["db"], scenario, features, output)
    assert proc.returncode != 0
    assert "temp" in (proc.stdout + proc.stderr)
    assert not output.exists()


def test_no_compatible_scenario_does_not_fall_back_to_legacy(tmp_path):
    lib = _build_library(tmp_path)
    features = _future_features(tmp_path)
    scenario = _scenario_json(tmp_path, horizon=999)
    output = tmp_path / "predictions.csv"

    proc = _run_predict(tmp_path, lib["db"], scenario, features, output)
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "no compatible scenario" in combined.lower() or "无兼容场景" in combined
    # 不启动旧流水线
    assert "电力需求智能预测分析系统" not in combined
    assert not output.exists()


@pytest.mark.parametrize(
    "scenario_kwargs",
    [{"include_signature": False}, {"signature": {}}],
    ids=["missing", "empty"],
)
def test_predict_requires_non_empty_signature(tmp_path, scenario_kwargs):
    db = _empty_db(tmp_path)
    features = _future_features(tmp_path)
    scenario = _scenario_json(tmp_path, **scenario_kwargs)
    output = tmp_path / "predictions.csv"

    proc = _run_predict(tmp_path, db, scenario, features, output)
    assert proc.returncode != 0
    assert "signature" in (proc.stdout + proc.stderr)
    assert not output.exists()


def test_features_row_count_must_equal_requested_forecast_steps(tmp_path):
    """§3.1.5：输出行数必须等于用户请求的预测长度，不接受任意长度的特征表。"""
    lib = _build_library(tmp_path)
    features = _future_features(tmp_path)  # 720 行
    scenario = _scenario_json(tmp_path, forecast_steps=168)
    output = tmp_path / "predictions.csv"

    proc = _run_predict(tmp_path, lib["db"], scenario, features, output)

    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "forecast_steps" in combined and "720" in combined
    assert not output.exists()
