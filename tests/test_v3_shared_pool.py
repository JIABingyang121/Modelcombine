"""V3 共享模型池脚本的契约测试。

覆盖：共享长表结构、逐窗口文件与断点续跑、H 滚动 / T 单次训练复用、计划因果性
校验、y 的目标时间轴、轨迹长度与有限性、缺历史时失败。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.v3_shared_pool as pool_module
from scripts.v3_shared_pool import (
    SharedPoolError,
    build_supervised,
    load_copy,
    main,
    validate_plan_causality,
    window_history,
)

DATASET = "aemo_vic"
HORIZONS = [24, 48]
TRAINING_HOURS = 8760


def _synthetic_copy() -> pd.DataFrame:
    ts = pd.date_range("2021-07-01 00:00", "2023-03-31 23:00", freq="h")
    hours = ts.hour.to_numpy(dtype=float)
    index = np.arange(len(ts), dtype=float)
    load = (
        5000.0
        + 1500.0 * np.sin(2 * np.pi * hours / 24.0)
        + 500.0 * np.sin(2 * np.pi * index / 168.0)
        + 0.02 * index
    )
    return pd.DataFrame({"timestamp": ts, "load": load})


def _write_inputs(tmp_path: Path, *, extra_test: bool = False) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "data"
    (data_root / DATASET).mkdir(parents=True)
    _synthetic_copy().to_csv(data_root / DATASET / "load.csv", index=False)

    definition = {
        "experiment": "v3_shared_pool_test",
        "pool": ["seasonal_naive", "random_forest"],
        "horizons": HORIZONS,
    }
    definition_path = tmp_path / "definition.json"
    definition_path.write_text(json.dumps(definition), encoding="utf-8")

    windows = [
        {"label": "H1", "role": "history", "dataset": DATASET,
         "forecast_origin": "2022-07-01 00:00:00"},
        {"label": "T1", "role": "test", "dataset": DATASET,
         "forecast_origin": "2023-01-01 00:00:00"},
    ]
    if extra_test:
        windows.append(
            {"label": "T2", "role": "test", "dataset": DATASET,
             "forecast_origin": "2023-02-01 00:00:00"}
        )
    plan = {
        "experiment": "v3_shared_pool_test",
        "datasets": [{"dataset": DATASET, "windows": windows}],
    }
    plan_path = tmp_path / "window_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return definition_path, plan_path, data_root


def _run(definition: Path, plan: Path, data_root: Path, out_dir: Path) -> int:
    return main([
        "--definition", str(definition),
        "--window-plan", str(plan),
        "--data-root", str(data_root),
        "--out-dir", str(out_dir),
    ])


def test_generates_shared_predictions_for_all_windows(tmp_path):
    definition, plan, data_root = _write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    assert _run(definition, plan, data_root, out_dir) == 0

    merged = pd.read_csv(out_dir / "shared_predictions.csv")
    expected_rows = 2 * 2 * sum(HORIZONS)
    assert len(merged) == expected_rows
    assert list(merged.columns) == [
        "dataset", "window_label", "role", "model", "forecast_steps", "timestamp", "yhat",
    ]
    assert set(merged["model"]) == {"seasonal_naive", "random_forest"}
    assert np.isfinite(merged["yhat"]).all()
    for (_label, _model, steps), n in (
        merged.groupby(["window_label", "model", "forecast_steps"])["timestamp"].nunique().items()
    ):
        assert n == steps

    manifest = json.loads((out_dir / "shared_pool_manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == expected_rows
    assert manifest["pool_seed"] == 42
    assert "rolling_8760h_before_origin" == manifest["training"]["history_windows"]


def test_window_files_allow_resume(tmp_path, capsys):
    definition, plan, data_root = _write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    assert _run(definition, plan, data_root, out_dir) == 0
    before = (out_dir / "shared_predictions.csv").read_bytes()

    assert _run(definition, plan, data_root, out_dir) == 0
    captured = capsys.readouterr()
    assert captured.out.count("跳过已完成") == 4  # 2 窗口 × 2 模型
    assert (out_dir / "shared_predictions.csv").read_bytes() == before


def test_test_windows_train_models_once_and_reuse(tmp_path, monkeypatch):
    definition, plan, data_root = _write_inputs(tmp_path, extra_test=True)
    out_dir = tmp_path / "out"
    calls = []
    original = pool_module.fit_pool_model

    def counting(model_type, x, y, params, seed):
        calls.append(model_type)
        return original(model_type, x, y, params, seed)

    monkeypatch.setattr(pool_module, "fit_pool_model", counting)
    assert _run(definition, plan, data_root, out_dir) == 0
    # H1 各训练一次；两个 T 窗口共享同一批 T 训练 → 每个模型共 2 次（H 1 + T 1）
    assert sorted(calls) == ["random_forest", "random_forest", "seasonal_naive", "seasonal_naive"]


def test_plan_causality_rejects_history_after_test(tmp_path):
    definition, plan, data_root = _write_inputs(tmp_path)
    data = json.loads(plan.read_text(encoding="utf-8"))
    data["datasets"][0]["windows"] = [
        {"label": "T1", "role": "test", "dataset": DATASET,
         "forecast_origin": "2022-09-01 00:00:00"},
        {"label": "H1", "role": "history", "dataset": DATASET,
         "forecast_origin": "2022-10-01 00:00:00"},
    ]
    plan.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SharedPoolError):
        _run(definition, plan, data_root, tmp_path / "out")


def test_expected_task_grid_from_plan(tmp_path):
    _, plan, _ = _write_inputs(tmp_path)
    data = json.loads(plan.read_text(encoding="utf-8"))
    grid = pool_module.expected_task_grid(data, HORIZONS, "test")
    assert grid == [(DATASET, "T1", 24), (DATASET, "T1", 48)]
    validate_plan_causality(data)


def test_history_window_uses_rolling_slice_and_test_uses_t1_slice(tmp_path):
    definition, plan, data_root = _write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    assert _run(definition, plan, data_root, out_dir) == 0
    merged = pd.read_csv(out_dir / "shared_predictions.csv")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    h1_last = merged[merged.window_label == "H1"]["timestamp"].max()
    t1_first = merged[merged.window_label == "T1"]["timestamp"].min()
    assert h1_last == pd.Timestamp("2022-07-01 00:00") + pd.Timedelta(hours=max(HORIZONS))
    assert t1_first == pd.Timestamp("2023-01-01 01:00")
    assert h1_last < t1_first


def test_build_supervised_has_datetime_target_index():
    copy = _synthetic_copy()
    frame = copy[copy.timestamp < pd.Timestamp("2022-07-01 00:00")].reset_index(drop=True)
    x, y = build_supervised(frame)
    assert list(x.columns) == ["hour", "dayofweek", "lag_1", "lag_24", "lag_168", "roll24_mean"]
    assert isinstance(y.index, pd.DatetimeIndex)
    assert y.index[0] == pd.Timestamp("2021-07-08 01:00")


def test_window_history_requires_full_720_hours(tmp_path):
    copy = _synthetic_copy().tail(100).reset_index(drop=True)
    with pytest.raises(SharedPoolError) as excinfo:
        window_history(copy, "2023-01-01 00:00:00")
    assert "720" in str(excinfo.value)


def test_missing_dataset_copy_fails(tmp_path):
    definition, plan, data_root = _write_inputs(tmp_path)
    (data_root / DATASET / "load.csv").unlink()
    with pytest.raises(FileNotFoundError):
        _run(definition, plan, data_root, tmp_path / "out")
