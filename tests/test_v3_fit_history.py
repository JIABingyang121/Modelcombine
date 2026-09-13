"""V3 历史阶段脚本的契约测试。

覆盖：记忆记录数与最佳方案、冠军单模型、Stacking 权重、MoLE 门控产物、
检索自检（T 同季检索 N/N + 写读一致 N/N）、H 任务网格强制、同季缺失时门控失败。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.v3_fit_history import (
    FitHistoryError,
    main,
    rank_memory_records,
    season_key,
)

DATASET = "aemo_vic"
POOL = ["seasonal_naive", "random_forest", "xgboost_reg"]
HORIZONS = [24, 48]


def _copy() -> pd.DataFrame:
    ts = pd.date_range("2021-01-01 00:00", "2023-12-31 23:00", freq="h")
    index = np.arange(len(ts), dtype=float)
    hours = ts.hour.to_numpy(dtype=float)
    doy = ts.dayofyear.to_numpy(dtype=float)
    load = (
        5000.0
        + 1500.0 * np.sin(2 * np.pi * hours / 24.0)
        + 1200.0 * np.cos(2 * np.pi * doy / 365.0)
        + 0.02 * index
    )
    return pd.DataFrame({"timestamp": ts, "load": load})


def _window(label: str, role: str, origin: str) -> dict:
    o = pd.Timestamp(origin)
    return {
        "label": label,
        "role": role,
        "dataset": DATASET,
        "forecast_origin": origin,
        "targets": {
            str(s): {
                "forecast_steps": s,
                "first_target": str(o + pd.Timedelta(hours=1)),
                "last_target": str(o + pd.Timedelta(hours=s)),
            }
            for s in HORIZONS
        },
    }


def _default_windows():
    return [
        _window("H1", "history", "2022-07-01 00:00:00"),
        _window("H2", "history", "2022-01-01 00:00:00"),
        _window("T1", "test", "2023-01-01 00:00:00"),
    ]


def _write_inputs(tmp_path: Path, *, windows=None) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "data"
    (data_root / DATASET).mkdir(parents=True)
    copy = _copy()
    copy.to_csv(data_root / DATASET / "load.csv", index=False)

    definition = {"experiment": "v3_fit_test", "pool": POOL, "horizons": HORIZONS}
    definition_path = tmp_path / "definition.json"
    definition_path.write_text(json.dumps(definition), encoding="utf-8")

    if windows is None:
        windows = _default_windows()
    plan = {"datasets": [{"dataset": DATASET, "windows": windows}]}
    plan_path = tmp_path / "window_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    truth_map = copy.set_index("timestamp")["load"]
    rng = np.random.default_rng(7)
    rows = []
    for window in windows:
        origin = pd.Timestamp(window["forecast_origin"])
        for horizon in HORIZONS:
            targets = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon, freq="h")
            truth = truth_map.reindex(targets).to_numpy(dtype=float)
            for i, model in enumerate(POOL):
                noise = rng.normal(0.0, 50.0 * (i + 1), size=horizon)
                for ts, yhat in zip(targets, truth + noise):
                    rows.append(
                        {
                            "dataset": DATASET,
                            "window_label": window["label"],
                            "role": window["role"],
                            "model": model,
                            "forecast_steps": horizon,
                            "timestamp": ts,
                            "yhat": float(yhat),
                        }
                    )
    shared_path = tmp_path / "shared.csv"
    pd.DataFrame(rows).to_csv(shared_path, index=False)
    return definition_path, plan_path, shared_path


def _run(tmp_path: Path, definition: Path, plan: Path, shared: Path, out: Path) -> int:
    return main([
        "--definition", str(definition),
        "--window-plan", str(plan),
        "--shared", str(shared),
        "--data-root", str(tmp_path / "data"),
        "--out-dir", str(out),
    ])


def test_fit_history_outputs_and_checks(tmp_path):
    definition, plan, shared = _write_inputs(tmp_path)
    out = tmp_path / "out"
    assert _run(tmp_path, definition, plan, shared, out) == 0

    memory = json.loads((out / "memory.json").read_text(encoding="utf-8"))
    assert len(memory["records"]) == 2 * len(HORIZONS)
    for record in memory["records"]:
        assert record["members"]
        assert set(record["members"]).issubset(set(POOL))
        assert len(record["scenario_profile"]) == 120
        assert record["validation_wape"] >= 0

    validation = json.loads((out / "pool_validation.json").read_text(encoding="utf-8"))
    assert validation["effective_pool"] == POOL
    assert validation["removed"] == {}

    champions = json.loads((out / "champions.json").read_text(encoding="utf-8"))
    assert champions[DATASET]["model"] in POOL
    assert champions[DATASET]["h_tasks"] == 2 * len(HORIZONS)

    stacking = json.loads((out / "stacking.json").read_text(encoding="utf-8"))
    assert set(stacking[DATASET]) == {str(h) for h in HORIZONS}
    for entry in stacking[DATASET].values():
        assert set(entry["weights"]) == set(POOL)

    assert (out / "mole_router.pt").is_file()
    meta = json.loads((out / "mole_router_meta.json").read_text(encoding="utf-8"))
    assert meta["datasets"][DATASET]["train_points"] == 2 * sum(HORIZONS)

    check = json.loads((out / "nn_check.json").read_text(encoding="utf-8"))
    assert check["test_retrieval_total"] == check["test_retrieval_passed"] == len(HORIZONS)
    assert check["readback_identity_total"] == check["readback_identity_passed"] == 2 * len(HORIZONS)
    assert check["passed"] is True


def test_same_season_retrieval_gate_fails(tmp_path):
    windows = [
        _window("H1", "history", "2022-07-01 00:00:00"),
        _window("H2", "history", "2022-10-01 00:00:00"),
        _window("T1", "test", "2023-01-01 00:00:00"),
    ]
    definition, plan, shared = _write_inputs(tmp_path, windows=windows)
    out = tmp_path / "out"
    assert _run(tmp_path, definition, plan, shared, out) == 1
    check = json.loads((out / "nn_check.json").read_text(encoding="utf-8"))
    assert check["test_retrieval_passed"] == 0
    assert check["test_retrieval_total"] == len(HORIZONS)
    assert check["passed"] is False


def test_missing_history_task_fails_grid(tmp_path):
    definition, plan, shared = _write_inputs(tmp_path)
    frame = pd.read_csv(shared)
    frame = frame[~((frame["window_label"] == "H2") & (frame["forecast_steps"] == 48))]
    frame.to_csv(shared, index=False)
    with pytest.raises(FitHistoryError) as excinfo:
        _run(tmp_path, definition, plan, shared, tmp_path / "out")
    assert "H 任务网格" in str(excinfo.value)


def testseason_key_uses_meteorological_seasons():
    # 12-2 / 3-5 / 6-8 / 9-11；6 与 8 同季，10 与 12 不同季
    assert season_key(6) == season_key(8)
    assert season_key(12) == season_key(1) == season_key(2)
    assert season_key(10) != season_key(12)
    assert season_key(3) == season_key(5)
    assert season_key(9) == season_key(11)


def test_nsw_like_same_season_retrieval_passes(tmp_path):
    windows = [
        _window("H1", "history", "2022-06-01 00:00:00"),
        _window("H2", "history", "2022-08-01 00:00:00"),
        _window("T1", "test", "2023-08-01 00:00:00"),
    ]
    definition, plan, shared = _write_inputs(tmp_path, windows=windows)
    out = tmp_path / "out"
    assert _run(tmp_path, definition, plan, shared, out) == 0
    check = json.loads((out / "nn_check.json").read_text(encoding="utf-8"))
    assert check["test_retrieval_passed"] == check["test_retrieval_total"] == len(HORIZONS)
    assert check["passed"] is True


def test_pool_validation_failure_stops_without_reduction(tmp_path):
    definition, plan, shared = _write_inputs(tmp_path)
    frame = pd.read_csv(shared)
    mask = (
        (frame["model"] == "seasonal_naive")
        & (frame["window_label"] == "H1")
        & (frame["forecast_steps"] == 24)
    )
    frame.loc[mask, "yhat"] = -1.0
    frame.to_csv(shared, index=False)
    out = tmp_path / "out"
    assert _run(tmp_path, definition, plan, shared, out) == 1
    validation = json.loads((out / "pool_validation.json").read_text(encoding="utf-8"))
    assert "seasonal_naive" in validation["removed"]
    assert not (out / "memory.json").exists()


def test_rank_memory_records_prefers_same_season(tmp_path):
    records = [
        {"dataset": DATASET, "window_label": "H_apr", "horizon": 24,
         "origin": "2024-04-01 00:00:00", "members": ["seasonal_naive"],
         "weights": {"seasonal_naive": 1.0}, "scenario_profile": [1.0, 0.0, 0.0]},
        {"dataset": DATASET, "window_label": "H_oct", "horizon": 24,
         "origin": "2024-10-01 00:00:00", "members": ["seasonal_naive"],
         "weights": {"seasonal_naive": 1.0}, "scenario_profile": [1.0, 0.0, 0.0]},
    ]
    query = [1.0, 0.0, 0.0]
    before = pd.Timestamp("2025-04-01 00:00:00")
    top_all = rank_memory_records(
        records, query, dataset=DATASET, horizon=24, before_origin=before, inclusive=False
    )[0]
    assert top_all["window_label"] == "H_oct"  # 无季节过滤时更近者优先
    top_apr = rank_memory_records(
        records, query, dataset=DATASET, horizon=24, before_origin=before,
        inclusive=False, same_season_as=season_key(4),
    )[0]
    assert top_apr["window_label"] == "H_apr"  # 同季过滤后只能取 4 月记录
