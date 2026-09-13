"""V3 历史阶段脚本的契约测试。

覆盖：记忆记录数与最佳方案、冠军单模型、Stacking 权重、MoLE 门控产物、
记忆读回自检（读回 N/N + 盲测覆盖 N/N）、无更早同季记录时覆盖检查失败。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v3_fit_history import main

DATASET = "aemo_vic"
POOL = ["seasonal_naive", "random_forest", "xgboost_reg"]
HORIZONS = [24, 48]


def _copy() -> pd.DataFrame:
    ts = pd.date_range("2021-07-01 00:00", "2023-01-31 23:00", freq="h")
    index = np.arange(len(ts), dtype=float)
    hours = ts.hour.to_numpy(dtype=float)
    load = 5000 + 1500 * np.sin(2 * np.pi * hours / 24.0) + 0.02 * index
    return pd.DataFrame({"timestamp": ts, "load": load})


def _write_inputs(tmp_path: Path, *, windows=None) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "data"
    (data_root / DATASET).mkdir(parents=True)
    copy = _copy()
    copy.to_csv(data_root / DATASET / "load.csv", index=False)

    definition = {"experiment": "v3_fit_test", "pool": POOL, "horizons": HORIZONS}
    definition_path = tmp_path / "definition.json"
    definition_path.write_text(json.dumps(definition), encoding="utf-8")

    if windows is None:
        windows = [
            {"label": "H1", "role": "history", "dataset": DATASET,
             "forecast_origin": "2022-07-01 00:00:00"},
            {"label": "H2", "role": "history", "dataset": DATASET,
             "forecast_origin": "2022-10-01 00:00:00"},
            {"label": "T1", "role": "test", "dataset": DATASET,
             "forecast_origin": "2023-01-01 00:00:00"},
        ]
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
    validation = json.loads((out / "pool_validation.json").read_text(encoding="utf-8"))
    assert validation["effective_pool"] == POOL
    assert validation["removed"] == {}
    for record in memory["records"]:
        assert record["members"]
        assert set(record["members"]).issubset(set(POOL))
        assert len(record["scenario_profile"]) == 120
        assert record["validation_wape"] >= 0

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
    assert check["readback_total"] == check["readback_passed"] == 2 * len(HORIZONS)
    assert check["coverage_total"] == check["coverage_passed"] == len(HORIZONS)
    assert check["passed"] is True


def test_missing_earlier_history_fails_coverage(tmp_path):
    windows = [
        {"label": "T1", "role": "test", "dataset": DATASET,
         "forecast_origin": "2022-09-01 00:00:00"},
        {"label": "H1", "role": "history", "dataset": DATASET,
         "forecast_origin": "2022-10-01 00:00:00"},
    ]
    definition, plan, shared = _write_inputs(tmp_path, windows=windows)
    out = tmp_path / "out"
    assert _run(tmp_path, definition, plan, shared, out) == 1
    check = json.loads((out / "nn_check.json").read_text(encoding="utf-8"))
    assert check["coverage_passed"] == 0
    assert check["coverage_total"] == len(HORIZONS)
    assert check["passed"] is False
