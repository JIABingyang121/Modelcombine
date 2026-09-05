"""组合器 interaction 分支的离线—在线一致性（走真实 run.py predict）。

方案 §3.3：第一版历史数据契约下不使用需要未来天气的 interaction；日历特征可由
未来时间戳生成。本用例构造一个残差随 hour 线性变化的成员，逼出真实的日历特征
interaction，再验证保存后的组合器在在线入口上逐点重放离线结果。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tests.forecast_steps_fixtures import (
    DATASET,
    REPO_ROOT,
    WEEKLY_FEATURES,
    fit_weekly,
    register_models,
    run_predict,
    task_of,
    write_dataset,
    write_history,
    write_scenario,
)

STEPS = 168
ROWS = 1100  # 720 小时 signature 窗口 + 168 步目标轨迹 + 余量


def _build_library_with_hour_dependent_residual(tmp_path: Path):
    raw_root = tmp_path / "features"
    db = tmp_path / "lib.sqlite3"
    frames = write_dataset(raw_root, rows=ROWS)
    register_models(db, tmp_path / "artifacts", [
        ("catboost_reg", fit_weekly(frames["train"]), WEEKLY_FEATURES),
        ("lgbm_reg", fit_weekly(frames["train"], hour_bias=3.0), WEEKLY_FEATURES),
    ])

    build = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_combinations_kg", "--model-library",
            "--datasets", DATASET, "--forecast-steps", str(STEPS),
            "--candidates", "catboost_reg", "lgbm_reg",
            "--raw-root", str(raw_root), "--out-root", str(tmp_path / "ml"),
            "--database", str(db), "--model-artifacts", str(tmp_path / "combo_artifacts"),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    report = json.loads((tmp_path / "ml" / "model_library_report.json").read_text())
    return task_of(report, STEPS), db, frames


def test_interaction_combo_replays_through_run_py_predict(tmp_path):
    task, db, frames = _build_library_with_hour_dependent_residual(tmp_path)
    assert task["has_interaction"] is True, "装置未触发 interaction，用例失去意义"

    history = write_history(tmp_path, frames["test"], task["test_origin"], "history.csv")
    scenario = write_scenario(tmp_path, forecast_steps=STEPS)
    output = tmp_path / "online.csv"

    predict = run_predict(db, scenario, history, output)
    assert predict.returncode == 0, predict.stdout + predict.stderr

    trace = json.loads(output.with_suffix(".trace.json").read_text())
    assert trace["has_interaction"] is True
    assert trace["forecast_steps"] == STEPS

    online = pd.read_csv(output)["yhat"].to_numpy(dtype=float)
    offline = np.asarray(task["test_trajectory"], dtype=float)
    assert len(online) == len(offline) == STEPS
    np.testing.assert_allclose(online, offline, rtol=0, atol=1e-8)
