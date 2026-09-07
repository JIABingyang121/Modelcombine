"""Piece 5（冻结实验定义）与 Piece 7（结果分析）。

冻结要点：只冻结一次、窗口必须齐全且落在数据范围内、训练截止取 T1 的输入历史起点、
深度方法三种子而确定性方法一次。
分析要点：一个任务 = (数据集, 测试窗口, 预测长度)，共 27 个；§8 的"总体更优"必须
同时满足平均 MAE 更低与至少赢 14/27；任务不齐或时间戳不一致时不产出结论。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_final_comparison import (
    EXPECTED_TASKS,
    METHOD_UNDER_TEST,
    WIN_RATE_MIN_TASKS,
    AnalysisError,
    analyse,
)
from scripts.freeze_final_experiment import (
    DEEP_METHOD_SEEDS,
    DETERMINISTIC_METHOD_SEEDS,
    FreezeError,
    build_definition,
)
from tests.forecast_steps_fixtures import (
    DATASET,
    REPO_ROOT,
    write_dataset,
    write_frozen_window_plan,
)

STEPS = 24
ROWS = 1800
DATASETS = ("pjm", "aemo_vic", "aemo_nsw")
METHODS = ("modelcombine", "random_forest", "xgboost")


# ------------------------------------------------------------------ Piece 5
@pytest.fixture
def frozen_inputs(tmp_path):
    raw_root = tmp_path / "raw"
    frames = write_dataset(tmp_path / "splits", rows=ROWS)
    plan_path = write_frozen_window_plan(raw_root, frames, forecast_steps=STEPS)
    plan = json.loads(plan_path.read_text())
    plan["datasets"][0]["fits"] = True
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return {"raw_root": raw_root, "plan": plan_path}


def test_definition_records_dates_cutoff_windows_methods_seeds_and_columns(frozen_inputs):
    definition = build_definition(
        raw_root=frozen_inputs["raw_root"], window_plan_path=frozen_inputs["plan"],
        datasets=[DATASET], methods=list(METHODS), forecast_steps=[STEPS], database=None,
    )

    assert definition["methods"] == list(METHODS)
    assert definition["forecast_horizons"] == {"H1": 24, "H2": 168, "H3": 720}
    assert definition["library_windows"] == ["S1", "S2", "S3"]
    assert definition["audit_windows"] == ["A"]
    assert definition["test_windows"] == ["T1", "T2", "T3"]
    assert definition["input_columns"] and definition["output_columns"][0] == "timestamp"

    # 确定性方法一次，深度方法三个种子
    for method in METHODS:
        assert definition["seeds"][method] == list(DETERMINISTIC_METHOD_SEEDS)
    deep = build_definition(
        raw_root=frozen_inputs["raw_root"], window_plan_path=frozen_inputs["plan"],
        datasets=[DATASET], methods=["modelcombine"], forecast_steps=[STEPS], database=None,
    )
    assert deep["seeds"]["modelcombine"] == list(DETERMINISTIC_METHOD_SEEDS)
    assert list(DEEP_METHOD_SEEDS) == [42, 43, 44]

    entry = definition["datasets"][0]
    plan = json.loads(frozen_inputs["plan"].read_text())
    origins = {o["label"]: o for o in plan["datasets"][0]["origins"]}
    # §5：训练截止取 T1 的输入历史起点
    assert entry["training_cutoff"] == origins["T1"]["history_start"]
    assert entry["training_rows"] > 0
    assert [w["label"] for w in entry["windows"]] == ["S1", "S2", "S3", "A", "T1", "T2", "T3"]


def test_freeze_refuses_when_capacity_or_windows_are_missing(frozen_inputs):
    plan = json.loads(frozen_inputs["plan"].read_text())
    plan["datasets"][0]["fits"] = False
    frozen_inputs["plan"].write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(FreezeError, match="容量不足"):
        build_definition(
            raw_root=frozen_inputs["raw_root"], window_plan_path=frozen_inputs["plan"],
            datasets=[DATASET], methods=list(METHODS), forecast_steps=[STEPS], database=None,
        )

    plan["datasets"][0]["fits"] = True
    plan["datasets"][0]["origins"] = [
        o for o in plan["datasets"][0]["origins"] if o["label"] != "T2"
    ]
    frozen_inputs["plan"].write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(FreezeError, match="缺少"):
        build_definition(
            raw_root=frozen_inputs["raw_root"], window_plan_path=frozen_inputs["plan"],
            datasets=[DATASET], methods=list(METHODS), forecast_steps=[STEPS], database=None,
        )


def test_freeze_refuses_unregistered_method(frozen_inputs):
    with pytest.raises(FreezeError, match="未在统一入口注册"):
        build_definition(
            raw_root=frozen_inputs["raw_root"], window_plan_path=frozen_inputs["plan"],
            datasets=[DATASET], methods=["time_moe"], forecast_steps=[STEPS], database=None,
        )


def test_freeze_does_not_overwrite_an_existing_definition(tmp_path, frozen_inputs):
    out = tmp_path / "experiment_definition.json"
    out.write_text("{}", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable, "scripts/freeze_final_experiment.py",
            "--raw-root", str(frozen_inputs["raw_root"]),
            "--window-plan", str(frozen_inputs["plan"]),
            "--datasets", DATASET, "--methods", *METHODS,
            "--forecast-steps", str(STEPS), "--out", str(out),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "只能冻结一次" in (proc.stdout + proc.stderr)
    assert out.read_text() == "{}"


# ------------------------------------------------------------------ Piece 7
def _long_table(mae_by_method, *, steps=(24, 168, 720), windows=("T1", "T2", "T3")):
    """构造 27 任务长表：每个方法的误差是固定偏移，便于精确断言。"""
    rows = []
    for dataset in DATASETS:
        for window in windows:
            for s in steps:
                stamps = pd.date_range("2026-01-01", periods=s, freq="h")
                y = np.full(s, 100.0)
                for method, offset in mae_by_method.items():
                    rows.append(pd.DataFrame({
                        "timestamp": stamps, "y_true": y, "yhat": y + offset,
                        "method": method, "dataset": dataset, "test_window": window,
                        "forecast_steps": s, "seed": 42,
                    }))
    return pd.concat(rows, ignore_index=True)


def test_analysis_produces_task_dataset_and_horizon_tables():
    frame = _long_table({METHOD_UNDER_TEST: 1.0, "xgboost": 2.0, "random_forest": 4.0})

    report = analyse(frame)

    assert report["conclusions"]["complete_grid"] is True
    assert len(report["task_metrics"]) == EXPECTED_TASKS * 3
    main = {r["method"]: r for r in report["main"]}
    assert main[METHOD_UNDER_TEST]["mae"] == pytest.approx(1.0)
    assert main[METHOD_UNDER_TEST]["tasks"] == EXPECTED_TASKS
    assert main["xgboost"]["mae"] == pytest.approx(2.0)
    # WAPE = 总绝对误差 / 总真值
    assert main[METHOD_UNDER_TEST]["wape"] == pytest.approx(1.0 / 100.0)
    assert len(report["by_dataset"]) == len(DATASETS) * 3
    assert len(report["by_forecast_steps"]) == 3 * 3


def test_wins_and_mean_rank_follow_per_task_mae():
    frame = _long_table({METHOD_UNDER_TEST: 1.0, "xgboost": 2.0, "random_forest": 4.0})

    ranking = analyse(frame)["ranking"]

    assert ranking["n_tasks"] == EXPECTED_TASKS
    rows = {r["method"]: r for r in ranking["methods"]}
    assert rows[METHOD_UNDER_TEST]["wins"] == EXPECTED_TASKS
    assert rows[METHOD_UNDER_TEST]["mean_rank"] == pytest.approx(1.0)
    assert rows["random_forest"]["mean_rank"] == pytest.approx(3.0)


def _long_table_per_task(offsets_by_task):
    """``offsets_by_task[(dataset, window, steps)] = {method: 误差偏移}``。"""
    rows = []
    for (dataset, window, steps), offsets in offsets_by_task.items():
        stamps = pd.date_range("2026-01-01", periods=steps, freq="h")
        y = np.full(steps, 100.0)
        for method, offset in offsets.items():
            rows.append(pd.DataFrame({
                "timestamp": stamps, "y_true": y, "yhat": y + offset,
                "method": method, "dataset": dataset, "test_window": window,
                "forecast_steps": steps, "seed": 42,
            }))
    return pd.concat(rows, ignore_index=True)


def test_overall_superiority_needs_both_mean_mae_and_win_count():
    """平均 MAE 更低但胜场只有 13/27 时，不得表述总体更优。

    13 个任务上大幅领先（0.1 对 5.0），14 个任务上小幅落后（1.0 对 0.9）：
    平均 MAE 明显更低，胜场却不到 14——两条必须同时满足，缺一不可。
    """
    tasks = [
        (dataset, window, steps)
        for dataset in DATASETS for window in ("T1", "T2", "T3")
        for steps in (24, 168, 720)
    ]
    assert len(tasks) == EXPECTED_TASKS
    offsets = {
        task: ({METHOD_UNDER_TEST: 0.1, "xgboost": 5.0} if index < 13
               else {METHOD_UNDER_TEST: 1.0, "xgboost": 0.9})
        for index, task in enumerate(tasks)
    }

    conclusions = analyse(_long_table_per_task(offsets))["conclusions"]
    versus = conclusions["comparisons"][0]

    assert versus["compared_tasks"] == EXPECTED_TASKS
    assert versus["wins"] == 13 < WIN_RATE_MIN_TASKS
    assert versus["mean_mae_lower"] is True
    assert versus["overall_better"] is False
    assert versus["relative_mae_change_pct"] < 0

    # 反面：把领先任务数提到 14，两条同时满足才允许表述总体更优
    offsets = {
        task: ({METHOD_UNDER_TEST: 0.1, "xgboost": 5.0} if index < 14
               else {METHOD_UNDER_TEST: 1.0, "xgboost": 0.9})
        for index, task in enumerate(tasks)
    }
    better = analyse(_long_table_per_task(offsets))["conclusions"]["comparisons"][0]
    assert better["wins"] == 14 and better["overall_better"] is True


def test_incomplete_grid_and_timestamp_mismatch_block_conclusions():
    partial = _long_table({METHOD_UNDER_TEST: 1.0, "xgboost": 2.0}, windows=("T1",))
    assert analyse(partial)["conclusions"]["complete_grid"] is False

    frame = _long_table({METHOD_UNDER_TEST: 1.0, "xgboost": 2.0})
    mask = (frame["method"] == "xgboost") & (frame["forecast_steps"] == 24)
    frame.loc[mask, "timestamp"] = pd.Timestamp("2030-01-01")
    with pytest.raises(AnalysisError, match="目标时间戳不一致"):
        analyse(frame)
