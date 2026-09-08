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
            datasets=[DATASET], methods=["timespeaks"], forecast_steps=[STEPS], database=None,
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
def _definition(methods, *, steps=(24, 168, 720), windows=("T1", "T2", "T3"), seeds=(42,)):
    return {
        "methods": list(methods),
        "seeds": {m: list(seeds) for m in methods},
        "test_windows": list(windows),
        "forecast_steps": list(steps),
        "datasets": [{"dataset": d} for d in DATASETS],
    }


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

    report = analyse(frame, _definition(set(frame["method"])))

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

    ranking = analyse(frame, _definition(set(frame["method"])))["ranking"]

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

    conclusions = analyse(_long_table_per_task(offsets), _definition([METHOD_UNDER_TEST, "xgboost"]))["conclusions"]
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
    better = analyse(_long_table_per_task(offsets), _definition([METHOD_UNDER_TEST, "xgboost"]))["conclusions"]["comparisons"][0]
    assert better["wins"] == 14 and better["overall_better"] is True


def test_incomplete_grid_and_timestamp_mismatch_block_conclusions():
    partial = _long_table({METHOD_UNDER_TEST: 1.0, "xgboost": 2.0}, windows=("T1",))
    assert analyse(partial, _definition(set(partial["method"])))["conclusions"]["complete_grid"] is False

    frame = _long_table({METHOD_UNDER_TEST: 1.0, "xgboost": 2.0})
    mask = (frame["method"] == "xgboost") & (frame["forecast_steps"] == 24)
    frame.loc[mask, "timestamp"] = frame.loc[mask, "timestamp"] + pd.Timedelta(days=365)
    with pytest.raises(AnalysisError, match="目标时间戳不一致"):
        analyse(frame, _definition(set(frame["method"])))


def test_definition_records_seed_policy_and_method_limitations(frozen_inputs):
    """深度方法三种子；iTransformer 因官方无种子参数按一次算；Time-MoE 限制必须落盘。"""
    definition = build_definition(
        raw_root=frozen_inputs["raw_root"], window_plan_path=frozen_inputs["plan"],
        datasets=[DATASET], methods=["itransformer", "mole", "time_moe"],
        external_config=EXTERNAL_CONFIG,
        forecast_steps=[STEPS], database=None,
    )

    # 只有 MoLE 真正接收请求种子，三种子才有意义
    assert definition["seeds"]["mole"] == list(DEEP_METHOD_SEEDS)
    # iTransformer 官方 run.py 固定 fix_seed=2023；Time-MoE 是零样本确定性贪心生成。
    # 两者跑多种子都只会得到相同结果，不构成独立样本，因此各跑一次。
    # 结果表记的是官方硬编码的 2023，冻结定义必须记同一个值，否则完整性核对必然失败
    assert definition["seeds"]["itransformer"] == [2023]
    assert definition["seeds"]["time_moe"] == list(DETERMINISTIC_METHOD_SEEDS)
    assert set(definition["seed_insensitive"]) == {"itransformer", "time_moe"}
    assert definition["official_fixed_seeds"]["itransformer"] == 2023

    # Time-MoE 的预训练截止不可证，必须随定义一起落盘
    assert "time_moe" in definition["method_limitations"]
    assert "预训练" in definition["method_limitations"]["time_moe"]
    assert "mole" not in definition["method_limitations"]


def test_conclusions_carry_method_limitations():
    """结论文件必须带上限制，避免 Time-MoE 的比较被读成同等条件下的比较。"""
    frame = _long_table({METHOD_UNDER_TEST: 1.0, "time_moe": 2.0, "mole": 3.0})

    conclusions = analyse(frame, _definition(set(frame["method"])))["conclusions"]

    assert "time_moe" in conclusions["method_limitations"]
    assert "mole" not in conclusions["method_limitations"]


def test_single_method_grid_is_not_treated_as_complete():
    """只有 Modelcombine 一个方法的 27 任务长表，不得被判成完整实验。

    这是旧实现的漏洞：只统计任务键，27 个任务键凑齐就 complete_grid=true，
    comparisons 却是空的——某个外部方法完全缺失也看不出来。
    """
    frame = _long_table({METHOD_UNDER_TEST: 1.0})
    definition = _definition([METHOD_UNDER_TEST, "xgboost", "mole"])

    report = analyse(frame, definition)

    completeness = report["completeness"]
    assert completeness["passed"] is False
    assert any("方法集合不符" in p for p in completeness["problems"])
    assert report["conclusions"]["comparisons"] == []


def test_missing_seed_for_one_method_blocks_conclusions():
    """某方法少跑一个种子时，即使任务齐全也不得给出总体更优结论。"""
    frame = _long_table({METHOD_UNDER_TEST: 1.0, "mole": 5.0})
    definition = _definition([METHOD_UNDER_TEST, "mole"])
    definition["seeds"]["mole"] = [42, 43, 44]  # 长表里只有 42

    report = analyse(frame, definition)

    assert report["completeness"]["passed"] is False
    assert any("种子集合" in p for p in report["completeness"]["problems"])
    versus = report["conclusions"]["comparisons"][0]
    assert versus["mean_mae_lower"] is True
    assert versus["overall_better"] is False, "产物不完整时不得表述总体更优"


def test_complete_run_matching_the_definition_allows_conclusions():
    frame = _long_table({METHOD_UNDER_TEST: 1.0, "mole": 5.0})
    definition = _definition([METHOD_UNDER_TEST, "mole"])

    report = analyse(frame, definition)

    assert report["completeness"]["passed"] is True
    versus = report["conclusions"]["comparisons"][0]
    assert versus["compared_all_tasks"] is True
    assert versus["wins"] == EXPECTED_TASKS
    assert versus["overall_better"] is True


def test_analysis_without_a_definition_cannot_conclude():
    frame = _long_table({METHOD_UNDER_TEST: 1.0, "mole": 5.0})

    report = analyse(frame)

    assert report["completeness"]["checked_against_definition"] is False
    assert report["conclusions"]["comparisons"][0]["overall_better"] is False


# --------------------------------------------- 多种子方法不得被读成"重复时间戳"
def _with_three_seeds(frame, method):
    rows = frame[frame["method"] == method]
    return pd.concat(
        [frame[frame["method"] != method]]
        + [rows.assign(seed=seed) for seed in (42, 43, 44)],
        ignore_index=True,
    )


def test_multi_seed_method_is_not_read_as_inconsistent_timestamps():
    """MoLE 三种子共 3H 行、H 个唯一时间戳；确定性方法只有 H 行。

    按方法把全部行的时间戳拼成元组来比较，会把这个正确结构判成"目标时间戳不一致"。
    """
    frame = _with_three_seeds(
        _long_table({METHOD_UNDER_TEST: 1.0, "mole": 2.0}), "mole"
    )
    definition = _definition([METHOD_UNDER_TEST, "mole"])
    definition["seeds"]["mole"] = [42, 43, 44]

    report = analyse(frame, definition)

    assert report["completeness"]["passed"] is True, report["completeness"]["problems"]
    assert report["conclusions"]["complete_grid"] is True


def test_seed_must_be_complete_in_every_task_not_only_globally():
    """某一个任务缺了 seed=44，但全局种子并集仍是 {42,43,44}——必须被查出来。"""
    frame = _with_three_seeds(
        _long_table({METHOD_UNDER_TEST: 1.0, "mole": 2.0}), "mole"
    )
    drop = (
        (frame["method"] == "mole") & (frame["seed"] == 44)
        & (frame["dataset"] == DATASETS[0]) & (frame["test_window"] == "T1")
        & (frame["forecast_steps"] == 24)
    )
    assert drop.any()
    frame = frame[~drop].reset_index(drop=True)
    definition = _definition([METHOD_UNDER_TEST, "mole"])
    definition["seeds"]["mole"] = [42, 43, 44]

    report = analyse(frame, definition)

    assert report["completeness"]["passed"] is False
    assert any("44" in p for p in report["completeness"]["problems"])
    assert all(
        c["overall_better"] is False for c in report["conclusions"]["comparisons"]
    )


def test_duplicate_timestamps_within_one_seed_are_rejected():
    """同一个 (方法, 种子) 内出现重复时间戳必须报错，行数对上也不行。"""
    frame = _long_table({METHOD_UNDER_TEST: 1.0, "mole": 2.0})
    task = (
        (frame["method"] == "mole") & (frame["dataset"] == DATASETS[0])
        & (frame["test_window"] == "T1") & (frame["forecast_steps"] == 24)
    )
    index = frame.index[task]
    frame.loc[index[-1], "timestamp"] = frame.loc[index[0], "timestamp"]

    with pytest.raises(AnalysisError, match="重复"):
        analyse(frame, _definition([METHOD_UNDER_TEST, "mole"]))


# ------------------------------------------- 外部方法的正式口径必须进入冻结定义
EXTERNAL_CONFIG = {
    "itransformer": {
        "repo": "/srv/iTransformer", "python": "/srv/venv/bin/python",
        "checkpoints": "/srv/ckpt",
        "commit": "c2426e68ca13f74aaec08045c5c724d8ad328124",
        "hyperparameters": {
            "seq_len": 96, "label_len": 48, "d_model": 512, "n_heads": 8,
            "e_layers": 3, "d_layers": 1, "d_ff": 512, "factor": 1, "dropout": 0.1,
            "train_epochs": 10, "batch_size": 32, "patience": 3,
            "learning_rate": 0.0001, "des": "final",
        },
    },
    "mole": {
        "repo": "/srv/MoLE", "python": "/srv/venv/bin/python", "checkpoints": "/srv/ckpt",
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "hyperparameters": {
            "seq_len": 336, "t_dim": 4, "train_epochs": 10, "batch_size": 32,
            "patience": 3, "learning_rate": 0.0001, "des": "final",
        },
    },
    "time_moe": {
        "repo": "/srv/Time-MoE", "python": "/srv/venv2/bin/python",
        "snapshot": "/srv/hf/models--Maple728--TimeMoE-50M/snapshots/abc",
        "commit": "89abcdef0123456789abcdef0123456789abcdef",
        "checkpoint_id": "Maple728/TimeMoE-50M@main",
        "context_length": 720,
    },
}


def test_definition_freezes_external_hyperparameters_code_version_and_checkpoint(
    frozen_inputs,
):
    definition = build_definition(
        raw_root=frozen_inputs["raw_root"], window_plan_path=frozen_inputs["plan"],
        datasets=[DATASET], methods=["itransformer", "mole", "time_moe"],
        forecast_steps=[STEPS], database=None, external_config=EXTERNAL_CONFIG,
    )

    external = definition["external"]
    assert external["itransformer"]["hyperparameters"]["d_model"] == 512
    assert external["itransformer"]["commit"] == EXTERNAL_CONFIG["itransformer"]["commit"]
    assert external["mole"]["hyperparameters"]["seq_len"] == 336
    assert external["time_moe"]["checkpoint_id"] == "Maple728/TimeMoE-50M@main"
    assert external["time_moe"]["context_length"] == 720
    # 机器本地路径不属于口径，不得写进定义
    for method, frozen in external.items():
        assert "repo" not in frozen and "python" not in frozen, method


def test_freeze_refuses_external_method_without_config(frozen_inputs):
    with pytest.raises(FreezeError, match="external-config"):
        build_definition(
            raw_root=frozen_inputs["raw_root"], window_plan_path=frozen_inputs["plan"],
            datasets=[DATASET], methods=["itransformer"], forecast_steps=[STEPS],
            database=None, external_config=None,
        )


def test_freeze_refuses_external_config_missing_commit(frozen_inputs):
    config = {"itransformer": dict(EXTERNAL_CONFIG["itransformer"])}
    config["itransformer"].pop("commit")
    with pytest.raises(FreezeError, match="commit"):
        build_definition(
            raw_root=frozen_inputs["raw_root"], window_plan_path=frozen_inputs["plan"],
            datasets=[DATASET], methods=["itransformer"], forecast_steps=[STEPS],
            database=None, external_config=config,
        )
