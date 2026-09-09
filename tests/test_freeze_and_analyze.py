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
from scripts.train_combinations_kg import TIMESTAMP_POLICY
from scripts.freeze_final_experiment import (
    DEEP_METHOD_SEEDS,
    DETERMINISTIC_METHOD_SEEDS,
    FreezeError,
    build_definition,
)
from tests.library_fixtures import make_complete_library
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
FIXTURE_CANDIDATES = ("lgbm_reg", "seasonal_naive")


@pytest.fixture
def frozen_inputs(tmp_path):
    """窗口计划 + 一个内部完整的模型库（DATASET 一个数据集、STEPS 一个长度）。

    正式冻结的"恰好 7 方法 / 3 数据集 / 3 长度"由 CLI 的 assert_formal_scope 强制；
    build_definition 本身只要求库完整、候选与 pipeline 一致，因此可以用小装置测。
    """
    raw_root = tmp_path / "raw"
    frames = write_dataset(tmp_path / "splits", rows=ROWS)
    plan_path = write_frozen_window_plan(raw_root, frames, forecast_steps=STEPS)
    plan = json.loads(plan_path.read_text())
    plan["datasets"][0]["fits"] = True
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    library = make_complete_library(
        tmp_path / "library", datasets=[DATASET], forecast_steps=[STEPS],
        candidates=list(FIXTURE_CANDIDATES),
    )
    return {"raw_root": raw_root, "plan": plan_path, "library": library}


def _freeze(frozen_inputs, **over):
    """带上库、候选与 pipeline 的 build_definition 调用。"""
    kwargs = dict(
        raw_root=frozen_inputs["raw_root"], window_plan_path=frozen_inputs["plan"],
        datasets=[DATASET], methods=list(METHODS), forecast_steps=[STEPS],
        database=frozen_inputs["library"]["database"],
        candidates=list(FIXTURE_CANDIDATES),
        pipeline_config=frozen_inputs["library"]["pipeline"],
        library_report=frozen_inputs["library"]["report"],
    )
    kwargs.update(over)
    return build_definition(**kwargs)


def test_definition_records_dates_cutoff_windows_methods_seeds_and_columns(frozen_inputs):
    definition = _freeze(frozen_inputs,
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
    deep = _freeze(frozen_inputs,
        methods=["modelcombine"],
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
        _freeze(frozen_inputs,
        )

    plan["datasets"][0]["fits"] = True
    plan["datasets"][0]["origins"] = [
        o for o in plan["datasets"][0]["origins"] if o["label"] != "T2"
    ]
    frozen_inputs["plan"].write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(FreezeError, match="缺少"):
        _freeze(frozen_inputs,
        )


def test_freeze_refuses_unregistered_method(frozen_inputs):
    with pytest.raises(FreezeError, match="未在统一入口注册"):
        _freeze(frozen_inputs,
            methods=["timespeaks"],
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
            "--database", str(frozen_inputs["library"]["database"]),
            "--library-report", str(frozen_inputs["library"]["report"]),
            "--candidates", *FIXTURE_CANDIDATES,
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
    definition = _freeze(frozen_inputs,
        methods=["itransformer", "mole", "time_moe"], external_config=EXTERNAL_CONFIG,
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
    definition = _freeze(frozen_inputs,
        methods=["itransformer", "mole", "time_moe"], external_config=EXTERNAL_CONFIG,
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
        _freeze(frozen_inputs,
            methods=["itransformer"], external_config=None,
        )


def test_freeze_refuses_external_config_missing_commit(frozen_inputs):
    config = {"itransformer": dict(EXTERNAL_CONFIG["itransformer"])}
    config["itransformer"].pop("commit")
    with pytest.raises(FreezeError, match="commit"):
        _freeze(frozen_inputs,
            methods=["itransformer"], external_config=config,
        )


# ------------------------------- 冻结定义必须真正约束正式运行（不只是被记录）
def _windows(dataset: str):
    return [
        {"label": label, "role": role,
         "history_start": f"2026-0{i+1}-01 00:00:00",
         "history_end": f"2026-0{i+1}-02 00:00:00",
         "forecast_origin": f"2026-0{i+1}-02 00:00:00",
         "targets": {"24": {"forecast_steps": 24,
                            "first_target": f"2026-0{i+1}-02 01:00:00",
                            "last_target": f"2026-0{i+1}-03 00:00:00"}}}
        for i, (label, role) in enumerate(
            [("S1", "library"), ("S2", "library"), ("S3", "library"), ("A", "audit"),
             ("T1", "test"), ("T2", "test"), ("T3", "test")])
    ]


def _run_definition(tmp_path, **over):
    """构造一份与运行参数一致的定义（含真实窗口计划文件），再逐项破坏它。"""
    import subprocess as _sp
    commit = _sp.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
    plan = {"datasets": [
        {"dataset": d, "fits": True,
         "origins": [{k: v for k, v in w.items() if k != "role"} | {"label": w["label"]}
                     for w in _windows(d)]}
        for d in DATASETS
    ]}
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    library = make_complete_library(
        tmp_path, datasets=DATASETS, forecast_steps=[24, 168, 720],
        candidates=["lgbm_reg", "seasonal_naive"],
    )
    base = {
        "repo_commit": commit,
        "datasets": [{"dataset": d, "windows": _windows(d)} for d in DATASETS],
        "test_windows": ["T1", "T2", "T3"],
        "forecast_steps": [24, 168, 720],
        "raw_root": str((tmp_path / "raw").resolve()),
        "window_plan": str((tmp_path / "plan.json").resolve()),
        "database": str(library["database"].resolve()),
        "candidates": ["lgbm_reg", "seasonal_naive"],
        "library_report": str(library["report"].resolve()),
        "timestamp_policy": TIMESTAMP_POLICY,
        "seeds": {"modelcombine": [42]},
    }
    base.update(over)
    return base


class _Args:
    def __init__(self, tmp_path, **over):
        self.datasets = list(DATASETS)
        self.windows = ["T1", "T2", "T3"]
        self.forecast_steps = [24, 168, 720]
        self.raw_root = tmp_path / "raw"
        self.window_plan = tmp_path / "plan.json"
        self.database = tmp_path / "lib.sqlite3"
        self.candidates = ["lgbm_reg", "seasonal_naive"]
        for k, v in over.items():
            setattr(self, k, v)


@pytest.fixture
def clean_repo(monkeypatch):
    """把主仓库状态打成"干净且正好是冻结的那个提交"。

    本机开发时工作树必然是脏的，正例断言无法依赖真实仓库状态；脏/干净判定本身由
    test_dirty_worktree_blocks_the_run 单独覆盖。
    """
    import scripts.final_comparison as fc
    commit = "a" * 40
    monkeypatch.setattr(fc, "repo_state", lambda: (commit, []))
    return commit


def test_run_parameters_must_match_the_frozen_definition(tmp_path, clean_repo):
    from scripts.final_comparison import FinalComparisonError, assert_matches_definition

    definition = _run_definition(tmp_path, repo_commit=clean_repo)
    assert assert_matches_definition(_Args(tmp_path), definition) is None

    for field, value, needle in [
        ("datasets", ["pjm"], "--datasets"),
        ("windows", ["T1"], "--windows"),
        ("forecast_steps", [24], "--forecast-steps"),
        ("raw_root", tmp_path / "other_raw", "--raw-root"),
        ("window_plan", tmp_path / "other_plan.json", "--window-plan"),
        ("database", tmp_path / "other.sqlite3", "--database"),
        ("candidates", ["lgbm_reg"], "--candidates"),
    ]:
        with pytest.raises(FinalComparisonError, match=needle):
            assert_matches_definition(_Args(tmp_path, **{field: value}), definition)


def test_repo_commit_mismatch_blocks_the_run(tmp_path, clean_repo):
    """冻结定义记的主仓库版本与当前不符时，不得在 T1—T3 上运行。"""
    from scripts.final_comparison import FinalComparisonError, assert_matches_definition

    definition = _run_definition(tmp_path, repo_commit="0" * 40)
    with pytest.raises(FinalComparisonError, match="主仓库版本"):
        assert_matches_definition(_Args(tmp_path), definition)


def test_dirty_worktree_blocks_the_run(tmp_path, monkeypatch):
    """HEAD 相同但受跟踪源码被改过，跑的就不是冻结的那份代码。"""
    import scripts.final_comparison as fc
    from scripts.final_comparison import FinalComparisonError, assert_matches_definition

    commit = "b" * 40
    monkeypatch.setattr(fc, "repo_state", lambda: (commit, [" M scripts/final_comparison.py"]))
    definition = _run_definition(tmp_path, repo_commit=commit)
    with pytest.raises(FinalComparisonError, match="未提交修改"):
        assert_matches_definition(_Args(tmp_path), definition)


def test_window_plan_overwritten_at_the_same_path_is_caught(tmp_path, clean_repo):
    """同一路径覆盖窗口计划：路径不变，内容变了，必须被抓住。"""
    from scripts.final_comparison import FinalComparisonError, assert_matches_definition

    definition = _run_definition(tmp_path, repo_commit=clean_repo)
    plan = json.loads((tmp_path / "plan.json").read_text())
    plan["datasets"][0]["origins"][4]["forecast_origin"] = "2099-01-01 00:00:00"
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(FinalComparisonError, match="窗口计划的内容与冻结定义不一致"):
        assert_matches_definition(_Args(tmp_path), definition)


# ------------------------------------------ 多份预测长表合并，且拒绝重复行
def test_multiple_prediction_files_are_merged(tmp_path):
    from scripts.analyze_final_comparison import load_predictions

    frame = _long_table({METHOD_UNDER_TEST: 1.0, "mole": 2.0})
    a = frame[frame["method"] == METHOD_UNDER_TEST]
    b = frame[frame["method"] == "mole"]
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    a.to_csv(pa, index=False)
    b.to_csv(pb, index=False)

    merged = load_predictions([pa, pb])

    assert len(merged) == len(frame)
    assert set(merged["method"]) == {METHOD_UNDER_TEST, "mole"}
    report = analyse(merged, _definition([METHOD_UNDER_TEST, "mole"]))
    assert report["conclusions"]["complete_grid"] is True


def test_duplicate_rows_across_files_are_refused(tmp_path):
    """同一份产物传两次会让该任务被算重，必须报错而不是去重。"""
    from scripts.analyze_final_comparison import AnalysisError, load_predictions

    frame = _long_table({METHOD_UNDER_TEST: 1.0, "mole": 2.0})
    path = tmp_path / "all.csv"
    frame.to_csv(path, index=False)

    load_predictions([path])
    with pytest.raises(AnalysisError, match="重复"):
        load_predictions([path, path])


def test_duplicate_rows_within_one_file_are_refused(tmp_path):
    from scripts.analyze_final_comparison import AnalysisError, load_predictions

    frame = _long_table({METHOD_UNDER_TEST: 1.0, "mole": 2.0})
    doubled = pd.concat([frame, frame.head(3)], ignore_index=True)
    path = tmp_path / "dup.csv"
    doubled.to_csv(path, index=False)

    with pytest.raises(AnalysisError, match="重复"):
        load_predictions([path])


# ----------------------------- 实验定义的时间戳策略必须在访问 T 之前形成闸门
@pytest.mark.parametrize("policy, needle", [
    (None, "时间戳策略"),
    ("raw_rows_as_is", "时间戳策略"),
])
def test_definition_timestamp_policy_must_match_before_run(tmp_path, clean_repo,
                                                           policy, needle):
    """缺失或被篡改都必须拒绝——不给默认值、不做迁移。

    旧定义因此全部失效，这是预期行为：同一段历史在两种策略下切出的序列不同。
    """
    from scripts.final_comparison import FinalComparisonError, assert_matches_definition

    definition = _run_definition(tmp_path, repo_commit=clean_repo)
    if policy is None:
        definition.pop("timestamp_policy")
    else:
        definition["timestamp_policy"] = policy

    with pytest.raises(FinalComparisonError, match=needle):
        assert_matches_definition(_Args(tmp_path), definition)


def test_definition_with_a_stale_library_report_policy_is_refused(tmp_path, clean_repo):
    """定义里策略对，但它指向的建库报告是旧策略建的——同样必须在读 T 之前拒绝。"""
    from scripts.final_comparison import FinalComparisonError, assert_matches_definition

    definition = _run_definition(tmp_path, repo_commit=clean_repo)
    report_path = Path(definition["library_report"])
    report = json.loads(report_path.read_text())
    report["timestamp_policy"] = "raw_rows_as_is"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(FinalComparisonError, match="不是用当前策略建的"):
        assert_matches_definition(_Args(tmp_path), definition)
