"""Stage 2 三基线质量门控（scripts/stage2_quality_gate.py）。

这套评估代码要在**看到真实 Q1—Q3 真值之前**冻结，所以它必须先在合成数据上把
以下几件事钉死：基线只由 H1—H3 决定、MASE 分母取训练段而不是查询窗口、门槛比较
方向和边界严格按 §11.2、运行不完整时不产出任何结论。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.stage2_quality_gate import (
    BASELINE_BEST_SINGLE,
    BASELINE_RIDGE,
    BASELINE_SEASONAL_NAIVE,
    EXIT_GATE_FAILED,
    EXIT_INCOMPLETE,
    EXIT_OK,
    MASE_SEASONAL_PERIOD,
    METHOD_MODELCOMBINE,
    METHODS,
    evaluate_gates,
)
from tests.forecast_steps_fixtures import (
    DATASET,
    FIXTURE_CANDIDATES,
    REPO_ROOT,
    seed_models,
    write_dataset,
    write_frozen_window_plan,
)

STEPS = 24
ROWS = 1800  # 训练段 + 720 小时 signature 窗口 + 7×24 目标


def _build_stage1(tmp_path: Path):
    """真实建库：冻结窗口计划 -> H1—H3 三条关系（A 作审计窗口）。"""
    raw_root = tmp_path / "raw"
    db = tmp_path / "lib.sqlite3"
    frames = write_dataset(tmp_path / "splits", rows=ROWS)
    seed_models(db, tmp_path / "artifacts", frames["train"])
    window_plan = write_frozen_window_plan(raw_root, frames, forecast_steps=STEPS)
    out_root = tmp_path / "library"

    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_combinations_kg", "--model-library",
            "--datasets", DATASET, "--forecast-steps", str(STEPS),
            "--candidates", *FIXTURE_CANDIDATES,
            "--raw-root", str(raw_root), "--window-plan", str(window_plan),
            "--out-root", str(out_root), "--database", str(db),
            "--model-artifacts", str(tmp_path / "combo_artifacts"),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return {
        "raw_root": raw_root, "db": db, "window_plan": window_plan,
        "library_report": out_root / "model_library_report.json",
    }


def _run_stage2(tmp_path: Path, built: dict, *, out_name="stage2", library_report=None):
    out = tmp_path / out_name
    proc = subprocess.run(
        [
            sys.executable, "scripts/stage2_quality_gate.py",
            "--database", str(built["db"]), "--raw-root", str(built["raw_root"]),
            "--window-plan", str(built["window_plan"]),
            "--library-report", str(library_report or built["library_report"]),
            "--datasets", DATASET, "--forecast-steps", str(STEPS),
            "--candidates", *FIXTURE_CANDIDATES,
            "--out", str(out),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return proc, out


@pytest.fixture(scope="module")
def stage2(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("stage2")
    built = _build_stage1(tmp_path)
    proc, out = _run_stage2(tmp_path, built)
    assert proc.returncode in (EXIT_OK, EXIT_GATE_FAILED), proc.stdout + proc.stderr
    return {
        "tmp_path": tmp_path, "built": built, "proc": proc, "out": out,
        "metrics": json.loads((out / "main_metrics.json").read_text()),
        "acceptance": json.loads((out / "acceptance.json").read_text()),
        "definition": json.loads((out / "experiment_definition.json").read_text()),
    }


# --------------------------------------------------------------- 产出与契约
def test_every_query_window_gets_all_four_methods(stage2):
    task = stage2["metrics"]["tasks"][0]
    assert [q["window"] for q in task["queries"]] == ["Q1", "Q2", "Q3"]

    for query in task["queries"]:
        assert set(query["metrics"]) == set(METHODS)
        for method in METHODS:
            scores = query["metrics"][method]
            assert set(scores) == {"mae", "rmse", "mase"}
            assert all(np.isfinite(v) and v >= 0 for v in scores.values())
        # §11.1：输出长度=请求长度=trace 长度，且在线不调用选择器
        assert query["n_rows"] == STEPS
        assert query["trace"]["forecast_steps"] == STEPS
        assert query["trace"]["n_rows"] == STEPS
        assert query["trace"]["selector_invoked"] is False

    predictions = stage2["out"] / "predictions"
    for window in ("Q1", "Q2", "Q3"):
        frame = pd.read_csv(predictions / f"{DATASET}_s{STEPS}_{window}.csv")
        assert len(frame) == STEPS
        assert set(METHODS) <= set(frame.columns) and "y" in frame.columns


def test_exit_code_matches_acceptance_verdict(stage2):
    expected = EXIT_OK if stage2["acceptance"]["passed"] else EXIT_GATE_FAILED
    assert stage2["proc"].returncode == expected


def test_frozen_constants_are_recorded_in_the_definition(stage2):
    frozen = stage2["definition"]["frozen_constants"]
    assert frozen["mase_seasonal_period"] == MASE_SEASONAL_PERIOD
    assert frozen["thresholds"]["11.2.1a_best_single_mean_ratio_lt"] == 1.00
    assert frozen["thresholds"]["11.2.1b_best_single_per_dataset_ratio_le"] == 1.03
    assert frozen["thresholds"]["11.2.2_seasonal_naive_per_task_ratio_lt"] == 1.00
    assert frozen["thresholds"]["11.2.3_ridge_mean_ratio_le"] == 1.00
    assert stage2["definition"]["library_windows"] == ["H1", "H2", "H3"]
    assert stage2["definition"]["query_windows"] == ["Q1", "Q2", "Q3"]


# ------------------------------------------------------- MASE 分母取训练段
def test_mase_denominator_comes_from_the_training_segment(stage2):
    task = stage2["metrics"]["tasks"][0]
    raw = pd.read_csv(stage2["built"]["raw_root"] / DATASET / "load.csv")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    plan = json.loads(stage2["built"]["window_plan"].read_text())
    h1 = next(
        o for o in plan["datasets"][0]["feasibility"][str(STEPS)]["origins"]
        if o["label"] == "H1"
    )
    segment = raw[raw["timestamp"] < pd.Timestamp(h1["history_start"])]["load"].to_numpy(float)
    expected = float(np.mean(np.abs(segment[MASE_SEASONAL_PERIOD:]
                                    - segment[:-MASE_SEASONAL_PERIOD])))

    assert task["mase_scale"] == pytest.approx(expected, rel=0, abs=1e-9)
    assert task["mase_train_segment"]["rows"] == len(segment)
    # mase 就是 mae / scale，且分母不是查询窗口上 Seasonal Naive 的 MAE
    query = task["queries"][0]
    scores = query["metrics"][METHOD_MODELCOMBINE]
    assert scores["mase"] == pytest.approx(scores["mae"] / task["mase_scale"], rel=1e-12)
    assert task["mase_scale"] != pytest.approx(
        query["metrics"][BASELINE_SEASONAL_NAIVE]["mae"], rel=1e-6
    )


# ----------------------------------------------- 基线只由 H1—H3 冻结，不看 Q
def test_baselines_are_frozen_on_validation_and_ignore_query_windows(tmp_path):
    """扰动 Q1—Q3 区段的真值：Best Single 的选择与 Ridge 的系数必须一字不变。

    这是 §8 公平性规则在 Stage 2 侧的对应保证——基线若被查询窗口影响，
    整个比较就不再是"未见窗口"。
    """
    built = _build_stage1(tmp_path)
    proc_a, out_a = _run_stage2(tmp_path, built, out_name="stage2_a")
    assert proc_a.returncode in (EXIT_OK, EXIT_GATE_FAILED), proc_a.stdout + proc_a.stderr
    task_a = json.loads((out_a / "main_metrics.json").read_text())["tasks"][0]

    plan = json.loads(built["window_plan"].read_text())
    q1 = next(
        o for o in plan["datasets"][0]["feasibility"][str(STEPS)]["origins"]
        if o["label"] == "Q1"
    )
    load_path = built["raw_root"] / DATASET / "load.csv"
    raw = pd.read_csv(load_path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    mask = raw["timestamp"] >= pd.Timestamp(q1["first_target"])
    assert mask.any(), "扰动区间为空，断言无意义"
    raw.loc[mask, "load"] = raw.loc[mask, "load"] + 500.0
    raw.to_csv(load_path, index=False)

    proc_b, out_b = _run_stage2(tmp_path, built, out_name="stage2_b")
    assert proc_b.returncode in (EXIT_OK, EXIT_GATE_FAILED), proc_b.stdout + proc_b.stderr
    task_b = json.loads((out_b / "main_metrics.json").read_text())["tasks"][0]

    assert task_a["validation_best_single"] == task_b["validation_best_single"]
    assert task_a["frozen_baseline_columns"] == task_b["frozen_baseline_columns"]
    assert task_a["ridge_coef"] == pytest.approx(task_b["ridge_coef"], abs=1e-12)
    assert task_a["ridge_intercept"] == pytest.approx(task_b["ridge_intercept"], abs=1e-12)
    assert task_a["validation_stacked_mae"] == pytest.approx(
        task_b["validation_stacked_mae"], abs=1e-12
    )
    # 扰动确实生效：Q 上的指标必须变了，否则上面的不变性是空断言
    assert task_a["task_metrics"][METHOD_MODELCOMBINE]["mae"] != pytest.approx(
        task_b["task_metrics"][METHOD_MODELCOMBINE]["mae"], rel=1e-6
    )


def test_validation_mismatch_with_library_report_is_incomplete_not_a_verdict(tmp_path):
    """Stage 2 重算的 validation 与建库报告对不上时，不得给出任何门槛结论。"""
    built = _build_stage1(tmp_path)
    report = json.loads(built["library_report"].read_text())
    for task in report["tasks"]:
        for scores in task["candidate_validation_mae"].values():
            scores["trajectory_mae"] = float(scores["trajectory_mae"]) + 1.0
    tampered = tmp_path / "tampered_report.json"
    tampered.write_text(json.dumps(report), encoding="utf-8")

    proc, out = _run_stage2(tmp_path, built, out_name="stage2_bad", library_report=tampered)

    assert proc.returncode == EXIT_INCOMPLETE
    assert "不是同一批 validation" in (proc.stdout + proc.stderr)
    assert not (out / "acceptance.json").exists()
    assert not (out / "main_metrics.json").exists()


# ----------------------------------------------------------- 门槛比较方向
def _task(dataset, steps, *, mc, best_single, seasonal, ridge):
    metrics = {
        METHOD_MODELCOMBINE: mc, BASELINE_BEST_SINGLE: best_single,
        BASELINE_SEASONAL_NAIVE: seasonal, BASELINE_RIDGE: ridge,
    }
    return {
        "dataset": dataset, "forecast_steps": steps,
        "task_metrics": {m: {"mae": v, "rmse": v, "mase": v} for m, v in metrics.items()},
        "queries": [
            {
                "window": w, "n_rows": steps,
                "trace": {"forecast_steps": steps, "n_rows": steps, "selector_invoked": False},
                "metrics": {m: {"mae": v, "rmse": v, "mase": v} for m, v in metrics.items()},
            }
            for w in ("Q1", "Q2", "Q3")
        ],
    }


def _rule(acceptance, name, steps):
    return next(
        r for r in acceptance["rules"]
        if r["rule"] == name and r.get("forecast_steps") == steps
    )


def test_ratio_exactly_one_fails_strict_rules_and_passes_non_strict():
    """§11.2 的比较方向必须严格区分：1a/2 是 `<`，1b/3 是 `<=`。"""
    tasks = [
        _task(ds, 24, mc=100.0, best_single=100.0, seasonal=100.0, ridge=100.0)
        for ds in ("pjm", "aemo_vic", "aemo_nsw")
    ]
    acceptance = evaluate_gates(tasks)

    assert _rule(acceptance, "11.2.1a", 24)["passed"] is False   # 1.00 < 1.00 不成立
    assert _rule(acceptance, "11.2.2", 24)["passed"] is False    # 1.00 < 1.00 不成立
    assert _rule(acceptance, "11.2.1b", 24)["passed"] is True    # 1.00 <= 1.03
    assert _rule(acceptance, "11.2.3", 24)["passed"] is True     # 1.00 <= 1.00
    assert acceptance["passed"] is False


def test_mean_ratio_is_equal_weight_over_datasets_not_over_pooled_mae():
    """三个区域负荷量级差一个数量级：先平均 MAE 再相除会让 PJM 单独决定结论。

    这里 PJM 的比值 1.02、两个 AEMO 的比值各 0.80，逐数据集等权平均 = 0.874 < 1.00
    应当通过；若改成先平均 MAE 再相除则约 1.019，会被 PJM 拖成不通过。
    """
    tasks = [
        _task("pjm", 168, mc=30600.0, best_single=30000.0, seasonal=1e9, ridge=1e9),
        _task("aemo_vic", 168, mc=320.0, best_single=400.0, seasonal=1e9, ridge=1e9),
        _task("aemo_nsw", 168, mc=400.0, best_single=500.0, seasonal=1e9, ridge=1e9),
    ]
    rule = _rule(evaluate_gates(tasks), "11.2.1a", 168)

    assert rule["value"] == pytest.approx((1.02 + 0.8 + 0.8) / 3, rel=1e-9)
    assert rule["passed"] is True
    pooled = (30600.0 + 320.0 + 400.0) / (30000.0 + 400.0 + 500.0)
    assert pooled > 1.0  # 另一种读法会得出相反结论，故必须写死一种


def test_per_dataset_rule_fails_when_a_single_dataset_exceeds_tolerance():
    tasks = [
        _task("pjm", 720, mc=104.0, best_single=100.0, seasonal=1e9, ridge=1e9),  # 1.04
        _task("aemo_vic", 720, mc=80.0, best_single=100.0, seasonal=1e9, ridge=1e9),
        _task("aemo_nsw", 720, mc=80.0, best_single=100.0, seasonal=1e9, ridge=1e9),
    ]
    acceptance = evaluate_gates(tasks)

    assert _rule(acceptance, "11.2.1a", 720)["passed"] is True  # 均值 0.9067 < 1.00
    rule = _rule(acceptance, "11.2.1b", 720)
    assert rule["passed"] is False and rule["worst_dataset"] == "pjm"


def test_seasonal_naive_rule_is_judged_per_task_not_on_average():
    """§11.2 第 2 条是 9 个任务逐个判定，一个任务输了就不通过。"""
    tasks = [
        _task("pjm", 24, mc=50.0, best_single=1e9, seasonal=100.0, ridge=1e9),
        _task("aemo_vic", 24, mc=50.0, best_single=1e9, seasonal=100.0, ridge=1e9),
        _task("aemo_nsw", 24, mc=101.0, best_single=1e9, seasonal=100.0, ridge=1e9),
    ]
    rule = _rule(evaluate_gates(tasks), "11.2.2", 24)

    assert rule["passed"] is False
    assert rule["per_dataset"]["aemo_nsw"] == pytest.approx(1.01)
    assert float(np.mean(list(rule["per_dataset"].values()))) < 1.0  # 平均会误判为通过


def test_functional_gate_catches_wrong_row_count_or_invoked_selector():
    task = _task("pjm", 24, mc=1.0, best_single=2.0, seasonal=2.0, ridge=2.0)
    task["queries"][0]["trace"]["selector_invoked"] = True
    task["queries"][1]["n_rows"] = 23
    acceptance = evaluate_gates([task])

    functional = [r for r in acceptance["rules"] if r["rule"] == "11.1.2/11.1.3"]
    assert [r["passed"] for r in functional] == [False, False, True]
    assert acceptance["passed"] is False
