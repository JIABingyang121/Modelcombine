"""最终对比实验统一入口（scripts/final_comparison.py）。

要守的是**契约**，不是某个方法的精度：所有方法都用同一份输入历史与同一批目标时间戳，
输出同一个长表 schema；任何方法产不出完整轨迹就立即停止，不静默截断、补齐或换算法。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.final_comparison import (
    OUTPUT_COLUMNS,
    FinalComparisonError,
    Request,
    available_methods,
    run_request,
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
ROWS = 1800


def _build(tmp_path: Path) -> dict:
    """真实建库（S1—S3 + A）+ 冻结窗口计划，T1—T3 供本入口使用。"""
    raw_root = tmp_path / "raw"
    db = tmp_path / "lib.sqlite3"
    frames = write_dataset(tmp_path / "splits", rows=ROWS)
    seed_models(db, tmp_path / "artifacts", frames["train"])
    window_plan = write_frozen_window_plan(raw_root, frames, forecast_steps=STEPS)
    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_combinations_kg", "--model-library",
            "--datasets", DATASET, "--forecast-steps", str(STEPS),
            "--candidates", *FIXTURE_CANDIDATES,
            "--raw-root", str(raw_root), "--window-plan", str(window_plan),
            "--out-root", str(tmp_path / "library"), "--database", str(db),
            "--model-artifacts", str(tmp_path / "combo_artifacts"),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return {"raw_root": raw_root, "db": db, "window_plan": window_plan}


def _run(tmp_path: Path, built: dict, *, methods, windows=("T1", "T2", "T3"), seeds=(42,)):
    out = tmp_path / "final.csv"
    proc = subprocess.run(
        [
            sys.executable, "scripts/final_comparison.py",
            "--methods", *methods, "--datasets", DATASET,
            "--windows", *windows, "--forecast-steps", str(STEPS),
            "--seeds", *[str(s) for s in seeds],
            "--raw-root", str(built["raw_root"]),
            "--window-plan", str(built["window_plan"]),
            "--database", str(built["db"]), "--out", str(out),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return proc, out


@pytest.fixture(scope="module")
def comparison(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("final")
    built = _build(tmp_path)
    proc, out = _run(
        tmp_path, built, methods=["modelcombine", "random_forest", "xgboost"]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return {"tmp_path": tmp_path, "built": built, "frame": pd.read_csv(out)}


def test_output_schema_is_fixed_and_identical_for_every_method(comparison):
    frame = comparison["frame"]
    assert list(frame.columns) == list(OUTPUT_COLUMNS)

    methods = {"modelcombine", "random_forest", "xgboost"}
    assert set(frame["method"]) == methods
    # 每个方法 × 3 个测试窗口 × 24 步
    assert len(frame) == len(methods) * 3 * STEPS
    for method in methods:
        rows = frame[frame["method"] == method]
        assert len(rows) == 3 * STEPS
        assert set(rows["test_window"]) == {"T1", "T2", "T3"}
        assert set(rows["forecast_steps"]) == {STEPS}
        assert set(rows["seed"]) == {42}
        assert rows["yhat"].notna().all() and np.isfinite(rows["yhat"]).all()


def test_all_methods_share_the_same_target_timestamps_and_truth(comparison):
    """§5：所有方法必须在完全相同的目标时间戳上被评价，y_true 也必须一致。"""
    frame = comparison["frame"]
    per_method = {
        method: rows.sort_values(["test_window", "timestamp"]).reset_index(drop=True)
        for method, rows in frame.groupby("method")
    }
    reference = per_method["modelcombine"]
    for method, rows in per_method.items():
        assert rows["timestamp"].tolist() == reference["timestamp"].tolist(), method
        np.testing.assert_allclose(
            rows["y_true"].to_numpy(dtype=float),
            reference["y_true"].to_numpy(dtype=float),
            rtol=0, atol=0, err_msg=method,
        )
    # 不同方法确实给出了不同预测，不是同一条轨迹复制三份
    assert not np.allclose(
        per_method["modelcombine"]["yhat"], per_method["random_forest"]["yhat"]
    )


def test_windows_do_not_overlap_and_follow_the_frozen_plan(comparison):
    frame = comparison["frame"]
    plan = json.loads(comparison["built"]["window_plan"].read_text())
    origins = {o["label"]: o for o in plan["datasets"][0]["origins"]}

    rows = frame[frame["method"] == "modelcombine"]
    last_end = None
    for label in ("T1", "T2", "T3"):
        window = rows[rows["test_window"] == label]
        stamps = pd.to_datetime(window["timestamp"]).sort_values()
        expected_first = pd.Timestamp(origins[label]["targets"][str(STEPS)]["first_target"])
        assert stamps.iloc[0] == expected_first
        assert len(stamps) == STEPS
        if last_end is not None:
            assert stamps.iloc[0] > last_end
        last_end = stamps.iloc[-1]


def test_seed_is_recorded_and_changes_stochastic_methods(comparison):
    """种子写进输出；随机方法换种子结果应当变化，确定性方法不受影响。"""
    tmp_path = comparison["tmp_path"] / "seeds"
    tmp_path.mkdir(exist_ok=True)
    proc, out = _run(
        tmp_path, comparison["built"], methods=["random_forest"],
        windows=("T1",), seeds=(42, 43),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    frame = pd.read_csv(out)
    assert set(frame["seed"]) == {42, 43}
    a = frame[frame["seed"] == 42]["yhat"].to_numpy(dtype=float)
    b = frame[frame["seed"] == 43]["yhat"].to_numpy(dtype=float)
    assert len(a) == len(b) == STEPS
    assert not np.allclose(a, b), "换种子后随机森林的输出应当变化，否则种子没接上"


def test_incomplete_trajectory_stops_immediately(comparison):
    """任一方法不能输出完整长度就立即停止，不截断也不补齐。"""
    from scripts import final_comparison

    built = comparison["built"]
    raw = final_comparison._library_raw_frame(built["raw_root"], DATASET)
    plan = json.loads(built["window_plan"].read_text())
    window = next(o for o in plan["datasets"][0]["origins"] if o["label"] == "T1")
    history, target = final_comparison._window_slice(
        raw,
        {**{k: v for k, v in window.items() if k != "targets"},
         **window["targets"][str(STEPS)]},
        STEPS, label="probe",
    )
    request = Request(
        dataset=DATASET, test_window="T1", forecast_steps=STEPS, seed=42,
        history=history, target_timestamps=target["timestamp"], raw=raw,
        window=window, database=built["db"],
    )

    final_comparison._ADAPTERS["_short"] = lambda req: np.ones(req.forecast_steps - 1)
    final_comparison._ADAPTERS["_nan"] = lambda req: np.full(req.forecast_steps, np.nan)
    try:
        with pytest.raises(FinalComparisonError, match="不一致"):
            run_request("_short", request)
        with pytest.raises(FinalComparisonError, match="非有限值"):
            run_request("_nan", request)
        with pytest.raises(FinalComparisonError, match="未注册"):
            run_request("nonexistent", request)
    finally:
        final_comparison._ADAPTERS.pop("_short", None)
        final_comparison._ADAPTERS.pop("_nan", None)


def test_registered_methods_cover_the_locally_implemented_ones():
    assert {"modelcombine", "random_forest", "xgboost"} <= set(available_methods())
