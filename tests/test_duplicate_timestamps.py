"""重复时间戳的统一处理（TIMESTAMP_POLICY = mean_load_by_timestamp）。

PJM 秋令时回拨会产生真实的重复小时（服务器盘点实测 2 个）。重复行会让窗口切片多出一行
——目标区间变成 steps+1 个点、lag/rolling 错位——而建库、冻结、正式运行三处若各按各的方式
处理，同一段历史会切出不同的序列。

这里的关键装置是把重复点放在**同时属于 S3 目标区间与 A 输入历史**的位置：窗口按 stride
排布时 S3 的目标区间恰好落在 A 的 720 小时历史里，所以一个重复点会同时影响建库的验证轨迹
和审计窗口的输入。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.train_combinations_kg import (
    TIMESTAMP_POLICY,
    _library_raw_frame,
    _library_raw_timestamps,
)
from tests.forecast_steps_fixtures import (
    DATASET,
    FIXTURE_CANDIDATES,
    REPO_ROOT,
    seed_models,
    write_dataset,
    write_frozen_window_plan,
)

ROWS = 1000
STEPS = 24


def _duplicate_inside_s3_target_and_a_history(raw_root: Path, plan_path: Path):
    """复制一个既在 S3 目标区间、又在 A 输入历史里的时刻，模拟 DST 回拨。"""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    origins = {o["label"]: o for o in plan["datasets"][0]["origins"]}
    s3 = origins["S3"]["targets"][str(STEPS)]
    target = pd.Timestamp(s3["first_target"]) + pd.Timedelta(hours=1)

    a_start = pd.Timestamp(origins["A"]["history_start"])
    a_end = pd.Timestamp(origins["A"]["history_end"])
    assert a_start <= target <= a_end, "装置无效：该时刻不在 A 的输入历史里"
    assert pd.Timestamp(s3["first_target"]) <= target <= pd.Timestamp(s3["last_target"])

    path = raw_root / DATASET / "load.csv"
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    row = frame[frame["timestamp"] == target]
    assert len(row) == 1
    original = float(row["load"].iloc[0])
    twin = row.copy()
    twin["load"] = original + 100.0
    pd.concat([frame, twin], ignore_index=True).sort_values("timestamp").to_csv(
        path, index=False
    )
    return target, original, original + 100.0


@pytest.fixture
def duplicated(tmp_path):
    raw_root = tmp_path / "raw"
    db = tmp_path / "lib.sqlite3"
    frames = write_dataset(tmp_path / "splits", rows=ROWS)
    seed_models(db, tmp_path / "artifacts", frames["train"])
    plan = write_frozen_window_plan(raw_root, frames, forecast_steps=STEPS)
    target, a, b = _duplicate_inside_s3_target_and_a_history(raw_root, plan)
    return {"raw_root": raw_root, "db": db, "plan": plan, "tmp": tmp_path,
            "target": target, "values": (a, b)}


# ------------------------------------------------------------------ 读取层
def test_duplicate_load_is_averaged_and_series_becomes_unique(duplicated):
    frame = _library_raw_frame(duplicated["raw_root"], DATASET)
    a, b = duplicated["values"]

    assert not frame["timestamp"].duplicated().any(), "序列必须唯一"
    hit = frame.loc[frame["timestamp"] == duplicated["target"], "load"]
    assert len(hit) == 1
    assert hit.iloc[0] == pytest.approx((a + b) / 2)


def test_timestamps_reader_returns_sorted_unique_and_reads_no_load(duplicated):
    stamps = _library_raw_timestamps(duplicated["raw_root"], DATASET)

    assert not stamps.duplicated().any()
    assert stamps.is_monotonic_increasing
    # 只读时间列：把负荷列写成非数字也不该影响它
    path = duplicated["raw_root"] / DATASET / "load.csv"
    frame = pd.read_csv(path)
    frame["load"] = "不是数字"
    frame.to_csv(path, index=False)
    assert len(_library_raw_timestamps(duplicated["raw_root"], DATASET)) == len(stamps)


def test_original_csv_is_not_modified(duplicated):
    path = duplicated["raw_root"] / DATASET / "load.csv"
    before = path.read_bytes()
    _library_raw_frame(duplicated["raw_root"], DATASET)
    _library_raw_timestamps(duplicated["raw_root"], DATASET)
    assert path.read_bytes() == before, "规范化只发生在读取时，不得回写原始 CSV"


# -------------------------------------------------- 建库：S3 目标 / A 历史回归
def test_build_survives_a_duplicate_spanning_s3_target_and_a_history(duplicated):
    """重复点同时落在 S3 目标区间与 A 输入历史里，建库必须照常完成。"""
    out = duplicated["tmp"] / "out"
    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_combinations_kg", "--model-library",
            "--datasets", DATASET, "--forecast-steps", str(STEPS),
            "--candidates", *FIXTURE_CANDIDATES,
            "--raw-root", str(duplicated["raw_root"]),
            "--window-plan", str(duplicated["plan"]),
            "--out-root", str(out), "--database", str(duplicated["db"]),
            "--model-artifacts", str(duplicated["tmp"] / "combo_artifacts"),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((out / "model_library_report.json").read_text())
    assert report["timestamp_policy"] == TIMESTAMP_POLICY
    assert len(report["tasks"]) == 3
    for task in report["tasks"]:
        assert len(task["val_trajectory"]) == STEPS, "目标区间不得因重复点多出一行"
        assert len(task["test_trajectory"]) == STEPS


def test_freeze_records_the_same_policy(duplicated):
    from scripts.freeze_final_experiment import _dataset_definition

    plan = json.loads(duplicated["plan"].read_text(encoding="utf-8"))
    plan["datasets"][0]["fits"] = True
    entry = _dataset_definition(duplicated["raw_root"], plan, DATASET)

    raw_rows = len(pd.read_csv(duplicated["raw_root"] / DATASET / "load.csv"))
    assert entry["rows"] == raw_rows - 1, "冻结按唯一时间戳计数"
    assert [w["label"] for w in entry["windows"]] == [
        "S1", "S2", "S3", "A", "T1", "T2", "T3"], "窗口日期不因去重改变"
