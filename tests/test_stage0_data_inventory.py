"""Stage 0 数据覆盖盘点（scripts/stage0_data_inventory.py）。

这份盘点的数字会被用来一次性冻结全部预测起点，所以它算错就等于整批实验口径
算错。这里守住三件事：缺口/重复能被如实识别；容量判定用的是**连续段**而不是
总跨度；装不下时非零退出而不是悄悄缩窗口。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_raw(root: Path, dataset: str, timestamps: pd.DatetimeIndex) -> None:
    (root / dataset).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"timestamp": timestamps, "load": np.arange(len(timestamps), dtype=float)}
    ).to_csv(root / dataset / "load.csv", index=False)


def _run(tmp_path: Path, root: Path, *, datasets, steps, trajectories=7, window=720):
    out = tmp_path / "inventory.json"
    proc = subprocess.run(
        [
            sys.executable, "scripts/stage0_data_inventory.py",
            "--raw-root", str(root), "--layout", "raw",
            "--datasets", *datasets,
            "--forecast-steps", *[str(s) for s in steps],
            "--trajectories", str(trajectories),
            "--signature-window", str(window),
            "--out", str(out),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    report = json.loads(out.read_text()) if out.exists() else None
    return proc, report


def test_reports_gaps_duplicates_and_uses_longest_contiguous_run(tmp_path):
    root = tmp_path / "data"
    # 总跨度足够，但中间被挖掉一段 -> 连续段不足，容量判定必须看连续段
    head = pd.date_range("2025-01-01", periods=900, freq="h")
    tail = pd.date_range(head[-1] + pd.Timedelta(hours=50), periods=900, freq="h")
    stamps = head.append(tail).append(pd.DatetimeIndex([tail[-1]]))  # 末尾一个重复
    _write_raw(root, "gappy", stamps)

    proc, report = _run(tmp_path, root, datasets=["gappy"], steps=[24], trajectories=7, window=720)

    entry = report["datasets"][0]
    assert entry["union"]["duplicate_timestamps"] == 1
    assert entry["union"]["gap_count"] == 1
    assert entry["union"]["largest_gap_hours"] == 50.0
    assert entry["union"]["longest_contiguous_run"]["hours"] == 900
    # 总跨度 1850 小时 > 需要的 888，但最长连续段 900 也够 -> 可行
    assert entry["feasibility"]["24"]["fits"] is True
    assert proc.returncode == 0


def test_insufficient_contiguous_coverage_exits_nonzero(tmp_path):
    root = tmp_path / "data"
    _write_raw(root, "short", pd.date_range("2025-01-01", periods=4345, freq="h"))

    proc, report = _run(tmp_path, root, datasets=["short"], steps=[168, 720])

    entry = report["datasets"][0]
    assert entry["feasibility"]["168"]["fits"] is True
    assert entry["feasibility"]["720"]["fits"] is False
    assert entry["feasibility"]["720"]["shortfall_hours"] == 720 + 7 * 720 - 4345
    assert report["all_feasible"] is False
    assert proc.returncode != 0, "装不下时必须停下来报告，不能悄悄缩窗口"


def test_proposed_origins_are_non_overlapping_and_have_full_history(tmp_path):
    root = tmp_path / "data"
    _write_raw(root, "ample", pd.date_range("2025-01-01", periods=6800, freq="h"))

    proc, report = _run(tmp_path, root, datasets=["ample"], steps=[720])
    assert proc.returncode == 0, proc.stdout + proc.stderr

    origins = report["datasets"][0]["feasibility"]["720"]["origins"]
    assert [o["label"] for o in origins] == ["H1", "H2", "H3", "A", "Q1", "Q2", "Q3"]
    # H1—H3 建库、A 是共用的冻结后审计窗口、Q1—Q3 未见查询（§6.3）
    assert [o["role"] for o in origins] == (
        ["library"] * 3 + ["audit"] + ["query"] * 3
    )

    run_start = pd.Timestamp(report["datasets"][0]["union"]["longest_contiguous_run"]["start"])
    previous_last_target = None
    for origin in origins:
        first = pd.Timestamp(origin["first_target"])
        last = pd.Timestamp(origin["last_target"])
        assert (last - first) == pd.Timedelta(hours=719)
        # 每个起点前都要有完整 720 小时真实历史
        assert pd.Timestamp(origin["history_start"]) >= run_start
        assert (pd.Timestamp(origin["forecast_origin"])
                - pd.Timestamp(origin["history_start"])) == pd.Timedelta(hours=719)
        # 目标区间互不重叠
        if previous_last_target is not None:
            assert first > previous_last_target
        previous_last_target = last
