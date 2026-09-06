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


def _run(tmp_path: Path, root: Path, *, datasets, steps, trajectories=7, window=720,
         out_name="inventory.json"):
    out = tmp_path / out_name
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
    # 总跨度远超需求，但中间被挖掉一段 -> 容量判定必须看**连续段**而不是总跨度
    head = pd.date_range("2025-01-01", periods=5000, freq="h")
    tail = pd.date_range(head[-1] + pd.Timedelta(hours=50), periods=2000, freq="h")
    stamps = head.append(tail).append(pd.DatetimeIndex([tail[-1]]))  # 末尾一个重复
    _write_raw(root, "gappy", stamps)

    proc, report = _run(tmp_path, root, datasets=["gappy"], steps=[24], trajectories=7, window=720)

    entry = report["datasets"][0]
    assert entry["union"]["duplicate_timestamps"] == 1
    assert entry["union"]["gap_count"] == 1
    assert entry["union"]["largest_gap_hours"] == 50.0
    assert entry["union"]["longest_contiguous_run"]["hours"] == 5000
    # 总跨度 7049 小时 > 需要的 5760，但最长连续段只有 5000 -> 不可行
    assert entry["union"]["span_hours"] > entry["required_contiguous_hours"]
    assert entry["fits"] is False
    assert entry["shortfall_hours"] == 720 + 7 * 720 - 5000
    assert proc.returncode != 0


def test_insufficient_contiguous_coverage_exits_nonzero(tmp_path):
    root = tmp_path / "data"
    _write_raw(root, "short", pd.date_range("2025-01-01", periods=4345, freq="h"))

    proc, report = _run(tmp_path, root, datasets=["short"], steps=[168, 720])

    entry = report["datasets"][0]
    assert entry["fits"] is False
    assert entry["shortfall_hours"] == 720 + 7 * 720 - 4345
    assert report["all_feasible"] is False
    assert proc.returncode != 0, "装不下时必须停下来报告，不能悄悄缩窗口"


def test_one_origin_serves_all_three_forecast_horizons(tmp_path):
    """同一个 S/T 起点同时产生 H1=24、H2=168、H3=720。

    三种长度必须使用**完全相同**的 history_start / history_end / forecast_origin；
    短长度的目标区间是长长度的前缀（H1 = 未来 720 小时里的前 24 小时）。
    窗口按最长长度 720 排布，因此 720 步目标互不重叠。
    """
    root = tmp_path / "data"
    _write_raw(root, "ample", pd.date_range("2025-01-01", periods=6800, freq="h"))

    proc, report = _run(tmp_path, root, datasets=["ample"], steps=[24, 168, 720])
    assert proc.returncode == 0, proc.stdout + proc.stderr

    entry = report["datasets"][0]
    # 容量统一按最长长度算，与请求哪几种长度无关
    assert entry["required_contiguous_hours"] == 720 + 7 * 720
    assert report["forecast_horizons"] == {"H1": 24, "H2": 168, "H3": 720}

    origins = entry["origins"]
    assert [o["label"] for o in origins] == ["S1", "S2", "S3", "A", "T1", "T2", "T3"]
    assert [o["role"] for o in origins] == ["library"] * 3 + ["audit"] + ["test"] * 3

    run_start = pd.Timestamp(entry["union"]["longest_contiguous_run"]["start"])
    previous_last = None
    for origin in origins:
        history_start = pd.Timestamp(origin["history_start"])
        history_end = pd.Timestamp(origin["history_end"])
        forecast_origin = pd.Timestamp(origin["forecast_origin"])
        assert history_start >= run_start
        # 输入历史恰好 720 小时，且以预测起点结尾
        assert history_end == forecast_origin
        assert (forecast_origin - history_start) == pd.Timedelta(hours=719)

        targets = origin["targets"]
        assert sorted(int(k) for k in targets) == [24, 168, 720]
        # 三种长度共享同一个起点：第一个目标时刻必然相同
        assert {pd.Timestamp(v["first_target"]) for v in targets.values()} == {
            forecast_origin + pd.Timedelta(hours=1)
        }
        for steps, window in targets.items():
            assert window["forecast_steps"] == int(steps)
            span = pd.Timestamp(window["last_target"]) - pd.Timestamp(window["first_target"])
            assert span == pd.Timedelta(hours=int(steps) - 1)
        # 短长度是长长度的真前缀
        assert (pd.Timestamp(targets["24"]["last_target"])
                < pd.Timestamp(targets["168"]["last_target"])
                < pd.Timestamp(targets["720"]["last_target"]))

        # 相邻窗口按最长长度隔开，720 步目标互不重叠
        if previous_last is not None:
            assert pd.Timestamp(targets["720"]["first_target"]) > previous_last
        previous_last = pd.Timestamp(targets["720"]["last_target"])


def test_requested_steps_do_not_change_origins(tmp_path):
    """只请求 24 步也必须给出同一套共享起点和 5760 小时容量判据。"""
    root = tmp_path / "data"
    _write_raw(root, "ample", pd.date_range("2025-01-01", periods=6800, freq="h"))

    _proc_all, report_all = _run(tmp_path, root, datasets=["ample"], steps=[24, 168, 720])
    _proc_one, report_one = _run(
        tmp_path, root, datasets=["ample"], steps=[24], out_name="one.json"
    )

    a = report_all["datasets"][0]
    b = report_one["datasets"][0]
    assert a["required_contiguous_hours"] == b["required_contiguous_hours"] == 720 + 7 * 720
    assert [o["forecast_origin"] for o in a["origins"]] == [
        o["forecast_origin"] for o in b["origins"]
    ]
