"""PJM hour-ending EPT → 连续固定 EST/UTC-5 的合成契约测试。

仅使用受控合成装置（原始格式 ``Datetime,PJME_MW``），不读取任何机器本地的
真实 ``data/pjm/load.csv``；真实数据的 40199 行验收属于转换质量门，不在 pytest。

覆盖：普通小时、春季跳时闭合、秋季重复小时分别保留且不平均、稳定排序保留重复
行原始顺序、严格逐小时连续、非 DST 缺口失败、CLI 输出与原始字节不变。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_pjm_hourly import _load_input, prepare, transform

ROOT = Path(__file__).resolve().parents[1]
RAW_HEADER = "Datetime,PJME_MW"
HOUR = pd.Timedelta(hours=1)


def _frame(labels, loads):
    return pd.DataFrame(
        {"timestamp": pd.to_datetime(labels), "load": [float(v) for v in loads]}
    )


def _write_raw_csv(path: Path, labels, loads) -> None:
    lines = [RAW_HEADER]
    for label, value in zip(labels, loads):
        lines.append(f"{label},{float(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_ordinary_hour_conversion():
    frame = _frame(["2014-06-01 12:00:00"], [100.0])
    out = transform(frame)
    assert len(out) == 1
    assert out["timestamp"].iloc[0] == pd.Timestamp("2014-06-01 11:00:00")
    assert out["load"].iloc[0] == 100.0


def test_spring_forward_becomes_continuous():
    frame = _frame(
        [
            "2014-03-09 00:00:00",
            "2014-03-09 01:00:00",
            "2014-03-09 02:00:00",
            "2014-03-09 04:00:00",
            "2014-03-09 05:00:00",
        ],
        [1, 2, 3, 4, 5],
    )
    out = transform(frame)
    assert out["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist() == [
        "2014-03-09 00:00",
        "2014-03-09 01:00",
        "2014-03-09 02:00",
        "2014-03-09 03:00",
        "2014-03-09 04:00",
    ]
    assert (out["timestamp"].diff().dropna() == HOUR).all()


def test_fall_back_duplicate_hours_kept_separate():
    frame = _frame(
        [
            "2014-11-02 00:00:00",
            "2014-11-02 01:00:00",
            "2014-11-02 02:00:00",
            "2014-11-02 02:00:00",
            "2014-11-02 03:00:00",
        ],
        [10, 20, 100, 200, 30],
    )
    out = transform(frame)
    assert len(out) == 5
    assert out["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist() == [
        "2014-11-01 23:00",
        "2014-11-02 00:00",
        "2014-11-02 01:00",
        "2014-11-02 02:00",
        "2014-11-02 03:00",
    ]
    assert out["timestamp"].nunique() == 5


def test_duplicate_hour_loads_are_not_averaged_and_keep_order():
    frame = _frame(
        ["2014-11-02 01:00:00", "2014-11-02 02:00:00", "2014-11-02 02:00:00"],
        [20, 100, 200],
    )
    out = transform(frame)
    # 第 1 个重复行=DST → EST 01:00；第 2 个重复行=STANDARD → EST 02:00
    assert out["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist() == [
        "2014-11-02 00:00",
        "2014-11-02 01:00",
        "2014-11-02 02:00",
    ]
    assert out["load"].tolist() == [20.0, 100.0, 200.0]
    assert 150.0 not in out["load"].tolist()


def test_stable_sort_preserves_duplicate_row_order(tmp_path):
    src = tmp_path / "PJME_hourly.csv"
    src.write_text(
        RAW_HEADER + "\n"
        "2014-11-02 02:00:00,100.0\n"
        "2014-11-02 02:00:00,200.0\n"
        "2014-11-02 01:00:00,50.0\n"
        "2014-11-02 03:00:00,300.0\n",
        encoding="utf-8",
    )
    frame = _load_input(src)
    assert frame["timestamp"].is_monotonic_increasing
    dup = frame[frame["timestamp"] == pd.Timestamp("2014-11-02 02:00:00")]
    assert dup["load"].tolist() == [100.0, 200.0]


def test_output_is_strictly_hourly_continuous():
    frame = _frame(
        pd.date_range("2014-01-01 00:00", "2014-01-02 00:00", freq="h").strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        range(25),
    )
    out = transform(frame)
    assert len(out) == 25
    assert (out["timestamp"].diff().dropna() == HOUR).all()
    assert int(out["timestamp"].duplicated().sum()) == 0


def test_non_dst_gap_fails(tmp_path):
    src = tmp_path / "PJME_hourly.csv"
    _write_raw_csv(src, ["2014-06-01 01:00:00", "2014-06-01 03:00:00"], [1, 2])
    with pytest.raises(ValueError) as excinfo:
        prepare(input_path=src, start="2014-01-01", out_root=tmp_path / "out")
    assert "非 DST 缺口" in str(excinfo.value)


def test_cli_outputs_and_raw_bytes_unchanged(tmp_path):
    src = tmp_path / "PJME_hourly.csv"
    out_root = tmp_path / "out"
    labels = pd.date_range("2014-06-01 00:00", "2014-06-01 05:00", freq="h").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    _write_raw_csv(src, labels, range(6))
    before = src.read_bytes()

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.prepare_pjm_hourly",
            "--input",
            str(src),
            "--start",
            "2014-01-01",
            "--out-root",
            str(out_root),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert src.read_bytes() == before

    load_path = out_root / "pjm" / "load.csv"
    report_path = out_root / "pjm_hourly_quality_report.json"
    assert load_path.exists() and report_path.exists()

    out = pd.read_csv(load_path)
    assert list(out.columns) == ["timestamp", "load"]
    assert len(out) == 6

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["timestamp_semantics"] == "interval_end"
    assert report["output_timezone"] == "fixed_UTC-5_EST"
    assert report["input"]["rows"] == 6
    assert report["output"]["rows"] == 6
    assert report["output"]["gaps"] == 0
    assert report["output"]["duplicate_timestamps"] == 0
    assert report["load_multiset_preserved"] is True
    assert report["load_sequence_preserved"] is True
