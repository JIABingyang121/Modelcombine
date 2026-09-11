"""PJM hour-ending EPT → 连续固定 EST/UTC-5 的契约测试。

覆盖：普通小时、春季跳时闭合、秋季重复小时分别保留且不平均、严格逐小时连续、
非 DST 缺口失败、原始 CSV 字节不变、真实 2014-2018 数据 40199 行且无缺口。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_pjm_hourly import prepare, transform

ROOT = Path(__file__).resolve().parents[1]
HEADER = "timestamp,load,region,region_type"
REAL_INPUT = ROOT / "data" / "pjm" / "load.csv"
HOUR = pd.Timedelta(hours=1)


def _frame(labels, loads):
    return pd.DataFrame(
        {"timestamp": pd.to_datetime(labels), "load": [float(v) for v in loads]}
    )


def _write_csv(path: Path, labels, loads) -> None:
    lines = [HEADER]
    for label, value in zip(labels, loads):
        lines.append(f"{label},{float(value)},PJME,real_grid")
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


def test_duplicate_hour_loads_are_not_averaged():
    frame = _frame(
        ["2014-11-02 01:00:00", "2014-11-02 02:00:00", "2014-11-02 02:00:00"],
        [20, 100, 200],
    )
    out = transform(frame)
    loads = sorted(out["load"].tolist())
    assert loads == [20.0, 100.0, 200.0]
    assert 150.0 not in loads


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
    src = tmp_path / "load.csv"
    _write_csv(src, ["2014-06-01 01:00:00", "2014-06-01 03:00:00"], [1, 2])
    with pytest.raises(ValueError) as excinfo:
        prepare(input_path=src, start="2014-01-01", out_root=tmp_path / "out")
    assert "非 DST 缺口" in str(excinfo.value)


def test_cli_outputs_and_raw_bytes_unchanged(tmp_path):
    src = tmp_path / "load.csv"
    out_root = tmp_path / "out"
    labels = pd.date_range("2014-06-01 00:00", "2014-06-01 05:00", freq="h").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    _write_csv(src, labels, range(6))
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
    assert report["load_one_to_one"] is True


@pytest.mark.skipif(not REAL_INPUT.exists(), reason="本机缺少 data/pjm/load.csv")
def test_real_2014_2018_data_is_40199_rows_without_gaps(tmp_path):
    report, _ = prepare(input_path=REAL_INPUT, start="2014-01-01", out_root=tmp_path)
    assert report["input"]["rows"] == 40199
    assert report["input"]["duplicate_labels"] == 4
    assert report["input"]["dst_expanded_labels"] == 4
    assert report["input"]["non_dst_gaps"] == []
    assert report["output"]["rows"] == 40199
    assert report["output"]["first_timestamp"] == "2014-01-01 01:00:00"
    assert report["output"]["last_timestamp"] == "2018-08-02 23:00:00"
    assert report["output"]["gaps"] == 0
    assert report["output"]["duplicate_timestamps"] == 0
    assert report["load_one_to_one"] is True
