"""EIA-930 PJM RTO → UTC hour-ending 小时级负荷的契约测试。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_pjm_rto_hourly import load_balance_files, prepare, transform

ROOT = Path(__file__).resolve().parents[1]
REAL_INPUT = ROOT / "data" / "external" / "pjm_eia930_2019plus_20260912" / "balance"
HEADER = '"Balancing Authority","UTC Time at End of Hour","Demand (MW) (Adjusted)"'


def _write_balance(path: Path, rows) -> None:
    lines = [HEADER]
    for utc, adjusted in rows:
        lines.append(f'"PJM","{utc}",{adjusted}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _utc(y, mo, d, h, ampm):
    return f"{mo:02d}/{d:02d}/{y} {h}:00:00 {ampm}"


def test_utc_hour_ending_is_preserved(tmp_path):
    _write_balance(tmp_path / "f.csv", [(_utc(2019, 1, 1, 6, "AM"), 100.0)])
    raw, _ = load_balance_files(tmp_path)
    out = transform(raw)
    assert out["timestamp"].iloc[0] == pd.Timestamp("2019-01-01 06:00:00")
    assert out["load"].iloc[0] == 100.0


def test_missing_adjusted_fails(tmp_path):
    _write_balance(tmp_path / "f.csv", [(_utc(2019, 1, 1, 6, "AM"), "")])
    raw, _ = load_balance_files(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        transform(raw)
    assert "Adjusted" in str(excinfo.value)


def test_non_finite_adjusted_fails(tmp_path):
    _write_balance(tmp_path / "f.csv", [(_utc(2019, 1, 1, 6, "AM"), "inf")])
    raw, _ = load_balance_files(tmp_path)
    with pytest.raises(ValueError):
        transform(raw)


def test_gap_fails(tmp_path):
    _write_balance(
        tmp_path / "f.csv",
        [(_utc(2019, 1, 1, 6, "AM"), 1.0), (_utc(2019, 1, 1, 8, "AM"), 2.0)],
    )
    raw, _ = load_balance_files(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        transform(raw)
    assert "逐小时连续" in str(excinfo.value)


def test_cli_and_raw_bytes_unchanged(tmp_path):
    src_dir = tmp_path / "balance"
    src_dir.mkdir()
    rows = [
        (_utc(2019, 1, 1, 6, "AM"), 100.0),
        (_utc(2019, 1, 1, 7, "AM"), 101.0),
        (_utc(2019, 1, 1, 8, "AM"), 102.0),
    ]
    _write_balance(src_dir / "f.csv", rows)
    before = (src_dir / "f.csv").read_bytes()
    out_root = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.prepare_pjm_rto_hourly",
            "--input-dir",
            str(src_dir),
            "--out-root",
            str(out_root),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert (src_dir / "f.csv").read_bytes() == before

    load_path = out_root / "pjm_rto" / "load.csv"
    report_path = out_root / "pjm_rto_hourly_quality_report.json"
    assert load_path.exists() and report_path.exists()
    out = pd.read_csv(load_path)
    assert list(out.columns) == ["timestamp", "load"]
    assert len(out) == 3
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["timezone_semantics"] == "UTC"
    assert report["timestamp_semantics"] == "hour_ending"
    assert report["load_field"] == "Demand (MW) (Adjusted)"
    assert report["gaps"] == 0


@pytest.mark.skipif(not REAL_INPUT.exists(), reason="本机缺少 EIA-930 PJM 数据")
def test_real_pjm_rto_formal_range(tmp_path):
    report, _ = prepare(
        REAL_INPUT, tmp_path, start="2019-01-01 06:00:00", end="2020-07-29 05:00:00"
    )
    assert report["rows"] == 13800
    assert report["gaps"] == 0
    assert report["duplicate_timestamps"] == 0
    assert report["first_timestamp"] == "2019-01-01 06:00:00"
    assert report["last_timestamp"] == "2020-07-29 05:00:00"
