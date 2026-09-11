"""AEMO 原始五分钟数据 → 统一小时级负荷的契约测试。

覆盖：聚合语义、右闭右标签、跨月/跨年接缝、精确重复与冲突、缺文件/缺列/区域/
PERIODTYPE 校验、不完整小时失败、原始文件字节不变、双区域 CLI 输出，以及旧下载
入口与唯一实现共用同一聚合函数。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.prepare_aemo_hourly import (
    SOURCE_ROWS_PER_HOUR,
    _read_month_file,
    aggregate_hourly,
    discover_month_files,
)

ROOT = Path(__file__).resolve().parents[1]
HEADER = "REGION,SETTLEMENTDATE,TOTALDEMAND,RRP,PERIODTYPE"


def _ts_str(value) -> str:
    return pd.Timestamp(value).strftime("%Y/%m/%d %H:%M:%S")


def _raw(region, timestamps, values, *, source_file=None, periodtype="TRADE", rrp="1.00"):
    frame = pd.DataFrame(
        {
            "REGION": region,
            "SETTLEMENTDATE": [_ts_str(t) for t in timestamps],
            "TOTALDEMAND": [str(v) for v in values],
            "RRP": rrp,
            "PERIODTYPE": periodtype,
        }
    )
    if source_file is not None:
        frame["source_file"] = source_file
    return frame


def _hour_timestamps(start: str):
    return pd.date_range(start, periods=SOURCE_ROWS_PER_HOUR, freq="5min")


def test_normal_hour_is_mean_of_twelve_five_minute_values():
    ts = _hour_timestamps("2022-03-01 00:05")
    frame = _raw("VIC1", ts, list(range(SOURCE_ROWS_PER_HOUR)))
    hourly, stats = aggregate_hourly(frame, region="VIC1")
    assert len(hourly) == 1
    assert hourly["timestamp"].iloc[0] == pd.Timestamp("2022-03-01 01:00:00")
    assert hourly["load"].iloc[0] == pytest.approx(sum(range(12)) / 12)
    assert stats["source_rows_per_hour_distribution"] == {"12": 1}


def test_hour_label_0100_uses_0005_to_0100_not_0105():
    ts = pd.date_range("2022-03-01 00:05", "2022-03-01 02:00", freq="5min")
    frame = _raw("VIC1", ts, [float(i) for i in range(len(ts))])
    hourly, _ = aggregate_hourly(frame, region="VIC1")
    assert list(hourly["timestamp"]) == [
        pd.Timestamp("2022-03-01 01:00:00"),
        pd.Timestamp("2022-03-01 02:00:00"),
    ]
    assert hourly["load"].iloc[0] == pytest.approx(sum(range(12)) / 12)
    assert hourly["load"].iloc[1] == pytest.approx(sum(range(12, 24)) / 12)


def test_0000_interval_belongs_to_previous_hour():
    ts = pd.date_range("2021-12-31 23:05", "2022-01-01 00:00", freq="5min")
    frame = _raw("VIC1", ts, [1.0] * len(ts))
    hourly, stats = aggregate_hourly(frame, region="VIC1")
    assert len(hourly) == 1
    assert hourly["timestamp"].iloc[0] == pd.Timestamp("2022-01-01 00:00:00")
    assert stats["hourly_rows_per_year"] == {2021: 1}


def test_month_seam_is_continuous_and_full():
    jan = pd.date_range("2022-01-31 23:05", "2022-02-01 00:00", freq="5min")
    feb = pd.date_range("2022-02-01 00:05", "2022-02-01 01:00", freq="5min")
    frame = pd.concat(
        [_raw("VIC1", jan, [1.0] * 12), _raw("VIC1", feb, [2.0] * 12)],
        ignore_index=True,
    )
    hourly, stats = aggregate_hourly(frame, region="VIC1")
    assert list(hourly["timestamp"]) == [
        pd.Timestamp("2022-02-01 00:00:00"),
        pd.Timestamp("2022-02-01 01:00:00"),
    ]
    assert stats["source_rows_per_hour_distribution"] == {"12": 2}
    assert stats["output_gaps"] == 0


def test_year_seam_produces_2026_01_01_0000_label():
    ts = pd.date_range("2025-01-01 00:05", "2026-01-01 00:00", freq="5min")
    frame = _raw("VIC1", ts, [1.0] * len(ts))
    hourly, stats = aggregate_hourly(frame, region="VIC1")
    assert hourly["timestamp"].iloc[-1] == pd.Timestamp("2026-01-01 00:00:00")
    assert stats["hourly_rows"] == 8760
    assert stats["hourly_rows_per_year"] == {2025: 8760}
    assert stats["output_gaps"] == 0


def test_exact_cross_file_duplicate_kept_once():
    ts = _hour_timestamps("2022-03-01 00:05")
    base = _raw("VIC1", ts, list(range(12)), source_file="a.csv")
    twin = base.iloc[[3]].copy()
    twin["source_file"] = "b.csv"
    frame = pd.concat([base, twin], ignore_index=True)
    hourly, stats = aggregate_hourly(frame, region="VIC1")
    assert stats["raw_duplicate_keys"] == 1
    assert stats["exact_duplicate_rows_dropped"] == 1
    assert stats["conflict_duplicate_keys"] == 0
    assert len(hourly) == 1
    assert stats["source_rows_per_hour_distribution"] == {"12": 1}


def test_conflicting_duplicate_fails_immediately():
    ts = _hour_timestamps("2022-03-01 00:05")
    base = _raw("VIC1", ts, list(range(12)), source_file="a.csv")
    twin = base.iloc[[3]].copy()
    twin["source_file"] = "b.csv"
    twin["TOTALDEMAND"] = "9999"
    frame = pd.concat([base, twin], ignore_index=True)
    with pytest.raises(ValueError) as excinfo:
        aggregate_hourly(frame, region="VIC1")
    message = str(excinfo.value)
    assert "冲突" in message
    assert "a.csv" in message and "b.csv" in message
    assert "TOTALDEMAND" in message


def test_missing_month_file_fails(tmp_path):
    region_dir = tmp_path / "VIC1"
    region_dir.mkdir()
    for month in range(1, 12):  # 缺 12 月
        (region_dir / f"PRICE_AND_DEMAND_2022{month:02d}_VIC1.csv").write_text(
            HEADER + "\n", encoding="utf-8"
        )
    with pytest.raises(FileNotFoundError) as excinfo:
        discover_month_files(tmp_path, "VIC1", [2022])
    assert "2022-12" in str(excinfo.value)


def test_missing_required_column_fails(tmp_path):
    path = tmp_path / "PRICE_AND_DEMAND_202201_VIC1.csv"
    path.write_text(
        "REGION,SETTLEMENTDATE,TOTALDEMAND,PERIODTYPE\n"
        "VIC1,2022/01/01 00:05:00,1.0,TRADE\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        _read_month_file(path, "VIC1")
    assert "RRP" in str(excinfo.value)


def test_region_mismatch_fails(tmp_path):
    path = tmp_path / "PRICE_AND_DEMAND_202201_VIC1.csv"
    path.write_text(
        HEADER + "\nNSW1,2022/01/01 00:05:00,1.0,1.0,TRADE\n", encoding="utf-8"
    )
    with pytest.raises(ValueError) as excinfo:
        _read_month_file(path, "VIC1")
    assert "REGION" in str(excinfo.value)


def test_non_trade_periodtype_fails(tmp_path):
    path = tmp_path / "PRICE_AND_DEMAND_202201_VIC1.csv"
    path.write_text(
        HEADER + "\nVIC1,2022/01/01 00:05:00,1.0,1.0,SETTLEMENTDAY\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        _read_month_file(path, "VIC1")
    assert "TRADE" in str(excinfo.value)


def test_incomplete_hour_fails_without_interpolation():
    ts = pd.date_range("2022-03-01 00:05", "2022-03-01 00:55", freq="5min")  # 11 个点
    frame = _raw("VIC1", ts, [1.0] * len(ts))
    with pytest.raises(ValueError) as excinfo:
        aggregate_hourly(frame, region="VIC1")
    assert str(SOURCE_ROWS_PER_HOUR) in str(excinfo.value)


def _write_month_file(path: Path, region: str, timestamps) -> None:
    lines = [HEADER]
    for i, t in enumerate(timestamps):
        lines.append(f"{region},{_ts_str(t)},{100.0 + (i % 7):.2f},1.00,TRADE")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_region_year(root: Path, region: str, year: int) -> None:
    region_dir = root / region
    region_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range(f"{year}-01-01 00:05", f"{year + 1}-01-01 00:00", freq="5min")
    # AEMO 月文件按区间结束时刻归属：2023-01-01 00:00 属于 2022-12。
    periods = (ts - pd.Timedelta(minutes=5)).to_period("M")
    for month in range(1, 13):
        selected = ts[periods == pd.Period(f"{year}-{month:02d}", freq="M")]
        _write_month_file(
            region_dir / f"PRICE_AND_DEMAND_{year}{month:02d}_{region}.csv",
            region,
            selected,
        )


@pytest.fixture(scope="module")
def double_region_cli(tmp_path_factory):
    base = tmp_path_factory.mktemp("aemo_cli")
    input_root = base / "raw"
    out_root = base / "out"
    _write_region_year(input_root, "VIC1", 2022)
    _write_region_year(input_root, "NSW1", 2022)

    before = {p: p.read_bytes() for p in sorted(input_root.rglob("*.csv"))}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.prepare_aemo_hourly",
            "--input-root",
            str(input_root),
            "--years",
            "2022",
            "--regions",
            "VIC1",
            "NSW1",
            "--out-root",
            str(out_root),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    after = {p: p.read_bytes() for p in sorted(input_root.rglob("*.csv"))}
    return SimpleNamespace(
        base=base,
        input_root=input_root,
        out_root=out_root,
        proc=proc,
        before=before,
        after=after,
    )


def test_raw_csv_bytes_unchanged_after_cli(double_region_cli):
    run = double_region_cli
    assert run.proc.returncode == 0, run.proc.stderr
    assert run.before == run.after


def test_cli_writes_two_load_csv_and_report(double_region_cli):
    run = double_region_cli
    assert run.proc.returncode == 0, run.proc.stderr

    vic = run.out_root / "aemo_vic" / "load.csv"
    nsw = run.out_root / "aemo_nsw" / "load.csv"
    report_path = run.out_root / "aemo_hourly_quality_report.json"
    assert vic.exists() and nsw.exists() and report_path.exists()

    vic_df = pd.read_csv(vic)
    assert list(vic_df.columns) == ["timestamp", "load"]
    assert len(vic_df) == 8760
    assert vic_df["timestamp"].iloc[0] == "2022-01-01 01:00:00"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["region_mapping"] == {"VIC1": "aemo_vic", "NSW1": "aemo_nsw"}
    assert report["time_semantics"] == {
        "timezone_semantics": "fixed_UTC+10_AEST",
        "timestamp_semantics": "interval_end",
        "hourly_rule": "right_closed_right_labeled",
        "duplicate_policy": "drop_exact_duplicates_fail_on_conflict",
    }
    for region in ("VIC1", "NSW1"):
        meta = report["regions"][region]
        assert meta["hourly_rows"] == 8760
        assert meta["source_rows_per_hour_distribution"] == {"12": 8760}
        assert meta["output_duplicate_timestamps"] == 0
        assert meta["output_gaps"] == 0
        assert meta["incomplete_hours"] == []
        assert meta["raw_duplicate_keys"] == 0
        assert meta["non_numeric_or_non_finite"] == 0


def test_legacy_download_entries_reuse_single_aggregation():
    import scripts.download_aemo as download_vic
    import scripts.download_aemo_nsw as download_nsw
    from scripts.prepare_aemo_hourly import aggregate_hourly as shared

    assert download_vic.aggregate_hourly is shared
    assert download_nsw.aggregate_hourly is shared
    for name in ("download_aemo.py", "download_aemo_nsw.py"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "resample(" not in text
        assert "prepare_aemo_hourly" in text
