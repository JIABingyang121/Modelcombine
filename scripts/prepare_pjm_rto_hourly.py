"""EIA-930 PJM RTO 小时需求 → 统一小时级负荷（新数据集 ``pjm_rto``）。

口径（已批准方案）
------------------
- 时间列：``UTC Time at End of Hour``；**UTC、hour ending**，不做任何时区转换。
- 负荷列：仅 ``Demand (MW) (Adjusted)``；缺失或非有限立即失败。
- 输出严格为 ``timestamp,load``；时间戳唯一、严格逐小时连续。

注意：这是 **PJM RTO**，不是旧 PJME（PJM East）；不得拼接、不得复用旧结果。

正式入口::

    python -m scripts.prepare_pjm_rto_hourly \
        --input-dir data/external/pjm_eia930_2019plus_20260912/balance \
        --out-root <private experiment dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

UTC_END_COLUMN = "UTC Time at End of Hour"
DEMAND_COLUMN = "Demand (MW) (Adjusted)"
BALANCING_AUTHORITY_COLUMN = "Balancing Authority"
EXPECTED_BA = "PJM"
EIA_DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
TIMEZONE_SEMANTICS = "UTC"
TIMESTAMP_SEMANTICS = "hour_ending"
SOURCE_DESCRIPTION = (
    "EIA-930 Hourly Electric Grid Monitor, EIA930_BALANCE "
    "(Balancing Authority=PJM); load = Demand (MW) (Adjusted)"
)
OUTPUT_DATASET_DIR = "pjm_rto"
HOUR = pd.Timedelta(hours=1)


def _format_ts(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def load_balance_files(input_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    """读取目录内全部 EIA-930 BALANCE 文件（不做列名猜测）。"""
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"目录内没有 CSV: {input_dir}")
    frames: List[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        missing = [
            c
            for c in (
                BALANCING_AUTHORITY_COLUMN,
                UTC_END_COLUMN,
                DEMAND_COLUMN,
            )
            if c not in frame.columns
        ]
        if missing:
            raise ValueError(f"{path.name} 缺少必填列: {missing}")
        ba = sorted(set(frame[BALANCING_AUTHORITY_COLUMN].astype(str).str.strip()))
        if ba != [EXPECTED_BA]:
            raise ValueError(f"{path.name} 含非 {EXPECTED_BA} 的 BA: {ba}")
        utc = pd.to_datetime(frame[UTC_END_COLUMN], format=EIA_DATETIME_FORMAT, errors="coerce")
        if utc.isna().any():
            bad = frame.loc[utc.isna(), UTC_END_COLUMN].astype(str).head(5).tolist()
            raise ValueError(f"{path.name} UTC 时间无法解析: {bad}")
        load = pd.to_numeric(frame[DEMAND_COLUMN], errors="coerce")
        frames.append(
            pd.DataFrame(
                {"utc_end": utc, "load": load.astype(float), "source_file": path.name}
            )
        )
    return pd.concat(frames, ignore_index=True), [p.name for p in files]


def transform(
    raw: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """按 UTC hour-ending 输出 ``timestamp,load``（不做时区转换）。"""
    work = raw.sort_values("utc_end", kind="mergesort").reset_index(drop=True)
    if work["utc_end"].duplicated().any():
        dup = work.loc[work["utc_end"].duplicated(), "utc_end"].head(5).tolist()
        raise ValueError(f"UTC 时间戳重复: {dup}")
    work = work.assign(timestamp=work["utc_end"])
    if start is not None:
        work = work[work["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        work = work[work["timestamp"] <= pd.Timestamp(end)]
    work = work.reset_index(drop=True)
    if work.empty:
        raise ValueError("区间过滤后没有数据")
    values = work["load"].to_numpy(dtype="float64", na_value=np.nan)
    bad_load = ~np.isfinite(values)
    if bad_load.any():
        bad = work.loc[bad_load, "timestamp"].head(5).tolist()
        raise ValueError(f"{DEMAND_COLUMN} 缺失或非有限: {bad}")
    out = pd.DataFrame({"timestamp": work["timestamp"], "load": values})
    if len(out) > 1:
        diffs = out["timestamp"].diff().dropna()
        if not (diffs == HOUR).all():
            offenders = out.loc[out["timestamp"].diff() != HOUR, "timestamp"].head(10).tolist()
            raise ValueError(f"输出不是严格逐小时连续: {offenders}")
    return out


def prepare(
    input_dir: Path,
    out_root: Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Tuple[Dict[str, object], Path]:
    input_dir = Path(input_dir)
    out_root = Path(out_root)
    raw, files = load_balance_files(input_dir)
    hourly = transform(raw, start=start, end=end)

    out_path = out_root / OUTPUT_DATASET_DIR / "load.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out_path, index=False)

    report: Dict[str, object] = {
        "source": SOURCE_DESCRIPTION,
        "input_dir": str(input_dir),
        "input_files": files,
        "timezone_semantics": TIMEZONE_SEMANTICS,
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "load_field": DEMAND_COLUMN,
        "start_filter": start,
        "end_filter": end,
        "rows": int(len(hourly)),
        "first_timestamp": _format_ts(hourly["timestamp"].iloc[0]),
        "last_timestamp": _format_ts(hourly["timestamp"].iloc[-1]),
        "duplicate_timestamps": int(hourly["timestamp"].duplicated().sum()),
        "output_file": str(out_path),
    }
    if len(hourly) > 1:
        expected = pd.date_range(hourly["timestamp"].iloc[0], hourly["timestamp"].iloc[-1], freq="h")
        report["gaps"] = int(len(expected.difference(pd.DatetimeIndex(hourly["timestamp"]))))
    else:
        report["gaps"] = 0

    report_path = out_root / "pjm_rto_hourly_quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report, report_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="EIA-930 PJM RTO → UTC hour-ending 小时级负荷"
    )
    parser.add_argument(
        "--input-dir",
        default="data/external/pjm_eia930_2019plus_20260912/balance",
        help="EIA-930 BALANCE（PJM）CSV 目录",
    )
    parser.add_argument("--out-root", required=True, help="输出根目录")
    parser.add_argument("--start", default=None, help="UTC 起始标签（含）")
    parser.add_argument("--end", default=None, help="UTC 结束标签（含）")
    args = parser.parse_args(argv)
    report, report_path = prepare(
        Path(args.input_dir), Path(args.out_root), start=args.start, end=args.end
    )
    print(
        f"[prepare_pjm_rto_hourly] {report['output_file']} rows={report['rows']} "
        f"{report['first_timestamp']} .. {report['last_timestamp']}"
    )
    print(f"[prepare_pjm_rto_hourly] quality report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
