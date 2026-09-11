"""AEMO NEM PRICE_AND_DEMAND 原始五分钟数据 → 统一小时级负荷。

本模块是 AEMO 原始数据转换的**唯一实现**与正式 CLI。语义固定为：

- 时区：固定 UTC+10（AEST），不使用 Australia/Melbourne，不做夏令时转换；
- 时间戳：区间结束（interval_end）；
- 小时规则：右闭右标签，``resample("h", label="right", closed="right")``；
- 重复：``REGION + SETTLEMENTDATE`` 精确重复去重，字段冲突立即失败。

正式入口::

    python -m scripts.prepare_aemo_hourly \
        --input-root data/external/aemo_raw_20260912 \
        --years 2022 2023 2024 2025 \
        --regions VIC1 NSW1 \
        --out-root data/processed/aemo_hourly_2022_2025
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import timedelta, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "REGION",
    "SETTLEMENTDATE",
    "TOTALDEMAND",
    "RRP",
    "PERIODTYPE",
)
REGION_TO_DATASET: Mapping[str, str] = {"VIC1": "aemo_vic", "NSW1": "aemo_nsw"}
PERIODTYPE_EXPECTED = "TRADE"
SOURCE_ROWS_PER_HOUR = 12
SETTLEMENTDATE_FORMAT = "%Y/%m/%d %H:%M:%S"
AEST_TZ = timezone(timedelta(hours=10))
RAW_INTERVAL = pd.Timedelta(minutes=5)
HOURLY_INTERVAL = pd.Timedelta(hours=1)

TIMEZONE_SEMANTICS = "fixed_UTC+10_AEST"
TIMESTAMP_SEMANTICS = "interval_end"
HOURLY_RULE = "right_closed_right_labeled"
DUPLICATE_POLICY = "drop_exact_duplicates_fail_on_conflict"

_FILENAME_RE = re.compile(r"^PRICE_AND_DEMAND_(\d{4})(\d{2})_(?P<region>[A-Za-z0-9]+)\.csv$")


def _repo_commit() -> str:
    """返回处理代码所在提交；不可用时返回 "unknown"。"""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _format_ts(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def discover_month_files(
    input_root: Path, region: str, years: Sequence[int]
) -> Dict[Tuple[int, int], Path]:
    """按 ``区域 × 年 × 12 月`` 精确发现文件；缺月或数量不为 12 立即失败。"""
    region_dir = Path(input_root) / region
    if not region_dir.is_dir():
        raise FileNotFoundError(f"区域目录不存在: {region_dir}")

    found: Dict[Tuple[int, int], Path] = {}
    for path in sorted(region_dir.iterdir()):
        if not path.is_file():
            continue
        match = _FILENAME_RE.match(path.name)
        if not match:
            continue
        file_region = match.group("region")
        if file_region != region:
            raise ValueError(
                f"{path.name} 文件名区域 {file_region} 与目录区域 {region} 不一致"
            )
        key = (int(match.group(1)), int(match.group(2)))
        if key in found:
            raise ValueError(
                f"重复月份文件: {region} {key[0]}-{key[1]:02d}: "
                f"{found[key].name}, {path.name}"
            )
        found[key] = path

    missing: List[str] = []
    selected: Dict[Tuple[int, int], Path] = {}
    for year in years:
        for month in range(1, 13):
            key = (year, month)
            if key not in found:
                missing.append(f"{region} {year}-{month:02d}")
            else:
                selected[key] = found[key]
    if missing:
        raise FileNotFoundError("缺少月份文件: " + ", ".join(missing))
    for year in years:
        year_files = [p for (y, _m), p in found.items() if y == year]
        if len(year_files) != 12:
            raise ValueError(
                f"{region} {year} 必须恰好 12 个文件，实际 {len(year_files)} 个"
            )
    return selected


def _read_month_file(path: Path, region: str) -> pd.DataFrame:
    """读取单个月文件并做结构校验；不做列名猜测。"""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing_cols:
        raise ValueError(f"{path.name} 缺少必填列: {missing_cols}")

    region_values = sorted(set(frame["REGION"].astype(str).str.strip()))
    if region_values != [region]:
        raise ValueError(
            f"{path.name} 文件内 REGION={region_values} 与目录区域 {region} 不一致"
        )

    bad_period = frame["PERIODTYPE"].astype(str).str.strip() != PERIODTYPE_EXPECTED
    if bad_period.any():
        values = sorted(set(frame.loc[bad_period, "PERIODTYPE"].astype(str)))
        raise ValueError(
            f"{path.name} 存在非 {PERIODTYPE_EXPECTED} 的 PERIODTYPE: {values}"
        )

    frame = frame.copy()
    frame["source_file"] = path.name
    return frame


def load_region_raw(
    input_root: Path, region: str, years: Sequence[int]
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """读取某区域全部选中月份并附加 ``source_file`` 后合并。"""
    files = discover_month_files(input_root, region, years)
    frames: List[pd.DataFrame] = []
    files_per_year: Dict[int, int] = {int(y): 0 for y in years}
    for (year, _month), path in sorted(files.items()):
        frames.append(_read_month_file(path, region))
        files_per_year[year] += 1
    return pd.concat(frames, ignore_index=True), files_per_year


def _parse_settlement(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format=SETTLEMENTDATE_FORMAT, errors="coerce")
    bad = parsed.isna()
    if bad.any():
        examples = series[bad].astype(str).head(5).tolist()
        raise ValueError(f"SETTLEMENTDATE 无法按 AEMO 格式解析: {examples}")
    return parsed.dt.tz_localize(AEST_TZ)


def _validate_demand(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    bad = values.isna() | ~np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))
    if bad.any():
        examples = series[bad].astype(str).head(5).tolist()
        raise ValueError(f"TOTALDEMAND 存在非数字或非有限值: {examples}")
    return values.astype(float)


def _deduplicate(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """按 ``REGION + SETTLEMENTDATE`` 去重：精确重复丢弃，冲突立即失败。"""
    fields = list(REQUIRED_COLUMNS)
    work = raw.copy()
    if "_ts" not in work.columns:
        work["_ts"] = _parse_settlement(work["SETTLEMENTDATE"])
    if "source_file" not in work.columns:
        work["source_file"] = "<memory>"
    work = work.reset_index(drop=True)

    raw_duplicate_keys = 0
    exact_dropped = 0
    keep_positions: List[int] = []
    for (region, ts), group in work.groupby(["REGION", "_ts"], sort=False):
        if len(group) == 1:
            keep_positions.append(int(group.index[0]))
            continue
        raw_duplicate_keys += 1
        distinct = group[fields].drop_duplicates()
        if len(distinct) == 1:
            keep_positions.append(int(group.index[0]))
            exact_dropped += len(group) - 1
            continue
        conflict_fields = [c for c in fields if group[c].nunique(dropna=False) > 1]
        files = sorted(group["source_file"].astype(str).unique().tolist())
        raise ValueError(
            f"重复键字段冲突: REGION={region} SETTLEMENTDATE={ts} "
            f"冲突字段={conflict_fields} 文件={files}"
        )

    deduped = (
        work.loc[sorted(keep_positions)]
        .sort_values("_ts")
        .reset_index(drop=True)
    )
    stats = {
        "raw_duplicate_keys": raw_duplicate_keys,
        "exact_duplicate_rows_dropped": exact_dropped,
        "conflict_duplicate_keys": 0,
    }
    return deduped, stats


def aggregate_hourly(
    raw: pd.DataFrame, *, region: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """把合并后的原始五分钟数据聚合为统一小时级负荷。

    返回 ``(hourly_df, stats)``。``hourly_df`` 含 ``timestamp`` 与 ``load`` 两列，
    时间戳为无时区偏移的本地市场时间（区间结束，右闭右标签）。
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"缺少必填列: {missing_cols}")

    if region is not None:
        region_values = sorted(set(raw["REGION"].astype(str).str.strip()))
        if region_values != [region]:
            raise ValueError(f"REGION={region_values} 与期望区域 {region} 不一致")

    work = raw.copy()
    work["_ts"] = _parse_settlement(work["SETTLEMENTDATE"])
    demand = _validate_demand(work["TOTALDEMAND"])
    non_numeric_or_non_finite = 0
    work["_load"] = demand

    raw_start = work["_ts"].min()
    raw_end = work["_ts"].max()
    raw_year = (work["_ts"] - RAW_INTERVAL).dt.year
    raw_rows_per_year = {
        int(k): int(v) for k, v in raw_year.value_counts().sort_index().items()
    }

    deduped, dup_stats = _deduplicate(work)
    series = deduped.set_index("_ts")["_load"].sort_index()

    resampler = series.resample("h", label="right", closed="right")
    means = resampler.mean()
    counts = resampler.count()

    incomplete = counts[counts != SOURCE_ROWS_PER_HOUR]
    if len(incomplete) > 0:
        labels = [str(x) for x in incomplete.index[:20]]
        raise ValueError(
            f"存在来源记录数不等于 {SOURCE_ROWS_PER_HOUR} 的小时，"
            f"共 {len(incomplete)} 个: {labels}"
        )

    hourly = pd.DataFrame(
        {
            "timestamp": means.index.tz_localize(None),
            "load": means.to_numpy(dtype=float),
        }
    ).sort_values("timestamp").reset_index(drop=True)

    distribution = {
        str(int(k)): int(v) for k, v in counts.value_counts().sort_index().items()
    }
    output_year = (hourly["timestamp"] - HOURLY_INTERVAL).dt.year
    hourly_rows_per_year = {
        int(k): int(v) for k, v in output_year.value_counts().sort_index().items()
    }

    output_duplicate_timestamps = int(hourly["timestamp"].duplicated().sum())
    if len(hourly) > 0:
        expected = pd.date_range(hourly["timestamp"].iloc[0], hourly["timestamp"].iloc[-1], freq="h")
        output_gaps = int(len(expected.difference(pd.DatetimeIndex(hourly["timestamp"]))))
    else:
        output_gaps = 0

    stats: Dict[str, object] = {
        "raw_rows_total": int(len(work)),
        "raw_rows_per_year": raw_rows_per_year,
        "raw_start": raw_start,
        "raw_end": raw_end,
        "raw_duplicate_keys": dup_stats["raw_duplicate_keys"],
        "exact_duplicate_rows_dropped": dup_stats["exact_duplicate_rows_dropped"],
        "conflict_duplicate_keys": dup_stats["conflict_duplicate_keys"],
        "non_numeric_or_non_finite": non_numeric_or_non_finite,
        "source_rows_per_hour_distribution": distribution,
        "incomplete_hours": [],
        "hourly_rows": int(len(hourly)),
        "hourly_rows_per_year": hourly_rows_per_year,
        "output_start": hourly["timestamp"].iloc[0] if len(hourly) else None,
        "output_end": hourly["timestamp"].iloc[-1] if len(hourly) else None,
        "output_duplicate_timestamps": output_duplicate_timestamps,
        "output_gaps": output_gaps,
    }
    return hourly, stats


def prepare(
    input_root: Path,
    years: Sequence[int],
    regions: Sequence[str],
    out_root: Path,
) -> Tuple[Dict[str, object], Path]:
    """执行完整转换：先全部成功，再一次性写出 CSV 与质量报告。"""
    input_root = Path(input_root)
    out_root = Path(out_root)
    unknown = [r for r in regions if r not in REGION_TO_DATASET]
    if unknown:
        raise ValueError(f"不支持的区域: {unknown}; 仅支持 {sorted(REGION_TO_DATASET)}")

    computed: Dict[str, Tuple[pd.DataFrame, str, Dict[int, int], Dict[str, object]]] = {}
    for region in regions:
        raw, files_per_year = load_region_raw(input_root, region, years)
        hourly, stats = aggregate_hourly(raw, region=region)
        computed[region] = (hourly, REGION_TO_DATASET[region], files_per_year, stats)

    report: Dict[str, object] = {
        "processing_code_commit": _repo_commit(),
        "input_root": str(input_root),
        "selected_years": [int(y) for y in years],
        "region_mapping": {r: REGION_TO_DATASET[r] for r in regions},
        "regions": {},
        "time_semantics": {
            "timezone_semantics": TIMEZONE_SEMANTICS,
            "timestamp_semantics": TIMESTAMP_SEMANTICS,
            "hourly_rule": HOURLY_RULE,
            "duplicate_policy": DUPLICATE_POLICY,
        },
        "output_files": {},
    }

    for region, (hourly, dataset, files_per_year, stats) in computed.items():
        out_path = out_root / dataset / "load.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(out_path, index=False)
        report["output_files"][region] = str(out_path)
        report["regions"][region] = {
            "dataset": dataset,
            "input_files_per_year": {str(y): int(files_per_year[y]) for y in years},
            "raw_rows_per_year": stats["raw_rows_per_year"],
            "actual_start": _format_ts(stats["raw_start"]),
            "actual_end": _format_ts(stats["raw_end"]),
            "missing_months": [],
            "raw_duplicate_keys": stats["raw_duplicate_keys"],
            "exact_duplicate_rows_dropped": stats["exact_duplicate_rows_dropped"],
            "conflict_duplicate_keys": stats["conflict_duplicate_keys"],
            "non_numeric_or_non_finite": stats["non_numeric_or_non_finite"],
            "source_rows_per_hour_distribution": stats["source_rows_per_hour_distribution"],
            "incomplete_hours": stats["incomplete_hours"],
            "hourly_rows": stats["hourly_rows"],
            "hourly_rows_per_year": stats["hourly_rows_per_year"],
            "output_start": _format_ts(stats["output_start"]),
            "output_end": _format_ts(stats["output_end"]),
            "output_duplicate_timestamps": stats["output_duplicate_timestamps"],
            "output_gaps": stats["output_gaps"],
        }

    report_path = out_root / "aemo_hourly_quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report, report_path


def _parse_years(values: Sequence[str]) -> List[int]:
    years: List[int] = []
    for value in values:
        try:
            years.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"无效年份: {value}") from exc
    if not years:
        raise ValueError("至少需要一个年份")
    return years


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="AEMO 原始五分钟数据 → 统一小时级负荷（唯一实现）"
    )
    parser.add_argument(
        "--input-root", default="data/external/aemo_raw_20260912", help="原始数据根目录"
    )
    parser.add_argument(
        "--years", nargs="+", default=["2022", "2023", "2024", "2025"], help="选择年份"
    )
    parser.add_argument(
        "--regions", nargs="+", default=["VIC1", "NSW1"], help="选择区域"
    )
    parser.add_argument(
        "--out-root",
        default="data/processed/aemo_hourly_2022_2025",
        help="输出根目录",
    )
    args = parser.parse_args(argv)

    report, report_path = prepare(
        input_root=Path(args.input_root),
        years=_parse_years(args.years),
        regions=list(args.regions),
        out_root=Path(args.out_root),
    )
    for region, meta in report["regions"].items():
        print(
            f"[prepare_aemo_hourly] {region} -> {report['output_files'][region]} "
            f"rows={meta['hourly_rows']} {meta['output_start']} .. {meta['output_end']}"
        )
    print(f"[prepare_aemo_hourly] quality report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
