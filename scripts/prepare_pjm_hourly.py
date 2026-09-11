"""PJM East 小时负荷：本地 hour-ending EPT → 连续固定 EST/UTC-5 小时序列。

来源与语义（审查结论）
--------------------
- ``data/pjm/load.csv`` 由 ``scripts/convert_pjm.py`` 从 Kaggle
  ``robikscube/hourly-energy-consumption`` 的 ``PJME_hourly.csv``（列
  ``Datetime,PJME_MW``）转换而来；``scripts/download_pjm.py`` 负责取该文件。
- 时间戳为 **hour ending**，时区为 **Eastern Prevailing Time（EST/EDT，随
  夏令时切换）**。依据：PJM 官方 Data Miner API 指南对 ``_ept`` 的定义
  （Eastern Prevailing Time），以及数据本身春季跳时缺口恰为
  ``02:00->04:00``、秋季重复标签恰为 ``02:00``——两者只与 hour-ending 唯一一致。

转换规则
--------
1. 原 timestamp 视为 interval-end（本地 EPT）；
2. timestamp - 1h 得到 interval-start（本地 EPT）；
3. 按 ``America/New_York`` 解析 DST；秋季重复小时按**原始行顺序**区分
   （第 1 次为 DST，第 2 次为标准时）；
4. 转换到固定 UTC-5/EST；
5. 再加 1h 恢复 interval-end 标签；
6. 输出无时区偏移的固定 EST timestamp（``timestamp,load``）。

不平均、不插值、不制造负荷；非 DST 缺口立即失败。

正式入口::

    python -m scripts.prepare_pjm_hourly \
        --input data/pjm/load.csv \
        --start 2014-01-01 \
        --out-root data/processed/pjm_hourly_2014_2018
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EASTERN_TZ = "America/New_York"
FIXED_EST = timezone(timedelta(hours=-5))
OUTPUT_TIMEZONE = "fixed_UTC-5_EST"
INPUT_TIMEZONE = "America/New_York_Eastern_Prevailing_Time_EST_EDT"
TIMESTAMP_SEMANTICS = "interval_end"
DST_PARSING_RULE = (
    "America/New_York; ambiguous fall-back resolved by original row order "
    "(first occurrence DST, second occurrence STANDARD)"
)
SOURCE_DESCRIPTION = (
    "PJME_hourly.csv (Kaggle robikscube/hourly-energy-consumption v3, "
    "from PJM website hourly load)"
)
OUTPUT_DATASET_DIR = "pjm"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
HOUR = pd.Timedelta(hours=1)


def _load_input(path: Path) -> pd.DataFrame:
    """读取并校验 ``timestamp,load``；不做列名猜测。"""
    frame = pd.read_csv(path)
    missing = [c for c in ("timestamp", "load") if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} 缺少必填列: {missing}")

    ts = pd.to_datetime(frame["timestamp"], format=DATETIME_FORMAT, errors="coerce")
    if ts.isna().any():
        bad = frame.loc[ts.isna(), "timestamp"].astype(str).head(5).tolist()
        raise ValueError(f"{path} timestamp 无法解析: {bad}")

    load = pd.to_numeric(frame["load"], errors="coerce")
    nonfinite = load.isna() | ~np.isfinite(load.to_numpy(dtype="float64", na_value=np.nan))
    if nonfinite.any():
        bad = frame.loc[nonfinite, "load"].astype(str).head(5).tolist()
        raise ValueError(f"{path} load 存在非数字或非有限值: {bad}")

    return pd.DataFrame({"timestamp": ts, "load": load.astype(float)})


def _localize_interval_starts(ts_start: pd.Series) -> pd.DatetimeIndex:
    """把 interval-start（本地 EPT 无时区）按 America/New_York 解析 DST。

    秋季回拨的歧义时刻按原始行顺序区分：第 1 次出现为 DST（``ambiguous=True``），
    第 2 次为标准时（``False``）。
    """
    ranks = ts_start.groupby(ts_start).cumcount()
    folds = (ranks == 0).to_numpy()
    idx = pd.DatetimeIndex(ts_start)
    return idx.tz_localize(EASTERN_TZ, ambiguous=folds, nonexistent="raise")


def _classify_input_gaps(frame: pd.DataFrame) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """识别原始标签中的两小时缺口，并区分 DST 春季跳时与真实缺数据。"""
    labels = pd.Series(sorted(frame["timestamp"].unique()))
    diffs = labels.diff()
    two_hour: List[Dict[str, str]] = []
    non_dst: List[Dict[str, str]] = []
    for pos in labels.index[diffs == pd.Timedelta(hours=2)]:
        prev_label = labels.iloc[pos - 1]
        curr_label = labels.iloc[pos]
        # 春季跳时缺口的前一个标签可能是不存在的本地时刻（如 02:00），
        # 用 shift_forward 定位其 UTC，再与后一个标签比较真实物理间隔。
        prev_utc = (
            pd.Timestamp(prev_label)
            .tz_localize(EASTERN_TZ, nonexistent="shift_forward")
            .tz_convert("UTC")
        )
        curr_utc = (
            pd.Timestamp(curr_label)
            .tz_localize(EASTERN_TZ, nonexistent="shift_forward")
            .tz_convert("UTC")
        )
        gap = {"from": str(prev_label), "to": str(curr_label)}
        two_hour.append(gap)
        if (curr_utc - prev_utc) != HOUR:
            non_dst.append(gap)
    return two_hour, non_dst


def transform(frame: pd.DataFrame) -> pd.DataFrame:
    """执行 EPT hour-ending → 固定 EST hour-ending 的转换（不修改输入）。"""
    if frame.empty:
        raise ValueError("没有可转换的数据")
    ts_start = frame["timestamp"] - HOUR
    localized = _localize_interval_starts(ts_start)
    est_end = (
        localized.tz_convert("UTC").tz_convert(FIXED_EST) + HOUR
    ).tz_localize(None)
    out = pd.DataFrame({"timestamp": est_end, "load": frame["load"].to_numpy()})
    return out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def prepare(
    input_path: Path, start: str, out_root: Path
) -> Tuple[Dict[str, object], Path]:
    """完整转换：先校验成功，再一次性写出 CSV 与质量报告。"""
    input_path = Path(input_path)
    out_root = Path(out_root)
    start_ts = pd.Timestamp(start)

    frame = _load_input(input_path)
    frame = frame[(frame["timestamp"] - HOUR) >= start_ts].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"没有区间起点 >= {start_ts} 的数据")

    input_rows = int(len(frame))
    unique_labels = int(frame["timestamp"].nunique())
    duplicate_labels = int(frame["timestamp"][frame["timestamp"].duplicated()].nunique())
    extra_rows = input_rows - unique_labels

    two_hour_gaps, non_dst_gaps = _classify_input_gaps(frame)
    if non_dst_gaps:
        raise ValueError(f"输入存在非 DST 缺口，拒绝转换: {non_dst_gaps}")

    out = transform(frame)

    if out["timestamp"].duplicated().any():
        dup = out.loc[out["timestamp"].duplicated(keep=False), "timestamp"].head(5).tolist()
        raise ValueError(f"输出存在重复 timestamp: {dup}")
    if len(out) > 1:
        diffs = out["timestamp"].diff().dropna()
        if not (diffs == HOUR).all():
            offenders = out.loc[out["timestamp"].diff() != HOUR, "timestamp"].head(10).tolist()
            raise ValueError(f"输出不是严格逐小时连续: {offenders}")

    output_duplicate = int(out["timestamp"].duplicated().sum())
    if len(out) > 1:
        expected = pd.date_range(out["timestamp"].iloc[0], out["timestamp"].iloc[-1], freq="h")
        output_gaps = int(len(expected.difference(pd.DatetimeIndex(out["timestamp"]))))
    else:
        output_gaps = 0

    load_one_to_one = bool(
        sorted(out["load"].tolist()) == sorted(frame["load"].tolist())
        and len(out) == len(frame)
    )

    out_path = out_root / OUTPUT_DATASET_DIR / "load.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    report: Dict[str, object] = {
        "source": SOURCE_DESCRIPTION,
        "input_path": str(input_path),
        "input_timezone": INPUT_TIMEZONE,
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "dst_parsing_rule": DST_PARSING_RULE,
        "output_timezone": OUTPUT_TIMEZONE,
        "start_filter": f"interval_start >= {start_ts}",
        "input": {
            "rows": input_rows,
            "unique_timestamps": unique_labels,
            "duplicate_labels": duplicate_labels,
            "dst_expanded_labels": duplicate_labels,
            "extra_duplicate_rows": extra_rows,
            "first_timestamp": str(frame["timestamp"].min()),
            "last_timestamp": str(frame["timestamp"].max()),
            "two_hour_gaps": two_hour_gaps,
            "non_dst_gaps": non_dst_gaps,
        },
        "output": {
            "rows": int(len(out)),
            "first_timestamp": str(out["timestamp"].iloc[0]),
            "last_timestamp": str(out["timestamp"].iloc[-1]),
            "duplicate_timestamps": output_duplicate,
            "gaps": output_gaps,
            "step_hours": 1,
        },
        "load_one_to_one": load_one_to_one,
        "output_file": str(out_path),
    }

    report_path = out_root / "pjm_hourly_quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report, report_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="PJM hour-ending EPT → 连续固定 EST/UTC-5 小时序列（唯一实现）"
    )
    parser.add_argument("--input", default="data/pjm/load.csv", help="输入 load.csv")
    parser.add_argument("--start", default="2014-01-01", help="区间起点下界（含）")
    parser.add_argument(
        "--out-root",
        default="data/processed/pjm_hourly_2014_2018",
        help="输出根目录",
    )
    args = parser.parse_args(argv)

    report, report_path = prepare(
        input_path=Path(args.input), start=args.start, out_root=Path(args.out_root)
    )
    out = report["output"]
    print(
        f"[prepare_pjm_hourly] {report['output_file']} rows={out['rows']} "
        f"{out['first_timestamp']} .. {out['last_timestamp']}"
    )
    print(f"[prepare_pjm_hourly] quality report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
