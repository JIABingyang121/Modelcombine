#!/usr/bin/env python3
"""Stage 0：多预测长度实验的数据覆盖盘点（只读）。

方案 §12 Stage 0：只读输出三个数据集的时间范围、频率、重复时间戳、缺失区间，
确认能否容纳每个预测长度的 6 条不重叠轨迹；盘点后一次性写定具体日期。

容量与窗口口径：
- 每个预测起点之前必须有完整的 ``signature_window`` 小时真实历史（默认 720）；
- **同一个起点同时产生 H1=24、H2=168、H3=720**：三种长度共享同一份输入历史与
  同一个 ``forecast_origin``，短长度的目标区间是长长度的前缀；
- 因此窗口按**最长长度 720** 排布，相邻起点间隔 720 小时，使 720 步目标互不重叠；
- 窗口序列固定为 ``S1,S2,S3,A,T1,T2,T3``：S1—S3 建三条历史关系，A 是三条关系共用的
  冻结后审计窗口（不参与成员/权重/阈值/路由选择），T1—T3 是最终未见测试窗口；
- 因此需要一段连续的 ``signature_window + 7 * 720`` 小时数据，与本次请求了哪几种
  预测长度无关。

数据集装不下预设窗口时**停止并报告**（非零退出），不压缩该数据集窗口后继续
声称同口径。本脚本不写任何数据文件，只写一份盘点 JSON。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.model_store import SUPPORTED_FORECAST_STEPS

#: 窗口序列与各自角色。顺序即时间顺序，不可重排。
WINDOW_ROLES: tuple[tuple[str, str], ...] = (
    ("S1", "library"),
    ("S2", "library"),
    ("S3", "library"),
    ("A", "audit"),
    ("T1", "test"),
    ("T2", "test"),
    ("T3", "test"),
)

#: 预测长度的可读别名。H1/H2/H3 只表示长度，历史数据样例一律叫 S1/S2/S3。
FORECAST_HORIZONS: Dict[str, int] = {"H1": 24, "H2": 168, "H3": 720}
#: 窗口按最长长度排布，容量判据与请求了哪几种长度无关。
WINDOW_STRIDE_HOURS = max(FORECAST_HORIZONS.values())

SPLITS = ("train", "val", "test")
TARGET = "load"
STEP = pd.Timedelta(hours=1)

#: 原始序列的列名在各数据集间不统一（PJM 用 timestamp/load，AEMO 用 ds/y）。
_RAW_TIME_COLUMNS = ("timestamp", "ds", "datetime", "time")
_RAW_VALUE_COLUMNS = ("load", "y", "value", "demand")


def _read_series(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    time_col = next((c for c in _RAW_TIME_COLUMNS if c in frame.columns), None)
    value_col = next((c for c in _RAW_VALUE_COLUMNS if c in frame.columns), None)
    if time_col is None or value_col is None:
        raise ValueError(f"{path} 缺少可识别的时间/负荷列，实际列: {list(frame.columns)}")
    frame = frame[[time_col, value_col]].rename(
        columns={time_col: "timestamp", value_col: TARGET}
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    return frame.sort_values("timestamp").reset_index(drop=True)


def _source_files(root: Path, layout: str) -> Dict[str, Path]:
    """``features`` 是建库当前实际消费的切分特征文件；``raw`` 是原始负荷序列。

    两者的时间轴不同：切分文件在 train/val/test 边界上有 prepare_supervised 造成
    的整段缺口，不是一条无缝序列；冻结预测起点必须看原始序列。
    """
    if layout == "raw":
        return {"load": root / "load.csv"}
    return {split: root / f"{split}.csv" for split in SPLITS}


def _gaps(timestamps: pd.Series) -> List[Dict[str, Any]]:
    """相邻时间戳间隔不等于 1 小时的位置。"""
    deltas = timestamps.diff()
    breaks = []
    for idx in deltas.index[deltas.notna() & (deltas != STEP)]:
        breaks.append(
            {
                "after": str(timestamps.loc[idx - 1]),
                "before": str(timestamps.loc[idx]),
                "gap_hours": float(deltas.loc[idx].total_seconds() / 3600.0),
            }
        )
    return breaks


def _longest_contiguous_run(timestamps: pd.Series) -> Dict[str, Any]:
    """最长的逐小时连续段（起止与长度）。"""
    if timestamps.empty:
        return {"start": None, "end": None, "hours": 0}
    is_break = timestamps.diff() != STEP
    is_break.iloc[0] = True
    run_id = is_break.cumsum()
    sizes = run_id.value_counts()
    best = sizes.idxmax()
    run = timestamps[run_id == best]
    return {"start": str(run.iloc[0]), "end": str(run.iloc[-1]), "hours": int(len(run))}


def _proposed_origins(
    run_start: pd.Timestamp,
    run_hours: int,
    *,
    forecast_steps: Sequence[int],
    signature_window: int,
    trajectories: int,
) -> List[Dict[str, Any]]:
    """在连续段内自末尾向前排布 ``trajectories`` 个共享预测起点。

    每个起点给出一份 720 小时输入历史，以及三种预测长度各自的目标区间——它们从
    同一个起点截取不同长度的前缀。末尾对齐：最后一个窗口的 720 步目标贴着可用
    数据的末端，越靠近现在越有代表性。
    """
    first_origin_offset = run_hours - trajectories * WINDOW_STRIDE_HOURS
    origins = []
    for index in range(trajectories):
        origin_offset = first_origin_offset + index * WINDOW_STRIDE_HOURS
        origin = run_start + pd.Timedelta(hours=origin_offset - 1)
        label, role = (
            WINDOW_ROLES[index] if index < len(WINDOW_ROLES) else (f"W{index + 1}", "extra")
        )
        origins.append(
            {
                "label": label,
                "role": role,
                "history_start": str(origin - pd.Timedelta(hours=signature_window - 1)),
                "history_end": str(origin),
                "forecast_origin": str(origin),
                "targets": {
                    str(steps): {
                        "forecast_steps": int(steps),
                        "first_target": str(origin + STEP),
                        "last_target": str(origin + pd.Timedelta(hours=int(steps))),
                    }
                    for steps in sorted(forecast_steps)
                },
            }
        )
    return origins


def inventory_dataset(
    raw_root: Path,
    dataset: str,
    *,
    forecast_steps: List[int],
    signature_window: int,
    trajectories: int,
    layout: str = "features",
) -> Dict[str, Any]:
    record: Dict[str, Any] = {"dataset": dataset, "layout": layout, "splits": {}, "issues": []}
    root = raw_root / dataset
    if not root.exists():
        record["issues"].append(f"数据目录不存在: {root}")
        return record

    frames = []
    for split, path in _source_files(root, layout).items():
        if not path.exists():
            record["splits"][split] = {"present": False}
            record["issues"].append(f"缺少数据文件: {path}")
            continue
        frame = _read_series(path)
        duplicated = int(frame["timestamp"].duplicated().sum())
        record["splits"][split] = {
            "present": True,
            "rows": int(len(frame)),
            "start": str(frame["timestamp"].iloc[0]),
            "end": str(frame["timestamp"].iloc[-1]),
            "duplicate_timestamps": duplicated,
            "nan_load": int(frame[TARGET].isna().sum()),
            "gap_count": len(_gaps(frame["timestamp"])),
        }
        frames.append(frame)

    if not frames:
        return record

    union = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    gaps = _gaps(union["timestamp"])
    run = _longest_contiguous_run(union["timestamp"])
    record["union"] = {
        "rows": int(len(union)),
        "start": str(union["timestamp"].iloc[0]),
        "end": str(union["timestamp"].iloc[-1]),
        "span_hours": int(
            (union["timestamp"].iloc[-1] - union["timestamp"].iloc[0]).total_seconds() / 3600
        )
        + 1,
        "duplicate_timestamps": int(
            sum(f["timestamp"].duplicated().sum() for f in frames)
        ),
        "gap_count": len(gaps),
        "largest_gap_hours": max((g["gap_hours"] for g in gaps), default=0.0),
        "gaps": gaps[:20],
        "longest_contiguous_run": run,
    }

    # 起点由最长长度决定，一套共享给全部预测长度
    needed = signature_window + trajectories * WINDOW_STRIDE_HOURS
    fits = run["hours"] >= needed
    record["required_contiguous_hours"] = needed
    record["available_contiguous_hours"] = run["hours"]
    record["fits"] = bool(fits)
    if fits:
        record["origins"] = _proposed_origins(
            pd.Timestamp(run["start"]), run["hours"],
            forecast_steps=forecast_steps,
            signature_window=signature_window,
            trajectories=trajectories,
        )
    else:
        record["shortfall_hours"] = int(needed - run["hours"])
        record["issues"].append(
            f"最长连续段 {run['hours']} 小时 < 需要的 {needed} 小时"
            f"（缺 {needed - run['hours']} 小时）"
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 0 数据覆盖盘点（只读）")
    parser.add_argument("--raw-root", type=Path, default=Path("data/features"))
    parser.add_argument("--layout", choices=["features", "raw"], default="features",
                        help="features=建库消费的 <ds>/{train,val,test}.csv；"
                             "raw=原始负荷序列 <ds>/load.csv（冻结预测起点看这个）")
    parser.add_argument("--datasets", nargs="+", default=["pjm", "aemo_vic", "aemo_nsw"])
    parser.add_argument("--forecast-steps", nargs="+", type=int,
                        default=list(SUPPORTED_FORECAST_STEPS))
    parser.add_argument("--signature-window", type=int, default=720,
                        help="预测起点之前必须具备的真实历史小时数（§6.1）")
    parser.add_argument("--trajectories", type=int, default=len(WINDOW_ROLES),
                        help="每个数据集×预测长度需要的不重叠轨迹条数"
                             "（§6.2：S1,S2,S3,A,T1,T2,T3 共 7 条）")
    parser.add_argument("--out", type=Path, default=Path("reports/stage0/data_inventory.json"))
    args = parser.parse_args()

    raw_root = args.raw_root if args.raw_root.is_absolute() else PROJECT_ROOT / args.raw_root
    report: Dict[str, Any] = {
        "stage": "stage0_data_inventory",
        "raw_root": str(raw_root),
        "signature_window": args.signature_window,
        "trajectories": args.trajectories,
        "forecast_steps": list(args.forecast_steps),
        "forecast_horizons": dict(FORECAST_HORIZONS),
        "windows": [label for label, _role in WINDOW_ROLES],
        "layout": args.layout,
        "datasets": [],
    }
    for dataset in args.datasets:
        report["datasets"].append(
            inventory_dataset(
                raw_root, dataset,
                forecast_steps=list(args.forecast_steps),
                signature_window=args.signature_window,
                trajectories=args.trajectories,
                layout=args.layout,
            )
        )

    blocked = [
        {"dataset": entry["dataset"], "issues": entry["issues"]}
        for entry in report["datasets"] if entry["issues"]
    ]
    report["blocked"] = blocked
    report["all_feasible"] = not blocked

    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for entry in report["datasets"]:
        union = entry.get("union")
        if union:
            print(f"[stage0] {entry['dataset']}: {union['start']} -> {union['end']}"
                  f"，{union['rows']} 行，重复 {union['duplicate_timestamps']}，"
                  f"缺口 {union['gap_count']}，最长连续段 "
                  f"{union['longest_contiguous_run']['hours']} 小时")
        if "fits" in entry:
            mark = "OK" if entry["fits"] else "不足"
            print(f"           共享起点容量: {mark}"
                  f"（需要 {entry['required_contiguous_hours']}，可用 "
                  f"{entry['available_contiguous_hours']}）")
        for issue in entry["issues"]:
            print(f"           ！{entry['dataset']}: {issue}")
    print(f"\n[stage0] 盘点已保存: {out}")

    if blocked:
        print("[stage0] 存在无法容纳预设窗口的数据集，按方案 §5 停止并报告，"
              "不压缩窗口后继续声称同口径。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
