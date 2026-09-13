#!/usr/bin/env python3
"""V3 报告阶段：解封 T 真值，统一计算主指标、冠军交叉表与胜负。

主指标：每任务 WAPE；完全胜利 = 每任务 Modelcombine 同时低于最佳单模型、
最佳静态组合（等权 / Stacking）与动态组合（MoLE Router）。目标 N/N。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v3_shared_pool import SharedPoolError, expected_task_grid, load_copy

STATIC_METHODS = ("equal_weight", "stacking")
DYNAMIC_METHODS = ("mole_router",)


class ReportError(RuntimeError):
    """报告阶段不完整：真值取不到、任务不全等。"""


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _wape(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sum(np.abs(pred - truth)) / np.sum(np.abs(truth)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="V3 解封与报告")
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--window-plan", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    definition = _load_json(args.definition)
    plan = _load_json(args.window_plan)
    predictions = pd.read_csv(args.predictions)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    champions = _load_json(args.fit_dir / "champions.json")
    pool_validation = _load_json(args.fit_dir / "pool_validation.json")
    pool = list(pool_validation["effective_pool"])
    methods = sorted(predictions["method"].unique())
    horizons = [int(s) for s in definition["horizons"]]
    expected = expected_task_grid(plan, horizons, "test")
    observed = sorted(
        set(
            zip(
                predictions["dataset"],
                predictions["window_label"],
                predictions["forecast_steps"].astype(int),
            )
        )
    )
    if observed != sorted(expected):
        raise ReportError(
            f"T 任务网格与窗口计划不符：缺 {sorted(set(expected) - set(observed))}，"
            f"多 {sorted(set(observed) - set(expected))}"
        )

    truth_frames = []
    for entry in plan["datasets"]:
        copy = load_copy(args.data_root / entry["dataset"] / "load.csv")
        copy = copy.assign(dataset=entry["dataset"])
        truth_frames.append(copy[["dataset", "timestamp", "load"]])
    truth = pd.concat(truth_frames, ignore_index=True)
    merged = predictions.merge(truth, on=["dataset", "timestamp"], how="left")
    if merged["load"].isna().any():
        missing = merged.loc[merged["load"].isna(), ["dataset", "timestamp"]].head(3)
        raise ReportError(f"T 真值缺失，例如 {missing.to_dict(orient='records')}")

    task_rows: List[Dict[str, Any]] = []
    for (dataset, label, horizon, method), group in merged.groupby(
        ["dataset", "window_label", "forecast_steps", "method"]
    ):
        if len(group) != int(horizon):
            raise ReportError(f"{dataset}/{label}/h{horizon}/{method}: rows={len(group)}")
        task_rows.append(
            {
                "dataset": dataset,
                "window_label": label,
                "forecast_steps": int(horizon),
                "method": method,
                "wape": _wape(
                    group["yhat"].to_numpy(dtype=float),
                    group["load"].to_numpy(dtype=float),
                ),
            }
        )
    tasks = pd.DataFrame(task_rows)

    macro = {
        method: float(group["wape"].mean())
        for method, group in tasks.groupby("method")
    }

    per_task: List[Dict[str, Any]] = []
    full_wins = 0
    for (dataset, label, horizon), group in tasks.groupby(
        ["dataset", "window_label", "forecast_steps"]
    ):
        scores = dict(zip(group["method"], group["wape"]))
        best_single = min(
            ((m, scores[m]) for m in pool if m in scores), key=lambda x: x[1]
        )
        best_static = min(
            ((m, scores[m]) for m in STATIC_METHODS if m in scores), key=lambda x: x[1]
        )
        dynamic = min(
            ((m, scores[m]) for m in DYNAMIC_METHODS if m in scores), key=lambda x: x[1]
        )
        mine = scores["modelcombine"]
        full_win = (
            mine < best_single[1] and mine < best_static[1] and mine < dynamic[1]
        )
        full_wins += int(full_win)
        per_task.append(
            {
                "dataset": dataset,
                "window_label": label,
                "forecast_steps": int(horizon),
                "modelcombine_wape": mine,
                "best_single": {"model": best_single[0], "wape": best_single[1]},
                "best_static": {"method": best_static[0], "wape": best_static[1]},
                "dynamic": {"method": dynamic[0], "wape": dynamic[1]},
                "full_win": bool(full_win),
            }
        )
    total_tasks = len(per_task)

    table_rows = []
    for test_dataset in sorted(tasks["dataset"].unique()):
        row: Dict[str, Any] = {"test_dataset": test_dataset}
        for champion_dataset, info in sorted(champions.items()):
            subset = tasks[
                (tasks["dataset"] == test_dataset) & (tasks["method"] == info["model"])
            ]
            row[f"{champion_dataset}_champion"] = (
                float(subset["wape"].mean()) if not subset.empty else None
            )
        row["modelcombine"] = float(
            tasks[
                (tasks["dataset"] == test_dataset) & (tasks["method"] == "modelcombine")
            ]["wape"].mean()
        )
        table_rows.append(row)

    by_dataset = []
    for (dataset, method), group in tasks.groupby(["dataset", "method"]):
        by_dataset.append(
            {
                "dataset": dataset,
                "method": method,
                "tasks": int(len(group)),
                "macro_wape": float(group["wape"].mean()),
            }
        )
    by_horizon = []
    for (horizon, method), group in tasks.groupby(["forecast_steps", "method"]):
        by_horizon.append(
            {
                "forecast_steps": int(horizon),
                "method": method,
                "tasks": int(len(group)),
                "macro_wape": float(group["wape"].mean()),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "main_metrics.json").write_text(
        json.dumps(
            {"macro_wape": macro, "tasks": task_rows, "methods": methods},
            ensure_ascii=False, indent=2,
        ) + "\n", encoding="utf-8",
    )
    (args.out_dir / "wins.json").write_text(
        json.dumps(
            {
                "per_task": per_task,
                "full_wins": full_wins,
                "total_tasks": total_tasks,
                "target": f"{total_tasks}/{total_tasks}",
                "rule": "每任务 Modelcombine 同时低于最佳单模型、最佳静态组合与动态组合",
            },
            ensure_ascii=False, indent=2,
        ) + "\n", encoding="utf-8",
    )
    (args.out_dir / "champion_3x3.json").write_text(
        json.dumps(
            {"champions": champions, "table": table_rows},
            ensure_ascii=False, indent=2,
        ) + "\n", encoding="utf-8",
    )
    (args.out_dir / "by_dataset.json").write_text(
        json.dumps(by_dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "by_forecast_steps.json").write_text(
        json.dumps(by_horizon, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[v3-report] 完全胜利 {full_wins}/{total_tasks}；"
        f"Modelcombine 宏平均 WAPE={macro['modelcombine']:.4f}"
    )
    for method in sorted(macro, key=lambda m: macro[m]):
        print(f"[v3-report] {method:<28} WAPE={macro[method]:.4f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ReportError, SharedPoolError) as exc:
        print(f"[v3-report] 失败: {exc}", file=sys.stderr)
        sys.exit(1)
