#!/usr/bin/env python3
"""Piece 7：直接读取最终对比实验的预测长表，产出结果表。

输入是 ``final_comparison.py`` 写出的长表；本脚本只读、只统计，**不复现实验、不重训、
不在 T1—T3 上改参数重跑**。

产出：

```text
main_metrics.json      主结果表（每方法在 27 个任务上的 MAE/RMSE/WAPE 汇总）
by_dataset.json        分数据集结果表
by_forecast_steps.json 分预测长度结果表
ranking.json           胜场数与平均排名
conclusions.json       §8 结论规则的逐条布尔判定
```

一个"任务"= (数据集, 测试窗口, 预测长度)，共 3 × 3 × 3 = 27 个。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.final_comparison import OUTPUT_COLUMNS

METHOD_UNDER_TEST = "modelcombine"
TASK_KEYS = ("dataset", "test_window", "forecast_steps")
EXPECTED_TASKS = 27

#: §8 结论规则：同时满足才能说"总体优于某方法"。
WIN_RATE_MIN_TASKS = 14


class AnalysisError(RuntimeError):
    """产物不完整或口径不一致，任何结论都不成立。"""


def _metrics(group: pd.DataFrame) -> Dict[str, float]:
    y = group["y_true"].to_numpy(dtype=float)
    yhat = group["yhat"].to_numpy(dtype=float)
    error = np.abs(yhat - y)
    return {
        "mae": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean((yhat - y) ** 2))),
        "wape": float(np.sum(error) / np.sum(np.abs(y))),
        "n_rows": int(len(group)),
    }


def _validate(frame: pd.DataFrame) -> None:
    missing = [c for c in OUTPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise AnalysisError(f"预测长表缺少列: {missing}")
    if not np.isfinite(frame["yhat"]).all() or not np.isfinite(frame["y_true"]).all():
        raise AnalysisError("预测长表含非有限值")

    # 所有方法必须落在完全相同的目标时间戳上，否则不是同口径比较
    per_task = frame.groupby(list(TASK_KEYS))
    for task, group in per_task:
        stamps = {
            method: tuple(sorted(rows["timestamp"]))
            for method, rows in group.groupby("method")
        }
        if len(set(stamps.values())) != 1:
            raise AnalysisError(f"任务 {task} 的方法之间目标时间戳不一致")
        for method, rows in group.groupby("method"):
            if len(rows) != int(task[2]) * rows["seed"].nunique():
                raise AnalysisError(
                    f"任务 {task} 方法 {method} 行数 {len(rows)} 与 forecast_steps 不符"
                )


def _task_table(frame: pd.DataFrame) -> pd.DataFrame:
    """逐任务、逐方法的指标；多种子先在种子间取平均（§6 深度方法报均值）。"""
    rows: List[Dict[str, Any]] = []
    for (dataset, window, steps, method), group in frame.groupby(
        [*TASK_KEYS, "method"]
    ):
        per_seed = [_metrics(g) for _seed, g in group.groupby("seed")]
        rows.append({
            "dataset": dataset, "test_window": window, "forecast_steps": int(steps),
            "method": method, "seeds": int(group["seed"].nunique()),
            **{
                metric: float(np.mean([m[metric] for m in per_seed]))
                for metric in ("mae", "rmse", "wape")
            },
        })
    return pd.DataFrame(rows)


def _summarise(tasks: pd.DataFrame, by: Sequence[str]) -> List[Dict[str, Any]]:
    keys = [*by, "method"] if by else ["method"]
    out = []
    for key, group in tasks.groupby(keys):
        record = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        record.update({
            "tasks": int(len(group)),
            **{m: float(group[m].mean()) for m in ("mae", "rmse", "wape")},
        })
        out.append(record)
    return out


def _ranking(tasks: pd.DataFrame) -> Dict[str, Any]:
    """逐任务按 MAE 排名，统计胜场数与平均排名。"""
    methods = sorted(tasks["method"].unique())
    wins = {m: 0 for m in methods}
    ranks: Dict[str, List[float]] = {m: [] for m in methods}
    for _task, group in tasks.groupby(list(TASK_KEYS)):
        ordered = group.sort_values("mae")
        wins[ordered.iloc[0]["method"]] += 1
        for rank, (_i, row) in enumerate(ordered.iterrows(), start=1):
            ranks[row["method"]].append(float(rank))
    return {
        "n_tasks": int(tasks.groupby(list(TASK_KEYS)).ngroups),
        "methods": [
            {
                "method": m, "wins": wins[m],
                "mean_rank": float(np.mean(ranks[m])) if ranks[m] else None,
            }
            for m in sorted(methods, key=lambda m: (-wins[m], np.mean(ranks[m])))
        ],
    }


def _conclusions(tasks: pd.DataFrame) -> Dict[str, Any]:
    """§8：27 任务平均 MAE 更低 **且** 至少赢 14/27，才能说"总体优于"。"""
    if METHOD_UNDER_TEST not in set(tasks["method"]):
        raise AnalysisError(f"长表里没有 {METHOD_UNDER_TEST}")
    mine = tasks[tasks["method"] == METHOD_UNDER_TEST].set_index(list(TASK_KEYS))
    results = []
    for method in sorted(set(tasks["method"]) - {METHOD_UNDER_TEST}):
        other = tasks[tasks["method"] == method].set_index(list(TASK_KEYS))
        shared = mine.index.intersection(other.index)
        a, b = mine.loc[shared, "mae"], other.loc[shared, "mae"]
        wins = int((a < b).sum())
        results.append({
            "versus": method,
            "compared_tasks": int(len(shared)),
            "mean_mae_modelcombine": float(a.mean()),
            "mean_mae_versus": float(b.mean()),
            "relative_mae_change_pct": float((a.mean() - b.mean()) / b.mean() * 100.0),
            "wins": wins,
            "win_threshold": WIN_RATE_MIN_TASKS,
            "mean_mae_lower": bool(a.mean() < b.mean()),
            "overall_better": bool(a.mean() < b.mean() and wins >= WIN_RATE_MIN_TASKS),
        })
    return {
        "rule": "27 个任务平均 MAE 更低，且至少赢得 14/27，才能表述总体更优",
        "complete_grid": bool(
            tasks.groupby(list(TASK_KEYS)).ngroups == EXPECTED_TASKS
        ),
        "comparisons": results,
    }


def analyse(frame: pd.DataFrame) -> Dict[str, Any]:
    _validate(frame)
    tasks = _task_table(frame)
    return {
        "task_metrics": tasks.to_dict(orient="records"),
        "main": _summarise(tasks, []),
        "by_dataset": _summarise(tasks, ["dataset"]),
        "by_forecast_steps": _summarise(tasks, ["forecast_steps"]),
        "ranking": _ranking(tasks),
        "conclusions": _conclusions(tasks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Piece 7：最终对比实验结果分析")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="final_comparison.py 写出的长表 CSV")
    parser.add_argument("--relation-trace", type=Path, default=None,
                        help="可选：Modelcombine 逐窗口命中关系记录 JSON，原样并入产物")
    parser.add_argument("--out", type=Path, required=True, help="输出目录")
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    try:
        report = analyse(frame)
    except AnalysisError as exc:
        print(f"[analyze] 产物不完整，不产出结论: {exc}")
        return 1

    if args.relation_trace is not None:
        report["modelcombine_relations"] = json.loads(
            args.relation_trace.read_text(encoding="utf-8")
        )

    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("main_metrics.json", {"main": report["main"], "task_metrics": report["task_metrics"]}),
        ("by_dataset.json", report["by_dataset"]),
        ("by_forecast_steps.json", report["by_forecast_steps"]),
        ("ranking.json", report["ranking"]),
        ("conclusions.json", report["conclusions"]),
    ):
        (out / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if "modelcombine_relations" in report:
        (out / "modelcombine_relations.json").write_text(
            json.dumps(report["modelcombine_relations"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"[analyze] 结果已保存: {out}")
    for row in sorted(report["main"], key=lambda r: r["mae"]):
        print(f"[analyze] {row['method']:<28} MAE={row['mae']:.4f} "
              f"RMSE={row['rmse']:.4f} WAPE={row['wape']:.4f}（{row['tasks']} 个任务）")
    ranking = report["ranking"]
    print(f"[analyze] 共 {ranking['n_tasks']} 个任务")
    for row in ranking["methods"]:
        print(f"[analyze] {row['method']:<28} 胜场 {row['wins']}，平均排名 {row['mean_rank']:.2f}")
    conclusions = report["conclusions"]
    if not conclusions["complete_grid"]:
        print(f"[analyze] 任务数不是 {EXPECTED_TASKS}，不得按完整实验表述结论。")
    for row in conclusions["comparisons"]:
        verdict = "可表述总体更优" if row["overall_better"] else "不得表述总体更优"
        print(f"[analyze] vs {row['versus']:<28} 平均 MAE "
              f"{row['mean_mae_modelcombine']:.4f} / {row['mean_mae_versus']:.4f}"
              f"（{row['relative_mae_change_pct']:+.2f}%），胜 {row['wins']}/"
              f"{row['compared_tasks']} -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
