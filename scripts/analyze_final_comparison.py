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
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.final_comparison import OUTPUT_COLUMNS
from src.models.external_adapters import METHOD_LIMITATIONS

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


def _check_against_definition(
    frame: pd.DataFrame, definition: Mapping[str, Any]
) -> Dict[str, Any]:
    """严格核对冻结定义：方法集合、方法×任务覆盖、每方法的种子集合，都必须完全一致。

    只统计任务键是不够的——只有一个方法的 27 任务长表同样能凑出 27 个任务键，
    却根本没有可比较的对象。
    """
    expected_methods = set(definition["methods"])
    expected_seeds = {m: set(seeds) for m, seeds in definition["seeds"].items()}
    expected_tasks = {
        (d["dataset"], window, steps)
        for d in definition["datasets"]
        for window in definition["test_windows"]
        for steps in definition["forecast_steps"]
    }

    observed_methods = set(frame["method"])
    problems: List[str] = []
    if observed_methods != expected_methods:
        problems.append(
            f"方法集合不符：缺 {sorted(expected_methods - observed_methods)}，"
            f"多 {sorted(observed_methods - expected_methods)}"
        )
    for method in sorted(expected_methods & observed_methods):
        rows = frame[frame["method"] == method]
        tasks = {
            (d, w, int(s))
            for d, w, s in zip(rows["dataset"], rows["test_window"], rows["forecast_steps"])
        }
        if tasks != expected_tasks:
            problems.append(
                f"{method} 的任务集合不符：缺 {sorted(expected_tasks - tasks)}，"
                f"多 {sorted(tasks - expected_tasks)}"
            )
        # 种子必须在**每个任务**上都齐全：只看全局并集时，个别任务少跑一个种子会漏掉
        want = expected_seeds.get(method, set())
        seeds_by_task: Dict[tuple, set] = {}
        for d, w, st, seed in zip(
            rows["dataset"], rows["test_window"], rows["forecast_steps"], rows["seed"]
        ):
            seeds_by_task.setdefault((d, w, int(st)), set()).add(int(seed))
        bad = {task: got for task, got in seeds_by_task.items() if got != want}
        if bad:
            example = sorted(bad)[0]
            problems.append(
                f"{method} 有 {len(bad)}/{len(seeds_by_task)} 个任务的种子集合与冻结定义 "
                f"{sorted(want)} 不符，例如 {example} 只有 {sorted(bad[example])}"
            )
    return {
        "checked_against_definition": True,
        "expected_methods": sorted(expected_methods),
        "expected_tasks": len(expected_tasks),
        "problems": problems,
        "passed": not problems,
    }


def _validate(frame: pd.DataFrame) -> None:
    missing = [c for c in OUTPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise AnalysisError(f"预测长表缺少列: {missing}")
    if not np.isfinite(frame["yhat"]).all() or not np.isfinite(frame["y_true"]).all():
        raise AnalysisError("预测长表含非有限值")

    # 同口径比较的判据落在 **(方法, 种子)** 这一层：每一组都必须恰好覆盖同样的 H 个
    # 唯一时间戳。不能把一个方法所有种子的时间戳拼起来比——MoLE 三种子有 3H 行、
    # 确定性方法只有 H 行，正确的结果也会被判成"目标时间戳不一致"。
    for task, group in frame.groupby(list(TASK_KEYS)):
        steps = int(task[2])
        reference: tuple | None = None
        for (method, seed), rows in group.groupby(["method", "seed"]):
            stamps = tuple(sorted(rows["timestamp"]))
            if len(stamps) != steps:
                raise AnalysisError(
                    f"任务 {task} 方法 {method} 种子 {seed} 有 {len(stamps)} 行，"
                    f"与 forecast_steps={steps} 不符"
                )
            if len(set(stamps)) != steps:
                raise AnalysisError(
                    f"任务 {task} 方法 {method} 种子 {seed} 的目标时间戳有重复"
                )
            if reference is None:
                reference = stamps
            elif stamps != reference:
                raise AnalysisError(f"任务 {task} 的方法之间目标时间戳不一致")
        # 同一个目标时刻只能有一个真值，否则各方法比的不是同一个窗口
        if int(group.groupby("timestamp")["y_true"].nunique().max()) != 1:
            raise AnalysisError(f"任务 {task} 的同一时间戳出现了不同的 y_true")


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


def _conclusions(tasks: pd.DataFrame, complete: bool) -> Dict[str, Any]:
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
        full = complete and len(shared) == EXPECTED_TASKS
        results.append({
            "versus": method,
            "compared_tasks": int(len(shared)),
            "compared_all_tasks": bool(len(shared) == EXPECTED_TASKS),
            "mean_mae_modelcombine": float(a.mean()),
            "mean_mae_versus": float(b.mean()),
            "relative_mae_change_pct": float((a.mean() - b.mean()) / b.mean() * 100.0),
            "wins": wins,
            "win_threshold": WIN_RATE_MIN_TASKS,
            "mean_mae_lower": bool(a.mean() < b.mean()),
            # 产物与冻结定义不符、或没有覆盖全部 27 个任务时，一律不得表述总体更优
            "overall_better": bool(
                full and a.mean() < b.mean() and wins >= WIN_RATE_MIN_TASKS
            ),
        })
    return {
        "rule": "产物与冻结定义完全一致、覆盖全部 27 个任务、平均 MAE 更低、"
                "且至少赢得 14/27，四条同时成立才能表述总体更优",
        #: 与这些方法的比较不得被读成同等条件下的比较
        "method_limitations": {
            method: METHOD_LIMITATIONS[method]
            for method in sorted(set(tasks["method"])) if method in METHOD_LIMITATIONS
        },
        "complete_grid": bool(
            tasks.groupby(list(TASK_KEYS)).ngroups == EXPECTED_TASKS
        ),
        "comparisons": results,
    }


def analyse(
    frame: pd.DataFrame, definition: Mapping[str, Any] | None = None
) -> Dict[str, Any]:
    _validate(frame)
    completeness = (
        _check_against_definition(frame, definition) if definition is not None
        else {
            "checked_against_definition": False,
            "problems": ["未提供 experiment_definition.json，无法核对方法与种子集合"],
            "passed": False,
        }
    )
    tasks = _task_table(frame)
    return {
        "completeness": completeness,
        "task_metrics": tasks.to_dict(orient="records"),
        "main": _summarise(tasks, []),
        "by_dataset": _summarise(tasks, ["dataset"]),
        "by_forecast_steps": _summarise(tasks, ["forecast_steps"]),
        "ranking": _ranking(tasks),
        "conclusions": {
            **_conclusions(tasks, bool(completeness["passed"])),
            "completeness": completeness,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Piece 7：最终对比实验结果分析")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="final_comparison.py 写出的长表 CSV")
    parser.add_argument("--definition", type=Path, required=True,
                        help="freeze_final_experiment.py 冻结的 experiment_definition.json；"
                             "用于严格核对方法×任务×种子集合")
    parser.add_argument("--relation-trace", type=Path, default=None,
                        help="可选：Modelcombine 逐窗口命中关系记录 JSON，原样并入产物")
    parser.add_argument("--out", type=Path, required=True, help="输出目录")
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    definition = json.loads(args.definition.read_text(encoding="utf-8"))
    try:
        report = analyse(frame, definition)
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
    completeness = report["completeness"]
    if not completeness["passed"]:
        for problem in completeness["problems"]:
            print(f"[analyze] ！与冻结定义不符：{problem}")
        print("[analyze] 实验产物与冻结定义不一致，不得按完整实验表述结论。")
    conclusions = report["conclusions"]
    if not conclusions["complete_grid"]:
        print(f"[analyze] 任务数不是 {EXPECTED_TASKS}，不得按完整实验表述结论。")
    for method, limitation in conclusions["method_limitations"].items():
        print(f"[analyze] ！{method} 限制：{limitation}")
    for row in conclusions["comparisons"]:
        verdict = "可表述总体更优" if row["overall_better"] else "不得表述总体更优"
        print(f"[analyze] vs {row['versus']:<28} 平均 MAE "
              f"{row['mean_mae_modelcombine']:.4f} / {row['mean_mae_versus']:.4f}"
              f"（{row['relative_mae_change_pct']:+.2f}%），胜 {row['wins']}/"
              f"{row['compared_tasks']} -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
