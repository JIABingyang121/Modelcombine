#!/usr/bin/env python3
"""模型库完整性的只读预检，冻结与正式运行共用同一份判据。

为什么必须在读 T1—T3 之前做：`final_comparison.run()` 的第一个动作就是按窗口切出目标区间
的真值。空库、只建了一部分关系的库、或产物丢失的库，如果要等到第一个窗口跑起来才失败，
那就是"看过 T 窗口之后再修再跑"，等于对着测试窗口迭代。

判据全部由输入推导，不写死 24 / 27：候选来自 `configs/pipeline.yaml`，数据集与预测长度
来自调用方。正式实验传 3 数据集 × 8 候选 × 3 长度时，等价于 24 个基础模型与 27 条关系。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

#: 建库时每个 (数据集, 预测长度) 会产出的历史数据样例窗口。
SCENARIO_SAMPLES = ("S1", "S2", "S3")


class LibraryIncomplete(RuntimeError):
    """模型库与本批实验的口径不符，任何结论都不成立。"""


def _read_only(database: Path) -> sqlite3.Connection:
    path = Path(database)
    if not path.exists():
        raise LibraryIncomplete(f"模型库不存在: {path}")
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def assert_library_complete(
    database: Path,
    *,
    datasets: Sequence[str],
    forecast_steps: Sequence[int],
    candidates: Sequence[str],
    base_horizon: int,
    timestamp_policy: str,
    library_report: Path | None = None,
) -> Dict[str, Any]:
    """库必须恰好是这一批实验用的完整模型库。只读，不写任何东西。"""
    problems: List[str] = []
    expected_models = {
        f"{dataset}__h{base_horizon}__{model_type}"
        for dataset in datasets
        for model_type in candidates
    }
    expected_relations = len(datasets) * len(forecast_steps) * len(SCENARIO_SAMPLES)

    connection = _read_only(database)
    try:
        models = {
            row["model_id"]: row
            for row in connection.execute(
                "SELECT model_id, artifact_path, lifecycle_stage FROM models"
            )
        }
        observed = set(models)
        if observed != expected_models:
            problems.append(
                f"基础模型不符：缺 {sorted(expected_models - observed)}，"
                f"多 {sorted(observed - expected_models)}"
            )
        inactive = sorted(
            m for m, row in models.items() if row["lifecycle_stage"] != "active"
        )
        if inactive:
            problems.append(f"这些基础模型不是 active: {inactive}")
        absent = sorted(
            m for m, row in models.items() if not Path(row["artifact_path"]).exists()
        )
        if absent:
            problems.append(f"这些基础模型的产物不存在: {absent}")

        relation_count = int(
            connection.execute("SELECT COUNT(*) FROM scenario_data_combinations").fetchone()[0]
        )
        if relation_count != expected_relations:
            problems.append(
                f"关系数是 {relation_count}，应为 {expected_relations}"
                f"（{len(datasets)} 数据集 × {len(forecast_steps)} 长度 × "
                f"{len(SCENARIO_SAMPLES)} 个历史样例）"
            )
        grid = {
            (row["region"], int(row["forecast_steps"])): int(row["n"])
            for row in connection.execute(
                "SELECT s.region, s.forecast_steps, COUNT(*) n "
                "FROM scenario_data_combinations r JOIN scenarios s USING(scenario_id) "
                "GROUP BY 1, 2"
            )
        }
        for dataset in datasets:
            for steps in forecast_steps:
                got = grid.get((dataset, int(steps)), 0)
                if got != len(SCENARIO_SAMPLES):
                    problems.append(
                        f"{dataset} forecast_steps={steps} 有 {got} 条关系，"
                        f"应为 {len(SCENARIO_SAMPLES)}"
                    )

        combos = {
            row["combination_id"]: row["artifact_path"]
            for row in connection.execute(
                "SELECT combination_id, artifact_path FROM combinations"
            )
        }
        if len(combos) != expected_relations:
            problems.append(f"组合器数是 {len(combos)}，应为 {expected_relations}")
        missing_combo = sorted(
            str(cid) for cid, path in combos.items() if not Path(path).exists()
        )
        if missing_combo:
            problems.append(f"这些组合器产物不存在: combination_id={missing_combo}")
        empty_members = [
            int(row["combination_id"])
            for row in connection.execute(
                "SELECT c.combination_id, COUNT(m.model_id) n FROM combinations c "
                "LEFT JOIN combination_members m USING(combination_id) "
                "GROUP BY 1 HAVING n = 0"
            )
        ]
        if empty_members:
            problems.append(f"这些组合没有成员: combination_id={sorted(empty_members)}")
    finally:
        connection.close()

    report_summary: Dict[str, Any] = {}
    if library_report is not None:
        report_summary = _check_library_report(
            Path(library_report), datasets=datasets, forecast_steps=forecast_steps,
            timestamp_policy=timestamp_policy, problems=problems,
        )

    if problems:
        raise LibraryIncomplete(
            f"模型库 {database} 与本批实验口径不符：\n  " + "\n  ".join(problems)
        )
    return {
        "database": str(Path(database).resolve()),
        "models": len(expected_models),
        "relations": expected_relations,
        "combinations": expected_relations,
        **report_summary,
    }


def _check_library_report(
    path: Path, *, datasets: Sequence[str], forecast_steps: Sequence[int],
    timestamp_policy: str, problems: List[str],
) -> Dict[str, Any]:
    """建库报告必须是同一批：同一个时间戳策略 + 完整的 数据集 × 长度 × S1/S2/S3 矩阵。

    策略必须核对，否则"写进报告"只是记录不是闸门——用旧策略建出来的完整库照样能冻结出
    新定义，而同一段历史在两种策略下会切出不同的序列。缺失即拒绝，不给默认值。
    """
    if not path.exists():
        problems.append(f"建库报告不存在: {path}")
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    recorded = report.get("timestamp_policy")
    if recorded != timestamp_policy:
        problems.append(
            f"建库报告的时间戳策略是 {recorded!r}，当前是 {timestamp_policy!r}——"
            "这个库不是用当前策略建的"
        )
    tasks = report.get("tasks", [])
    expected = {
        (dataset, int(steps), sample)
        for dataset in datasets
        for steps in forecast_steps
        for sample in SCENARIO_SAMPLES
    }
    observed = [
        (t.get("dataset"), int(t.get("forecast_steps", -1)), t.get("scenario_sample"))
        for t in tasks
    ]
    if len(observed) != len(expected):
        problems.append(f"建库报告有 {len(observed)} 条任务，应为 {len(expected)}")
    if set(observed) != expected:
        problems.append(
            f"建库报告的任务矩阵不符：缺 {sorted(expected - set(observed))}，"
            f"多 {sorted(set(observed) - expected)}"
        )
    if len(set(observed)) != len(observed):
        problems.append("建库报告有重复任务——同一格子出现多次说明混了多个批次")
    return {
        "library_report": str(path.resolve()),
        "library_report_tasks": len(observed),
        "timestamp_policy": recorded,
    }


def assert_window_plan_unchanged(
    window_plan: Path, definition: Mapping[str, Any]
) -> None:
    """当前窗口计划的内容必须与冻结定义里逐字段一致。

    只比路径不够：冻结之后在同一路径覆盖 `data_inventory.json`，路径检查照样通过，
    但 `run()` 读的是被替换后的窗口——时间窗口实际上没有被冻结。
    """
    path = Path(window_plan)
    if not path.exists():
        raise LibraryIncomplete(f"窗口计划不存在: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    by_dataset = {entry["dataset"]: entry for entry in plan.get("datasets", [])}
    problems: List[str] = []
    for frozen in definition["datasets"]:
        dataset = frozen["dataset"]
        entry = by_dataset.get(dataset)
        if entry is None:
            problems.append(f"{dataset}: 当前窗口计划里没有这个数据集")
            continue
        origins = {o["label"]: o for o in entry.get("origins", [])}
        for window in frozen["windows"]:
            label = window["label"]
            origin = origins.get(label)
            if origin is None:
                problems.append(f"{dataset} {label}: 当前窗口计划里没有这个窗口")
                continue
            for field in ("history_start", "history_end", "forecast_origin"):
                if str(origin.get(field)) != str(window[field]):
                    problems.append(
                        f"{dataset} {label}.{field}: 当前 {origin.get(field)!r}，"
                        f"冻结值 {window[field]!r}"
                    )
            if origin.get("targets") != window["targets"]:
                problems.append(f"{dataset} {label}.targets 与冻结值不一致")
    if problems:
        raise LibraryIncomplete(
            "窗口计划的内容与冻结定义不一致（同一路径被覆盖过？）：\n  "
            + "\n  ".join(problems)
        )
