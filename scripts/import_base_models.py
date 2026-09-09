#!/usr/bin/env python3
"""把已训练好的基础模型从旧模型库导入一个**全新的空库**，不重新训练。

用途：旧库的 24 个 h=1 基础模型在训练截止上仍然合规、产物仍在，但它的 27 条关系是按
另一套历史窗口和另一版契约建的，必须重建。直接指向旧库建库会把新关系**追加**到旧关系
旁边（`add_data_profile` 每次自增主键，`UNIQUE(scenario_id, data_profile_id,
combination_id)` 永远不触发），在线匹配又会把两批关系一起排序。所以正确做法是：新建一个
空库，只把 `models` 行搬过去，关系从零重建。

本入口只做搬运，**不训练、不写源库、不复制任何关系数据**：

1. 候选集合取自 ``configs/pipeline.yaml`` 的 ``models:`` 段（与 ``train_baselines`` 登记的同一批）。
2. 重新读三份训练 CSV 的时间列，用 Stage 0 窗口计划执行
   ``max(train.timestamp) < S1.history_start`` 门控。
3. **全部检查通过之后**才创建目标库——门控没过时不留下任何半成品。
4. 只复制 ``models`` 表字段；scenarios / data_profiles / combinations /
   combination_members / scenario_data_combinations / prediction_runs 一律不复制。
5. 校验每个 ``artifact_path`` 实际存在，输出不含哈希的导入报告。

源库以 ``mode=ro`` 打开，物理上不可写。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_baselines import load_pipeline_model_params
from scripts.train_combinations_kg import MODEL_LIBRARY_BASE_HORIZON
from src.storage.model_store import ModelStore

#: 导入报告文件名（不含哈希）。
IMPORT_REPORT_FILENAME = "base_model_import_report.json"

#: 目标库里必须保持为空的表——本入口只搬 models，关系要在新库上重建。
RELATION_TABLES = (
    "scenarios",
    "data_profiles",
    "combinations",
    "combination_members",
    "scenario_data_combinations",
    "prediction_runs",
)

#: models 表的全部字段，按 schema 顺序原样搬运。
MODEL_COLUMNS = (
    "model_id",
    "model_type",
    "task_type",
    "artifact_path",
    "required_features_json",
    "model_params_json",
    "lifecycle_stage",
    "trained_at",
)

REQUIRED_LIFECYCLE = "active"


class ImportError_(RuntimeError):
    """导入的前置条件不成立，或搬运结果与要求不符。"""


def s1_history_starts(
    window_plan: Path, datasets: Sequence[str]
) -> Dict[str, pd.Timestamp]:
    """从 Stage 0 权威窗口计划取每个数据集的 S1 输入历史起点。"""
    plan = json.loads(Path(window_plan).read_text(encoding="utf-8"))
    by_dataset = {entry["dataset"]: entry for entry in plan.get("datasets", [])}
    starts: Dict[str, pd.Timestamp] = {}
    for dataset in datasets:
        entry = by_dataset.get(dataset)
        if entry is None:
            raise ImportError_(f"窗口计划里没有数据集 {dataset}: {window_plan}")
        origins = {origin["label"]: origin for origin in entry.get("origins", [])}
        if "S1" not in origins:
            raise ImportError_(f"{dataset} 的窗口计划里没有 S1: {window_plan}")
        starts[dataset] = pd.Timestamp(origins["S1"]["history_start"])
    return starts


def assert_training_ends_before_s1(
    feature_root: Path, datasets: Sequence[str], starts: Mapping[str, pd.Timestamp]
) -> Dict[str, Dict[str, Any]]:
    """训练数据必须**严格早于** S1 输入历史起点。

    重新读一次训练 CSV 的时间列，不信任旧库里记录的任何时间——旧库的 `data_profiles`
    记的是它自己那套窗口，与本次训练截止判据无关。只读 timestamp 列，不读负荷值。
    """
    ranges: Dict[str, Dict[str, Any]] = {}
    for dataset in datasets:
        path = Path(feature_root) / dataset / "train.csv"
        if not path.exists():
            raise ImportError_(f"{dataset} 缺少训练切分: {path}")
        head = pd.read_csv(path, nrows=1)
        column = next(
            (c for c in head.columns if c.lower() in ("timestamp", "ts", "datetime", "date")),
            None,
        )
        if column is None:
            raise ImportError_(f"{dataset} 的 train.csv 没有时间列: {path}（列={list(head.columns)}）")
        # 正式复用判据不接受损坏输入：coerce+dropna 会让无法解析的时间悄悄消失，
        # 门控可能因此在坏数据上继续通过
        try:
            stamps = pd.to_datetime(
                pd.read_csv(path, usecols=[column])[column], errors="raise"
            )
        except (ValueError, TypeError) as exc:
            raise ImportError_(f"{dataset} 的 train.csv 有无法解析的时间戳: {path}（{exc}）") from exc
        if stamps.empty:
            raise ImportError_(f"{dataset} 的 train.csv 没有可用时间戳: {path}")
        last = stamps.max()
        start = starts[dataset]
        if not last < start:
            raise ImportError_(
                f"{dataset} 的训练数据截止到 {last}，未严格早于 S1 输入历史起点 {start}；"
                "这批基础模型不能复用到当前窗口计划。本入口不截断、不重新切分"
            )
        ranges[dataset] = {
            "path": str(path),
            "start": str(stamps.min()),
            "end": str(last),
            "rows": int(len(stamps)),
        }
    return ranges


def read_source_models(
    source: Path, datasets: Sequence[str], model_types: Sequence[str]
) -> List[sqlite3.Row]:
    """只读地取回声明的 24 个基础模型行，并逐条校验。

    多一个少一个都失败：缩水的候选池会让后续建库悄悄用更小的集合产出正式数字。
    """
    if not Path(source).exists():
        raise ImportError_(f"源库不存在: {source}")
    connection = sqlite3.connect(f"file:{Path(source).resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = {r["name"] for r in connection.execute("PRAGMA table_info(models)")}
        missing_columns = [c for c in MODEL_COLUMNS if c not in columns]
        if missing_columns:
            raise ImportError_(f"源库 models 表缺少字段 {missing_columns}: {source}")
        expected = [
            f"{dataset}__h{MODEL_LIBRARY_BASE_HORIZON}__{model_type}"
            for dataset in datasets
            for model_type in model_types
        ]
        rows = []
        for model_id in expected:
            row = connection.execute(
                f"SELECT {', '.join(MODEL_COLUMNS)} FROM models WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            if row is None:
                raise ImportError_(f"源库里没有 {model_id}——候选不完整，不导入")
            if row["lifecycle_stage"] != REQUIRED_LIFECYCLE:
                raise ImportError_(
                    f"{model_id} 的 lifecycle_stage 是 {row['lifecycle_stage']}，"
                    f"要求 {REQUIRED_LIFECYCLE}"
                )
            if not Path(row["artifact_path"]).exists():
                raise ImportError_(f"{model_id} 的产物不存在: {row['artifact_path']}")
            rows.append(row)
        total = connection.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        if int(total) != len(expected):
            raise ImportError_(
                f"源库 models 共 {total} 行，声明的候选是 {len(expected)} 个——"
                "源库含本次未声明的模型，不做部分导入"
            )
        return rows
    finally:
        connection.close()


def write_target(target: Path, rows: Sequence[sqlite3.Row]) -> int:
    """创建全新目标库并写入 models 行。目标库必须尚不存在。"""
    if Path(target).exists():
        raise ImportError_(f"目标库已存在，不覆盖: {target}")
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    store = ModelStore(str(target))
    try:
        store.create_schema()
        with store.connection:
            store.connection.executemany(
                f"INSERT INTO models ({', '.join(MODEL_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(MODEL_COLUMNS))})",
                [tuple(row[c] for c in MODEL_COLUMNS) for row in rows],
            )
        imported = int(store.connection.execute("SELECT COUNT(*) FROM models").fetchone()[0])
        if imported != len(rows):
            raise ImportError_(f"目标库写入 {imported} 行，应为 {len(rows)} 行")
        for table in RELATION_TABLES:
            count = int(store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            if count:
                raise ImportError_(f"目标库的 {table} 非空（{count} 行）——本入口只搬 models")
        return imported
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把已训练的基础模型从旧模型库导入一个全新的空库（不重新训练）"
    )
    parser.add_argument("--source", type=Path, required=True, help="源模型库 SQLite（只读）")
    parser.add_argument("--target", type=Path, required=True, help="目标模型库 SQLite（必须尚不存在）")
    parser.add_argument("--features", type=Path, required=True,
                        help="训练切分根目录，读 <features>/<dataset>/train.csv 的时间列")
    parser.add_argument("--window-plan", type=Path, required=True,
                        help="Stage 0 权威窗口计划；据此取 S1 输入历史起点做训练截止门控")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--pipeline-config", type=Path, default=Path("configs/pipeline.yaml"),
                        help="候选集合的唯一真源（models: 段）")
    parser.add_argument("--out", type=Path, required=True, help="导入报告输出目录")
    args = parser.parse_args()

    pipeline_config = (
        args.pipeline_config if args.pipeline_config.is_absolute()
        else PROJECT_ROOT / args.pipeline_config
    )
    model_types = list(load_pipeline_model_params(pipeline_config))
    if not model_types:
        raise ImportError_(f"{pipeline_config} 的 models: 段为空")

    # 门控在任何写操作之前：不满足时不创建目标库、不留半成品
    starts = s1_history_starts(args.window_plan, args.datasets)
    ranges = assert_training_ends_before_s1(args.features, args.datasets, starts)
    rows = read_source_models(args.source, args.datasets, model_types)
    imported = write_target(args.target, rows)

    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / IMPORT_REPORT_FILENAME
    report_path.write_text(
        json.dumps(
            {
                "source": str(args.source),
                "target": str(args.target),
                "datasets": list(args.datasets),
                "base_horizon": MODEL_LIBRARY_BASE_HORIZON,
                "models": model_types,
                "pipeline_config": str(pipeline_config),
                "s1_history_starts": {k: str(v) for k, v in starts.items()},
                "training_ranges": ranges,
                "imported_model_count": imported,
                "all_artifacts_present": True,
                "relation_tables_empty": list(RELATION_TABLES),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[import] 已导入 {imported} 个基础模型到 {args.target}")
    print(f"[import] 关系表全部为空，关系需在新库上重建")
    print(f"[import] 报告: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
