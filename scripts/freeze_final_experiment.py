#!/usr/bin/env python3
"""Piece 5：冻结最终对比实验，生成唯一的 ``experiment_definition.json``。

记录数据日期、训练截止时间、S/A/T 起点、方法、预测长度、随机种子与输入列。生成之后
这份文件就是这一批实验的口径真源；正式测试打开后不得回头改它再在同一批 T1—T3 上重跑。

只读输入（窗口计划、原始序列），只写一份定义文件；不训练、不预测、不碰模型库。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.final_comparison import BASE_FEATURES, OUTPUT_COLUMNS, available_methods
from src.models.external_adapters import METHOD_LIMITATIONS, METHOD_SEEDS
from scripts.stage0_data_inventory import FORECAST_HORIZONS, WINDOW_ROLES
from scripts.train_combinations_kg import _library_raw_frame
from src.storage.model_store import SUPPORTED_FORECAST_STEPS

#: 深度方法固定三个种子；确定性方法只跑一次。
DEEP_METHOD_SEEDS = (42, 43, 44)
DETERMINISTIC_METHOD_SEEDS = (42,)

#: 需要多种子的深度方法。
#:
#: **iTransformer 不在其中**：服务器探针实测其官方 `run.py` 没有 CLI 种子参数，内部
#: 硬编码 `fix_seed=2023`（见 probes/itransformer/probe_manifest.json 的
#: `seed_argument`）。对它跑三个种子只会得到三份完全相同的结果，报告"三种子均值±标准差"
#: 会把"标准差为 0"读成鲁棒性，实际是由构造决定的。因此它按确定性方法只跑一次，
#: 并在定义里记下不敏感的原因。
DEEP_METHODS = ("mole", "time_moe")
SEED_INSENSITIVE_REASONS = {
    "itransformer": (
        "官方 run.py 无 CLI 种子参数，内部固定 fix_seed=2023；多种子会产出相同结果，"
        "不构成独立样本"
    ),
}

LIBRARY_WINDOWS = tuple(label for label, role in WINDOW_ROLES if role == "library")
AUDIT_WINDOWS = tuple(label for label, role in WINDOW_ROLES if role == "audit")
TEST_WINDOWS = tuple(label for label, role in WINDOW_ROLES if role == "test")


class FreezeError(RuntimeError):
    """定义无法冻结：窗口缺失、数据不覆盖窗口、方法未注册等。"""


def _dataset_definition(
    raw_root: Path, window_plan: Dict[str, Any], dataset: str
) -> Dict[str, Any]:
    entry = next(
        (e for e in window_plan["datasets"] if e["dataset"] == dataset), None
    )
    if entry is None:
        raise FreezeError(f"窗口计划里没有数据集 {dataset}")
    if not entry.get("fits", False):
        raise FreezeError(f"{dataset}: 窗口计划判定容量不足，不能冻结实验")

    origins = {o["label"]: o for o in entry["origins"]}
    missing = [
        label for label, _role in WINDOW_ROLES if label not in origins
    ]
    if missing:
        raise FreezeError(f"{dataset}: 窗口计划缺少 {missing}")

    raw = _library_raw_frame(raw_root, dataset)
    data_start, data_end = raw["timestamp"].iloc[0], raw["timestamp"].iloc[-1]

    # §5：所有参数只能用 T1 之前的数据确定 -> 训练数据右边界取 T1 的输入历史起点（不含）
    training_cutoff = pd.Timestamp(origins[TEST_WINDOWS[0]]["history_start"])
    if training_cutoff <= data_start:
        raise FreezeError(f"{dataset}: T1 之前没有可用训练数据")

    for label, origin in origins.items():
        last = max(
            pd.Timestamp(t["last_target"]) for t in origin["targets"].values()
        )
        if pd.Timestamp(origin["history_start"]) < data_start or last > data_end:
            raise FreezeError(
                f"{dataset} {label}: 窗口超出原始数据范围 {data_start}~{data_end}"
            )

    return {
        "dataset": dataset,
        "data_start": str(data_start),
        "data_end": str(data_end),
        "rows": int(len(raw)),
        "training_cutoff": str(training_cutoff),
        "training_rows": int((raw["timestamp"] < training_cutoff).sum()),
        "windows": [
            {
                "label": label,
                "role": next(r for l, r in WINDOW_ROLES if l == label),
                "history_start": origins[label]["history_start"],
                "history_end": origins[label]["history_end"],
                "forecast_origin": origins[label]["forecast_origin"],
                "targets": origins[label]["targets"],
            }
            for label, _role in WINDOW_ROLES
        ],
    }


def build_definition(
    *,
    raw_root: Path, window_plan_path: Path, datasets: Sequence[str],
    methods: Sequence[str], forecast_steps: Sequence[int], database: Path | None,
) -> Dict[str, Any]:
    unknown = [m for m in methods if m not in available_methods()]
    if unknown:
        raise FreezeError(
            f"方法未在统一入口注册: {unknown}；已注册 {available_methods()}"
        )
    plan = json.loads(window_plan_path.read_text(encoding="utf-8"))
    definitions = [
        _dataset_definition(raw_root, plan, dataset) for dataset in datasets
    ]
    seeds = {
        method: list(DEEP_METHOD_SEEDS if method in DEEP_METHODS
                     else DETERMINISTIC_METHOD_SEEDS)
        for method in methods
    }
    limitations = {
        method: METHOD_LIMITATIONS[method]
        for method in methods if method in METHOD_LIMITATIONS
    }
    seed_notes = {
        method: SEED_INSENSITIVE_REASONS[method]
        for method in methods if method in SEED_INSENSITIVE_REASONS
    }
    return {
        "experiment": "final_comparison",
        "window_plan": str(window_plan_path),
        "raw_root": str(raw_root),
        "database": None if database is None else str(database),
        "methods": list(methods),
        "seeds": seeds,
        "seed_insensitive": seed_notes,
        "official_fixed_seeds": {
            method: METHOD_SEEDS[method] for method in methods if method in METHOD_SEEDS
        },
        #: 必须随结果一起报告的方法级限制（如 Time-MoE 的预训练截止不可证）
        "method_limitations": limitations,
        "forecast_steps": [int(s) for s in forecast_steps],
        "forecast_horizons": dict(FORECAST_HORIZONS),
        "library_windows": list(LIBRARY_WINDOWS),
        "audit_windows": list(AUDIT_WINDOWS),
        "test_windows": list(TEST_WINDOWS),
        "input_columns": list(BASE_FEATURES),
        "output_columns": list(OUTPUT_COLUMNS),
        "expected_result_rows": (
            len(datasets) * len(TEST_WINDOWS) * len(forecast_steps)
        ),
        "datasets": definitions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Piece 5：冻结最终对比实验定义")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--window-plan", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--forecast-steps", nargs="+", type=int,
                        default=list(SUPPORTED_FORECAST_STEPS))
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    if out.exists():
        print(f"[freeze] {out} 已存在：这一批的定义只能冻结一次，不覆盖。")
        return 1
    try:
        definition = build_definition(
            raw_root=args.raw_root, window_plan_path=args.window_plan,
            datasets=args.datasets, methods=args.methods,
            forecast_steps=args.forecast_steps, database=args.database,
        )
    except FreezeError as exc:
        print(f"[freeze] 无法冻结: {exc}")
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(definition, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[freeze] 已冻结: {out}")
    for entry in definition["datasets"]:
        print(f"[freeze] {entry['dataset']}: 数据 {entry['data_start']} ~ "
              f"{entry['data_end']}，训练截止 {entry['training_cutoff']}"
              f"（{entry['training_rows']} 行）")
    print(f"[freeze] 方法与种子: {definition['seeds']}")
    for method, reason in definition["seed_insensitive"].items():
        print(f"[freeze] {method} 对种子不敏感：{reason}")
    for method, limitation in definition["method_limitations"].items():
        print(f"[freeze] ！{method} 限制：{limitation}")
    print(f"[freeze] 每种方法预期结果数: {definition['expected_result_rows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
