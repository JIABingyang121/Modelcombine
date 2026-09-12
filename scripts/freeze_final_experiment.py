#!/usr/bin/env python3
"""Piece 5：冻结最终对比实验，生成唯一的 ``experiment_definition.json``。

记录数据日期、训练截止时间、S/A/T 起点、方法、预测长度、随机种子与输入列。生成之后
这份文件就是这一批实验的口径真源；正式测试打开后不得回头改它再在同一批 T1—T3 上重跑。

只读输入（窗口计划、原始序列），只写一份定义文件；不训练、不预测、不碰模型库。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.final_comparison import (
    BASE_FEATURES,
    OUTPUT_COLUMNS,
    FinalComparisonError,
    available_methods,
    repo_state,
)
from src.models.external_adapters import (
    EXTERNAL_METHODS,
    FROZEN_EXTERNAL_KEYS,
    METHOD_LIMITATIONS,
    OFFICIAL_FIXED_SEED,
    REQUIRED_HYPERPARAMETERS,
)
from scripts.stage0_data_inventory import FORECAST_HORIZONS, WINDOW_ROLES
from scripts.library_preflight import LibraryIncomplete, assert_library_complete
from scripts.train_baselines import load_pipeline_model_params
from scripts.train_combinations_kg import (
    MODEL_LIBRARY_BASE_HORIZON,
    TIMESTAMP_POLICY,
    _library_raw_timestamps,
)
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
DEEP_METHODS = ("mole",)
SEED_INSENSITIVE_REASONS = {
    "itransformer": (
        "官方 run.py 无 CLI 种子参数，内部固定 fix_seed=2023；多种子会产出相同结果，"
        "不构成独立样本"
    ),
    "time_moe": (
        "零样本推理 + 确定性贪心生成，无任务训练；多种子会产出相同结果，"
        "不得包装成三个独立样本"
    ),
}

#: 正式实验批准的数据集，冻结时不接受子集。
APPROVED_DATASETS = ("pjm_rto", "aemo_vic", "aemo_nsw")

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

    # 冻结只需要数据覆盖范围，只读时间列——不把任何窗口的负荷值载入内存
    stamps = _library_raw_timestamps(raw_root, dataset)
    data_start, data_end = stamps.iloc[0], stamps.iloc[-1]

    # B 训练截止：优先窗口计划里的显式值（v2 契约），缺省回退 T1.history_start（旧口径）。
    plan_cutoff = entry.get("training_cutoff")
    training_cutoff = (
        pd.Timestamp(plan_cutoff)
        if plan_cutoff is not None
        else pd.Timestamp(origins[TEST_WINDOWS[0]]["history_start"])
    )
    if training_cutoff <= data_start:
        raise FreezeError(f"{dataset}: training_cutoff 之前没有可用训练数据")

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
        "rows": int(len(stamps)),
        "training_cutoff": str(training_cutoff),
        "training_rows": int((stamps < training_cutoff).sum()),
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


def _frozen_external(
    methods: Sequence[str], external_config: Mapping[str, Any] | None
) -> Dict[str, Dict[str, Any]]:
    """把外部方法的正式口径固定进定义：超参数、官方代码版本、Time-MoE 权重标识。

    机器本地路径（repo/python/checkpoints/snapshot）不是口径，不写进定义——换台机器
    路径就变了，但超参数和官方版本不能变。
    """
    requested = [m for m in methods if m in EXTERNAL_METHODS]
    if not requested:
        return {}
    if not external_config:
        raise FreezeError(
            f"{requested} 走官方外部实现，必须用 --external-config 提供正式超参数、"
            "官方代码版本（commit）与 Time-MoE 权重标识，否则定义冻结后仍能改口径重跑"
        )
    frozen: Dict[str, Dict[str, Any]] = {}
    for method in requested:
        config = external_config.get(method)
        if not config:
            raise FreezeError(f"--external-config 里没有 {method} 的配置")
        missing = [k for k in FROZEN_EXTERNAL_KEYS[method] if k not in config]
        if missing:
            raise FreezeError(f"{method} 的 external-config 缺少 {missing}")
        if method in REQUIRED_HYPERPARAMETERS:
            absent = [
                k for k in REQUIRED_HYPERPARAMETERS[method]
                if k not in config["hyperparameters"]
            ]
            if absent:
                raise FreezeError(f"{method} 的 hyperparameters 缺少 {absent}")
        frozen[method] = {k: config[k] for k in FROZEN_EXTERNAL_KEYS[method]}
    return frozen


def _repo_commit() -> str:
    """冻结时的主仓库版本；正式运行必须在同一个 commit 上进行。

    只记 HEAD，不额外判定工作树是否干净——"HEAD 相同但工作树被改过"这个缺口留在流程上
    （服务器是执行环境，同步用 ff-only，运行前工作树本就应当干净），不在这里加检查。
    """
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise FreezeError(f"读不到主仓库 HEAD，无法冻结版本\n{completed.stderr.strip()}")
    return completed.stdout.strip()


def assert_formal_scope(
    methods: Sequence[str], datasets: Sequence[str], forecast_steps: Sequence[int]
) -> None:
    """正式冻结的口径不接受子集。

    定义一旦写出，分析器就把它当作"预期的全部"：只冻结 modelcombine 的定义会让分析器
    认为一个方法就是完整实验，七方法对比根本无从证明。
    """
    if sorted(methods) != sorted(available_methods()):
        raise FreezeError(
            f"正式冻结必须覆盖全部已注册方法 {sorted(available_methods())}，"
            f"收到 {sorted(methods)}"
        )
    if sorted(datasets) != sorted(APPROVED_DATASETS):
        raise FreezeError(f"数据集必须恰好是 {list(APPROVED_DATASETS)}，收到 {sorted(datasets)}")
    if sorted(int(s) for s in forecast_steps) != sorted(SUPPORTED_FORECAST_STEPS):
        raise FreezeError(
            f"预测长度必须恰好是 {list(SUPPORTED_FORECAST_STEPS)}，收到 {sorted(forecast_steps)}"
        )


def build_formal_definition(**kwargs) -> Dict[str, Any]:
    """**正式冻结的唯一入口。**

    依次做四件事：强制正式范围 → 取实际 HEAD → 要求受跟踪文件干净 → 把**实际 HEAD**
    交给底层构造函数。

    冻结侧也必须拒绝脏工作树：否则可以用改过的代码生成定义，却把干净的 HEAD 写进
    ``repo_commit``；之后在那个干净提交上运行时，运行侧只看到 HEAD 相符，无法证明定义
    确实由该提交生成。

    ``repo_commit`` 不接受调用方提供——定义的来源只能是实际 HEAD，否则可以伪造。
    **这里没有 bypass 参数，也不要加。**
    """
    if "repo_commit" in kwargs:
        raise FreezeError(
            "正式冻结入口不接受调用方提供的 repo_commit：定义的来源必须是实际 HEAD"
        )
    assert_formal_scope(
        kwargs["methods"], kwargs["datasets"], kwargs["forecast_steps"]
    )
    try:
        head, dirty = repo_state()
    except FinalComparisonError as exc:
        raise FreezeError(str(exc)) from exc
    if dirty:
        raise FreezeError(
            "主仓库受跟踪文件有未提交修改，冻结出的定义无法对应任何一个提交：\n  "
            + "\n  ".join(dirty[:20])
        )
    return build_definition(**kwargs, repo_commit=head)


def build_definition(
    *,
    raw_root: Path, window_plan_path: Path, datasets: Sequence[str],
    methods: Sequence[str], forecast_steps: Sequence[int], database: Path | None,
    external_config: Mapping[str, Any] | None = None,
    candidates: Sequence[str] = (),
    repo_commit: str | None = None,
    pipeline_config: Path = PROJECT_ROOT / "configs" / "pipeline.yaml",
    library_report: Path | None = None,
) -> Dict[str, Any]:
    """底层构造函数：校验库完整性与候选一致性，产出定义内容。

    **不是正式冻结入口**——它不强制正式范围，可以用小规模装置做单元测试。正式冻结必须走
    :func:`build_formal_definition`。
    """
    unknown = [m for m in methods if m not in available_methods()]
    if unknown:
        raise FreezeError(
            f"方法未在统一入口注册: {unknown}；已注册 {available_methods()}"
        )
    approved_candidates = list(load_pipeline_model_params(pipeline_config))
    if sorted(candidates) != sorted(approved_candidates):
        raise FreezeError(
            f"候选池必须与 {pipeline_config} 的 models: 段一致 {approved_candidates}，"
            f"收到 {sorted(candidates)}"
        )
    if database is None:
        raise FreezeError("正式冻结必须提供 --database：modelcombine 与 stack 都要用它")
    if library_report is None:
        raise FreezeError(
            "正式冻结必须提供 --library-report：时间戳策略与 27 任务矩阵都记在建库报告里，"
            "没有它就没法证明这个库是用当前口径建的"
        )
    try:
        library = assert_library_complete(
            database, datasets=datasets, forecast_steps=forecast_steps,
            candidates=approved_candidates, base_horizon=MODEL_LIBRARY_BASE_HORIZON,
            timestamp_policy=TIMESTAMP_POLICY, library_report=library_report,
        )
    except LibraryIncomplete as exc:
        raise FreezeError(str(exc)) from exc
    external = _frozen_external(methods, external_config)
    plan = json.loads(window_plan_path.read_text(encoding="utf-8"))
    definitions = [
        _dataset_definition(raw_root, plan, dataset) for dataset in datasets
    ]
    # 官方内部硬编码种子的方法（iTransformer 的 fix_seed=2023），结果表记的就是那个值，
    # 定义里必须记同一个值，否则完整性核对必然判"种子集合不符"。
    seeds = {
        method: (
            [OFFICIAL_FIXED_SEED[method]] if method in OFFICIAL_FIXED_SEED
            else list(DEEP_METHOD_SEEDS if method in DEEP_METHODS
                      else DETERMINISTIC_METHOD_SEEDS)
        )
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
        #: 正式运行时逐项比对的口径。路径按绝对路径记，避免相对路径在不同 cwd 下"看着相同"。
        "repo_commit": repo_commit if repo_commit is not None else _repo_commit(),
        "window_plan": str(Path(window_plan_path).resolve()),
        "raw_root": str(Path(raw_root).resolve()),
        "database": None if database is None else str(Path(database).resolve()),
        "candidates": sorted(candidates),
        #: 原始序列里重复时刻的处理策略。建库、冻结、正式运行必须是同一个。
        "timestamp_policy": TIMESTAMP_POLICY,
        "library_report": None if library_report is None else str(Path(library_report).resolve()),
        "library_preflight": library,
        "methods": list(methods),
        "seeds": seeds,
        "seed_insensitive": seed_notes,
        "official_fixed_seeds": {
            method: OFFICIAL_FIXED_SEED[method]
            for method in methods if method in OFFICIAL_FIXED_SEED
        },
        #: 必须随结果一起报告的方法级限制（如 Time-MoE 的预训练截止不可证）
        "method_limitations": limitations,
        #: 外部官方实现的正式口径。运行时由 final_comparison.py --definition 强制使用。
        "external": external,
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
    parser.add_argument("--external-config", type=Path, default=None,
                        help="外部官方实现的配置 JSON；跑 itransformer/mole/time_moe 时必需，"
                             "其中的超参数、commit 与 checkpoint_id 会被写进定义")
    parser.add_argument("--library-report", type=Path, required=True,
                        help="建库产出的 model_library_report.json；核对时间戳策略与 27 任务矩阵")
    parser.add_argument("--pipeline-config", type=Path,
                        default=PROJECT_ROOT / "configs" / "pipeline.yaml",
                        help="候选池的唯一真源")
    parser.add_argument("--candidates", nargs="+", default=[],
                        help="冻结候选池；stack_ensembles_reproduction 与建库共用同一批")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    if out.exists():
        print(f"[freeze] {out} 已存在：这一批的定义只能冻结一次，不覆盖。")
        return 1
    try:
        definition = build_formal_definition(
            raw_root=args.raw_root, window_plan_path=args.window_plan,
            datasets=args.datasets, methods=args.methods,
            forecast_steps=args.forecast_steps, database=args.database,
            external_config=(
                json.loads(args.external_config.read_text(encoding="utf-8"))
                if args.external_config else None
            ),
            candidates=args.candidates,
            pipeline_config=args.pipeline_config,
            library_report=args.library_report,
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
