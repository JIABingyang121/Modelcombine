#!/usr/bin/env python3
"""V3 盲测阶段：只读共享池 T 窗口预测，生成全部方法的 T 预测并做验收。

- 不读取任何 T 真值；真值只在 ``v3_report.py`` 解封。
- 方法：有效池单模型、等权平均、Stacking、MoLE Router、Modelcombine。
- 验收（设计 §8）：共享列与数值一致、预测完整有限非负、Modelcombine 记忆均早于
  对应 T 起点、生成预测时未读取 T 真值。
"""
from __future__ import annotations

import os

# 门控推理为纯 CPU 计算。必须在**任何项目模块导入之前**（v3_fit_history →
# v3_shared_pool → model_registry → deep_learning 会间接导入 torch）隐藏 CUDA，
# 避免服务器上 cu130 torch 与驱动不匹配导致 accelerator 提前初始化失败。
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

from scripts.v3_fit_history import (
    rank_memory_records,
    router_features,
    scenario_profile,
)
from scripts.v3_shared_pool import SharedPoolError, expected_task_grid, load_copy

OUTPUT_COLUMNS = ("method", "dataset", "window_label", "forecast_steps", "timestamp", "yhat")


class BlindPredictError(RuntimeError):
    """盲测阶段不完整：任务取不出、方法产不出完整预测等。"""


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def t_tasks(shared: pd.DataFrame, pool: Sequence[str]) -> List[Dict[str, Any]]:
    tasks = []
    test = shared[shared["role"] == "test"]
    for (dataset, label, horizon), group in test.groupby(
        ["dataset", "window_label", "forecast_steps"]
    ):
        pivot = group.pivot_table(
            index="timestamp", columns="model", values="yhat", aggfunc="first"
        ).sort_index()
        missing = [m for m in pool if m not in pivot.columns]
        if missing:
            raise BlindPredictError(f"{dataset}/{label}/h{horizon} 缺少模型列: {missing}")
        tasks.append(
            {
                "dataset": dataset,
                "window_label": label,
                "horizon": int(horizon),
                "origin": pivot.index.min() - pd.Timedelta(hours=1),
                "timestamps": pivot.index,
                "yhat": pivot[list(pool)].to_numpy(dtype=float),
            }
        )
    return tasks


def predict_stacking(
    task: Mapping[str, Any], stacking: Mapping[str, Any], pool: Sequence[str]
) -> np.ndarray:
    entry = stacking[task["dataset"]][str(task["horizon"])]
    weights = np.array([entry["weights"][m] for m in pool], dtype=float)
    return task["yhat"] @ weights + float(entry["intercept"])


def predict_router(
    task: Mapping[str, Any],
    meta: Mapping[str, Any],
    state: Mapping[str, Any],
    pool: Sequence[str],
) -> np.ndarray:
    import torch
    from torch import nn

    dataset = task["dataset"]
    info = meta["datasets"][dataset]
    features = router_features(task["yhat"], task["timestamps"], task["horizon"])
    mean = np.asarray(info["scaler_mean"], dtype=float)
    std = np.asarray(info["scaler_std"], dtype=float)
    x = torch.tensor((features - mean) / std, dtype=torch.float32)
    router = nn.Sequential(nn.Linear(x.shape[1], len(pool)))
    router.load_state_dict(state[dataset])
    router.eval()
    with torch.no_grad():
        weights = torch.softmax(router(x), dim=1).numpy()
    return (weights * task["yhat"]).sum(axis=1)


def predict_modelcombine(
    task: Mapping[str, Any],
    memory: Mapping[str, Any],
    profile: Sequence[float],
    pool: Sequence[str],
) -> Dict[str, Any]:
    ranked = rank_memory_records(
        memory["records"],
        profile,
        dataset=task["dataset"],
        horizon=task["horizon"],
        before_origin=task["origin"],
        inclusive=False,
    )
    if not ranked:
        raise BlindPredictError(
            f"{task['dataset']}/{task['window_label']}/h{task['horizon']} 没有更早的记忆记录"
        )
    record = ranked[0]
    index = {m: i for i, m in enumerate(pool)}
    weights = np.zeros(len(pool), dtype=float)
    for member, weight in record["weights"].items():
        if member not in index:
            raise BlindPredictError(
                f"记忆成员 {member} 不在有效池中: {record['dataset']}/{record['window_label']}"
            )
        weights[index[member]] = float(weight)
    return {
        "prediction": task["yhat"] @ weights,
        "retrieved": {
            "dataset": record["dataset"],
            "window_label": record["window_label"],
            "horizon": record["horizon"],
            "origin": record["origin"],
            "members": record["members"],
            "weights": record["weights"],
            "validation_wape": record["validation_wape"],
        },
    }


def build_rows(
    method: str, task: Mapping[str, Any], values: np.ndarray
) -> List[Dict[str, Any]]:
    values = np.asarray(values, dtype=float).ravel()
    if len(values) != len(task["timestamps"]):
        raise BlindPredictError(
            f"{method} {task['dataset']}/{task['window_label']}/h{task['horizon']}: "
            f"长度 {len(values)} != {len(task['timestamps'])}"
        )
    return [
        {
            "method": method,
            "dataset": task["dataset"],
            "window_label": task["window_label"],
            "forecast_steps": task["horizon"],
            "timestamp": ts,
            "yhat": float(value),
        }
        for ts, value in zip(task["timestamps"], values)
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="V3 盲测：全方法 T 预测 + 验收")
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--window-plan", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    definition = _load_json(args.definition)
    plan = _load_json(args.window_plan)
    shared = pd.read_csv(args.shared)
    shared["timestamp"] = pd.to_datetime(shared["timestamp"])
    validation = _load_json(args.fit_dir / "pool_validation.json")
    pool = list(validation["effective_pool"])
    memory = _load_json(args.fit_dir / "memory.json")
    stacking = _load_json(args.fit_dir / "stacking.json")
    router_meta = _load_json(args.fit_dir / "mole_router_meta.json")

    import torch

    router_state = torch.load(args.fit_dir / "mole_router.pt", weights_only=True)
    tasks = t_tasks(shared, pool)
    horizons = [int(s) for s in definition["horizons"]]
    expected = expected_task_grid(plan, horizons, "test")
    observed = [(t["dataset"], t["window_label"], t["horizon"]) for t in tasks]
    if sorted(observed) != sorted(expected):
        raise BlindPredictError(
            f"T 任务网格与窗口计划不符：缺 {sorted(set(expected) - set(observed))}，"
            f"多 {sorted(set(observed) - set(expected))}"
        )
    forbidden = {
        entry["dataset"]: [
            (
                pd.Timestamp(w["targets"][str(step)]["first_target"]),
                pd.Timestamp(w["targets"][str(step)]["last_target"]),
            )
            for w in entry["windows"]
            if w["role"] == "test"
            for step in horizons
        ]
        for entry in plan["datasets"]
    }
    copies = {
        entry["dataset"]: load_copy(args.data_root / entry["dataset"] / "load.csv")
        for entry in plan["datasets"]
    }

    rows: List[Dict[str, Any]] = []
    retrievals: List[Dict[str, Any]] = []
    truth_read_evidence: List[Dict[str, Any]] = []
    methods = [*pool, "equal_weight", "stacking", "mole_router", "modelcombine"]
    for task in tasks:
        copy = copies[task["dataset"]]
        accessed_start = task["origin"] - pd.Timedelta(hours=719)
        accessed = copy[
            (copy["timestamp"] >= accessed_start) & (copy["timestamp"] <= task["origin"])
        ]
        overlaps = int(
            sum(
                ((accessed["timestamp"] >= lo) & (accessed["timestamp"] <= hi)).sum()
                for lo, hi in forbidden[task["dataset"]]
            )
        )
        truth_read_evidence.append(
            {
                "dataset": task["dataset"],
                "window_label": task["window_label"],
                "horizon": task["horizon"],
                "accessed_start": str(accessed_start),
                "accessed_end": str(task["origin"]),
                "overlap_with_test_targets": overlaps,
            }
        )
        if overlaps:
            raise BlindPredictError(
                f"{task['dataset']}/{task['window_label']}: 场景画像读取了 T 目标区间"
            )
        profile = scenario_profile(copy, task["origin"])
        for i, model in enumerate(pool):
            rows.extend(build_rows(model, task, task["yhat"][:, i]))
        rows.extend(build_rows("equal_weight", task, task["yhat"].mean(axis=1)))
        rows.extend(build_rows("stacking", task, predict_stacking(task, stacking, pool)))
        rows.extend(
            build_rows(
                "mole_router", task, predict_router(task, router_meta, router_state, pool)
            )
        )
        combined = predict_modelcombine(task, memory, profile, pool)
        rows.extend(build_rows("modelcombine", task, combined["prediction"]))
        retrievals.append(
            {
                "dataset": task["dataset"],
                "window_label": task["window_label"],
                "horizon": task["horizon"],
                "test_origin": str(task["origin"]),
                **combined["retrieved"],
            }
        )

    frame = pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(
        ["dataset", "window_label", "forecast_steps", "method", "timestamp"]
    ).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "predictions.csv", index=False)

    # 验收 1：单模型输出必须与共享表（T 窗口）逐值一致
    shared_test = shared[shared["role"] == "test"]
    singles_identical = True
    for model in pool:
        shared_model = shared_test[shared_test["model"] == model][
            ["dataset", "window_label", "forecast_steps", "timestamp", "yhat"]
        ].rename(columns={"yhat": "shared_yhat"})
        output_model = frame[frame["method"] == model][
            ["dataset", "window_label", "forecast_steps", "timestamp", "yhat"]
        ].rename(columns={"yhat": "output_yhat"})
        merged = output_model.merge(
            shared_model, on=["dataset", "window_label", "forecast_steps", "timestamp"],
            how="outer", indicator=True,
        )
        if len(merged) != len(output_model) or (merged["_merge"] != "both").any():
            singles_identical = False
            break
        if not np.array_equal(
            merged["output_yhat"].to_numpy(), merged["shared_yhat"].to_numpy()
        ):
            singles_identical = False
            break

    # 验收 2：完整、有限、非负
    problems: List[str] = []
    for (dataset, label, horizon, method), group in frame.groupby(
        ["dataset", "window_label", "forecast_steps", "method"]
    ):
        if len(group) != int(horizon):
            problems.append(f"{dataset}/{label}/h{horizon}/{method}: rows={len(group)}")
        elif not np.isfinite(group["yhat"]).all():
            problems.append(f"{dataset}/{label}/h{horizon}/{method}: non_finite")
        elif (group["yhat"] < 0).any():
            problems.append(f"{dataset}/{label}/h{horizon}/{method}: negative")

    # 验收 3：记忆记录必须早于对应 T 起点
    memory_before = all(
        pd.Timestamp(r["origin"]) < pd.Timestamp(r["test_origin"]) for r in retrievals
    )

    acceptance = {
        "shared_table": str(args.shared),
        "effective_pool": pool,
        "methods": methods,
        "checks": {
            "shared_columns_identical": {
                "passed": bool(singles_identical),
                "detail": "全部单模型输出与共享表逐值相等（含行集合）",
            },
            "complete_finite_nonnegative": {
                "passed": not problems,
                "problems": problems[:20],
            },
            "memory_before_test_origin": {
                "passed": bool(memory_before),
                "retrievals": retrievals,
            },
            "no_truth_read": {
                "passed": all(
                    e["overlap_with_test_targets"] == 0 for e in truth_read_evidence
                ),
                "detail": "T 场景画像只读各自起点前 720 小时观测；与全部 T 目标区间零相交",
                "evidence": truth_read_evidence,
            },
        },
    }
    acceptance["passed"] = all(c["passed"] for c in acceptance["checks"].values())
    (args.out_dir / "acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[v3-blind] {len(frame)} 行；验收 {'通过' if acceptance['passed'] else '未通过'}"
    )
    if not acceptance["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (BlindPredictError, SharedPoolError) as exc:
        print(f"[v3-blind] 失败: {exc}", file=sys.stderr)
        sys.exit(1)
