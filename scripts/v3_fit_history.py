#!/usr/bin/env python3
"""V3 历史阶段：用 H 窗口建立 Modelcombine 记忆、冠军单模型、静态与动态组合器。

只读 H 窗口真值（来自数据副本），不读取任何 T 真值。产物：

- ``memory.json``           每 (数据集, H 窗口, 长度) 的最佳方案：成员、权重、
                            K 折验证 WAPE、场景特征与日历。
- ``champions.json``        每数据集 H 宏平均 WAPE 最低的单模型。
- ``stacking.json``         每 (数据集, 长度) 在 H 上学习并冻结的线性权重。
- ``mole_router.pt`` / ``mole_router_meta.json``
                            每数据集的 MoLE 风格线性门控网络（共享池预测 +
                            场景特征 → 7 模型权重）。
- ``nn_check.json``         记忆读回自检（每条记录必须取回自己）与盲测覆盖检查。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v3_shared_pool import POOL, SharedPoolError, load_copy

CV_FOLDS = 4
SCENARIO_POINTS = 120
ROUTER_SEED = 42
ROUTER_EPOCHS = 300
ROUTER_LR = 1e-3


class FitHistoryError(RuntimeError):
    """历史阶段不完整：任务取不出、记忆读回不一致等。"""


def validate_pool(shared: pd.DataFrame, pool: Sequence[str]) -> Dict[str, Any]:
    """设计 §2：H 预测必须完整、有限且非负，否则该模型从共享池统一移除。"""
    history = shared[shared["role"] == "history"]
    problems: Dict[str, List[str]] = {}
    for model in pool:
        sub = history[history["model"] == model]
        if sub.empty:
            problems[model] = ["missing: H 窗口没有任何预测"]
            continue
        bad: List[str] = []
        for (dataset, label, horizon), group in sub.groupby(
            ["dataset", "window_label", "forecast_steps"]
        ):
            rows = group.sort_values("timestamp")
            reason = None
            if len(rows) != int(horizon):
                reason = f"rows={len(rows)}"
            elif rows["timestamp"].duplicated().any():
                reason = "duplicate_timestamps"
            elif not np.isfinite(rows["yhat"]).all():
                reason = "non_finite"
            elif (rows["yhat"] < 0).any():
                reason = "negative"
            if reason:
                bad.append(f"{dataset}/{label}/h{horizon}: {reason}")
        if bad:
            problems[model] = bad
    effective = [m for m in pool if m not in problems]
    return {
        "declared_pool": list(pool),
        "effective_pool": effective,
        "removed": {m: problems[m][:5] for m in problems},
        "passed": bool(effective),
    }


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_shared(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def h_tasks(shared: pd.DataFrame, pool: Sequence[str]) -> List[Dict[str, Any]]:
    tasks = []
    history = shared[shared["role"] == "history"]
    for (dataset, label, horizon), group in history.groupby(
        ["dataset", "window_label", "forecast_steps"]
    ):
        pivot = group.pivot_table(
            index="timestamp", columns="model", values="yhat", aggfunc="first"
        ).sort_index()
        missing = [m for m in pool if m not in pivot.columns]
        if missing:
            raise FitHistoryError(f"{dataset}/{label}/h{horizon} 缺少模型列: {missing}")
        tasks.append(
            {
                "dataset": dataset,
                "window_label": label,
                "horizon": int(horizon),
                "origin": str(group["timestamp"].min() - pd.Timedelta(hours=1)),
                "timestamps": pivot.index,
                "yhat": pivot[list(pool)].to_numpy(dtype=float),
            }
        )
    return tasks


def scenario_profile(copy: pd.DataFrame, origin: str, points: int = SCENARIO_POINTS) -> List[float]:
    origin_ts = pd.Timestamp(origin)
    start = origin_ts - pd.Timedelta(hours=719)
    window = copy[(copy["timestamp"] >= start) & (copy["timestamp"] <= origin_ts)]
    if len(window) != 720:
        raise FitHistoryError(f"场景 {origin} 历史不足 720 小时（{len(window)}）")
    values = window["load"].to_numpy(dtype=float)
    grid = np.linspace(0.0, 1.0, points)
    resampled = np.interp(grid, np.linspace(0.0, 1.0, len(values)), values)
    std = float(resampled.std())
    if std <= 0:
        raise FitHistoryError(f"场景 {origin} 历史为常数，无法构造相似度特征")
    return ((resampled - resampled.mean()) / std).tolist()


def _fold_indices(n: int, folds: int = CV_FOLDS) -> List[np.ndarray]:
    return [idx for idx in np.array_split(np.arange(n), folds) if len(idx) > 0]


def _fit_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    weights, *_ = np.linalg.lstsq(x, y, rcond=None)
    return weights


def _cv_wape(x: np.ndarray, y: np.ndarray) -> float:
    pooled = np.empty_like(y)
    for fold in _fold_indices(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[fold] = False
        weights = _fit_weights(x[mask], y[mask])
        pooled[fold] = x[fold] @ weights
    return float(np.sum(np.abs(pooled - y)) / np.sum(np.abs(y)))


def _wape(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(np.abs(pred - y)) / np.sum(np.abs(y)))


def evaluate_candidates(
    yhat: np.ndarray, truth: np.ndarray, pool: Sequence[str]
) -> Tuple[List[str], Dict[str, float], float, float]:
    """枚举全部非空子集；单模型取原值（权重 1），多模型用最小二乘权重。"""
    n_models = yhat.shape[1]
    best: Optional[Tuple[float, Tuple[int, ...]]] = None
    for size in range(1, n_models + 1):
        for subset in combinations(range(n_models), size):
            columns = list(subset)
            x = yhat[:, columns]
            if size == 1:
                validation = _wape(x[:, 0], truth)
            else:
                validation = _cv_wape(x, truth)
            if best is None or validation < best[0]:
                best = (validation, subset)
    assert best is not None
    validation_wape, subset = best
    columns = list(subset)
    members = [pool[i] for i in columns]
    if len(columns) == 1:
        weights = {members[0]: 1.0}
        insample = validation_wape
    else:
        fitted = _fit_weights(yhat[:, columns], truth)
        weights = {m: float(w) for m, w in zip(members, fitted)}
        insample = _wape(yhat[:, columns] @ fitted, truth)
    return members, weights, float(validation_wape), float(insample)


def compute_champions(
    tasks: Sequence[Mapping[str, Any]], pool: Sequence[str]
) -> Dict[str, Any]:
    per_dataset: Dict[str, Dict[str, List[float]]] = {}
    for task in tasks:
        scores = per_dataset.setdefault(task["dataset"], {m: [] for m in pool})
        for i, model in enumerate(pool):
            scores[model].append(_wape(task["yhat"][:, i], task["truth"]))
    champions = {}
    for dataset, scores in per_dataset.items():
        macro = {m: float(np.mean(v)) for m, v in scores.items()}
        best = min(macro, key=lambda m: macro[m])
        champions[dataset] = {
            "model": best,
            "macro_wape": macro[best],
            "macro_wape_by_model": {m: round(v, 6) for m, v in macro.items()},
            "h_tasks": len(next(iter(scores.values()))),
        }
    return champions


def fit_stacking(
    tasks: Sequence[Mapping[str, Any]], pool: Sequence[str]
) -> Dict[str, Any]:
    by_key: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for task in tasks:
        by_key.setdefault((task["dataset"], task["horizon"]), []).append(task)
    stacking: Dict[str, Any] = {}
    for (dataset, horizon), group in by_key.items():
        x = np.vstack([t["yhat"] for t in group])
        y = np.concatenate([t["truth"] for t in group])
        design = np.hstack([x, np.ones((len(y), 1))])
        weights, *_ = np.linalg.lstsq(design, y, rcond=None)
        stacking.setdefault(dataset, {})[str(horizon)] = {
            "weights": {m: float(w) for m, w in zip(pool, weights[:-1])},
            "intercept": float(weights[-1]),
            "train_points": int(len(y)),
        }
    return stacking


def router_features(yhat: np.ndarray, timestamps: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    lead = np.arange(1, len(timestamps) + 1, dtype=float) / 720.0
    hour = timestamps.hour.to_numpy(dtype=float)
    dow = timestamps.dayofweek.to_numpy(dtype=float)
    return np.column_stack(
        [
            yhat,
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            lead,
        ]
    )


def train_routers(
    tasks: Sequence[Mapping[str, Any]], pool: Sequence[str], out_dir: Path
) -> Dict[str, Any]:
    import torch
    from torch import nn

    torch.manual_seed(ROUTER_SEED)
    meta: Dict[str, Any] = {"feature_names": [f"pred_{m}" for m in pool] + [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "lead_scaled",
    ], "epochs": ROUTER_EPOCHS, "learning_rate": ROUTER_LR, "seed": ROUTER_SEED, "datasets": {}}
    state: Dict[str, Any] = {}
    by_dataset: Dict[str, List[Mapping[str, Any]]] = {}
    for task in tasks:
        by_dataset.setdefault(task["dataset"], []).append(task)
    for dataset, group in by_dataset.items():
        features = np.vstack([
            router_features(t["yhat"], t["timestamps"], t["horizon"]) for t in group
        ])
        truth = np.concatenate([t["truth"] for t in group])
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1.0
        x = torch.tensor((features - mean) / std, dtype=torch.float32)
        y = torch.tensor(truth, dtype=torch.float32)
        predictions = torch.tensor(features[:, : len(pool)], dtype=torch.float32)
        router = nn.Sequential(nn.Linear(x.shape[1], len(pool)))
        optimizer = torch.optim.Adam(router.parameters(), lr=ROUTER_LR)
        for _epoch in range(ROUTER_EPOCHS):
            optimizer.zero_grad()
            weights = torch.softmax(router(x), dim=1)
            combined = (weights * predictions).sum(dim=1)
            loss = torch.mean((combined - y) ** 2)
            loss.backward()
            optimizer.step()
        state[dataset] = router.state_dict()
        meta["datasets"][dataset] = {
            "scaler_mean": mean.tolist(),
            "scaler_std": std.tolist(),
            "train_points": int(len(y)),
            "final_mse": float(loss.item()),
        }
    torch.save(state, out_dir / "mole_router.pt")
    (out_dir / "mole_router_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def _similarity(profile: np.ndarray, other: Sequence[float]) -> float:
    other_array = np.asarray(other, dtype=float)
    return float(
        np.dot(profile, other_array)
        / (np.linalg.norm(profile) * np.linalg.norm(other_array))
    )


def rank_memory_records(
    records: Sequence[Mapping[str, Any]],
    profile: Sequence[float],
    *,
    dataset: str,
    horizon: int,
    before_origin: pd.Timestamp,
    inclusive: bool,
) -> List[Mapping[str, Any]]:
    """按场景相似度检索记忆；平局时更近的记录优先（最近相似场景）。"""
    query = np.asarray(profile, dtype=float)
    candidates = [
        r
        for r in records
        if r["dataset"] == dataset
        and r["horizon"] == horizon
        and (
            pd.Timestamp(r["origin"]) <= before_origin
            if inclusive
            else pd.Timestamp(r["origin"]) < before_origin
        )
    ]
    return sorted(
        candidates,
        key=lambda r: (
            -_similarity(query, r["scenario_profile"]),
            -pd.Timestamp(r["origin"]).value,
        ),
    )


def memory_readback_check(
    records: Sequence[Mapping[str, Any]], t_queries: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    readback_failures = []
    for record in records:
        ranked = rank_memory_records(
            records,
            record["scenario_profile"],
            dataset=record["dataset"],
            horizon=record["horizon"],
            before_origin=pd.Timestamp(record["origin"]),
            inclusive=True,
        )
        top = ranked[0]
        if not (
            top["window_label"] == record["window_label"]
            and top["horizon"] == record["horizon"]
        ):
            readback_failures.append(
                {"expected": f"{record['dataset']}/{record['window_label']}/h{record['horizon']}",
                 "got": f"{top['dataset']}/{top['window_label']}/h{top['horizon']}"}
            )
    coverage_failures = []
    for query in t_queries:
        ranked = rank_memory_records(
            records,
            query["scenario_profile"],
            dataset=query["dataset"],
            horizon=query["horizon"],
            before_origin=pd.Timestamp(query["origin"]),
            inclusive=False,
        )
        if not ranked:
            coverage_failures.append(
                f"{query['dataset']}/{query['window_label']}/h{query['horizon']}"
            )
    return {
        "readback_total": len(records),
        "readback_passed": len(records) - len(readback_failures),
        "readback_failures": readback_failures[:10],
        "coverage_total": len(t_queries),
        "coverage_passed": len(t_queries) - len(coverage_failures),
        "coverage_failures": coverage_failures[:10],
        "passed": not readback_failures and not coverage_failures,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="V3 历史阶段：记忆 + 冠军 + 组合器")
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--window-plan", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    definition = _load_json(args.definition)
    plan = _load_json(args.window_plan)
    declared_pool = list(definition["pool"])
    shared = load_shared(args.shared)
    pool_validation = validate_pool(shared, declared_pool)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pool_validation.json").write_text(
        json.dumps(pool_validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not pool_validation["passed"]:
        print("[v3-fit] 共享池校验后没有可用模型", file=sys.stderr)
        return 1
    if pool_validation["removed"]:
        print(f"[v3-fit] 移除不合格模型: {sorted(pool_validation['removed'])}")
    pool = pool_validation["effective_pool"]
    tasks = h_tasks(shared, pool)

    copies = {
        entry["dataset"]: load_copy(args.data_root / entry["dataset"] / "load.csv")
        for entry in plan["datasets"]
    }
    for task in tasks:
        copy = copies[task["dataset"]]
        truth = copy.set_index("timestamp")["load"].reindex(task["timestamps"])
        if truth.isna().any():
            raise FitHistoryError(
                f"{task['dataset']}/{task['window_label']}/h{task['horizon']} 真值缺失"
            )
        task["truth"] = truth.to_numpy(dtype=float)

    records = []
    for task in tasks:
        members, weights, validation_wape, insample_wape = evaluate_candidates(
            task["yhat"], task["truth"], pool
        )
        origin = task["origin"]
        records.append(
            {
                "dataset": task["dataset"],
                "window_label": task["window_label"],
                "horizon": task["horizon"],
                "origin": origin,
                "members": members,
                "weights": weights,
                "validation_wape": round(validation_wape, 6),
                "insample_wape": round(insample_wape, 6),
                "scenario_profile": scenario_profile(copies[task["dataset"]], origin),
                "calendar": {
                    "month": int(pd.Timestamp(origin).month),
                    "dayofweek": int(pd.Timestamp(origin).dayofweek),
                },
            }
        )
    memory = {
        "experiment": definition["experiment"],
        "pool": pool,
        "cv_folds": CV_FOLDS,
        "scenario_points": SCENARIO_POINTS,
        "candidates": "全部非空子集（单模型原值，多模型最小二乘权重，K 折验证选择）",
        "records": records,
    }

    t_queries = []
    test = shared[shared["role"] == "test"]
    for (dataset, label, horizon), group in test.groupby(
        ["dataset", "window_label", "forecast_steps"]
    ):
        origin = str(group["timestamp"].min() - pd.Timedelta(hours=1))
        t_queries.append(
            {
                "dataset": dataset,
                "window_label": label,
                "horizon": int(horizon),
                "origin": origin,
                "scenario_profile": scenario_profile(copies[dataset], origin),
            }
        )

    champions = compute_champions(tasks, pool)
    stacking = fit_stacking(tasks, pool)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "memory.json").write_text(
        json.dumps(memory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "champions.json").write_text(
        json.dumps(champions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "stacking.json").write_text(
        json.dumps(stacking, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    train_routers(tasks, pool, args.out_dir)
    check = memory_readback_check(records, t_queries)
    (args.out_dir / "nn_check.json").write_text(
        json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[v3-fit] 记忆 {len(records)} 条；读回 {check['readback_passed']}/"
        f"{check['readback_total']}，盲测覆盖 {check['coverage_passed']}/"
        f"{check['coverage_total']}"
    )
    if not check["passed"]:
        print("[v3-fit] 记忆自检未通过，不能进入 T", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FitHistoryError, SharedPoolError) as exc:
        print(f"[v3-fit] 失败: {exc}", file=sys.stderr)
        sys.exit(1)
