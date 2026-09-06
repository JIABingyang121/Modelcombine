#!/usr/bin/env python3
"""Stage 2：三基线质量门控（方案 §7.1 基线、§9.1 指标、§11.2 第 1—3 条门槛）。

在 Q1—Q3 三个未见查询窗口上，把 Modelcombine 的检索—重放结果与三个基础基线
逐点对比：

```text
Modelcombine            检索 H1—H3 已存关系并重放（run.py predict 同一条代码路径）
Seasonal Naive (168)    重复查询窗口自己历史的最近 168 小时
Validation Best Single  只用 H1—H3 validation 选出的最优单模型，不看 Q
Ridge Stacking          所有合格候选的固定 Ridge 融合，无场景检索、无可变成员
```

**冻结声明**：本文件中的指标定义、基线定义、Ridge 超参与门槛数值都是预注册常量，
在查看 Q1—Q3 真值之前写死。Stage 2 跑完之后不得回头调整这些常量、再在同一批 Q
窗口上重新宣称通过（§12 Stage 4 的停止规则）。

**任务集合必须恰好等于 3×3**：允许只跑一部分、或额外跑别的数据集/长度做诊断，但
覆盖度规则要求任务集合与 9 个核心格子**完全相等**——缺格子不行，多格子同样不行。
多出来的任务会混进同一长度的等权平均，把本应只由 PJM/VIC/NSW 三者决定的指标改掉，
所以逐长度的比值规则也要求该长度的数据集集合**恰好**是这三个、且该长度本身是核心
长度。不存在用子集或超集拿到 passed=true 的路径。

退出码：0=全部门槛通过；2=运行完整但门槛未通过（按 §12 停止，不进 Stage 3）；
1=运行不完整（数据缺失、窗口产不出轨迹等），此时不产出结论。
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_combinations_kg import (
    MODEL_LIBRARY_BASE_HORIZON,
    MODEL_LIBRARY_BUSINESS_DOMAIN,
    MODEL_LIBRARY_COUNTRY_BY_REGION,
    MODEL_LIBRARY_TASK_TYPE,
    _frozen_windows,
    _library_candidate_models,
    _library_raw_frame,
)
from src.models.artifacts import load_artifact
from src.models.trajectory_forecast import (
    TrajectoryForecastError,
    calendar_frame,
    generate_member_trajectory,
    generate_trajectory_matrix,
)
from src.storage.model_store import SUPPORTED_FORECAST_STEPS, ModelStore

# ---------------------------------------------------------------- 预注册常量
#: §6.2 的窗口角色。H1—H3 是 validation（基线在这里选定/拟合），Q1—Q3 是未见查询。
#: A 是建库阶段的审计窗口，Stage 2 完全不碰。
LIBRARY_WINDOWS: Tuple[str, ...] = ("H1", "H2", "H3")
QUERY_WINDOWS: Tuple[str, ...] = ("Q1", "Q2", "Q3")

#: §9.1：MASE 分母用训练段的 mean(|y_t - y_{t-168}|)，不用 test 上 Seasonal Naive
#: 的 MAE 临时充当分母。"训练段"在这里冻结为：原始序列中严格早于 H1 历史窗口起点
#: 的全部数据——它与任何评价窗口、任何窗口的输入历史都不重叠。
MASE_SEASONAL_PERIOD = 168

#: §7.1 Ridge Stacking：固定融合，不做场景检索也不选成员子集。超参在看 Q 之前写死。
#:
#: "所有合格候选"在这里冻结为**输入契约下合格的全部候选**（已登记、产物存在、能由
#: (timestamp, load) 派生出完整轨迹），而不是建库里再经 filter_weak_models 得到的
#: safe_models。理由：§7.1 的措辞是"所有…固定…不用可变成员子集"，其对照物正是
#: Modelcombine 的按场景变成员；若改用逐窗口 safe_models 的交集，本装置上会退化成
#: 单模型（H2/H3 的 safe 只剩一个），那就不再是一个静态融合基线了。Ridge 的 L2
#: 本身会压低劣质列的权重，这正是静态 stacking 该做的事。
RIDGE_STACKING_ALPHA = 1.0
RIDGE_STACKING_FIT_INTERCEPT = True

#: §11.1.4：离线冻结轨迹与保存后在线重放必须逐值一致。
REPLAY_TOLERANCE = 1e-8

#: §5：核心任务是 3 数据集 × 3 预测长度 = 9 个任务。少一个格子，"等权平均"就不是
#: 方案定义的那个口径，任何"通过"结论都不成立。
REQUIRED_DATASETS: Tuple[str, ...] = ("pjm", "aemo_vic", "aemo_nsw")
REQUIRED_FORECAST_STEPS: Tuple[int, ...] = (24, 168, 720)

METHOD_MODELCOMBINE = "modelcombine"
BASELINE_SEASONAL_NAIVE = "seasonal_naive_168"
BASELINE_BEST_SINGLE = "validation_best_single"
BASELINE_RIDGE = "ridge_stacking"
METHODS: Tuple[str, ...] = (
    METHOD_MODELCOMBINE, BASELINE_SEASONAL_NAIVE, BASELINE_BEST_SINGLE, BASELINE_RIDGE,
)

#: §11.2 第 1—3 条。比值一律是 Modelcombine / 基线，越小越好。
#: "三数据集等权平均比值"冻结为**逐数据集比值的等权平均**，而不是先平均 MAE 再相除
#: ——三个区域负荷量级差一个数量级，先平均 MAE 会让 PJM 单独决定结论。
THRESHOLD_BEST_SINGLE_MEAN_RATIO = 1.00        # 严格小于
THRESHOLD_BEST_SINGLE_PER_DATASET_RATIO = 1.03  # 小于等于
THRESHOLD_SEASONAL_NAIVE_PER_TASK_RATIO = 1.00  # 严格小于（9 个任务逐个判定）
THRESHOLD_RIDGE_MEAN_RATIO = 1.00               # 小于等于

EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_GATE_FAILED = 2


class Stage2Error(RuntimeError):
    """运行不完整：数据缺失、窗口产不出轨迹、与建库口径不一致等。"""


# ------------------------------------------------------------------ 指标
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(diff * diff)))


def mase_scale(series: pd.Series) -> float:
    """训练段的 mean(|y_t - y_{t-168}|)。"""
    values = np.asarray(series, dtype=float)
    if len(values) <= MASE_SEASONAL_PERIOD:
        raise Stage2Error(
            f"MASE 训练段只有 {len(values)} 点，不足以计算 {MASE_SEASONAL_PERIOD} 步季节差分"
        )
    diffs = np.abs(values[MASE_SEASONAL_PERIOD:] - values[:-MASE_SEASONAL_PERIOD])
    scale = float(np.mean(diffs))
    if not np.isfinite(scale) or scale <= 0:
        raise Stage2Error(f"MASE 缩放分母非正或非有限: {scale}")
    return scale


# ------------------------------------------------------------------ 窗口切片
def _window_slice(
    raw: pd.DataFrame, window: Mapping[str, Any], forecast_steps: int, *, label: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """取一个冻结窗口的输入历史与其完整目标轨迹。"""
    history = raw[
        (raw["timestamp"] >= pd.Timestamp(window["history_start"]))
        & (raw["timestamp"] <= pd.Timestamp(window["forecast_origin"]))
    ].reset_index(drop=True)
    targets = raw[
        (raw["timestamp"] >= pd.Timestamp(window["first_target"]))
        & (raw["timestamp"] <= pd.Timestamp(window["last_target"]))
    ].reset_index(drop=True)
    if len(targets) != forecast_steps:
        raise Stage2Error(
            f"{label}: 目标区间只有 {len(targets)} 个点，与 forecast_steps={forecast_steps} 不符"
        )
    if history.empty:
        raise Stage2Error(f"{label}: 输入历史为空")
    return history, targets


def _candidate_matrix(
    members: Mapping[str, Dict[str, Any]],
    history: pd.DataFrame,
    *,
    forecast_steps: int,
    country: str,
    label: str,
    required_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """一个窗口上全部候选的轨迹矩阵。

    ``required_columns`` 给定时（Q 窗口），任何一个已在 validation 冻结的候选产不出
    轨迹都是不完整任务，直接失败——不静默丢列，否则基线的成员集合会被 Q 改变。
    """
    matrix, skipped = generate_trajectory_matrix(
        members=members, history=history, forecast_steps=forecast_steps, country=country
    )
    if required_columns is not None:
        missing = [c for c in required_columns if c not in matrix.columns]
        if missing:
            reasons = {entry["model_type"]: entry["reason"] for entry in skipped}
            raise Stage2Error(
                f"{label}: 已在 validation 冻结的候选 {missing} 在该窗口产不出轨迹"
                f"（{[reasons.get(m) for m in missing]}）"
            )
    return matrix


# ------------------------------------------------------------------ 冻结基线
def _freeze_baselines(
    members: Mapping[str, Dict[str, Any]],
    *,
    raw: pd.DataFrame,
    windows: Mapping[str, Mapping[str, Any]],
    forecast_steps: int,
    country: str,
    dataset: str,
) -> Dict[str, Any]:
    """只用 H1—H3 定下 Validation Best Single 与 Ridge Stacking，绝不看 Q。"""
    blocks: List[pd.DataFrame] = []
    targets: List[np.ndarray] = []
    per_window_mae: Dict[str, Dict[str, float]] = {}
    columns: Optional[List[str]] = None

    for label in LIBRARY_WINDOWS:
        history, target = _window_slice(
            raw, windows[label], forecast_steps, label=f"{dataset} {label}"
        )
        matrix = _candidate_matrix(
            members, history, forecast_steps=forecast_steps, country=country,
            label=f"{dataset} {label}",
        )
        available = sorted(c for c in matrix.columns if c in members)
        if not available:
            raise Stage2Error(f"{dataset} {label}: 没有任何候选能产出 validation 轨迹")
        columns = available if columns is None else [c for c in columns if c in available]
        y = target["load"].to_numpy(dtype=float)
        blocks.append(matrix)
        targets.append(y)
        per_window_mae[label] = {c: mae(y, matrix[c].to_numpy(float)) for c in available}

    if not columns:
        raise Stage2Error(f"{dataset}: H1—H3 没有共同的合格候选，无法冻结基线")

    stacked = np.vstack([block[columns].to_numpy(dtype=float) for block in blocks])
    y_stacked = np.concatenate(targets)

    # Validation Best Single：三个 H 窗口合并后的 MAE 最小者；完全相同按模型名确定性排序
    single_mae = {c: mae(y_stacked, stacked[:, index]) for index, c in enumerate(columns)}
    best_single = min(sorted(columns), key=lambda c: (single_mae[c], c))

    ridge = Ridge(alpha=RIDGE_STACKING_ALPHA, fit_intercept=RIDGE_STACKING_FIT_INTERCEPT)
    ridge.fit(stacked, y_stacked)

    return {
        "columns": columns,
        "validation_window_mae": per_window_mae,
        "validation_stacked_mae": single_mae,
        "best_single": best_single,
        "ridge": ridge,
        "ridge_coef": {c: float(w) for c, w in zip(columns, ridge.coef_)},
        "ridge_intercept": float(ridge.intercept_),
    }


def _cross_check_library_report(
    report: Mapping[str, Any],
    *,
    dataset: str,
    forecast_steps: int,
    declared_candidates: Sequence[str],
    validation_window_mae: Mapping[str, Mapping[str, float]],
    tolerance: float = REPLAY_TOLERANCE,
) -> None:
    """Stage 2 必须和建库是同一批：同样的三个历史窗口、同样的候选集合、同样的
    validation 轨迹。

    任何一项对不上都说明窗口计划、模型库或候选集合与建库时不一致，此时门槛结论没有
    意义，直接判运行不完整，而不是继续算。
    """
    tasks = [
        task for task in report.get("tasks", [])
        if task.get("dataset") == dataset
        and int(task.get("forecast_steps", -1)) == int(forecast_steps)
    ]
    if not tasks:
        raise Stage2Error(
            f"{dataset} forecast_steps={forecast_steps}: 建库报告里找不到对应任务，"
            "无法核对 Stage 2 与建库是否同一批 validation"
        )

    windows = [task.get("library_window") for task in tasks]
    if sorted(w for w in windows if w is not None) != sorted(LIBRARY_WINDOWS):
        raise Stage2Error(
            f"{dataset} forecast_steps={forecast_steps}: 建库报告的历史窗口是 {windows}，"
            f"要求恰好各一条 {list(LIBRARY_WINDOWS)}——关系不齐或有重复批次，不能用于门槛判定"
        )

    expected_candidates = sorted(set(declared_candidates))
    for task in tasks:
        recorded_candidates = sorted(set(task.get("declared_candidates") or []))
        if recorded_candidates != expected_candidates:
            raise Stage2Error(
                f"{dataset} {task['library_window']}: 建库声明的候选是 {recorded_candidates}，"
                f"Stage 2 传入的是 {expected_candidates}——基线与建库不是同一批候选"
            )
    for task in tasks:
        label = task["library_window"]
        if label not in validation_window_mae:
            raise Stage2Error(
                f"{dataset} {label}: Stage 2 没有重算这个历史窗口的 validation 轨迹"
            )
        recorded = task.get("candidate_validation_mae") or {}
        recomputed = validation_window_mae[label]
        for model_type, scores in recorded.items():
            if model_type not in recomputed:
                raise Stage2Error(
                    f"{dataset} {label}: 建库用过的候选 {model_type} 在 Stage 2 重算中缺失"
                )
            delta = abs(float(scores["trajectory_mae"]) - recomputed[model_type])
            if delta > tolerance:
                raise Stage2Error(
                    f"{dataset} {label}: 候选 {model_type} 的 validation 轨迹 MAE 与建库"
                    f"报告差 {delta:.3e} > {tolerance:g}，Stage 2 与建库不是同一批 validation"
                )


# ------------------------------------------------------------------ 在线检索
def _offline_replay(
    trace: Mapping[str, Any],
    matrix: pd.DataFrame,
    timestamps: Sequence[Any],
    country: str,
    *,
    label: str,
) -> np.ndarray:
    """用命中关系的已保存组合器，在本窗口独立重算一遍最终预测。

    §11.1.4 的门槛对象：这条轨迹与在线输出必须逐值一致。它独立地加载组合器产物、
    独立地取成员轨迹并按保存的权重融合，因此在线侧一旦用错成员、权重、成员顺序或
    interaction 的特征表，都会在这里暴露。
    """
    predictor = load_artifact(Path(trace["artifact_paths"]["combination"]))
    missing = [m for m in predictor.member_ids if m not in matrix.columns]
    if missing:
        raise Stage2Error(
            f"{label}: 命中关系的成员 {missing} 不在本窗口的候选轨迹矩阵里，无法做离线重放"
        )
    base = {m: matrix[m].to_numpy(dtype=float) for m in predictor.member_ids}
    return np.asarray(
        predictor.predict(base, calendar_frame(timestamps, country)), dtype=float
    )


def _modelcombine_prediction(
    *, database: Path, dataset: str, forecast_steps: int, history: pd.DataFrame, workdir: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """走 run.py predict 的同一条代码路径：只检索已存关系并重放。"""
    from src.pipeline.main import library_predict

    workdir.mkdir(parents=True, exist_ok=True)
    scenario_path = workdir / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            {
                "task_type": MODEL_LIBRARY_TASK_TYPE,
                "business_domain": MODEL_LIBRARY_BUSINESS_DOMAIN,
                "region": dataset,
                "freq": "h",
                "forecast_steps": int(forecast_steps),
            }
        ),
        encoding="utf-8",
    )
    history_path = workdir / "history.csv"
    history[["timestamp", "load"]].to_csv(history_path, index=False)
    output_path = workdir / "forecast.csv"
    trace = library_predict(
        database=str(database),
        scenario=str(scenario_path),
        features=None,
        history=str(history_path),
        output=str(output_path),
    )
    return pd.read_csv(output_path)["yhat"].to_numpy(dtype=float), trace


# ------------------------------------------------------------------ 单任务
def evaluate_task(
    *,
    store: ModelStore,
    database: Path,
    raw_root: Path,
    dataset: str,
    forecast_steps: int,
    windows: Mapping[str, Mapping[str, Any]],
    declared_candidates: Sequence[str],
    library_report: Mapping[str, Any],
    predictions_dir: Path,
    workdir: Path,
) -> Dict[str, Any]:
    country = MODEL_LIBRARY_COUNTRY_BY_REGION.get(dataset)
    if country is None:
        raise Stage2Error(f"未知数据集 {dataset}：无法确定日历特征所用节假日日历")
    raw = _library_raw_frame(raw_root, dataset)

    members, skipped = _library_candidate_models(
        store, dataset=dataset, base_horizon=MODEL_LIBRARY_BASE_HORIZON,
        model_types=declared_candidates,
    )

    # MASE 分母：严格早于 H1 输入历史起点的训练段
    train_cut = pd.Timestamp(windows[LIBRARY_WINDOWS[0]]["history_start"])
    train_segment = raw[raw["timestamp"] < train_cut]
    scale = mase_scale(train_segment["load"])

    frozen = _freeze_baselines(
        members, raw=raw, windows=windows, forecast_steps=forecast_steps,
        country=country, dataset=dataset,
    )
    _cross_check_library_report(
        library_report, dataset=dataset, forecast_steps=forecast_steps,
        declared_candidates=declared_candidates,
        validation_window_mae=frozen["validation_window_mae"],
    )

    columns = frozen["columns"]
    queries: List[Dict[str, Any]] = []
    for label in QUERY_WINDOWS:
        history, target = _window_slice(
            raw, windows[label], forecast_steps, label=f"{dataset} {label}"
        )
        y = target["load"].to_numpy(dtype=float)
        matrix = _candidate_matrix(
            members, history, forecast_steps=forecast_steps, country=country,
            label=f"{dataset} {label}", required_columns=columns,
        )
        design = matrix[columns].to_numpy(dtype=float)

        yhat_mc, trace = _modelcombine_prediction(
            database=database, dataset=dataset, forecast_steps=forecast_steps,
            history=history, workdir=workdir / f"{dataset}_s{forecast_steps}_{label}",
        )
        # seasonal_naive 的轨迹只由查询窗口自己的历史决定，不消费任何模型产物
        seasonal = generate_member_trajectory(
            model=None, model_type="seasonal_naive", required_features=[],
            history=history, forecast_steps=forecast_steps, country=country,
        )
        # §11.1.4：在线输出必须与离线重放逐值一致
        replay = _offline_replay(
            trace, matrix, target["timestamp"], country,
            label=f"{dataset} {label}",
        )
        replay_error = float(np.max(np.abs(replay - yhat_mc))) if len(replay) else float("inf")

        predictions = {
            METHOD_MODELCOMBINE: yhat_mc,
            BASELINE_SEASONAL_NAIVE: seasonal,
            BASELINE_BEST_SINGLE: matrix[frozen["best_single"]].to_numpy(dtype=float),
            BASELINE_RIDGE: np.asarray(frozen["ridge"].predict(design), dtype=float),
        }
        for method, values in predictions.items():
            if len(values) != forecast_steps or not np.isfinite(values).all():
                raise Stage2Error(
                    f"{dataset} {label} {method}: 预测长度 {len(values)} 或存在非有限值"
                )

        predictions_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": target["timestamp"], "y": y, **predictions}).to_csv(
            predictions_dir / f"{dataset}_s{forecast_steps}_{label}.csv", index=False
        )

        queries.append(
            {
                "window": label,
                "forecast_origin": str(windows[label]["forecast_origin"]),
                "first_target": str(target["timestamp"].iloc[0]),
                "last_target": str(target["timestamp"].iloc[-1]),
                "n_rows": int(len(y)),
                "trace": {
                    "scenario_id": trace["scenario_id"],
                    "relation_id": trace["relation_id"],
                    "combination_id": trace["combination_id"],
                    "selector_invoked": trace["selector_invoked"],
                    "forecast_steps": trace["forecast_steps"],
                    "n_rows": trace["n_rows"],
                    "member_types": trace["member_types"],
                },
                "replay_max_abs_error": replay_error,
                "metrics": {
                    method: {
                        "mae": mae(y, values),
                        "rmse": rmse(y, values),
                        "mase": mae(y, values) / scale,
                    }
                    for method, values in predictions.items()
                },
            }
        )

    task_metrics = {
        method: {
            metric: float(np.mean([q["metrics"][method][metric] for q in queries]))
            for metric in ("mae", "rmse", "mase")
        }
        for method in METHODS
    }
    return {
        "dataset": dataset,
        "forecast_steps": int(forecast_steps),
        "mase_scale": scale,
        "mase_train_segment": {
            "start": str(train_segment["timestamp"].iloc[0]) if len(train_segment) else None,
            "end": str(train_segment["timestamp"].iloc[-1]) if len(train_segment) else None,
            "rows": int(len(train_segment)),
        },
        "declared_candidates": list(declared_candidates),
        "ineligible_candidates": skipped,
        "frozen_baseline_columns": columns,
        "validation_best_single": frozen["best_single"],
        "validation_stacked_mae": frozen["validation_stacked_mae"],
        "ridge_coef": frozen["ridge_coef"],
        "ridge_intercept": frozen["ridge_intercept"],
        "queries": queries,
        "task_metrics": task_metrics,
    }


# ------------------------------------------------------------------ 门槛判定
def _ratio(task: Mapping[str, Any], baseline: str) -> float:
    return (
        task["task_metrics"][METHOD_MODELCOMBINE]["mae"]
        / task["task_metrics"][baseline]["mae"]
    )


def evaluate_gates(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """§11.2 第 1—3 条，逐 forecast_steps 判定；外加覆盖度与 §11.1 功能门槛。"""
    by_steps: Dict[int, List[Mapping[str, Any]]] = {}
    for task in tasks:
        by_steps.setdefault(int(task["forecast_steps"]), []).append(task)

    # 覆盖度：任务集合必须"恰好等于"3×3。缺格子不是方案口径；多格子会混进同一长度的
    # 等权平均，同样不是方案口径。判据同时看**记录条数**和**去重后的格子集合**：
    # 只比集合的话，同一个格子重复出现会被 set 折叠，10 条记录也能判成 9 格通过，
    # 而重复记录在逐长度的 {dataset: ratio} 里会静默覆盖掉原来那条。
    counts = Counter((t["dataset"], int(t["forecast_steps"])) for t in tasks)
    present = set(counts)
    expected = {(d, s) for d in REQUIRED_DATASETS for s in REQUIRED_FORECAST_STEPS}
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    duplicated = sorted(cell for cell, n in counts.items() if n > 1)
    rules: List[Dict[str, Any]] = [{
        "rule": "coverage",
        "description": "任务集合必须恰好等于 3 数据集 × 3 预测长度 = 9 个任务，"
                       "不多不少、不重复",
        "expected_tasks": len(expected), "present_tasks": len(present & expected),
        "task_records": len(tasks),
        "missing_tasks": [{"dataset": d, "forecast_steps": s} for d, s in missing],
        "extra_tasks": [{"dataset": d, "forecast_steps": s} for d, s in extra],
        "duplicate_tasks": [
            {"dataset": d, "forecast_steps": s, "records": counts[(d, s)]}
            for d, s in duplicated
        ],
        "passed": len(tasks) == len(expected) and present == expected,
    }]
    for steps in sorted(by_steps):
        group = by_steps[steps]
        bs_ratios = {t["dataset"]: _ratio(t, BASELINE_BEST_SINGLE) for t in group}
        bs_mean = float(np.mean(list(bs_ratios.values())))
        # 等权平均只有在"该长度的数据集恰好是这三个"时才是方案定义的口径：多一个
        # 数据集就会把它的比值一起平均进去，改掉本应由 PJM/VIC/NSW 决定的结论。
        # group 行数也要恰好是三条：重复任务会在 {dataset: ratio} 里静默覆盖，
        # 只看键集合看不出来。
        core_steps = steps in REQUIRED_FORECAST_STEPS
        complete = (
            core_steps
            and len(group) == len(REQUIRED_DATASETS)
            and set(bs_ratios) == set(REQUIRED_DATASETS)
        )
        rules.append({
            "rule": "11.2.1a", "forecast_steps": steps,
            "description": "相对 Validation Best Single：三数据集等权平均 MAE 比值 < 1.00",
            "value": bs_mean, "threshold": THRESHOLD_BEST_SINGLE_MEAN_RATIO,
            "comparison": "<", "per_dataset": bs_ratios, "datasets_complete": complete,
            "core_forecast_steps": core_steps,
            "passed": bool(complete and bs_mean < THRESHOLD_BEST_SINGLE_MEAN_RATIO),
        })
        worst_dataset = max(bs_ratios, key=bs_ratios.get) if bs_ratios else None
        rules.append({
            "rule": "11.2.1b", "forecast_steps": steps,
            "description": "相对 Validation Best Single：每个数据集 MAE 比值 <= 1.03",
            "value": bs_ratios[worst_dataset] if worst_dataset else None,
            "threshold": THRESHOLD_BEST_SINGLE_PER_DATASET_RATIO,
            "comparison": "<=", "worst_dataset": worst_dataset, "per_dataset": bs_ratios,
            "datasets_complete": complete,
            "core_forecast_steps": core_steps,
            "passed": bool(complete and all(
                r <= THRESHOLD_BEST_SINGLE_PER_DATASET_RATIO for r in bs_ratios.values()
            )),
        })
        sn_ratios = {t["dataset"]: _ratio(t, BASELINE_SEASONAL_NAIVE) for t in group}
        rules.append({
            "rule": "11.2.2", "forecast_steps": steps,
            "description": "相对 Seasonal Naive (168)：每个数据集×长度任务比值 < 1.00",
            "value": max(sn_ratios.values()) if sn_ratios else None,
            "threshold": THRESHOLD_SEASONAL_NAIVE_PER_TASK_RATIO,
            "comparison": "<", "per_dataset": sn_ratios, "datasets_complete": complete,
            "core_forecast_steps": core_steps,
            "passed": bool(complete and all(
                r < THRESHOLD_SEASONAL_NAIVE_PER_TASK_RATIO for r in sn_ratios.values()
            )),
        })
        ridge_ratios = {t["dataset"]: _ratio(t, BASELINE_RIDGE) for t in group}
        ridge_mean = float(np.mean(list(ridge_ratios.values())))
        rules.append({
            "rule": "11.2.3", "forecast_steps": steps,
            "description": "相对 Ridge Stacking：该长度的等权平均 MAE 比值 <= 1.00",
            "value": ridge_mean, "threshold": THRESHOLD_RIDGE_MEAN_RATIO,
            "comparison": "<=", "per_dataset": ridge_ratios, "datasets_complete": complete,
            "core_forecast_steps": core_steps,
            "passed": bool(complete and ridge_mean <= THRESHOLD_RIDGE_MEAN_RATIO),
        })

    functional = []
    for task in tasks:
        for query in task["queries"]:
            functional.append({
                "rule": "11.1.2/11.1.3", "dataset": task["dataset"],
                "forecast_steps": task["forecast_steps"], "window": query["window"],
                "description": "输出行数=请求长度=trace 长度，且 selector_invoked=false",
                "passed": bool(
                    query["n_rows"] == task["forecast_steps"]
                    and query["trace"]["forecast_steps"] == task["forecast_steps"]
                    and query["trace"]["n_rows"] == task["forecast_steps"]
                    and query["trace"]["selector_invoked"] is False
                ),
            })
            functional.append({
                "rule": "11.1.4", "dataset": task["dataset"],
                "forecast_steps": task["forecast_steps"], "window": query["window"],
                "description": "在线输出与离线冻结组合器重放逐值一致（<= 1e-8）",
                "value": query["replay_max_abs_error"], "threshold": REPLAY_TOLERANCE,
                "comparison": "<=",
                "passed": bool(query["replay_max_abs_error"] <= REPLAY_TOLERANCE),
            })
    all_rules = rules + functional
    return {
        "rules": all_rules,
        "passed": all(rule["passed"] for rule in all_rules),
        "failed_rules": [r for r in all_rules if not r["passed"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 三基线质量门控")
    parser.add_argument("--database", type=Path, required=True, help="V3 SQLite 模型库")
    parser.add_argument("--raw-root", type=Path, required=True, help="原始序列根目录")
    parser.add_argument("--window-plan", type=Path, required=True,
                        help="Stage 0 冻结的窗口计划 JSON")
    parser.add_argument("--library-report", type=Path, required=True,
                        help="建库报告 model_library_report.json（用于核对同一批 validation）")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--forecast-steps", nargs="+", type=int,
                        default=list(SUPPORTED_FORECAST_STEPS))
    parser.add_argument("--candidates", nargs="+", required=True,
                        help="本批声明的候选模型类型，必须与建库时一致")
    parser.add_argument("--out", type=Path, required=True, help="输出目录")
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    library_report = json.loads(args.library_report.read_text(encoding="utf-8"))

    definition = {
        "stage": "stage2_quality_gate",
        "database": str(args.database),
        "raw_root": str(args.raw_root),
        "window_plan": str(args.window_plan),
        "library_report": str(args.library_report),
        "datasets": list(args.datasets),
        "forecast_steps": list(args.forecast_steps),
        "declared_candidates": list(args.candidates),
        "library_windows": list(LIBRARY_WINDOWS),
        "query_windows": list(QUERY_WINDOWS),
        "methods": list(METHODS),
        "frozen_constants": {
            "ridge_stacking_alpha": RIDGE_STACKING_ALPHA,
            "ridge_stacking_fit_intercept": RIDGE_STACKING_FIT_INTERCEPT,
            "mase_seasonal_period": MASE_SEASONAL_PERIOD,
            "replay_tolerance": REPLAY_TOLERANCE,
            "required_datasets": list(REQUIRED_DATASETS),
            "required_forecast_steps": list(REQUIRED_FORECAST_STEPS),
            "mase_train_segment": "原始序列中严格早于 H1 history_start 的全部数据",
            "mean_ratio_definition": "逐数据集比值的等权平均（不是先平均 MAE 再相除）",
            "qualified_candidates_definition":
                "输入契约下合格的全部候选（已登记、产物存在、能产出完整轨迹），"
                "不再经 filter_weak_models；Best Single 与 Ridge Stacking 共用这一个池",
            "thresholds": {
                "11.2.1a_best_single_mean_ratio_lt": THRESHOLD_BEST_SINGLE_MEAN_RATIO,
                "11.2.1b_best_single_per_dataset_ratio_le":
                    THRESHOLD_BEST_SINGLE_PER_DATASET_RATIO,
                "11.2.2_seasonal_naive_per_task_ratio_lt":
                    THRESHOLD_SEASONAL_NAIVE_PER_TASK_RATIO,
                "11.2.3_ridge_mean_ratio_le": THRESHOLD_RIDGE_MEAN_RATIO,
            },
        },
    }
    (out / "experiment_definition.json").write_text(
        json.dumps(definition, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    store = ModelStore(str(args.database))
    tasks: List[Dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="stage2_") as tmp:
            for dataset in args.datasets:
                for steps in args.forecast_steps:
                    windows = _frozen_windows(args.window_plan, dataset, int(steps))
                    tasks.append(
                        evaluate_task(
                            store=store, database=args.database, raw_root=args.raw_root,
                            dataset=dataset, forecast_steps=int(steps), windows=windows,
                            declared_candidates=args.candidates,
                            library_report=library_report,
                            predictions_dir=out / "predictions", workdir=Path(tmp),
                        )
                    )
    except Stage2Error as exc:
        print(f"[stage2] 运行不完整，不产出门槛结论: {exc}")
        return EXIT_INCOMPLETE
    finally:
        store.close()

    (out / "main_metrics.json").write_text(
        json.dumps({"tasks": tasks}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    acceptance = evaluate_gates(tasks)
    (out / "acceptance.json").write_text(
        json.dumps(acceptance, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    for task in tasks:
        metrics = task["task_metrics"]
        print(f"[stage2] {task['dataset']} s={task['forecast_steps']}: "
              + "，".join(f"{m} MAE={metrics[m]['mae']:.4f}" for m in METHODS))
    coverage = next(r for r in acceptance["rules"] if r["rule"] == "coverage")
    if not coverage["passed"]:
        print(f"[stage2] 覆盖度不符：记录 {coverage['task_records']} 条，"
              f"缺少 {coverage['missing_tasks']}，多余 {coverage['extra_tasks']}，"
              f"重复 {coverage['duplicate_tasks']}；"
              "这不是恰好 3×3 的口径，所有等权平均门槛一律判不通过。")
    for rule in acceptance["rules"]:
        if rule["rule"] == "11.1.4" and not rule["passed"]:
            print(f"[stage2] {rule['dataset']} s={rule['forecast_steps']} "
                  f"{rule['window']}: 在线输出与离线重放最大逐值误差 "
                  f"{rule['value']:.3e} > {REPLAY_TOLERANCE:g}")
    for rule in acceptance["rules"]:
        if rule["rule"].startswith("11.2") and rule.get("value") is not None:
            print(f"[stage2] {rule['rule']} s={rule.get('forecast_steps')}: "
                  f"{rule['value']:.4f} {rule['comparison']} {rule['threshold']} -> "
                  f"{'通过' if rule['passed'] else '未通过'}")
    print(f"\n[stage2] 结果已保存: {out}")

    if not acceptance["passed"]:
        print("[stage2] 基础质量门槛未通过：按 §12 停止，不投入三个外部深度方法的全量训练。")
        return EXIT_GATE_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
