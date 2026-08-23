"""Task 7：九任务 System A/B 影子对照运行器。

固定任务集合：PJM / AEMO VIC / AEMO NSW × h=1/6/24，共 9 个任务。

每个任务只构建一次候选预测矩阵；Protocol B interaction 开/关共用该矩阵，
不重训、不重载预测文件。报告同时给出逐任务明细、等权平均、按样本量加权
平均和 interaction 胜/负任务数，并逐任务记录 System A/combinator 参考、
Protocol A 参考、validation-selected 最佳单模型、guard 状态与各阶段耗时。

运行方式（服务器正式运行前先做 PJM h=1 smoke）：

    python scripts/run_system_ab_shadow.py \
        --datasets pjm --horizons 1 \
        --output result/ab_convergence/shadow_9tasks_smoke.json

    python scripts/run_system_ab_shadow.py \
        --output result/ab_convergence/shadow_9tasks/shadow_9tasks.json

注意：
- 本脚本不训练基线。它读取 reports/baselines 下已有的 val/test 预测文件；
  缺失的 AEMO 基线需先由 scripts/train_baselines.py 生成。
- 每个任务使用独立 KGFeedbackStore，不加载历史反馈文件，避免跨任务污染。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.combination_utils import (
    DATASET_HORIZONS,
    filter_weak_models,
    get_common_models,
    load_predictions_safe,
)
from src.eval.kg.config import (
    KG_EXTENDED_POOL_MIN_LOADED,
    KG_SEASONAL_NAIVE_AS_FROZEN_EXPERT,
    KG_STRICT_EXTENDED_POOL,
    _build_extended_pool_strategies,
    _build_kg_model_candidates,
    RUNTIME_PREDICTIONS_KEY,
)
from src.eval.kg.data_io import _load_extended_pool_for_split
from src.eval.kg.feedback import KGFeedbackStore
from src.eval.kg.protocol_a import kg_combination_pred_only
from src.core.solver import build_protocol_b_context, build_solver
from src.core.index import IndexManager
from src.graph.model_graph import ModelGraph
from src.models.uncertainty import UncertaintyGate

import scripts.train_combinations_kg as kg_runner

REPORT_SCHEMA_VERSION = "task7-shadow.4"
# 关系强度对照与关系证据门槛加入后，quality_gates/aggregates 的**结构**变了。
# 旧版报告（v4d 等）必须仍按旧口径可复核，否则已发布的验收证据会失去可验证性。
LEGACY_REPORT_SCHEMA_VERSIONS = ("task7-shadow.3",)
RANDOM_SEED = 42
INTERACTION_TIE_TOLERANCE = 1e-12

TASK_DATASETS = ("pjm", "aemo_vic", "aemo_nsw")
TASK_HORIZONS = (1, 6, 24)

KEY_DEPENDENCIES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "torch",
    "pytorch-lightning",
    "prophet",
    "statsmodels",
    "cmdstanpy",
)

# 报告键名 -> import 模块名。scikit-learn 的包名是 sklearn，
# pytorch-lightning 的包名是 lightning。
DEPENDENCY_IMPORT_NAMES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "torch": "torch",
    "pytorch-lightning": "lightning",
    "prophet": "prophet",
    "statsmodels": "statsmodels",
    "cmdstanpy": "cmdstanpy",
}

# 与 train_combinations_kg.run_dataset_kg 保持一致的矩阵准备参数默认值。
DEFAULT_FILTER_THRESHOLD = 2.0
DEFAULT_ELIGIBLE_MASE_HARD = float("inf")


def build_task_specs() -> List[Dict[str, Any]]:
    """固定九任务集合：PJM / AEMO VIC / AEMO NSW × h=1/6/24。"""
    return [
        {"dataset": dataset, "horizon": horizon}
        for dataset in TASK_DATASETS
        for horizon in TASK_HORIZONS
    ]


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) <= 0:
        return None
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) <= 0:
        return None
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    return {
        "mae": _safe_mae(y_true, y_pred),
        "rmse": _safe_rmse(y_true, y_pred),
    }


def evaluate_system_a_matrix_reference(
    *,
    dataset: str,
    horizon: int,
    df_test: pd.DataFrame,
    selected_models: Sequence[str],
    weights: Mapping[str, float],
    scenario_id: str,
    path_id: str,
) -> Dict[str, Any]:
    """在冻结的同任务预测矩阵上评估旧 System A 选择与权重。

    这里刻意保留旧 ``WeightedBlender`` 的权重语义：不重归一化，只把
    System A 选中的模型列按原权重相加。任何模型列缺失都会使参照无效，
    禁止像旧训练路径那样只用成功模型继续计算并把降级结果标成 ``ok``。
    """
    selected = list(selected_models)
    base = {
        "dataset": dataset,
        "horizon": int(horizon),
        "reference_mode": "shared_prediction_matrix",
        "models": selected,
        "weights": dict(weights),
        "scenario_id": scenario_id,
        "path_id": path_id,
        "n_test": int(len(df_test)),
        "data_sha_test": _data_sha(df_test),
    }
    if not selected:
        return {**base, "status": "invalid_reference", "reason": "System A selected no models"}
    if "y" not in df_test.columns or df_test.empty:
        return {
            **base,
            "status": "invalid_reference",
            "reason": "shared prediction matrix has no non-empty y column",
        }

    missing = [model for model in selected if model not in df_test.columns]
    if missing:
        return {
            **base,
            "status": "invalid_reference",
            "reason": f"selected models missing from shared prediction matrix: {missing}",
            "missing_models": missing,
        }

    missing_weights = [model for model in selected if model not in weights]
    weight_values = np.asarray([weights.get(model, np.nan) for model in selected], dtype=float)
    if missing_weights or not np.isfinite(weight_values).all():
        return {
            **base,
            "status": "invalid_reference",
            "reason": f"System A weights missing or non-finite: {missing_weights}",
            "missing_weights": missing_weights,
        }

    y_test = df_test["y"].to_numpy(dtype=float)
    pred_test = df_test[selected].to_numpy(dtype=float) @ weight_values
    metrics = metric_summary(y_test, pred_test)
    if metrics["mae"] is None or not np.isfinite(metrics["mae"]):
        return {
            **base,
            "status": "invalid_reference",
            "reason": "shared prediction matrix produced no finite System A evaluation pairs",
        }
    prediction_sha = hashlib.sha256(np.asarray(pred_test, dtype=float).tobytes()).hexdigest()
    return {
        **base,
        "status": "ok",
        "metrics": metrics,
        "prediction_sha": prediction_sha,
    }


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return "<absent>"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data_sha(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _collect_dependency_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for dep in KEY_DEPENDENCIES:
        import_name = DEPENDENCY_IMPORT_NAMES.get(dep, dep)
        try:
            module = importlib.import_module(import_name)
            versions[dep] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            versions[dep] = f"unavailable: {type(exc).__name__}"
    return versions


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return completed.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {type(exc).__name__}"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


# ---------------------------------------------------------------------------
# 预测矩阵构建（每个任务只调用一次）
# ---------------------------------------------------------------------------

def build_task_matrix(
    *,
    dataset: str,
    horizon: int,
    models: Sequence[str],
    pred_root: Path,
    raw_root: Optional[Path],
    filter_threshold: float = DEFAULT_FILTER_THRESHOLD,
    naive_as_frozen_expert: bool = KG_SEASONAL_NAIVE_AS_FROZEN_EXPERT,
    eligible_mase_hard: float = DEFAULT_ELIGIBLE_MASE_HARD,
) -> Dict[str, Any]:
    """加载预测文件并执行与 run_dataset_kg 一致的过滤/清洗，产出任务矩阵。

    返回字段：
    - df_val_kg / df_test_kg：Protocol A/B 直接消费的预测矩阵
    - df_raw_val / df_raw_test：原始特征
    - safe_models：过滤后的候选模型列
    - base_model_cols：safe_models 中属于基础模型的列
    - metadata：过滤理由、valid_ratio、impute 等信息
    """
    base_models = list(models)
    df_val = load_predictions_safe(pred_root, dataset, horizon, base_models, "val")
    df_test = load_predictions_safe(pred_root, dataset, horizon, base_models, "test")

    frozen_naive_meta: Dict[str, Any] = {
        "enabled": bool(naive_as_frozen_expert),
        "loaded": False,
        "skip_reason": None,
        "source": None,
    }
    if naive_as_frozen_expert:
        naive_val, naive_val_meta = _load_extended_pool_for_split(
            df_val,
            split="val",
            dataset=dataset,
            horizon=horizon,
            pred_root=pred_root,
            combo_root=None,
            strategies=["seasonal_naive"],
            allow_in_sample=False,
        )
        naive_test, naive_test_meta = _load_extended_pool_for_split(
            df_test,
            split="test",
            dataset=dataset,
            horizon=horizon,
            pred_root=pred_root,
            combo_root=None,
            strategies=["seasonal_naive"],
            allow_in_sample=False,
        )
        df_val, df_test = naive_val, naive_test
        val_loaded = naive_val_meta.get("loaded", []) if isinstance(naive_val_meta, dict) else []
        test_loaded = naive_test_meta.get("loaded", []) if isinstance(naive_test_meta, dict) else []
        frozen_naive_meta["source"] = "standalone_row_id_merge"
        frozen_naive_meta["loaded"] = bool(
            "seasonal_naive" in val_loaded and "seasonal_naive" in test_loaded
        )
        if not frozen_naive_meta["loaded"]:
            frozen_naive_meta["skip_reason"] = "missing_or_merge_failed"

    candidate_models = list(dict.fromkeys(base_models + (["seasonal_naive"] if naive_as_frozen_expert else [])))
    model_cols = get_common_models(df_val, df_test, candidate_models)
    if len(model_cols) < 2:
        raise RuntimeError(
            f"{dataset} h={horizon}: 预测文件共同模型数不足 2：{model_cols}"
        )

    # 弱模型过滤（与 run_dataset_kg 完全相同的入口）
    eligible_filter_reasons: Dict[str, List[str]] = {m: [] for m in candidate_models}
    model_valid_ratio: Dict[str, float] = {}
    model_valid_pairs: Dict[str, int] = {}
    model_fair_mae: Dict[str, float] = {}
    y_val_arr = np.asarray(df_val["y"].values, dtype=float)
    for m in candidate_models:
        if m not in model_cols:
            eligible_filter_reasons[m].append("missing_pred_file_val_or_test")
            continue
        eligible_filter_reasons[m].append("alignment_ok")
        try:
            from src.eval.kg.data_io import _valid_pair_mae

            fair_mae, v_ratio, v_count = _valid_pair_mae(
                y_val_arr, np.asarray(df_val[m].values, dtype=float)
            )
            model_valid_ratio[m] = v_ratio
            model_valid_pairs[m] = v_count
            model_fair_mae[m] = fair_mae
            if v_count <= 0 or not np.isfinite(fair_mae):
                model_fair_mae[m] = float("inf")
                eligible_filter_reasons[m].append(f"insufficient_valid_pairs:{v_count}")
        except Exception as exc:  # noqa: BLE001
            model_fair_mae[m] = float("inf")
            model_valid_ratio[m] = 0.0
            model_valid_pairs[m] = 0
            eligible_filter_reasons[m].append(f"mae_compute_failed:{type(exc).__name__}")

    df_val_for_filter, _, filter_impute_meta = kg_runner._impute_prediction_nans(
        df_val, df_test, model_cols
    )
    safe_models, filter_meta = filter_weak_models(
        df_val_for_filter, model_cols, threshold_ratio=filter_threshold, horizon=horizon
    )
    filter_meta["nan_imputation_for_filter"] = filter_impute_meta

    for m in model_cols:
        if m not in safe_models:
            removed = filter_meta.get("stability_removed") or {}
            if m in removed:
                reason = f"safe_cols_stability:{removed[m]}"
            else:
                soft_th = filter_meta.get("soft_threshold")
                mae_v = model_fair_mae.get(m)
                if mae_v is not None and np.isfinite(mae_v) and soft_th is not None:
                    reason = f"safe_cols_threshold:{mae_v:.4f}>{float(soft_th):.4f}"
                elif model_valid_pairs.get(m, 0) <= 0:
                    reason = "safe_cols_threshold:insufficient_valid_pairs"
                else:
                    reason = "safe_cols_filtered"
            eligible_filter_reasons.setdefault(m, []).append(reason)

    if np.isfinite(eligible_mase_hard):
        seasonal_period = kg_runner._infer_seasonal_period(df_val)
        mase_removed = []
        for m in list(safe_models):
            mase_like = kg_runner._estimate_mase_like(
                df_val, m, seasonal_period=seasonal_period
            )
            if mase_like is not None and np.isfinite(mase_like) and mase_like > float(eligible_mase_hard):
                mase_removed.append(m)
                eligible_filter_reasons.setdefault(m, []).append(
                    f"mase_hard:{mase_like:.4f}>{float(eligible_mase_hard):.4f}"
                )
        if mase_removed:
            candidate_kept = [m for m in safe_models if m not in mase_removed]
            if len(candidate_kept) >= 2:
                safe_models = candidate_kept

    if len(safe_models) < 2:
        raise RuntimeError(
            f"{dataset} h={horizon}: 过滤后模型数不足 2：{safe_models}"
        )

    df_val_kg, df_test_kg, safe_impute_meta = kg_runner._impute_prediction_nans(
        df_val, df_test, safe_models
    )

    df_raw_val = None
    df_raw_test = None
    raw_meta = {"val_loaded": False, "test_loaded": False, "val_path": None, "test_path": None}
    if raw_root:
        val_candidates = [
            raw_root / dataset / f"val_h{horizon}.csv",
            raw_root / dataset / "val.csv",
            raw_root / dataset / f"val_{horizon}.csv",
        ]
        test_candidates = [
            raw_root / dataset / f"test_h{horizon}.csv",
            raw_root / dataset / "test.csv",
            raw_root / dataset / f"test_{horizon}.csv",
        ]
        raw_val_path = next((p for p in val_candidates if p.exists()), None)
        raw_test_path = next((p for p in test_candidates if p.exists()), None)
        if raw_val_path is not None:
            try:
                df_raw_val = pd.read_csv(raw_val_path)
                raw_meta["val_loaded"] = True
                raw_meta["val_path"] = str(raw_val_path)
            except Exception as exc:  # noqa: BLE001
                raw_meta["val_error"] = str(exc)
        if raw_test_path is not None:
            try:
                df_raw_test = pd.read_csv(raw_test_path)
                raw_meta["test_loaded"] = True
                raw_meta["test_path"] = str(raw_test_path)
            except Exception as exc:  # noqa: BLE001
                raw_meta["test_error"] = str(exc)

    kg_input_cols = list(dict.fromkeys(safe_models + ["y", "timestamp"]))
    if "seasonal_naive" in df_val_kg.columns and "seasonal_naive" in df_test_kg.columns:
        kg_input_cols = list(dict.fromkeys(kg_input_cols + ["seasonal_naive"]))

    for m in safe_models:
        eligible_filter_reasons.setdefault(m, []).append("eligible_pass_all_gates")

    model_node_types = {
        m: ("naive_baseline" if m == "seasonal_naive" else ("base_model" if m in base_models else "extended_pool"))
        for m in safe_models
    }

    return {
        "df_val_kg": df_val_kg[kg_input_cols],
        "df_test_kg": df_test_kg[kg_input_cols],
        "df_raw_val": df_raw_val,
        "df_raw_test": df_raw_test,
        "safe_models": list(safe_models),
        "base_model_cols": [m for m in safe_models if m in base_models],
        "metadata": {
            "filter": filter_meta,
            "safe_models": list(safe_models),
            "common_base_models": [m for m in model_cols if m in base_models],
            "eligible_models": list(safe_models),
            "eligible_filter_reasons": eligible_filter_reasons,
            "model_valid_ratio": model_valid_ratio,
            "model_valid_pairs": model_valid_pairs,
            "model_fair_mae": {
                m: v for m, v in model_fair_mae.items() if np.isfinite(v)
            },
            "nan_imputation_safe_models": safe_impute_meta,
            "candidate_models_count": int(len(candidate_models)),
            "common_models_count": int(len(model_cols)),
            "frozen_naive": frozen_naive_meta,
            "raw": raw_meta,
            "model_node_types": model_node_types,
            "n_val": int(len(df_val_kg)),
            "n_test": int(len(df_test_kg)),
        },
    }


def build_candidate_outcome_audit(metadata: Mapping[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """把候选不可用与质量过滤分开记录，避免空 ``failed_models`` 误导。"""
    safe_models = set(metadata.get("safe_models") or [])
    reasons_by_model = metadata.get("eligible_filter_reasons") or {}
    failed_models: Dict[str, List[str]] = {}
    filtered_models: Dict[str, List[str]] = {}
    failure_markers = (
        "missing_pred_file",
        "mae_compute_failed",
        "insufficient_valid_pairs",
    )
    if isinstance(reasons_by_model, Mapping):
        for model, raw_reasons in reasons_by_model.items():
            reasons = list(raw_reasons or [])
            if model in safe_models:
                continue
            if any(any(marker in str(reason) for marker in failure_markers) for reason in reasons):
                failed_models[str(model)] = reasons
            else:
                filtered_models[str(model)] = reasons
    return {
        "failed_models": failed_models,
        "filtered_models": filtered_models,
    }


# ---------------------------------------------------------------------------
# Protocol 运行（全部基于同一矩阵）
# ---------------------------------------------------------------------------

@contextmanager
def _interaction_disabled_for(dataset: str):
    """临时按数据集禁用 interaction，退出后恢复。不重训、不重建矩阵。"""
    import src.eval.kg.config as kg_config
    import src.eval.kg.protocol_b as protocol_b

    original_config = kg_config.PROTOCOL_B_DISABLE_INTERACTION_DATASETS
    original_protocol_b = protocol_b.PROTOCOL_B_DISABLE_INTERACTION_DATASETS
    disabled = set(original_config) | {dataset}
    kg_config.PROTOCOL_B_DISABLE_INTERACTION_DATASETS = set(disabled)
    protocol_b.PROTOCOL_B_DISABLE_INTERACTION_DATASETS = set(disabled)
    try:
        yield
    finally:
        kg_config.PROTOCOL_B_DISABLE_INTERACTION_DATASETS = original_config
        protocol_b.PROTOCOL_B_DISABLE_INTERACTION_DATASETS = original_protocol_b


def _relation_edges_found(raw: Mapping[str, Any]) -> List[str]:
    """本次运行实际消费到的关系边（scenario->model 的 recommended_for）。"""
    meta = (
        ((raw.get("val") or {}).get("weight_meta") or {})
        .get("protocol_b_selection_meta", {})
        .get("relation_strength")
    ) or {}
    return list(meta.get("edges_found") or [])


def _run_protocol_b_on_matrix(
    *,
    dataset: str,
    horizon: int,
    matrix: Mapping[str, Any],
    feedback_store: KGFeedbackStore,
    trace_path: Optional[Path],
    relation_graph: Any = None,
    write_relations: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
    """在已构建好的矩阵上运行 Protocol B（solver 路径），开启运行时精确预测。

    与 scripts/train_combinations_kg._run_protocol_b_with_solver 保持同一
    solver 入口，但额外开启 return_predictions，确保 test_mae_on/off 直接
    由引擎交出的 pred_test 计算。
    """
    manifests = kg_runner._build_runtime_manifests(list(matrix["safe_models"]))
    index_manager = IndexManager.with_defaults(manifests=manifests)
    ctx = build_protocol_b_context(
        dataset=dataset,
        horizon=horizon,
        df_val=matrix["df_val_kg"],
        df_test=matrix["df_test_kg"],
        df_raw_val=matrix["df_raw_val"],
        df_raw_test=matrix["df_raw_test"],
        model_cols=list(matrix["safe_models"]),
        base_model_cols=list(matrix["base_model_cols"]),
        feedback_store=feedback_store,
        return_predictions=True,
    )
    # §11#7：把关系图交给 solver 的**消费**侧。未传时关系项恒为中性，
    # v5 将无法评估本功能，故报告层会断言至少有任务 edges_found 非空。
    ctx.model_graph = relation_graph
    solver_kwargs: Dict[str, Any] = {
        "manifests": manifests,
        "index_manager": index_manager,
        "uncertainty_gate": UncertaintyGate(threshold=float("inf")),
    }
    # 写入侧（Hawkes temporal stage）只在 warm-up 打开。正式测量必须只读：
    # 一旦测量也写图，任务的运行顺序就会改变关系状态，结果不可复现。
    if relation_graph is not None and write_relations:
        solver_kwargs["temporal_relation_graph"] = relation_graph
        solver_kwargs["temporal_relation_create_missing"] = True
    solver = build_solver("protocol_b", **solver_kwargs)
    normalized, trace = solver.solve(
        ctx,
        trace_path=str(trace_path) if trace_path is not None else None,
    )
    raw = normalized.get("raw", normalized) if isinstance(normalized, dict) else normalized
    predictions = normalized.get("predictions") if isinstance(normalized, dict) else None
    return raw, predictions, trace


def _run_protocol_a_on_matrix(matrix: Mapping[str, Any], dataset: str, horizon: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    raw = kg_combination_pred_only(
        matrix["df_val_kg"],
        matrix["df_test_kg"],
        list(matrix["safe_models"]),
        horizon,
        dataset_name=dataset,
        return_predictions=True,
    )
    elapsed = time.perf_counter() - t0
    predictions = raw.pop(RUNTIME_PREDICTIONS_KEY, None)
    split = raw.get("test") or {}
    y_test = np.asarray(matrix["df_test_kg"]["y"].values, dtype=float)
    pred_test = (
        np.asarray(predictions["test"], dtype=float)
        if isinstance(predictions, dict) and "test" in predictions
        else None
    )
    metrics = metric_summary(y_test, pred_test) if pred_test is not None else {"mae": None, "rmse": None}
    return {
        "status": "ok",
        "protocol": raw.get("protocol"),
        "models": list(split.get("selected_models") or []),
        "weights": dict(split.get("weights") or {}),
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "weight_meta": split.get("weight_meta") if isinstance(split, dict) else None,
    }


def _protocol_b_split_summary(raw: Mapping[str, Any], predictions: Mapping[str, Any]) -> Dict[str, Any]:
    split = raw.get("test") or {}
    if not isinstance(split, dict):
        split = {}
    weight_meta = split.get("weight_meta") or {}
    if not isinstance(weight_meta, dict):
        weight_meta = {}
    guard = weight_meta.get("protocol_b_guard") or {}
    guard_config = weight_meta.get("guard_config") or {}
    if not isinstance(guard, dict):
        guard = {}
    if not isinstance(guard_config, dict):
        guard_config = {}
    fallback_target = guard.get("fallback_target")
    fallback_reason = guard.get("reason")
    if fallback_target is None and isinstance(guard_config, dict):
        fallback_target = guard_config.get("final_fallback_target")
        fallback_reason = guard_config.get("final_fallback_reason")

    final_interaction_branch = weight_meta.get("interaction_branch") or {}
    candidate_interaction_branch = weight_meta.get("interaction_branch_candidate") or {}
    post_adjustment = weight_meta.get("post_adjustment") or {}
    if not isinstance(final_interaction_branch, dict):
        final_interaction_branch = {}
    if not isinstance(candidate_interaction_branch, dict):
        candidate_interaction_branch = {}
    if not isinstance(post_adjustment, dict):
        post_adjustment = {}
    interaction_branch = candidate_interaction_branch or final_interaction_branch
    interaction_evaluated = bool(interaction_branch)
    interaction_candidate_applied = bool(interaction_branch.get("applied"))
    post_adjustment_applied = bool(post_adjustment.get("applied"))
    final_prediction_contains_interaction = bool(
        interaction_candidate_applied
        and fallback_target is None
        and bool(final_interaction_branch)
        and not post_adjustment_applied
    )
    if fallback_target is not None:
        interaction_status_reason = f"guard_fallback:{fallback_target}"
    elif final_prediction_contains_interaction:
        interaction_status_reason = "applied_to_final_prediction"
    elif interaction_candidate_applied and post_adjustment_applied:
        interaction_status_reason = "overwritten_by_post_adjustment"
    elif not interaction_evaluated:
        interaction_status_reason = "not_evaluated"
    else:
        disabled_reason = (
            interaction_branch.get("disabled_reason_code")
            or interaction_branch.get("disabled_reason")
        )
        reject_reasons = interaction_branch.get("reject_reasons") or []
        if disabled_reason:
            interaction_status_reason = f"disabled:{disabled_reason}"
        elif reject_reasons:
            interaction_status_reason = f"rejected:{','.join(map(str, reject_reasons))}"
        else:
            interaction_status_reason = "evaluated_not_applied"

    oof_raw = interaction_branch.get("cv_mae_raw_guard")
    oof_inter = interaction_branch.get("cv_mae_interaction_guard")
    if interaction_branch.get("cv_guard_source") != "oof_blocked_cv":
        oof_raw = None
        oof_inter = None

    return {
        "protocol": raw.get("protocol"),
        "models": list(split.get("selected_models") or []),
        "weights": dict(split.get("weights") or {}),
        "guard": {
            "fallback_target": fallback_target,
            "fallback_reason": fallback_reason,
            "guard_config": guard_config,
        },
        "interaction_branch": interaction_branch,
        "interaction_evaluated": interaction_evaluated,
        "interaction_candidate_applied": interaction_candidate_applied,
        "post_adjustment": post_adjustment,
        "post_adjustment_applied": post_adjustment_applied,
        "final_prediction_contains_interaction": final_prediction_contains_interaction,
        "interaction_status_reason": interaction_status_reason,
        "interaction_applied": final_prediction_contains_interaction,
        "val_mae_raw": interaction_branch.get("val_mae_raw"),
        "val_mae_interaction": interaction_branch.get("val_mae_interaction"),
        "val_mae_delta": (
            interaction_branch.get("val_mae_interaction") - interaction_branch.get("val_mae_raw")
            if interaction_branch.get("val_mae_interaction") is not None
            and interaction_branch.get("val_mae_raw") is not None
            else None
        ),
        "oof_mae_raw": oof_raw,
        "oof_mae_interaction": oof_inter,
        "oof_mae_delta": (
            oof_inter - oof_raw
            if oof_inter is not None and oof_raw is not None
            else None
        ),
        "cv_oof_coverage": interaction_branch.get("cv_oof_coverage"),
        "cv_guard_source": interaction_branch.get("cv_guard_source"),
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# System A/combinator 参考（真实数据，可选）
# ---------------------------------------------------------------------------

def _combinator_worker(
    queue,
    dataset: str,
    horizon: int,
    feature_root: str,
    tmpdir: str,
    shared_test_matrix: pd.DataFrame,
    shared_models: Sequence[str],
) -> None:
    """子进程入口：执行旧 System A 参考并回传 JSON 兼容结果。

    参数使用字符串而不是 Path，避免 fork 模式下跨进程传对象的不确定性；
    在子进程内再转回 Path。
    """
    try:
        result = run_combinator_reference(
            dataset=dataset,
            horizon=horizon,
            feature_root=Path(feature_root),
            tmpdir=Path(tmpdir),
            shared_test_matrix=shared_test_matrix,
            shared_models=shared_models,
        )
        queue.put({"status": "returned", "result": result})
    except BaseException as exc:  # noqa: BLE001
        queue.put({"status": "exception", "error": f"{type(exc).__name__}: {exc}"})


def run_combinator_reference_with_timeout(
    *,
    dataset: str,
    horizon: int,
    feature_root: Path,
    tmpdir: Path,
    shared_test_matrix: pd.DataFrame,
    shared_models: Sequence[str],
    timeout_seconds: float = 900.0,
) -> Dict[str, Any]:
    """带超时的旧 System A 选择；超时会被记录并使完整九任务质量门槛失败。"""
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_combinator_worker,
        args=(
            queue,
            dataset,
            int(horizon),
            str(feature_root),
            str(tmpdir),
            shared_test_matrix,
            list(shared_models),
        ),
    )
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        return {
            "status": "timeout",
            "reason": f"legacy System A exceeded {timeout_seconds:.0f}s and was terminated",
        }
    if proc.exitcode != 0:
        return {"status": "failed", "reason": f"worker exited with code {proc.exitcode}"}
    try:
        message = queue.get(timeout=5)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": f"failed to read worker result: {exc}"}
    if message.get("status") == "returned":
        return message.get("result") or {"status": "failed", "reason": "empty worker result"}
    return {
        "status": "failed",
        "reason": f"worker {message.get('status')}: {message.get('error')}",
    }


def run_combinator_reference(
    *,
    dataset: str,
    horizon: int,
    feature_root: Path,
    tmpdir: Path,
    shared_test_matrix: pd.DataFrame,
    shared_models: Sequence[str],
) -> Dict[str, Any]:
    """运行旧 System A 选择，并在同一 horizon 冻结矩阵上评估其组合。"""
    import src.pipeline.main as pipeline_main
    from src.pipeline.main import PowerPredictionPipeline

    train_path = feature_root / dataset / "train.csv"
    if not train_path.exists():
        return {
            "status": "not_run",
            "reason": f"training features not found for {dataset}: {train_path}",
        }

    try:
        train = pd.read_csv(train_path)
    except Exception as exc:  # noqa: BLE001
        return {"status": "not_run", "reason": f"failed to load features: {exc}"}

    if "region" not in train.columns or train["region"].nunique() < 1:
        return {"status": "not_run", "reason": "feature frame has no region column"}
    region = str(train["region"].dropna().iloc[0])
    train = train[train["region"] == region].copy()
    if train.empty:
        return {"status": "not_run", "reason": "empty train after region filter"}

    tmpdir.mkdir(parents=True, exist_ok=True)
    # 注意：PowerPredictionPipeline 必须在真实 PROJECT_ROOT 下初始化，否则
    # load_configs 会去隔离目录找 configs/model_assets.yaml 并失败。初始化完成后
    # 再切换 PROJECT_ROOT，使后续构建图谱/选择审计只写隔离目录（与 Task 6A 一致）。
    pipeline = PowerPredictionPipeline()
    old_root = pipeline_main.PROJECT_ROOT
    old_graph_path = os.environ.get("MODELCOMBINE_GRAPH_STATE_PATH")
    isolated_graph = tmpdir / "graph_state.pkl"
    source_graph = PROJECT_ROOT / "reports" / "graph_state.pkl"
    if source_graph.exists():
        shutil.copy2(source_graph, isolated_graph)
    os.environ["MODELCOMBINE_GRAPH_STATE_PATH"] = str(isolated_graph)
    pipeline_main.PROJECT_ROOT = str(tmpdir)
    try:
        graph = pipeline.build_model_graph()
        t0 = time.perf_counter()
        selected, weights, scenario_id, path_id = pipeline.select_models_for_region(
            region,
            train,
            graph,
            available_models_override=list(shared_models),
        )
        selection_seconds = time.perf_counter() - t0
    finally:
        pipeline_main.PROJECT_ROOT = old_root
        if old_graph_path is None:
            os.environ.pop("MODELCOMBINE_GRAPH_STATE_PATH", None)
        else:
            os.environ["MODELCOMBINE_GRAPH_STATE_PATH"] = old_graph_path

    reference = evaluate_system_a_matrix_reference(
        dataset=dataset,
        horizon=horizon,
        df_test=shared_test_matrix,
        selected_models=selected,
        weights=weights,
        scenario_id=scenario_id,
        path_id=path_id,
    )
    return {
        **reference,
        "region": region,
        "selection_seconds": selection_seconds,
        "elapsed_seconds": selection_seconds,
        "history_records_loaded": len(pipeline.historical_scenarios),
    }


# ---------------------------------------------------------------------------
# 任务运行
# ---------------------------------------------------------------------------

def run_task(
    *,
    dataset: str,
    horizon: int,
    models: Sequence[str],
    pred_root: Path,
    raw_root: Optional[Path],
    out_root: Path,
    filter_threshold: float,
    seed: int,
    run_combinator: bool = True,
    combinator_timeout_seconds: float = 900.0,
    feature_root: Optional[Path] = None,
    baseline_provenance: Optional[Mapping[str, Any]] = None,
    relation_graph: Any = None,
) -> Dict[str, Any]:
    _seed_everything(seed)
    task_out = out_root / dataset / f"h{horizon}"
    task_out.mkdir(parents=True, exist_ok=True)

    record: Dict[str, Any] = {
        "dataset": dataset,
        "horizon": horizon,
        "status": "unknown",
    }
    t_start = time.perf_counter()
    try:
        t0 = time.perf_counter()
        matrix = build_task_matrix(
            dataset=dataset,
            horizon=horizon,
            models=models,
            pred_root=pred_root,
            raw_root=raw_root,
            filter_threshold=filter_threshold,
        )
        matrix_seconds = time.perf_counter() - t0
        if baseline_provenance is not None:
            from scripts.train_baselines import verify_task_artifacts

            # 验证全部已加载的基础候选，不只验证过滤后的最终 safe_models：
            # 被过滤的旧 CSV 也可能改变过滤阈值，不能让它绕过来源检查。
            verified_models = list(matrix["metadata"]["common_base_models"])
            # seasonal_naive 被 _build_kg_model_candidates() 从基础候选中剔除、
            # 作为 frozen expert 单独加载，因此不会出现在 common_base_models 里。
            # 但它现在由 train_baselines 按配置真实训练并产出带哈希的产物
            # （见 16d2c0a），只要本轮确实加载了它，就必须一并核对来源，
            # 否则它的预测可以来自与本轮基线无关的旧 CSV。
            frozen_naive = (matrix["metadata"].get("frozen_naive") or {})
            if frozen_naive.get("loaded") and "seasonal_naive" not in verified_models:
                verified_models.append("seasonal_naive")
            verify_task_artifacts(
                baseline_provenance,
                pred_root=pred_root,
                dataset=dataset,
                horizon=horizon,
                models=verified_models,
            )
        candidate_audit = build_candidate_outcome_audit(matrix["metadata"])
        record["matrix"] = {
            "elapsed_seconds": matrix_seconds,
            "n_val": int(len(matrix["df_val_kg"])),
            "n_test": int(len(matrix["df_test_kg"])),
            "safe_models": list(matrix["safe_models"]),
            "common_base_models": list(matrix["metadata"]["common_base_models"]),
            **candidate_audit,
            "data_sha_val": _data_sha(matrix["df_val_kg"]),
            "data_sha_test": _data_sha(matrix["df_test_kg"]),
            "raw_val_loaded": bool(matrix["metadata"]["raw"]["val_loaded"]),
            "raw_test_loaded": bool(matrix["metadata"]["raw"]["test_loaded"]),
        }
        record.update(
            {k: matrix["metadata"][k] for k in ("filter", "eligible_filter_reasons", "frozen_naive")}
        )

        feedback_store = KGFeedbackStore(learning_rate=0.1)

        # §11#7 关系强度 warm-up（同任务，只写不测）。
        # 九个任务的 dataset/horizon 不同 -> scenario_id 不同，而消费侧按当前
        # scenario_id 精确匹配，所以"前一个任务写、后一个任务读"不成立：
        # PJM h=1 写的边 PJM h=6 根本读不到。闭环只能落在同一个任务上。
        # warm-up 之后立即停止写入，让正式测量在同一份冻结的关系状态上进行，
        # 任务运行顺序才不会改变结果。预测矩阵仍只构建一次。
        if relation_graph is not None:
            warm_raw, warm_pred, warm_trace = _run_protocol_b_on_matrix(
                dataset=dataset,
                horizon=horizon,
                matrix=matrix,
                feedback_store=KGFeedbackStore(learning_rate=0.1),
                trace_path=task_out / "protocol_b_trace_relation_warmup.json",
                relation_graph=relation_graph,
                write_relations=True,
            )
            warm_scenario_id = getattr(warm_trace, "scenario_id", None)
            edges_written: List[str] = []
            if warm_scenario_id and relation_graph.G.has_node(warm_scenario_id):
                edges_written = sorted(
                    tgt
                    for _s, tgt, d in relation_graph.G.out_edges(warm_scenario_id, data=True)
                    if d.get("edge_type") == "recommended_for"
                )
            record["relation_warmup"] = {
                "scenario_id": warm_scenario_id,
                "selected_models": list(_protocol_b_split_summary(warm_raw, warm_pred)["models"]),
                "edges_written": edges_written,
                "edges_consumed": _relation_edges_found(warm_raw),
            }

        # Protocol B：interaction 开（默认）
        t0 = time.perf_counter()
        raw_on, pred_on, trace_on = _run_protocol_b_on_matrix(
            dataset=dataset,
            horizon=horizon,
            matrix=matrix,
            feedback_store=feedback_store,
            trace_path=task_out / "protocol_b_trace_on.json",
            relation_graph=relation_graph,
        )
        protocol_b_on_seconds = time.perf_counter() - t0
        b_on = _protocol_b_split_summary(raw_on, pred_on)
        # §11#7：记录本任务实际消费到的关系边，供"实验是否有效"门槛判定
        record["relation_strength_edges_found"] = _relation_edges_found(raw_on)
        y_test = np.asarray(matrix["df_test_kg"]["y"].values, dtype=float)
        pred_test_on = (
            np.asarray(b_on["predictions"]["test"], dtype=float)
            if isinstance(b_on["predictions"], dict) and "test" in b_on["predictions"]
            else None
        )
        if pred_test_on is None:
            raise RuntimeError("Protocol B interaction-on did not return runtime test predictions")
        record["test_mae_on"] = _safe_mae(y_test, pred_test_on)
        record["test_rmse_on"] = _safe_rmse(y_test, pred_test_on)
        record["protocol_b_on"] = {
            "protocol": b_on["protocol"],
            "models": b_on["models"],
            "weights": b_on["weights"],
            "guard": b_on["guard"],
            "interaction_branch": b_on["interaction_branch"],
            "interaction_evaluated": b_on["interaction_evaluated"],
            "interaction_candidate_applied": b_on["interaction_candidate_applied"],
            "post_adjustment": b_on["post_adjustment"],
            "post_adjustment_applied": b_on["post_adjustment_applied"],
            "final_prediction_contains_interaction": b_on["final_prediction_contains_interaction"],
            "interaction_status_reason": b_on["interaction_status_reason"],
            "elapsed_seconds": protocol_b_on_seconds,
            "trace_stages": [s.get("stage") for s in getattr(trace_on, "stages", [])],
        }

        # Protocol B：interaction 关（共用同一矩阵）
        t0 = time.perf_counter()
        with _interaction_disabled_for(dataset):
            raw_off, pred_off, trace_off = _run_protocol_b_on_matrix(
                dataset=dataset,
                horizon=horizon,
                matrix=matrix,
                feedback_store=KGFeedbackStore(learning_rate=0.1),
                trace_path=task_out / "protocol_b_trace_off.json",
                relation_graph=relation_graph,
            )
        protocol_b_off_seconds = time.perf_counter() - t0
        b_off = _protocol_b_split_summary(raw_off, pred_off)
        pred_test_off = (
            np.asarray(b_off["predictions"]["test"], dtype=float)
            if isinstance(b_off["predictions"], dict) and "test" in b_off["predictions"]
            else None
        )
        if pred_test_off is None:
            raise RuntimeError("Protocol B interaction-off did not return runtime test predictions")
        record["test_mae_off"] = _safe_mae(y_test, pred_test_off)
        record["test_rmse_off"] = _safe_rmse(y_test, pred_test_off)
        record["test_mae_delta"] = record["test_mae_on"] - record["test_mae_off"]
        record["protocol_b_off"] = {
            "protocol": b_off["protocol"],
            "models": b_off["models"],
            "weights": b_off["weights"],
            "guard": b_off["guard"],
            "interaction_branch": b_off["interaction_branch"],
            "interaction_evaluated": b_off["interaction_evaluated"],
            "interaction_candidate_applied": b_off["interaction_candidate_applied"],
            "post_adjustment": b_off["post_adjustment"],
            "post_adjustment_applied": b_off["post_adjustment_applied"],
            "final_prediction_contains_interaction": b_off["final_prediction_contains_interaction"],
            "interaction_status_reason": b_off["interaction_status_reason"],
            "elapsed_seconds": protocol_b_off_seconds,
            "trace_stages": [s.get("stage") for s in getattr(trace_off, "stages", [])],
        }

        # §11#7 关系强度对照臂：与上面的 on 臂**只差关系强度项**
        # （interaction 两臂都开）。现有 on/off 是 interaction 对照，两臂的关系
        # 强度完全一样，不能冒充关系强度收益对照，故单独跑一次中性臂。
        record["test_mae_relation_neutral"] = None
        record["test_mae_relation_delta"] = None
        record["relation_contrast"] = None
        if relation_graph is not None:
            t0 = time.perf_counter()
            raw_neutral, pred_neutral, _trace_neutral = _run_protocol_b_on_matrix(
                dataset=dataset,
                horizon=horizon,
                matrix=matrix,
                feedback_store=KGFeedbackStore(learning_rate=0.1),
                trace_path=task_out / "protocol_b_trace_relation_neutral.json",
                relation_graph=None,
                write_relations=False,
            )
            relation_neutral_seconds = time.perf_counter() - t0
            b_neutral = _protocol_b_split_summary(raw_neutral, pred_neutral)
            pred_test_neutral = (
                np.asarray(b_neutral["predictions"]["test"], dtype=float)
                if isinstance(b_neutral["predictions"], dict) and "test" in b_neutral["predictions"]
                else None
            )
            if pred_test_neutral is None:
                raise RuntimeError(
                    "Protocol B relation-neutral arm did not return runtime test predictions"
                )
            record["test_mae_relation_neutral"] = _safe_mae(y_test, pred_test_neutral)
            record["test_mae_relation_delta"] = (
                record["test_mae_on"] - record["test_mae_relation_neutral"]
            )
            record["relation_contrast"] = {
                "arm_definition": (
                    "enabled=消费当前场景 recommended_for 边；neutral=关系项恒为中性；"
                    "两臂 interaction 均开启；delta<0 表示关系强度带来收益"
                ),
                "enabled_mae": record["test_mae_on"],
                "neutral_mae": record["test_mae_relation_neutral"],
                "delta": record["test_mae_relation_delta"],
                "enabled_models": list(b_on["models"]),
                "neutral_models": list(b_neutral["models"]),
                "models_changed": list(b_on["models"]) != list(b_neutral["models"]),
                "enabled_edges_found": list(record["relation_strength_edges_found"]),
                "neutral_edges_found": _relation_edges_found(raw_neutral),
                "elapsed_seconds": relation_neutral_seconds,
            }

        # 逐任务必填的 interaction 字段（来自 interaction 开的那次）
        record["interaction_applied"] = bool(b_on["interaction_applied"])
        record["interaction_evaluated"] = bool(b_on["interaction_evaluated"])
        record["interaction_candidate_applied"] = bool(b_on["interaction_candidate_applied"])
        record["post_adjustment_applied"] = bool(b_on["post_adjustment_applied"])
        record["final_prediction_contains_interaction"] = bool(
            b_on["final_prediction_contains_interaction"]
        )
        record["interaction_status_reason"] = b_on["interaction_status_reason"]
        record["val_mae_raw"] = b_on["val_mae_raw"]
        record["val_mae_interaction"] = b_on["val_mae_interaction"]
        record["val_mae_delta"] = b_on["val_mae_delta"]
        record["oof_mae_raw"] = b_on["oof_mae_raw"]
        record["oof_mae_interaction"] = b_on["oof_mae_interaction"]
        record["oof_mae_delta"] = b_on["oof_mae_delta"]
        record["cv_oof_coverage"] = b_on["cv_oof_coverage"]
        record["n_val"] = int(len(matrix["df_val_kg"]))
        record["protocol"] = b_on["protocol"]
        record["fallback_target"] = b_on["guard"].get("fallback_target")
        record["selected_models"] = list(b_on["models"])
        record["selected_models_off"] = list(b_off["models"])
        record["selection_diff"] = {
            "on_models": list(b_on["models"]),
            "off_models": list(b_off["models"]),
            "models_changed": list(b_on["models"]) != list(b_off["models"]),
            "weights_changed": dict(b_on["weights"]) != dict(b_off["weights"]),
            "protocol_changed": b_on["protocol"] != b_off["protocol"],
        }

        # Protocol A 参考（同一矩阵）
        record["protocol_a"] = _run_protocol_a_on_matrix(matrix, dataset, horizon)

        # validation-selected 最佳单模型（同一矩阵，不使用 test 标签）
        y_val = np.asarray(matrix["df_val_kg"]["y"].values, dtype=float)
        val_maes = {
            m: float(np.mean(np.abs(matrix["df_val_kg"][m].to_numpy(dtype=float) - y_val)))
            for m in matrix["safe_models"]
        }
        best_single = min(matrix["safe_models"], key=lambda m: (val_maes[m], m))
        record["best_single"] = {
            "model": best_single,
            "validation_mae": val_maes[best_single],
            "test": metric_summary(
                y_test, matrix["df_test_kg"][best_single].to_numpy(dtype=float)
            ),
            "selection_uses_test_labels": False,
            "per_model": {
                m: {
                    "validation_mae": val_maes[m],
                    "test": metric_summary(
                        y_test, matrix["df_test_kg"][m].to_numpy(dtype=float)
                    ),
                }
                for m in matrix["safe_models"]
            },
        }

        # System A/combinator 参考：真实场景选择 + 同一 horizon 冻结预测矩阵评估。
        if run_combinator:
            with tempfile.TemporaryDirectory(prefix=f"shadow_combinator_{dataset}_h{horizon}_") as tmp:
                record["system_a"] = run_combinator_reference_with_timeout(
                    dataset=dataset,
                    horizon=horizon,
                    feature_root=feature_root or (PROJECT_ROOT / "data" / "features"),
                    tmpdir=Path(tmp),
                    shared_test_matrix=matrix["df_test_kg"],
                    shared_models=list(matrix["safe_models"]),
                    timeout_seconds=combinator_timeout_seconds,
                )
        else:
            record["system_a"] = {
                "status": "not_run",
                "reason": "run_combinator disabled by --skip-combinator",
            }

        # Protocol B 汇总参考
        record["protocol_b"] = {
            "on": {
                "mae": record["test_mae_on"],
                "rmse": record["test_rmse_on"],
                "protocol": b_on["protocol"],
                "models": b_on["models"],
            },
            "off": {
                "mae": record["test_mae_off"],
                "rmse": record["test_rmse_off"],
                "protocol": b_off["protocol"],
                "models": b_off["models"],
            },
        }
        record["guard"] = {
            "on": b_on["guard"],
            "off": b_off["guard"],
        }
        record["timing"] = {
            "matrix_sec": matrix_seconds,
            "protocol_b_on_sec": protocol_b_on_seconds,
            "protocol_b_off_sec": protocol_b_off_seconds,
            "protocol_a_sec": record["protocol_a"].get("elapsed_seconds"),
            "system_a_sec": (
                record["system_a"].get("elapsed_seconds")
                if isinstance(record["system_a"], dict)
                else None
            ),
            "total_sec": time.perf_counter() - t_start,
        }
        record["status"] = "ok"
    except BaseException as exc:  # noqa: BLE001
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = time.perf_counter() - t_start
    return record


# ---------------------------------------------------------------------------
# 聚合与报告验证
# ---------------------------------------------------------------------------

NUMERIC_AGG_FIELDS = (
    "test_mae_on",
    "test_mae_off",
    "test_mae_delta",
    "val_mae_raw",
    "val_mae_interaction",
    "val_mae_delta",
    "oof_mae_raw",
    "oof_mae_interaction",
    "oof_mae_delta",
    "cv_oof_coverage",
)

# 关系强度对照（schema >= task7-shadow.4）。与 interaction 对照分开统计：
# 两者的对照变量不同，混在一起会让"关系强度收益"读起来像 interaction 收益。
RELATION_AGG_FIELDS = (
    "test_mae_relation_neutral",
    "test_mae_relation_delta",
)


def aggregate_summary(
    tasks: Sequence[Mapping[str, Any]],
    *,
    include_relation: bool = True,
) -> Dict[str, Any]:
    task_details = []
    for t in tasks:
        detail = {
            "dataset": t.get("dataset"),
            "horizon": t.get("horizon"),
            "n_val": t.get("n_val"),
            "interaction_applied": t.get("interaction_applied"),
            "test_mae_on": t.get("test_mae_on"),
            "test_mae_off": t.get("test_mae_off"),
            "test_mae_delta": t.get("test_mae_delta"),
            "val_mae_delta": t.get("val_mae_delta"),
            "oof_mae_delta": t.get("oof_mae_delta"),
            "cv_oof_coverage": t.get("cv_oof_coverage"),
            "protocol": t.get("protocol"),
            "fallback_target": t.get("fallback_target"),
            "selected_models": t.get("selected_models"),
        }
        if include_relation:
            detail["test_mae_relation_neutral"] = t.get("test_mae_relation_neutral")
            detail["test_mae_relation_delta"] = t.get("test_mae_relation_delta")
            detail["relation_strength_edges_found"] = list(
                t.get("relation_strength_edges_found") or []
            )
        task_details.append(detail)

    agg_fields = NUMERIC_AGG_FIELDS + (RELATION_AGG_FIELDS if include_relation else ())
    equal_weight_average: Dict[str, Optional[float]] = {}
    sample_weighted_average: Dict[str, Optional[float]] = {}
    for field in agg_fields:
        values = []
        for t in tasks:
            v = t.get(field)
            if v is not None and np.isscalar(v) and np.isfinite(v):
                values.append(float(v))
        equal_weight_average[field] = float(np.mean(values)) if values else None

        weighted_values = []
        weights = []
        for t in tasks:
            v = t.get(field)
            n = t.get("n_val")
            if v is not None and np.isscalar(v) and np.isfinite(v) and n:
                weighted_values.append(float(v))
                weights.append(float(n))
        sample_weighted_average[field] = (
            float(np.average(weighted_values, weights=weights)) if weighted_values else None
        )

    wins = losses = ties = 0
    for t in tasks:
        delta = t.get("test_mae_delta")
        if delta is None or not np.isfinite(delta):
            ties += 1
        elif delta < -INTERACTION_TIE_TOLERANCE:
            wins += 1
        elif delta > INTERACTION_TIE_TOLERANCE:
            losses += 1
        else:
            ties += 1

    summary: Dict[str, Any] = {
        "task_details": task_details,
        "equal_weight_average": equal_weight_average,
        "sample_weighted_average": sample_weighted_average,
        "interaction_win_loss_count": {
            "interaction_wins": wins,
            "interaction_losses": losses,
            "interaction_ties": ties,
            "total_tasks": len(task_details),
        },
    }
    if include_relation:
        r_wins = r_losses = r_ties = 0
        for t in tasks:
            delta = t.get("test_mae_relation_delta")
            if delta is None or not np.isfinite(delta):
                r_ties += 1
            elif delta < -INTERACTION_TIE_TOLERANCE:
                r_wins += 1
            elif delta > INTERACTION_TIE_TOLERANCE:
                r_losses += 1
            else:
                r_ties += 1
        summary["relation_win_loss_count"] = {
            "relation_wins": r_wins,
            "relation_losses": r_losses,
            "relation_ties": r_ties,
            "total_tasks": len(task_details),
        }
    return summary


def _finite_metric(payload: Any, *path: str) -> Optional[float]:
    current = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if current is None or not np.isscalar(current):
        return None
    try:
        value = float(current)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def evaluate_relation_evidence_gate(
    tasks: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """至少一个成功任务真的消费到了关系边，否则本轮无法评估关系强度功能。

    字段缺失一律按"无证据"处理：旧版报告没有该字段，不能因此蒙混通过。
    """
    with_edges = [
        f"{t.get('dataset')} h={t.get('horizon')}"
        for t in tasks
        if t.get("status") == "ok" and (t.get("relation_strength_edges_found") or [])
    ]
    issues = []
    if not with_edges:
        issues.append(
            "no successful task consumed any relation edge "
            "(relation_strength_edges_found empty or missing for all tasks); "
            "this run cannot evaluate relation-strength scoring"
        )
    return {
        "passed": bool(with_edges),
        "issues": issues,
        "tasks_with_edges": len(with_edges),
        "tasks": with_edges,
    }


def evaluate_quality_gates(
    tasks: Sequence[Mapping[str, Any]],
    *,
    trace_root: Path,
    include_relation_gate: bool = True,
) -> Dict[str, Any]:
    """按 Task 7 预先写定口径计算可机器判定的质量门槛。

    ``include_relation_gate=False`` 复现关系强度接入前的门槛集合，供旧 schema
    报告（task7-shadow.3）复核使用。
    """
    task_keys = {(task.get("dataset"), task.get("horizon")) for task in tasks}
    expected_keys = {(spec["dataset"], spec["horizon"]) for spec in build_task_specs()}
    task_issues = []
    if len(tasks) != 9 or task_keys != expected_keys:
        task_issues.append("tasks are not the fixed nine-task set")
    task_issues.extend(
        f"{task.get('dataset')} h={task.get('horizon')}: status={task.get('status')}"
        for task in tasks
        if task.get("status") != "ok"
    )
    all_tasks = {"passed": not task_issues, "issues": task_issues}

    system_a_issues = []
    best_single_issues = []
    system_a_maes: List[float] = []
    protocol_b_maes: List[float] = []
    for task in tasks:
        label = f"{task.get('dataset')} h={task.get('horizon')}"
        system_a = task.get("system_a")
        matrix = task.get("matrix")
        best_single = task.get("best_single")
        a_mae = _finite_metric(system_a, "metrics", "mae")
        b_mae = _finite_metric(task, "protocol_b", "on", "mae")
        if not isinstance(system_a, Mapping) or system_a.get("status") != "ok":
            system_a_issues.append(f"{label}: System A reference is not ok")
        elif system_a.get("reference_mode") != "shared_prediction_matrix":
            system_a_issues.append(f"{label}: System A did not use shared prediction matrix")
        elif a_mae is None:
            system_a_issues.append(f"{label}: System A MAE is not finite")
        elif not isinstance(matrix, Mapping):
            system_a_issues.append(f"{label}: task matrix metadata is missing")
        else:
            if system_a.get("dataset") != task.get("dataset"):
                system_a_issues.append(f"{label}: System A dataset does not match task")
            if system_a.get("horizon") != task.get("horizon"):
                system_a_issues.append(f"{label}: System A horizon does not match task")
            if system_a.get("data_sha_test") != matrix.get("data_sha_test"):
                system_a_issues.append(f"{label}: System A test matrix hash does not match task")
            if system_a.get("n_test") != matrix.get("n_test"):
                system_a_issues.append(f"{label}: System A test row count does not match task")
            safe_models = set(matrix.get("safe_models") or [])
            selected_models = set(system_a.get("models") or [])
            if not selected_models or not selected_models.issubset(safe_models):
                system_a_issues.append(
                    f"{label}: System A selected models are not a subset of the shared pool"
                )
        if not isinstance(best_single, Mapping):
            best_single_issues.append(f"{label}: best-single reference is missing")
        else:
            if best_single.get("selection_uses_test_labels") is not False:
                best_single_issues.append(f"{label}: best-single selection is not validation-only")
            matrix_models = (
                set(matrix.get("safe_models") or [])
                if isinstance(matrix, Mapping)
                else set()
            )
            if best_single.get("model") not in matrix_models:
                best_single_issues.append(f"{label}: best-single model is outside shared pool")
        if a_mae is not None:
            system_a_maes.append(a_mae)
        if b_mae is not None:
            protocol_b_maes.append(b_mae)
    system_a_valid = {"passed": not system_a_issues, "issues": system_a_issues}
    best_single_valid = {"passed": not best_single_issues, "issues": best_single_issues}

    numeric_issues = []
    for task in tasks:
        label = f"{task.get('dataset')} h={task.get('horizon')}"
        on = _finite_metric(task, "test_mae_on")
        off = _finite_metric(task, "test_mae_off")
        delta = _finite_metric(task, "test_mae_delta")
        protocol_b_on = _finite_metric(task, "protocol_b", "on", "mae")
        if on is None or off is None or delta is None:
            numeric_issues.append(f"{label}: on/off/delta is missing or non-finite")
        elif not np.isclose(delta, on - off, rtol=0.0, atol=1e-9):
            numeric_issues.append(f"{label}: test_mae_delta does not equal on-off")
        if on is not None and protocol_b_on is not None and not np.isclose(
            on, protocol_b_on, rtol=0.0, atol=1e-9
        ):
            numeric_issues.append(f"{label}: protocol_b.on.mae differs from test_mae_on")
        if protocol_b_on is None:
            numeric_issues.append(f"{label}: Protocol B MAE is missing or non-finite")
        if _finite_metric(task, "best_single", "test", "mae") is None:
            numeric_issues.append(f"{label}: best-single MAE is missing or non-finite")
    numeric_consistency = {"passed": not numeric_issues, "issues": numeric_issues}

    average_passed = (
        len(system_a_maes) == 9
        and len(protocol_b_maes) == 9
        and float(np.mean(protocol_b_maes)) <= float(np.mean(system_a_maes)) * 1.01
    )
    average_vs_a = {
        "passed": bool(average_passed),
        "threshold_ratio": 1.01,
        "protocol_b_mean_mae": float(np.mean(protocol_b_maes)) if len(protocol_b_maes) == 9 else None,
        "system_a_mean_mae": float(np.mean(system_a_maes)) if len(system_a_maes) == 9 else None,
    }

    per_task_a_failures = []
    per_task_best_failures = []
    for task in tasks:
        label = f"{task.get('dataset')} h={task.get('horizon')}"
        b_mae = _finite_metric(task, "protocol_b", "on", "mae")
        a_mae = _finite_metric(task, "system_a", "metrics", "mae")
        best_mae = _finite_metric(task, "best_single", "test", "mae")
        if b_mae is None or a_mae is None or b_mae > a_mae * 1.03:
            per_task_a_failures.append(label)
        if b_mae is None or best_mae is None or b_mae > best_mae * 1.01:
            per_task_best_failures.append(label)
    per_task_vs_a = {
        "passed": not per_task_a_failures,
        "threshold_ratio": 1.03,
        "failures": per_task_a_failures,
    }
    per_task_vs_best = {
        "passed": not per_task_best_failures,
        "threshold_ratio": 1.01,
        "failures": per_task_best_failures,
    }

    trace_issues = []
    valid_traces = 0
    for spec in build_task_specs():
        for mode in ("on", "off"):
            relative = Path(spec["dataset"]) / f"h{spec['horizon']}" / f"protocol_b_trace_{mode}.json"
            path = trace_root / relative
            if not path.exists():
                trace_issues.append(f"missing:{relative}")
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                trace_issues.append(f"invalid:{relative}:{type(exc).__name__}")
                continue
            if not isinstance(payload, Mapping) or not isinstance(payload.get("stages"), list):
                trace_issues.append(f"invalid_schema:{relative}")
                continue
            valid_traces += 1
    trace_integrity = {
        "passed": not trace_issues and valid_traces == 18,
        "expected": 18,
        "valid": valid_traces,
        "issues": trace_issues,
    }

    gates = {
        "all_tasks_successful": all_tasks,
        "system_a_references_valid": system_a_valid,
        "best_single_references_valid": best_single_valid,
        "average_vs_system_a_1pct": average_vs_a,
        "per_task_vs_system_a_3pct": per_task_vs_a,
        "per_task_vs_best_single_1pct": per_task_vs_best,
        "numeric_consistency": numeric_consistency,
        "trace_integrity": trace_integrity,
    }
    if include_relation_gate:
        # §11#7：没有任何关系边被消费时，本轮无法评估关系强度，判失败而非通过
        gates["relation_strength_evidence"] = evaluate_relation_evidence_gate(tasks)
    gates["status"] = "passed" if all(gate["passed"] for gate in gates.values()) else "failed"
    return gates


def validate_shadow_report(
    report: Mapping[str, Any],
    *,
    trace_root: Optional[Path] = None,
) -> None:
    version = report.get("schema_version")
    if version == REPORT_SCHEMA_VERSION:
        relation_required = True
    elif version in LEGACY_REPORT_SCHEMA_VERSIONS:
        # 旧版报告按接入关系强度之前的口径复核，其证据仍然有效
        relation_required = False
    else:
        raise ValueError(f"unexpected schema_version: {version}")

    specs = report.get("task_specs")
    if specs != build_task_specs():
        raise ValueError("task_specs does not match the fixed nine-task set")

    tasks = report.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 9:
        raise ValueError(f"report must contain exactly 9 tasks, got {len(tasks) if isinstance(tasks, list) else type(tasks)}")

    expected_keys = {(s["dataset"], s["horizon"]) for s in build_task_specs()}
    actual_keys = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("each task must be a dict")
        key = (task.get("dataset"), task.get("horizon"))
        if key not in expected_keys:
            raise ValueError(f"unexpected task key: {key}")
        actual_keys.add(key)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        raise ValueError(f"missing tasks: {missing}")

    per_task_required = (
        "interaction_applied",
        "interaction_evaluated",
        "interaction_candidate_applied",
        "post_adjustment_applied",
        "final_prediction_contains_interaction",
        "interaction_status_reason",
        "val_mae_raw",
        "val_mae_interaction",
        "val_mae_delta",
        "oof_mae_raw",
        "oof_mae_interaction",
        "oof_mae_delta",
        "cv_oof_coverage",
        "test_mae_on",
        "test_mae_off",
        "test_mae_delta",
        "n_val",
        "horizon",
        "dataset",
        "protocol",
        "fallback_target",
        "selected_models",
        "system_a",
        "protocol_a",
        "protocol_b",
        "best_single",
        "guard",
        "selection_diff",
    )
    if relation_required:
        per_task_required = per_task_required + (
            "relation_warmup",
            "relation_contrast",
            "relation_strength_edges_found",
            "test_mae_relation_neutral",
            "test_mae_relation_delta",
        )
    for i, task in enumerate(tasks):
        if task.get("status") != "ok":
            raise ValueError(f"task {i} not successful: {task.get('error')}")
        for field in per_task_required:
            if field not in task:
                raise ValueError(f"task {i} missing required field {field}")

    aggregates = report.get("aggregates")
    if not isinstance(aggregates, dict):
        raise ValueError("report missing aggregates")
    for key in (
        "task_details",
        "equal_weight_average",
        "sample_weighted_average",
        "interaction_win_loss_count",
    ):
        if key not in aggregates:
            raise ValueError(f"aggregates missing {key}")
    if len(aggregates["task_details"]) != 9:
        raise ValueError("aggregates.task_details must contain 9 rows")
    recomputed_aggregates = aggregate_summary(tasks, include_relation=relation_required)
    if aggregates != recomputed_aggregates:
        raise ValueError("aggregates do not match recomputed task metrics")
    win_loss = aggregates["interaction_win_loss_count"]
    if win_loss.get("interaction_wins") is None or win_loss.get("interaction_losses") is None:
        raise ValueError("interaction_win_loss_count incomplete")
    if win_loss.get("interaction_wins") + win_loss.get("interaction_losses") + win_loss.get("interaction_ties", 0) != 9:
        raise ValueError("interaction_win_loss_count does not sum to 9")

    meta = report.get("_meta")
    if not isinstance(meta, dict):
        raise ValueError("report missing _meta")
    for key in ("code_commit", "random_seed", "data_hashes", "python", "environment"):
        if key not in meta:
            raise ValueError(f"_meta missing {key}")
    env = meta.get("environment")
    if not isinstance(env, dict):
        raise ValueError("_meta.environment must be a dict")
    for dep in KEY_DEPENDENCIES:
        if dep not in env:
            raise ValueError(f"_meta.environment missing dependency version {dep}")
    if meta.get("baseline_provenance_required"):
        provenance = meta.get("baseline_provenance")
        if not isinstance(provenance, Mapping) or provenance.get("status") != "verified":
            raise ValueError("_meta.baseline_provenance is required and must be verified")

    recorded_gates = report.get("quality_gates")
    if not isinstance(recorded_gates, Mapping):
        raise ValueError("report missing quality_gates")
    resolved_trace_root = trace_root
    if resolved_trace_root is None and meta.get("out_root"):
        candidate = Path(str(meta["out_root"]))
        if candidate.exists():
            resolved_trace_root = candidate
    if resolved_trace_root is None:
        raise ValueError("quality_gates validation requires an existing trace_root")
    recomputed_gates = evaluate_quality_gates(
        tasks, trace_root=resolved_trace_root, include_relation_gate=relation_required
    )
    if dict(recorded_gates) != recomputed_gates:
        raise ValueError("quality_gates do not match recomputed metrics or trace files")
    if recomputed_gates.get("status") != "passed":
        raise ValueError("quality_gates failed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Task 7 九任务 System A/B 影子对照")
    parser.add_argument("--pred-root", type=Path, default=PROJECT_ROOT / "reports" / "baselines")
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data" / "features")
    parser.add_argument("--feature-root", type=Path, default=PROJECT_ROOT / "data" / "features")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=None)
    parser.add_argument("--filter-threshold", type=float, default=DEFAULT_FILTER_THRESHOLD)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("MODELCOMBINE_SEED", "42")))
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pipeline.yaml",
        help="要求基线来源与此默认入口配置一致",
    )
    parser.add_argument(
        "--require-baseline-provenance",
        action="store_true",
        help="仅接受带有当前 pipeline.yaml 哈希和候选种子策略的基线预测",
    )
    parser.add_argument("--skip-combinator", action="store_true", help="跳过旧 System A 参考（默认运行）")
    parser.add_argument("--combinator-timeout", type=float, default=900.0)
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "result" / "ab_convergence" / "shadow_9tasks")
    args = parser.parse_args()

    pred_root = args.pred_root if args.pred_root.is_absolute() else PROJECT_ROOT / args.pred_root
    raw_root = args.raw_root if args.raw_root.is_absolute() else PROJECT_ROOT / args.raw_root
    feature_root = args.feature_root if args.feature_root.is_absolute() else PROJECT_ROOT / args.feature_root
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    out_root = args.out_root if args.out_root.is_absolute() else PROJECT_ROOT / args.out_root
    pipeline_config = (
        args.pipeline_config
        if args.pipeline_config.is_absolute()
        else PROJECT_ROOT / args.pipeline_config
    )
    # §11#7：v5 全程共用一张关系图，作为九个任务关系状态的**统一容器**。
    # 注意它不是"前一个任务写、后一个任务读"：消费侧按当前 scenario_id 精确
    # 匹配，而九个任务的 dataset/horizon 不同、场景 ID 也不同，跨任务读不到。
    # 闭环靠的是每个任务内部的 warm-up（写）→ 正式测量（只读）两轮。
    # 图为空时关系项恒为中性，报告的 relation_strength_evidence 门槛会判失败，
    # 提示该轮无法评估本功能，而不是让它悄悄显示通过。
    relation_graph = ModelGraph()
    print("关系图: 已创建（九任务共用容器；闭环为每任务内 warm-up -> 只读测量两轮）")

    baseline_provenance: Dict[str, Any] = {"status": "not_required"}
    if args.require_baseline_provenance:
        from scripts.train_baselines import load_verified_baseline_provenance

        try:
            baseline_provenance = {
                "status": "verified",
                **load_verified_baseline_provenance(pred_root, pipeline_config),
            }
        except ValueError as exc:
            print(f"[validate] baseline provenance failed: {exc}")
            return 2

    selected_datasets = set(args.datasets) if args.datasets else set(TASK_DATASETS)
    selected_horizons = set(args.horizons) if args.horizons else set(TASK_HORIZONS)
    specs = [
        s for s in build_task_specs()
        if s["dataset"] in selected_datasets and s["horizon"] in selected_horizons
    ]
    if not specs:
        parser.error("selected datasets/horizons produce no tasks")

    models = _build_kg_model_candidates()
    print(f"KG 基础候选模型: {models}")
    print(f"任务数: {len(specs)}（完整 9 任务需要 {len(build_task_specs())} 个）")
    print(f"run_combinator: {not args.skip_combinator}")
    if args.skip_combinator:
        print("已 --skip-combinator：System A 参考将记录为 not_run")

    tasks: List[Dict[str, Any]] = []
    for index, spec in enumerate(specs, 1):
        dataset, horizon = spec["dataset"], spec["horizon"]
        print(f"\n[{index}/{len(specs)}] {dataset} h={horizon}")
        record = run_task(
            relation_graph=relation_graph,
            dataset=dataset,
            horizon=horizon,
            models=models,
            pred_root=pred_root,
            raw_root=raw_root,
            out_root=out_root,
            filter_threshold=args.filter_threshold,
            seed=args.seed,
            run_combinator=not args.skip_combinator,
            combinator_timeout_seconds=args.combinator_timeout,
            feature_root=feature_root,
            baseline_provenance=(baseline_provenance if args.require_baseline_provenance else None),
        )
        print(
            f"  status={record['status']} test_mae_on={record.get('test_mae_on')} "
            f"test_mae_off={record.get('test_mae_off')} "
            f"interaction_applied={record.get('interaction_applied')}"
        )
        tasks.append(record)

    failed = [t for t in tasks if t.get("status") != "ok"]
    full_specs = build_task_specs()
    if len(specs) != len(full_specs):
        print(f"  [warn] 本次仅运行 {len(specs)}/{len(full_specs)} 个任务，不满足 9/9 门槛")

    data_hashes: Dict[str, str] = {}
    for spec in specs:
        ds = spec["dataset"]
        feature_train = feature_root / ds / "train.csv"
        data_hashes[ds] = _sha256_file(feature_train)

    full_run = len(specs) == len(full_specs)
    quality_gates = (
        evaluate_quality_gates(tasks, trace_root=out_root)
        if full_run
        else {
            "status": "not_evaluated",
            "reason": f"subset run contains {len(specs)}/9 tasks",
        }
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "task_specs": build_task_specs(),
        "run_specs": specs,
        "tasks": tasks,
        "aggregates": aggregate_summary(tasks),
        "quality_gates": quality_gates,
        "_meta": {
            "code_commit": _git_commit(),
            "random_seed": args.seed,
            "data_hashes": data_hashes,
            "python": sys.executable,
            "environment": _collect_dependency_versions(),
            "pred_root": str(pred_root),
            "pipeline_config": str(pipeline_config),
            "baseline_provenance_required": args.require_baseline_provenance,
            "baseline_provenance": baseline_provenance,
            "raw_root": str(raw_root),
            "out_root": str(out_root),
            "filter_threshold": args.filter_threshold,
            "run_combinator": not args.skip_combinator,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"\n报告已保存: {output}")

    if len(specs) == len(full_specs):
        try:
            validate_shadow_report(report, trace_root=out_root)
            print("[validate] 报告 schema、9 任务完整性与质量门槛检查通过")
        except ValueError as exc:
            print(f"[validate] 报告未通过: {exc}")
            return 2
    else:
        print("[validate] 跳过完整 9 任务校验（当前为 smoke/子集运行）")

    if failed:
        print(f"[fail] {len(failed)} 个任务失败")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
