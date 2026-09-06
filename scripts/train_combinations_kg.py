"""KG evaluation runner (modularized)."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
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
    ensure_dir,
)
from src.eval.kg.config import (
    KG_EXTENDED_POOL_MIN_LOADED,
    KG_SEASONAL_NAIVE_AS_FROZEN_EXPERT,
    KG_STRICT_EXTENDED_POOL,
    _build_extended_pool_strategies,
    _build_kg_model_candidates,
)
from src.eval.kg.data_io import _load_extended_pool_for_split
try:
    from src.eval.kg.data_io import _valid_pair_mae
except ImportError:
    # Backward-compatible fallback for older data_io.py on remote nodes.
    def _valid_pair_mae(y_true: np.ndarray, y_pred: np.ndarray):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        n_valid = int(mask.sum())
        n_total = len(y_true)
        valid_ratio = float(n_valid / n_total) if n_total > 0 else 0.0
        if n_valid <= 0:
            return float("inf"), valid_ratio, n_valid
        mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
        if not np.isfinite(mae):
            return float("inf"), valid_ratio, n_valid
        return mae, valid_ratio, n_valid
from src.eval.kg.protocol_a import kg_combination_pred_only
from src.eval.kg.protocol_b import (
    evaluate_fixed_protocol_b_combination,
    kg_combination_with_features,
)
from src.eval.kg.data_io import _align_raw_to_pred
from src.eval.kg.feedback import KGFeedbackStore, make_config_fingerprint
from src.core.scenario_id import compute_scenario_id
from src.models.artifacts import load_artifact, save_artifact
from src.models.combination_predictor import (
    COMBINATION_PREDICTOR_KEY,
    CombinationPredictor,
)
from src.models.trajectory_forecast import (
    calendar_frame,
    future_timestamps,
    generate_trajectory_matrix,
)
from src.storage.model_store import (
    SUPPORTED_FORECAST_STEPS,
    ModelStore,
    validate_forecast_steps,
)
from scripts.train_baselines import prepare_supervised
from src.eval.kg import load_candidate_audit_map, load_health_enabled_map
from src.core.enums import ModelLifecycleStage, TaskType
from src.core.index import IndexManager
from src.core.schema import ModelManifest
from src.core.solver import build_protocol_b_context, build_solver
from src.graph.model_graph import ModelGraph
from src.graph.temporal_relations import HawkesRelationUpdater
from src.models.uncertainty import UncertaintyGate


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _temporal_relations_enabled() -> bool:
    """实验开关：是否在 Protocol B 求解链挂载 Hawkes 时间衰减关系更新 stage。

    默认关闭，保持既有数值/行为不变；置真时按调度反馈事件动态更新模型关系
    强度，并在 SelectionTrace 中产出 TemporalRelationUpdate 记录。
    """
    return _env_flag("MODELCOMBINE_KG_ENABLE_TEMPORAL_RELATIONS")


def _build_temporal_relation_updater() -> HawkesRelationUpdater:
    """从环境变量读取 Hawkes 核参数，缺省用库内默认（base=0.5, alpha=0.2）。"""
    kwargs: Dict[str, Any] = {}
    for env_name, key in (
        ("MODELCOMBINE_KG_TEMPORAL_BASE_STRENGTH", "base_strength"),
        ("MODELCOMBINE_KG_TEMPORAL_ALPHA", "alpha"),
        ("MODELCOMBINE_KG_TEMPORAL_BETA", "beta"),
    ):
        raw = os.environ.get(env_name)
        if raw is not None and raw.strip():
            kwargs[key] = float(raw)
    return HawkesRelationUpdater(**kwargs)


def _dump_temporal_relation_snapshot(graph: ModelGraph, path: Path) -> None:
    """导出动态关系图快照（节点/边+权重），便于离线核对关系强度变化。"""
    edges = [
        {
            "source": src,
            "target": dst,
            "edge_type": data.get("edge_type"),
            "weight": data.get("weight"),
            "dynamic_strength": data.get("dynamic_strength"),
            "event_count": data.get("event_count"),
            "last_event_at": data.get("last_event_at"),
            "event_evidence_refs": data.get("event_evidence_refs", []),
        }
        for src, dst, data in graph.G.edges(data=True)
    ]
    snapshot = {
        "n_nodes": graph.G.number_of_nodes(),
        "n_edges": graph.G.number_of_edges(),
        "edges": edges,
    }
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _stable_json_hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _parse_csv_env_str_set(name: str, default: str = "") -> set:
    raw = os.environ.get(name, default)
    if raw is None:
        return set()
    vals = [v.strip() for v in str(raw).split(",")]
    return {v for v in vals if v}


def _parse_csv_env_int_set(name: str, default: str = "") -> set:
    out = set()
    for v in _parse_csv_env_str_set(name, default=default):
        try:
            out.add(int(v))
        except Exception:
            continue
    return out


def _build_runtime_manifests(
    model_ids: List[str],
    business_domain: str = "load_forecast",
) -> Dict[str, ModelManifest]:
    """Build task-local manifests after the existing KG filtering gates.

    The script has already applied health, audit, MASE and safe-column gates before
    Protocol B. These runtime manifests keep the solver capability stage auditable
    without changing the already-approved safe model pool.
    """
    return {
        model_id: ModelManifest(
            model_id=model_id,
            task_types=[TaskType.FORECASTING],
            business_domains=[business_domain],
            input_constraints={"features": []},
            output_schema={"yhat": "float"},
            resource_cost={},
            lifecycle_stage=ModelLifecycleStage.ACTIVE,
        )
        for model_id in model_ids
    }


def _protocol_b_trace_path(out_root: Path, dataset: str, horizon: int) -> Path:
    return out_root / dataset / "traces" / f"protocol_b_solver_h{int(horizon)}.json"


def _run_protocol_b_with_solver(
    *,
    dataset: str,
    horizon: int,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    df_raw_val: Optional[pd.DataFrame],
    df_raw_test: Optional[pd.DataFrame],
    model_cols: List[str],
    base_model_cols: List[str],
    feedback_store: KGFeedbackStore,
    trace_path: Optional[Path] = None,
    signal_ablation_profile_paths: Optional[List[Path]] = None,
    signal_kg_result_paths: Optional[List[Path]] = None,
    temporal_relation_graph: Optional[ModelGraph] = None,
) -> tuple[Dict[str, Any], Any]:
    manifests = _build_runtime_manifests(model_cols)
    if signal_ablation_profile_paths or signal_kg_result_paths:
        index_manager = IndexManager.with_report_signals(
            manifests=manifests,
            ablation_profile_paths=signal_ablation_profile_paths,
            kg_result_paths=signal_kg_result_paths,
        )
    else:
        index_manager = IndexManager.with_defaults(manifests=manifests)
    ctx = build_protocol_b_context(
        dataset=dataset,
        horizon=horizon,
        df_val=df_val,
        df_test=df_test,
        df_raw_val=df_raw_val,
        df_raw_test=df_raw_test,
        model_cols=model_cols,
        base_model_cols=base_model_cols,
        feedback_store=feedback_store,
    )
    solver_kwargs: Dict[str, Any] = dict(
        manifests=manifests,
        index_manager=index_manager,
        uncertainty_gate=UncertaintyGate(threshold=float("inf")),
    )
    # 实验开关：仅当传入动态关系图时才挂载 Hawkes 时序关系 stage，
    # 保证默认路径（关闭）与既有 build_solver 调用签名/数值完全不变。
    if temporal_relation_graph is not None:
        solver_kwargs.update(
            temporal_relation_graph=temporal_relation_graph,
            temporal_relation_updater=_build_temporal_relation_updater(),
            temporal_relation_create_missing=True,
        )
    solver = build_solver("protocol_b", **solver_kwargs)
    normalized, trace = solver.solve(
        ctx,
        trace_path=str(trace_path) if trace_path is not None else None,
    )
    raw = normalized.get("raw", normalized) if isinstance(normalized, dict) else normalized
    return raw, trace


def _infer_seasonal_period(df: pd.DataFrame) -> int:
    if "timestamp" not in df.columns:
        return 24
    try:
        ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna().sort_values()
        if len(ts) >= 3:
            delta = ts.diff().dropna().median()
            if pd.notna(delta):
                delta_sec = float(delta.total_seconds())
                if np.isfinite(delta_sec) and delta_sec > 0:
                    inferred = int(round(86400.0 / delta_sec))
                    if 2 <= inferred <= 24 * 12:
                        return inferred
    except Exception:
        return 24
    return 24

def _estimate_mase_like(
    df_val: pd.DataFrame,
    model: str,
    seasonal_period: int,
) -> Optional[float]:
    if model not in df_val.columns or "y" not in df_val.columns:
        return None
    y = np.asarray(df_val["y"].values, dtype=float)
    pred = np.asarray(df_val[model].values, dtype=float)
    if len(y) != len(pred) or len(y) < 5:
        return None
    finite_mask = np.isfinite(y) & np.isfinite(pred)
    if int(finite_mask.sum()) < max(8, len(y) // 20):
        return None
    abs_err = np.abs(y[finite_mask] - pred[finite_mask])
    model_mae = float(np.mean(abs_err))
    if not np.isfinite(model_mae):
        return None

    denom = None
    if "seasonal_naive" in df_val.columns:
        seasonal_pred = np.asarray(df_val["seasonal_naive"].values, dtype=float)
        seasonal_mask = np.isfinite(y) & np.isfinite(seasonal_pred)
        if int(seasonal_mask.sum()) > 0:
            base_mae = float(np.mean(np.abs(y[seasonal_mask] - seasonal_pred[seasonal_mask])))
        else:
            base_mae = float("nan")
        if np.isfinite(base_mae) and base_mae > 1e-12:
            denom = base_mae
    if denom is None and len(y) > seasonal_period:
        naive_abs = np.abs(y[seasonal_period:] - y[:-seasonal_period])
        if len(naive_abs) > 0:
            base_mae = float(np.mean(naive_abs))
            if np.isfinite(base_mae) and base_mae > 1e-12:
                denom = base_mae
    if denom is None or denom <= 1e-12:
        return None
    return float(model_mae / denom)


def _finite_mae_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    valid_pairs = int(mask.sum())
    total = int(len(y_true))
    valid_ratio = float(valid_pairs / total) if total > 0 else 0.0
    if valid_pairs <= 0:
        return {"mae": None, "valid_pairs": valid_pairs, "valid_ratio": valid_ratio}
    mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
    if not np.isfinite(mae):
        mae = None
    return {"mae": mae, "valid_pairs": valid_pairs, "valid_ratio": valid_ratio}


def _impute_prediction_nans(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    model_cols: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """按列用 val 中位数填补预测缺失，保证 KG 训练矩阵可数值化。"""
    out_val = df_val.copy()
    out_test = df_test.copy()
    impute_meta: Dict[str, Dict[str, Any]] = {}
    global_fallback = float(np.nanmean(np.asarray(df_val["y"].values, dtype=float)))
    if not np.isfinite(global_fallback):
        global_fallback = 0.0

    for m in model_cols:
        if m not in out_val.columns or m not in out_test.columns:
            continue
        v_val = np.asarray(out_val[m].values, dtype=float)
        v_test = np.asarray(out_test[m].values, dtype=float)
        miss_val = ~np.isfinite(v_val)
        miss_test = ~np.isfinite(v_test)
        if not (bool(miss_val.any()) or bool(miss_test.any())):
            continue
        finite_val = v_val[np.isfinite(v_val)]
        if finite_val.size > 0:
            fill_value = float(np.median(finite_val))
            fill_source = "val_median"
        else:
            finite_test = v_test[np.isfinite(v_test)]
            if finite_test.size > 0:
                fill_value = float(np.median(finite_test))
                fill_source = "test_median"
            else:
                fill_value = global_fallback
                fill_source = "global_y_mean"
        v_val = np.where(np.isfinite(v_val), v_val, fill_value)
        v_test = np.where(np.isfinite(v_test), v_test, fill_value)
        out_val[m] = v_val
        out_test[m] = v_test
        impute_meta[m] = {
            "fill_source": fill_source,
            "fill_value": fill_value,
            "val_missing_count": int(miss_val.sum()),
            "test_missing_count": int(miss_test.sum()),
        }
    return out_val, out_test, impute_meta


def _compute_slice_evidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_slices: int = 3,
) -> Dict[str, Any]:
    """P1-3: 分时段 MAE 证据（头段/中段/尾段 + 高误差区域）。

    用于诊断 val 改善是否在时间维度上均匀分布。
    """
    n = len(y_true)
    if n < 6:
        return {"skip": "too_few_samples"}
    ae = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    finite_mask = np.isfinite(ae)

    edges = np.linspace(0, n, n_slices + 1, dtype=int)
    temporal_slices = {}
    labels = ["head", "mid", "tail"] if n_slices == 3 else [f"slice_{i}" for i in range(n_slices)]
    for i in range(n_slices):
        s, e = int(edges[i]), int(edges[i + 1])
        sl = ae[s:e]
        sl_mask = finite_mask[s:e]
        valid = int(sl_mask.sum())
        if valid > 0:
            temporal_slices[labels[i]] = {
                "mae": float(np.mean(sl[sl_mask])),
                "n": int(e - s),
                "n_valid": valid,
            }

    # 高误差区域（top-10% absolute error 的 MAE）
    high_err_meta = {}
    valid_ae = ae[finite_mask]
    if len(valid_ae) > 10:
        p90 = float(np.percentile(valid_ae, 90))
        high_mask = ae >= p90
        high_n = int(high_mask.sum())
        if high_n > 0:
            high_err_meta = {
                "p90_threshold": p90,
                "n_high_error": high_n,
                "mae_high_error": float(np.mean(ae[high_mask])),
            }

    # tail_full_ratio：尾段 MAE / 全段 MAE
    tail_label = labels[-1] if labels else "tail"
    full_mae = float(np.mean(valid_ae)) if len(valid_ae) > 0 else None
    tail_mae = temporal_slices.get(tail_label, {}).get("mae")
    tail_full_ratio = None
    if (
        full_mae is not None
        and tail_mae is not None
        and np.isfinite(full_mae)
        and np.isfinite(tail_mae)
        and full_mae > 1e-10
    ):
        tail_full_ratio = float(tail_mae / full_mae)

    return {
        "temporal_slices": temporal_slices,
        "high_error": high_err_meta,
        "full_mae": full_mae,
        "tail_full_ratio": tail_full_ratio,
    }


def _resolve_pool_mode(extended_pool: bool, has_candidate_audit: bool) -> str:
    if not extended_pool:
        return "g0_no_extended"
    if has_candidate_audit:
        return "g2_extended_with_audit"
    return "g1_extended_no_audit"

def run_dataset_kg(dataset: str, horizons: List[int], models: List[str],
                   pred_root: Path, raw_root: Optional[Path], out_root: Path,
                   filter_threshold: float = 2.0,
                   health_enabled_map: Optional[Dict[str, Dict[str, List[str]]]] = None,
                   extended_pool: bool = False,
                   naive_as_frozen_expert: bool = KG_SEASONAL_NAIVE_AS_FROZEN_EXPERT,
                   combo_root: Optional[Path] = None,
                   allow_in_sample_extended_pool: bool = False,
                   strict_extended_pool: bool = KG_STRICT_EXTENDED_POOL,
                   min_extended_loaded: int = KG_EXTENDED_POOL_MIN_LOADED,
                   candidate_audit_map: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
                   strict_candidate_audit: bool = False,
                   min_audit_accepted: int = 0,
                   seed: int = 42,
                   eligible_mase_hard: float = float("inf"),
                   signal_ablation_profile_paths: Optional[List[Path]] = None,
                   signal_kg_result_paths: Optional[List[Path]] = None) -> Dict:
    """运行单个数据集的 KG 评估"""
    results = {}

    # 闭环反馈：在 horizon 循环外初始化，跨步长写回（h=1 → h=6 → h=24）。
    # 若上次运行保存了 feedback 文件，则自动加载以实现跨 session 学习。
    feedback_lr = float(os.environ.get("MODELCOMBINE_KG_FEEDBACK_LR", "0.1"))
    feedback_store = KGFeedbackStore(learning_rate=feedback_lr)
    feedback_save_path = out_root / f"{dataset}_kg_feedback.pkl"
    config_fingerprint = make_config_fingerprint(
        list(models),
        extra={
            "dataset": dataset,
            "extended_pool": bool(extended_pool),
            "has_candidate_audit": bool(candidate_audit_map),
            "strict_candidate_audit": bool(strict_candidate_audit),
            "min_audit_accepted": int(min_audit_accepted),
            "filter_threshold": round(float(filter_threshold), 6),
            "seed": int(seed),
            "strict_extended_pool": bool(strict_extended_pool),
            "min_extended_loaded": int(min_extended_loaded),
            "allow_in_sample_extended_pool": bool(allow_in_sample_extended_pool),
        },
    )
    if os.environ.get("MODELCOMBINE_KG_FEEDBACK_LOAD_PREV", "0").strip() in {"1", "true", "yes"}:
        feedback_store.load(str(feedback_save_path), config_fingerprint=config_fingerprint)

    # 实验开关：Hawkes 时序关系图在 horizon 循环外初始化，跨步长累积调度反馈事件。
    # 默认关闭；置真时每次 Protocol B 求解后写入 scenario→path/model 关系边并落 trace。
    temporal_relation_graph = ModelGraph() if _temporal_relations_enabled() else None
    if temporal_relation_graph is not None:
        print("  [temporal] Hawkes 时序关系 stage 已启用（实验模式）")

    extended_strategies = _build_extended_pool_strategies()
    pool_mode = _resolve_pool_mode(extended_pool=extended_pool, has_candidate_audit=bool(candidate_audit_map))
    eligible_definition = {
        "pred_file_exists_val_test": True,
        "health_pass_required": bool(health_enabled_map),
        "alignment_row_id_required": True,
        "safe_cols_rule": "val_mae<=best_val_mae*r_or_kmin_backfill",
        "kmin": 3,
        "kmax": None,
        "r": float(filter_threshold),
        "mase_hard": float(eligible_mase_hard) if np.isfinite(eligible_mase_hard) else None,
    }
    eligible_definition_hash = _stable_json_hash(eligible_definition)

    for h in horizons:
        print(f"  处理 {dataset} h={h}...")
        
        try:
            # 基础模型走严格 y 对齐校验；seasonal_naive 作为 frozen expert
            # 通过 row_id merge 单独接入，避免 DST 重复时间戳导致 y 校验误判。
            load_models = list(models)
            df_val = load_predictions_safe(pred_root, dataset, h, load_models, "val")
            df_test = load_predictions_safe(pred_root, dataset, h, load_models, "test")
        except (FileNotFoundError, ValueError) as e:
            print(f"    跳过: {e}")
            continue

        extended_meta = {"enabled": False}
        frozen_naive_meta: Dict[str, Any] = {
            "enabled": bool(naive_as_frozen_expert),
            "loaded": False,
            "skip_reason": None,
            "source": None,
            "val_loaded": [],
            "test_loaded": [],
        }
        if naive_as_frozen_expert and not extended_pool:
            naive_val, naive_val_meta = _load_extended_pool_for_split(
                df_val,
                split="val",
                dataset=dataset,
                horizon=h,
                pred_root=pred_root,
                combo_root=combo_root,
                strategies=["seasonal_naive"],
                allow_in_sample=allow_in_sample_extended_pool,
            )
            naive_test, naive_test_meta = _load_extended_pool_for_split(
                df_test,
                split="test",
                dataset=dataset,
                horizon=h,
                pred_root=pred_root,
                combo_root=combo_root,
                strategies=["seasonal_naive"],
                allow_in_sample=allow_in_sample_extended_pool,
            )
            df_val, df_test = naive_val, naive_test
            val_loaded = naive_val_meta.get("loaded", []) if isinstance(naive_val_meta, dict) else []
            test_loaded = naive_test_meta.get("loaded", []) if isinstance(naive_test_meta, dict) else []
            frozen_naive_meta["val_loaded"] = list(val_loaded) if isinstance(val_loaded, list) else []
            frozen_naive_meta["test_loaded"] = list(test_loaded) if isinstance(test_loaded, list) else []
            frozen_naive_meta["source"] = "standalone_row_id_merge"
            frozen_naive_meta["loaded"] = bool(
                "seasonal_naive" in frozen_naive_meta["val_loaded"]
                and "seasonal_naive" in frozen_naive_meta["test_loaded"]
            )
            if not frozen_naive_meta["loaded"]:
                frozen_naive_meta["skip_reason"] = "missing_or_merge_failed"
            else:
                print("    frozen_naive: seasonal_naive 已通过 row_id merge 接入")

        if extended_pool:
            ext_val, ext_val_meta = _load_extended_pool_for_split(
                df_val,
                split="val",
                dataset=dataset,
                horizon=h,
                pred_root=pred_root,
                combo_root=combo_root,
                strategies=extended_strategies,
                allow_in_sample=allow_in_sample_extended_pool,
            )
            ext_test, ext_test_meta = _load_extended_pool_for_split(
                df_test,
                split="test",
                dataset=dataset,
                horizon=h,
                pred_root=pred_root,
                combo_root=combo_root,
                strategies=extended_strategies,
                allow_in_sample=allow_in_sample_extended_pool,
            )
            df_val, df_test = ext_val, ext_test
            val_loaded_n = len(ext_val_meta.get("loaded", []))
            test_loaded_n = len(ext_test_meta.get("loaded", []))
            common_loaded_n = len(
                set(ext_val_meta.get("loaded", [])) & set(ext_test_meta.get("loaded", []))
            )
            extended_meta = {
                "enabled": True,
                "val": ext_val_meta,
                "test": ext_test_meta,
                "strict": bool(strict_extended_pool),
                "min_loaded": int(min_extended_loaded),
                "val_loaded_count": int(val_loaded_n),
                "test_loaded_count": int(test_loaded_n),
                "common_loaded_count": int(common_loaded_n),
            }
            if ext_val_meta["loaded"] or ext_test_meta["loaded"]:
                print(
                    f"    扩展候选池: val_loaded={ext_val_meta['loaded']} "
                    f"test_loaded={ext_test_meta['loaded']}"
                )
            if naive_as_frozen_expert:
                val_loaded_ext = ext_val_meta.get("loaded", []) if isinstance(ext_val_meta, dict) else []
                test_loaded_ext = ext_test_meta.get("loaded", []) if isinstance(ext_test_meta, dict) else []
                val_has_naive = isinstance(val_loaded_ext, list) and ("seasonal_naive" in val_loaded_ext)
                test_has_naive = isinstance(test_loaded_ext, list) and ("seasonal_naive" in test_loaded_ext)
                frozen_naive_meta["source"] = "extended_pool"
                frozen_naive_meta["val_loaded"] = ["seasonal_naive"] if val_has_naive else []
                frozen_naive_meta["test_loaded"] = ["seasonal_naive"] if test_has_naive else []
                frozen_naive_meta["loaded"] = bool(val_has_naive and test_has_naive)
                if not frozen_naive_meta["loaded"]:
                    frozen_naive_meta["skip_reason"] = "missing_in_extended_pool_val_or_test"
                else:
                    frozen_naive_meta["skip_reason"] = None
            if strict_extended_pool:
                if (
                    val_loaded_n < min_extended_loaded
                    or test_loaded_n < min_extended_loaded
                    or common_loaded_n < min_extended_loaded
                ):
                    raise RuntimeError(
                        "strict_extended_pool_failed:"
                        f"{dataset} h={h}, val_loaded={val_loaded_n}, "
                        f"test_loaded={test_loaded_n}, common_loaded={common_loaded_n}, "
                        f"min_required={min_extended_loaded}"
                    )
        
        candidate_models = list(models)
        if naive_as_frozen_expert:
            candidate_models = list(dict.fromkeys(candidate_models + ["seasonal_naive"]))
        if extended_pool:
            candidate_models = sorted(set(candidate_models) | set(extended_strategies))
        model_cols = get_common_models(df_val, df_test, candidate_models)
        if len(model_cols) < 2:
            print(f"    跳过: 模型数不足")
            continue

        eligible_filter_reasons: Dict[str, List[str]] = {}
        model_mae_map: Dict[str, float] = {}
        model_fair_mae_map: Dict[str, float] = {}
        model_valid_ratio_map: Dict[str, float] = {}
        model_valid_pairs_map: Dict[str, int] = {}
        y_val_arr = np.asarray(df_val["y"].values, dtype=float)
        for m in candidate_models:
            eligible_filter_reasons[m] = []
            if m not in model_cols:
                eligible_filter_reasons[m].append("missing_pred_file_val_or_test")
            else:
                eligible_filter_reasons[m].append("alignment_ok")
                try:
                    fair_mae, v_ratio, v_count = _valid_pair_mae(
                        y_val_arr, np.asarray(df_val[m].values, dtype=float)
                    )
                    model_valid_ratio_map[m] = v_ratio
                    model_valid_pairs_map[m] = v_count
                    model_fair_mae_map[m] = fair_mae
                    if v_count <= 0 or not np.isfinite(fair_mae):
                        model_mae_map[m] = float("inf")
                        eligible_filter_reasons[m].append(
                            f"insufficient_valid_pairs:{v_count}"
                        )
                    else:
                        model_mae_map[m] = fair_mae
                except Exception:
                    model_mae_map[m] = float("inf")
                    model_fair_mae_map[m] = float("inf")
                    model_valid_ratio_map[m] = 0.0
                    model_valid_pairs_map[m] = 0
                    eligible_filter_reasons[m].append("mae_compute_failed")

        # 弱模型过滤
        df_val_for_filter, _, filter_impute_meta = _impute_prediction_nans(df_val, df_test, model_cols)
        safe_models, filter_meta = filter_weak_models(
            df_val_for_filter, model_cols, threshold_ratio=filter_threshold, horizon=h
        )
        filter_meta["nan_imputation_for_filter"] = filter_impute_meta
        print(f"    过滤: {len(model_cols)} -> {len(safe_models)} 模型")

        # 健康白名单过滤（与 modelcombine 的健康诊断对齐）
        health_meta = {
            "applied": False,
            "enabled_models": None,
            "removed_models": [],
            "skip_reason": None,
        }
        audit_meta = {
            "applied": False,
            "accepted_models": [],
            "removed_models": [],
            "skip_reason": None,
            "strict": bool(strict_candidate_audit),
            "min_accepted": int(min_audit_accepted),
        }
        if health_enabled_map:
            enabled_models = health_enabled_map.get(dataset, {}).get(str(h), [])
            if enabled_models:
                enabled_set = set(enabled_models)
                base_models_in_safe = [m for m in safe_models if m in models]
                extended_in_safe = [m for m in safe_models if m not in models]
                removed_models = [m for m in base_models_in_safe if m not in enabled_set]
                filtered_models = [m for m in base_models_in_safe if m in enabled_set] + extended_in_safe
                health_meta["enabled_models"] = sorted(enabled_models)
                health_meta["removed_models"] = removed_models
                for m in removed_models:
                    eligible_filter_reasons.setdefault(m, []).append("health_disabled")
                if len(filtered_models) >= 2:
                    safe_models = filtered_models
                    health_meta["applied"] = True
                    if removed_models:
                        print(f"    健康白名单过滤: 移除 {removed_models}")
                else:
                    health_meta["skip_reason"] = (
                        f"白名单后模型不足2个 ({len(filtered_models)})，保留原 safe_models"
                    )
                    print(f"    健康白名单过滤跳过: {health_meta['skip_reason']}")
            else:
                health_meta["skip_reason"] = "当前任务未找到 enabled 白名单"

        if candidate_audit_map:
            task_audit = candidate_audit_map.get(dataset, {}).get(str(h))
            if isinstance(task_audit, dict):
                accepted = task_audit.get("accepted", [])
                accepted_set = set(accepted if isinstance(accepted, list) else [])
                task_candidates = task_audit.get("candidates", {})
                audited_candidates = set(task_candidates.keys()) if isinstance(task_candidates, dict) else set()
                base_models_in_safe = [m for m in safe_models if m in models]
                ext_models_in_safe = [m for m in safe_models if m not in models]
                ext_kept_audited = [m for m in ext_models_in_safe if (m in audited_candidates) and (m in accepted_set)]
                ext_kept = ext_kept_audited
                ext_removed = [
                    m for m in ext_models_in_safe
                    if (m in audited_candidates) and (m not in accepted_set)
                ]
                ext_removed_unaudited = [m for m in ext_models_in_safe if m not in audited_candidates]
                # unaudited 模型 soft-pass（默认关闭）。支持按数据集/步长/模型白名单限域启用，
                # 避免 global soft-pass 污染 pjm 等稳定任务。
                soft_pass_enabled = str(os.environ.get(
                    "MODELCOMBINE_KG_AUDIT_SOFT_PASS_ENABLED", "0"
                )).strip().lower() in {"1", "true", "yes", "on"}
                _soft_pass_ratio = float(os.environ.get(
                    "MODELCOMBINE_KG_AUDIT_SOFT_PASS_RATIO", "2.0"
                ))
                _soft_pass_max_per_task = max(int(os.environ.get(
                    "MODELCOMBINE_KG_AUDIT_SOFT_PASS_MAX_PER_TASK", "2"
                )), 0)
                _soft_pass_dataset_scope = _parse_csv_env_str_set(
                    "MODELCOMBINE_KG_AUDIT_SOFT_PASS_DATASETS", ""
                )
                _soft_pass_horizon_scope = _parse_csv_env_int_set(
                    "MODELCOMBINE_KG_AUDIT_SOFT_PASS_HORIZONS", ""
                )
                _soft_pass_allowlist = _parse_csv_env_str_set(
                    "MODELCOMBINE_KG_AUDIT_SOFT_PASS_ALLOWLIST", ""
                )
                soft_pass_task_enabled = bool(soft_pass_enabled)
                if soft_pass_task_enabled and _soft_pass_dataset_scope and dataset not in _soft_pass_dataset_scope:
                    soft_pass_task_enabled = False
                if soft_pass_task_enabled and _soft_pass_horizon_scope and int(h) not in _soft_pass_horizon_scope:
                    soft_pass_task_enabled = False
                best_base_mae = min(
                    (model_mae_map.get(m, float("inf")) for m in base_models_in_safe),
                    default=float("inf"),
                )
                soft_pass_candidates = []
                ext_still_removed_unaudited = []
                for m in ext_removed_unaudited:
                    m_mae = model_mae_map.get(m, float("inf"))
                    if (
                        soft_pass_task_enabled
                        and (not _soft_pass_allowlist or m in _soft_pass_allowlist)
                        and
                        np.isfinite(m_mae)
                        and np.isfinite(best_base_mae)
                        and best_base_mae > 0
                        and m_mae <= best_base_mae * _soft_pass_ratio
                    ):
                        soft_pass_candidates.append((m, float(m_mae)))
                    else:
                        ext_still_removed_unaudited.append(m)
                soft_pass_candidates.sort(key=lambda kv: kv[1])
                ext_soft_passed = [m for m, _ in soft_pass_candidates[:_soft_pass_max_per_task]]
                soft_pass_not_selected = {m for m, _ in soft_pass_candidates[_soft_pass_max_per_task:]}
                if soft_pass_not_selected:
                    ext_still_removed_unaudited.extend(sorted(soft_pass_not_selected))
                ext_kept = ext_kept_audited + ext_soft_passed
                audit_meta["accepted_models"] = ext_kept_audited
                audit_meta["soft_pass_enabled"] = bool(soft_pass_enabled)
                audit_meta["soft_pass_task_enabled"] = bool(soft_pass_task_enabled)
                audit_meta["soft_pass_ratio"] = float(_soft_pass_ratio)
                audit_meta["soft_pass_max_per_task"] = int(_soft_pass_max_per_task)
                audit_meta["soft_pass_dataset_scope"] = sorted(_soft_pass_dataset_scope)
                audit_meta["soft_pass_horizon_scope"] = sorted(_soft_pass_horizon_scope)
                audit_meta["soft_pass_allowlist"] = sorted(_soft_pass_allowlist)
                audit_meta["soft_passed_models"] = ext_soft_passed
                audit_meta["removed_models"] = ext_removed
                audit_meta["removed_unaudited_models"] = ext_still_removed_unaudited
                audit_meta["pass_through_models"] = []
                for m in ext_removed:
                    eligible_filter_reasons.setdefault(m, []).append("candidate_audit_rejected")
                for m in ext_still_removed_unaudited:
                    eligible_filter_reasons.setdefault(m, []).append("candidate_audit_missing_record")
                for m in ext_soft_passed:
                    eligible_filter_reasons.setdefault(m, []).append("candidate_audit_soft_pass")
                if strict_candidate_audit and len(ext_kept_audited) < int(min_audit_accepted):
                    raise RuntimeError(
                        "strict_candidate_audit_failed:"
                        f"{dataset} h={h}, accepted={len(ext_kept_audited)}, "
                        f"min_required={int(min_audit_accepted)}, accepted_models={ext_kept_audited}"
                    )
                filtered_models = base_models_in_safe + ext_kept
                if len(filtered_models) >= 2:
                    safe_models = filtered_models
                    audit_meta["applied"] = True
                    removed_all = ext_removed + ext_still_removed_unaudited
                    if removed_all:
                        print(f"    候选审计过滤: 移除 {removed_all}")
                    if ext_soft_passed:
                        print(
                            f"    候选审计soft-pass(ENABLED={soft_pass_enabled}, task_enabled={soft_pass_task_enabled}, "
                            f"ratio={_soft_pass_ratio}, max={_soft_pass_max_per_task}): "
                            f"{ext_soft_passed}"
                        )
                else:
                    audit_meta["skip_reason"] = (
                        f"审计过滤后模型不足2个 ({len(filtered_models)})，保留原 safe_models"
                    )
                    print(f"    候选审计过滤跳过: {audit_meta['skip_reason']}")
            else:
                audit_meta["skip_reason"] = "当前任务未找到 candidate audit 记录"
                if strict_candidate_audit and int(min_audit_accepted) > 0:
                    raise RuntimeError(
                        f"strict_candidate_audit_failed:{dataset} h={h}, "
                        "missing_task_audit_record"
                    )

        seasonal_period = _infer_seasonal_period(df_val)
        for m in model_cols:
            if m not in safe_models:
                if m in (filter_meta.get("stability_removed") or {}):
                    reason = f"safe_cols_stability:{filter_meta['stability_removed'][m]}"
                else:
                    soft_th = filter_meta.get("soft_threshold")
                    mae_v = model_mae_map.get(m)
                    if mae_v is not None and np.isfinite(mae_v) and soft_th is not None:
                        reason = f"safe_cols_threshold:{mae_v:.4f}>{float(soft_th):.4f}"
                    elif model_valid_pairs_map.get(m, 0) <= 0:
                        reason = "safe_cols_threshold:insufficient_valid_pairs"
                    else:
                        reason = "safe_cols_filtered"
                eligible_filter_reasons.setdefault(m, []).append(reason)
        if np.isfinite(eligible_mase_hard):
            mase_removed: List[str] = []
            for m in list(safe_models):
                mase_like = _estimate_mase_like(df_val, m, seasonal_period=seasonal_period)
                if mase_like is not None and np.isfinite(mase_like) and mase_like > float(eligible_mase_hard):
                    mase_removed.append(m)
                    eligible_filter_reasons.setdefault(m, []).append(
                        f"mase_hard:{mase_like:.4f}>{float(eligible_mase_hard):.4f}"
                    )
            if mase_removed:
                candidate_kept = [m for m in safe_models if m not in mase_removed]
                if len(candidate_kept) >= 2:
                    safe_models = candidate_kept
                else:
                    print("    MASE 硬阈值过滤后不足2个模型，跳过本次 MASE 过滤")

        if len(safe_models) < 2:
            print(f"    跳过: 白名单/过滤后模型数不足 ({len(safe_models)})")
            continue

        # Protocol A/B 前对 safe_models 缺失预测做稳健填补，避免 NaN 传播到 Ridge。
        df_val_kg, df_test_kg, safe_impute_meta = _impute_prediction_nans(df_val, df_test, safe_models)
        
        # 尝试加载原始特征
        df_raw_val = None
        df_raw_test = None
        if raw_root:
            val_candidates = [
                raw_root / dataset / f"val_h{h}.csv",
                raw_root / dataset / "val.csv",
                raw_root / dataset / f"val_{h}.csv",
            ]
            test_candidates = [
                raw_root / dataset / f"test_h{h}.csv",
                raw_root / dataset / "test.csv",
                raw_root / dataset / f"test_{h}.csv",
            ]
            raw_val_path = next((p for p in val_candidates if p.exists()), None)
            raw_test_path = next((p for p in test_candidates if p.exists()), None)
            if raw_val_path is not None:
                try:
                    df_raw_val = pd.read_csv(raw_val_path)
                    print(f"    已加载原始特征: {raw_val_path} ({max(0, len(df_raw_val.columns) - 2)} 列)")
                except Exception as e:
                    print(f"    原始特征加载失败: {e}")
            if raw_test_path is not None:
                try:
                    df_raw_test = pd.read_csv(raw_test_path)
                except Exception as e:
                    print(f"    原始 test 特征加载失败: {e}")
            if raw_val_path is None or raw_test_path is None:
                print("    原始 val/test 特征不完整，Protocol B 可能回退到 Protocol A")
        
        model_node_types = {
            m: (
                "naive_baseline"
                if m == "seasonal_naive"
                else ("base_model" if m in models else "extended_pool")
            )
            for m in safe_models
        }
        # naive 元信息增强
        naive_fair_mae = model_fair_mae_map.get("seasonal_naive")
        naive_valid_ratio = model_valid_ratio_map.get("seasonal_naive")
        naive_valid_pairs = model_valid_pairs_map.get("seasonal_naive")
        naive_in_eligible = "seasonal_naive" in safe_models
        for m in safe_models:
            eligible_filter_reasons.setdefault(m, []).append("eligible_pass_all_gates")
        res_h = {
            "_meta": {
                "filter": filter_meta,
                "safe_models": safe_models,
                "eligible_models": safe_models,
                "eligible_filter_reasons": eligible_filter_reasons,
                "eligible_definition": eligible_definition,
                "eligible_definition_hash": eligible_definition_hash,
                "kmin": 3,
                "kmax": len(model_cols),
                "r": float(filter_threshold),
                "mase_hard": float(eligible_mase_hard) if np.isfinite(eligible_mase_hard) else None,
                "health_filter": health_meta,
                "candidate_audit": audit_meta,
                "extended_pool": extended_meta,
                "frozen_naive": frozen_naive_meta,
                "pool_mode": pool_mode,
                "seed": int(seed),
                "model_node_types": model_node_types,
                "model_valid_ratio": model_valid_ratio_map,
                "model_valid_pairs": model_valid_pairs_map,
                "model_fair_mae": {
                    m: v for m, v in model_fair_mae_map.items()
                    if np.isfinite(v)
                },
                "nan_imputation_safe_models": safe_impute_meta,
                "candidate_models_count": int(len(candidate_models)),
                "common_models_count": int(len(model_cols)),
                "naive_meta": {
                    "in_eligible": naive_in_eligible,
                    "fair_mae": float(naive_fair_mae) if naive_fair_mae is not None and np.isfinite(naive_fair_mae) else None,
                    "valid_ratio": float(naive_valid_ratio) if naive_valid_ratio is not None else None,
                    "valid_pairs": int(naive_valid_pairs) if naive_valid_pairs is not None else None,
                    "node_type": "naive_baseline",
                    "frozen_expert": bool(naive_as_frozen_expert),
                },
            }
        }
        kg_input_cols = list(dict.fromkeys(safe_models + ["y", "timestamp"]))
        if "seasonal_naive" in df_val_kg.columns and "seasonal_naive" in df_test_kg.columns:
            kg_input_cols = list(dict.fromkeys(kg_input_cols + ["seasonal_naive"]))
        
        # Protocol A: 仅预测
        a_val: dict = {}  # 若 Protocol A 失败，write_back 块读到空 dict 而非 NameError
        try:
            t0_a = time.perf_counter()
            res_a = kg_combination_pred_only(
                df_val_kg[kg_input_cols],
                df_test_kg[kg_input_cols],
                safe_models, h, dataset_name=dataset
            )
            a_runtime = time.perf_counter() - t0_a
            res_h["kg_protocol_a"] = res_a
            res_h["protocol_A"] = res_a
            res_h["_meta"]["runtime_protocol_a_sec"] = float(a_runtime)
            print(f"    Protocol A MAE: val={res_a['val']['mae']:.2f}, test={res_a['test']['mae']:.2f}")
            a_val = res_a.get("val", {}) if isinstance(res_a, dict) else {}
            if isinstance(a_val, dict):
                if a_val.get("selected_models") is not None:
                    print(f"    Protocol A selected_models: {a_val.get('selected_models')}")
                selected_a = a_val.get("selected_models", [])
                selected_a = selected_a if isinstance(selected_a, list) else []
                res_h["_meta"]["naive_selected_by_protocol_a"] = "seasonal_naive" in selected_a
                res_h["_meta"]["naive_selected_ratio_protocol_a"] = 1.0 if "seasonal_naive" in selected_a else 0.0
                a_wm = a_val.get("weight_meta", {})
                if isinstance(a_wm, dict) and a_wm.get("fallback"):
                    print(f"    Protocol A fallback: {a_wm.get('fallback')}")
            # P1-3: Protocol A slice 证据
            if isinstance(res_a, dict):
                for split_key in ("val", "test"):
                    split_data = res_a.get(split_key, {})
                    if isinstance(split_data, dict):
                        sel_a = split_data.get("selected_models", [])
                        w_a = split_data.get("weights", {})
                        df_split = df_val_kg if split_key == "val" else df_test_kg
                        y_split = np.asarray(df_split["y"].values, dtype=float)
                        if sel_a and isinstance(w_a, dict) and w_a:
                            pred_a_arr = np.zeros(len(y_split), dtype=float)
                            for m, w in w_a.items():
                                if m in df_split.columns:
                                    pred_a_arr += np.asarray(df_split[m].values, dtype=float) * float(w)
                            split_data["slice_evidence"] = _compute_slice_evidence(y_split, pred_a_arr)
        except Exception as e:
            print(f"    Protocol A 失败: {e}")
            res_h["kg_protocol_a"] = {"error": str(e)}
            res_h["protocol_A"] = {"error": str(e)}
        
        # Protocol B: 预测+特征
        try:
            t0_b = time.perf_counter()
            base_model_cols_for_guard = [m for m in safe_models if m in models]
            protocol_b_kwargs = {
                "dataset_name": dataset,
                "base_model_cols": base_model_cols_for_guard,
                "feedback_store": feedback_store,  # 闭环反馈：读注入
            }
            try:
                protocol_b_trace_path = _protocol_b_trace_path(out_root, dataset, h)
                res_b, protocol_b_trace = _run_protocol_b_with_solver(
                    dataset=dataset,
                    horizon=h,
                    df_val=df_val_kg[kg_input_cols],
                    df_test=df_test_kg[kg_input_cols],
                    df_raw_val=df_raw_val,
                    df_raw_test=df_raw_test,
                    model_cols=safe_models,
                    base_model_cols=base_model_cols_for_guard,
                    feedback_store=feedback_store,
                    trace_path=protocol_b_trace_path,
                    signal_ablation_profile_paths=signal_ablation_profile_paths,
                    signal_kg_result_paths=signal_kg_result_paths,
                    temporal_relation_graph=temporal_relation_graph,
                )
                res_h.setdefault("_meta", {})["protocol_b_solver"] = {
                    "enabled": True,
                    "trace_path": str(protocol_b_trace_path),
                    "trace_stages": [
                        s.get("stage")
                        for s in getattr(protocol_b_trace, "stages", [])
                    ],
                }
            except TypeError as call_err:
                # Backward compatibility: older protocol_b signature has no base_model_cols.
                if "unexpected keyword argument 'base_model_cols'" not in str(call_err):
                    raise
                print("    [compat] protocol_b 不支持 base_model_cols，回退旧签名调用")
                protocol_b_kwargs.pop("base_model_cols", None)
                protocol_b_kwargs.pop("feedback_store", None)
                res_b = kg_combination_with_features(
                    df_val_kg[kg_input_cols],
                    df_test_kg[kg_input_cols],
                    df_raw_val, df_raw_test, safe_models, h,
                    **protocol_b_kwargs,
                )
                res_h.setdefault("_meta", {})["protocol_b_compat_retry_without_base_model_cols"] = True
                # compat 路径剥除了 feedback_store，apply_to_graph 未执行，需标记可观测
                res_h["_meta"]["feedback_apply_skipped"] = True
                res_h["_meta"]["protocol_b_solver"] = {
                    "enabled": False,
                    "fallback": "direct_protocol_b_compat",
                }
            b_runtime = time.perf_counter() - t0_b

            # 闭环反馈写回：用 val-MAE（B vs A 参照）更新 feedback_store，供后续 horizon 使用。
            # 优先使用 selected_models_b_candidate（guard 触发前的 B 候选），
            # 无法获取时回退 selected_models（guard 通过的最终选择）。
            try:
                _b_val = res_b.get("val", {}) if isinstance(res_b, dict) else {}
                # Medium-1 fix: prefer B-candidate over guard fallback
                _fb_selected = (_b_val.get("selected_models_b_candidate") or
                                _b_val.get("selected_models") or [])
                _fb_weights = (_b_val.get("weights_b_candidate") or
                               _b_val.get("weights") or {})
                # High-1/2 fix: val-only signal; quality normalized via Protocol A reference
                _fb_val_mae_b = _b_val.get("mae", float("nan"))
                _fb_val_mae_a = a_val.get("mae", float("nan")) if isinstance(a_val, dict) else float("nan")
                if (_fb_selected and isinstance(_fb_weights, dict)
                        and np.isfinite(_fb_val_mae_b) and np.isfinite(_fb_val_mae_a)):
                    feedback_store.write_back(
                        selected=list(_fb_selected),
                        weights=dict(_fb_weights),
                        val_mae_b=float(_fb_val_mae_b),
                        val_mae_a=float(_fb_val_mae_a),
                        horizon=int(h),
                        dataset=str(dataset),
                    )
                    print(f"    [feedback] written back: h={h}, "
                          f"selected={_fb_selected}, val_mae_b={_fb_val_mae_b:.2f}, "
                          f"val_mae_a={_fb_val_mae_a:.2f}")
            except Exception as _fb_err:
                print(f"    [feedback] write_back failed (non-fatal): {_fb_err}")
            res_h["kg_protocol_b"] = res_b
            res_h["protocol_B"] = res_b
            res_h["_meta"]["runtime_protocol_b_sec"] = float(b_runtime)
            b_val_meta = res_b.get("val", {}) if isinstance(res_b, dict) else {}
            if isinstance(b_val_meta, dict):
                res_h["_meta"]["protocol_b_reasoning_used_rate"] = b_val_meta.get("reasoning_used_rate")
                res_h["_meta"]["protocol_b_reasoning_mode"] = b_val_meta.get("reasoning_mode")
                selected_b = b_val_meta.get("selected_models", [])
                if not selected_b:
                    selected_b = b_val_meta.get("selected_models_b_candidate", [])
                selected_b = selected_b if isinstance(selected_b, list) else []
                res_h["_meta"]["naive_selected_by_protocol_b"] = "seasonal_naive" in selected_b
                res_h["_meta"]["naive_selected_ratio_protocol_b"] = 1.0 if "seasonal_naive" in selected_b else 0.0
                b_wm = b_val_meta.get("weight_meta", {})
                b_guard_meta = b_wm.get("protocol_b_guard") if isinstance(b_wm, dict) else {}
                b_guard_cfg = b_wm.get("guard_config") if isinstance(b_wm, dict) else {}
                audit_soft = audit_meta.get("soft_passed_models", []) if isinstance(audit_meta, dict) else []
                audit_soft = audit_soft if isinstance(audit_soft, list) else []
                selected_soft = [m for m in selected_b if m in set(audit_soft)]
                selected_base = [m for m in selected_b if m in set(models)]
                selected_extended = [m for m in selected_b if m not in set(models)]
                res_h["_meta"]["protocol_b"] = {
                    "selected_models": selected_b,
                    "selected_soft_passed_models": selected_soft,
                    "selected_base_models_count": int(len(selected_base)),
                    "selected_extended_models_count": int(len(selected_extended)),
                    "chosen_protocol": res_b.get("protocol") if isinstance(res_b, dict) else None,
                    "guard": b_guard_meta if isinstance(b_guard_meta, dict) else {},
                    "guard_config": b_guard_cfg if isinstance(b_guard_cfg, dict) else {},
                    "fallback_target": (
                        b_guard_meta.get("fallback_target")
                        if isinstance(b_guard_meta, dict)
                        else None
                    ),
                    "fallback_reason": (
                        b_guard_meta.get("reason")
                        if isinstance(b_guard_meta, dict)
                        else None
                    ),
                }
            print(f"    Protocol B MAE: val={res_b['val']['mae']:.2f}, test={res_b['test']['mae']:.2f}")
            if isinstance(b_val_meta, dict):
                if b_val_meta.get("selected_models") is not None:
                    print(f"    Protocol B selected_models: {b_val_meta.get('selected_models')}")
                b_wm = b_val_meta.get("weight_meta", {})
                if isinstance(b_wm, dict):
                    guard_meta = b_wm.get("protocol_b_guard")
                    if isinstance(guard_meta, dict):
                        print(f"    Protocol B guard: {guard_meta.get('reason')}")
            # P1-3: Protocol B slice 证据
            if isinstance(res_b, dict):
                for split_key in ("val", "test"):
                    split_data = res_b.get(split_key, {})
                    if isinstance(split_data, dict):
                        sel_b = split_data.get("selected_models", [])
                        w_b = split_data.get("weights", {})
                        df_split = df_val_kg if split_key == "val" else df_test_kg
                        y_split = np.asarray(df_split["y"].values, dtype=float)
                        if sel_b and isinstance(w_b, dict) and w_b:
                            pred_b_arr = np.zeros(len(y_split), dtype=float)
                            for m, w in w_b.items():
                                if m in df_split.columns:
                                    pred_b_arr += np.asarray(df_split[m].values, dtype=float) * float(w)
                            split_data["slice_evidence"] = _compute_slice_evidence(y_split, pred_b_arr)
        except Exception as e:
            print(f"    Protocol B 失败: {e}")
            res_h["kg_protocol_b"] = {"error": str(e)}
            res_h["protocol_B"] = {"error": str(e)}
        
        results[h] = res_h

    # 闭环反馈持久化：保存本次 session 的写回结果，供下次运行加载。
    try:
        feedback_store.save(str(feedback_save_path), config_fingerprint=config_fingerprint)
        fb_summary = feedback_store.summary()
        print(f"[feedback] {dataset} summary: "
              f"models={fb_summary['n_model_scores']}, "
              f"pairs={fb_summary['n_pair_scores']}, "
              f"history={fb_summary['n_history_entries']}")
    except Exception as _save_err:
        print(f"[feedback] save failed (non-fatal): {_save_err}")

    # 实验开关：导出 Hawkes 动态关系图快照，便于核对“反馈事件→关系强度变化”。
    if temporal_relation_graph is not None:
        try:
            snapshot_path = out_root / f"{dataset}_temporal_relations.json"
            _dump_temporal_relation_snapshot(temporal_relation_graph, snapshot_path)
            print(f"[temporal] {dataset} 关系图快照已保存: {snapshot_path} "
                  f"(nodes={temporal_relation_graph.G.number_of_nodes()}, "
                  f"edges={temporal_relation_graph.G.number_of_edges()})")
        except Exception as _temporal_err:
            print(f"[temporal] snapshot save failed (non-fatal): {_temporal_err}")

    return results


# ==========================================================================
# 离线模型库构建（SQLite 模型库 Task 4）
# ==========================================================================

MODEL_LIBRARY_TASK_TYPE = "load_forecast"
MODEL_LIBRARY_BUSINESS_DOMAIN = "power_load"
MODEL_LIBRARY_STRATEGY = "protocol_b_combination"
TARGET = "load"

#: 基础预测器的单步语义。用户请求的完整轨迹长度是 forecast_steps，与之正交。
MODEL_LIBRARY_BASE_HORIZON = 1
#: 预测长度的可读别名：H1=未来一天、H2=未来一周、H3=未来三十天。
#: 数据库真正的字段始终是 forecast_steps，不额外增加重复的 H 字段；这里只进报告。
#: 注意：H1/H2/H3 从此只表示预测长度，历史数据样例一律叫 S1/S2/S3。
FORECAST_HORIZON_LABELS: Dict[int, str] = {24: "H1", 168: "H2", 720: "H3"}
#: 构造场景/数据特征的历史窗口（小时）。离线与在线必须取同一个窗口长度，
#: 否则两侧 signature 不可比，相似度检索失去意义。
SIGNATURE_WINDOW = 720
#: 用户在线只需提交这两列；离线建库也只从这两列派生成员输入。
TRAJECTORY_INPUT_COLUMNS = ("timestamp", "load")
MODEL_LIBRARY_COUNTRY_BY_REGION = {"pjm": "US", "aemo_vic": "AU", "aemo_nsw": "AU"}
#: seasonal_naive 在 KG 基础候选里被排除（扩展池节点），但它在轨迹契约下有
#: 明确定义的多步行为，是模型库的合法成员。
MODEL_LIBRARY_EXTRA_CANDIDATES = ("seasonal_naive",)


def _model_library_candidate_types() -> List[str]:
    candidates = list(_build_kg_model_candidates())
    for extra in MODEL_LIBRARY_EXTRA_CANDIDATES:
        if extra not in candidates:
            candidates.append(extra)
    return candidates


def _infer_freq(timestamps: pd.Series) -> str:
    ts = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()
    inferred = pd.infer_freq(ts) if len(ts) >= 3 else None
    if inferred:
        return inferred
    if len(ts) >= 2:
        delta = ts.diff().dropna().median()
        if pd.notna(delta):
            if delta <= pd.Timedelta(hours=1):
                return "h"
            if delta <= pd.Timedelta(days=1):
                return "D"
    return "h"


def _scenario_signature(
    df_val: pd.DataFrame,
    df_raw_val: Optional[pd.DataFrame],
    horizon: int,
    freq: str,
) -> Dict[str, float]:
    y = np.asarray(df_val["y"].values, dtype=float)
    y = y[np.isfinite(y)]
    signature: Dict[str, float] = {
        "horizon": float(horizon),
        "n_samples": float(len(y)),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "y_cv": float(np.std(y) / (abs(np.mean(y)) + 1e-6)),
    }
    if df_raw_val is not None and len(df_raw_val):
        from src.selector.scenario_similarity import PowerScenarioAnalyzer

        frame = _align_raw_to_pred(df_raw_val.copy(), df_val.copy())
        frame = frame.reset_index(drop=True)
        frame["load"] = pd.Series(y[: len(frame)])
        extra = PowerScenarioAnalyzer().extract_scenario_signature(frame)
        signature.update({k: float(v) for k, v in extra.items() if np.isfinite(v)})
    return {k: round(float(v), 8) for k, v in signature.items()}


def history_window_signature(
    history: pd.DataFrame,
    *,
    freq: str,
    base_horizon: int = MODEL_LIBRARY_BASE_HORIZON,
    window: int = SIGNATURE_WINDOW,
) -> Dict[str, float]:
    """由预测起点之前 ``window`` 小时的 ``(timestamp, load)`` 生成场景签名。

    离线建库与在线查询共用这一个实现：两侧取同样长度的窗口、派生同样的 hour
    列、走同样的统计口径，signature 的 key 集与数值才可比。签名里不放
    ``forecast_steps``——它是硬契约过滤条件，不是"越接近越相似"的数值特征。
    """
    missing = [column for column in TRAJECTORY_INPUT_COLUMNS if column not in history.columns]
    if missing:
        raise ValueError(f"signature 需要的列缺失: {missing}")
    if len(history) < window:
        raise ValueError(
            f"signature 需要预测起点之前至少 {window} 小时历史，收到 {len(history)} 行"
        )
    frame = history.tail(window)[list(TRAJECTORY_INPUT_COLUMNS)].reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["load"] = pd.to_numeric(frame["load"])
    frame["hour"] = frame["timestamp"].dt.hour
    target_frame = frame.rename(columns={"load": "y"})
    return _scenario_signature(target_frame, frame, base_horizon, freq)


def _forecast_origin_raw_frame(
    raw_root: Optional[Path], dataset: str, split: str, horizon: int
) -> Optional[pd.DataFrame]:
    """预测起点特征 X(t)，附目标时间戳 t+H。

    基础模型训练是 X(t) -> y(t+H)；组合 interaction 若也从这份特征取值（而不是
    目标时刻的原始特征），离线评估、保存后重放、以及在线 run.py predict 三处
    使用的就是同一份特征，无需调用方提供未来天气。
    """
    if raw_root is None:
        return None
    path = raw_root / dataset / f"{split}.csv"
    if not path.exists():
        return None
    features, _y, target_ts, _freq = prepare_supervised(pd.read_csv(path), TARGET, horizon)
    frame = features.reset_index(drop=True)
    frame.insert(0, "timestamp", pd.to_datetime(target_ts).values)
    return frame


def _library_candidate_models(
    store: ModelStore,
    *,
    dataset: str,
    base_horizon: int,
    model_types: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, str]]]:
    """取回本次建库声明的全部基础预测器产物。

    候选完整性规则（正式实验口径）：``model_types`` 是本次建库**声明**的候选集合，
    默认取 ``_model_library_candidate_types()``——它与 ``configs/pipeline.yaml``
    的 ``models:`` 段同源，正是 ``train_baselines`` 会登记的那一批。声明的候选
    未登记或产物文件丢失，都属于**建库不完整**，直接整体失败，绝不用一个缩水的
    候选池产出正式数字。

    只有"已登记、但在历史数据契约下产不出完整轨迹"才是可记录的无资格
    （需要未来外生变量的模型、以及无法消费用户最新历史的 arima），由
    :func:`generate_member_trajectory` 在轨迹生成时判定并记入 skipped。
    """
    members: Dict[str, Dict[str, Any]] = {}
    skipped: List[Dict[str, str]] = []
    for model_type in model_types:
        model_id = f"{dataset}__h{base_horizon}__{model_type}"
        row = store.get_model(model_id)
        if row is None:
            raise RuntimeError(
                f"model library build failed for {dataset}: 声明的候选 {model_type} 未登记在"
                f"模型库（期望 model_id={model_id}）——建库不完整，不产出报告"
            )
        artifact_path = Path(row["artifact_path"])
        if not artifact_path.exists():
            raise RuntimeError(
                f"model library build failed for {dataset}: 模型 {model_id} 已登记但产物文件"
                f"不存在: {artifact_path}——建库不完整，不产出报告"
            )
        members[model_type] = {
            "model_id": model_id,
            "model": load_artifact(artifact_path),
            "model_type": row["model_type"],
            "required_features": list(row["required_features"]),
        }
    return members, skipped


def _library_split_frame(raw_root: Path, dataset: str, split: str) -> pd.DataFrame:
    path = raw_root / dataset / f"{split}.csv"
    if not path.exists():
        raise RuntimeError(f"model library build needs {path}")
    frame = pd.read_csv(path)
    missing = [column for column in ("timestamp", TARGET) if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} 缺少必填列: {missing}")
    frame = frame[["timestamp", TARGET]].rename(columns={TARGET: "load"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["load"] = pd.to_numeric(frame["load"])
    return frame.sort_values("timestamp").reset_index(drop=True)


def _library_raw_frame(raw_root: Path, dataset: str) -> pd.DataFrame:
    path = raw_root / dataset / "load.csv"
    if not path.exists():
        raise RuntimeError(f"model library build needs {path}")
    frame = pd.read_csv(path)
    missing = [column for column in ("timestamp", TARGET) if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} 缺少必填列: {missing}")
    frame = frame[["timestamp", TARGET]].rename(columns={TARGET: "load"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["load"] = pd.to_numeric(frame["load"])
    return frame.sort_values("timestamp").reset_index(drop=True)


def _frozen_windows(window_plan: Path, dataset: str, forecast_steps: int) -> Dict[str, Dict[str, Any]]:
    """读取 Stage 0 冻结的七个共享预测窗口，并取出该长度的目标区间。

    七个窗口的输入历史与预测起点由所有预测长度共享；``forecast_steps`` 只决定从
    同一个起点截取多长的目标前缀。
    """
    plan = json.loads(window_plan.read_text(encoding="utf-8"))
    dataset_entry = next(entry for entry in plan["datasets"] if entry["dataset"] == dataset)
    windows = {}
    for origin in dataset_entry["origins"]:
        target = origin["targets"][str(forecast_steps)]
        windows[origin["label"]] = {
            **{k: v for k, v in origin.items() if k != "targets"},
            "first_target": target["first_target"],
            "last_target": target["last_target"],
        }
    return windows


def _scenario_sample_frame(
    raw_root: Path,
    dataset: str,
    window: Mapping[str, Any],
) -> pd.DataFrame:
    """取一个冻结窗口的 720 小时历史和其完整目标轨迹。"""
    frame = _library_raw_frame(raw_root, dataset)
    start = pd.Timestamp(window["history_start"])
    end = pd.Timestamp(window["last_target"])
    return frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].reset_index(drop=True)


def _library_origin_split(
    frame: pd.DataFrame, forecast_steps: int, *, label: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """把一个切分拆成「预测起点之前的历史」与「该起点之后的完整目标轨迹」。"""
    needed = SIGNATURE_WINDOW + forecast_steps
    if len(frame) < needed:
        raise RuntimeError(
            f"{label} 只有 {len(frame)} 行，不足以容纳 {SIGNATURE_WINDOW} 小时 signature 窗口 "
            f"+ {forecast_steps} 步完整轨迹（需要 {needed} 行）"
        )
    cut = len(frame) - forecast_steps
    history = frame.iloc[:cut].reset_index(drop=True)
    targets = frame.iloc[cut:].reset_index(drop=True)
    expected = future_timestamps(history, forecast_steps)
    if not targets["timestamp"].reset_index(drop=True).equals(pd.Series(expected)):
        raise RuntimeError(
            f"{label} 的目标时间戳不是预测起点之后连续的 {forecast_steps} 个小时点；"
            "本方案不在缺口上伪造目标"
        )
    return history, targets


def _library_trajectory_matrix(
    members: Mapping[str, Dict[str, Any]],
    *,
    frame: pd.DataFrame,
    forecast_steps: int,
    country: str,
    label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, str]], pd.Timestamp]:
    """生成一个预测起点下的完整轨迹矩阵与其日历特征表。"""
    history, targets = _library_origin_split(frame, forecast_steps, label=label)
    matrix, skipped = generate_trajectory_matrix(
        members=members,
        history=history,
        forecast_steps=forecast_steps,
        country=country,
    )
    matrix["y"] = targets["load"].to_numpy(dtype=float)
    calendar = calendar_frame(matrix["target_timestamp"], country)
    return matrix, calendar, skipped, history["timestamp"].iloc[-1]


def _library_candidate_maes(matrix: pd.DataFrame, model_cols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """逐候选记录「单点（lead=1）」与「完整轨迹」两种 validation 表现。

    两者的排序可以不同——这正是 §2.2 指出的问题：单点最优不等于长轨迹最稳。
    """
    y = matrix["y"].to_numpy(dtype=float)
    lead1 = matrix["lead"].to_numpy(dtype=int) == 1
    report: Dict[str, Dict[str, float]] = {}
    for column in model_cols:
        pred = matrix[column].to_numpy(dtype=float)
        report[column] = {
            "lead1_mae": float(np.mean(np.abs(pred[lead1] - y[lead1]))),
            "trajectory_mae": float(np.mean(np.abs(pred - y))),
        }
    return report


def _build_library_task(
    store: ModelStore,
    *,
    dataset: str,
    forecast_steps: int,
    model_types: Sequence[str],
    raw_root: Optional[Path],
    artifact_dir: Path,
    filter_threshold: float,
    base_horizon: int = MODEL_LIBRARY_BASE_HORIZON,
    val_frame: Optional[pd.DataFrame] = None,
    test_frame: Optional[pd.DataFrame] = None,
    scenario_sample: Optional[str] = None,
    audit_window: Optional[str] = None,
) -> Dict[str, Any]:
    """按完整轨迹建立一条 scenario-data-combination 关系（方案 §3.3）。

    指定任务不得静默跳过：原始数据缺失 / 无可用候选 / 过滤后无候选 / 重放不一致
    均直接失败。
    """
    forecast_steps = validate_forecast_steps(forecast_steps)
    if raw_root is None:
        raise RuntimeError(
            f"model library build needs --raw-root for {dataset} forecast_steps={forecast_steps}"
        )
    country = MODEL_LIBRARY_COUNTRY_BY_REGION.get(dataset)
    if country is None:
        raise RuntimeError(f"未知数据集 {dataset}：无法确定日历特征所用节假日日历")

    candidates, skipped = _library_candidate_models(
        store, dataset=dataset, base_horizon=base_horizon, model_types=model_types
    )
    if not candidates:
        raise RuntimeError(
            f"model library build failed for {dataset} forecast_steps={forecast_steps}: "
            f"没有任何已登记的 h={base_horizon} 候选模型"
        )

    label = f"{dataset} forecast_steps={forecast_steps}"
    if scenario_sample is not None:
        label = f"{label} {scenario_sample}"
    if val_frame is None:
        val_frame = _library_split_frame(raw_root, dataset, "val")
    val_matrix, val_calendar, val_skipped, val_origin = _library_trajectory_matrix(
        candidates,
        frame=val_frame,
        forecast_steps=forecast_steps,
        country=country,
        label=f"{label} val",
    )
    # 候选资格由离线边界判定：能在 val 上产出轨迹的才进入 test 与组合枚举。
    eligible = {name: candidates[name] for name in candidates if name in val_matrix.columns}
    skipped.extend(val_skipped)
    for entry in skipped:
        print(f"  [model-library] {label}: 候选 {entry['model_type']} 无资格——{entry['reason']}")
    if not eligible:
        raise RuntimeError(
            f"model library build failed for {label}: 所有候选都无法在当前输入契约下产出完整轨迹"
        )

    if test_frame is None:
        test_frame = _library_split_frame(raw_root, dataset, "test")
    test_matrix, test_calendar, test_skipped, test_origin = _library_trajectory_matrix(
        eligible,
        frame=test_frame,
        forecast_steps=forecast_steps,
        country=country,
        label=f"{label} test",
    )
    if test_skipped:
        raise RuntimeError(
            f"model library build failed for {label}: 候选在 test 起点无法产出轨迹 {test_skipped}"
        )

    model_cols = sorted(eligible)
    candidate_maes = _library_candidate_maes(val_matrix, model_cols)
    # min_keep=1：递归轨迹下一个成员可能放大若干数量级，强行保底反而会把数值
    # 爆炸的列塞进每一个被枚举的子集。安全候选就是能稳定跑完整条轨迹的那些。
    # horizon 在这里只用于挑选 naive 基线的容忍度，取的是"预测多远"，因此传
    # forecast_steps；下面 Protocol B 的 horizon 是基础预测器的单步语义，传
    # base_horizon。两处含义不同，不要统一。
    safe_models, filter_meta = filter_weak_models(
        val_matrix.rename(columns={"target_timestamp": "timestamp"}),
        model_cols,
        threshold_ratio=filter_threshold,
        min_keep=1,
        horizon=forecast_steps,
    )
    if not safe_models:
        raise RuntimeError(
            f"model library build failed for {label}: 弱模型/稳定性过滤后无可用模型"
        )
    safe_models = sorted(safe_models)

    cols = list(dict.fromkeys(list(safe_models) + ["y", "timestamp"]))
    dval = val_matrix.rename(columns={"target_timestamp": "timestamp"})[cols]
    dtest = test_matrix.rename(columns={"target_timestamp": "timestamp"})[cols]

    # 非空子集全枚举：不写死二/三模型上限。相同实际成员只保留 validation 最小者。
    #
    # 枚举时把 val 同时当作引擎的 test 入参：Protocol B 会用 val/test 两侧的
    # **预测分布** 估 PSI 得到 drift_level，再据此做稳定性过滤与 interaction 门控
    # （src/eval/kg/stability.py:270）。只要真实 test 预测进入这一步，即使 test
    # 标签完全没参与拟合，候选集合和被选组合也会随 test 分布改变。方案 §3.3.6 要求
    # test 只在冻结后评价，因此这里从结构上不给引擎任何 test 数据；冻结时刻本来
    # 也没有合法的未来窗口可用于估漂移。
    y_val = dval["y"].to_numpy(dtype=float)
    candidates_by_members: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for size in range(1, len(safe_models) + 1):
        for members in itertools.combinations(safe_models, size):
            res = evaluate_fixed_protocol_b_combination(
                dval, dval, val_calendar, val_calendar,
                selected_models=list(members),
                horizon=base_horizon,
                dataset_name=dataset,
                base_model_cols=list(safe_models),
                return_combination_predictor=True,
            )
            predictor = res[COMBINATION_PREDICTOR_KEY]
            effective = tuple(predictor.member_ids)
            if len(effective) == 1:
                # 只剩一个成员时就是"用这个模型"，不是"这个模型的加权组合"：
                # 权重固定为 1、不带 interaction。让 Ridge 自由拟合单列系数得到的
                # 是一个一般不等于 1 的缩放（实测两个装置上分别是 0.9993 和 1.0087，
                # 两个方向都出现过），等于给单模型关系加了一个无理由的系统性偏置。
                # 排名也用这份原样预测，保证选出来的和存下去的是同一个东西。
                predictor = CombinationPredictor(
                    member_ids=[effective[0]],
                    linear_weights=[1.0],
                    strategy=predictor.strategy,
                    interaction=None,
                )
                val_prediction = dval[effective[0]].to_numpy(dtype=float)
                val_mae = float(np.mean(np.abs(val_prediction - y_val)))
            else:
                val_prediction = np.asarray(
                    res["_runtime_predictions"]["val"], dtype=float
                )
                val_mae = float(res["val"]["mae"])
            existing = candidates_by_members.get(effective)
            if existing is None or val_mae < existing["validation_mae"]:
                candidates_by_members[effective] = {
                    "requested_members": list(members),
                    "validation_mae": val_mae,
                    "predictor": predictor,
                    "val_prediction": val_prediction,
                }

    # 排名目标是整条 validation 轨迹的 MAE；完全相同才按成员 ID 元组确定性排序。
    best_key = min(
        candidates_by_members,
        key=lambda k: (candidates_by_members[k]["validation_mae"], k),
    )
    best = candidates_by_members[best_key]
    predictor = best["predictor"]

    freq = _infer_freq(dval["timestamp"])
    val_history = val_frame.iloc[: len(val_frame) - forecast_steps]
    signature = history_window_signature(val_history, freq=freq, base_horizon=base_horizon)
    scenario_id = compute_scenario_id(
        signature, prefix=f"{dataset}_h{base_horizon}_s{forecast_steps}"
    )
    effective_model_ids = [eligible[m]["model_id"] for m in predictor.member_ids]
    artifact_path = artifact_dir / dataset / f"{scenario_id}__combo.pkl"
    save_artifact(predictor, artifact_path)

    # §9.3：保存后立即重放，与离线最终预测逐点比对；比较对象是完整 validation
    # 轨迹。不可重放的组合任何数据库写入都不发生。
    reloaded = load_artifact(artifact_path)
    val_replay = np.asarray(
        reloaded.predict(
            {m: dval[m].to_numpy(dtype=float) for m in reloaded.member_ids},
            val_calendar,
        ),
        dtype=float,
    )
    max_abs = float(np.max(np.abs(val_replay - best["val_prediction"])))
    if max_abs > 1e-8:
        raise RuntimeError(
            f"model library build failed for {label}: "
            f"组合器重放误差 {max_abs:.3e} > 1e-8（val），不写入数据库"
        )

    # test 只在组合、权重和全部规则冻结之后评价：由已保存的组合器重放 test 轨迹，
    # 不再走一次引擎。这条轨迹也正是在线入口应当逐点复现的目标。
    test_prediction = np.asarray(
        reloaded.predict(
            {m: dtest[m].to_numpy(dtype=float) for m in reloaded.member_ids},
            test_calendar,
        ),
        dtype=float,
    )
    test_mae = float(np.mean(np.abs(test_prediction - dtest["y"].to_numpy(dtype=float))))

    if store.get_scenario(scenario_id) is None:
        store.add_scenario(
            scenario_id=scenario_id,
            task_type=MODEL_LIBRARY_TASK_TYPE,
            business_domain=MODEL_LIBRARY_BUSINESS_DOMAIN,
            region=dataset,
            horizon=base_horizon,
            forecast_steps=forecast_steps,
            freq=freq,
            signature=signature,
        )
    data_profile_id = store.add_data_profile(
        scenario_id=scenario_id,
        data_ref=str(raw_root / dataset),
        target_column="load",
        features=sorted(TRAJECTORY_INPUT_COLUMNS),
        sample_count=int(len(dval)),
        start_at=str(dval["timestamp"].min()),
        end_at=str(dval["timestamp"].max()),
        signature=signature,
    )

    combination_id = store.add_combination(
        MODEL_LIBRARY_STRATEGY,
        str(artifact_path),
        [
            (model_id, order, float(weight))
            for order, (model_id, weight) in enumerate(
                zip(effective_model_ids, predictor.linear_weights)
            )
        ],
    )
    relation_id = store.add_relation(
        scenario_id,
        data_profile_id,
        combination_id,
        validation_mae=best["validation_mae"],
        test_mae=test_mae,
    )
    print(
        f"  [model-library] {label}: 最佳组合 {list(predictor.member_ids)} "
        f"(val_mae={best['validation_mae']:.4f}, "
        f"{len(candidates_by_members)} 个去重候选组合)"
    )
    return {
        "dataset": dataset,
        "base_horizon": base_horizon,
        "forecast_steps": forecast_steps,
        "forecast_horizon": FORECAST_HORIZON_LABELS[forecast_steps],
        "scenario_sample": scenario_sample,
        "audit_window": audit_window,
        "scenario_id": scenario_id,
        "data_profile_id": data_profile_id,
        "combination_id": combination_id,
        "relation_id": relation_id,
        "safe_models": list(safe_models),
        "declared_candidates": list(model_types),
        "skipped_candidates": skipped,
        "candidate_validation_mae": candidate_maes,
        "filter_excluded": list(filter_meta.get("final_excluded", [])),
        "requested_members": best["requested_members"],
        "effective_members": effective_model_ids,
        "linear_weights": [float(w) for w in predictor.linear_weights],
        "has_interaction": predictor.interaction is not None,
        "validation_mae": best["validation_mae"],
        "test_mae": test_mae,
        "enumerated_combinations": [
            {
                "members": list(members),
                "validation_mae": entry["validation_mae"],
            }
            for members, entry in sorted(
                candidates_by_members.items(), key=lambda kv: (kv[1]["validation_mae"], kv[0])
            )
        ],
        "val_origin": str(val_origin),
        "test_origin": str(test_origin),
        "test_target_timestamps": [str(ts) for ts in dtest["timestamp"]],
        "val_trajectory": best["val_prediction"].tolist(),
        "test_trajectory": test_prediction.tolist(),
    }


def build_model_library(
    *,
    datasets: Optional[Sequence[str]],
    selected_forecast_steps: Optional[Sequence[int]],
    model_types: Sequence[str],
    raw_root: Optional[Path],
    out_root: Path,
    database: Path,
    artifact_dir: Path,
    filter_threshold: float,
    window_plan: Optional[Path] = None,
) -> Dict[str, Any]:
    ensure_dir(out_root)
    store = ModelStore(str(database))
    store.create_schema()
    steps_filter = [
        validate_forecast_steps(s)
        for s in (selected_forecast_steps or SUPPORTED_FORECAST_STEPS)
    ]
    dataset_filter = list(datasets) if datasets else list(DATASET_HORIZONS)

    report: Dict[str, Any] = {
        "base_horizon": MODEL_LIBRARY_BASE_HORIZON,
        "signature_window": SIGNATURE_WINDOW,
        "candidate_model_types": list(model_types),
        "tasks": [],
    }
    try:
        for dataset in dataset_filter:
            for forecast_steps in steps_filter:
                if window_plan is None:
                    report["tasks"].append(
                        _build_library_task(
                            store,
                            dataset=dataset,
                            forecast_steps=int(forecast_steps),
                            model_types=model_types,
                            raw_root=raw_root,
                            artifact_dir=artifact_dir,
                            filter_threshold=filter_threshold,
                        )
                    )
                    continue

                windows = _frozen_windows(window_plan, dataset, int(forecast_steps))
                audit_frame = _scenario_sample_frame(raw_root, dataset, windows["A"])
                for label in ("S1", "S2", "S3"):
                    report["tasks"].append(
                        _build_library_task(
                            store,
                            dataset=dataset,
                            forecast_steps=int(forecast_steps),
                            model_types=model_types,
                            raw_root=raw_root,
                            artifact_dir=artifact_dir,
                            filter_threshold=filter_threshold,
                            val_frame=_scenario_sample_frame(raw_root, dataset, windows[label]),
                            test_frame=audit_frame,
                            scenario_sample=label,
                            audit_window="A",
                        )
                    )
    finally:
        store.close()

    (out_root / "model_library_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n[model-library] 报告已保存: {out_root / 'model_library_report.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description="KG 组合策略评估")
    parser.add_argument("--pred-root", type=Path, default=Path("reports/baselines"),
                        help="预测文件根目录")
    parser.add_argument("--raw-root", type=Path, default=Path("data/features"),
                        help="原始特征根目录（Protocol B，默认 data/features）")
    parser.add_argument("--out-root", type=Path, default=Path("reports/combos_kg"),
                        help="输出目录")
    parser.add_argument("--filter-threshold", type=float, default=2.0,
                        help="弱模型过滤阈值")
    parser.add_argument("--health-config", type=Path, default=None,
                        help="可选：routing_config.json 路径，用于健康白名单过滤")
    parser.add_argument("--extended-pool", action="store_true",
                        help="启用扩展候选池（gating/soft/adaptive/seasonal_naive 等）")
    parser.add_argument("--naive-as-frozen-expert", action="store_true", default=KG_SEASONAL_NAIVE_AS_FROZEN_EXPERT,
                        help="将 seasonal_naive 作为 KG 可选冻结专家节点（进入 eligible/safe_models）")
    parser.add_argument("--no-naive-as-frozen-expert", dest="naive_as_frozen_expert", action="store_false",
                        help="关闭 seasonal_naive 冻结专家节点")
    parser.add_argument("--combo-root", type=Path, default=Path("reports/modelcombine"),
                        help="组合预测根目录（用于扩展候选池兜底加载）")
    parser.add_argument("--allow-in-sample-extended-pool", action="store_true",
                        help="允许将 in-sample 的扩展策略预测用于 val（默认严格禁止，防泄漏）")
    parser.add_argument("--strict-extended-pool", action="store_true", default=KG_STRICT_EXTENDED_POOL,
                        help="启用扩展池时，要求每个任务至少加载 min-extended-loaded 个候选（默认开启）")
    parser.add_argument("--no-strict-extended-pool", dest="strict_extended_pool", action="store_false",
                        help="关闭扩展池严格门禁（仅调试用）")
    parser.add_argument("--min-extended-loaded", type=int, default=KG_EXTENDED_POOL_MIN_LOADED,
                        help="扩展池严格模式下每个 split 至少成功加载的扩展策略数量（默认1）")
    parser.add_argument("--candidate-audit", type=Path, default=None,
                        help="可选：离线候选审计 JSON 路径（仅允许 accepted 节点入池）")
    parser.add_argument("--strict-candidate-audit", action="store_true", default=False,
                        help="启用 candidate audit 严格门禁（accepted 数不足时失败）")
    parser.add_argument("--min-audit-accepted", type=int, default=0,
                        help="strict-candidate-audit 模式下每任务最少 accepted 扩展节点数")
    parser.add_argument("--fail-on-protocol-b-error", action="store_true", default=True,
                        help="若任一任务 Protocol B 运行失败则返回非零退出码（默认开启）")
    parser.add_argument("--allow-protocol-b-error", dest="fail_on_protocol_b_error", action="store_false",
                        help="允许 Protocol B 个别任务失败并继续（仅调试用）")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("MODELCOMBINE_SEED", "42")),
                        help="运行随机种子（用于结果元数据固化）")
    parser.add_argument("--eligible-mase-hard", type=float, default=float("inf"),
                        help="Eligible 口径 MASE 硬阈值；<=该阈值才可入 eligible/safe（默认不启用）")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="仅运行指定数据集")
    parser.add_argument("--horizons", nargs="*", type=int, default=None,
                        help="仅运行指定预测步长，例如 --horizons 1")
    parser.add_argument("--signal-ablation-profile", nargs="*", type=Path, default=None,
                        help="可选：加载 ablation_profile.json 作为 latency 信号源")
    parser.add_argument("--signal-kg-results", nargs="*", type=Path, default=None,
                        help="可选：加载 kg_results.json 作为 latency/drift 信号源")
    # P1-1: G0/G1/G2 消融分解
    parser.add_argument("--model-library", action="store_true",
                        help="离线模型库构建模式：按完整轨迹枚举任意非空组合、写入最佳关系到 SQLite")
    parser.add_argument("--forecast-steps", nargs="*", type=int, default=None,
                        help=f"--model-library 的用户预测长度，默认全部 "
                             f"{list(SUPPORTED_FORECAST_STEPS)}")
    parser.add_argument("--candidates", nargs="*", default=None,
                        help="本次建库声明的候选模型类型；默认取 pipeline.yaml models: 段"
                             "（train_baselines 登记的同一批）。声明的候选未登记或产物丢失"
                             "一律整体失败，不静默缩小候选池")
    parser.add_argument("--database", type=Path, default=None,
                        help="SQLite 模型库路径（--model-library 必需）")
    parser.add_argument("--model-artifacts", type=Path, default=None,
                        help="组合预测器产物目录（--model-library 必需）")
    parser.add_argument("--window-plan", type=Path, default=None,
                        help="Stage 0 冻结窗口 JSON；指定时用 S1—S3 建库，并共用 A 评价")
    parser.add_argument("--ablation-mode", type=str, default=None,
                        choices=["g0", "g1", "g2", "all"],
                        help="消融实验模式: g0=无扩展候选, g1=扩展无审计, g2=扩展+审计, all=依次跑全部")
    args = parser.parse_args()
    np.random.seed(int(args.seed))
    
    # 支持相对路径
    project_root = Path(__file__).parent.parent
    pred_root = args.pred_root if args.pred_root.is_absolute() else project_root / args.pred_root
    out_root = args.out_root if args.out_root.is_absolute() else project_root / args.out_root
    raw_root = None
    if args.raw_root:
        raw_root = args.raw_root if args.raw_root.is_absolute() else project_root / args.raw_root
    combo_root = args.combo_root if args.combo_root.is_absolute() else project_root / args.combo_root

    if args.model_library:
        if args.database is None or args.model_artifacts is None:
            parser.error("--model-library 需要同时提供 --database 与 --model-artifacts")
        build_model_library(
            datasets=args.datasets,
            selected_forecast_steps=args.forecast_steps,
            model_types=args.candidates or _model_library_candidate_types(),
            raw_root=raw_root,
            out_root=out_root,
            database=args.database,
            artifact_dir=args.model_artifacts,
            filter_threshold=float(args.filter_threshold),
            window_plan=args.window_plan,
        )
        return
    health_config_path = None
    candidate_audit_path = None
    if args.health_config:
        health_config_path = (
            args.health_config if args.health_config.is_absolute()
            else project_root / args.health_config
        )
    else:
        health_config_path = None
        print("未指定 --health-config，跳过健康白名单过滤 (不自动候选)")
    health_enabled_map = load_health_enabled_map(health_config_path)
    if args.candidate_audit:
        candidate_audit_path = (
            args.candidate_audit if args.candidate_audit.is_absolute()
            else project_root / args.candidate_audit
        )
    candidate_audit_map = load_candidate_audit_map(candidate_audit_path)
    if args.ablation_mode in {"g2", "all"} and not candidate_audit_map:
        raise ValueError(
            "--ablation-mode g2/all requires --candidate-audit with valid task records"
        )
    if health_enabled_map:
        print(f"已加载健康白名单: {health_config_path}")
    if candidate_audit_map:
        print(f"已加载 candidate audit: {candidate_audit_path}")
    kg_models = _build_kg_model_candidates()
    print(f"KG 基础候选模型: {kg_models}")
    if args.extended_pool:
        print(f"KG 扩展候选池: {_build_extended_pool_strategies()}")
    print(f"KG seasonal_naive 冻结专家: {args.naive_as_frozen_expert}")
    print(f"KG run seed: {args.seed}")
    if args.extended_pool:
        print(
            f"扩展候选池已启用: combo_root={combo_root}, "
            f"allow_in_sample={args.allow_in_sample_extended_pool}, "
            f"strict={args.strict_extended_pool}, min_loaded={args.min_extended_loaded}"
        )
    if np.isfinite(args.eligible_mase_hard):
        print(f"Eligible MASE hard 阈值: {args.eligible_mase_hard}")
    if args.strict_candidate_audit:
        print(
            f"candidate audit 严格模式: enabled, min_accepted={args.min_audit_accepted}, "
            f"path={candidate_audit_path}"
        )
    signal_ablation_profile_paths = [
        p if p.is_absolute() else project_root / p
        for p in (args.signal_ablation_profile or [])
    ]
    signal_kg_result_paths = [
        p if p.is_absolute() else project_root / p
        for p in (args.signal_kg_results or [])
    ]
    if signal_ablation_profile_paths or signal_kg_result_paths:
        print(
            "信号索引输入: "
            f"ablation_profiles={signal_ablation_profile_paths}, "
            f"kg_results={signal_kg_result_paths}"
        )
    
    ensure_dir(out_root)

    # P1-1: 消融分解 —— ablation_mode 覆盖 pool 设置
    ablation_configs: List[Dict[str, Any]] = []
    if args.ablation_mode == "all":
        ablation_configs = [
            {"mode": "g0", "extended_pool": False, "candidate_audit_map": None, "suffix": "g0"},
            {"mode": "g1", "extended_pool": True, "candidate_audit_map": None, "suffix": "g1"},
            {"mode": "g2", "extended_pool": True, "candidate_audit_map": candidate_audit_map, "suffix": "g2"},
        ]
    elif args.ablation_mode == "g0":
        ablation_configs = [{"mode": "g0", "extended_pool": False, "candidate_audit_map": None, "suffix": "g0"}]
    elif args.ablation_mode == "g1":
        ablation_configs = [{"mode": "g1", "extended_pool": True, "candidate_audit_map": None, "suffix": "g1"}]
    elif args.ablation_mode == "g2":
        ablation_configs = [{"mode": "g2", "extended_pool": True, "candidate_audit_map": candidate_audit_map, "suffix": "g2"}]
    else:
        # 非消融模式：正常运行
        ablation_configs = [{
            "mode": None,
            "extended_pool": args.extended_pool,
            "candidate_audit_map": candidate_audit_map,
            "suffix": None,
        }]

    for abl_cfg in ablation_configs:
        abl_mode = abl_cfg["mode"]
        abl_extended = abl_cfg["extended_pool"]
        abl_audit = abl_cfg["candidate_audit_map"]
        abl_suffix = abl_cfg["suffix"]
        abl_out_root = out_root / abl_suffix if abl_suffix else out_root
        if abl_suffix:
            ensure_dir(abl_out_root)
            print(f"\n{'='*60}")
            print(f"消融模式: {abl_mode.upper()} → 输出 {abl_out_root}")
            print(f"{'='*60}")

        all_results = {}
        selected = set(args.datasets) if args.datasets else None
        selected_horizons = set(int(h) for h in args.horizons) if args.horizons else None
        
        for dataset, horizons in DATASET_HORIZONS.items():
            if selected and dataset not in selected:
                continue
            run_horizons = [
                int(h)
                for h in horizons
                if selected_horizons is None or int(h) in selected_horizons
            ]
            if not run_horizons:
                continue
            
            print(f"\n处理数据集: {dataset}")
            results = run_dataset_kg(
                dataset, run_horizons, kg_models,
                pred_root, raw_root, abl_out_root,
                args.filter_threshold,
                health_enabled_map=health_enabled_map,
                extended_pool=abl_extended,
                naive_as_frozen_expert=args.naive_as_frozen_expert,
                combo_root=combo_root,
                allow_in_sample_extended_pool=args.allow_in_sample_extended_pool,
                strict_extended_pool=args.strict_extended_pool if abl_mode != "g0" else False,
                min_extended_loaded=args.min_extended_loaded,
                candidate_audit_map=abl_audit,
                strict_candidate_audit=args.strict_candidate_audit if abl_audit else False,
                min_audit_accepted=args.min_audit_accepted,
                seed=args.seed,
                eligible_mase_hard=float(args.eligible_mase_hard),
                signal_ablation_profile_paths=signal_ablation_profile_paths,
                signal_kg_result_paths=signal_kg_result_paths,
            )
            all_results[dataset] = results
        
        # 保存结果
        json_results = {}
        for ds, horizons in all_results.items():
            json_results[ds] = {}
            for h, data in horizons.items():
                json_results[ds][str(h)] = data
        run_meta = {
            "seed": int(args.seed),
            "ablation_mode": abl_mode,
            "pool_mode": _resolve_pool_mode(
                extended_pool=bool(abl_extended),
                has_candidate_audit=bool(abl_audit),
            ),
            "naive_as_frozen_expert": bool(args.naive_as_frozen_expert),
            "eligible_mase_hard": float(args.eligible_mase_hard) if np.isfinite(args.eligible_mase_hard) else None,
            "guard_config": {
                "last_block_ratio_env": os.environ.get("MODELCOMBINE_KG_B_LAST_BLOCK_RATIO"),
                "min_w_multiplier_env": os.environ.get("MODELCOMBINE_KG_B_HIGH_DRIFT_MIN_W_MULTIPLIER"),
                "complexity_penalty_env": os.environ.get("MODELCOMBINE_KG_B_COMPLEXITY_PENALTY_ENABLED"),
            },
        }
        json_results["_run_meta"] = run_meta

        with (abl_out_root / "kg_results.json").open("w") as f:
            json.dump(json_results, f, indent=2, default=str)
        protocol_b_errors: List[str] = []
        for ds_name, ds_data in json_results.items():
            if not isinstance(ds_data, dict):
                continue
            for h_key, h_payload in ds_data.items():
                if not isinstance(h_payload, dict):
                    continue
                b_payload = h_payload.get("kg_protocol_b")
                if isinstance(b_payload, dict) and b_payload.get("error"):
                    protocol_b_errors.append(f"{ds_name} h={h_key}: {b_payload.get('error')}")
        if protocol_b_errors:
            print("检测到 Protocol B 失败任务:")
            for item in protocol_b_errors:
                print(f"  - {item}")
            if args.fail_on_protocol_b_error:
                raise RuntimeError(
                    f"Protocol B 失败任务数={len(protocol_b_errors)}，已触发 fail-on-protocol-b-error"
                )

        print(f"\n结果已保存到: {abl_out_root}")

if __name__ == "__main__":
    main()
