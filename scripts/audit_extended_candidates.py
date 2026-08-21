#!/usr/bin/env python3
"""
离线审计扩展候选策略（入 KG 前门禁）：
1) 误差相关性（与现有候选池的冗余度）
2) blocked-CV 增益（候选 + 最佳参考 的组合是否优于参考）
3) tail 稳定性（末段窗口是否恶化）
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.combination_utils import DATASET_HORIZONS, fit_ridge_robust
from src.utils.blocked_cv import blocked_cv_select_alpha, blocked_cv_splits, resolve_blocked_cv_config


SAFE_VAL_MODES = {"oof", "blocked_cv", "blocked_cv_blend", "online_rolling", "deterministic"}


@dataclass
class AuditConfig:
    max_corr_threshold: float = 0.90
    median_corr_threshold: float = 0.50
    min_cv_improve_pct: float = 0.30
    max_tail_degrade_ratio: float = 0.01
    tail_ratio: float = 0.20
    require_safe_val_mode: bool = True
    drift_low_threshold: float = 0.15
    drift_high_threshold: float = 0.30
    high_drift_min_accepted: int = 1
    high_drift_preferred_candidates: Tuple[str, ...] = ("rl_qms", "stacking_safe")
    high_drift_rescue_max_mae_gap_ratio: float = 0.04
    high_drift_rescue_min_cv_improve_pct: float = -0.05
    high_drift_rescue_max_tail_degrade_ratio: float = 0.02
    short_h_max: int = 6
    long_h_min: int = 24
    short_h_cv_multiplier: float = 0.75
    long_h_cv_multiplier: float = 1.25
    short_h_tail_multiplier: float = 1.50
    long_h_tail_multiplier: float = 0.75
    short_h_rescue_gap_multiplier: float = 1.25
    long_h_rescue_gap_multiplier: float = 0.75
    short_h_rescue_min_cv_multiplier: float = 1.20
    long_h_rescue_min_cv_multiplier: float = 0.50
    short_h_rescue_tail_multiplier: float = 1.25
    long_h_rescue_tail_multiplier: float = 0.75


def _effective_task_thresholds(config: AuditConfig, horizon: int) -> Dict[str, float]:
    """Horizon-aware thresholds: short horizon relax, long horizon tighten."""
    cv_mul = 1.0
    tail_mul = 1.0
    rescue_gap_mul = 1.0
    rescue_min_cv_mul = 1.0
    rescue_tail_mul = 1.0

    if horizon <= int(config.short_h_max):
        cv_mul = float(config.short_h_cv_multiplier)
        tail_mul = float(config.short_h_tail_multiplier)
        rescue_gap_mul = float(config.short_h_rescue_gap_multiplier)
        rescue_min_cv_mul = float(config.short_h_rescue_min_cv_multiplier)
        rescue_tail_mul = float(config.short_h_rescue_tail_multiplier)
    elif horizon >= int(config.long_h_min):
        cv_mul = float(config.long_h_cv_multiplier)
        tail_mul = float(config.long_h_tail_multiplier)
        rescue_gap_mul = float(config.long_h_rescue_gap_multiplier)
        rescue_min_cv_mul = float(config.long_h_rescue_min_cv_multiplier)
        rescue_tail_mul = float(config.long_h_rescue_tail_multiplier)

    min_cv_improve = float(config.min_cv_improve_pct) * cv_mul
    max_tail_degrade = max(float(config.max_tail_degrade_ratio) * tail_mul, 0.0)
    rescue_max_mae_gap = max(float(config.high_drift_rescue_max_mae_gap_ratio) * rescue_gap_mul, 0.0)
    rescue_min_cv_improve = float(config.high_drift_rescue_min_cv_improve_pct) * rescue_min_cv_mul
    rescue_max_tail_degrade = max(
        float(config.high_drift_rescue_max_tail_degrade_ratio) * rescue_tail_mul,
        0.0,
    )
    return {
        "min_cv_improve_pct": min_cv_improve,
        "max_tail_degrade_ratio": max_tail_degrade,
        "rescue_max_mae_gap_ratio": rescue_max_mae_gap,
        "rescue_min_cv_improve_pct": rescue_min_cv_improve,
        "rescue_max_tail_degrade_ratio": rescue_max_tail_degrade,
    }


def _resolve_cv_config(n_samples: int, horizon: int) -> Tuple[int, int, int]:
    return resolve_blocked_cv_config(n_samples=n_samples, horizon=horizon)


def _ensure_row_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "row_id" in out.columns:
        out["row_id"] = out["row_id"].astype(str)
        if not out["row_id"].duplicated().any():
            return out
    if "timestamp" not in out.columns:
        out = out.reset_index(drop=True)
        out["row_id"] = [f"row_{i}" for i in range(len(out))]
        return out
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    tmp = pd.DataFrame({"_ts": ts, "_orig_idx": np.arange(len(out))})
    tmp = tmp.sort_values(["_ts", "_orig_idx"]).reset_index(drop=True)
    tmp["_k"] = tmp.groupby("_ts", dropna=False).cumcount().astype(str)
    stable = tmp["_ts"].astype(str) + "_" + tmp["_k"]
    out = out.iloc[tmp["_orig_idx"].values].reset_index(drop=True).copy()
    out["row_id"] = stable.astype(str).values
    return out


def _load_pred(pred_root: Path, dataset: str, horizon: int, strategy: str, split: str) -> Optional[pd.DataFrame]:
    path = pred_root / dataset / f"{split}_pred_h{horizon}_{strategy}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, dtype={"row_id": "string"})
    except Exception:
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
    if "pred" not in df.columns:
        return None
    if "y" not in df.columns:
        return None
    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df))
    df = _ensure_row_id(df)
    cols = ["row_id", "timestamp", "y", "pred"]
    return df[[c for c in cols if c in df.columns]].copy()


def _load_val_mode(pred_root: Path, dataset: str, horizon: int, strategy: str) -> str:
    meta_path = pred_root / dataset / f"val_pred_h{horizon}_{strategy}.meta.json"
    if not meta_path.exists():
        return "unknown"
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return str(payload.get("val_eval_mode", "unknown"))
    except Exception:
        return "unknown"


def _list_available_strategies(pred_root: Path, dataset: str, horizon: int, split: str) -> List[str]:
    ds_dir = pred_root / dataset
    if not ds_dir.exists():
        return []
    prefix = f"{split}_pred_h{horizon}_"
    out = []
    for p in ds_dir.glob(f"{prefix}*.csv"):
        name = p.name[len(prefix):-4]
        if name:
            out.append(name)
    return sorted(set(out))


def _corr_abs(res1: np.ndarray, res2: np.ndarray) -> float:
    m = np.isfinite(res1) & np.isfinite(res2)
    if int(np.sum(m)) < 20:
        return 1.0
    res1 = res1[m]
    res2 = res2[m]
    if float(np.std(res1)) < 1e-10 or float(np.std(res2)) < 1e-10:
        return 1.0
    c = np.corrcoef(res1, res2)[0, 1]
    if not np.isfinite(c):
        return 1.0
    return float(abs(c))


def _compute_psi_1d(base: np.ndarray, target: np.ndarray, bins: int = 10) -> Optional[float]:
    base = np.asarray(base, dtype=float)
    target = np.asarray(target, dtype=float)
    base = base[np.isfinite(base)]
    target = target[np.isfinite(target)]
    if len(base) < 20 or len(target) < 20:
        return None
    try:
        qs = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(base, qs)
        edges = np.unique(edges)
        if len(edges) <= 2:
            return None
        base_hist, _ = np.histogram(base, bins=edges)
        tgt_hist, _ = np.histogram(target, bins=edges)
        base_ratio = np.clip(base_hist / max(int(base_hist.sum()), 1), 1e-6, 1.0)
        tgt_ratio = np.clip(tgt_hist / max(int(tgt_hist.sum()), 1), 1e-6, 1.0)
        psi = np.sum((base_ratio - tgt_ratio) * np.log(base_ratio / tgt_ratio))
        return float(psi) if np.isfinite(psi) else None
    except Exception:
        return None


def _estimate_task_drift_from_references(
    *,
    pred_root: Path,
    dataset: str,
    horizon: int,
    references: List[str],
    config: AuditConfig,
) -> Dict[str, Any]:
    psi_map: Dict[str, float] = {}
    for ref in references:
        r_val = _load_pred(pred_root, dataset, horizon, ref, "val")
        r_test = _load_pred(pred_root, dataset, horizon, ref, "test")
        if r_val is None or r_test is None:
            continue
        psi = _compute_psi_1d(r_val["pred"].values.astype(float), r_test["pred"].values.astype(float), bins=10)
        if psi is not None and np.isfinite(psi):
            psi_map[ref] = float(psi)
    if psi_map:
        vals = np.array(list(psi_map.values()), dtype=float)
        median_psi = float(np.median(vals))
        max_psi = float(np.max(vals))
    else:
        median_psi = 0.0
        max_psi = 0.0
    if median_psi >= float(config.drift_high_threshold):
        drift_level = "high"
    elif median_psi >= float(config.drift_low_threshold):
        drift_level = "medium"
    else:
        drift_level = "low"
    return {
        "drift_level": drift_level,
        "median_psi": median_psi,
        "max_psi": max_psi,
        "psi_by_reference": psi_map,
    }


def _maybe_apply_high_drift_rescue(
    *,
    task_out: Dict[str, Any],
    config: AuditConfig,
    task_thresholds: Optional[Dict[str, float]] = None,
) -> None:
    drift = task_out.get("drift", {})
    if not isinstance(drift, dict) or drift.get("drift_level") != "high":
        return
    accepted = task_out.get("accepted", [])
    if not isinstance(accepted, list):
        return
    min_accepted = max(int(config.high_drift_min_accepted), 0)
    if len(accepted) >= min_accepted:
        return

    candidates_payload = task_out.get("candidates", {})
    if not isinstance(candidates_payload, dict):
        return

    preferred = list(config.high_drift_preferred_candidates)
    thresholds = task_thresholds or {}
    rescue_max_mae_gap_ratio = float(
        thresholds.get("rescue_max_mae_gap_ratio", config.high_drift_rescue_max_mae_gap_ratio)
    )
    rescue_min_cv_improve_pct = float(
        thresholds.get("rescue_min_cv_improve_pct", config.high_drift_rescue_min_cv_improve_pct)
    )
    rescue_max_tail_degrade_ratio = float(
        thresholds.get("rescue_max_tail_degrade_ratio", config.high_drift_rescue_max_tail_degrade_ratio)
    )
    # 按“优先名单 -> 更高 CV 改进 -> 更小 MAE gap”排序救援候选。
    rescue_pool: List[Tuple[float, float, str]] = []
    for idx, cand in enumerate(preferred):
        info = candidates_payload.get(cand, {})
        if not isinstance(info, dict):
            continue
        if str(info.get("val_eval_mode", "unknown")) not in SAFE_VAL_MODES:
            continue
        best_ref_mae = info.get("best_reference_mae")
        cand_mae = info.get("candidate_mae")
        # Fix1-HIGH: 优先使用 effective_cv_improve（online_rolling 已替换为 rolling_improve_pct），
        # 兜底使用 cv_improve_pct，保持与 audit_task 主路径的口径一致。
        effective_cv_improve = (
            info.get("effective_cv_improve")
            if info.get("effective_cv_improve") is not None
            else info.get("cv_improve_pct")
        )
        tail_degrade_ratio = info.get("tail_degrade_ratio")
        if not (
            isinstance(best_ref_mae, (float, int))
            and isinstance(cand_mae, (float, int))
            and np.isfinite(best_ref_mae)
            and np.isfinite(cand_mae)
            and float(best_ref_mae) > 1e-10
        ):
            continue
        if cand_mae > float(best_ref_mae) * (1.0 + rescue_max_mae_gap_ratio):
            continue
        if effective_cv_improve is None or not np.isfinite(float(effective_cv_improve)):
            continue
        if float(effective_cv_improve) < rescue_min_cv_improve_pct:
            continue
        if tail_degrade_ratio is None or not np.isfinite(float(tail_degrade_ratio)):
            continue
        if float(tail_degrade_ratio) > rescue_max_tail_degrade_ratio:
            continue
        mae_gap_ratio = float(cand_mae) / max(float(best_ref_mae), 1e-10)
        # idx 用于保持优先名单顺序。
        rescue_pool.append((-float(effective_cv_improve), mae_gap_ratio + idx * 1e-6, cand))

    rescue_pool.sort()
    need = max(0, min_accepted - len(accepted))
    rescued: List[str] = []
    for _, _, cand in rescue_pool:
        if cand in accepted:
            continue
        accepted.append(cand)
        rescued.append(cand)
        if cand in task_out.get("rejected", []):
            task_out["rejected"] = [x for x in task_out["rejected"] if x != cand]
        info = candidates_payload.get(cand, {})
        if isinstance(info, dict):
            info["accept"] = True
            info["reason"] = "accepted_high_drift_rescue"
            info["rescue"] = {
                "applied": True,
                "policy": "preferred_candidate_safe_relax",
                "drift_level": "high",
            }
        if len(rescued) >= need:
            break

    task_out["rescue"] = {
        "enabled": True,
        "drift_level": "high",
        "preferred_candidates": preferred,
        "requested_min_accepted": min_accepted,
        "thresholds": {
            "rescue_max_mae_gap_ratio": rescue_max_mae_gap_ratio,
            "rescue_min_cv_improve_pct": rescue_min_cv_improve_pct,
            "rescue_max_tail_degrade_ratio": rescue_max_tail_degrade_ratio,
        },
        "rescued": rescued,
        "accepted_after_rescue": list(accepted),
    }


def _blocked_cv_mae_from_pred(y: np.ndarray, pred: np.ndarray, horizon: int) -> Optional[float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    if int(np.sum(mask)) < 50:
        return None
    y = y[mask]
    pred = pred[mask]
    n_folds, min_train, gap = _resolve_cv_config(len(y), horizon)
    splits = blocked_cv_splits(len(y), n_folds=n_folds, min_train=min_train, gap=gap)
    if not splits:
        return None
    scores = []
    for _, val_idx in splits:
        if len(val_idx) == 0:
            continue
        scores.append(float(np.mean(np.abs(y[val_idx] - pred[val_idx]))))
    return float(np.mean(scores)) if scores else None


def _blocked_cv_pair_blend_mae(
    y: np.ndarray,
    ref_pred: np.ndarray,
    cand_pred: np.ndarray,
    horizon: int,
) -> Optional[float]:
    mask = np.isfinite(y) & np.isfinite(ref_pred) & np.isfinite(cand_pred)
    if int(np.sum(mask)) < 50:
        return None
    y = y[mask]
    ref_pred = ref_pred[mask]
    cand_pred = cand_pred[mask]
    X = np.column_stack([ref_pred, cand_pred])
    n_folds, min_train, gap = _resolve_cv_config(len(y), horizon)
    best_alpha, _ = blocked_cv_select_alpha(
        X,
        y,
        alphas=[0.1, 1.0, 10.0, 100.0],
        n_folds=n_folds,
        min_train=min_train,
        positive=False,
        fit_intercept=False,
        gap=gap,
    )
    splits = blocked_cv_splits(len(y), n_folds=n_folds, min_train=min_train, gap=gap)
    if not splits:
        return None
    scores = []
    for train_idx, val_idx in splits:
        if len(val_idx) == 0:
            continue
        reg, _ = fit_ridge_robust(
            X[train_idx],
            y[train_idx],
            alpha=float(best_alpha),
            positive=False,
            fit_intercept=False,
        )
        pred = reg.predict(X[val_idx])
        scores.append(float(np.mean(np.abs(y[val_idx] - pred))))
    return float(np.mean(scores)) if scores else None


def _full_pair_tail_degrade_ratio(
    y: np.ndarray,
    ref_pred: np.ndarray,
    cand_pred: np.ndarray,
    horizon: int,
    tail_ratio: float,
) -> Optional[float]:
    mask = np.isfinite(y) & np.isfinite(ref_pred) & np.isfinite(cand_pred)
    if int(np.sum(mask)) < 50:
        return None
    y = y[mask]
    ref_pred = ref_pred[mask]
    cand_pred = cand_pred[mask]
    X = np.column_stack([ref_pred, cand_pred])
    n_folds, min_train, gap = _resolve_cv_config(len(y), horizon)
    best_alpha, _ = blocked_cv_select_alpha(
        X,
        y,
        alphas=[0.1, 1.0, 10.0, 100.0],
        n_folds=n_folds,
        min_train=min_train,
        positive=False,
        fit_intercept=False,
        gap=gap,
    )
    reg, _ = fit_ridge_robust(
        X,
        y,
        alpha=float(best_alpha),
        positive=False,
        fit_intercept=False,
    )
    blend_pred = reg.predict(X)
    tail_n = max(32, int(len(y) * tail_ratio))
    tail_n = min(max(tail_n, 1), len(y))
    tail_slice = slice(len(y) - tail_n, len(y))
    ref_tail = float(np.mean(np.abs(y[tail_slice] - ref_pred[tail_slice])))
    blend_tail = float(np.mean(np.abs(y[tail_slice] - blend_pred[tail_slice])))
    if not np.isfinite(ref_tail) or ref_tail <= 1e-10:
        return None
    return float((blend_tail - ref_tail) / ref_tail)


def _rolling_mae_improve(
    y: np.ndarray,
    ref_pred: np.ndarray,
    cand_pred: np.ndarray,
    window: int = 48,
) -> Optional[float]:
    """Fix5b: 滚动窗口 MAE 改进率，专为 online_rolling 策略设计。
    用 50% 重叠滑动窗口（step=window//2）计算候选策略相对参考模型的平均 MAE 改进百分比。
    与 blocked CV 不同，此方法保留时序顺序，不做 train/val 拆分，
    更适合评估顺序依赖的在线策略价值。
    """
    mask = np.isfinite(y) & np.isfinite(ref_pred) & np.isfinite(cand_pred)
    if int(np.sum(mask)) < max(window, 50):
        return None
    y = y[mask]
    ref_pred = ref_pred[mask]
    cand_pred = cand_pred[mask]
    n = len(y)
    step = max(window // 2, 1)
    ref_maes, cand_maes = [], []
    for start in range(0, n - window + 1, step):
        sl = slice(start, start + window)
        ref_maes.append(float(np.mean(np.abs(y[sl] - ref_pred[sl]))))
        cand_maes.append(float(np.mean(np.abs(y[sl] - cand_pred[sl]))))
    if not ref_maes:
        return None
    ref_mean = float(np.mean(ref_maes))
    cand_mean = float(np.mean(cand_maes))
    if ref_mean <= 1e-10:
        return None
    return float((ref_mean - cand_mean) / ref_mean * 100.0)


def audit_task(
    *,
    pred_root: Path,
    dataset: str,
    horizon: int,
    candidates: List[str],
    config: AuditConfig,
) -> Dict[str, Any]:
    available_val = _list_available_strategies(pred_root, dataset, horizon, "val")
    references = [s for s in available_val if s not in set(candidates)]
    task_out: Dict[str, Any] = {
        "accepted": [],
        "rejected": [],
        "candidates": {},
        "references": references,
    }
    if not references:
        for c in candidates:
            task_out["candidates"][c] = {
                "accept": False,
                "reason": "no_reference_models",
            }
            task_out["rejected"].append(c)
        return task_out
    drift_meta = _estimate_task_drift_from_references(
        pred_root=pred_root,
        dataset=dataset,
        horizon=horizon,
        references=references,
        config=config,
    )
    task_out["drift"] = drift_meta
    task_thresholds = _effective_task_thresholds(config, horizon)
    task_out["thresholds"] = task_thresholds

    for cand in candidates:
        result: Dict[str, Any] = {"accept": False}
        c_val = _load_pred(pred_root, dataset, horizon, cand, "val")
        c_test = _load_pred(pred_root, dataset, horizon, cand, "test")
        if c_val is None or c_test is None:
            result["reason"] = "missing_val_or_test_prediction"
            task_out["candidates"][cand] = result
            task_out["rejected"].append(cand)
            continue

        val_mode = _load_val_mode(pred_root, dataset, horizon, cand)
        result["val_eval_mode"] = val_mode
        if config.require_safe_val_mode and val_mode not in SAFE_VAL_MODES:
            result["reason"] = f"unsafe_val_mode:{val_mode}"
            task_out["candidates"][cand] = result
            task_out["rejected"].append(cand)
            continue

        corr_vals: List[float] = []
        ref_rows: List[Tuple[str, float, pd.DataFrame]] = []
        for ref in references:
            r_val = _load_pred(pred_root, dataset, horizon, ref, "val")
            if r_val is None:
                continue
            merged = c_val[["row_id", "y", "pred"]].merge(
                r_val[["row_id", "pred"]].rename(columns={"pred": "pred_ref"}),
                on="row_id",
                how="inner",
            )
            if len(merged) < 50:
                continue
            y = merged["y"].values.astype(float)
            cand_pred = merged["pred"].values.astype(float)
            ref_pred = merged["pred_ref"].values.astype(float)
            valid = np.isfinite(y) & np.isfinite(cand_pred) & np.isfinite(ref_pred)
            if int(np.sum(valid)) < 50:
                continue
            y = y[valid]
            cand_pred = cand_pred[valid]
            ref_pred = ref_pred[valid]
            corr_vals.append(_corr_abs(cand_pred - y, ref_pred - y))
            ref_mae = float(np.mean(np.abs(y - ref_pred)))
            ref_rows.append((ref, ref_mae, merged.loc[valid].copy()))

        if not ref_rows:
            result["reason"] = "insufficient_reference_overlap"
            task_out["candidates"][cand] = result
            task_out["rejected"].append(cand)
            continue

        ref_rows.sort(key=lambda x: x[1])
        best_ref, best_ref_mae, best_merged = ref_rows[0]
        y = best_merged["y"].values.astype(float)
        cand_pred = best_merged["pred"].values.astype(float)
        ref_pred = best_merged["pred_ref"].values.astype(float)
        valid = np.isfinite(y) & np.isfinite(cand_pred) & np.isfinite(ref_pred)
        if int(np.sum(valid)) < 50:
            result["reason"] = "insufficient_finite_overlap"
            task_out["candidates"][cand] = result
            task_out["rejected"].append(cand)
            continue
        y = y[valid]
        cand_pred = cand_pred[valid]
        ref_pred = ref_pred[valid]
        cand_mae = float(np.mean(np.abs(y - cand_pred)))

        max_abs_corr = float(np.max(corr_vals)) if corr_vals else 1.0
        median_abs_corr = float(np.median(corr_vals)) if corr_vals else 1.0
        cv_ref_mae = _blocked_cv_mae_from_pred(y, ref_pred, horizon)
        cv_combo_mae = _blocked_cv_pair_blend_mae(y, ref_pred, cand_pred, horizon)
        tail_degrade_ratio = _full_pair_tail_degrade_ratio(y, ref_pred, cand_pred, horizon, config.tail_ratio)
        cv_improve_pct = None
        if cv_ref_mae is not None and cv_combo_mae is not None and cv_ref_mae > 1e-10:
            cv_improve_pct = float((cv_ref_mae - cv_combo_mae) / cv_ref_mae * 100.0)

        # Fix5: online_rolling 策略差异化审计
        # 在线策略的预测依赖历史反馈序列，残差天然与参考模型高相关，
        # 且 blocked CV blend（静态评估）无法体现其时序适应优势。
        # 对 online_rolling 放宽 corr 阈值，并用 rolling MAE 改进率替代 cv_improve 判断。
        is_online_rolling = (val_mode == "online_rolling")
        if is_online_rolling:
            effective_max_corr = min(config.max_corr_threshold + 0.08, 0.97)
            effective_median_corr = min(config.median_corr_threshold + 0.20, 0.85)
        else:
            effective_max_corr = config.max_corr_threshold
            effective_median_corr = config.median_corr_threshold

        corr_ok_base = (
            max_abs_corr <= effective_max_corr
            and median_abs_corr <= effective_median_corr
        )

        # Fix5b: online_rolling 用 rolling MAE 改进替代 blocked CV improve
        rolling_improve_pct: Optional[float] = None
        if is_online_rolling:
            rolling_improve_pct = _rolling_mae_improve(
                y, ref_pred, cand_pred,
                window=max(48, int(len(y) * 0.08)),
            )

        if is_online_rolling and rolling_improve_pct is not None:
            effective_cv_improve = rolling_improve_pct
        else:
            effective_cv_improve = cv_improve_pct

        cv_ok = (
            (effective_cv_improve is not None and effective_cv_improve >= task_thresholds["min_cv_improve_pct"])
            or cand_mae <= best_ref_mae * 0.998
        )
        tail_ok = (
            tail_degrade_ratio is not None
            and tail_degrade_ratio <= task_thresholds["max_tail_degrade_ratio"]
        )
        corr_ok = corr_ok_base or (
            effective_cv_improve is not None
            and effective_cv_improve >= max(0.5, task_thresholds["min_cv_improve_pct"] * 2.0)
        )

        reasons = []
        if not corr_ok:
            reasons.append(
                f"high_corr:max={max_abs_corr:.3f},median={median_abs_corr:.3f}"
            )
        if not cv_ok:
            reasons.append(
                "cv_improve_insufficient:"
                f"{effective_cv_improve if effective_cv_improve is not None else 'NA'}"
            )
        if not tail_ok:
            reasons.append(
                "tail_degrade_too_high:"
                f"{tail_degrade_ratio if tail_degrade_ratio is not None else 'NA'}"
            )

        result.update({
            "best_reference": best_ref,
            "best_reference_mae": best_ref_mae,
            "candidate_mae": cand_mae,
            "max_abs_corr": max_abs_corr,
            "median_abs_corr": median_abs_corr,
            "cv_ref_mae": cv_ref_mae,
            "cv_combo_mae": cv_combo_mae,
            "cv_improve_pct": cv_improve_pct,
            "rolling_improve_pct": rolling_improve_pct,
            "effective_cv_improve": effective_cv_improve,
            "tail_degrade_ratio": tail_degrade_ratio,
        })

        if corr_ok and cv_ok and tail_ok:
            result["accept"] = True
            result["reason"] = "accepted"
            task_out["accepted"].append(cand)
        else:
            result["reason"] = ";".join(reasons) if reasons else "rejected"
            task_out["rejected"].append(cand)
        task_out["candidates"][cand] = result

    _maybe_apply_high_drift_rescue(task_out=task_out, config=config, task_thresholds=task_thresholds)
    return task_out


def main() -> None:
    parser = argparse.ArgumentParser(description="离线审计扩展候选策略")
    parser.add_argument("--pred-root", type=Path, required=True, help="预测根目录（baselines）")
    parser.add_argument("--out", type=Path, required=True, help="输出 JSON 路径")
    parser.add_argument("--datasets", nargs="*", default=None, help="仅审计指定数据集")
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=["rl_qms", "stacking_safe"],
        help="待审计候选策略",
    )
    parser.add_argument("--max-corr-threshold", type=float, default=0.90)
    parser.add_argument("--median-corr-threshold", type=float, default=0.50)
    parser.add_argument("--min-cv-improve-pct", type=float, default=0.30)
    parser.add_argument("--max-tail-degrade-ratio", type=float, default=0.01)
    parser.add_argument("--tail-ratio", type=float, default=0.20)
    parser.add_argument("--drift-low-threshold", type=float, default=0.15,
                        help="任务漂移分层 low 阈值（基于预测分布 PSI 中位数）")
    parser.add_argument("--drift-high-threshold", type=float, default=0.30,
                        help="任务漂移分层 high 阈值（基于预测分布 PSI 中位数）")
    parser.add_argument("--high-drift-min-accepted", type=int, default=1,
                        help="高漂移任务至少保留的扩展候选数（通过救援策略兜底）")
    parser.add_argument("--high-drift-preferred-candidates", nargs="*",
                        default=["rl_qms", "stacking_safe"],
                        help="高漂移救援优先候选（默认 rl_qms stacking_safe）")
    parser.add_argument("--high-drift-rescue-max-mae-gap-ratio", type=float, default=0.04,
                        help="高漂移救援：candidate_mae 相对 best_ref_mae 最大允许劣化比例")
    parser.add_argument("--high-drift-rescue-min-cv-improve-pct", type=float, default=-0.05,
                        help="高漂移救援：最小允许 CV 改进百分比（可为负，表示小幅退化可接受）")
    parser.add_argument("--high-drift-rescue-max-tail-degrade-ratio", type=float, default=0.02,
                        help="高漂移救援：tail 最多允许劣化比例")
    parser.add_argument("--short-h-max", type=int, default=6,
                        help="短步长上界（<=该值视为 short）")
    parser.add_argument("--long-h-min", type=int, default=24,
                        help="长步长下界（>=该值视为 long）")
    parser.add_argument("--short-h-cv-multiplier", type=float, default=0.75,
                        help="短步长 CV 阈值乘子（<1 放宽）")
    parser.add_argument("--long-h-cv-multiplier", type=float, default=1.25,
                        help="长步长 CV 阈值乘子（>1 收紧）")
    parser.add_argument("--short-h-tail-multiplier", type=float, default=1.50,
                        help="短步长 tail 阈值乘子（>1 放宽）")
    parser.add_argument("--long-h-tail-multiplier", type=float, default=0.75,
                        help="长步长 tail 阈值乘子（<1 收紧）")
    parser.add_argument("--short-h-rescue-gap-multiplier", type=float, default=1.25,
                        help="短步长 rescue mae-gap 阈值乘子")
    parser.add_argument("--long-h-rescue-gap-multiplier", type=float, default=0.75,
                        help="长步长 rescue mae-gap 阈值乘子")
    parser.add_argument("--short-h-rescue-min-cv-multiplier", type=float, default=1.20,
                        help="短步长 rescue min-cv 阈值乘子")
    parser.add_argument("--long-h-rescue-min-cv-multiplier", type=float, default=0.50,
                        help="长步长 rescue min-cv 阈值乘子")
    parser.add_argument("--short-h-rescue-tail-multiplier", type=float, default=1.25,
                        help="短步长 rescue tail 阈值乘子")
    parser.add_argument("--long-h-rescue-tail-multiplier", type=float, default=0.75,
                        help="长步长 rescue tail 阈值乘子")
    parser.add_argument("--allow-unsafe-val-mode", action="store_true",
                        help="允许 unknown/in_sample val_mode（默认禁止）")
    parser.add_argument("--strict-min-accepted", action="store_true",
                        help="启用每任务最少 accepted 数门禁（失败返回非零）")
    parser.add_argument("--min-accepted-per-task", type=int, default=0,
                        help="strict-min-accepted 模式下每任务最少 accepted 数")
    args = parser.parse_args()

    pred_root = args.pred_root if args.pred_root.is_absolute() else PROJECT_ROOT / args.pred_root
    out_path = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or list(DATASET_HORIZONS.keys())
    config = AuditConfig(
        max_corr_threshold=float(args.max_corr_threshold),
        median_corr_threshold=float(args.median_corr_threshold),
        min_cv_improve_pct=float(args.min_cv_improve_pct),
        max_tail_degrade_ratio=float(args.max_tail_degrade_ratio),
        tail_ratio=float(args.tail_ratio),
        require_safe_val_mode=not args.allow_unsafe_val_mode,
        drift_low_threshold=float(args.drift_low_threshold),
        drift_high_threshold=float(args.drift_high_threshold),
        high_drift_min_accepted=int(args.high_drift_min_accepted),
        high_drift_preferred_candidates=tuple(args.high_drift_preferred_candidates or []),
        high_drift_rescue_max_mae_gap_ratio=float(args.high_drift_rescue_max_mae_gap_ratio),
        high_drift_rescue_min_cv_improve_pct=float(args.high_drift_rescue_min_cv_improve_pct),
        high_drift_rescue_max_tail_degrade_ratio=float(args.high_drift_rescue_max_tail_degrade_ratio),
        short_h_max=int(args.short_h_max),
        long_h_min=int(args.long_h_min),
        short_h_cv_multiplier=float(args.short_h_cv_multiplier),
        long_h_cv_multiplier=float(args.long_h_cv_multiplier),
        short_h_tail_multiplier=float(args.short_h_tail_multiplier),
        long_h_tail_multiplier=float(args.long_h_tail_multiplier),
        short_h_rescue_gap_multiplier=float(args.short_h_rescue_gap_multiplier),
        long_h_rescue_gap_multiplier=float(args.long_h_rescue_gap_multiplier),
        short_h_rescue_min_cv_multiplier=float(args.short_h_rescue_min_cv_multiplier),
        long_h_rescue_min_cv_multiplier=float(args.long_h_rescue_min_cv_multiplier),
        short_h_rescue_tail_multiplier=float(args.short_h_rescue_tail_multiplier),
        long_h_rescue_tail_multiplier=float(args.long_h_rescue_tail_multiplier),
    )

    payload: Dict[str, Any] = {
        "meta": {
            "pred_root": str(pred_root),
            "candidates": args.candidates,
            "config": {
                "max_corr_threshold": config.max_corr_threshold,
                "median_corr_threshold": config.median_corr_threshold,
                "min_cv_improve_pct": config.min_cv_improve_pct,
                "max_tail_degrade_ratio": config.max_tail_degrade_ratio,
                "tail_ratio": config.tail_ratio,
                "require_safe_val_mode": config.require_safe_val_mode,
                "drift_low_threshold": config.drift_low_threshold,
                "drift_high_threshold": config.drift_high_threshold,
                "high_drift_min_accepted": config.high_drift_min_accepted,
                "high_drift_preferred_candidates": list(config.high_drift_preferred_candidates),
                "high_drift_rescue_max_mae_gap_ratio": config.high_drift_rescue_max_mae_gap_ratio,
                "high_drift_rescue_min_cv_improve_pct": config.high_drift_rescue_min_cv_improve_pct,
                "high_drift_rescue_max_tail_degrade_ratio": config.high_drift_rescue_max_tail_degrade_ratio,
                "short_h_max": config.short_h_max,
                "long_h_min": config.long_h_min,
                "short_h_cv_multiplier": config.short_h_cv_multiplier,
                "long_h_cv_multiplier": config.long_h_cv_multiplier,
                "short_h_tail_multiplier": config.short_h_tail_multiplier,
                "long_h_tail_multiplier": config.long_h_tail_multiplier,
                "short_h_rescue_gap_multiplier": config.short_h_rescue_gap_multiplier,
                "long_h_rescue_gap_multiplier": config.long_h_rescue_gap_multiplier,
                "short_h_rescue_min_cv_multiplier": config.short_h_rescue_min_cv_multiplier,
                "long_h_rescue_min_cv_multiplier": config.long_h_rescue_min_cv_multiplier,
                "short_h_rescue_tail_multiplier": config.short_h_rescue_tail_multiplier,
                "long_h_rescue_tail_multiplier": config.long_h_rescue_tail_multiplier,
            },
        },
        "tasks": {},
    }
    rows: List[Dict[str, Any]] = []
    gate_violations: List[str] = []

    for ds in datasets:
        horizons = DATASET_HORIZONS.get(ds, [])
        payload["tasks"][ds] = {}
        for h in horizons:
            task = audit_task(
                pred_root=pred_root,
                dataset=ds,
                horizon=h,
                candidates=args.candidates,
                config=config,
            )
            payload["tasks"][ds][str(h)] = task
            if args.strict_min_accepted and len(task.get("accepted", [])) < int(args.min_accepted_per_task):
                gate_violations.append(
                    f"{ds} h={h}: accepted={len(task.get('accepted', []))} < {int(args.min_accepted_per_task)}"
                )
            for cand, info in task.get("candidates", {}).items():
                row = {
                    "dataset": ds,
                    "horizon": h,
                    "candidate": cand,
                    "accept": bool(info.get("accept", False)),
                    "reason": info.get("reason"),
                    "rescued": bool((info.get("rescue") or {}).get("applied", False)),
                    "val_eval_mode": info.get("val_eval_mode"),
                    "task_drift_level": (task.get("drift") or {}).get("drift_level"),
                    "task_median_psi": (task.get("drift") or {}).get("median_psi"),
                    "best_reference": info.get("best_reference"),
                    "max_abs_corr": info.get("max_abs_corr"),
                    "median_abs_corr": info.get("median_abs_corr"),
                    "cv_ref_mae": info.get("cv_ref_mae"),
                    "cv_combo_mae": info.get("cv_combo_mae"),
                    "cv_improve_pct": info.get("cv_improve_pct"),
                    "tail_degrade_ratio": info.get("tail_degrade_ratio"),
                }
                rows.append(row)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    total_tasks = sum(len(DATASET_HORIZONS.get(ds, [])) for ds in datasets)
    accepted_total = sum(
        len(payload["tasks"].get(ds, {}).get(str(h), {}).get("accepted", []))
        for ds in datasets for h in DATASET_HORIZONS.get(ds, [])
    )
    print(f"[DONE] candidate audit json: {out_path}")
    print(f"[DONE] candidate audit csv: {csv_path}")
    print(f"[INFO] tasks={total_tasks}, total_accepted_candidates={accepted_total}")
    if gate_violations:
        print("[ERROR] strict-min-accepted violations:")
        for item in gate_violations:
            print(f"  - {item}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
