"""Drift estimation and drift-aware weighting utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    DRIFT_PSI_HIGH_THRESHOLD,
    DRIFT_PSI_LOW_THRESHOLD,
    KG_DRIFT_DECAY_MAX_MULTIPLIER,
    KG_TEMPORAL_WEIGHT_MIN_RATIO,
    PROTOCOL_B_GLOBAL_MIN_IMPROVEMENT,
    PROTOCOL_B_LARGE_SAMPLE_THRESHOLD,
)

def _compute_psi_1d(base: np.ndarray, target: np.ndarray, bins: int = 10) -> Optional[float]:
    base = np.asarray(base, dtype=float)
    target = np.asarray(target, dtype=float)
    base = base[np.isfinite(base)]
    target = target[np.isfinite(target)]
    if len(base) < 20 or len(target) < 20:
        return None
    try:
        qs = np.linspace(0, 1, bins + 1)
        edges = np.quantile(base, qs)
        edges = np.unique(edges)
        if len(edges) <= 2:
            return None
        base_hist, _ = np.histogram(base, bins=edges)
        tgt_hist, _ = np.histogram(target, bins=edges)
        base_ratio = np.clip(base_hist / max(base_hist.sum(), 1), 1e-6, 1.0)
        tgt_ratio = np.clip(tgt_hist / max(tgt_hist.sum(), 1), 1e-6, 1.0)
        psi = np.sum((base_ratio - tgt_ratio) * np.log(base_ratio / tgt_ratio))
        return float(psi)
    except Exception:
        return None

def _estimate_prediction_space_drift(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    model_cols: List[str],
) -> Dict[str, Any]:
    psi_map: Dict[str, float] = {}
    for c in model_cols:
        if c not in df_val.columns or c not in df_test.columns:
            continue
        psi = _compute_psi_1d(df_val[c].values, df_test[c].values, bins=10)
        if psi is not None and np.isfinite(psi):
            psi_map[c] = float(psi)
    if psi_map:
        psi_values = np.array(list(psi_map.values()), dtype=float)
        median_psi = float(np.median(psi_values))
        max_psi = float(np.max(psi_values))
    else:
        median_psi = 0.0
        max_psi = 0.0
    if median_psi >= DRIFT_PSI_HIGH_THRESHOLD:
        level = "high"
    elif median_psi >= DRIFT_PSI_LOW_THRESHOLD:
        level = "medium"
    else:
        level = "low"
    return {
        "psi_by_model": psi_map,
        "median_psi": median_psi,
        "max_psi": max_psi,
        "drift_level": level,
    }

def _compute_temporal_weights(
    n_samples: int,
    decay: float,
    min_ratio: Optional[float] = None,
) -> Optional[np.ndarray]:
    """计算时间衰减样本权重（近期样本更高）。"""
    if decay <= 0 or n_samples <= 1:
        return None
    t = np.arange(n_samples - 1, -1, -1, dtype=float)
    weights = np.exp(-decay * t)
    min_w_ratio = KG_TEMPORAL_WEIGHT_MIN_RATIO if min_ratio is None else float(min_ratio)
    min_w_ratio = min(max(min_w_ratio, 0.0), 1.0)
    if min_w_ratio > 0 and np.isfinite(weights).all():
        max_w = float(np.max(weights))
        if np.isfinite(max_w) and max_w > 0:
            min_w = max_w * min_w_ratio
            weights = np.maximum(weights, min_w)
    weights = weights / np.mean(weights)
    return weights

def _resolve_drift_aware_temporal_decay(
    *,
    base_decay: float,
    drift_level: str,
    drift_median_psi: Optional[float],
) -> Tuple[float, Dict[str, Any]]:
    """
    根据漂移等级与 PSI 中位数自适应调整时间衰减。
    PSI 越高，越强调近期样本，缓解 val->test 外推偏差。
    """
    decay = max(float(base_decay), 0.0)
    psi = None
    if drift_median_psi is not None:
        try:
            psi = float(drift_median_psi)
        except Exception:
            psi = None
    multiplier = 1.0
    if drift_level == "high":
        multiplier = max(multiplier, 2.0)
    elif drift_level == "medium":
        multiplier = max(multiplier, 1.3)
    if psi is not None and np.isfinite(psi) and psi > 0:
        multiplier = max(multiplier, 1.0 + min(psi, 1.0) * 2.5)
        if psi >= DRIFT_PSI_HIGH_THRESHOLD:
            multiplier = max(multiplier, 2.5 + min((psi - DRIFT_PSI_HIGH_THRESHOLD) * 2.0, 2.5))
    multiplier = min(max(multiplier, 1.0), KG_DRIFT_DECAY_MAX_MULTIPLIER)
    decay = decay * multiplier
    return decay, {
        "base_decay": float(base_decay),
        "effective_decay": float(decay),
        "multiplier": float(multiplier),
        "drift_level": str(drift_level),
        "drift_median_psi": float(psi) if psi is not None and np.isfinite(psi) else None,
    }

def _resolve_protocol_b_global_min_improve(n_val: int, drift_level: str) -> float:
    """
    Protocol B 全局最小改进门槛分层：
    - 大样本放宽到 0
    - 高漂移场景适度放宽，避免被微小阈值过度回退
    """
    val = float(PROTOCOL_B_GLOBAL_MIN_IMPROVEMENT)
    if n_val >= PROTOCOL_B_LARGE_SAMPLE_THRESHOLD:
        return 0.0
    if drift_level == "high":
        if n_val >= 2000:
            return min(val, 0.0002)
        return min(val, 0.0003)
    if drift_level == "medium" and n_val >= 1500:
        return min(val, 0.0003)
    return val
