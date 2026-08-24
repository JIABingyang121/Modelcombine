"""基于 blocked-CV/OOF 边际贡献的带符号关系证据（Task 8.3 Task 5）。

关系事件的方向和幅度只由 blocked-CV/OOF 决定；样本内 validation gain 与最终
Ridge 权重仅作审计字段。OOF 不可用时不得回退到样本内指标。
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .model_selection import fit_static_weight_ridge

# 固定中性区间：|oof_gain| <= 0.005 时不产生事件，v6 期间不得修改。
RELATION_GAIN_DEADBAND = 0.005


def _skipped(reason: str) -> Dict[str, Any]:
    return {
        "eligible": False,
        "skip_reason": reason,
        "deadband": RELATION_GAIN_DEADBAND,
        "evidence_mode": None,
        "by_model": {},
    }


def classify_relation_gain(
    *,
    validation_gain: float,
    oof_gain: Optional[float],
    final_weight: float,
) -> Dict[str, Any]:
    """按 OOF 边际贡献分类；validation_gain 与 final_weight 只作审计。"""
    audit = {
        "validation_gain": validation_gain,
        "oof_gain": oof_gain,
        "final_weight": final_weight,
    }
    if oof_gain is None or not np.isfinite(oof_gain):
        return {**audit, "polarity": "neutral", "magnitude": 0.0, "skip_reason": "no_oof_evidence"}
    if abs(oof_gain) <= RELATION_GAIN_DEADBAND:
        return {**audit, "polarity": "neutral", "magnitude": 0.0, "skip_reason": "oof_gain_in_deadband"}
    return {
        **audit,
        "polarity": "positive" if oof_gain > 0 else "negative",
        "magnitude": min(abs(oof_gain), 1.0),
        "skip_reason": None,
    }


def _validation_metrics(
    df_val: pd.DataFrame,
    models: Sequence[str],
    **metric_kwargs: Any,
) -> tuple[Optional[float], Optional[float]]:
    """复用生产 Ridge 的 alpha 候选与时序加权，计算样本内 MAE 与 blocked-CV/OOF。

    样本内 MAE 取第一个预测数组；OOF 取 ridge_meta["cv_mae"]（blocked-CV 选出
    alpha 时的验证 MAE）。返回 (val_mae, oof_mae)。
    """
    if not models:
        return None, None
    pred_val, _pred_test, _weights, meta = fit_static_weight_ridge(
        df_val,
        df_val,
        selected_models=list(models),
        **metric_kwargs,
    )
    y = df_val["y"].to_numpy()
    val_mae = float(np.mean(np.abs(pred_val - y)))
    cv_mae = meta.get("cv_mae")
    oof_mae = float(cv_mae) if cv_mae is not None and np.isfinite(float(cv_mae)) else None
    return val_mae, oof_mae


def compute_relation_feedback_evidence(
    *,
    df_val: pd.DataFrame,
    candidate_models: Sequence[str],
    selected_models: Sequence[str],
    horizon: int,
    final_weights: Mapping[str, float],
    fallback_target: Optional[str],
    alpha_candidates: Optional[List[float]],
    temporal_decay: Optional[float],
    temporal_decay_meta: Optional[Dict[str, Any]],
    temporal_min_weight_ratio: Optional[float],
) -> Dict[str, Any]:
    """为每个最终模型计算验证边际贡献证据；guard 回退时跳过并留痕。"""
    if fallback_target is not None:
        return _skipped(f"guard_fallback:{fallback_target}")
    metric_kwargs = {
        "horizon": horizon,
        "alpha_candidates": alpha_candidates,
        "temporal_decay": temporal_decay,
        "temporal_decay_meta": temporal_decay_meta,
        "temporal_min_weight_ratio": temporal_min_weight_ratio,
    }
    full_val, full_oof = _validation_metrics(df_val, selected_models, **metric_kwargs)
    by_model: Dict[str, Any] = {}
    for model in selected_models:
        reference = [m for m in selected_models if m != model]
        if not reference:
            alternatives = [m for m in candidate_models if m != model]
            if not alternatives:
                continue
            reference = [min(
                alternatives,
                key=lambda m: float(np.mean(np.abs(df_val[m].to_numpy() - df_val["y"].to_numpy()))),
            )]
        ref_val, ref_oof = _validation_metrics(df_val, reference, **metric_kwargs)
        gain_val = (ref_val - full_val) / max(ref_val, 1e-12) if ref_val is not None and full_val is not None else 0.0
        gain_oof = (
            (ref_oof - full_oof) / max(ref_oof, 1e-12)
            if ref_oof is not None and full_oof is not None
            else None
        )
        by_model[model] = classify_relation_gain(
            validation_gain=gain_val,
            oof_gain=gain_oof,
            final_weight=float(final_weights.get(model, 0.0)),
        )
    has_oof = any(
        item["oof_gain"] is not None and np.isfinite(item["oof_gain"])
        for item in by_model.values()
    )
    return {
        "eligible": has_oof,
        "skip_reason": None if has_oof else "no_oof_evidence",
        "deadband": RELATION_GAIN_DEADBAND,
        "evidence_mode": "blocked_cv_oof" if has_oof else None,
        "by_model": by_model,
    }
