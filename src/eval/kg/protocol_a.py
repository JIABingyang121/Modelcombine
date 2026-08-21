"""Protocol A implementation (prediction-only KG)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.eval.combination_utils import evaluate
from src.eval.metrics import evaluate as evaluate_with_tail
from src.eval.strategy_naming import get_strategy_naming
from src.graph.model_graph import ModelGraph

from .config import *
from .config import _extended_pool_penalty_models
from .conflict import check_conflict
from .drift import _compute_temporal_weights, _resolve_drift_aware_temporal_decay
from .model_selection import (
    _cleanup_zero_weight_models_and_refit,
    _find_reliable_small_combo,
    _fit_ridge_and_mae,
    _select_stepwise_alpha,
    fit_static_weight_ridge,
    forward_stepwise_select,
)
from .stability import _dedup_and_stability_filter, adaptive_max_models, _adaptive_stepwise_min_improve

def _merge_eval_metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    base = evaluate(pred, y)
    try:
        rich = evaluate_with_tail(y, pred)
    except Exception:
        rich = {}
    if isinstance(rich, dict):
        base.update(rich)
    return base

def kg_combination_pred_only(df_val: pd.DataFrame, df_test: pd.DataFrame,
                              model_cols: List[str], horizon: int,
                              corr_threshold: float = 0.5,
                              complementary_weight: float = 1.2,
                              dataset_name: Optional[str] = None,
                              return_predictions: bool = False) -> Dict:
    """
    KG 组合 - 仅使用预测数据
    
    步骤:
    1. 计算模型 MAE
    2. 计算误差相关性 -> complementary/conflict 边
    3. 基于图谱推理选择模型组合
    """
    y_val = df_val["y"].values
    y_test = df_test["y"].values
    
    # 计算 MAE
    maes = {}
    for m in model_cols:
        maes[m] = np.mean(np.abs(df_val[m].values - y_val))

    filter_ctx = _dedup_and_stability_filter(
        df_val=df_val,
        df_test=df_test,
        model_cols=model_cols,
        maes=maes,
        horizon=horizon,
    )
    model_cols = filter_ctx["model_cols"]
    drift_meta = filter_ctx["drift_meta"]
    drift_level = filter_ctx["drift_level"]
    stability_meta = filter_ctx["stability_meta"]
    stability_removed = filter_ctx["stability_removed"]
    error_corrs = filter_ctx["error_corrs"]
    
    # 构建图
    mg = ModelGraph()
    
    for m in model_cols:
        mg.add_model_node(m, {"mae": maes[m]})
    
    for (m1, m2), corr in error_corrs.items():
        # 低相关 -> complementary (双向)
        if corr < corr_threshold:
            mg.add_relation(m1, m2, "complementary", weight=1.0 - corr)
            mg.add_relation(m2, m1, "complementary", weight=1.0 - corr)
        # 高相关 -> conflict (双向)
        elif corr > 0.9:
            mg.add_relation(m1, m2, "conflict", weight=corr)
            mg.add_relation(m2, m1, "conflict", weight=corr)
    
    # 基于性能和互补性打分（用于解释和排序先验）
    model_scores = {}
    base_raw = {m: 1.0 / (maes[m] + 1e-6) for m in model_cols}
    base_max = max(base_raw.values()) if base_raw else 1.0
    high_drift_penalty_models = _extended_pool_penalty_models() if drift_level == "high" else set()
    for m in model_cols:
        # 基础分归一化，避免与互补加成量纲失配。
        base_score = base_raw[m] / (base_max + 1e-10)
        
        # 互补性加成
        complementary_bonus = 0.0
        for neighbor in mg.neighbors_by_type(m, "complementary"):
            if neighbor in model_cols:
                edge_weight = mg.G[m][neighbor].get("weight", 0.5)
                complementary_bonus += edge_weight * complementary_weight
        
        score = base_score + complementary_bonus
        if m in high_drift_penalty_models:
            score *= KG_HIGH_DRIFT_EXTENDED_SCORE_FACTOR
        model_scores[m] = score

    max_models = adaptive_max_models(len(df_val), len(model_cols))
    if len(df_val) < 1000:
        max_models = min(max_models, 2)
    if drift_level == "high":
        max_models = max(2, max_models - 1)
    if dataset_name == "aemo_nsw" and horizon >= 24 and drift_level == "high":
        max_models = min(max_models, 2)
    # P0-3: h24 max_models 硬上限
    if horizon >= KG_LONG_HORIZON_MIN_H:
        max_models = min(max_models, KG_LONG_HORIZON_MAX_MODELS_CAP)
    candidate_order = [m for m, _ in sorted(model_scores.items(), key=lambda x: x[1], reverse=True)]
    drift_decay, drift_decay_meta = _resolve_drift_aware_temporal_decay(
        base_decay=KG_RIDGE_TEMPORAL_DECAY,
        drift_level=drift_level,
        drift_median_psi=drift_meta.get("median_psi"),
    )
    sample_weight = _compute_temporal_weights(len(y_val), drift_decay)
    alpha_candidates = HIGH_DRIFT_ALPHA_CANDIDATES if drift_level == "high" else None
    # P0-3: 长步长 alpha 倍率加强正则化
    if horizon >= KG_LONG_HORIZON_MIN_H and alpha_candidates is None:
        alpha_candidates = [a * KG_LONG_HORIZON_ALPHA_MULTIPLIER for a in [1.0, 10.0, 50.0, 100.0, 500.0]]
    stepwise_min_improve = _adaptive_stepwise_min_improve(
        drift_level=drift_level,
        n_candidates=len(model_cols),
        n_val=len(df_val),
    )

    stepwise_alpha = _select_stepwise_alpha(
        X_val=df_val[model_cols].values,
        y_val=y_val,
        sample_weight=sample_weight,
        horizon=horizon,
        alphas=alpha_candidates,
        default_alpha=1.0,
    )
    stepwise_selected, stepwise_meta = forward_stepwise_select(
        df_val=df_val,
        candidate_models=candidate_order,
        base_maes=maes,
        max_models=max_models,
        horizon=horizon,
        min_improve_ratio=stepwise_min_improve,
        sample_weight=sample_weight,
        conflict_checker=lambda trial: len(check_conflict(mg, trial)) > 0,
        alpha=stepwise_alpha,
    )
    selected = stepwise_selected
    if len(selected) < 2:
        # 回退：按图打分+冲突过滤补齐
        selected = []
        for m in candidate_order:
            if not check_conflict(mg, selected + [m]):
                selected.append(m)
            if len(selected) >= max(2, max_models):
                break
    if len(selected) < 2:
        selected = sorted(model_cols, key=lambda m: maes[m])[:min(2, len(model_cols))]
    
    # static_weight 风格赋权（Ridge）
    pred_val, pred_test, weights, ridge_meta = fit_static_weight_ridge(
        df_val=df_val,
        df_test=df_test,
        selected_models=selected,
        horizon=horizon,
        alpha_candidates=alpha_candidates,
        temporal_decay=drift_decay,
        temporal_decay_meta=drift_decay_meta,
    )
    selected, pred_val, pred_test, weights, ridge_meta = _cleanup_zero_weight_models_and_refit(
        df_val=df_val,
        df_test=df_test,
        selected_models=selected,
        pred_val=pred_val,
        pred_test=pred_test,
        weights=weights,
        ridge_meta=ridge_meta,
        horizon=horizon,
        alpha_candidates=alpha_candidates,
        temporal_decay=drift_decay,
        temporal_decay_meta=drift_decay_meta,
    )
    ridge_meta["selection_meta"] = {
        "max_models": int(max_models),
        "stepwise_alpha": float(stepwise_alpha),
        "stepwise": stepwise_meta,
        "stepwise_min_improve_ratio": float(stepwise_min_improve),
        "high_drift_extended_score_factor": (
            float(KG_HIGH_DRIFT_EXTENDED_SCORE_FACTOR) if drift_level == "high" else 1.0
        ),
        "drift": drift_meta,
        "stability": {
            "by_model": stability_meta,
            "removed_models": stability_removed,
        },
    }

    # fail-safe: 使用 blocked-CV MAE 做保守判据，降低 in-sample 偏乐观风险
    base_cv_maes = {}
    for m in model_cols:
        base_cv_maes[m] = _fit_ridge_and_mae(
            df_val[[m]].values,
            y_val,
            alpha=stepwise_alpha,
            horizon=horizon,
            sample_weight=sample_weight,
        )
    guard_anchor_candidates: List[str] = []
    seasonal_anchor = "seasonal_naive"
    if (
        horizon >= 24
        and
        seasonal_anchor in df_val.columns
        and seasonal_anchor in df_test.columns
        and seasonal_anchor not in model_cols
        and len(model_cols) > 0
    ):
        seasonal_val_mae = float(np.mean(np.abs(df_val[seasonal_anchor].values - y_val)))
        best_model_val_mae = float(min(maes.get(m, np.inf) for m in model_cols))
        if np.isfinite(seasonal_val_mae) and np.isfinite(best_model_val_mae):
            if seasonal_val_mae <= best_model_val_mae * KG_GUARD_NAIVE_RELAX_RATIO:
                base_cv_maes[seasonal_anchor] = _fit_ridge_and_mae(
                    df_val[[seasonal_anchor]].values,
                    y_val,
                    alpha=stepwise_alpha,
                    horizon=horizon,
                    sample_weight=sample_weight,
                )
                guard_anchor_candidates.append(seasonal_anchor)
    best_model = min(base_cv_maes, key=base_cv_maes.get) if base_cv_maes else min(model_cols, key=lambda m: maes[m])
    best_base_cv_mae = float(base_cv_maes.get(best_model, np.inf))
    kg_cv_mae = _fit_ridge_and_mae(
        df_val[selected].values,
        y_val,
        alpha=stepwise_alpha,
        horizon=horizon,
        sample_weight=sample_weight,
    )
    fallback_eps = DRIFT_HIGH_GUARD_FALLBACK_EPS if drift_level == "high" else 0.05
    ridge_meta["guard_config"] = {
        "fallback_eps": float(fallback_eps),
        "temporal_weight_min_ratio": float(KG_TEMPORAL_WEIGHT_MIN_RATIO),
        "best_base_cv_mae": float(best_base_cv_mae) if np.isfinite(best_base_cv_mae) else None,
        "kg_cv_mae": float(kg_cv_mae) if np.isfinite(kg_cv_mae) else None,
        "guard_anchor_candidates": guard_anchor_candidates,
        "guard_anchor_relax_ratio": float(KG_GUARD_NAIVE_RELAX_RATIO),
    }
    if np.isfinite(best_base_cv_mae) and np.isfinite(kg_cv_mae) and kg_cv_mae > best_base_cv_mae * (1.0 + fallback_eps):
        fallback_meta: Dict[str, Any] = {
            "mode": "single_model",
            "best_model": best_model,
            "best_base_cv_mae": float(best_base_cv_mae),
            "kg_cv_mae": float(kg_cv_mae),
        }
        reliable_pair = _find_reliable_small_combo(
            df_val=df_val,
            model_cols=model_cols,
            maes=maes,
            y_val=y_val,
            horizon=horizon,
            alpha=stepwise_alpha,
            sample_weight=sample_weight,
            best_single_cv_mae=best_base_cv_mae,
            fallback_eps=fallback_eps,
        )
        if reliable_pair is not None:
            pair_models = reliable_pair["pair"]
            p_val, p_test, p_weights, p_meta = fit_static_weight_ridge(
                df_val=df_val,
                df_test=df_test,
                selected_models=pair_models,
                horizon=horizon,
                alpha_candidates=alpha_candidates,
                temporal_decay=drift_decay,
                temporal_decay_meta=drift_decay_meta,
            )
            selected = pair_models
            pred_val = p_val
            pred_test = p_test
            weights = p_weights
            ridge_meta.update(p_meta)
            fallback_meta.update({
                "mode": "reliable_small_combo",
                "reliable_pair": reliable_pair,
                "pair_models": pair_models,
            })
            ridge_meta["fallback"] = (
                "kg_worse_than_best_single_cv_but_use_reliable_small_combo:"
                f"{kg_cv_mae:.4f}>{best_base_cv_mae:.4f}*(1+{fallback_eps:.3f})"
            )
        else:
            pred_val = df_val[best_model].values
            pred_test = df_test[best_model].values
            weights = {best_model: 1.0}
            selected = [best_model]
            ridge_meta["fallback"] = (
                f"kg_worse_than_best_single_cv:{kg_cv_mae:.4f}>{best_base_cv_mae:.4f}*(1+{fallback_eps:.3f})"
            )
        ridge_meta["fallback_meta"] = fallback_meta

    # 小样本长步长守卫：多窗口检测，避免单个 tail 切片偶然性。
    if (
        horizon >= PROTOCOL_A_SMALL_SAMPLE_LONG_H_MIN_H
        and len(y_val) < PROTOCOL_A_SMALL_SAMPLE_LONG_H_MAX_VAL_SAMPLES
        and len(model_cols) > 0
    ):
        tail_n = max(
            PROTOCOL_A_SMALL_SAMPLE_LONG_H_MIN_TAIL,
            int(len(y_val) * PROTOCOL_A_SMALL_SAMPLE_LONG_H_TAIL_RATIO),
        )
        tail_n = min(max(tail_n, 1), len(y_val))
        win_sizes = sorted(set([
            tail_n,
            max(PROTOCOL_A_SMALL_SAMPLE_LONG_H_MIN_TAIL, tail_n // 2),
        ]))
        if len(win_sizes) < 2 and len(y_val) > 1:
            alt_w = max(1, tail_n - 1)
            if alt_w not in win_sizes:
                win_sizes.append(alt_w)
                win_sizes = sorted(win_sizes)
        window_details = []
        guard_triggered = False
        best_tail_model = min(model_cols, key=lambda m: maes[m])
        best_tail_mae = float(maes[best_tail_model])
        kg_tail_mae = float(np.mean(np.abs(pred_val - y_val)))
        trigger_msg = None

        for w in win_sizes:
            if w <= 0:
                continue
            w = min(w, len(y_val))
            tail_slice = slice(len(y_val) - w, len(y_val))
            kg_w_mae = float(np.mean(np.abs(pred_val[tail_slice] - y_val[tail_slice])))
            tail_maes = {
                m: float(np.mean(np.abs(df_val[m].values[tail_slice] - y_val[tail_slice])))
                for m in model_cols
            }
            best_w_model = min(tail_maes, key=tail_maes.get)
            best_w_mae = tail_maes[best_w_model]
            degraded = kg_w_mae > best_w_mae * (1.0 + PROTOCOL_A_SMALL_SAMPLE_LONG_H_MAX_TAIL_DEGRADATION)
            window_details.append({
                "window": int(w),
                "kg_tail_mae": kg_w_mae,
                "best_tail_model": best_w_model,
                "best_tail_mae": best_w_mae,
                "degraded": bool(degraded),
            })
            if degraded and not guard_triggered:
                guard_triggered = True
                best_tail_model = best_w_model
                best_tail_mae = best_w_mae
                kg_tail_mae = kg_w_mae
                trigger_msg = (
                    "small_sample_long_horizon_window_guard:"
                    f"{kg_w_mae:.4f}>{best_w_mae:.4f}*(1+{PROTOCOL_A_SMALL_SAMPLE_LONG_H_MAX_TAIL_DEGRADATION:.3f})"
                    f"@window={w}"
                )

        ridge_meta["small_sample_long_horizon_guard"] = {
            "enabled": True,
            "tail_n": int(tail_n),
            "windows": window_details,
            "kg_tail_mae": kg_tail_mae,
            "best_tail_model": best_tail_model,
            "best_tail_mae": best_tail_mae,
            "max_tail_degradation": PROTOCOL_A_SMALL_SAMPLE_LONG_H_MAX_TAIL_DEGRADATION,
        }
        if guard_triggered:
            pred_val = df_val[best_tail_model].values
            pred_test = df_test[best_tail_model].values
            weights = {best_tail_model: 1.0}
            selected = [best_tail_model]
            fallback_tag = trigger_msg or "small_sample_long_horizon_window_guard"
            prev_fallback = ridge_meta.get("fallback")
            ridge_meta["fallback"] = f"{prev_fallback};{fallback_tag}" if prev_fallback else fallback_tag

    reasoning_meta = {
        "used_reasoning_path": False,
        "reasoning_disabled": True,
        "reasoning_mode": "off",
    }

    naming = get_strategy_naming("kg_protocol_a", default_category="kg_core")

    result = {
        "val": {
            **_merge_eval_metrics(pred_val, y_val),
            "strategy_id": naming["strategy_id"],
            "strategy_display_name": naming["display_name"],
            "method_family": naming["method_family"],
            "method_route": naming["method_route"],
            "core_route": naming["core_route"],
            "selected_models": selected,
            "weights": weights,
            "model_scores": model_scores,
            "error_correlations": {f"{k[0]}_vs_{k[1]}": v for k, v in error_corrs.items()},
            "conflicts_detected": [f"{c[0]}_vs_{c[1]}" for c in check_conflict(mg, model_cols)],
            "weight_meta": ridge_meta,
            **reasoning_meta,
        },
        "test": {
            **_merge_eval_metrics(pred_test, y_test),
            "strategy_id": naming["strategy_id"],
            "strategy_display_name": naming["display_name"],
            "method_family": naming["method_family"],
            "method_route": naming["method_route"],
            "core_route": naming["core_route"],
            "selected_models": selected,
            "weights": weights,
            "weight_meta": ridge_meta,
            **reasoning_meta,
        },
        "protocol": "A_pred_only"
    }
    if return_predictions:
        # 运行时预测：默认不产出；调用方取用后应立即摘除，勿写入 trace/实验 JSON。
        result[RUNTIME_PREDICTIONS_KEY] = {
            "val": np.asarray(pred_val, dtype=float),
            "test": np.asarray(pred_test, dtype=float),
        }
    return result
