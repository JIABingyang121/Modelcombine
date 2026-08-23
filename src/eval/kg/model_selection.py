"""KG model selection and ridge fitting utilities."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.eval.combination_utils import fit_ridge_robust
from src.graph.model_graph import ModelGraph
from src.utils.blocked_cv import blocked_cv_select_alpha, blocked_cv_splits

from .config import (
    KG_HIGH_DRIFT_EXTENDED_SCORE_FACTOR,
    KG_RIDGE_TEMPORAL_DECAY,
    KG_ZERO_WEIGHT_CLEANUP_THRESHOLD,
    PROTOCOL_B_RELATION_STRENGTH_NEUTRAL,
    PROTOCOL_B_RELATION_STRENGTH_WEIGHT,
    PROTOCOL_STEPWISE_TIE_RELATIVE_TOLERANCE,
    PROTOCOL_STEPWISE_MIN_IMPROVEMENT,
)
from .conflict import _lookup_pair_corr, check_conflict, compute_error_correlations
from .data_io import _resolve_cv_config
from .drift import _compute_temporal_weights
from .stability import _adaptive_stepwise_min_improve
from .config import _extended_pool_penalty_models

def _select_stepwise_alpha(
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: Optional[np.ndarray],
    horizon: int,
    alphas: Optional[List[float]] = None,
    default_alpha: float = 1.0,
) -> float:
    """为 stepwise 阶段选取与最终 Ridge 更一致的 alpha。"""
    try:
        n_folds, min_train, gap = _resolve_cv_config(len(y_val), horizon)
        best_alpha, cv_mae = blocked_cv_select_alpha(
            X_val,
            y_val,
            alphas=alphas,
            n_folds=n_folds,
            min_train=min_train,
            positive=True,
            fit_intercept=False,
            sample_weight=sample_weight,
            gap=gap,
        )
        if np.isfinite(cv_mae):
            return float(best_alpha)
    except Exception:
        pass
    return float(default_alpha)

def fit_static_weight_ridge(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    selected_models: List[str],
    horizon: int,
    alpha_candidates: Optional[List[float]] = None,
    temporal_decay: Optional[float] = None,
    temporal_decay_meta: Optional[Dict[str, Any]] = None,
    temporal_min_weight_ratio: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict]:
    """
    使用 static_weight 风格的 Ridge 赋权（positive=True, fit_intercept=False）。
    失败时回退到 MAE 倒数权重。
    """
    y_val = df_val["y"].values
    X_val = df_val[selected_models].values
    X_test = df_test[selected_models].values

    meta = {
        "weight_method": "ridge_static_weight",
        "fallback": None,
    }
    effective_decay = float(KG_RIDGE_TEMPORAL_DECAY if temporal_decay is None else temporal_decay)
    effective_decay = max(effective_decay, 0.0)
    sample_weight = _compute_temporal_weights(
        len(y_val),
        effective_decay,
        min_ratio=temporal_min_weight_ratio,
    )
    effective_min_ratio = (
        float(temporal_min_weight_ratio)
        if temporal_min_weight_ratio is not None
        else None
    )
    if sample_weight is not None:
        # 时间衰减：近期样本权重更高，缓解 val->test 轻度分布偏移。
        meta["temporal_weighting"] = {
            "enabled": True,
            "decay": float(effective_decay),
            "min_ratio": effective_min_ratio,
            "min_w": float(np.min(sample_weight)),
            "max_w": float(np.max(sample_weight)),
        }
    else:
        meta["temporal_weighting"] = {
            "enabled": False,
            "decay": float(effective_decay),
            "min_ratio": effective_min_ratio,
        }
    if isinstance(temporal_decay_meta, dict) and temporal_decay_meta:
        meta["temporal_weighting"]["drift_adaptive"] = temporal_decay_meta

    try:
        n_folds, min_train, gap = _resolve_cv_config(len(y_val), horizon)
        best_alpha, cv_mae = blocked_cv_select_alpha(
            X_val, y_val,
            alphas=alpha_candidates,
            n_folds=n_folds, min_train=min_train,
            positive=True, fit_intercept=False,
            sample_weight=sample_weight,
            gap=gap,
        )
        if not np.isfinite(cv_mae):
            best_alpha = 10.0

        reg, solver_meta = fit_ridge_robust(
            X_val, y_val,
            alpha=float(best_alpha),
            positive=True,
            fit_intercept=False,
            sample_weight=sample_weight,
        )
        pred_val = reg.predict(X_val)
        pred_test = reg.predict(X_test)
        weights = dict(zip(selected_models, reg.coef_.tolist()))
        meta.update({
            "best_alpha": float(best_alpha),
            "cv_mae": float(cv_mae) if np.isfinite(cv_mae) else None,
            "solver_meta": solver_meta,
        })
        return pred_val, pred_test, weights, meta
    except Exception as e:
        maes = [np.mean(np.abs(df_val[m].values - y_val)) for m in selected_models]
        inv = np.array([1.0 / (mae + 1e-6) for mae in maes])
        w = inv / (inv.sum() + 1e-8)
        pred_val = X_val @ w
        pred_test = X_test @ w
        weights = dict(zip(selected_models, w.tolist()))
        meta["fallback"] = f"inverse_mae_due_to_ridge_error:{e}"
        return pred_val, pred_test, weights, meta

def _cleanup_zero_weight_models_and_refit(
    *,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    selected_models: List[str],
    pred_val: np.ndarray,
    pred_test: np.ndarray,
    weights: Dict[str, float],
    ridge_meta: Dict[str, Any],
    horizon: int,
    alpha_candidates: Optional[List[float]],
    temporal_decay: Optional[float],
    temporal_decay_meta: Optional[Dict[str, Any]],
    temporal_min_weight_ratio: Optional[float] = None,
    zero_threshold: float = KG_ZERO_WEIGHT_CLEANUP_THRESHOLD,
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, float], Dict[str, Any]]:
    """
    清理近零权重模型并重拟合，减少噪声维度对高漂移场景的干扰。
    仅在“删除后仍至少 2 个模型”时触发，避免退化成单模型偶然路径。
    """
    selected = list(selected_models)
    nonzero_selected = [
        m for m in selected
        if abs(float(weights.get(m, 0.0))) > float(zero_threshold)
    ]
    cleanup_meta: Dict[str, Any] = {
        "applied": False,
        "threshold": float(zero_threshold),
        "before": selected,
        "after": nonzero_selected,
    }
    if 1 < len(nonzero_selected) < len(selected):
        pred_val, pred_test, weights, refit_meta = fit_static_weight_ridge(
            df_val=df_val,
            df_test=df_test,
            selected_models=nonzero_selected,
            horizon=horizon,
            alpha_candidates=alpha_candidates,
            temporal_decay=temporal_decay,
            temporal_decay_meta=temporal_decay_meta,
            temporal_min_weight_ratio=temporal_min_weight_ratio,
        )
        selected = nonzero_selected
        cleanup_meta["applied"] = True
        cleanup_meta["refit"] = True
        cleanup_meta["refit_meta"] = refit_meta
        ridge_meta.update(refit_meta)
    ridge_meta["zero_weight_cleanup"] = cleanup_meta
    return selected, pred_val, pred_test, weights, ridge_meta

def _find_reliable_small_combo(
    *,
    df_val: pd.DataFrame,
    model_cols: List[str],
    maes: Dict[str, float],
    y_val: np.ndarray,
    horizon: int,
    alpha: float,
    sample_weight: Optional[np.ndarray],
    best_single_cv_mae: float,
    fallback_eps: float,
) -> Optional[Dict[str, Any]]:
    """
    在 fallback 前尝试寻找“可靠二模型组合”，避免直接回退单模型损失多样性。
    策略：
    - 候选只取前若干低 MAE 模型；
    - 仅考虑低相关对（|corr| <= 0.85）；
    - 要求 pair_cv_mae 明显不差于 best_single_cv_mae。
    """
    if len(model_cols) < 2:
        return None
    top = sorted(model_cols, key=lambda m: maes.get(m, float("inf")))[:min(6, len(model_cols))]
    if len(top) < 2:
        return None
    corr_map = compute_error_correlations(df_val, top)
    best_pair: Optional[List[str]] = None
    best_pair_cv = float("inf")
    best_pair_corr = None
    for i, m1 in enumerate(top):
        for m2 in top[i + 1:]:
            pair_corr = abs(_lookup_pair_corr(corr_map, m1, m2, default=1.0))
            if pair_corr > 0.85:
                continue
            pair_cv = _fit_ridge_and_mae(
                df_val[[m1, m2]].values,
                y_val,
                alpha=alpha,
                horizon=horizon,
                sample_weight=sample_weight,
            )
            if np.isfinite(pair_cv) and pair_cv < best_pair_cv:
                best_pair = [m1, m2]
                best_pair_cv = float(pair_cv)
                best_pair_corr = float(pair_corr)
    if best_pair is None:
        return None
    # 比 best single 略松但仍保守：允许在 fallback_eps 的一半以内。
    if not np.isfinite(best_single_cv_mae):
        return None
    accept_th = best_single_cv_mae * (1.0 + max(fallback_eps * 0.5, 0.005))
    if best_pair_cv > accept_th:
        return None
    return {
        "pair": best_pair,
        "pair_cv_mae": best_pair_cv,
        "pair_corr": best_pair_corr,
        "accept_threshold": float(accept_th),
    }

def _fit_ridge_and_mae(
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float,
    horizon: int,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    n_folds, min_train, gap = _resolve_cv_config(len(y_val), horizon)
    splits = blocked_cv_splits(len(y_val), n_folds=n_folds, min_train=min_train, gap=gap)
    if not splits:
        reg, _ = fit_ridge_robust(
            X_val,
            y_val,
            alpha=float(alpha),
            positive=True,
            fit_intercept=False,
            sample_weight=sample_weight,
        )
        pred = reg.predict(X_val)
        return float(np.mean(np.abs(pred - y_val)))

    fold_scores: List[float] = []
    for train_idx, val_idx in splits:
        if len(val_idx) == 0:
            continue
        sw = sample_weight[train_idx] if sample_weight is not None else None
        reg, _ = fit_ridge_robust(
            X_val[train_idx],
            y_val[train_idx],
            alpha=float(alpha),
            positive=True,
            fit_intercept=False,
            sample_weight=sw,
        )
        pred = reg.predict(X_val[val_idx])
        fold_scores.append(float(np.mean(np.abs(pred - y_val[val_idx]))))
    if not fold_scores:
        return float("inf")
    return float(np.mean(fold_scores))

def _is_better_beyond_tolerance(
    trial_mae: float,
    incumbent_mae: float,
    tie_rel_tol: float,
) -> bool:
    """trial 是否**确实**优于在位者，而不是落在数值噪声里。

    候选的 CV MAE 来自 lbfgs 迭代解（tol=1e-4），差异小于 ``tie_rel_tol`` 时
    两者实质并列，此时保留在位者——在位者由调用方给定的候选顺序决定，
    因此结果可复现，且 Protocol B 的 base_score 排序仍然说了算。
    """
    if not np.isfinite(trial_mae):
        return False
    if not np.isfinite(incumbent_mae):
        return True
    return trial_mae < incumbent_mae - abs(incumbent_mae) * tie_rel_tol


def _is_tied_within_tolerance(
    trial_mae: float,
    incumbent_mae: float,
    tie_rel_tol: float,
) -> bool:
    """两项有限 MAE 是否落在 stepwise 的相对并列容差内。"""
    if not (np.isfinite(trial_mae) and np.isfinite(incumbent_mae)):
        return False
    return abs(trial_mae - incumbent_mae) <= abs(incumbent_mae) * tie_rel_tol


def forward_stepwise_select(
    df_val: pd.DataFrame,
    candidate_models: List[str],
    base_maes: Dict[str, float],
    max_models: int,
    horizon: int,
    min_improve_ratio: float = 0.005,
    sample_weight: Optional[np.ndarray] = None,
    conflict_checker: Optional[Callable[[List[str]], bool]] = None,
    alpha: float = 1.0,
    respect_candidate_order: bool = False,
    tie_rel_tol: float = PROTOCOL_STEPWISE_TIE_RELATIVE_TOLERANCE,
) -> Tuple[List[str], Dict[str, float]]:
    """
    前向逐步选择：每一步仅在组合 MAE 相对改善达到阈值时才扩展。
    """
    if not candidate_models:
        return [], {"stopped_early": True, "reason": "no_candidates"}
    if len(candidate_models) == 1:
        return candidate_models.copy(), {"stopped_early": True, "reason": "single_candidate"}

    # respect_candidate_order 仅由 Protocol B 传真：它已按 (-base_score, mae)
    # 排好序，base_score 含关系强度项（§11#7）；此处若无条件按 MAE 重排，
    # 会把调用方算好的排序整个丢弃，关系强度永远影响不到最终选择。
    # 默认仍按 MAE 排序，以保持 Protocol A 与既有 guard 基线行为不变。
    if respect_candidate_order:
        ordered = list(candidate_models)
    else:
        ordered = sorted(candidate_models, key=lambda m: base_maes.get(m, float("inf")))
    best_single = ordered[0]
    selected = [best_single]
    best_mae = _fit_ridge_and_mae(
        df_val[[best_single]].values,
        df_val["y"].values,
        alpha=alpha,
        horizon=horizon,
        sample_weight=sample_weight,
    )
    trace = [{"step": 0, "selected": best_single, "mae": best_mae}]
    y_val = df_val["y"].values

    candidate_failures: Dict[str, str] = {}
    candidate_evaluations: List[Dict[str, Any]] = []
    while len(selected) < max_models:
        best_candidate = None
        best_candidate_mae = best_mae
        evaluations: List[Dict[str, Any]] = []
        for m in ordered:
            if m in selected:
                continue
            trial = selected + [m]
            if conflict_checker is not None and conflict_checker(trial):
                continue
            try:
                trial_mae = _fit_ridge_and_mae(
                    df_val[trial].values,
                    y_val,
                    alpha=alpha,
                    horizon=horizon,
                    sample_weight=sample_weight,
                )
            except Exception as exc:
                # 静默 continue 会让间歇性求解失败无法定位：同一份数据两次运行
                # 选出不同组合，却查不到任何线索。失败必须留痕。
                error = f"{type(exc).__name__}: {exc}"
                candidate_failures[m] = error
                evaluations.append({"model": m, "decision": "failure", "error": error})
                continue
            comparison_mae = best_candidate_mae
            if _is_better_beyond_tolerance(trial_mae, best_candidate_mae, tie_rel_tol):
                best_candidate_mae = trial_mae
                best_candidate = m
                decision = "new_best"
            elif _is_tied_within_tolerance(trial_mae, comparison_mae, tie_rel_tol):
                decision = "tie_kept"
            else:
                decision = "not_better"
            evaluations.append({
                "model": m,
                "trial_mae": float(trial_mae),
                "comparison_mae": float(comparison_mae),
                "decision": decision,
            })

        candidate_evaluations.append({
            "step": len(selected),
            "incumbent_models": list(selected),
            "incumbent_mae": float(best_mae),
            "evaluations": evaluations,
        })

        if best_candidate is None:
            trace.append({"step": len(selected), "selected": None, "mae": best_mae, "stop": "no_valid_candidate"})
            break

        rel_improve = (best_mae - best_candidate_mae) / max(best_mae, 1e-10)
        if rel_improve < min_improve_ratio:
            trace.append({
                "step": len(selected),
                "selected": best_candidate,
                "mae": best_candidate_mae,
                "stop": "improvement_too_small",
                "rel_improve": rel_improve,
            })
            break

        selected.append(best_candidate)
        best_mae = best_candidate_mae
        trace.append({"step": len(selected) - 1, "selected": best_candidate, "mae": best_mae, "rel_improve": rel_improve})

    return selected, {
        "stopped_early": len(selected) < max_models,
        "final_mae": best_mae,
        "min_improve_ratio": float(min_improve_ratio),
        "tie_rel_tol": float(tie_rel_tol),
        "candidate_failures": candidate_failures,
        "candidate_evaluations": candidate_evaluations,
        "trace": trace,
    }

def select_models_protocol_b(
    mg: ModelGraph,
    model_cols: List[str],
    maes: Dict[str, float],
    error_corrs: Dict[Tuple[str, str], float],
    feat_model_corrs: Dict[Tuple[str, str], float],
    horizon: int,
    df_val: Optional[pd.DataFrame] = None,
    max_models: int = 5,
    feature_bonus_weight: float = 0.6,
    corr_penalty_weight: float = 0.35,
    feature_diversity_weight: float = 0.25,
    beam_width: int = 3,
    drift_level: str = "low",
    relation_strength_weight: float = PROTOCOL_B_RELATION_STRENGTH_WEIGHT,
) -> Tuple[List[str], Dict[str, float], Dict[str, float], Dict]:
    """
    Protocol B 专用模型选择：
    - 基础性能 + 互补边奖励 + 特征关联奖励
    - 与已选模型高相关时惩罚（鼓励误差多样性）
    """
    feature_bonus = {m: 0.0 for m in model_cols}
    feat_cnt = {m: 0 for m in model_cols}
    for (_, m), corr in feat_model_corrs.items():
        if m in feature_bonus:
            feature_bonus[m] += float(corr)
            feat_cnt[m] += 1
    for m in model_cols:
        if feat_cnt[m] > 0:
            feature_bonus[m] = feature_bonus[m] / feat_cnt[m]

    base_raw = {m: 1.0 / (maes[m] + 1e-6) for m in model_cols}
    base_max = max(base_raw.values()) if base_raw else 1.0
    base_scores = {}
    high_drift_penalty_models = _extended_pool_penalty_models() if drift_level == "high" else set()

    # 关系强度（§11#7）：Hawkes/反馈写在 scenario->model 的 recommended_for 边上，
    # 此前没有任何决策路径读它。这里把它作为逐模型加分项计入候选评分。
    # 取 dynamic_strength 优先、否则 weight；无边时用中性值，使该项恰好为 0。
    relation_strength: Dict[str, float] = {}
    for m in model_cols:
        values = []
        if mg.G.has_node(m):
            for src, _tgt, data in mg.G.in_edges(m, data=True):
                if data.get("edge_type") != "recommended_for":
                    continue
                raw = data.get("dynamic_strength", data.get("weight"))
                if raw is None:
                    continue
                val = float(raw)
                if np.isfinite(val):
                    values.append(val)
        relation_strength[m] = (
            float(np.mean(values)) if values else PROTOCOL_B_RELATION_STRENGTH_NEUTRAL
        )
    relation_contribution = {
        m: relation_strength_weight * (relation_strength[m] - PROTOCOL_B_RELATION_STRENGTH_NEUTRAL)
        for m in model_cols
    }

    for m in model_cols:
        base_norm = base_raw[m] / (base_max + 1e-10)
        comp_bonus = 0.0
        for neighbor in mg.neighbors_by_type(m, "complementary"):
            if neighbor in model_cols:
                comp_bonus += float(mg.G[m][neighbor].get("weight", 0.5))
        score = (
            base_norm
            + comp_bonus
            + feature_bonus_weight * feature_bonus[m]
            + relation_contribution[m]
        )
        if m in high_drift_penalty_models:
            score *= KG_HIGH_DRIFT_EXTENDED_SCORE_FACTOR
        base_scores[m] = score

    # 基于 Feature->Model 相关性的“特征画像”差异，用于鼓励在不同特征区域互补的模型组合。
    feature_names = sorted({feat for feat, _ in feat_model_corrs.keys()})
    feature_matrix = {}
    if feature_names:
        for m in model_cols:
            vec = np.array(
                [float(feat_model_corrs.get((feat, m), 0.0)) for feat in feature_names],
                dtype=float,
            )
            feature_matrix[m] = vec

    def _feature_diversity(combo: List[str]) -> float:
        if len(combo) <= 1 or not feature_matrix:
            return 0.0
        vals = []
        for i, m1 in enumerate(combo):
            v1 = feature_matrix.get(m1)
            if v1 is None:
                continue
            n1 = float(np.linalg.norm(v1))
            for m2 in combo[i + 1:]:
                v2 = feature_matrix.get(m2)
                if v2 is None:
                    continue
                n2 = float(np.linalg.norm(v2))
                if n1 < 1e-12 or n2 < 1e-12:
                    continue
                cos = float(np.dot(v1, v2) / (n1 * n2 + 1e-12))
                vals.append(max(0.0, 1.0 - cos))
        return float(np.mean(vals)) if vals else 0.0

    def _score_combo(combo: List[str]) -> float:
        if not combo:
            return -np.inf
        base = float(np.sum([base_scores[m] for m in combo]))
        if len(combo) <= 1:
            corr_pen = 0.0
        else:
            pair_corrs = []
            for i, m1 in enumerate(combo):
                for m2 in combo[i + 1:]:
                    pair_corrs.append(abs(_lookup_pair_corr(error_corrs, m1, m2, default=0.0)))
            corr_pen = float(np.mean(pair_corrs)) if pair_corrs else 0.0
        feature_div = _feature_diversity(combo)
        # 组合越大，相关性惩罚影响越明显；同时加入特征多样性奖励。
        return base - corr_penalty_weight * corr_pen * len(combo) + feature_diversity_weight * feature_div

    max_len = min(max_models, len(model_cols))
    beam: List[List[str]] = [[]]
    scored: Dict[Tuple[str, ...], float] = {}
    for _ in range(max_len):
        candidates: Dict[Tuple[str, ...], float] = {}
        for combo in beam:
            used = set(combo)
            for m in model_cols:
                if m in used:
                    continue
                new_combo = combo + [m]
                if check_conflict(mg, new_combo):
                    continue
                key = tuple(new_combo)
                score = _score_combo(new_combo)
                prev = candidates.get(key)
                if prev is None or score > prev:
                    candidates[key] = score
                scored[key] = score
        if not candidates:
            break
        top = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[:max(1, beam_width)]
        beam = [list(k) for k, _ in top]

    valid_scored = [(list(k), s) for k, s in scored.items() if len(k) >= 2]
    if valid_scored:
        selected = max(valid_scored, key=lambda kv: kv[1])[0]
    else:
        selected = []

    # 在图打分基础上再做一次前向逐步选择，直接优化验证集 MAE。
    stepwise_meta = {}
    stepwise_alpha = 1.0
    stepwise_min_improve = PROTOCOL_STEPWISE_MIN_IMPROVEMENT
    if df_val is not None and len(selected) >= 1:
        stepwise_min_improve = _adaptive_stepwise_min_improve(
            drift_level=drift_level,
            n_candidates=len(model_cols),
            n_val=len(df_val),
        )
        sample_weight = _compute_temporal_weights(len(df_val), KG_RIDGE_TEMPORAL_DECAY)
        stepwise_alpha = _select_stepwise_alpha(
            X_val=df_val[model_cols].values,
            y_val=df_val["y"].values,
            sample_weight=sample_weight,
            horizon=horizon,
            default_alpha=1.0,
        )
        candidate_order = sorted(
            model_cols,
            key=lambda m: (-(base_scores.get(m, 0.0)), maes.get(m, float("inf"))),
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
            # 仅 Protocol B 使用含关系强度的候选排序；Protocol A 保持按 MAE 排序
            respect_candidate_order=True,
        )
        if len(stepwise_selected) >= 2:
            selected = stepwise_selected

    if len(selected) < 2:
        selected = sorted(model_cols, key=lambda m: maes[m])[:min(3, len(model_cols))]
    meta = {
        "feature_diversity_weight": float(feature_diversity_weight),
        "drift_level": drift_level,
        "stepwise_alpha": float(stepwise_alpha),
        "stepwise_min_improve_ratio": float(stepwise_min_improve),
        "stepwise_meta": stepwise_meta,
        # 关系强度评分项（§11#7）：留痕以便回答"为什么这条候选排在前面"。
        # by_model 为读到的强度，contribution 为已乘权重、已减中性的实际加分。
        "relation_strength": {
            "weight": float(relation_strength_weight),
            "neutral": float(PROTOCOL_B_RELATION_STRENGTH_NEUTRAL),
            "by_model": {m: float(relation_strength[m]) for m in model_cols},
            "contribution": {m: float(relation_contribution[m]) for m in model_cols},
            "edges_found": sorted(
                m for m in model_cols
                if relation_strength[m] != PROTOCOL_B_RELATION_STRENGTH_NEUTRAL
            ),
        },
    }
    return selected, base_scores, feature_bonus, meta

def _fallback_selection(
    mg: ModelGraph,
    model_cols: List[str],
    maes: Dict[str, float],
    max_models: int = 5,
) -> List[str]:
    """性能+互补性回退选择"""
    model_scores = {}
    for m in model_cols:
        base_score = 1.0 / (maes[m] + 1e-6)
        bonus = 0.0
        for neighbor in mg.neighbors_by_type(m, "complementary"):
            if neighbor in model_cols:
                bonus += mg.G[m][neighbor].get("weight", 0.5)
        model_scores[m] = base_score + bonus
    
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    
    selected = []
    for m, _ in sorted_models:
        if not check_conflict(mg, selected + [m]):
            selected.append(m)
        if len(selected) >= min(max_models, len(model_cols)):
            break
    
    return selected if len(selected) >= 2 else model_cols[:min(max(2, max_models), len(model_cols))]
