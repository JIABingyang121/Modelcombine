"""Protocol B implementation (prediction + raw-feature KG)."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.eval.combination_utils import evaluate, fit_ridge_robust
from src.eval.metrics import evaluate as evaluate_with_tail
from src.eval.strategy_naming import get_strategy_naming
from src.graph.model_graph import ModelGraph
from src.utils.blocked_cv import blocked_cv_select_alpha, blocked_cv_splits

from .config import *
from .conflict import (
    _lookup_pair_corr,
    compute_feature_model_correlation_safe,
)
from .data_io import _align_raw_to_pred, _blocked_cv_mae_from_pred, _resolve_cv_config
from .drift import (
    _compute_temporal_weights,
    _resolve_drift_aware_temporal_decay,
    _resolve_protocol_b_global_min_improve,
)
from .model_selection import (
    _cleanup_zero_weight_models_and_refit,
    _compute_feature_bonus_map,
    fit_static_weight_ridge,
    pair_eligibility_from_cleanup,
    select_models_protocol_b,
)
from .reasoning_evidence import build_reasoning_evidence
from .relation_feedback import compute_relation_feedback_evidence
from .feedback import KGFeedbackStore
from .protocol_a import kg_combination_pred_only
from .stability import _dedup_and_stability_filter, adaptive_max_models

def _merge_eval_metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    base = evaluate(pred, y)
    try:
        rich = evaluate_with_tail(y, pred)
    except Exception:
        rich = {}
    if isinstance(rich, dict):
        base.update(rich)
    return base


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) <= 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

def _interaction_oof_cv_metrics(
    *,
    X_inter_val: np.ndarray,
    residual_target: np.ndarray,
    base_pred: np.ndarray,
    y_true: np.ndarray,
    alpha: float,
    horizon: int,
    sample_weight: Optional[np.ndarray],
) -> Optional[Dict[str, float]]:
    """Leak-free blocked-CV OOF metrics for interaction residual branch."""
    n_folds, min_train, gap = _resolve_cv_config(len(y_true), horizon)
    splits = blocked_cv_splits(len(y_true), n_folds=n_folds, min_train=min_train, gap=gap)
    if not splits:
        return None

    oof_pred = np.full(len(y_true), np.nan, dtype=float)
    for train_idx, val_idx in splits:
        if len(train_idx) < 2 or len(val_idx) == 0:
            continue
        sw_train = sample_weight[train_idx] if sample_weight is not None else None
        try:
            reg_fold, _ = fit_ridge_robust(
                X_inter_val[train_idx],
                residual_target[train_idx],
                alpha=float(alpha),
                positive=False,
                fit_intercept=False,
                sample_weight=sw_train,
            )
            oof_pred[val_idx] = base_pred[val_idx] + reg_fold.predict(X_inter_val[val_idx])
        except Exception:
            continue

    mask = np.isfinite(oof_pred)
    if int(mask.sum()) < max(10, len(y_true) // 20):
        return None

    mae_raw = float(np.mean(np.abs(base_pred[mask] - y_true[mask])))
    mae_inter = float(np.mean(np.abs(oof_pred[mask] - y_true[mask])))
    return {
        "mae_raw_oof": mae_raw,
        "mae_inter_oof": mae_inter,
        "oof_coverage": float(np.mean(mask.astype(float))),
    }

def _unified_oof_mae(
    *,
    df_val: pd.DataFrame,
    models: Sequence[str],
    horizon: int,
    alpha: float,
    sample_weight: Optional[np.ndarray],
    interaction_features: Optional[np.ndarray] = None,
    interaction_alpha: Optional[float] = None,
) -> Tuple[float, int]:
    """在同一组 blocked-CV 折上计算一个方案的折外 MAE（Task 8.3 Task 11）。

    每一折都重新拟合线性组合（正约束 Ridge、无截距）；给出 interaction 设计矩阵
    时，在同一折内用该折的残差重新拟合交互残差模型再叠加。A/B 两侧调用同一函数、
    同样的折配置、同样的 alpha 与样本权重，返回 (折外 MAE, 覆盖样本数)，
    覆盖数用于确认两侧评价范围一致。

    只读取 ``df_val``，签名不含任何 test 帧。拟合失败直接向上抛：该判据只在
    ``len(df_val) >= 500`` 时可达，此时折必然存在且非退化，任何异常都是真问题。
    """
    y = np.asarray(df_val["y"].values, dtype=float)
    X = np.asarray(df_val[list(models)].values, dtype=float)
    n_folds, min_train, gap = _resolve_cv_config(len(y), horizon)
    splits = blocked_cv_splits(len(y), n_folds=n_folds, min_train=min_train, gap=gap)

    oof = np.full(len(y), np.nan, dtype=float)
    for train_idx, val_idx in splits:
        sw = sample_weight[train_idx] if sample_weight is not None else None
        reg, _ = fit_ridge_robust(
            X[train_idx], y[train_idx],
            alpha=float(alpha), positive=True, fit_intercept=False,
            sample_weight=sw,
        )
        pred_va = reg.predict(X[val_idx])
        if interaction_features is not None:
            residual = y[train_idx] - reg.predict(X[train_idx])
            reg_i, _ = fit_ridge_robust(
                interaction_features[train_idx], residual,
                alpha=float(interaction_alpha),
                positive=False, fit_intercept=False,
                sample_weight=sw,
            )
            pred_va = pred_va + reg_i.predict(interaction_features[val_idx])
        oof[val_idx] = pred_va

    mask = np.isfinite(oof)
    return float(np.mean(np.abs(oof[mask] - y[mask]))), int(mask.sum())


def kg_combination_with_features(df_val: pd.DataFrame, df_test: pd.DataFrame,
                                  df_raw_val: Optional[pd.DataFrame],
                                  df_raw_test: Optional[pd.DataFrame],
                                  model_cols: List[str], horizon: int,
                                  corr_threshold: float = 0.5,
                                  dataset_name: Optional[str] = None,
                                  base_model_cols: Optional[List[str]] = None,
                                  feedback_store: Optional[KGFeedbackStore] = None,
                                  return_predictions: bool = False,
                                  relation_graph: Optional[ModelGraph] = None,
                                  relation_scenario_id: Optional[str] = None,
                                  _fixed_selected_models: Optional[List[str]] = None,
                                  _skip_final_guard: bool = False) -> Dict:
    """
    KG 组合 - 使用预测+原始特征

    步骤:
    1. Protocol A 的所有步骤
    2. 额外: Feature -> Model 边（特征-误差相关性）
    3. 场景签名 + infer_optimal_path_by_reasoning
    """
    
    def _attach_predictions(result: Dict, pred_val_arr, pred_test_arr) -> Dict:
        """按需挂载运行时预测。默认关闭时返回结构与体积完全不变。"""
        if return_predictions:
            result[RUNTIME_PREDICTIONS_KEY] = {
                "val": np.asarray(pred_val_arr, dtype=float),
                "test": np.asarray(pred_test_arr, dtype=float),
            }
        return result

    # 诊断私有控制（Task 8.3 Task 4）：跳过 guard 必须有固定候选，否则没有意义。
    if _skip_final_guard and _fixed_selected_models is None:
        raise ValueError("_skip_final_guard requires _fixed_selected_models (diagnostic fixed pair)")

    # Protocol A 作为稳定参考与回退候选（用于 Protocol B 保护机制）
    # 回退分支要交出精确预测，A 侧也需同步产出。
    protocol_a_reference = kg_combination_pred_only(
        df_val, df_test, model_cols, horizon, corr_threshold, dataset_name=dataset_name,
        return_predictions=return_predictions,
    )
    val_mae_a = ((protocol_a_reference.get("val") or {}).get("mae"))
    test_mae_a = ((protocol_a_reference.get("test") or {}).get("mae"))
    if base_model_cols is None:
        base_model_cols = []

    # 如果没有原始特征，回退到 Protocol A
    if df_raw_val is None or len(df_raw_val) == 0 or df_raw_test is None or len(df_raw_test) == 0:
        result = protocol_a_reference
        result["protocol"] = "B_fallback_to_A_no_raw"
        return result
    
    y_val = df_val["y"].values
    y_test = df_test["y"].values
    
    # 计算 MAE
    maes = {}
    for m in model_cols:
        maes[m] = _safe_mae(y_val, np.asarray(df_val[m].values, dtype=float))
    base_model_candidates = [m for m in (base_model_cols or []) if m in model_cols]
    best_base_model = (
        min(base_model_candidates, key=lambda m: maes.get(m, float("inf")))
        if base_model_candidates
        else None
    )
    val_mae_best_base = (
        _safe_mae(y_val, np.asarray(df_val[best_base_model].values, dtype=float))
        if best_base_model is not None
        else None
    )
    test_mae_best_base = (
        _safe_mae(y_test, np.asarray(df_test[best_base_model].values, dtype=float))
        if best_base_model is not None
        else None
    )

    drift_level = "low"
    drift_median_psi: Optional[float] = None
    try:
        drift_payload = (
            (protocol_a_reference.get("val", {}) or {})
            .get("weight_meta", {})
            .get("selection_meta", {})
            .get("drift", {})
        )
        drift_level = drift_payload.get("drift_level", "low")
        drift_median_psi = drift_payload.get("median_psi")
    except Exception:
        drift_level = "low"
        drift_median_psi = None
    drift_level_raw = str(drift_level)
    drift_level_source = "protocol_a"
    drift_median_psi_float: Optional[float] = None
    if drift_median_psi is not None:
        try:
            drift_median_psi_float = float(drift_median_psi)
        except Exception:
            drift_median_psi_float = None
    # PSI deadband stabilization: only mark as high when psi clearly exceeds upper edge.
    if drift_median_psi_float is not None and np.isfinite(drift_median_psi_float):
        drift_level_source = "psi_deadband"
        if drift_median_psi_float >= PROTOCOL_B_PSI_DEADBAND_HIGH:
            drift_level = "high"
        elif drift_median_psi_float <= PROTOCOL_B_PSI_DEADBAND_LOW:
            drift_level = "low"
        else:
            drift_level = "medium"

    filter_ctx = _dedup_and_stability_filter(
        df_val=df_val,
        df_test=df_test,
        model_cols=model_cols,
        maes=maes,
        horizon=horizon,
        drift_level_override=drift_level,
        print_suffix="(B)",
    )
    model_cols = filter_ctx["model_cols"]
    stability_meta = filter_ctx["stability_meta"]
    stability_removed = filter_ctx["stability_removed"]
    error_corrs = filter_ctx["error_corrs"]
    
    # 构建图
    mg = ModelGraph()
    
    for m in model_cols:
        mg.add_model_node(m, {"mae": maes[m]})

    # 注入**当前场景**的关系强度边（§11#7 生产接线）。
    # 只复制 relation_scenario_id 指向的 recommended_for 边：若把所有场景的边
    # 都读进来取平均，无关场景会改变本次决策。未提供图或场景时不复制任何边，
    # 此时评分中的关系项恒为中性、行为与接入前一致。
    relation_edges_injected = 0
    if relation_graph is not None and relation_scenario_id:
        src_graph = relation_graph.G
        if src_graph.has_node(relation_scenario_id):
            for _src, tgt, data in src_graph.out_edges(relation_scenario_id, data=True):
                if data.get("edge_type") != "recommended_for" or tgt not in model_cols:
                    continue
                strength = data.get("dynamic_strength", data.get("weight"))
                if strength is None:
                    continue
                mg.G.add_edge(
                    relation_scenario_id,
                    tgt,
                    edge_type="recommended_for",
                    weight=float(strength),
                    dynamic_strength=float(strength),
                )
                relation_edges_injected += 1
    
    for (m1, m2), corr in error_corrs.items():
        if corr < corr_threshold:
            mg.add_relation(m1, m2, "complementary", weight=1.0 - corr)
            mg.add_relation(m2, m1, "complementary", weight=1.0 - corr)
    # 高误差相关只作连续惩罚（在 selector 的 corr_penalty 里），不再写成硬 conflict。

    # 复制外部图谱的显式 conflict 边（Task 8.3 Task 3）：只有两端都是当前候选、
    # edge_type == "conflict" 的模型间边才被复制，来源记为 external_graph。
    explicit_conflict_edges_consumed = 0
    if relation_graph is not None:
        for src, tgt, data in relation_graph.G.edges(data=True):
            if data.get("edge_type") != "conflict":
                continue
            if src not in model_cols or tgt not in model_cols:
                continue
            weight = float(data.get("weight", 1.0))
            mg.G.add_edge(src, tgt, edge_type="conflict", weight=weight, source="external_graph")
            mg.G.add_edge(tgt, src, edge_type="conflict", weight=weight, source="external_graph")
            explicit_conflict_edges_consumed += 1

    # 闭环反馈：将上一步长的历史性能分注入互补边权重
    feedback_apply_meta: Dict[str, Any] = {"enabled": False}
    if feedback_store is not None:
        fb_result = feedback_store.apply_to_graph(mg)
        feedback_apply_meta = {"enabled": True, **fb_result}

    # 识别特征列
    exclude_cols = {"timestamp", "y", "row_id", "_stable_key", "_ts_dt", "_orig_idx"}
    feature_cols = [c for c in df_raw_val.columns if c not in exclude_cols and c in df_raw_test.columns]
    
    if not feature_cols:
        result = protocol_a_reference
        result["protocol"] = "B_fallback_to_A_no_features"
        return result
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
    
    # 添加 Feature -> Model 边
    feat_model_corrs = compute_feature_model_correlation_safe(
        df_raw_val.copy(), df_val.copy(), model_cols, feature_cols
    )
    
    for feat in feature_cols:
        mg.add_feature_node(feat, description=f"Raw feature: {feat}")
    
    for (feat, model), corr in feat_model_corrs.items():
        if corr > 0.1:  # 有意义的相关性
            mg.add_feature_model_edge(feat, model, required=corr > 0.3)
    
    # 场景节点
    scenario_id = f"scenario_h{horizon}"
    scenario_attrs = {
        "horizon": horizon,
        "horizon_class": "short" if horizon <= 1 else ("medium" if horizon <= 6 else "long"),
        "n_samples": len(df_val)
    }
    mg.add_scenario_node(scenario_id, scenario_attrs)
    
    for feat in feature_cols[:10]:  # 限制数量
        mg.add_scenario_feature_edge(scenario_id, feat, weight=1.0)
    
    # 显式推理证据（可控启用）：
    # - off: 完全关闭（disabled，贡献为 0）
    # - hybrid / prefer: 都只提供有来源的评分证据，不再覆盖 stepwise 结果。
    reasoning_evidence = build_reasoning_evidence(
        graph=mg,
        scenario_id=scenario_id,
        available_features=set(feature_cols),
        model_cols=model_cols,
        max_models=max(2, max_models),
        mode=PROTOCOL_B_REASONING_MODE,
    )
    reasoning_used = reasoning_evidence.source == "historical_evidence"
    reasoning_top_path = reasoning_evidence.paths[0] if reasoning_evidence.paths else None
    reasoning_top_models = (reasoning_top_path or {}).get("models", [])

    reasoning_meta = {
        "used_reasoning_path": reasoning_used,
        "reasoning_used_rate": 1.0 if reasoning_used else 0.0,
        "reasoning_mode": PROTOCOL_B_REASONING_MODE,
        "reasoning_source": reasoning_evidence.source,
        "reasoning_path": reasoning_top_path.get("path_id") if reasoning_top_path else None,
        "reasoning_score": reasoning_top_path.get("final_score") if reasoning_top_path else None,
        "reasoning_disabled": PROTOCOL_B_REASONING_MODE == "off",
        "reasoning_candidate_paths": len(reasoning_evidence.paths),
        "reasoning_contribution": dict(reasoning_evidence.contribution_by_model),
        "cold_start_no_evidence": reasoning_evidence.cold_start_no_evidence,
    }

    # 拟合参数在选择之前就已确定（只取决于 drift/dataset/horizon）。提前解析出来，
    # 让 selector 的 pair 资格判定用与最终拟合完全相同的 Ridge 配置。
    drift_decay, drift_decay_meta = _resolve_drift_aware_temporal_decay(
        base_decay=KG_RIDGE_TEMPORAL_DECAY,
        drift_level=drift_level,
        drift_median_psi=drift_median_psi,
    )
    temporal_min_w_ratio = None
    if drift_level == "high":
        temporal_min_w_ratio = min(
            1.0,
            KG_TEMPORAL_WEIGHT_MIN_RATIO * PROTOCOL_B_HIGH_DRIFT_MIN_W_MULTIPLIER,
        )
    if dataset_name in PROTOCOL_B_MIN_W_OVERRIDE_DATASETS:
        base_ratio = temporal_min_w_ratio if temporal_min_w_ratio is not None else KG_TEMPORAL_WEIGHT_MIN_RATIO
        temporal_min_w_ratio = min(1.0, max(float(base_ratio), float(PROTOCOL_B_MIN_W_OVERRIDE_VALUE)))

    # P0-3: h24 alpha 加强正则化
    b_alpha_candidates = HIGH_DRIFT_ALPHA_CANDIDATES if drift_level == "high" else None
    if horizon >= KG_LONG_HORIZON_MIN_H and b_alpha_candidates is None:
        b_alpha_candidates = [a * KG_LONG_HORIZON_ALPHA_MULTIPLIER for a in [1.0, 10.0, 50.0, 100.0, 500.0]]

    pair_fit_config = {
        "alpha_candidates": b_alpha_candidates,
        "temporal_decay": drift_decay,
        "temporal_decay_meta": drift_decay_meta,
        "temporal_min_weight_ratio": temporal_min_w_ratio,
    }

    # Protocol B 差异化：特征奖励 + 误差相关惩罚（避免 A/B 完全一致）
    feature_bonus_weight = PROTOCOL_B_FEATURE_BONUS_WEIGHT
    corr_penalty_weight = PROTOCOL_B_CORR_PENALTY_WEIGHT
    if len(df_val) < PROTOCOL_B_SMALL_SAMPLE_THRESHOLD:
        feature_bonus_weight += 0.2
        corr_penalty_weight *= 0.5
    if _fixed_selected_models is not None:
        # 固定二模型诊断（Task 8.3 Task 4）：真正绕过 selector，不诱导评分。
        # 拟合、interaction、post-adjustment 仍复用生产实现；feature_bonus_map
        # 与 selector 用同一公式，保证 post-adjustment 与生产一致。
        selected = [m for m in _fixed_selected_models if m in model_cols]
        feature_bonus_map = _compute_feature_bonus_map(model_cols, feat_model_corrs)
        b_scores = {m: 0.0 for m in model_cols}
        b_select_meta = {
            "diagnostic_mode": "fixed_pair",
            "requested_models": list(selected),
            "candidate_order": list(selected),
            "selector_output": list(selected),
            "constraint_decisions": [],
            "stepwise_adoption": {
                "stepwise_output": None,
                "adopted": False,
                "not_adopted_reason": "diagnostic_fixed_pair",
                "candidate_source": "diagnostic_fixed_pair",
            },
            "pair_eligibility": {
                "checked": False,
                "reason_not_checked": "diagnostic_fixed_pair",
            },
            "pair_diagnostics": {},
            "score_components": {},
            "relation_strength": {
                "weight": float(PROTOCOL_B_RELATION_STRENGTH_WEIGHT),
                "neutral": float(PROTOCOL_B_RELATION_STRENGTH_NEUTRAL),
                "by_model": {m: float(PROTOCOL_B_RELATION_STRENGTH_NEUTRAL) for m in model_cols},
                "contribution": {m: 0.0 for m in model_cols},
                "edges_found": [],
            },
        }
    else:
        def full_pair_evaluator(pair: Sequence[str]) -> Dict:
            return evaluate_fixed_protocol_b_candidate(
                df_val, df_test, df_raw_val, df_raw_test,
                selected_models=pair,
                horizon=horizon,
                dataset_name=dataset_name,
                base_model_cols=base_model_cols,
            )

        selected, b_scores, feature_bonus_map, b_select_meta = select_models_protocol_b(
            mg=mg,
            model_cols=model_cols,
            maes=maes,
            error_corrs=error_corrs,
            feat_model_corrs=feat_model_corrs,
            horizon=horizon,
            df_val=df_val,
            max_models=max_models,
            feature_bonus_weight=feature_bonus_weight,
            corr_penalty_weight=corr_penalty_weight,
            feature_diversity_weight=PROTOCOL_B_FEATURE_DIVERSITY_WEIGHT,
            drift_level=drift_level,
            reasoning_contribution=reasoning_evidence.contribution_by_model,
            dataset_name=dataset_name,
            base_model_cols=base_model_cols,
            pair_fit_config=pair_fit_config,
            full_pair_evaluator=full_pair_evaluator,
        )
    # 选择流程（Task 8.3 Task 3）：selector 之后不再有任何改写 selected 的代码，
    # post_selector_mutations 恒为空；constraint_decisions 已在 selector 内记录。
    selection_flow = {
        "candidate_order": list(b_select_meta["candidate_order"]),
        "selector_output": list(selected),
        "reasoning": {
            "used": bool(reasoning_used),
            "mode": PROTOCOL_B_REASONING_MODE,
            "source": reasoning_evidence.source,
            "path": reasoning_meta["reasoning_path"],
            "path_models": list(reasoning_top_models),
            "contribution": dict(reasoning_evidence.contribution_by_model),
        },
        "constraint_decisions": list(b_select_meta["constraint_decisions"]),
        # stepwise 输出是否被采用、pair 资格判定过程（Task 8.3 Task 10）
        "stepwise_adoption": dict(b_select_meta["stepwise_adoption"]),
        "pair_eligibility": dict(b_select_meta["pair_eligibility"]),
        "post_selector_mutations": [],
        "final_selected_before_fit": list(selected),
    }
    # selector 判定"没有任何合格 pair"时，最终回退目标要在下方 guard 处改成最佳
    # 单模型：Protocol A 保留旧语义，其输出本身可能带零权重模型，回退到 A 会把刚
    # 被判为不合格的退化组合重新带进最终结果。
    no_eligible_pair = (
        (b_select_meta.get("pair_eligibility") or {}).get("outcome") == "no_eligible_pair"
    )

    pred_val, pred_test, weights, ridge_meta = fit_static_weight_ridge(
        df_val=df_val,
        df_test=df_test,
        selected_models=selected,
        horizon=horizon,
        alpha_candidates=b_alpha_candidates,
        temporal_decay=drift_decay,
        temporal_decay_meta=drift_decay_meta,
        temporal_min_weight_ratio=temporal_min_w_ratio,
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
        alpha_candidates=b_alpha_candidates,
        temporal_decay=drift_decay,
        temporal_decay_meta=drift_decay_meta,
        temporal_min_weight_ratio=temporal_min_w_ratio,
    )
    # fitted_models 只可能因拟合后的近零权重清理与 final_selected_before_fit 不同。
    selection_flow["fitted_models"] = list(selected)
    ridge_meta["protocol_b_selection_meta"] = {
        "max_models": int(max_models),
        "high_drift_extended_score_factor": (
            float(KG_HIGH_DRIFT_EXTENDED_SCORE_FACTOR) if drift_level == "high" else 1.0
        ),
        "explicit_conflict_edges_consumed": explicit_conflict_edges_consumed,
        **b_select_meta,
        "selection_flow": selection_flow,
        "stability": {
            "by_model": stability_meta,
            "removed_models": stability_removed,
        },
    }

    # Protocol B 差异化主分支：在主干预测上叠加“模型 x 特征”交互残差。
    applied_interaction_spec: Optional[Dict[str, Any]] = None
    interaction_meta: Dict[str, Any] = {
        "enabled": False,
        "applied": False,
        "disabled_reason": None,
        "disabled_reason_code": None,
        "reject_reasons": [],
        "selected_features": [],
        "n_interactions": 0,
    }
    disable_interaction = (
        drift_level == "high"
        and int(horizon) >= int(PROTOCOL_B_DISABLE_INTERACTION_HIGH_DRIFT_MIN_H)
    )
    # P0-2: 支持按数据集显式禁用 interaction（如 NSW 低信噪比场景）
    disable_by_dataset = bool(
        PROTOCOL_B_DISABLE_INTERACTION_DATASETS
        and dataset_name in PROTOCOL_B_DISABLE_INTERACTION_DATASETS
    )
    if disable_by_dataset:
        disable_interaction = True
    if disable_interaction:
        if disable_by_dataset:
            interaction_meta["disabled_reason"] = (
                f"dataset_disable_interaction: dataset={dataset_name}"
            )
            interaction_meta["disabled_reason_code"] = "dataset_guard"
        else:
            interaction_meta["disabled_reason"] = (
                "high_drift_long_horizon_guard: "
                f"drift_level={drift_level}, horizon={horizon}, "
                f"min_h={int(PROTOCOL_B_DISABLE_INTERACTION_HIGH_DRIFT_MIN_H)}"
            )
            interaction_meta["disabled_reason_code"] = "drift_guard"
    else:
        try:
            raw_val_aligned = _align_raw_to_pred(df_raw_val.copy(), df_val.copy())
            raw_test_aligned = _align_raw_to_pred(df_raw_test.copy(), df_test.copy())
            if len(raw_val_aligned) == len(df_val) and len(raw_test_aligned) == len(df_test):
                numeric_feature_cols = []
                for feat in feature_cols:
                    if feat not in raw_val_aligned.columns or feat not in raw_test_aligned.columns:
                        continue
                    v = pd.to_numeric(raw_val_aligned[feat], errors="coerce")
                    if np.isfinite(v).sum() < max(30, len(v) // 10):
                        continue
                    numeric_feature_cols.append(feat)

                feat_scores = {}
                for feat in numeric_feature_cols:
                    scores = [
                        float(corr) for (f, m), corr in feat_model_corrs.items()
                        if f == feat and m in selected and np.isfinite(corr)
                    ]
                    if scores:
                        feat_scores[feat] = float(np.mean(scores))
                selected_feats = [
                    f for f, s in sorted(feat_scores.items(), key=lambda kv: kv[1], reverse=True)
                    if s > 0.15
                ][:3]

                if selected_feats:
                    interaction_meta["enabled"] = True
                    interaction_meta["selected_features"] = selected_feats
                    inter_val_cols = []
                    inter_test_cols = []
                    inter_names = []
                    for feat in selected_feats:
                        f_val = pd.to_numeric(raw_val_aligned[feat], errors="coerce")
                        f_mu = float(np.nanmean(f_val)) if np.isfinite(f_val).any() else 0.0
                        f_val = f_val.fillna(f_mu).values.astype(float) - f_mu
                        f_test = pd.to_numeric(raw_test_aligned[feat], errors="coerce").fillna(f_mu).values.astype(float) - f_mu
                        for m in selected:
                            inter_val_cols.append(df_val[m].values.astype(float) * f_val)
                            inter_test_cols.append(df_test[m].values.astype(float) * f_test)
                            inter_names.append(f"{m}__x__{feat}")

                    if inter_val_cols:
                        X_inter_val = np.column_stack(inter_val_cols)
                        X_inter_test = np.column_stack(inter_test_cols)
                        residual_target = y_val - pred_val
                        n_folds_i, min_train_i, gap_i = _resolve_cv_config(len(y_val), horizon)
                        inter_sample_weight = _compute_temporal_weights(
                            len(y_val),
                            drift_decay,
                            min_ratio=temporal_min_w_ratio,
                        )
                        alpha_i, cv_i = blocked_cv_select_alpha(
                            X_inter_val,
                            residual_target,
                            alphas=[0.1, 1.0, 10.0, 100.0],
                            n_folds=n_folds_i,
                            min_train=min_train_i,
                            positive=False,
                            fit_intercept=False,
                            sample_weight=inter_sample_weight,
                            gap=gap_i,
                        )
                        reg_i, solver_meta_i = fit_ridge_robust(
                            X_inter_val,
                            residual_target,
                            alpha=float(alpha_i),
                            positive=False,
                            fit_intercept=False,
                            sample_weight=inter_sample_weight,
                        )
                        pred_val_i = pred_val + reg_i.predict(X_inter_val)
                        pred_test_i = pred_test + reg_i.predict(X_inter_test)
                        val_mae_raw = float(np.mean(np.abs(pred_val - y_val)))
                        val_mae_i = float(np.mean(np.abs(pred_val_i - y_val)))
                        cv_mae_raw_leaky = _blocked_cv_mae_from_pred(y_val, pred_val, horizon)
                        cv_mae_i_leaky = _blocked_cv_mae_from_pred(y_val, pred_val_i, horizon)
                        oof_cv = _interaction_oof_cv_metrics(
                            X_inter_val=X_inter_val,
                            residual_target=residual_target,
                            base_pred=pred_val,
                            y_true=y_val,
                            alpha=float(alpha_i),
                            horizon=horizon,
                            sample_weight=inter_sample_weight,
                        )
                        cv_mae_raw_guard = cv_mae_raw_leaky
                        cv_mae_i_guard = cv_mae_i_leaky
                        cv_guard_source = "leaky_blocked_cv"
                        if oof_cv is not None:
                            cv_mae_raw_guard = float(oof_cv["mae_raw_oof"])
                            cv_mae_i_guard = float(oof_cv["mae_inter_oof"])
                            cv_guard_source = "oof_blocked_cv"
                        cv_pass = True
                        if cv_mae_raw_guard is not None and cv_mae_i_guard is not None:
                            cv_pass = cv_mae_i_guard <= cv_mae_raw_guard * (
                                1.0 + PROTOCOL_B_POST_ADJUST_MAX_DEGRADATION
                            )

                        def _tail_p95(y_true: np.ndarray, pred_arr: np.ndarray) -> float:
                            ae = np.abs(y_true - pred_arr)
                            return float(np.percentile(ae, 95))

                        tail_raw = _tail_p95(y_val, pred_val)
                        tail_i = _tail_p95(y_val, pred_val_i)
                        tail_pass = tail_i <= tail_raw * (1.0 + PROTOCOL_B_POST_ADJUST_MAX_DEGRADATION)

                        # 额外门禁：末段窗口 MAE 不恶化（避免 val 平均改善但 tail 崩溃）
                        tail_window = max(
                            int(len(y_val) * PROTOCOL_B_LAST_BLOCK_RATIO),
                            max(PROTOCOL_B_LAST_BLOCK_MIN_SAMPLES, int(horizon) * 2),
                        )
                        tail_window = min(max(tail_window, 1), len(y_val))
                        tail_slice = slice(len(y_val) - tail_window, len(y_val))
                        tail_window_raw = float(np.mean(np.abs(pred_val[tail_slice] - y_val[tail_slice])))
                        tail_window_i = float(np.mean(np.abs(pred_val_i[tail_slice] - y_val[tail_slice])))
                        tail_window_pass = True
                        if PROTOCOL_B_LAST_BLOCK_GUARD_ENABLED:
                            tail_window_pass = (
                                np.isfinite(tail_window_i)
                                and tail_window_i <= tail_window_raw * (1.0 + PROTOCOL_B_POST_ADJUST_MAX_DEGRADATION)
                            )

                        reject_reasons: List[str] = []
                        if not cv_pass:
                            reject_reasons.append("cv_fail")
                        if not tail_pass:
                            reject_reasons.append("tail_fail")
                        if PROTOCOL_B_LAST_BLOCK_GUARD_ENABLED and not tail_window_pass:
                            reject_reasons.append("last_block_fail")

                        accept_interaction = (
                            np.isfinite(val_mae_i)
                            and val_mae_i <= val_mae_raw * (1.0 + PROTOCOL_B_POST_ADJUST_MAX_DEGRADATION)
                            and cv_pass
                            and tail_pass
                            and tail_window_pass
                        )
                        if not accept_interaction and not reject_reasons:
                            reject_reasons.append("val_guard")
                        if accept_interaction:
                            pred_val = pred_val_i
                            pred_test = pred_test_i
                            interaction_meta["applied"] = True
                            # 供 guard 的统一折外比较逐折重拟合同一交互结构使用
                            applied_interaction_spec = {
                                "features": X_inter_val,
                                "alpha": float(alpha_i),
                                "sample_weight": inter_sample_weight,
                            }
                        interaction_meta.update({
                            "n_interactions": int(X_inter_val.shape[1]),
                            "alpha": float(alpha_i),
                            "cv_mae": float(cv_i) if np.isfinite(cv_i) else None,
                            "val_mae_raw": val_mae_raw,
                            "val_mae_interaction": val_mae_i,
                            "cv_mae_raw": float(cv_mae_raw_leaky) if cv_mae_raw_leaky is not None else None,
                            "cv_mae_interaction": float(cv_mae_i_leaky) if cv_mae_i_leaky is not None else None,
                            "cv_mae_raw_guard": (
                                float(cv_mae_raw_guard) if cv_mae_raw_guard is not None else None
                            ),
                            "cv_mae_interaction_guard": (
                                float(cv_mae_i_guard) if cv_mae_i_guard is not None else None
                            ),
                            "cv_guard_source": cv_guard_source,
                            "cv_oof_coverage": (
                                float(oof_cv["oof_coverage"]) if isinstance(oof_cv, dict) else None
                            ),
                            "cv_pass": bool(cv_pass),
                            "tail_p95_raw": float(tail_raw),
                            "tail_p95_interaction": float(tail_i),
                            "tail_pass": bool(tail_pass),
                            "tail_window": int(tail_window),
                            "last_block_guard_enabled": bool(PROTOCOL_B_LAST_BLOCK_GUARD_ENABLED),
                            "last_block_ratio": float(PROTOCOL_B_LAST_BLOCK_RATIO),
                            "tail_window_mae_raw": float(tail_window_raw),
                            "tail_window_mae_interaction": float(tail_window_i),
                            "tail_window_pass": bool(tail_window_pass),
                            "reject_reasons": reject_reasons,
                            "interaction_terms": inter_names,
                            "solver_meta": solver_meta_i,
                        })
                else:
                    interaction_meta["disabled_reason"] = "no_reliable_feature_signal"
                    interaction_meta["disabled_reason_code"] = "small_sample"
            else:
                interaction_meta["disabled_reason"] = "raw_pred_alignment_failed"
                interaction_meta["disabled_reason_code"] = "small_sample"
        except Exception as e:
            interaction_meta["error"] = str(e)
            interaction_meta["disabled_reason_code"] = "cv_fail"
    ridge_meta["interaction_branch"] = interaction_meta
    ridge_meta["interaction_disable_reason"] = interaction_meta.get("disabled_reason_code")

    # Protocol B 差异化后处理：在 Ridge 权重基础上施加
    # feature bonus（奖励）与 error-corr（惩罚）重标定。
    w_vec = np.array([weights[m] for m in selected], dtype=float)
    feature_bonus_vec = np.array([feature_bonus_map.get(m, 0.0) for m in selected], dtype=float)
    corr_pen_vec = np.array([
        np.mean([
            abs(_lookup_pair_corr(error_corrs, m, mm, default=0.0))
            for mm in selected if mm != m
        ]) if len(selected) > 1 else 0.0
        for m in selected
    ], dtype=float)
    adjust_bonus_scale = PROTOCOL_B_ADJUST_BONUS_SCALE
    adjust_penalty_scale = PROTOCOL_B_ADJUST_PENALTY_SCALE
    if len(df_val) < PROTOCOL_B_SMALL_SAMPLE_THRESHOLD:
        adjust_bonus_scale += 0.2
        adjust_penalty_scale *= 0.5
    adjust_gain = np.exp(adjust_bonus_scale * feature_bonus_vec - adjust_penalty_scale * corr_pen_vec)

    if np.any(np.abs(adjust_gain - adjust_gain.mean()) > 1e-8):
        w_adj = np.maximum(w_vec, 0.0) * adjust_gain
        if w_adj.sum() > 1e-10:
            # 先归一化，再根据原始 ridge 权重和决定目标尺度：
            # - 原始和在合理区间时，保持其语义
            # - 否则回到 sum=1 的凸组合，避免尺度漂移
            w_adj = w_adj / (w_adj.sum() + 1e-12)
            raw_weight_sum = float(np.sum(w_vec))
            if 0.5 <= raw_weight_sum <= 1.5:
                target_sum = raw_weight_sum
            else:
                target_sum = 1.0
            w_adj = w_adj * target_sum

            pred_val_adj = df_val[selected].values @ w_adj
            pred_test_adj = df_test[selected].values @ w_adj

            val_mae_raw = float(np.mean(np.abs(pred_val - y_val)))
            val_mae_adj = float(np.mean(np.abs(pred_val_adj - y_val)))

            # Sanity check：后处理轻微恶化即可拒绝（默认阈值 3%）。
            accept_adjustment = (
                np.isfinite(val_mae_adj)
                and val_mae_adj <= val_mae_raw * (1.0 + PROTOCOL_B_POST_ADJUST_MAX_DEGRADATION)
            )
            if accept_adjustment:
                pred_val = pred_val_adj
                pred_test = pred_test_adj
                weights = dict(zip(selected, w_adj.tolist()))

            ridge_meta["post_adjustment"] = {
                "mode": "feature_bonus_minus_error_corr",
                "applied": bool(accept_adjustment),
                "raw_weight_sum": raw_weight_sum,
                "adjusted_weight_sum": float(np.sum(w_adj)),
                "val_mae_raw": val_mae_raw,
                "val_mae_adjusted": val_mae_adj,
                "feature_bonus": dict(zip(selected, feature_bonus_vec.tolist())),
                "corr_penalty": dict(zip(selected, corr_pen_vec.tolist())),
            }

    # Protocol B 软回退保护：
    # 1) 验证集明显劣于 A（默认>5%）时回退；
    # 2) 小样本场景（默认 n<500）下，若相对 A 未取得最小改善（默认1%）则回退。
    val_mae_b = _safe_mae(y_val, np.asarray(pred_val, dtype=float))
    val_mae_b_finite = bool(np.isfinite(val_mae_b))
    fallback_reason = None
    if not val_mae_b_finite:
        fallback_reason = "invalid_val_mae_b_non_finite"
    fallback_eps = PROTOCOL_B_VAL_FALLBACK_EPS
    if drift_level == "high":
        fallback_eps = min(fallback_eps, PROTOCOL_B_HIGH_DRIFT_VAL_FALLBACK_EPS)
    small_sample_min_improve = PROTOCOL_B_SMALL_SAMPLE_MIN_IMPROVEMENT
    global_min_improve = _resolve_protocol_b_global_min_improve(len(df_val), drift_level)
    tail_holdout_ratio = PROTOCOL_B_TAIL_HOLDOUT_RATIO
    tail_holdout_max_ratio = PROTOCOL_B_TAIL_HOLDOUT_MAX_RATIO
    if drift_level == "high":
        tail_holdout_max_ratio = min(tail_holdout_max_ratio, PROTOCOL_B_TAIL_HOLDOUT_MAX_RATIO_HIGH)
    elif drift_level == "medium":
        tail_holdout_max_ratio = min(tail_holdout_max_ratio, PROTOCOL_B_TAIL_HOLDOUT_MAX_RATIO_MEDIUM)
    tail_start = int(len(y_val) * (1.0 - tail_holdout_ratio))
    tail_start = min(max(tail_start, 0), max(len(y_val) - 1, 0))
    tail_mae_b = None
    tail_full_ratio_b = None
    if len(y_val) > 1 and tail_start < len(y_val):
        tail_slice = slice(tail_start, len(y_val))
        tail_mae_b = _safe_mae(
            np.asarray(y_val[tail_slice], dtype=float),
            np.asarray(pred_val[tail_slice], dtype=float),
        )
        if np.isfinite(val_mae_b) and val_mae_b > 1e-10:
            tail_full_ratio_b = tail_mae_b / val_mae_b
    ridge_meta["guard_config"] = {
        "val_mae_b_finite": bool(val_mae_b_finite),
        "val_mae_b_non_finite": bool(not val_mae_b_finite),
        "val_mae_b_raw": val_mae_b,
        "drift_level_raw": drift_level_raw,
        "drift_level_effective": drift_level,
        "drift_level_source": drift_level_source,
        "drift_median_psi": (
            float(drift_median_psi_float)
            if drift_median_psi_float is not None and np.isfinite(drift_median_psi_float)
            else None
        ),
        "psi_deadband_low": float(PROTOCOL_B_PSI_DEADBAND_LOW),
        "psi_deadband_high": float(PROTOCOL_B_PSI_DEADBAND_HIGH),
        "val_fallback_eps": fallback_eps,
        "high_drift_val_fallback_eps": float(PROTOCOL_B_HIGH_DRIFT_VAL_FALLBACK_EPS),
        "high_drift_min_w_multiplier": float(PROTOCOL_B_HIGH_DRIFT_MIN_W_MULTIPLIER),
        "temporal_min_weight_ratio_override": (
            float(temporal_min_w_ratio) if temporal_min_w_ratio is not None else None
        ),
        "effective_min_w": (
            ridge_meta.get("temporal_weighting", {}).get("min_w")
            if isinstance(ridge_meta.get("temporal_weighting"), dict)
            else None
        ),
        "global_min_improvement": global_min_improve,
        "small_sample_threshold": PROTOCOL_B_SMALL_SAMPLE_THRESHOLD,
        "small_sample_min_improvement": small_sample_min_improve,
        "large_sample_threshold": PROTOCOL_B_LARGE_SAMPLE_THRESHOLD,
        "tail_holdout_ratio": float(tail_holdout_ratio),
        "tail_holdout_max_ratio": float(tail_holdout_max_ratio),
        "tail_holdout_max_ratio_medium": float(PROTOCOL_B_TAIL_HOLDOUT_MAX_RATIO_MEDIUM),
        "tail_holdout_max_ratio_high": float(PROTOCOL_B_TAIL_HOLDOUT_MAX_RATIO_HIGH),
        "long_horizon_min_h": int(KG_LONG_HORIZON_MIN_H),
        "long_horizon_min_improvement": float(PROTOCOL_B_LONG_H_MIN_IMPROVEMENT),
        "last_block_guard_enabled": bool(PROTOCOL_B_LAST_BLOCK_GUARD_ENABLED),
        "last_block_ratio": float(PROTOCOL_B_LAST_BLOCK_RATIO),
        "last_block_min_samples": int(PROTOCOL_B_LAST_BLOCK_MIN_SAMPLES),
        "dataset_last_block_guard_datasets": sorted(list(PROTOCOL_B_DATASET_LAST_BLOCK_GUARD_DATASETS)),
        "dataset_last_block_guard_ratio": float(PROTOCOL_B_DATASET_LAST_BLOCK_GUARD_RATIO),
        "dataset_last_block_guard_min_samples": int(PROTOCOL_B_DATASET_LAST_BLOCK_GUARD_MIN_SAMPLES),
        "dataset_last_block_guard_max_degradation": float(PROTOCOL_B_DATASET_LAST_BLOCK_GUARD_MAX_DEGRADATION),
        "high_drift_overfit_val_improve_threshold": float(PROTOCOL_B_HIGH_DRIFT_OVERFIT_VAL_IMPROVE_THRESHOLD),
        "high_drift_overfit_tail_ratio_max": float(PROTOCOL_B_HIGH_DRIFT_OVERFIT_TAIL_RATIO_MAX),
        "high_drift_a_preferred_min_improve": float(PROTOCOL_B_HIGH_DRIFT_A_PREFERRED_MIN_IMPROVE),
        "high_drift_a_preferred_tail_ratio": float(PROTOCOL_B_HIGH_DRIFT_A_PREFERRED_TAIL_RATIO),
        "best_single_scope_report": str(PROTOCOL_B_BEST_SINGLE_SCOPE),
        "best_single_scope_guard": str(PROTOCOL_B_GUARD_BEST_SINGLE_SCOPE),
        "enforce_base_model_in_selection": bool(PROTOCOL_B_ENFORCE_BASE_MODEL_IN_SELECTION),
        "enforce_base_model_mode": str(PROTOCOL_B_ENFORCE_BASE_MODEL_MODE),
        "enforce_base_model_min_h": int(PROTOCOL_B_ENFORCE_BASE_MODEL_MIN_H),
        "enforce_base_model_datasets": sorted(list(PROTOCOL_B_ENFORCE_BASE_MODEL_DATASETS)),
        "tail_mae_b": float(tail_mae_b) if tail_mae_b is not None and np.isfinite(tail_mae_b) else None,
        "tail_full_ratio_b": (
            float(tail_full_ratio_b) if tail_full_ratio_b is not None and np.isfinite(tail_full_ratio_b) else None
        ),
        "interaction_disable_reason": ridge_meta.get("interaction_disable_reason"),
        "interaction_reject_reasons": (
            (ridge_meta.get("interaction_branch", {}) or {}).get("reject_reasons", [])
            if isinstance(ridge_meta.get("interaction_branch", {}), dict)
            else []
        ),
    }

    best_single_candidates = list(model_cols)
    if PROTOCOL_B_GUARD_BEST_SINGLE_SCOPE == "base_models_only":
        base_candidates = [m for m in (base_model_cols or []) if m in model_cols]
        if base_candidates:
            best_single_candidates = base_candidates
    best_single_model = (
        min(best_single_candidates, key=lambda m: maes.get(m, float("inf")))
        if best_single_candidates
        else (min(model_cols, key=lambda m: maes.get(m, float("inf"))) if model_cols else None)
    )
    ridge_meta["guard_config"]["best_single_candidates_count"] = int(len(best_single_candidates))
    ridge_meta["guard_config"]["best_single_candidates"] = list(best_single_candidates)
    ridge_meta["guard_config"]["best_base_model"] = best_base_model
    ridge_meta["guard_config"]["base_candidates_count"] = int(len(base_model_candidates))
    ridge_meta["guard_config"]["base_candidates"] = list(base_model_candidates)
    val_mae_best_single = (
        _safe_mae(y_val, np.asarray(df_val[best_single_model].values, dtype=float))
        if best_single_model is not None
        else None
    )
    test_mae_best_single = (
        _safe_mae(y_test, np.asarray(df_test[best_single_model].values, dtype=float))
        if best_single_model is not None
        else None
    )
    ridge_meta["guard_config"]["val_mae_best_base"] = val_mae_best_base
    ridge_meta["guard_config"]["test_mae_best_base"] = test_mae_best_base
    fallback_target = "protocol_a"
    rel_improve_b_vs_a = None
    if np.isfinite(val_mae_b) and val_mae_a is not None:
        rel_improve_b_vs_a = (float(val_mae_a) - val_mae_b) / max(float(val_mae_a), 1e-10)
        if val_mae_b > float(val_mae_a) * (1.0 + fallback_eps):
            fallback_reason = (
                f"val_guard: B({val_mae_b:.4f}) > A({float(val_mae_a):.4f})"
                f" * (1+{fallback_eps:.3f})"
            )
        elif (
            horizon >= KG_LONG_HORIZON_MIN_H
            and rel_improve_b_vs_a < PROTOCOL_B_LONG_H_MIN_IMPROVEMENT
        ):
            fallback_reason = (
                "long_horizon_min_improve_guard: "
                f"rel_improve={rel_improve_b_vs_a:.4f} < {PROTOCOL_B_LONG_H_MIN_IMPROVEMENT:.4f}, "
                f"horizon={horizon}"
            )
        elif (
            global_min_improve > 0
            and val_mae_b >= float(val_mae_a) * (1.0 - global_min_improve)
        ):
            fallback_reason = (
                f"global_min_improve_guard: B({val_mae_b:.4f}) >= "
                f"A({float(val_mae_a):.4f}) * (1-{global_min_improve:.4f}), "
                f"improvement < {global_min_improve * 100:.2f}%"
            )
        elif len(df_val) < PROTOCOL_B_SMALL_SAMPLE_THRESHOLD:
            rel_improve = (float(val_mae_a) - val_mae_b) / max(float(val_mae_a), 1e-10)
            if rel_improve < small_sample_min_improve:
                fallback_reason = (
                    f"small_sample_guard: n_val={len(df_val)} < {PROTOCOL_B_SMALL_SAMPLE_THRESHOLD}, "
                    f"rel_improve={rel_improve:.4f} < {small_sample_min_improve:.4f}"
                )
        elif (
            fallback_reason is None
            and tail_full_ratio_b is not None
            and np.isfinite(tail_full_ratio_b)
            and tail_full_ratio_b > tail_holdout_max_ratio
        ):
            fallback_reason = (
                f"tail_holdout_guard: tail/full={tail_full_ratio_b:.4f} > "
                f"{tail_holdout_max_ratio:.4f} (ratio={tail_holdout_ratio:.2f})"
            )
        elif fallback_reason is None and drift_level == "high":
            interaction_guard_ok = True
            interaction_guard = ridge_meta.get("interaction_branch", {})
            if isinstance(interaction_guard, dict):
                for gate_key in ("cv_pass", "tail_pass", "tail_window_pass"):
                    if gate_key in interaction_guard and not bool(interaction_guard.get(gate_key)):
                        interaction_guard_ok = False
                        break
            tail_guard_ok = (
                tail_full_ratio_b is not None
                and np.isfinite(tail_full_ratio_b)
                and tail_full_ratio_b <= (
                    PROTOCOL_B_HIGH_DRIFT_OVERFIT_TAIL_RATIO_MAX_SHORT_H
                    if horizon <= PROTOCOL_B_HIGH_DRIFT_OVERFIT_TAIL_RATIO_SHORT_H_MAX
                    else PROTOCOL_B_HIGH_DRIFT_OVERFIT_TAIL_RATIO_MAX
                )
            )
            tail_ratio_repr = (
                f"{tail_full_ratio_b:.4f}"
                if tail_full_ratio_b is not None and np.isfinite(tail_full_ratio_b)
                else "nan"
            )
            if (
                rel_improve_b_vs_a is not None
                and rel_improve_b_vs_a > PROTOCOL_B_HIGH_DRIFT_OVERFIT_VAL_IMPROVE_THRESHOLD
                and (not tail_guard_ok or not interaction_guard_ok)
            ):
                # 样本内改善过大本身不是过拟合证据（Task 8.3 Task 11）。真实九任务
                # 里该任务全部候选的 tail/full 几乎相同、都越过门槛，复合判据退化成
                # "越好越拒"。这里在同一组折、同一评价范围上分别算 A、B 的折外 MAE：
                # 折外支持 B 就不以样本内改善为由回退；不支持或不可用维持保守回退。
                # 阈值一律不动，只增加一条证据。
                oof_alpha = float(ridge_meta.get("best_alpha") or 1.0)
                oof_sample_weight = _compute_temporal_weights(
                    len(df_val), drift_decay, min_ratio=temporal_min_w_ratio,
                )
                models_a = list(
                    (protocol_a_reference.get("val") or {}).get("selected_models") or []
                )
                oof_mae_a, oof_cov_a = _unified_oof_mae(
                    df_val=df_val, models=models_a, horizon=horizon,
                    alpha=oof_alpha, sample_weight=oof_sample_weight,
                )
                oof_mae_b, oof_cov_b = _unified_oof_mae(
                    df_val=df_val, models=list(selected), horizon=horizon,
                    alpha=oof_alpha, sample_weight=oof_sample_weight,
                    interaction_features=(applied_interaction_spec or {}).get("features"),
                    interaction_alpha=(applied_interaction_spec or {}).get("alpha"),
                )
                oof_supports_b = oof_mae_b < oof_mae_a
                ridge_meta["guard_config"]["high_drift_overfit_oof_check"] = {
                    "evaluated": True,
                    "protocol": "same_blocked_cv_folds_same_alpha_and_sample_weight",
                    "alpha": oof_alpha,
                    "models_a": models_a,
                    "models_b": list(selected),
                    "interaction_included_for_b": applied_interaction_spec is not None,
                    "oof_mae_a": oof_mae_a,
                    "oof_mae_b": oof_mae_b,
                    "oof_coverage_a": oof_cov_a,
                    "oof_coverage_b": oof_cov_b,
                    "rel_improve_b_vs_a": float(rel_improve_b_vs_a),
                    "tail_full_ratio_b": (
                        float(tail_full_ratio_b)
                        if tail_full_ratio_b is not None and np.isfinite(tail_full_ratio_b)
                        else None
                    ),
                    "supports_b": bool(oof_supports_b),
                }
                if not oof_supports_b:
                    fallback_reason = (
                        "high_drift_overfit_guard: "
                        f"rel_improve={rel_improve_b_vs_a:.4f} > {PROTOCOL_B_HIGH_DRIFT_OVERFIT_VAL_IMPROVE_THRESHOLD:.4f}, "
                        f"tail/full={tail_ratio_repr}, "
                        f"interaction_guard_ok={interaction_guard_ok}, "
                        f"oof_supports_b=False(oof_a={oof_mae_a:.4f}, oof_b={oof_mae_b:.4f})"
                    )
            elif (
                rel_improve_b_vs_a is not None
                and rel_improve_b_vs_a < PROTOCOL_B_HIGH_DRIFT_A_PREFERRED_MIN_IMPROVE
                and tail_full_ratio_b is not None
                and np.isfinite(tail_full_ratio_b)
                and tail_full_ratio_b > PROTOCOL_B_HIGH_DRIFT_A_PREFERRED_TAIL_RATIO
            ):
                fallback_reason = (
                    "A_preferred_by_drift_guard: "
                    f"rel_improve={rel_improve_b_vs_a:.4f} < {PROTOCOL_B_HIGH_DRIFT_A_PREFERRED_MIN_IMPROVE:.4f}, "
                    f"tail/full={tail_full_ratio_b:.4f} > {PROTOCOL_B_HIGH_DRIFT_A_PREFERRED_TAIL_RATIO:.4f}"
                )

    dataset_last_block_guard_enabled = bool(dataset_name in PROTOCOL_B_DATASET_LAST_BLOCK_GUARD_DATASETS)
    last_block_mae_a = None
    last_block_mae_b = None
    if (
        dataset_last_block_guard_enabled
        and val_mae_a is not None
        and np.isfinite(val_mae_a)
        and np.isfinite(val_mae_b)
        and len(y_val) > 1
    ):
        lb_size = max(
            int(len(y_val) * PROTOCOL_B_DATASET_LAST_BLOCK_GUARD_RATIO),
            max(PROTOCOL_B_DATASET_LAST_BLOCK_GUARD_MIN_SAMPLES, int(horizon) * 2),
        )
        lb_size = min(max(lb_size, 1), len(y_val))
        lb_slice = slice(len(y_val) - lb_size, len(y_val))
        # NSW 强化：last_block 使用时间衰减加权 MAE，近期样本权重更高
        lb_temporal_w = _compute_temporal_weights(
            lb_size,
            max(drift_decay, KG_RIDGE_TEMPORAL_DECAY),
            min_ratio=temporal_min_w_ratio,
        )
        def _weighted_mae(y_true_sl, y_pred_sl, weights_sl=None):
            ae = np.abs(np.asarray(y_true_sl, dtype=float) - np.asarray(y_pred_sl, dtype=float))
            mask = np.isfinite(ae)
            if int(mask.sum()) <= 0:
                return float("nan")
            if weights_sl is not None and len(weights_sl) == len(ae):
                w = weights_sl[mask]
                return float(np.sum(ae[mask] * w) / np.sum(w))
            return float(np.mean(ae[mask]))

        last_block_mae_a = float("nan")
        a_val_weights = (protocol_a_reference.get("val", {}) or {}).get("weights", {})
        if isinstance(a_val_weights, dict) and a_val_weights:
            try:
                pred_a = np.zeros(len(y_val), dtype=float)
                used = 0
                for m, w in a_val_weights.items():
                    if m in df_val.columns:
                        pred_a += np.asarray(df_val[m].values, dtype=float) * float(w)
                        used += 1
                if used > 0:
                    last_block_mae_a = _weighted_mae(
                        y_val[lb_slice], pred_a[lb_slice], lb_temporal_w,
                    )
            except Exception:
                last_block_mae_a = float("nan")
        if not np.isfinite(last_block_mae_a):
            last_block_mae_a = float(val_mae_a)
        last_block_mae_b = _weighted_mae(
            y_val[lb_slice], pred_val[lb_slice], lb_temporal_w,
        )
        if (
            np.isfinite(last_block_mae_a)
            and np.isfinite(last_block_mae_b)
            and last_block_mae_a > 1e-10
            and last_block_mae_b > last_block_mae_a * (1.0 + PROTOCOL_B_DATASET_LAST_BLOCK_GUARD_MAX_DEGRADATION)
        ):
            lb_reason = (
                "last_block_guard: "
                f"B_last({last_block_mae_b:.4f}) > A_last({last_block_mae_a:.4f})"
                f" * (1+{PROTOCOL_B_DATASET_LAST_BLOCK_GUARD_MAX_DEGRADATION:.3f})"
            )
            fallback_reason = lb_reason if not fallback_reason else f"{fallback_reason};{lb_reason}"
            fallback_target = "best_single"
    ridge_meta["guard_config"]["dataset_last_block_guard_enabled"] = dataset_last_block_guard_enabled
    ridge_meta["guard_config"]["last_block_mae_a"] = (
        float(last_block_mae_a) if last_block_mae_a is not None and np.isfinite(last_block_mae_a) else None
    )
    ridge_meta["guard_config"]["last_block_mae_b"] = (
        float(last_block_mae_b) if last_block_mae_b is not None and np.isfinite(last_block_mae_b) else None
    )

    complexity_guard_enabled = (
        PROTOCOL_B_COMPLEXITY_PENALTY_ENABLED
        and (not PROTOCOL_B_COMPLEXITY_PENALTY_DATASETS or (dataset_name in PROTOCOL_B_COMPLEXITY_PENALTY_DATASETS))
    )
    rel_improve_b_vs_best_single = None
    complexity_penalty_triggered = False
    complexity_penalty_reason = None
    effective_complexity_min_improve = None
    complexity_penalty_target = (
        "best_single"
        if PROTOCOL_B_COMPLEXITY_PENALTY_FALLBACK == "best_single"
        else "protocol_a"
    )
    if (
        complexity_guard_enabled
        and np.isfinite(val_mae_b)
        and val_mae_best_single is not None
        and np.isfinite(val_mae_best_single)
        and val_mae_best_single > 1e-10
    ):
        rel_improve_b_vs_best_single = (
            float(val_mae_best_single) - float(val_mae_b)
        ) / max(float(val_mae_best_single), 1e-10)
        # P1: eligible 模型少（<=5）时降低门槛，因为组合空间小导致改进率天然低
        effective_complexity_min_improve = PROTOCOL_B_COMPLEXITY_PENALTY_MIN_REL_IMPROVE
        if len(model_cols) <= PROTOCOL_B_COMPLEXITY_PENALTY_FEW_MODELS_THRESHOLD:
            effective_complexity_min_improve = PROTOCOL_B_COMPLEXITY_PENALTY_MIN_REL_IMPROVE_FEW_MODELS
        effective_complexity_min_improve = (
            float(effective_complexity_min_improve) * float(PROTOCOL_B_COMPLEXITY_PENALTY_SOFT_MARGIN)
        )
        if rel_improve_b_vs_best_single < effective_complexity_min_improve:
            complexity_penalty_triggered = True
            complexity_penalty_reason = (
                "complexity_penalty_guard: "
                f"rel_improve_vs_best_single={rel_improve_b_vs_best_single:.4f} < "
                f"{effective_complexity_min_improve:.4f}"
                f" (soft_margin={PROTOCOL_B_COMPLEXITY_PENALTY_SOFT_MARGIN:.2f})"
                f" (n_models={len(model_cols)})"
            )

    if complexity_penalty_triggered and complexity_penalty_reason:
        fallback_reason = (
            complexity_penalty_reason
            if not fallback_reason
            else f"{fallback_reason};{complexity_penalty_reason}"
        )
        if complexity_penalty_target == "best_single":
            fallback_target = "best_single"

    if (
        fallback_reason
        and fallback_target == "protocol_a"
        and val_mae_a is not None
        and np.isfinite(val_mae_a)
        and val_mae_best_single is not None
        and np.isfinite(val_mae_best_single)
        and float(val_mae_a) > float(val_mae_best_single)
    ):
        fallback_target = "best_single"
        extra_reason = (
            "A_worse_than_best_single_guard: "
            f"A({float(val_mae_a):.4f}) > best_single({float(val_mae_best_single):.4f})"
        )
        fallback_reason = f"{fallback_reason};{extra_reason}"

    # 无合格 pair 时改走最佳单模型分支（Task 8.3 Task 10）。只改回退目标，不动任何
    # guard 阈值、不改 Protocol A：否则 selector 刚拒绝的退化组合会经由 A 的输出
    # 重新出现在最终结果里。
    if (
        fallback_reason
        and fallback_target == "protocol_a"
        and no_eligible_pair
        and best_single_model is not None
    ):
        fallback_target = "best_single"
        fallback_reason = f"{fallback_reason};no_eligible_pair_fallback_to_best_single"

    ridge_meta["guard_config"].update({
        "complexity_penalty_enabled": bool(complexity_guard_enabled),
        "complexity_penalty_min_rel_improve": float(PROTOCOL_B_COMPLEXITY_PENALTY_MIN_REL_IMPROVE),
        "complexity_penalty_soft_margin": float(PROTOCOL_B_COMPLEXITY_PENALTY_SOFT_MARGIN),
        "complexity_penalty_effective_threshold": (
            float(effective_complexity_min_improve)
            if effective_complexity_min_improve is not None and np.isfinite(effective_complexity_min_improve)
            else None
        ),
        "complexity_penalty_fallback": str(PROTOCOL_B_COMPLEXITY_PENALTY_FALLBACK),
        "complexity_penalty_triggered": bool(complexity_penalty_triggered),
        "complexity_penalty_reason": complexity_penalty_reason,
        "complexity_penalty_target": complexity_penalty_target,
        "best_single_model": best_single_model,
        "val_mae_best_single": val_mae_best_single,
        "test_mae_best_single": test_mae_best_single,
        "rel_improve_b_vs_best_single": rel_improve_b_vs_best_single,
    })

    # P0-3: 多窗口一致性守卫（h24+）
    # 在 3 个不同大小的时间窗口上比较 B vs best_single，
    # 若多数窗口 B 恶化则认为 val 上的改善不可靠。
    multi_window_guard_meta: Dict[str, Any] = {"enabled": False}
    if (
        PROTOCOL_B_MULTI_WINDOW_GUARD_ENABLED
        and horizon >= PROTOCOL_B_MULTI_WINDOW_GUARD_MIN_H
        and not fallback_reason  # 已有回退原因时不再追加
        and best_single_model is not None
        and best_single_model in df_val.columns
        and np.isfinite(val_mae_b)
        and len(y_val) > 1
    ):
        multi_window_guard_meta["enabled"] = True
        best_single_pred_val = np.asarray(df_val[best_single_model].values, dtype=float)
        mw_window_ratios = [0.15, 0.25, 0.40]
        mw_details = []
        mw_degraded_count = 0
        for ratio in mw_window_ratios:
            w_size = max(int(len(y_val) * ratio), max(int(horizon) * 2, 16))
            w_size = min(w_size, len(y_val))
            w_slice = slice(len(y_val) - w_size, len(y_val))
            w_mae_b = _safe_mae(y_val[w_slice], pred_val[w_slice])
            w_mae_single = _safe_mae(y_val[w_slice], best_single_pred_val[w_slice])
            degraded = (
                np.isfinite(w_mae_b) and np.isfinite(w_mae_single)
                and w_mae_single > 1e-10
                and w_mae_b > w_mae_single * (1.0 + PROTOCOL_B_MULTI_WINDOW_GUARD_MAX_DEGRADATION)
            )
            if degraded:
                mw_degraded_count += 1
            mw_details.append({
                "window_ratio": ratio,
                "window_size": int(w_size),
                "mae_b": float(w_mae_b) if np.isfinite(w_mae_b) else None,
                "mae_single": float(w_mae_single) if np.isfinite(w_mae_single) else None,
                "degraded": bool(degraded),
            })
        multi_window_guard_meta.update({
            "windows": mw_details,
            "degraded_count": mw_degraded_count,
            "majority_threshold": int(PROTOCOL_B_MULTI_WINDOW_GUARD_MAJORITY),
            "max_degradation": float(PROTOCOL_B_MULTI_WINDOW_GUARD_MAX_DEGRADATION),
            "best_single_model": best_single_model,
        })
        if mw_degraded_count >= int(PROTOCOL_B_MULTI_WINDOW_GUARD_MAJORITY):
            mw_reason = (
                f"multi_window_consistency_guard: {mw_degraded_count}/{len(mw_window_ratios)} "
                f"windows degraded vs {best_single_model}, "
                f"horizon={horizon}, threshold={PROTOCOL_B_MULTI_WINDOW_GUARD_MAX_DEGRADATION:.4f}"
            )
            fallback_reason = mw_reason if not fallback_reason else f"{fallback_reason};{mw_reason}"
            fallback_target = "best_single"
            multi_window_guard_meta["triggered"] = True
        else:
            multi_window_guard_meta["triggered"] = False
    ridge_meta["guard_config"]["multi_window_guard"] = multi_window_guard_meta

    # 带符号关系反馈证据（Task 8.3 Task 5）：方向与幅度只由 blocked-CV/OOF 决定。
    # guard 回退时跳过并留痕；样本内 validation gain 与 Ridge 权重只作审计。
    effective_fallback_target = fallback_target if fallback_reason else None
    relation_feedback = compute_relation_feedback_evidence(
        df_val=df_val,
        candidate_models=model_cols,
        selected_models=selected,
        horizon=horizon,
        final_weights=weights,
        fallback_target=effective_fallback_target,
        alpha_candidates=b_alpha_candidates,
        temporal_decay=drift_decay,
        temporal_decay_meta=drift_decay_meta,
        temporal_min_weight_ratio=temporal_min_w_ratio,
    )
    ridge_meta["relation_feedback"] = relation_feedback

    if fallback_reason and not _skip_final_guard:
        naming_b = get_strategy_naming("kg_protocol_b", default_category="kg_core")
        guarded = copy.deepcopy(protocol_a_reference)
        guarded["protocol"] = "B_fallback_to_A_guard"
        if fallback_target == "best_single":
            fallback_model = best_single_model
            if (
                fallback_model is None
                or fallback_model not in df_val.columns
                or fallback_model not in df_test.columns
            ):
                if (
                    best_base_model is not None
                    and best_base_model in df_val.columns
                    and best_base_model in df_test.columns
                ):
                    fallback_model = best_base_model
                    fallback_reason = (
                        f"{fallback_reason};best_single_unavailable_use_best_base:{best_base_model}"
                        if fallback_reason
                        else f"best_single_unavailable_use_best_base:{best_base_model}"
                    )
                else:
                    fallback_target = "protocol_a"
                    fallback_reason = (
                        f"{fallback_reason};best_single_column_missing:{best_single_model}"
                        if fallback_reason
                        else f"best_single_column_missing:{best_single_model}"
                    )
            if fallback_target == "best_single" and fallback_model is not None:
                assert fallback_model in df_val.columns, f"best_single {fallback_model} not in df_val"
                assert fallback_model in df_test.columns, f"best_single {fallback_model} not in df_test"
                guarded = {
                    "val": {
                        **_merge_eval_metrics(df_val[fallback_model].values, y_val),
                        "selected_models": [fallback_model],
                        "weights": {fallback_model: 1.0},
                        "weight_meta": {},
                    },
                    "test": {
                        **_merge_eval_metrics(df_test[fallback_model].values, y_test),
                        "selected_models": [fallback_model],
                        "weights": {fallback_model: 1.0},
                        "weight_meta": {},
                    },
                    "protocol": "B_fallback_to_best_single_guard",
                }
        guard_cfg = ridge_meta.get("guard_config", {})
        if isinstance(guard_cfg, dict):
            guard_cfg["final_fallback_target"] = fallback_target
            guard_cfg["final_fallback_reason"] = fallback_reason
        for split in ("val", "test"):
            split_payload = guarded.get(split, {})
            if not isinstance(split_payload, dict):
                continue
            split_payload["strategy_id"] = naming_b["strategy_id"]
            split_payload["strategy_display_name"] = naming_b["display_name"]
            split_payload["method_family"] = naming_b["method_family"]
            split_payload["method_route"] = naming_b["method_route"]
            split_payload["core_route"] = naming_b["core_route"]
            weight_meta = split_payload.setdefault("weight_meta", {})
            if isinstance(weight_meta, dict):
                # 回退分支的 weight_meta 来自 Protocol A（best_single 时为空），
                # 不含 B 的 selection meta。把关系强度评分项回填进来，否则
                # "B 本来按什么排序、关系强度起了多大作用"在回退时无法回溯。
                b_meta_src = ridge_meta.get("protocol_b_selection_meta") or {}
                # stepwise 轨迹同理：回退时更需要知道 B 走了哪几步、
                # 是否在并列容差内做了取舍，否则两次运行结果不同时无从定位。
                for _key in (
                    "relation_strength",
                    "stepwise_meta",
                    "stepwise_alpha",
                    "stepwise_min_improve_ratio",
                    "score_components",
                    "pair_diagnostics",
                    "candidate_order",
                    "constraint_decisions",
                    "selector_output",
                    "selection_flow",
                ):
                    _val = b_meta_src.get(_key)
                    if _val is None:
                        continue
                    b_sel_meta = weight_meta.setdefault("protocol_b_selection_meta", {})
                    if isinstance(b_sel_meta, dict):
                        b_sel_meta[_key] = _val
                # 最终输出已经回退，但仍保留 B 候选分支的 interaction 审计信息。
                # 单独命名，避免误写成最终预测实际包含了 interaction。
                weight_meta["interaction_branch_candidate"] = copy.deepcopy(
                    ridge_meta.get("interaction_branch", {})
                )
                guard_cfg_out = dict(guard_cfg) if isinstance(guard_cfg, dict) else {}
                weight_meta["guard_config"] = guard_cfg_out
                guard_cfg_out["final_fallback_target"] = fallback_target
                guard_cfg_out["final_fallback_reason"] = fallback_reason
                weight_meta["protocol_b_guard"] = {
                    "fallback_to_a": fallback_target == "protocol_a",
                    "fallback_to_best_single": fallback_target == "best_single",
                    "fallback_target": fallback_target,
                    "reason": fallback_reason,
                    "val_mae_a": float(val_mae_a) if val_mae_a is not None else None,
                    "val_mae_b": val_mae_b,
                    "test_mae_a": float(test_mae_a) if test_mae_a is not None else None,
                    "best_single_model": best_single_model,
                    "val_mae_best_single": val_mae_best_single,
                    "test_mae_best_single": test_mae_best_single,
                    "best_base_model": best_base_model,
                    "val_mae_best_base": val_mae_best_base,
                    "test_mae_best_base": test_mae_best_base,
                    "val_fallback_eps": fallback_eps,
                    "global_min_improvement": global_min_improve,
                    "small_sample_min_improvement": small_sample_min_improve,
                }
            split_payload.update(reasoning_meta)
        # 回填 B 的候选元信息，便于分析为何被回退。
        guarded_val = guarded.get("val", {})
        if isinstance(guarded_val, dict):
            guarded_val["selected_models_b_candidate"] = selected
            guarded_val["weights_b_candidate"] = weights
            guarded_val["n_feature_model_edges"] = len(feat_model_corrs)
            guarded_val["n_features_used"] = len(feature_cols)
            guarded_val["model_scores_b"] = b_scores
            guarded_val["feature_bonus_weight"] = feature_bonus_weight
            guarded_val["corr_penalty_weight"] = corr_penalty_weight
            guarded_val.update(reasoning_meta)
        guarded_test = guarded.get("test", {})
        if isinstance(guarded_test, dict):
            guarded_test["selected_models_b_candidate"] = selected
            guarded_test["weights_b_candidate"] = weights
        if return_predictions and RUNTIME_PREDICTIONS_KEY not in guarded:
            # 回退到 Protocol A 时 guarded 由 protocol_a_reference 深拷贝而来，
            # 已带其运行时预测；这里只需补最佳单模型分支——它是新构造的 dict，
            # 预测就是该模型列本身。
            bs_models = list(guarded_test.get("selected_models") or [])
            if (
                len(bs_models) == 1
                and bs_models[0] in df_val.columns
                and bs_models[0] in df_test.columns
            ):
                _attach_predictions(
                    guarded,
                    df_val[bs_models[0]].values,
                    df_test[bs_models[0]].values,
                )
        guarded["relation_feedback"] = relation_feedback
        return guarded

    guard_cfg = ridge_meta.get("guard_config", {})
    if isinstance(guard_cfg, dict):
        guard_cfg["final_fallback_target"] = None
        guard_cfg["final_fallback_reason"] = None

    naming = get_strategy_naming("kg_protocol_b", default_category="kg_core")

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
            "n_feature_model_edges": len(feat_model_corrs),
            "n_features_used": len(feature_cols),
            "model_scores_b": b_scores,
            "feature_bonus_weight": feature_bonus_weight,
            "corr_penalty_weight": corr_penalty_weight,
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
        "protocol": "B_pred_features",
        "feedback_apply_meta": feedback_apply_meta,
        "relation_feedback": relation_feedback,
    }
    if _fixed_selected_models is not None:
        # 固定二模型诊断（Task 8.3 Task 4）：记录诊断标志与 guard 会做什么，
        # 但不执行回退。退化资格按近零权重清理后的 zero_weight_cleanup["after"]
        # 判定——二模型下生产清理不触发 refit，但"after"已反映实际有效模型。
        cleanup_after = (ridge_meta.get("zero_weight_cleanup") or {}).get("after")
        effective_models, eligible_pair, degenerate_reason = pair_eligibility_from_cleanup(
            selected, cleanup_after,
        )
        result["diagnostic_mode"] = "fixed_pair"
        result["requested_models"] = list(_fixed_selected_models)
        result["effective_models"] = effective_models
        result["eligible_pair"] = eligible_pair
        result["degenerate_reason"] = degenerate_reason
        result["fallback_target"] = None
        result["guard_would_fallback_to"] = fallback_target if fallback_reason else None
        result["guard_would_fallback_reason"] = fallback_reason
    # pred_val/pred_test 是引擎最终采用的预测：可能已叠加交互残差、也可能被
    # post_adjustment 覆盖回线性组合。交出它们本身，调用方不必再猜。
    return _attach_predictions(result, pred_val, pred_test)


def evaluate_fixed_protocol_b_candidate(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    df_raw_val: Optional[pd.DataFrame],
    df_raw_test: Optional[pd.DataFrame],
    *,
    selected_models: Sequence[str],
    horizon: int,
    dataset_name: Optional[str],
    base_model_cols: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """诊断专用固定二模型求值入口（Task 8.3 Task 4）。

    真正绕过候选选择器与最终 guard（不通过调高候选分数诱导选择），
    拟合、interaction 与 post-adjustment 复用生产实现。固定 pair 之外还记录
    guard 会回退到什么目标（但不执行），以及近零权重清理后的退化标记。
    """
    if len(set(selected_models)) != 2:
        raise ValueError("fixed candidate evaluation requires exactly two distinct models")
    return kg_combination_with_features(
        df_val, df_test, df_raw_val, df_raw_test,
        list(selected_models), horizon,
        dataset_name=dataset_name,
        base_model_cols=list(base_model_cols or selected_models),
        return_predictions=True,
        _fixed_selected_models=list(selected_models),
        _skip_final_guard=True,
    )
