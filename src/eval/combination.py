import os
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from strategies.rl_qms import RLQMSStrategy
from strategies.mole_router import MoLERouterStrategy
from strategies.dash_tta import DASHTTAStrategy

from src.selector.scenario_optimizer import (
    DirectWeightGatingNetwork,
    AdaptiveBucketSelector,
    ScenarioSimilarityEnhancer,
    should_use_optimized_strategy,
    DEFAULT_KG_COMPONENT_STRATEGIES,
)
from src.utils.blocked_cv import blocked_cv_select_alpha, blocked_cv_splits

from src.eval.data_utils import MIN_SAMPLES_FOR_DYNAMIC
from src.eval.strategy_config import (
    SOFTGATING_ALPHA_CANDIDATES,
    KG_COMPONENT_MODEL_POOL_MODE,
    DIRECT_WEIGHT_GATING_TEMPERATURE,
    SCENARIO_SIMILARITY_WEIGHT_MODE,
    SCENARIO_SIMILARITY_TEMPERATURE,
    DRIFT_DECAY_BASE,
    DRIFT_DECAY_SLOPE,
    DRIFT_DECAY_MAX,
)
from src.eval.dynamic_strategies import ScenarioGatingNetwork, ScenarioBucketSelector
from src.eval.metrics import compute_extreme_weights, evaluate_slices, compute_dynamic_cost, ROBUST_CFG
from src.eval.model_selection import select_models_by_history
from src.eval.combination_utils import fit_ridge_robust
from src.eval.strategy_naming import get_strategy_naming

EXPECTED_BASELINE_CLASSIC = [
    "simple_avg_safe",
    "static_weight_safe",
    "stacking_safe",
    "simple_avg",
    "static_weight",
    "stacking",
    "dynamic_avg",
    "dynamic_weight",
    "dynamic_stacking",
]
EXPECTED_BASELINE_SOTA = ["rl_qms", "mole_router"]
# 历史兼容：EXPECTED_KG_COMPONENT 保留原名称，语义上等同于 kg component 分支。
EXPECTED_KG_COMPONENT_ABLATION = [
    "gating_network",
    "soft_gating",
    "scenario_bucket",
    "gating_network_v2",
    "adaptive_bucket",
    "scenario_similarity",
]
EXPECTED_KG_COMPONENT = EXPECTED_KG_COMPONENT_ABLATION
EXPECTED_STRATEGIES = EXPECTED_BASELINE_CLASSIC + EXPECTED_BASELINE_SOTA + EXPECTED_KG_COMPONENT


def _parse_csv_env_list(env_name: str) -> List[str]:
    raw = os.environ.get(env_name, "")
    if not raw or not raw.strip():
        return []
    return [x.strip() for x in raw.split(",") if x and x.strip()]


def _run_baselines_classic(
    *,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    safe_cols: List[str],
    dynamic_cols: List[str],
    y_val: np.ndarray,
    sample_weight: np.ndarray | None,
    drift_events: List[Any] | None,
    drift_events_val: List[Any] | None,
    drift_events_test: List[Any] | None,
    record_failure,
) -> tuple[Dict[str, Dict], Dict[str, Any]]:
    baselines_classic: Dict[str, Dict] = {}

    if dynamic_cols:
        X_val_dyn = val_df[dynamic_cols].fillna(val_df[dynamic_cols].mean()).values
        X_test_dyn = test_df[dynamic_cols].fillna(test_df[dynamic_cols].mean()).values
        sa_val = np.nanmean(val_df[dynamic_cols].values, axis=1)
        sa_test = np.nanmean(test_df[dynamic_cols].values, axis=1)
        baselines_classic["dynamic_avg"] = {
            "val_pred": sa_val,
            "test_pred": sa_test,
            "weights": {m: 1 / len(dynamic_cols) for m in dynamic_cols},
            "selected_models": dynamic_cols,
            "category": "baseline_classic",
        }

        alpha_dyn, _ = blocked_cv_select_alpha(
            X_val_dyn, y_val,
            n_folds=3, min_train=50,
            positive=True, fit_intercept=False,
            sample_weight=sample_weight,
        )

        reg_sw_dyn, sw_dyn_solver_meta = fit_ridge_robust(
            X_val_dyn, y_val,
            alpha=alpha_dyn, positive=True, fit_intercept=False,
            sample_weight=sample_weight,
        )
        baselines_classic["dynamic_weight"] = {
            "val_pred": reg_sw_dyn.predict(X_val_dyn),
            "test_pred": reg_sw_dyn.predict(X_test_dyn),
            "weights": dict(zip(dynamic_cols, reg_sw_dyn.coef_)),
            "selected_models": dynamic_cols,
            "alpha": alpha_dyn,
            "solver_meta": sw_dyn_solver_meta,
            "category": "baseline_classic",
        }

        alpha_dyn_stack, _ = blocked_cv_select_alpha(
            X_val_dyn, y_val,
            n_folds=3, min_train=50,
            positive=True, fit_intercept=True,
            sample_weight=sample_weight,
        )

        reg_st_dyn, st_dyn_solver_meta = fit_ridge_robust(
            X_val_dyn, y_val,
            alpha=alpha_dyn_stack, positive=True, fit_intercept=True,
            sample_weight=sample_weight,
        )
        baselines_classic["dynamic_stacking"] = {
            "val_pred": reg_st_dyn.predict(X_val_dyn),
            "test_pred": reg_st_dyn.predict(X_test_dyn),
            "weights": dict(zip(dynamic_cols, reg_st_dyn.coef_)),
            "intercept": reg_st_dyn.intercept_,
            "selected_models": dynamic_cols,
            "alpha": alpha_dyn_stack,
            "solver_meta": st_dyn_solver_meta,
            "category": "baseline_classic",
        }
    else:
        record_failure("dynamic_avg", "no_models")
        record_failure("dynamic_weight", "no_models")
        record_failure("dynamic_stacking", "no_models")

    X_val_safe = val_df[safe_cols].fillna(val_df[safe_cols].mean()).values
    X_test_safe = test_df[safe_cols].fillna(test_df[safe_cols].mean()).values

    n_val = len(y_val)
    max_psi = 0.0
    drift_events_for_decay = drift_events_val if drift_events_val is not None else drift_events
    drift_event_count_val = len(drift_events_for_decay or [])
    drift_event_count_test = len(drift_events_test or [])
    for ev in (drift_events_for_decay or []):
        if isinstance(ev, dict):
            metric_type = str(ev.get("metric_type", "")).lower()
            value = ev.get("value", 0.0)
        else:
            metric_type = str(getattr(ev, "metric_type", "")).lower()
            value = getattr(ev, "value", 0.0)
        if metric_type != "psi":
            continue
        try:
            max_psi = max(max_psi, float(value))
        except Exception:
            continue

    if max_psi > 0:
        # 漂移越强，越强调近期样本；具体斜率/上限由 strategy_config 中常量控制。
        beta_decay = min(DRIFT_DECAY_MAX, DRIFT_DECAY_BASE + DRIFT_DECAY_SLOPE * max_psi)
        beta_decay_source = "drift_adaptive"
    else:
        beta_decay = DRIFT_DECAY_BASE
        beta_decay_source = "fixed"
    time_weights = np.exp(-beta_decay * np.arange(n_val - 1, -1, -1))
    time_weights = time_weights / time_weights.mean()

    sa_val_safe = np.nanmean(val_df[safe_cols].values, axis=1)
    sa_test_safe = np.nanmean(test_df[safe_cols].values, axis=1)
    baselines_classic["simple_avg_safe"] = {
        "val_pred": sa_val_safe,
        "test_pred": sa_test_safe,
        "weights": {m: 1 / len(safe_cols) for m in safe_cols},
        "selected_models": safe_cols,
        "category": "baseline_classic",
        "scope": "safe",
    }

    alpha_safe, _ = blocked_cv_select_alpha(
        X_val_safe, y_val,
        n_folds=3, min_train=50,
        positive=True, fit_intercept=False,
        sample_weight=time_weights,
    )

    reg_sw, sw_solver_meta = fit_ridge_robust(
        X_val_safe, y_val,
        alpha=alpha_safe, positive=True, fit_intercept=False,
        sample_weight=time_weights,
    )
    baselines_classic["static_weight_safe"] = {
        "val_pred": reg_sw.predict(X_val_safe),
        "test_pred": reg_sw.predict(X_test_safe),
        "weights": dict(zip(safe_cols, reg_sw.coef_)),
        "selected_models": safe_cols,
        "alpha": alpha_safe,
        "solver_meta": sw_solver_meta,
        "category": "baseline_classic",
        "scope": "safe",
        "ridge_decay_beta": beta_decay,
        "ridge_decay_source": beta_decay_source,
        "drift_max_psi": max_psi,
        "drift_event_count": drift_event_count_val,
        "drift_event_count_val": drift_event_count_val,
        "drift_event_count_test": drift_event_count_test,
    }

    alpha_safe_stack, _ = blocked_cv_select_alpha(
        X_val_safe, y_val,
        n_folds=3, min_train=50,
        positive=True, fit_intercept=True,
        sample_weight=time_weights,
    )

    reg_st, st_solver_meta = fit_ridge_robust(
        X_val_safe, y_val,
        alpha=alpha_safe_stack, positive=True, fit_intercept=True,
        sample_weight=time_weights,
    )
    baselines_classic["stacking_safe"] = {
        "val_pred": reg_st.predict(X_val_safe),
        "test_pred": reg_st.predict(X_test_safe),
        "weights": dict(zip(safe_cols, reg_st.coef_)),
        "intercept": reg_st.intercept_,
        "selected_models": safe_cols,
        "alpha": alpha_safe_stack,
        "solver_meta": st_solver_meta,
        "category": "baseline_classic",
        "scope": "safe",
    }

    # 兼容旧策略名
    baselines_classic["simple_avg"] = {**baselines_classic["simple_avg_safe"], "legacy_alias": True}
    baselines_classic["static_weight"] = {**baselines_classic["static_weight_safe"], "legacy_alias": True}
    baselines_classic["stacking"] = {**baselines_classic["stacking_safe"], "legacy_alias": True}

    drift_meta = {
        "drift_event_count": drift_event_count_val,
        "drift_event_count_val": drift_event_count_val,
        "drift_event_count_test": drift_event_count_test,
        "drift_max_psi": max_psi,
        "ridge_decay_beta": beta_decay,
        "ridge_decay_source": beta_decay_source,
    }
    return baselines_classic, drift_meta


def _run_baselines_sota(
    *,
    safe_cols: List[str],
    P_val: np.ndarray,
    P_test: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    val_df_filled: pd.DataFrame,
    test_df_filled: pd.DataFrame,
    model_cols: List[str],
    horizon: int,
    naive_scale: float | None,
    structural_baseline_val: np.ndarray | None,
    structural_baseline_test: np.ndarray | None,
    structural_baseline_name: str,
    record_failure,
) -> Dict[str, Dict]:
    baselines_sota: Dict[str, Dict] = {}

    try:
        print("    [perf] 训练 SOTA: rl_qms")
        rl_val = RLQMSStrategy(
            Nq=72, Nsp=4, alpha=0.1, gamma=0.8, Ne=100, em_metric="ape",
            seed=42, warm_start_from_val=False,
            active_models=safe_cols,
            switch_penalty=0.5,
            test_epsilon=0.0,
        )
        rl_val.fit(P_val, y_val, model_names=model_cols)
        out_val = rl_val.predict(P_val, y_test=y_val, model_names=model_cols)

        rl = RLQMSStrategy(
            Nq=72, Nsp=4, alpha=0.1, gamma=0.8, Ne=100, em_metric="ape",
            seed=42, warm_start_from_val=True,
            active_models=safe_cols,
            switch_penalty=0.5,
            test_epsilon=0.0,
        )
        rl.fit(P_val, y_val, model_names=model_cols)
        out_test = rl.predict(P_test, y_test=y_test, model_names=model_cols)
        chosen_test = out_test.chosen_models

        baselines_sota["rl_qms"] = {
            "val_pred": out_val.y_pred_test,
            "test_pred": out_test.y_pred_test,
            "weights": "per_step_one_hot",
            "avg_models_used": 1.0,
            "active_models": safe_cols,
            "meta": out_test.meta,
            "chosen_models": chosen_test,
            "weights_log": out_test.weights_log,
            "category": "baseline_sota",
            "description": "RL-QMS (IEEE TSG 2020)：Q-learning 动态选择单模型",
        }
    except Exception as e:
        print(f"    [rl_qms] 失败: {e}")
        record_failure("rl_qms", e)

    try:
        print("    [perf] 训练 SOTA: mole_router")
        mole = MoLERouterStrategy(
            hidden_dim=16,
            epochs=200,
            lr=1e-2,
            batch_size=256,
            head_dropout=0.1,
            seed=42,
            active_models=safe_cols,
            temperature=1.5,
            temporal_holdout=0.2,
            weight_clip=0.8,
        )
        mole.fit(P_val, y_val, ctx_val=val_df_filled, model_names=model_cols)
        out_val = mole.predict(P_val, y_test=y_val, ctx_test=val_df_filled, model_names=model_cols)
        out_test = mole.predict(P_test, y_test=y_test, ctx_test=test_df_filled, model_names=model_cols)
        chosen_test = np.argmax(out_test.weights_log, axis=1) if out_test.weights_log is not None else None

        baselines_sota["mole_router"] = {
            "val_pred": out_val.y_pred_test,
            "test_pred": out_test.y_pred_test,
            "weights": "per_sample_softmax",
            "avg_models_used": float(len(safe_cols)),
            "active_models": safe_cols,
            "meta": out_test.meta,
            "weights_log": out_test.weights_log,
            "chosen_models": chosen_test,
            "category": "baseline_sota",
            "description": "MoLE Router (ICML 2024)：MLP 根据时间特征预测软权重",
        }
    except Exception as e:
        print(f"    [mole_router] 失败: {e}")
        record_failure("mole_router", e)

    return baselines_sota


def _run_dash_tta_node(
    *,
    safe_cols: List[str],
    P_val: np.ndarray,
    P_test: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    val_df_filled: pd.DataFrame,
    test_df_filled: pd.DataFrame,
    model_cols: List[str],
    horizon: int,
    naive_scale: float | None,
    structural_baseline_val: np.ndarray | None,
    structural_baseline_test: np.ndarray | None,
    structural_baseline_name: str,
    record_failure,
) -> Dict[str, Dict]:
    """DASH-TTA 仅作为 KG 扩展节点候选导出，不纳入 SOTA 对比集合。"""
    baselines_node: Dict[str, Dict] = {}
    try:
        print("    [perf] 训练 KG node: dash_tta")
        delay = max(1, int(horizon))
        scale = float(naive_scale) if naive_scale is not None else 1.0
        # Fix4: horizon_adaptive=True 自动按 delay 调整 window 和 fallback_delta
        dash_val = DASHTTAStrategy(
            delay=delay,
            scale=scale,
            active_models=safe_cols,
            baseline_mode="simple_avg",
            horizon_adaptive=True,
        )
        dash_val.fit(P_val, y_val, ctx_val=val_df_filled, model_names=model_cols)
        out_val = dash_val.predict(
            P_val,
            y_test=y_val,
            ctx_test=val_df_filled,
            model_names=model_cols,
            baseline_series=structural_baseline_val,
        )

        dash = DASHTTAStrategy(
            delay=delay,
            scale=scale,
            active_models=safe_cols,
            baseline_mode="simple_avg",
            horizon_adaptive=True,
        )
        dash.fit(P_val, y_val, ctx_val=val_df_filled, model_names=model_cols)
        out_test = dash.predict(
            P_test,
            y_test=y_test,
            ctx_test=test_df_filled,
            model_names=model_cols,
            baseline_series=structural_baseline_test,
        )

        baselines_node["dash_tta"] = {
            "val_pred": out_val.y_pred_test,
            "test_pred": out_test.y_pred_test,
            "weights": "per_step_dynamic_with_structural_baseline",
            "avg_models_used": float(len(safe_cols)),
            "active_models": safe_cols,
            "meta": out_test.meta,
            "chosen_models": out_test.chosen_models,
            "weights_log": out_test.weights_log,
            "category": "kg_node",
            "description": (
                "DASH-TTA v2（KG node only）：延迟反馈 + 漂移门控 + 结构化兜底"
                f"（baseline={structural_baseline_name}）"
            ),
        }
    except Exception as e:
        print(f"    [dash_tta] 失败: {e}")
        record_failure("dash_tta", e)
    return baselines_node


def _run_kg_component_strategies(
    *,
    skip_complex_kg_component: bool,
    allowed_kg_component_strategies: List[str],
    model_cols: List[str],
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df_filled: pd.DataFrame,
    test_df_filled: pd.DataFrame,
    train_df_filled: pd.DataFrame | None,
    y_val: np.ndarray,
    kg_component_active_cols: List[str],
    baselines_classic: Dict[str, Dict],
    robust_cfg: Dict[str, float],
    naive_scale: float | None,
    horizon: int,
    threshold_ratio: float,
    alpha_fallback: float,
    record_failure,
) -> Dict[str, Dict]:
    kg_components: Dict[str, Dict] = {}

    if skip_complex_kg_component:
        return kg_components

    def _resolve_oof_cv(n_samples: int) -> tuple[int, int, int]:
        gap = max(int(horizon), 1)
        if n_samples < 1000:
            return 2, min(300, max(80, n_samples // 3)), gap
        return 3, min(200, max(80, n_samples // 4)), gap

    def _compute_blocked_oof(builder, fallback_pred: np.ndarray) -> tuple[np.ndarray | None, float]:
        n = len(val_df_filled)
        n_folds, min_train, gap = _resolve_oof_cv(n)
        splits = blocked_cv_splits(n, n_folds=n_folds, min_train=min_train, gap=gap)
        if not splits:
            return None, 0.0
        oof_pred = np.full(n, np.nan)
        for train_idx, val_idx in splits:
            if len(val_idx) == 0:
                continue
            try:
                fold_train = val_df_filled.iloc[train_idx].copy()
                fold_val = val_df_filled.iloc[val_idx].copy()
                fold_pred = builder(fold_train, fold_val)
                if fold_pred is None:
                    continue
                fold_pred = np.asarray(fold_pred, dtype=float)
                if len(fold_pred) != len(val_idx):
                    continue
                oof_pred[val_idx] = fold_pred
            except Exception:
                continue
        valid_mask = np.isfinite(oof_pred)
        coverage = float(valid_mask.mean()) if n > 0 else 0.0
        if valid_mask.sum() == 0:
            return None, coverage
        oof_pred[~valid_mask] = np.asarray(fallback_pred, dtype=float)[~valid_mask]
        return oof_pred, coverage

    if "gating_network" in allowed_kg_component_strategies:
        try:
            gating = ScenarioGatingNetwork(
                model_cols,
                method="ridge",
                top_m=min(3, len(kg_component_active_cols)),
                robust_cfg=robust_cfg,
                active_models=kg_component_active_cols,
                corr_penalty_enabled=True,
                corr_penalty_scale=0.2,
            )
            gating.fit(val_df_filled)
            gn_val, _ = gating.predict(val_df_filled)
            gn_test, test_used = gating.predict(test_df_filled)
            gn_fallback = np.nanmean(val_df_filled[kg_component_active_cols].values, axis=1)
            gn_oof, gn_oof_cov = _compute_blocked_oof(
                lambda tr, va: (
                    (lambda _g: (_g.fit(tr), _g.predict(va)[0]))(
                        ScenarioGatingNetwork(
                            model_cols,
                            method="ridge",
                            top_m=min(3, len(kg_component_active_cols)),
                            robust_cfg=robust_cfg,
                            active_models=kg_component_active_cols,
                            corr_penalty_enabled=True,
                            corr_penalty_scale=0.2,
                        )
                    )[1]
                ),
                gn_fallback,
            )
            payload = {
                "val_pred": gn_val,
                "test_pred": gn_test,
                "weights": "per_sample_dynamic",
                "avg_models_used": float(test_used),
                "active_models": kg_component_active_cols,
                "category": "kg_component",
                "description": "场景门控网络：基于场景特征预测模型误差，误差倒数加权",
            }
            if gn_oof is not None and gn_oof_cov > 0:
                payload["val_pred_oof"] = gn_oof
                payload["val_oof_coverage"] = gn_oof_cov
            kg_components["gating_network"] = payload
        except Exception as e:
            print(f"    [gating_network] 失败: {e}")
            record_failure("gating_network", e)

    if "soft_gating" in allowed_kg_component_strategies:
        try:
            static_test_pred = baselines_classic.get("static_weight_safe", {}).get("test_pred")
            gn_test_pred = kg_components.get("gating_network", {}).get("test_pred")
            oof_dynamic = np.full(len(y_val), np.nan)
            oof_static = np.full(len(y_val), np.nan)

            if static_test_pred is not None and gn_test_pred is not None:
                _sg_n_folds, _sg_min_train, _sg_gap = _resolve_oof_cv(len(y_val))
                sg_folds = blocked_cv_splits(len(y_val), n_folds=_sg_n_folds, min_train=max(30, _sg_min_train), gap=_sg_gap)
                _oof_fold_safe_sets = []

                if sg_folds:
                    for fold_train_idx, fold_val_idx in sg_folds:
                        _fold_train_raw = val_df.iloc[fold_train_idx].copy()
                        _fold_val_raw = val_df.iloc[fold_val_idx].copy()
                        for _m in model_cols:
                            if _m in _fold_train_raw.columns:
                                _ft_mean = _fold_train_raw[_m].mean()
                                _ft_mean = _ft_mean if not np.isnan(_ft_mean) else 0.0
                                _fold_train_raw[_m] = _fold_train_raw[_m].fillna(_ft_mean)
                                _fold_val_raw[_m] = _fold_val_raw[_m].fillna(_ft_mean)
                        _fold_train_df = _fold_train_raw
                        _fold_val_df = _fold_val_raw
                        y_ft = y_val[fold_train_idx]

                        _fold_scores = {m: float(np.mean(np.abs(
                            _fold_train_df["y"].values - _fold_train_df[m].values
                        ))) for m in model_cols}
                        _fold_best_mae = min(_fold_scores.values()) if _fold_scores else 0.0
                        _fold_mase = {}
                        if naive_scale is not None and naive_scale > 1e-8:
                            _fold_mase = {m: mae / naive_scale for m, mae in _fold_scores.items()}
                        _fold_safe = [
                            m for m, mae in _fold_scores.items()
                            if mae <= _fold_best_mae * threshold_ratio
                            and _fold_mase.get(m, 0) < 1.0
                        ]
                        if not _fold_safe:
                            _fold_safe = [m for m, mae in _fold_scores.items()
                                          if mae <= _fold_best_mae * threshold_ratio]
                        if not _fold_safe:
                            _fold_safe = model_cols
                        _fold_scene_fit = _fold_safe
                        _oof_fold_safe_sets.append(set(_fold_safe))

                        X_ft_safe = _fold_train_df[_fold_safe].fillna(0).values
                        try:
                            _fold_alpha, _ = blocked_cv_select_alpha(
                                X_ft_safe, y_ft,
                                n_folds=2, min_train=20,
                                positive=True, fit_intercept=False,
                            )
                        except Exception:
                            _fold_alpha = alpha_fallback

                        X_fv_safe = _fold_val_df[_fold_safe].fillna(0).values

                        try:
                            _fold_reg, _ = fit_ridge_robust(
                                X_ft_safe, y_ft,
                                alpha=_fold_alpha, positive=True, fit_intercept=False,
                            )
                            oof_static[fold_val_idx] = _fold_reg.predict(X_fv_safe)
                        except Exception:
                            oof_static[fold_val_idx] = np.nanmean(
                                _fold_val_df[_fold_safe].values, axis=1)

                        try:
                            _fold_gn = ScenarioGatingNetwork(
                                model_cols, method="ridge",
                                top_m=min(3, len(_fold_scene_fit)),
                                robust_cfg=robust_cfg, active_models=_fold_scene_fit
                            )
                            _fold_gn.fit(_fold_train_df)
                            _fold_pred, _ = _fold_gn.predict(_fold_val_df)
                            oof_dynamic[fold_val_idx] = _fold_pred
                        except Exception:
                            oof_dynamic[fold_val_idx] = np.nanmean(
                                _fold_val_df[_fold_safe].values, axis=1)

                    oof_mask = ~np.isnan(oof_dynamic) & ~np.isnan(oof_static)
                    if oof_mask.sum() > 20:
                        best_alpha = 0.5
                        best_rmse = float('inf')
                        for a in SOFTGATING_ALPHA_CANDIDATES:
                            blended = (1 - a) * oof_dynamic[oof_mask] + a * oof_static[oof_mask]
                            rmse_a = float(np.sqrt(np.mean((y_val[oof_mask] - blended) ** 2)))
                            if rmse_a < best_rmse:
                                best_rmse = rmse_a
                                best_alpha = a
                    else:
                        best_alpha = 0.5
                else:
                    best_alpha = 0.5

                if _oof_fold_safe_sets:
                    _final_pool = set(kg_component_active_cols)
                    _oof_union = set().union(*_oof_fold_safe_sets)
                    _jaccard = (len(_final_pool & _oof_union) /
                                max(len(_final_pool | _oof_union), 1))
                    if _jaccard < 0.3:
                        _shrink = 0.3
                        _old_alpha = best_alpha
                        best_alpha = best_alpha * (1 - _shrink) + 0.5 * _shrink
                        print(f"    [SoftGating] OOF/final 模型池 Jaccard={_jaccard:.2f}<0.3, "
                              f"alpha {_old_alpha:.2f}→{best_alpha:.2f} (向0.5收缩)")
                    elif _jaccard < 0.6:
                        _shrink = 0.1
                        _old_alpha = best_alpha
                        best_alpha = best_alpha * (1 - _shrink) + 0.5 * _shrink
                        print(f"    [SoftGating] OOF/final 模型池 Jaccard={_jaccard:.2f}<0.6, "
                              f"alpha {_old_alpha:.2f}→{best_alpha:.2f} (轻收缩)")

                gn_val_pred_full = kg_components.get("gating_network", {}).get("val_pred")
                static_val_pred_full = baselines_classic.get("static_weight_safe", {}).get("val_pred")
                sg_val = (1 - best_alpha) * gn_val_pred_full + best_alpha * static_val_pred_full
                sg_test = (1 - best_alpha) * gn_test_pred + best_alpha * static_test_pred

                payload = {
                    "val_pred": sg_val,
                    "test_pred": sg_test,
                    "weights": f"soft_blend_alpha={best_alpha:.2f}",
                    "alpha_static": best_alpha,
                    "category": "kg_component",
                    "description": f"SoftGating(OOF): (1-{best_alpha:.2f})*dynamic + {best_alpha:.2f}*static",
                }
                oof_mask = ~np.isnan(oof_dynamic) & ~np.isnan(oof_static)
                if oof_mask.sum() > 0:
                    sg_oof = np.full(len(y_val), np.nan)
                    sg_oof[oof_mask] = (1 - best_alpha) * oof_dynamic[oof_mask] + best_alpha * oof_static[oof_mask]
                    fallback = (1 - best_alpha) * np.nanmean(val_df_filled[kg_component_active_cols].values, axis=1) + best_alpha * static_val_pred_full
                    sg_oof[np.isnan(sg_oof)] = fallback[np.isnan(sg_oof)]
                    payload["val_pred_oof"] = sg_oof
                    payload["val_oof_coverage"] = float(oof_mask.mean())
                kg_components["soft_gating"] = payload
            else:
                record_failure("soft_gating", "missing_dependencies")
        except Exception as e:
            print(f"    [soft_gating] 失败: {e}")
            record_failure("soft_gating", e)

    if "scenario_bucket" in allowed_kg_component_strategies:
        try:
            bucket_sel = ScenarioBucketSelector(
                model_cols,
                min_bucket_size=30,
                robust_cfg=robust_cfg,
                active_models=kg_component_active_cols,
                top_k_models=3,
                use_soft_fusion=True,
                corr_penalty_enabled=True,
                corr_penalty_scale=0.2,
            )
            bucket_sel.fit(val_df_filled)
            sb_val, _ = bucket_sel.predict(val_df_filled)
            sb_test, test_avg_models = bucket_sel.predict(test_df_filled)
            sb_fallback = np.nanmean(val_df_filled[kg_component_active_cols].values, axis=1)
            sb_oof, sb_oof_cov = _compute_blocked_oof(
                lambda tr, va: (
                    (lambda _b: (_b.fit(tr), _b.predict(va)[0]))(
                        ScenarioBucketSelector(
                            model_cols,
                            min_bucket_size=30,
                            robust_cfg=robust_cfg,
                            active_models=kg_component_active_cols,
                            top_k_models=3,
                            use_soft_fusion=True,
                            corr_penalty_enabled=True,
                            corr_penalty_scale=0.2,
                        )
                    )[1]
                ),
                sb_fallback,
            )
            payload = {
                "val_pred": sb_val,
                "test_pred": sb_test,
                "weights": "per_bucket_dynamic",
                "avg_models_used": test_avg_models,
                "active_models": kg_component_active_cols,
                "bucket_info": {
                    "n_buckets": len(bucket_sel.bucket_best_models),
                    "buckets": list(bucket_sel.bucket_best_models.keys())[:10],
                },
                "category": "kg_component",
                "description": "场景分桶：按时段/周末/节假日/负荷模式分桶学习独立权重",
            }
            if sb_oof is not None and sb_oof_cov > 0:
                payload["val_pred_oof"] = sb_oof
                payload["val_oof_coverage"] = sb_oof_cov
            kg_components["scenario_bucket"] = payload
        except Exception as e:
            print(f"    [scenario_bucket] 失败: {e}")
            record_failure("scenario_bucket", e)

    if "gating_network_v2" in allowed_kg_component_strategies:
        try:
            gating_v2 = DirectWeightGatingNetwork(
                model_cols=model_cols,
                temperature=DIRECT_WEIGHT_GATING_TEMPERATURE,
                cv_folds=3,
                use_enhanced_features=True,
                active_models=kg_component_active_cols,
            )
            gating_v2.fit(val_df_filled)
            gn2_val, _ = gating_v2.predict(val_df_filled)
            gn2_test, test_used = gating_v2.predict(test_df_filled)
            gn2_fallback = np.nanmean(val_df_filled[kg_component_active_cols].values, axis=1)
            gn2_oof, gn2_oof_cov = _compute_blocked_oof(
                lambda tr, va: (
                    (lambda _g2: (_g2.fit(tr), _g2.predict(va)[0]))(
                        DirectWeightGatingNetwork(
                            model_cols=model_cols,
                            temperature=DIRECT_WEIGHT_GATING_TEMPERATURE,
                            cv_folds=3,
                            use_enhanced_features=True,
                            active_models=kg_component_active_cols,
                        )
                    )[1]
                ),
                gn2_fallback,
            )
            payload = {
                "val_pred": gn2_val,
                "test_pred": gn2_test,
                "weights": "per_sample_dynamic",
                "avg_models_used": float(test_used),
                "active_models": kg_component_active_cols,
                "best_alpha": gating_v2._best_alpha,
                "n_features": len(gating_v2.ctx_cols),
                "category": "kg_component",
                "description": "场景门控网络V2：直接预测权重 + 增强场景特征 + CV正则化",
            }
            if gn2_oof is not None and gn2_oof_cov > 0:
                payload["val_pred_oof"] = gn2_oof
                payload["val_oof_coverage"] = gn2_oof_cov
            kg_components["gating_network_v2"] = payload
        except Exception as e:
            print(f"    [gating_network_v2] 失败: {e}")
            record_failure("gating_network_v2", e)

    if "adaptive_bucket" in allowed_kg_component_strategies:
        try:
            adaptive_bucket = AdaptiveBucketSelector(
                model_cols=model_cols,
                max_depth=4,
                min_bucket_size=50,
                use_enhanced_features=True,
                active_models=kg_component_active_cols,
            )
            adaptive_bucket.fit(val_df_filled)
            ab_val, _ = adaptive_bucket.predict(val_df_filled)
            ab_test, test_avg = adaptive_bucket.predict(test_df_filled)
            ab_fallback = np.nanmean(val_df_filled[kg_component_active_cols].values, axis=1)
            ab_oof, ab_oof_cov = _compute_blocked_oof(
                lambda tr, va: (
                    (lambda _ab: (_ab.fit(tr), _ab.predict(va)[0]))(
                        AdaptiveBucketSelector(
                            model_cols=model_cols,
                            max_depth=4,
                            min_bucket_size=50,
                            use_enhanced_features=True,
                            active_models=kg_component_active_cols,
                        )
                    )[1]
                ),
                ab_fallback,
            )
            payload = {
                "val_pred": ab_val,
                "test_pred": ab_test,
                "weights": "per_bucket_dynamic",
                "avg_models_used": test_avg,
                "active_models": kg_component_active_cols,
                "n_buckets": len(adaptive_bucket.bucket_weights),
                "n_features": len(adaptive_bucket.ctx_cols),
                "category": "kg_component",
                "description": "自适应分桶：决策树学习场景边界 + 增强场景特征",
            }
            if ab_oof is not None and ab_oof_cov > 0:
                payload["val_pred_oof"] = ab_oof
                payload["val_oof_coverage"] = ab_oof_cov
            kg_components["adaptive_bucket"] = payload
        except Exception as e:
            print(f"    [adaptive_bucket] 失败: {e}")
            record_failure("adaptive_bucket", e)

    if "scenario_similarity" in allowed_kg_component_strategies:
        try:
            similarity = ScenarioSimilarityEnhancer(
                model_cols=model_cols,
                n_neighbors=10,
                temperature=SCENARIO_SIMILARITY_TEMPERATURE,
                use_enhanced_features=True,
                active_models=kg_component_active_cols,
                weight_mode=SCENARIO_SIMILARITY_WEIGHT_MODE,
            )
            if train_df_filled is not None and len(train_df_filled) > 10:
                similarity.fit(train_df_filled)
            else:
                print(f"    [scenario_similarity] 警告: 无训练集，使用 val fit（可能有泄露）")
                similarity.fit(val_df_filled)

            ss_val, _ = similarity.predict(val_df_filled)
            ss_test, test_avg = similarity.predict(test_df_filled)
            ss_fallback = np.nanmean(val_df_filled[kg_component_active_cols].values, axis=1)
            ss_oof, ss_oof_cov = _compute_blocked_oof(
                lambda tr, va: (
                    (lambda _ss: (_ss.fit(tr), _ss.predict(va, is_val_prediction=False)[0]))(
                        ScenarioSimilarityEnhancer(
                            model_cols=model_cols,
                            n_neighbors=10,
                            temperature=SCENARIO_SIMILARITY_TEMPERATURE,
                            use_enhanced_features=True,
                            active_models=kg_component_active_cols,
                            weight_mode=SCENARIO_SIMILARITY_WEIGHT_MODE,
                        )
                    )[1]
                ),
                ss_fallback,
            )
            payload = {
                "val_pred": ss_val,
                "test_pred": ss_test,
                "weights": "per_sample_dynamic",
                "avg_models_used": test_avg,
                "active_models": kg_component_active_cols,
                "n_neighbors": similarity.n_neighbors,
                "n_features": len(similarity.ctx_cols),
                "fit_on": "train" if train_df_filled is not None else "val",
                "category": "kg_component",
                "description": "场景相似度增强：kNN匹配训练集相似场景权重（无泄露）",
            }
            if ss_oof is not None and ss_oof_cov > 0:
                payload["val_pred_oof"] = ss_oof
                payload["val_oof_coverage"] = ss_oof_cov
            kg_components["scenario_similarity"] = payload
        except Exception as e:
            print(f"    [scenario_similarity] 失败: {e}")
            record_failure("scenario_similarity", e)

    return kg_components


def combine_predictions(val_df: pd.DataFrame, test_df: pd.DataFrame, model_cols: List[str],
                        dynamic_select: bool = True, select_strategy: str = "top_k",
                        top_k: int = 3, threshold_ratio: float = 2.0,
                        robust_cfg: Dict[str, float] | None = None,
                        naive_scale: float = None, horizon: int = 1,
                        dataset_name: str = None,
                        train_df: pd.DataFrame = None,
                        health_enabled_models: List[str] = None,
                        health_scene_fit_models: List[str] = None,
                        drift_events: List[Any] | None = None,
                        drift_events_val: List[Any] | None = None,
                        drift_events_test: List[Any] | None = None,
                        structural_baseline_val: np.ndarray | None = None,
                        structural_baseline_test: np.ndarray | None = None,
                        structural_baseline_name: str = "seasonal_naive") -> Dict[str, Dict]:
    """
    Calibrate weights on val, apply to test.
    """
    robust_cfg = robust_cfg or ROBUST_CFG
    failed_strategies = []

    def _record_failure(name: str, err: Exception | str) -> None:
        failed_strategies.append({
            "strategy": name,
            "reason": str(err),
        })

    if not model_cols:
        for strat in EXPECTED_STRATEGIES:
            _record_failure(strat, "no_models")
        return {
            "_meta": {
                "baselines_classic": [],
                "baselines_sota": [],
                "kg_component": [],
                "skipped_complex_kg_component": True,
                "recommended_fallback": None,
                "fallback_ratio": 1.0,
                "failed_strategies": failed_strategies,
                "skip_reason": "no_models",
            }
        }

    # 对模型列做列均值填充
    val_df_filled = val_df.copy()
    test_df_filled = test_df.copy()
    train_df_filled = train_df.copy() if train_df is not None else None

    for m in model_cols:
        col_mean_val = val_df_filled[m].mean() if m in val_df_filled else 0.0
        col_mean_test = col_mean_val
        if m in val_df_filled:
            val_df_filled[m] = val_df_filled[m].fillna(col_mean_val)
        if m in test_df_filled:
            test_df_filled[m] = test_df_filled[m].fillna(col_mean_test)
        if train_df_filled is not None and m in train_df_filled:
            col_mean_train = train_df_filled[m].mean() if m in train_df_filled else col_mean_val
            train_df_filled[m] = train_df_filled[m].fillna(col_mean_train)

    model_scores = {m: mean_absolute_error(val_df_filled["y"].values, val_df_filled[m].values) for m in model_cols}

    if dynamic_select and len(model_cols) > 1:
        selected_cols, _selection_scores = select_models_by_history(
            val_df_filled, model_cols,
            strategy=select_strategy,
            top_k=top_k,
            threshold_ratio=threshold_ratio,
            robust_cfg=robust_cfg
        )
    else:
        selected_cols = model_cols

    best_mae = min(model_scores.values()) if model_scores else 0.0

    model_mase = {}
    if naive_scale is not None and naive_scale > 1e-8:
        model_mase = {m: mae / naive_scale for m, mae in model_scores.items()}

    mase_threshold = 1.0
    safe_cols = [
        m for m, mae in model_scores.items()
        if mae <= best_mae * threshold_ratio
        and model_mase.get(m, 0) < mase_threshold
    ]
    n_val_rows = len(val_df_filled)
    mase_relaxed_threshold = mase_threshold
    if n_val_rows < 1000 and len(safe_cols) < 3 and model_mase:
        mase_relaxed_threshold = mase_threshold * 1.5
        relaxed_cols = [
            m for m, mae in model_scores.items()
            if mae <= best_mae * threshold_ratio
            and model_mase.get(m, 0) < mase_relaxed_threshold
        ]
        if len(relaxed_cols) > len(safe_cols):
            safe_cols = relaxed_cols
            print(
                "    [safe_cols] 小样本放宽 MASE 阈值: "
                f"{mase_threshold:.2f} -> {mase_relaxed_threshold:.2f}, "
                f"safe_models={len(safe_cols)}"
            )

    if not safe_cols:
        safe_cols = [m for m, mae in model_scores.items() if mae <= best_mae * threshold_ratio]
    if not safe_cols:
        safe_cols = model_cols

    if health_enabled_models is not None:
        health_safe = [m for m in safe_cols if m in health_enabled_models]
        disabled_by_health = [m for m in safe_cols if m not in health_enabled_models]
        if health_safe:
            if disabled_by_health:
                print(f"    [P1.1 健康诊断] 下线模型: {disabled_by_health}")
            safe_cols = health_safe
        else:
            print(f"    [P1.1 健康诊断] 警告: 健康诊断排除了所有模型，回退到原 safe_cols")

    scene_fit_cols = safe_cols
    if health_scene_fit_models is not None:
        scene_fit_safe = [m for m in safe_cols if m in health_scene_fit_models]
        scene_limited = [m for m in safe_cols if m not in health_scene_fit_models]
        if scene_fit_safe:
            scene_fit_cols = scene_fit_safe
            if scene_limited:
                print(f"    [P2.2 场景路由] 动态策略排除 scene_limited: {scene_limited}")
        else:
            print(f"    [P2.2 场景路由] 警告: scene_fit 为空，动态策略回退到 safe_cols")

    # kg_component 策略模型池可配置：
    # - safe: 与 static/stacking 对齐（默认）
    # - scene_fit: 使用场景受限后的模型池
    if KG_COMPONENT_MODEL_POOL_MODE == "scene_fit":
        kg_component_active_cols = scene_fit_cols
    else:
        kg_component_active_cols = safe_cols

    if not kg_component_active_cols:
        kg_component_active_cols = safe_cols if safe_cols else model_cols

    if KG_COMPONENT_MODEL_POOL_MODE == "safe" and set(kg_component_active_cols) != set(scene_fit_cols):
        print("    [kg_component_pool] 使用 safe_cols 作为动态模型池（覆盖 scene_fit 限制）")

    excluded_by_mase = [
        m for m in model_cols
        if m not in safe_cols and model_mase.get(m, 0) >= mase_relaxed_threshold
    ]
    if excluded_by_mase:
        print(f"    [safe_cols] MASE>={mase_relaxed_threshold:.2f} 排除: {excluded_by_mase}")

    y_val = val_df_filled["y"].values
    y_test = test_df_filled["y"].values
    P_val = val_df_filled[model_cols].values
    P_test = test_df_filled[model_cols].values

    dynamic_cols = [m for m in selected_cols if m in safe_cols]
    if not dynamic_cols:
        dynamic_cols = safe_cols
        if dynamic_cols:
            print("    [dynamic] 选择列为空，回退到 safe_cols")

    sample_weight = None
    if robust_cfg.get("enable", False) and len(dynamic_cols) > 0:
        base_pred = np.nanmean(val_df_filled[dynamic_cols].values, axis=1)
        abs_err = np.abs(y_val - base_pred)
        sample_weight = compute_extreme_weights(y_val, abs_err, robust_cfg)

    results = {}
    baselines_classic = {}
    baselines_sota = {}
    kg_components = {}

    alpha_fallback = 100.0 if horizon >= 14 else (10.0 if horizon >= 7 else 1.0)

    baselines_classic, drift_meta = _run_baselines_classic(
        val_df=val_df,
        test_df=test_df,
        safe_cols=safe_cols,
        dynamic_cols=dynamic_cols,
        y_val=y_val,
        sample_weight=sample_weight,
        drift_events=drift_events,
        drift_events_val=drift_events_val,
        drift_events_test=drift_events_test,
        record_failure=_record_failure,
    )
    drift_event_count_val = drift_meta["drift_event_count_val"]
    drift_event_count_test = drift_meta["drift_event_count_test"]
    max_psi = drift_meta["drift_max_psi"]
    beta_decay = drift_meta["ridge_decay_beta"]
    beta_decay_source = drift_meta["ridge_decay_source"]

    ctx_cols = [c for c in val_df_filled.columns if c.startswith("ctx_")]
    n_ctx_features = len(ctx_cols)
    n_samples = len(val_df_filled)

    ds_min_samples = MIN_SAMPLES_FOR_DYNAMIC.get(dataset_name, 500)

    force_skip_complex = os.environ.get("MODELCOMBINE_SKIP_COMPLEX_KG_COMPONENT", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    fallback_strategy = "static_weight_safe"
    allowed_kg_component_strategies: List[str] = []
    use_complex = False
    if not force_skip_complex:
        use_complex = n_samples >= ds_min_samples and n_ctx_features >= 2
        allowed_kg_component_strategies = DEFAULT_KG_COMPONENT_STRATEGIES.copy() if use_complex else []
        if use_complex and 100 <= n_samples < 500 and n_ctx_features >= 2:
            # 小样本即便通过了数据集阈值，也需要确认是否仍启用复杂策略。
            _use, _fallback, _allowed = should_use_optimized_strategy(
                n_samples, n_ctx_features, dataset_name=dataset_name
            )
            if not _use:
                use_complex = False
                fallback_strategy = _fallback
                allowed_kg_component_strategies = []
            elif _allowed:
                allowed_kg_component_strategies = _allowed
        elif not use_complex:
            use_complex, fallback_strategy, allowed_kg_component_strategies = should_use_optimized_strategy(
                n_samples, n_ctx_features, dataset_name=dataset_name
            )

    skip_complex_kg_component = (not use_complex) or force_skip_complex
    if force_skip_complex:
        fallback_strategy = "static_weight_safe"
        allowed_kg_component_strategies = []
        print("    [perf] MODELCOMBINE_SKIP_COMPLEX_KG_COMPONENT=1，跳过复杂 kg_component 策略")
    if not skip_complex_kg_component:
        allow_kg_component = _parse_csv_env_list("MODELCOMBINE_KG_COMPONENT_STRATEGIES")
        disable_kg_component = set(_parse_csv_env_list("MODELCOMBINE_DISABLE_KG_COMPONENT_STRATEGIES"))
        if allow_kg_component:
            allow_set = {s for s in allow_kg_component if s in DEFAULT_KG_COMPONENT_STRATEGIES}
            before = list(allowed_kg_component_strategies)
            allowed_kg_component_strategies = [s for s in allowed_kg_component_strategies if s in allow_set]
            if before != allowed_kg_component_strategies:
                print(f"    [perf] MODELCOMBINE_KG_COMPONENT_STRATEGIES 生效: {allowed_kg_component_strategies}")
        if disable_kg_component:
            before = list(allowed_kg_component_strategies)
            allowed_kg_component_strategies = [s for s in allowed_kg_component_strategies if s not in disable_kg_component]
            if before != allowed_kg_component_strategies:
                print(f"    [perf] MODELCOMBINE_DISABLE_KG_COMPONENT_STRATEGIES 生效: 移除 {sorted(disable_kg_component)}")
        if not allowed_kg_component_strategies:
            skip_complex_kg_component = True
            fallback_strategy = "static_weight_safe"
            print("    [perf] kg_component 策略列表为空，自动跳过复杂 kg_component")
    if skip_complex_kg_component:
        print(f"    [场景降级] 样本量不足 ({n_samples}), 跳过本项目复杂策略，建议回退策略: {fallback_strategy}")
    elif set(allowed_kg_component_strategies) != set(DEFAULT_KG_COMPONENT_STRATEGIES):
        print(f"    [场景降级] 小样本仅启用 kg_component 子集: {allowed_kg_component_strategies}")

    kg_components.update(_run_kg_component_strategies(
        skip_complex_kg_component=skip_complex_kg_component,
        allowed_kg_component_strategies=allowed_kg_component_strategies,
        model_cols=model_cols,
        val_df=val_df,
        test_df=test_df,
        val_df_filled=val_df_filled,
        test_df_filled=test_df_filled,
        train_df_filled=train_df_filled,
        y_val=y_val,
        kg_component_active_cols=kg_component_active_cols,
        baselines_classic=baselines_classic,
        robust_cfg=robust_cfg,
        naive_scale=naive_scale,
        horizon=horizon,
        threshold_ratio=threshold_ratio,
        alpha_fallback=alpha_fallback,
        record_failure=_record_failure,
    ))

    skip_sota = os.environ.get("MODELCOMBINE_SKIP_SOTA", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    enable_dash_tta_node = os.environ.get("MODELCOMBINE_KG_INCLUDE_DASH_TTA", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if skip_sota:
        print("    [perf] MODELCOMBINE_SKIP_SOTA=1，跳过 SOTA 策略 (rl_qms/mole_router)")
        _record_failure("rl_qms", "skipped_by_perf_MODELCOMBINE_SKIP_SOTA")
        _record_failure("mole_router", "skipped_by_perf_MODELCOMBINE_SKIP_SOTA")
    else:
        baselines_sota.update(_run_baselines_sota(
            safe_cols=safe_cols,
            P_val=P_val,
            P_test=P_test,
            y_val=y_val,
            y_test=y_test,
            val_df_filled=val_df_filled,
            test_df_filled=test_df_filled,
            model_cols=model_cols,
            horizon=horizon,
            naive_scale=naive_scale,
            structural_baseline_val=structural_baseline_val,
            structural_baseline_test=structural_baseline_test,
            structural_baseline_name=structural_baseline_name,
            record_failure=_record_failure,
        ))
    if enable_dash_tta_node:
        baselines_sota.update(_run_dash_tta_node(
            safe_cols=safe_cols,
            P_val=P_val,
            P_test=P_test,
            y_val=y_val,
            y_test=y_test,
            val_df_filled=val_df_filled,
            test_df_filled=test_df_filled,
            model_cols=model_cols,
            horizon=horizon,
            naive_scale=naive_scale,
            structural_baseline_val=structural_baseline_val,
            structural_baseline_test=structural_baseline_test,
            structural_baseline_name=structural_baseline_name,
            record_failure=_record_failure,
        ))

    results.update(baselines_classic)
    results.update(baselines_sota)
    results.update(kg_components)

    for strategy_name, strategy_payload in results.items():
        if strategy_name == "_meta" or not isinstance(strategy_payload, dict):
            continue
        naming = get_strategy_naming(
            strategy_name, default_category=strategy_payload.get("category")
        )
        strategy_payload.setdefault("strategy_display_name", naming["display_name"])
        strategy_payload.setdefault("method_family", naming["method_family"])
        strategy_payload.setdefault("method_route", naming["method_route"])
        strategy_payload.setdefault("core_route", naming["core_route"])

    _expected_kg_component = EXPECTED_KG_COMPONENT
    _actual_kg_component = list(kg_components.keys())
    _kg_component_fallback_ratio = 1.0 if skip_complex_kg_component else (
        max(0.0, 1.0 - len(_actual_kg_component) / max(len(_expected_kg_component), 1))
    )

    results["_meta"] = {
        "baselines_classic": list(baselines_classic.keys()),
        "baselines_sota": list(baselines_sota.keys()),
        "kg_component": list(kg_components.keys()),
        "kg_component_model_pool_mode": KG_COMPONENT_MODEL_POOL_MODE,
        "kg_component_active_models": kg_component_active_cols,
        "kg_component_allowed_strategies": allowed_kg_component_strategies,
        "scenario_similarity_weight_mode": SCENARIO_SIMILARITY_WEIGHT_MODE,
        "scenario_similarity_temperature": SCENARIO_SIMILARITY_TEMPERATURE,
        "drift_event_count": drift_event_count_val,
        "drift_event_count_val": drift_event_count_val,
        "drift_event_count_test": drift_event_count_test,
        "strategy_naming_version": "v1",
        "drift_max_psi": max_psi,
        "ridge_decay_beta": beta_decay,
        "ridge_decay_source": beta_decay_source,
        "ridge_decay_base": DRIFT_DECAY_BASE,
        "ridge_decay_slope": DRIFT_DECAY_SLOPE,
        "ridge_decay_max": DRIFT_DECAY_MAX,
        "skipped_complex_kg_component": skip_complex_kg_component,
        "recommended_fallback": fallback_strategy if skip_complex_kg_component else None,
        "fallback_ratio": _kg_component_fallback_ratio,
        "skip_sota": bool(skip_sota),
        "failed_strategies": failed_strategies,
    }

    return results
