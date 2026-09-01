"""
组合策略通用工具 - 所有脚本共享

修复内容:
1. Time-OOF 防止测试集泄漏
2. 弱模型过滤
3. 稳定的数据对齐
4. 约束优化
"""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.exceptions import ConvergenceWarning


# ============================================================
# 兼容性处理
# ============================================================

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))


# ============================================================
# 配置
# ============================================================

DATASET_HORIZONS = {
    "pjm": [1, 6, 24],
    "aemo_vic": [1, 6, 24],
    "aemo_nsw": [1, 6, 24],
}

# 单一真源：从 model_assets.yaml 的 active 模型派生，消除硬编码清单漂移。
# 回退：manifest 读取失败时用历史硬编码列表，保证 eval 框架不因配置问题崩溃。
try:
    from ..core.manifest_loader import load_manifests, active_model_ids
    MODELS = sorted(active_model_ids(load_manifests()))
except Exception as _e:  # pragma: no cover
    print(f"[combination_utils][WARN] manifest 加载失败，回退硬编码 MODELS: {_e}")
    MODELS = [
        "xgboost_reg", "lgbm_reg", "catboost_reg",
        "prophet", "arima", "power_difference", "multimodal_fusion",
        "seasonal_naive",
    ]

STRATEGY_CONFIG = {
    "simple_avg": {
        "oof_valid": True,
        "auto_select_eligible": True,
        "requires_nonneg_weights": False,
    },
    "smart_weighted": {
        "oof_valid": False,
        "auto_select_eligible": False,
        "requires_nonneg_weights": True,
    },
    "static_weight": {
        "oof_valid": True,
        "auto_select_eligible": True,
        "requires_nonneg_weights": True,
    },
    "stacking": {
        "oof_valid": True,
        "auto_select_eligible": True,
        "requires_nonneg_weights": False,
    },
    "constrained_opt": {
        "oof_valid": False,
        "auto_select_eligible": False,
        "requires_nonneg_weights": True,
    }
}

try:
    DEFAULT_FILTER_WEAK_THRESHOLD_RATIO = float(
        os.environ.get("MODELCOMBINE_FILTER_WEAK_THRESHOLD_RATIO", "2.0")
    )
except Exception:
    DEFAULT_FILTER_WEAK_THRESHOLD_RATIO = 2.0

try:
    FILTER_WEAK_LATE_WINDOW_RATIO = float(
        os.environ.get("MODELCOMBINE_FILTER_WEAK_LATE_WINDOW_RATIO", "0.2")
    )
except Exception:
    FILTER_WEAK_LATE_WINDOW_RATIO = 0.2
FILTER_WEAK_LATE_WINDOW_RATIO = min(max(FILTER_WEAK_LATE_WINDOW_RATIO, 0.05), 0.4)

try:
    FILTER_WEAK_MAX_LATE_MAE_RATIO = float(
        os.environ.get("MODELCOMBINE_FILTER_WEAK_MAX_LATE_MAE_RATIO", "1.5")
    )
except Exception:
    FILTER_WEAK_MAX_LATE_MAE_RATIO = 1.5
FILTER_WEAK_MAX_LATE_MAE_RATIO = max(FILTER_WEAK_MAX_LATE_MAE_RATIO, 1.0)

try:
    FILTER_WEAK_MAX_CV_MAE_COEF_VAR = float(
        os.environ.get("MODELCOMBINE_FILTER_WEAK_MAX_CV_MAE_COEF_VAR", "0.5")
    )
except Exception:
    FILTER_WEAK_MAX_CV_MAE_COEF_VAR = 0.5
FILTER_WEAK_MAX_CV_MAE_COEF_VAR = max(FILTER_WEAK_MAX_CV_MAE_COEF_VAR, 0.0)

try:
    FILTER_WEAK_RMSE_NAIVE_MARGIN = float(
        os.environ.get("MODELCOMBINE_FILTER_WEAK_RMSE_NAIVE_MARGIN", "1.05")
    )
except Exception:
    FILTER_WEAK_RMSE_NAIVE_MARGIN = 1.05
FILTER_WEAK_RMSE_NAIVE_MARGIN = max(FILTER_WEAK_RMSE_NAIVE_MARGIN, 1.0)

try:
    FILTER_WEAK_RMSE_NAIVE_MARGIN_LONG_H = float(
        os.environ.get("MODELCOMBINE_FILTER_WEAK_RMSE_NAIVE_MARGIN_LONG_H", "1.20")
    )
except Exception:
    FILTER_WEAK_RMSE_NAIVE_MARGIN_LONG_H = 1.20
FILTER_WEAK_RMSE_NAIVE_MARGIN_LONG_H = max(FILTER_WEAK_RMSE_NAIVE_MARGIN_LONG_H, 1.0)

try:
    FILTER_WEAK_LONG_H_MIN = int(
        os.environ.get("MODELCOMBINE_FILTER_WEAK_LONG_H_MIN", "24")
    )
except Exception:
    FILTER_WEAK_LONG_H_MIN = 24
FILTER_WEAK_LONG_H_MIN = max(FILTER_WEAK_LONG_H_MIN, 1)


# ============================================================
# 伪回归器（等权回退用）
# ============================================================

class EqualWeightRegressor:
    """等权回退用伪回归器"""
    def __init__(self, n_features: int):
        self.coef_ = np.ones(n_features) / n_features
        self.intercept_ = 0.0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_


# ============================================================
# Ridge 拟合（稳健版）
# ============================================================

def fit_ridge_robust(X: np.ndarray, y: np.ndarray, 
                     alpha: float = 1.0,
                     positive: bool = True,
                     fit_intercept: bool = False,
                     sample_weight: Optional[np.ndarray] = None) -> Tuple[Any, Dict]:
    """
    稳健的 Ridge 拟合
    
    特点:
    1. 自动探测 sklearn 版本支持
    2. 手动处理负权重（如果 positive=True 但版本不支持）
    3. 等权回退兜底
    4. 所有路径都有明确 return
    """
    meta = {
        "solver_used": None, 
        "positive_enforced": False, 
        "fallback_reason": None,
        "alpha": alpha
    }
    tol = 1e-3 if X.shape[1] > 10 else 1e-4
    max_iter = 10000
    meta["tol"] = tol
    meta["max_iter"] = max_iter
    
    # 尝试 1: positive=True + lbfgs (sklearn >= 1.0)
    if positive:
        for try_idx, cur_alpha in enumerate([alpha, alpha * 10.0], start=1):
            try:
                reg = Ridge(
                    alpha=cur_alpha,
                    fit_intercept=fit_intercept,
                    positive=True,
                    solver="lbfgs",
                    max_iter=max_iter,
                    tol=tol,
                )
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter("always", ConvergenceWarning)
                    reg.fit(X, y, sample_weight=sample_weight)
                has_conv_warning = any(
                    issubclass(w.category, ConvergenceWarning) for w in ws
                )
                if has_conv_warning and try_idx == 1:
                    meta["fallback_reason"] = "lbfgs_convergence_warning_retry_alpha_x10"
                    continue
                meta["solver_used"] = "lbfgs_positive"
                meta["positive_enforced"] = True
                meta["alpha_effective"] = float(cur_alpha)
                if has_conv_warning:
                    meta["fallback_reason"] = "lbfgs_convergence_warning_persisted"
                return reg, meta
            except (TypeError, ValueError):
                break
            except Exception as e:
                if try_idx == 1:
                    meta["fallback_reason"] = f"lbfgs_positive_failed_retry_alpha_x10:{e}"
                    continue
                meta["fallback_reason"] = f"lbfgs_positive_failed:{e}"
                break
    
    # 尝试 2: 标准 Ridge
    try:
        reg = Ridge(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol)
        reg.fit(X, y, sample_weight=sample_weight)
        meta["solver_used"] = "default"
        
        # 如果需要正权重但版本不支持，手动处理
        if positive and hasattr(reg, 'coef_'):
            neg_mask = reg.coef_ < 0
            if neg_mask.any():
                reg.coef_ = np.maximum(reg.coef_, 0)
                if not fit_intercept:
                    coef_sum = reg.coef_.sum()
                    if coef_sum > 1e-8:
                        reg.coef_ = reg.coef_ / coef_sum
                    else:
                        reg.coef_ = np.ones_like(reg.coef_) / len(reg.coef_)
                meta["positive_enforced"] = True
                meta["fallback_reason"] = "manual_clip"
        
        return reg, meta
        
    except Exception as e:
        meta["fallback_reason"] = f"ridge_failed: {e}"
    
    # 尝试 3: 等权回退
    meta["solver_used"] = "equal_weight_fallback"
    return EqualWeightRegressor(X.shape[1]), meta


# ============================================================
# 稳定键生成
# ============================================================

def generate_stable_key(df: pd.DataFrame, 
                        ts_col: str = "timestamp",
                        secondary_col: str = None) -> Tuple[pd.Series, pd.DataFrame]:
    """
    生成稳定的行键
    
    Args:
        df: 输入 DataFrame
        ts_col: 时间戳列名
        secondary_col: 二级排序列
    
    Returns:
        stable_key: 稳定键 Series
        sorted_df: 排序后的 DataFrame
    """
    df = df.copy()
    
    df["_ts_dt"] = pd.to_datetime(df[ts_col])
    
    if secondary_col and secondary_col in df.columns:
        sort_cols = ["_ts_dt", secondary_col]
    elif "row_id" in df.columns:
        sort_cols = ["_ts_dt", "row_id"]
    else:
        df["_orig_idx"] = df.index
        sort_cols = ["_ts_dt", "_orig_idx"]
    
    df = df.sort_values(sort_cols).reset_index(drop=True)
    
    stable_key: pd.Series = (
        df["_ts_dt"].astype(str) + "_" + 
        df.groupby("_ts_dt").cumcount().astype(str)
    )
    
    return stable_key, df


# ============================================================
# 数据加载
# ============================================================

def load_predictions_safe(pred_root: Path, dataset: str, horizon: int,
                          models: List[str], split: str) -> pd.DataFrame:
    """
    安全的预测加载
    
    特点:
    1. 优先使用 row_id，否则生成稳定键
    2. 键唯一性校验
    3. y 值一致性校验
    """
    frames = []
    y_ref = None
    key_col = "row_id"
    
    for mid in models:
        p = pred_root / dataset / f"{split}_pred_h{horizon}_{mid}.csv"
        if not p.exists():
            continue
        
        df = pd.read_csv(p)
        
        if "timestamp" not in df.columns:
            raise ValueError(f"模型 {mid} 缺少 timestamp 列，无法生成 row_id")

        # 强制 row_id：优先使用原始 row_id；缺失或重复时按 timestamp+序号自动重建。
        if "row_id" not in df.columns or df["row_id"].duplicated().any():
            stable_key, df = generate_stable_key(df, "timestamp")
            df["row_id"] = stable_key.astype(str).values
            warnings.warn(f"模型 {mid}: 缺少/重复 row_id，已按 timestamp 稳定序自动生成 row_id")
        else:
            df["row_id"] = df["row_id"].astype(str)

        if df[key_col].duplicated().any():
            raise ValueError(f"模型 {mid} 存在重复 row_id")
        
        df = df.rename(columns={"pred": mid})
        
        # y 一致性校验
        if y_ref is None:
            y_ref = df[[key_col, "timestamp", "y"]].copy()
        else:
            check = y_ref.merge(
                df[[key_col, "y"]], 
                on=key_col, 
                suffixes=("_ref", "_new"),
                how="inner"
            )
            
            if len(check) != len(y_ref):
                raise ValueError(f"模型 {mid} merge 后行数变化")
            
            y_diff = np.abs(check["y_ref"] - check["y_new"])
            if y_diff.max() > 1e-6:
                raise ValueError(f"模型 {mid} 的 y 值不一致")
        
        frames.append(df[[key_col, mid]])
    
    if not frames:
        raise FileNotFoundError(f"No predictions for {dataset} h={horizon} {split}")
    
    # Merge
    merged = y_ref.copy()
    for f in frames:
        merged = merged.merge(f, on=key_col, how="inner")
    
    if len(merged) != len(y_ref):
        raise ValueError(f"最终 merge 行数异常")
    
    # 清理并排序
    if "_ts_dt" not in merged.columns:
        merged["_ts_dt"] = pd.to_datetime(merged["timestamp"])
    merged = merged.sort_values(["_ts_dt", "row_id"]).reset_index(drop=True)
    
    drop_cols = [c for c in ["_stable_key", "_ts_dt", "_orig_idx"] if c in merged.columns]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    
    return merged


def get_common_models(df_val: pd.DataFrame, df_test: pd.DataFrame, 
                      candidate_models: List[str]) -> List[str]:
    """获取 val 和 test 都有的模型列"""
    val_cols = set(df_val.columns)
    test_cols = set(df_test.columns)
    common = val_cols & test_cols & set(candidate_models)
    return sorted(common)


# ============================================================
# 弱模型过滤
# ============================================================

def filter_weak_models(df_val: pd.DataFrame, model_cols: List[str],
                       threshold_ratio: float = DEFAULT_FILTER_WEAK_THRESHOLD_RATIO,
                       hard_cap_ratio: float = 5.0,
                       min_keep: int = 3,
                       corr_exemption_threshold: float = 0.5,
                       horizon: Optional[int] = None) -> Tuple[List[str], Dict]:
    """
    弱模型过滤
    
    规则:
    1. MAE > best * threshold_ratio 则过滤
    2. 但 MAE <= best * hard_cap_ratio 且误差相关性低可豁免
    3. 必保留 best_model
    4. 至少保留 min_keep 个模型
    """
    y_val = df_val["y"].values.astype(float)
    
    maes = {}
    valid_ratios = {}
    errors = {}
    for m in model_cols:
        pred = df_val[m].values.astype(float)
        err = pred - y_val
        mask = np.isfinite(err)
        n_valid = int(mask.sum())
        valid_ratios[m] = float(n_valid / len(err)) if len(err) > 0 else 0.0
        if n_valid > 0:
            maes[m] = float(np.mean(np.abs(err[mask])))
        else:
            maes[m] = float("inf")
        errors[m] = err
    
    finite_maes = {m: v for m, v in maes.items() if np.isfinite(v)}
    if finite_maes:
        best_mae = min(finite_maes.values())
        best_model = min(finite_maes, key=finite_maes.get)
    else:
        best_mae = float("inf")
        best_model = model_cols[0] if model_cols else None
    
    soft_threshold = best_mae * threshold_ratio
    hard_threshold = best_mae * hard_cap_ratio
    
    preliminary_safe = [m for m, mae in maes.items() if mae <= soft_threshold]
    preliminary_excluded = [m for m in model_cols if m not in preliminary_safe]
    
    # 相关性豁免
    exempted = []
    if preliminary_excluded and len(preliminary_safe) > 0:
        safe_errors = np.column_stack([errors[m] for m in preliminary_safe])
        safe_avg_error = np.nanmean(safe_errors, axis=1)
        
        for m in preliminary_excluded:
            if not np.isfinite(maes[m]) or maes[m] > hard_threshold:
                continue
            
            m_error = errors[m]
            corr_mask = np.isfinite(m_error) & np.isfinite(safe_avg_error)
            if int(corr_mask.sum()) >= 3:
                corr = np.corrcoef(m_error[corr_mask], safe_avg_error[corr_mask])[0, 1]
            else:
                corr = np.nan
            if np.isnan(corr):
                corr = 1.0
            
            if abs(corr) < corr_exemption_threshold:
                exempted.append(m)
    
    safe_models = preliminary_safe + exempted

    # 稳定性约束：末段劣化 + 分块方差 + naive 基线对比。
    unstable_reasons: Dict[str, str] = {}
    n_samples = len(y_val)
    if n_samples >= 80:
        naive_period = 24
        if "timestamp" in df_val.columns:
            try:
                ts = pd.to_datetime(df_val["timestamp"], errors="coerce").dropna().sort_values()
                if len(ts) >= 3:
                    delta = ts.diff().dropna().median()
                    if pd.notna(delta):
                        delta_sec = float(delta.total_seconds())
                        if np.isfinite(delta_sec) and delta_sec > 0:
                            inferred = int(round(86400.0 / delta_sec))
                            if 2 <= inferred <= 24 * 12:
                                naive_period = inferred
            except Exception:
                pass

        split_idx = int(n_samples * (1.0 - FILTER_WEAK_LATE_WINDOW_RATIO))
        split_idx = min(max(split_idx, 1), n_samples - 1)
        cv_blocks = 5
        fold_edges = np.linspace(0, n_samples, cv_blocks + 1, dtype=int)
        valid_naive = n_samples > naive_period

        for m in list(safe_models):
            abs_err = np.abs(errors[m])
            mae_early = float(np.nanmean(abs_err[:split_idx])) if split_idx > 0 else float(np.nanmean(abs_err))
            mae_late = float(np.nanmean(abs_err[split_idx:])) if split_idx < n_samples else mae_early

            # Rule 1: 后 20% 相比前 80% 明显恶化
            if np.isfinite(mae_early) and mae_early > 1e-10:
                late_ratio = mae_late / mae_early
                if late_ratio > FILTER_WEAK_MAX_LATE_MAE_RATIO:
                    unstable_reasons[m] = f"unstable_late:{late_ratio:.3f}>{FILTER_WEAK_MAX_LATE_MAE_RATIO:.3f}"
                    continue

            # Rule 2: 分块 MAE 变异系数过大（时间不稳定）
            fold_maes = []
            for i in range(cv_blocks):
                s = fold_edges[i]
                e = fold_edges[i + 1]
                if e - s < 10:
                    continue
                fold_mae = float(np.mean(abs_err[s:e]))
                if np.isfinite(fold_mae):
                    fold_maes.append(fold_mae)
            if len(fold_maes) >= 3:
                fm = float(np.mean(fold_maes))
                fs = float(np.std(fold_maes))
                if fm > 1e-10:
                    coef_var = fs / fm
                    if coef_var > FILTER_WEAK_MAX_CV_MAE_COEF_VAR:
                        unstable_reasons[m] = f"unstable_cv:{coef_var:.3f}>{FILTER_WEAK_MAX_CV_MAE_COEF_VAR:.3f}"
                        continue

            # Rule 3: 相比季节 naive 更差（长期步长放宽阈值，避免 h24 过度误杀）
            if valid_naive:
                y_shift = y_val[:-naive_period]
                y_curr = y_val[naive_period:]
                model_pred = np.asarray(df_val[m].values[naive_period:], dtype=float)
                if len(model_pred) == len(y_curr) and len(y_curr) > 0:
                    naive_rmse = float(np.sqrt(np.mean((y_curr - y_shift) ** 2)))
                    model_rmse = float(np.sqrt(np.mean((y_curr - model_pred) ** 2)))
                    if np.isfinite(naive_rmse) and naive_rmse > 1e-10:
                        rmse_ratio = model_rmse / naive_rmse
                        rmse_threshold = (
                            FILTER_WEAK_RMSE_NAIVE_MARGIN_LONG_H
                            if (horizon is not None and int(horizon) >= FILTER_WEAK_LONG_H_MIN)
                            else FILTER_WEAK_RMSE_NAIVE_MARGIN
                        )
                        if rmse_ratio > rmse_threshold:
                            unstable_reasons[m] = (
                                f"rmse_worse_than_naive:{rmse_ratio:.3f}>{rmse_threshold:.3f}"
                            )
                            continue

        if unstable_reasons:
            safe_models = [m for m in safe_models if m not in unstable_reasons]
    
    # 兜底
    if best_model is not None and best_model not in safe_models:
        safe_models.insert(0, best_model)
    
    if len(safe_models) < min_keep:
        sorted_models = sorted(maes.items(), key=lambda x: x[1])
        for m, _ in sorted_models:
            if m not in safe_models:
                safe_models.append(m)
            if len(safe_models) >= min_keep:
                break
    
    if len(safe_models) == 0:
        if best_model is not None:
            safe_models = [best_model]
        else:
            safe_models = list(model_cols[:max(1, min_keep)])

    return safe_models, {
        "best_model": best_model,
        "best_mae": best_mae,
        "soft_threshold": soft_threshold,
        "hard_threshold": hard_threshold,
        "exempted_by_correlation": exempted,
        "stability_removed": unstable_reasons,
        "final_excluded": [m for m in model_cols if m not in safe_models],
        "final_count": len(safe_models),
        "valid_ratios": valid_ratios,
        "rmse_naive_margin": (
            FILTER_WEAK_RMSE_NAIVE_MARGIN_LONG_H
            if (horizon is not None and int(horizon) >= FILTER_WEAK_LONG_H_MIN)
            else FILTER_WEAK_RMSE_NAIVE_MARGIN
        ),
        "rmse_naive_margin_base": FILTER_WEAK_RMSE_NAIVE_MARGIN,
        "rmse_naive_margin_long_h": FILTER_WEAK_RMSE_NAIVE_MARGIN_LONG_H,
        "long_h_min": FILTER_WEAK_LONG_H_MIN,
        "horizon": int(horizon) if horizon is not None else None,
    }


# ============================================================
# Time-OOF 组合学习
# ============================================================

def time_oof_combination(df_val: pd.DataFrame, model_cols: List[str],
                         n_folds: int = 3, min_train: int = 100,
                         gap: int = 1,
                         alpha_candidates: List[float] = None,
                         positive: bool = True,
                         fit_intercept: bool = False) -> Tuple[np.ndarray, float, Dict]:
    """
    时间序列 OOF 组合学习
    
    关键: OOF 训练时使用与最终模型相同的约束
    """
    if alpha_candidates is None:
        alpha_candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    X = df_val[model_cols].values
    y = df_val["y"].values
    n = len(y)
    
    meta = {
        "n_samples": n,
        "n_folds": n_folds,
        "gap": gap,
        "positive": positive,
        "fit_intercept": fit_intercept
    }
    
    # 自动调参
    effective_folds = n_folds
    effective_min_train = min_train
    
    available = n - min_train - gap * n_folds
    if available < n_folds * 10:
        effective_folds = max(2, n_folds - 1)
        effective_min_train = max(30, min_train // 2)
        meta["auto_adjusted"] = True
    
    fold_size = (n - effective_min_train - gap * effective_folds) // effective_folds
    
    if fold_size <= 5:
        meta["skipped"] = True
        meta["reason"] = "fold_size_too_small"
        return np.full(n, np.nan), 1.0, meta
    
    # CV 选 alpha
    alpha_oof_mae = {}
    
    for alpha in alpha_candidates:
        oof_pred_alpha = np.full(n, np.nan)
        
        for fold in range(effective_folds):
            train_end = effective_min_train + fold * fold_size
            val_start = train_end + gap
            val_end = val_start + fold_size if fold < effective_folds - 1 else n
            
            if val_end <= val_start or val_start >= n:
                continue
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_fold_val = X[val_start:val_end]
            
            reg, _ = fit_ridge_robust(
                X_train, y_train, 
                alpha=alpha, 
                positive=positive,
                fit_intercept=fit_intercept
            )
            oof_pred_alpha[val_start:val_end] = reg.predict(X_fold_val)
        
        valid_mask = ~np.isnan(oof_pred_alpha)
        if valid_mask.sum() > 0:
            mae = np.mean(np.abs(y[valid_mask] - oof_pred_alpha[valid_mask]))
            alpha_oof_mae[alpha] = mae
    
    if not alpha_oof_mae:
        meta["skipped"] = True
        meta["reason"] = "all_alpha_failed"
        return np.full(n, np.nan), 1.0, meta
    
    best_alpha = min(alpha_oof_mae, key=alpha_oof_mae.get)
    meta["best_alpha"] = best_alpha
    meta["alpha_oof_mae"] = {str(k): float(v) for k, v in alpha_oof_mae.items()}
    
    # 用最优 alpha 生成最终 OOF
    oof_pred = np.full(n, np.nan)
    
    for fold in range(effective_folds):
        train_end = effective_min_train + fold * fold_size
        val_start = train_end + gap
        val_end = val_start + fold_size if fold < effective_folds - 1 else n
        
        if val_end <= val_start or val_start >= n:
            continue
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_fold_val = X[val_start:val_end]
        
        reg, _ = fit_ridge_robust(
            X_train, y_train,
            alpha=best_alpha,
            positive=positive,
            fit_intercept=fit_intercept
        )
        oof_pred[val_start:val_end] = reg.predict(X_fold_val)
    
    return oof_pred, best_alpha, meta


# ============================================================
# 约束优化
# ============================================================

def optimize_weights_constrained(X: np.ndarray, y: np.ndarray,
                                  model_names: List[str] = None,
                                  min_improvement: float = 0.005) -> Tuple[np.ndarray, Dict]:
    """
    约束权重优化: w >= 0, sum(w) = 1
    
    流程: SLSQP -> NNLS -> 等权
    """
    from scipy.optimize import minimize, nnls
    
    n_models = X.shape[1]
    model_names = model_names or [f"model_{i}" for i in range(n_models)]
    
    meta = {
        "solver_attempts": [],
        "final_solver": None,
        "improvement_pct": None,
        "model_names": model_names
    }
    
    def objective(w):
        return np.mean((y - X @ w) ** 2)
    
    w0 = np.ones(n_models) / n_models
    baseline_obj = objective(w0)
    meta["baseline_mse"] = float(baseline_obj)
    
    # SLSQP
    try:
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(objective, w0, method="SLSQP",
                         bounds=bounds, constraints=constraints,
                         options={"maxiter": 500, "ftol": 1e-8})
        
        improvement = (baseline_obj - result.fun) / (baseline_obj + 1e-10)
        
        meta["solver_attempts"].append({
            "solver": "SLSQP",
            "success": result.success,
            "mse": float(result.fun),
            "improvement_pct": float(improvement * 100)
        })
        
        if result.success and improvement >= min_improvement:
            meta["final_solver"] = "SLSQP"
            meta["improvement_pct"] = float(improvement * 100)
            return result.x, meta
            
    except Exception as e:
        meta["solver_attempts"].append({"solver": "SLSQP", "error": str(e)})
    
    # NNLS
    try:
        w_nnls, _ = nnls(X, y)
        w_sum = w_nnls.sum()
        
        if w_sum > 1e-8:
            w_nnls = w_nnls / w_sum
            obj_nnls = objective(w_nnls)
            improvement = (baseline_obj - obj_nnls) / (baseline_obj + 1e-10)
            
            meta["solver_attempts"].append({
                "solver": "NNLS",
                "mse": float(obj_nnls),
                "improvement_pct": float(improvement * 100)
            })
            
            if improvement >= min_improvement:
                meta["final_solver"] = "NNLS"
                meta["improvement_pct"] = float(improvement * 100)
                return w_nnls, meta
                
    except Exception as e:
        meta["solver_attempts"].append({"solver": "NNLS", "error": str(e)})
    
    # 等权回退
    meta["final_solver"] = "equal_weight"
    meta["improvement_pct"] = 0.0
    
    return w0, meta


# ============================================================
# 策略选择
# ============================================================

def select_best_strategy(results: Dict, use_oof: bool = True) -> Tuple[str, Dict]:
    """
    选择最佳策略
    
    规则:
    1. 仅考虑 auto_select_eligible=True 且 _oof_valid=True 的策略
    2. 无有效策略时回退到 static_weight > simple_avg
    """
    meta = {
        "use_oof": use_oof,
        "considered_strategies": [],
        "excluded_strategies": []
    }
    
    val_metrics = results.get("val_oof" if use_oof else "val_insample", {})
    
    candidates = {}
    
    for strategy, config in STRATEGY_CONFIG.items():
        if strategy not in val_metrics:
            continue
        
        data = val_metrics[strategy]
        
        if not config.get("auto_select_eligible", True):
            meta["excluded_strategies"].append({
                "strategy": strategy,
                "reason": "not_auto_select_eligible"
            })
            continue
        
        if use_oof:
            if not isinstance(data, dict) or not data.get("_oof_valid", False):
                meta["excluded_strategies"].append({
                    "strategy": strategy,
                    "reason": data.get("_reason", "oof_invalid") if isinstance(data, dict) else "invalid_data"
                })
                continue
        
        if isinstance(data, dict) and "mae" in data and data["mae"] is not None:
            candidates[strategy] = data["mae"]
            meta["considered_strategies"].append(strategy)
    
    if not candidates:
        if "static_weight" in val_metrics:
            meta["fallback"] = "static_weight"
            return "static_weight", meta
        
        meta["fallback"] = "simple_avg"
        return "simple_avg", meta
    
    best = min(candidates, key=candidates.get)
    meta["best_strategy"] = best
    meta["best_mae"] = candidates[best]
    
    return best, meta


# ============================================================
# 评估指标
# ============================================================

def evaluate(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(root_mean_squared_error(y, pred))
    }


def ensure_dir(path: Path) -> None:
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
