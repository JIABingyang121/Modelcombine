import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, mean_squared_error

# [Fix] sklearn 1.4+ 废弃 squared=False，改用 root_mean_squared_error
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    # sklearn < 1.4 兼容
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))


# 逐样本动态组合的鲁棒性配置（用于抑制极端误差）
ROBUST_CFG = {
    "enable": True,
    "high_load_q": 0.90,     # 高负荷分位点
    "tail_q": 0.90,          # 高误差分位点
    "high_load_weight": 2.0, # 高负荷样本权重增益
    "tail_weight": 1.5,      # 高误差样本权重增益
    "max_weight": 4.0,       # 权重上限
}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, naive_scale: float = None) -> Dict[str, float]:
    """
    综合评估指标体系：
    - 基础指标: MAE, RMSE
    - 相对误差: sMAPE (0-200%), MASE (用训练集 naive scale)
    - 峰值指标: peak_weighted_mae, top10_mae, top10_rmse
    - 鲁棒性: p95_ae, max_ae, tail_rmse
    """
    abs_errors = np.abs(y_true - y_pred)

    # === 基础指标 ===
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))

    # === 相对误差指标 ===
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    smape = float(np.mean(2 * abs_errors / denom) * 100)

    # MASE
    mase = np.nan
    if naive_scale is not None and naive_scale > 1e-8:
        mase = float(mae / naive_scale)

    # === 峰值指标 ===
    median_y = np.median(np.abs(y_true)) + 1e-8
    weights = np.abs(y_true) / median_y
    peak_weighted_mae = float(np.sum(weights * abs_errors) / np.sum(weights))

    threshold = np.percentile(np.abs(y_true), 90)
    top_mask = np.abs(y_true) >= threshold
    if top_mask.sum() > 0:
        top10_mae = float(np.mean(abs_errors[top_mask]))
        top10_rmse = float(np.sqrt(np.mean(abs_errors[top_mask] ** 2)))
    else:
        top10_mae = mae
        top10_rmse = rmse

    # === 鲁棒性指标 ===
    p95_ae = float(np.percentile(abs_errors, 95))
    max_ae = float(np.max(abs_errors))
    error_threshold = np.percentile(abs_errors, 90)
    tail_mask = abs_errors >= error_threshold
    if tail_mask.sum() > 0:
        tail_rmse = float(np.sqrt(np.mean(abs_errors[tail_mask] ** 2)))
    else:
        tail_rmse = rmse

    result = {
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "peak_weighted_mae": peak_weighted_mae,
        "top10_mae": top10_mae,
        "top10_rmse": top10_rmse,
        "p95_ae": p95_ae,
        "max_ae": max_ae,
        "tail_rmse": tail_rmse,
    }
    if not np.isnan(mase):
        result["mase"] = mase

    return result


def compute_extreme_weights(y_true: np.ndarray, abs_errors: np.ndarray | None = None,
                            cfg: Dict[str, float] | None = None) -> np.ndarray:
    """为极端点生成样本权重（高负荷 + 高误差）"""
    cfg = cfg or ROBUST_CFG
    if not cfg.get("enable", False):
        return np.ones_like(y_true, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    weights = np.ones_like(y_true, dtype=float)
    if y_true.size == 0:
        return weights

    abs_y = np.abs(y_true)
    load_thr = np.percentile(abs_y, cfg["high_load_q"] * 100)
    weights += (abs_y >= load_thr).astype(float) * float(cfg.get("high_load_weight", 0.0))

    if abs_errors is not None:
        abs_errors = np.asarray(abs_errors, dtype=float)
        if abs_errors.size > 0:
            err_thr = np.percentile(abs_errors, cfg["tail_q"] * 100)
            weights += (abs_errors >= err_thr).astype(float) * float(cfg.get("tail_weight", 0.0))

    max_w = cfg.get("max_weight")
    if max_w is not None:
        weights = np.minimum(weights, float(max_w))

    # 归一化到均值≈1
    mean_w = float(np.mean(weights))
    if mean_w > 0:
        weights = weights / mean_w

    return weights


def robust_mae(y_true: np.ndarray, y_pred: np.ndarray, cfg: Dict[str, float] | None = None) -> float:
    """高负荷+高误差加权 MAE，用于动态选择与权重学习"""
    cfg = cfg or ROBUST_CFG
    if not cfg.get("enable", False):
        return float(mean_absolute_error(y_true, y_pred))
    abs_errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    weights = compute_extreme_weights(y_true, abs_errors, cfg)
    return float(np.sum(weights * abs_errors) / (np.sum(weights) + 1e-8))


def seasonal_naive(y: np.ndarray, sp: int) -> np.ndarray:
    """构建季节性 naive 预测: y_t = y_{t-sp}"""
    s = pd.Series(y)
    return s.shift(sp).values


def compute_naive_scale(y_train: np.ndarray, sp: int) -> float:
    """
    在训练集上计算季节性 naive 的 MAE（用作 MASE 的分母）
    这是 Hyndman 标准 MASE 定义
    """
    y_naive = seasonal_naive(y_train, sp)
    valid_mask = np.arange(len(y_train)) >= sp
    if valid_mask.sum() > 0:
        return float(np.mean(np.abs(y_train[valid_mask] - y_naive[valid_mask])))
    return float(np.mean(np.abs(y_train - y_naive)))


def evaluate_slices(df: pd.DataFrame, pred_col: str, model_cols: List[str] = None) -> Dict[str, Dict]:
    """
    按场景切片评估指标
    """
    slices = {}
    y_true = df["y"].values
    y_pred = df[pred_col].values
    abs_errors = np.abs(y_true - y_pred)

    def slice_metrics(mask):
        if mask.sum() < 10:
            return None
        errs = abs_errors[mask]
        return {
            "n": int(mask.sum()),
            "mae": float(np.mean(errs)),
            "rmse": float(np.sqrt(np.mean(errs ** 2))),
            "p95_ae": float(np.percentile(errs, 95)),
        }

    if "ctx_hour" in df.columns:
        df = df.copy()
        hours = df["ctx_hour"].fillna(12).astype(int)
        conditions = [
            (hours >= 7) & (hours < 10),
            (hours >= 10) & (hours < 17),
            (hours >= 17) & (hours < 21),
        ]
        choices = ["morning_peak", "daytime", "evening_peak"]
        df["_period"] = np.select(conditions, choices, default="night")

        for period in ["morning_peak", "daytime", "evening_peak", "night"]:
            mask = (df["_period"] == period).values
            metrics = slice_metrics(mask)
            if metrics:
                slices[f"period_{period}"] = metrics

    if "ctx_is_weekend" in df.columns:
        for label, val in [("weekend", True), ("weekday", False)]:
            mask = (df["ctx_is_weekend"] == val).values
            metrics = slice_metrics(mask)
            if metrics:
                slices[label] = metrics

    if "ctx_is_holiday" in df.columns:
        for label, val in [("holiday", True), ("non_holiday", False)]:
            mask = (df["ctx_is_holiday"] == val).values
            metrics = slice_metrics(mask)
            if metrics:
                slices[label] = metrics

    thr = np.percentile(np.abs(y_true), 90)
    top_mask = np.abs(y_true) >= thr
    metrics = slice_metrics(top_mask)
    if metrics:
        slices["top10_load"] = metrics

    y_diff = np.abs(np.diff(y_true, prepend=y_true[0]))
    p33, p67 = np.percentile(y_diff, [33, 67])
    for label, low, high in [("vol_low", 0, p33), ("vol_mid", p33, p67), ("vol_high", p67, np.inf)]:
        mask = (y_diff >= low) & (y_diff < high)
        metrics = slice_metrics(mask)
        if metrics:
            slices[label] = metrics

    return slices


def compute_dynamic_cost(combo_name: str, combo_info: Dict, total_models: int,
                         actual_models_used: float = None) -> Dict[str, float]:
    """
    计算动态策略的成本指标
    """
    if actual_models_used is not None:
        models_used = actual_models_used
    elif "selected_models" in combo_info:
        models_used = len(combo_info["selected_models"])
    elif combo_name in ["simple_avg", "static_weight", "stacking", "simple_avg_safe", "static_weight_safe", "stacking_safe"]:
        models_used = total_models
    elif combo_name == "gating_network":
        models_used = total_models
    elif combo_name == "scenario_bucket":
        models_used = 2
    else:
        models_used = total_models

    return {
        "models_used": models_used,
        "model_ratio": models_used / total_models if total_models > 0 else 1.0,
        "cost_savings": 1.0 - (models_used / total_models) if total_models > 0 else 0.0,
    }
