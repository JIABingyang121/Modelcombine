"""KG model stability and candidate filtering utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    MODEL_STABILITY_RMSE_NAIVE_MARGIN,
    MODEL_STABILITY_RMSE_NAIVE_MARGIN_LONG_H,
    MODEL_STABILITY_UNSTABLE_LATE_REQUIRES_CORROBORATION,
    PROTOCOL_STEPWISE_MIN_IMPROVEMENT,
)
from .conflict import compute_error_correlations
from .drift import _estimate_prediction_space_drift

def _infer_daily_seasonal_period_from_timestamp(
    df: pd.DataFrame,
    *,
    ts_col: str = "timestamp",
    default_period: int = 24,
) -> int:
    """
    基于时间间隔推断“1天对应多少步”：
    - 小时级 -> 24
    - 半小时级 -> 48
    """
    if ts_col not in df.columns:
        return int(default_period)
    try:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
        ts = ts.dropna().sort_values()
        if len(ts) < 3:
            return int(default_period)
        delta = ts.diff().dropna().median()
        if pd.isna(delta):
            return int(default_period)
        delta_sec = float(delta.total_seconds())
        if not np.isfinite(delta_sec) or delta_sec <= 0:
            return int(default_period)
        inferred = int(round(86400.0 / delta_sec))
        if inferred < 2 or inferred > 24 * 12:
            return int(default_period)
        return inferred
    except Exception:
        return int(default_period)

def _estimate_model_stability(
    df_val: pd.DataFrame,
    model_cols: List[str],
    *,
    late_window_ratio: float = 0.2,
) -> Dict[str, Dict[str, Any]]:
    """
    估计模型时间稳定性（仅用 val）：
    - late_ratio: 末段 MAE / 前段 MAE
    - cv_coef_var: 分块 MAE 变异系数
    - rmse_ratio_vs_naive: 相对 seasonal naive 的 RMSE 比
    """
    y_val = np.asarray(df_val["y"].values, dtype=float)
    n = len(y_val)
    out: Dict[str, Dict[str, Any]] = {}
    if n <= 1:
        return out

    split_idx = int(n * (1.0 - late_window_ratio))
    split_idx = min(max(split_idx, 1), n - 1)
    edges = np.linspace(0, n, 6, dtype=int)  # 5 blocks

    naive_period = _infer_daily_seasonal_period_from_timestamp(df_val, default_period=24)
    naive_rmse = None
    if n > naive_period:
        y_shift = y_val[:-naive_period]
        y_curr = y_val[naive_period:]
        if len(y_curr) > 0:
            naive_rmse = float(np.sqrt(np.mean((y_curr - y_shift) ** 2)))
            if not np.isfinite(naive_rmse):
                naive_rmse = None

    for m in model_cols:
        pred = np.asarray(df_val[m].values, dtype=float)
        abs_err = np.abs(pred - y_val)
        mae_early = float(np.mean(abs_err[:split_idx])) if split_idx > 0 else float(np.mean(abs_err))
        mae_late = float(np.mean(abs_err[split_idx:])) if split_idx < n else mae_early
        late_ratio = None
        if np.isfinite(mae_early) and mae_early > 1e-10:
            late_ratio = float(mae_late / mae_early)

        fold_maes = []
        for i in range(5):
            s = edges[i]
            e = edges[i + 1]
            if e - s < 10:
                continue
            f_mae = float(np.mean(abs_err[s:e]))
            if np.isfinite(f_mae):
                fold_maes.append(f_mae)
        cv_coef_var = None
        if len(fold_maes) >= 3:
            f_mean = float(np.mean(fold_maes))
            f_std = float(np.std(fold_maes))
            if f_mean > 1e-10:
                cv_coef_var = float(f_std / f_mean)

        rmse_ratio_vs_naive = None
        if naive_rmse is not None and naive_rmse > 1e-10 and n > naive_period:
            model_rmse = float(np.sqrt(np.mean((y_val[naive_period:] - pred[naive_period:]) ** 2)))
            if np.isfinite(model_rmse):
                rmse_ratio_vs_naive = float(model_rmse / naive_rmse)

        out[m] = {
            "late_ratio": late_ratio,
            "cv_coef_var": cv_coef_var,
            "rmse_ratio_vs_naive": rmse_ratio_vs_naive,
            "unstable_late": bool(late_ratio is not None and late_ratio > 1.5),
            "unstable_cv": bool(cv_coef_var is not None and cv_coef_var > 0.5),
            "worse_than_naive": bool(
                rmse_ratio_vs_naive is not None and rmse_ratio_vs_naive > MODEL_STABILITY_RMSE_NAIVE_MARGIN
            ),
            "naive_period": int(naive_period),
        }
    return out

def _collect_stability_reasons(
    *,
    model_name: str,
    stability: Dict[str, Any],
    drift_level: str,
    horizon: int,
) -> List[str]:
    reasons: List[str] = []
    rmse_ratio = stability.get("rmse_ratio_vs_naive")
    worse_than_naive_effective = False
    # 阈值上提到条件之外：Task 6B 的佐证判据也要引用它，避免依赖分支内绑定。
    rmse_threshold = (
        MODEL_STABILITY_RMSE_NAIVE_MARGIN_LONG_H
        if horizon >= 24
        else MODEL_STABILITY_RMSE_NAIVE_MARGIN
    )
    if rmse_ratio is not None and np.isfinite(rmse_ratio):
        worse_than_naive_effective = bool(rmse_ratio > rmse_threshold)

    if stability.get("unstable_late"):
        reasons.append("unstable_late")
    if stability.get("unstable_cv"):
        reasons.append("unstable_cv")
    # 记录修正前判据，保证 guard 变化在 trace/meta 中可对照（合一计划 Task 6B）。
    reasons_before_corroboration = sorted(set(reasons))
    # 使用自适应阈值判定 worse_than_naive：
    # h>=24 用 MODEL_STABILITY_RMSE_NAIVE_MARGIN_LONG_H (默认 1.20)
    # h<24  用 MODEL_STABILITY_RMSE_NAIVE_MARGIN (默认 1.05)
    if worse_than_naive_effective:
        reasons.append("worse_than_naive")
    if (
        drift_level == "high"
        and horizon >= 24
        and model_name in {"prophet", "arima"}
        and (worse_than_naive_effective or stability.get("unstable_late") or stability.get("unstable_cv"))
    ):
        reasons.append("high_drift_long_h_unstable_external")

    resolved = sorted(set(reasons))
    stability["reasons_before_corroboration"] = list(reasons_before_corroboration)

    # Task 6B 最小修正：unstable_late 只度量验证窗前后段误差比，不含准确度信息。
    # 当它是**唯一**原因、且该候选已通过既有准确度下限（未触发 worse_than_naive）
    # 时，降级为软标记而不硬删除——复用既有阈值，不引入新阈值、不放宽 1.5。
    if (
        MODEL_STABILITY_UNSTABLE_LATE_REQUIRES_CORROBORATION
        and resolved == ["unstable_late"]
        and rmse_ratio is not None
        and np.isfinite(rmse_ratio)
        and not worse_than_naive_effective
    ):
        stability["retained_despite_unstable_late"] = True
        stability["retention_reason"] = (
            f"unstable_late_only_without_accuracy_corroboration: "
            f"rmse_ratio_vs_naive={rmse_ratio:.4f} <= threshold={rmse_threshold:.2f}"
        )
        return []

    stability["retained_despite_unstable_late"] = False
    return resolved

def _deduplicate_models_by_corr(
    model_cols: List[str],
    maes: Dict[str, float],
    error_corrs: Dict[Tuple[str, str], float],
    threshold: float = 0.999,
) -> Tuple[List[str], List[str]]:
    """
    去除几乎完全同质的模型预测（|corr| > threshold）。
    保留 MAE 更低者，移除更差者。
    """
    removed = set()
    active = set(model_cols)
    for (m1, m2), corr in error_corrs.items():
        if abs(float(corr)) <= threshold:
            continue
        if m1 not in active or m2 not in active:
            continue
        if m1 in removed or m2 in removed:
            continue
        worse = m1 if maes.get(m1, float("inf")) >= maes.get(m2, float("inf")) else m2
        removed.add(worse)
        active.discard(worse)
    filtered = [m for m in model_cols if m not in removed]
    return filtered, sorted(removed)

def adaptive_max_models(n_val: int, n_models_available: int) -> int:
    """样本越少，组合模型数上限越保守。"""
    if n_models_available <= 0:
        return 0
    if n_val < 200:
        return min(2, n_models_available)
    if n_val < 500:
        return min(3, n_models_available)
    if n_val < 2000:
        return min(4, n_models_available)
    return min(5, n_models_available)

def _adaptive_stepwise_min_improve(
    *,
    drift_level: str,
    n_candidates: int,
    n_val: int,
) -> float:
    """
    Stepwise 改进阈值自适应：
    - high drift 下更保守，避免 val 过拟合扩张候选
    - high drift 且候选充足（0.005），候选稀缺（0.010）
    - low/medium 使用默认阈值
    - 小样本仍保持更保守门槛（>=0.01）
    """
    if drift_level == "high":
        min_improve = 0.005 if n_candidates >= 4 else 0.010
    else:
        min_improve = PROTOCOL_STEPWISE_MIN_IMPROVEMENT
    if n_val < 1000:
        min_improve = max(min_improve, 0.01)
    return float(min(max(min_improve, 0.0005), 0.05))

def _dedup_and_stability_filter(
    *,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    model_cols: List[str],
    maes: Dict[str, float],
    horizon: int,
    drift_level_override: Optional[str] = None,
    print_suffix: str = "",
) -> Dict[str, Any]:
    """
    公共候选清洗：
    1) 同质去重
    2) 高漂移下稳定性过滤
    3) 输出过滤后的 error_corrs
    """
    work_cols = list(model_cols)
    error_corrs_all = compute_error_correlations(df_val, work_cols)
    work_cols, dedup_removed = _deduplicate_models_by_corr(work_cols, maes, error_corrs_all, threshold=0.999)
    if dedup_removed:
        print(f"    去重: 移除同质预测模型 {dedup_removed}")
    if len(work_cols) < 2:
        work_cols = sorted(maes.keys(), key=lambda m: maes[m])[:min(2, len(maes))]

    if drift_level_override is None:
        drift_meta = _estimate_prediction_space_drift(df_val, df_test, work_cols)
        drift_level = str(drift_meta.get("drift_level", "low"))
    else:
        drift_level = str(drift_level_override)
        drift_meta = {"drift_level": drift_level}

    stability_meta = _estimate_model_stability(df_val, work_cols)
    stability_removed: Dict[str, str] = {}

    if drift_level == "high":
        filtered_models: List[str] = []
        for m in work_cols:
            reasons = _collect_stability_reasons(
                model_name=m,
                stability=stability_meta.get(m, {}),
                drift_level=drift_level,
                horizon=horizon,
            )
            if reasons:
                stability_removed[m] = ",".join(reasons)
            else:
                filtered_models.append(m)
        if len(filtered_models) >= 2:
            work_cols = filtered_models
            tag = f"{print_suffix}" if print_suffix else ""
            print(f"    稳定性过滤{tag}: 移除 {stability_removed}")

    if len(work_cols) < 2:
        work_cols = sorted(maes.keys(), key=lambda m: maes[m])[:min(2, len(maes))]

    error_corrs = compute_error_correlations(df_val, work_cols)
    return {
        "model_cols": work_cols,
        "error_corrs_all": error_corrs_all,
        "error_corrs": error_corrs,
        "dedup_removed": dedup_removed,
        "drift_meta": drift_meta,
        "drift_level": drift_level,
        "stability_meta": stability_meta,
        "stability_removed": stability_removed,
    }
