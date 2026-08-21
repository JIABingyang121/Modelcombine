"""
P2.3  Blocked (expanding-window) cross-validation utilities.

所有需要在时间序列验证集上做超参数搜索的场景都应使用此模块，
确保 CV 折严格按照时间递增，训练折永远在验证折之前，杜绝未来数据泄露。

两个公开函数：
  - blocked_cv_splits : 生成 (train_idx, val_idx) 对
  - blocked_cv_select_alpha : 用 Ridge 在 blocked CV 上选最优 alpha
"""

from __future__ import annotations

from typing import List, Tuple, Callable, Optional

import numpy as np
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------------------
# 通用 blocked / expanding-window split 生成器
# ---------------------------------------------------------------------------

def resolve_blocked_cv_config(n_samples: int, horizon: int) -> Tuple[int, int, int]:
    """
    统一 blocked CV 配置口径：
    - n_val < 1000: 2-fold, 更大的训练窗
    - 其余: 3-fold
    - gap = horizon
    """
    gap = max(int(horizon), 0)
    if n_samples < 1000:
        return 2, min(300, max(80, n_samples // 3)), gap
    return 3, min(200, max(80, n_samples // 4)), gap


def blocked_cv_splits(
    n_samples: int,
    n_folds: int = 3,
    min_train: int = 50,
    gap: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window time-series CV splits.
    先固定最小训练窗 min_train，再将剩余窗口按 n_folds 近似均分为验证折。

    Args:
        n_samples:  总样本数
        n_folds:    折数
        min_train:  训练折最小长度（样本不够时返回空列表）
        gap:        训练-验证间跳过的样本数（可选，默认 0）

    Returns:
        列表，每项 (train_indices, val_indices)
    """
    if n_samples <= 0 or n_folds <= 0:
        return []
    if n_samples < min_train + 2:
        return []

    # 先保障训练窗下限，再把剩余窗口平均分给 n_folds 个验证折。
    available = n_samples - min_train - gap
    if available <= 0:
        return []
    val_size = max(1, available // n_folds)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        train_end = min_train + i * val_size
        val_start = train_end + gap
        if val_start >= n_samples:
            break
        val_end = n_samples if i == n_folds - 1 else min(val_start + val_size, n_samples)
        if val_end <= val_start:
            continue
        splits.append((np.arange(train_end), np.arange(val_start, val_end)))
    return splits


# ---------------------------------------------------------------------------
# 用 blocked CV 选 Ridge alpha（最常见的使用场景）
# ---------------------------------------------------------------------------

def blocked_cv_select_alpha(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alphas: Optional[List[float]] = None,
    n_folds: int = 3,
    min_train: int = 50,
    positive: bool = True,
    fit_intercept: bool = False,
    sample_weight: Optional[np.ndarray] = None,
    scorer: Optional[Callable] = None,
    gap: int = 0,
    max_iter: int = 10000,
    tol: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Blocked CV 选择 Ridge 回归最优 alpha。

    Args:
        X:              特征矩阵 (n, d)
        y:              目标向量 (n,)
        alphas:         候选 alpha 值列表
        n_folds:        CV 折数
        min_train:      最小训练折长度
        positive:       是否约束系数为正
        fit_intercept:  是否拟合截距
        sample_weight:  样本权重（可选）
        scorer:         评分函数 scorer(y_true, y_pred) -> float (越小越好)
                        默认 MAE

    Returns:
        (best_alpha, best_score)
    """
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]

    if scorer is None:
        scorer = lambda yt, yp: float(np.mean(np.abs(yt - yp)))

    if tol is None:
        tol = 1e-3 if X.shape[1] > 10 else 1e-4

    splits = blocked_cv_splits(len(X), n_folds=n_folds, min_train=min_train, gap=gap)
    if not splits:
        # 样本太少，返回最大正则化
        return alphas[-1], float("inf")

    best_alpha = alphas[len(alphas) // 2]  # 默认中间值
    best_score = float("inf")

    for alpha in alphas:
        fold_scores: List[float] = []
        for train_idx, val_idx in splits:
            try:
                # solver 选择：positive=True 需要 lbfgs；否则用默认 auto
                solver = "lbfgs" if positive else "auto"
                reg = Ridge(
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    positive=positive,
                    solver=solver,
                    max_iter=max_iter,
                    tol=tol,
                )
                sw = sample_weight[train_idx] if sample_weight is not None else None
                reg.fit(X[train_idx], y[train_idx], sample_weight=sw)
                pred = reg.predict(X[val_idx])
                fold_scores.append(scorer(y[val_idx], pred))
            except Exception:
                continue

        if fold_scores:
            mean_score = float(np.mean(fold_scores))
            if mean_score < best_score:
                best_score = mean_score
                best_alpha = alpha

    return best_alpha, best_score


# ---------------------------------------------------------------------------
# 用 blocked CV 选 SoftGating alpha
# ---------------------------------------------------------------------------

def blocked_cv_select_softgating_alpha(
    y_val: np.ndarray,
    dynamic_pred: np.ndarray,
    static_pred: np.ndarray,
    alpha_candidates: List[float],
    n_folds: int = 3,
    min_train: int = 30,
) -> Tuple[float, float]:
    """
    Blocked CV 选择 SoftGating 的混合系数 alpha。
    w_final = (1-alpha)*dynamic + alpha*static

    Args:
        y_val:             验证集真实值
        dynamic_pred:      动态策略预测
        static_pred:       静态策略预测
        alpha_candidates:  alpha 候选值
        n_folds:           CV 折数
        min_train:         最小训练折长度

    Returns:
        (best_alpha, best_rmse)
    """
    splits = blocked_cv_splits(len(y_val), n_folds=n_folds, min_train=min_train)
    if not splits:
        # 样本太少，默认 0.5
        blended = 0.5 * dynamic_pred + 0.5 * static_pred
        rmse = float(np.sqrt(np.mean((y_val - blended) ** 2)))
        return 0.5, rmse

    best_alpha = 0.5
    best_score = float("inf")

    for a in alpha_candidates:
        fold_scores: List[float] = []
        for _, val_idx in splits:
            blended = (1 - a) * dynamic_pred[val_idx] + a * static_pred[val_idx]
            rmse = float(np.sqrt(np.mean((y_val[val_idx] - blended) ** 2)))
            fold_scores.append(rmse)

        mean_score = float(np.mean(fold_scores))
        if mean_score < best_score:
            best_score = mean_score
            best_alpha = a

    return best_alpha, best_score
