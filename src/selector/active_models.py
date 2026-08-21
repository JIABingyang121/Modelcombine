"""
统一的 Active Models 过滤器

解决问题：
- Prophet/ARIMA 在 PJM 上 MASE>1.9，远差于 naive，不应参与动态选择
- 把"坏模型"包进候选池会拖垮组合效果（如 Simple Avg MAE=1521 vs 最佳 457）

过滤规则：
1. 绝对阈值：val_MAE > median_MAE * threshold_ratio 的模型被排除
2. MASE 阈值：MASE > mase_threshold 的模型被排除（比 naive 还差的不用）
3. 至少保留 min_models 个模型（避免极端情况全部被排除）
4. 输出 active_pool.json 以便调试和复现

使用方式：
```python
from src.selector.active_models import ActiveModelsFilter

filter = ActiveModelsFilter(threshold_ratio=2.0, mase_threshold=1.5, min_models=2)
active_models = filter.fit(val_df, model_cols, naive_scale)
# 之后只用 active_models 作为动态策略的候选池
```
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def compute_mase(y_true: np.ndarray, y_pred: np.ndarray, naive_scale: float) -> float:
    """计算 MASE = MAE / naive_scale"""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if naive_scale > 0:
        return mae / naive_scale
    return float('inf')


class ActiveModelsFilter:
    """
    统一的模型候选池过滤器
    
    两级过滤:
    1. MASE 过滤: 排除 MASE > mase_threshold 的模型（比 naive 差太多的不用）
    2. 相对过滤: 排除 MAE > percentile(val_MAE, threshold_pct) 的模型
    
    回退策略:
    - 若过滤后模型数 < min_models，按 MAE 升序保留前 min_models 个
    - 至少保留 1 个模型
    """
    
    def __init__(
        self,
        mase_threshold: float = 1.5,
        threshold_pct: float = 75,  # 保留 MAE 低于 75 分位的模型
        min_models: int = 2,
        verbose: bool = True,
    ):
        """
        Args:
            mase_threshold: MASE 阈值，超过此值的模型被排除
            threshold_pct: MAE 分位阈值（0-100），超过此分位的模型被排除
            min_models: 最少保留的模型数
            verbose: 是否打印过滤日志
        """
        self.mase_threshold = mase_threshold
        self.threshold_pct = threshold_pct
        self.min_models = min_models
        self.verbose = verbose
        
        # 拟合后的状态
        self.model_scores: Dict[str, Dict[str, float]] = {}
        self.active_models: List[str] = []
        self.excluded_models: List[str] = []
        self.exclusion_reasons: Dict[str, str] = {}
        self._is_fitted = False
    
    def fit(
        self,
        val_df: pd.DataFrame,
        model_cols: List[str],
        naive_scale: float,
        y_col: str = "y",
    ) -> List[str]:
        """
        在验证集上计算各模型表现并过滤
        
        Args:
            val_df: 验证集 DataFrame，需包含 y_col 列和各 model_cols 列
            model_cols: 所有候选模型名列表
            naive_scale: 用于计算 MASE 的 naive 基准误差（从训练集计算）
            y_col: 真值列名
        
        Returns:
            active_models: 通过过滤的模型列表
        """
        if y_col not in val_df.columns:
            raise ValueError(f"y_col={y_col} not in val_df.columns")
        
        y_true = val_df[y_col].values
        available_models = [m for m in model_cols if m in val_df.columns]
        
        if not available_models:
            raise ValueError("No model columns found in val_df")
        
        # 计算各模型的 MAE 和 MASE
        self.model_scores = {}
        for m in available_models:
            y_pred = val_df[m].values
            # 处理 NaN
            valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if valid_mask.sum() < 10:
                if self.verbose:
                    print(f"    [ActiveModelsFilter] {m}: too few valid samples, excluded")
                self.exclusion_reasons[m] = "too_few_valid_samples"
                continue
            y_t = y_true[valid_mask]
            y_p = y_pred[valid_mask]
            mae = float(mean_absolute_error(y_t, y_p))
            mase = compute_mase(y_t, y_p, naive_scale)
            self.model_scores[m] = {"mae": mae, "mase": mase}
        
        if not self.model_scores:
            # 全部模型都无效，回退到全部模型
            self.active_models = available_models
            self._is_fitted = True
            return self.active_models
        
        # 按 MAE 升序排序
        sorted_models = sorted(self.model_scores.items(), key=lambda x: x[1]["mae"])
        
        # Step 1: MASE 过滤
        mase_passed = []
        for m, scores in sorted_models:
            if scores["mase"] <= self.mase_threshold:
                mase_passed.append(m)
            else:
                self.exclusion_reasons[m] = f"mase={scores['mase']:.2f} > {self.mase_threshold}"
        
        # Step 2: 分位过滤
        if len(mase_passed) > 0:
            maes = [self.model_scores[m]["mae"] for m in mase_passed]
            pct_threshold = np.percentile(maes, self.threshold_pct)
            final_passed = []
            for m in mase_passed:
                if self.model_scores[m]["mae"] <= pct_threshold:
                    final_passed.append(m)
                else:
                    self.exclusion_reasons[m] = f"mae={self.model_scores[m]['mae']:.2f} > p{self.threshold_pct}={pct_threshold:.2f}"
        else:
            final_passed = []
        
        # 回退策略：至少保留 min_models 个
        if len(final_passed) < self.min_models:
            # 按 MAE 升序取前 min_models 个
            all_by_mae = [m for m, _ in sorted_models]
            final_passed = all_by_mae[:self.min_models]
            if self.verbose:
                print(f"    [ActiveModelsFilter] 过滤后仅剩 {len(final_passed)} 个，回退到 top-{self.min_models}")
        
        self.active_models = final_passed
        self.excluded_models = [m for m in available_models if m not in final_passed]
        self._is_fitted = True
        
        if self.verbose:
            print(f"    [ActiveModelsFilter] 活跃模型 ({len(self.active_models)}): {self.active_models}")
            if self.excluded_models:
                print(f"    [ActiveModelsFilter] 排除模型 ({len(self.excluded_models)}): {self.excluded_models}")
                for m in self.excluded_models:
                    reason = self.exclusion_reasons.get(m, "unknown")
                    print(f"        {m}: {reason}")
        
        return self.active_models
    
    def get_active_models(self) -> List[str]:
        """获取活跃模型列表"""
        if not self._is_fitted:
            raise RuntimeError("Filter not fitted. Call fit() first.")
        return self.active_models
    
    def get_model_scores(self) -> Dict[str, Dict[str, float]]:
        """获取所有模型的评分"""
        return self.model_scores
    
    def save(self, path: Path | str) -> None:
        """保存过滤结果到 JSON 文件"""
        path = Path(path)
        result = {
            "active_models": self.active_models,
            "excluded_models": self.excluded_models,
            "exclusion_reasons": self.exclusion_reasons,
            "model_scores": {m: {k: float(v) for k, v in scores.items()} for m, scores in self.model_scores.items()},
            "config": {
                "mase_threshold": self.mase_threshold,
                "threshold_pct": self.threshold_pct,
                "min_models": self.min_models,
            },
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Path | str) -> "ActiveModelsFilter":
        """从 JSON 文件加载过滤结果"""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        
        config = result.get("config", {})
        filter_obj = cls(
            mase_threshold=config.get("mase_threshold", 1.5),
            threshold_pct=config.get("threshold_pct", 75),
            min_models=config.get("min_models", 2),
            verbose=False,
        )
        filter_obj.active_models = result.get("active_models", [])
        filter_obj.excluded_models = result.get("excluded_models", [])
        filter_obj.exclusion_reasons = result.get("exclusion_reasons", {})
        filter_obj.model_scores = result.get("model_scores", {})
        filter_obj._is_fitted = True
        return filter_obj


def filter_predictions(
    P: np.ndarray,
    model_names: List[str],
    active_models: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    根据 active_models 过滤预测矩阵
    
    Args:
        P: 预测矩阵 [N, M]
        model_names: 模型名列表，长度 M
        active_models: 活跃模型列表
    
    Returns:
        P_filtered: 过滤后的预测矩阵 [N, len(active_indices)]
        filtered_names: 过滤后的模型名列表
    """
    active_indices = [i for i, m in enumerate(model_names) if m in active_models]
    if not active_indices:
        # 无活跃模型，返回原矩阵
        return P, model_names
    
    P_filtered = P[:, active_indices]
    filtered_names = [model_names[i] for i in active_indices]
    return P_filtered, filtered_names
