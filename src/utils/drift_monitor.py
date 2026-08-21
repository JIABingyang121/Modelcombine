"""
漂移监控模块 (P3.1)

功能：
1. 计算关键特征的 PSI (Population Stability Index)
2. 计算残差分布的 KS (Kolmogorov-Smirnov) 检验
3. 仅对大数据集启用（默认 PJM / AEMO）
4. 触发规则：PSI > 0.2 或 KS > 0.1 时记录 drift_event

窗口粒度：
- PJM: 168h（一周）
- AEMO: 168h（一周）
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import numpy as np
from datetime import datetime


# ============================================================================
# 配置常量
# ============================================================================

# 每个数据集的窗口大小和是否启用
DRIFT_CONFIG = {
    "pjm": {"enabled": True, "window_size": 168, "min_samples": 500},
    "aemo_vic": {"enabled": True, "window_size": 168, "min_samples": 500},
    "aemo_nsw": {"enabled": True, "window_size": 168, "min_samples": 500},
}

# 默认阈值
DEFAULT_PSI_THRESHOLD = 0.2
DEFAULT_KS_THRESHOLD = 0.1
SCORE_PENALTY = 0.05  # 每个 drift event 的基础惩罚


@dataclass
class DriftEvent:
    """单个漂移事件"""
    dataset: str
    horizon: int
    window_start: int
    window_end: int
    metric_type: str  # "psi" or "ks"
    feature_name: str
    value: float
    threshold: float
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DriftMonitor:
    """
    漂移监控器
    
    对滑动窗口内的特征分布变化和残差分布变化进行检测。
    """
    
    def __init__(
        self,
        psi_threshold: float = DEFAULT_PSI_THRESHOLD,
        ks_threshold: float = DEFAULT_KS_THRESHOLD,
        n_bins: int = 10,
    ):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.n_bins = n_bins
    
    @staticmethod
    def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """
        计算 PSI (Population Stability Index)
        
        PSI = Σ (p_i - q_i) * ln(p_i / q_i)
        
        Args:
            reference: 参考分布（训练集/前一窗口）
            current: 当前分布
            n_bins: 分箱数
        
        Returns:
            PSI 值（>0.2 通常认为分布有显著变化）
        """
        if len(reference) < n_bins or len(current) < n_bins:
            return 0.0
        
        # 用参考分布的分位数作为分箱边界
        breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        # 去除重复断点
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 3:
            return 0.0
        
        ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
        cur_counts = np.histogram(current, bins=breakpoints)[0].astype(float)
        
        # 归一化为概率
        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()
        
        # 防零
        eps = 1e-6
        ref_pct = np.clip(ref_pct, eps, 1.0)
        cur_pct = np.clip(cur_pct, eps, 1.0)
        
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return psi
    
    @staticmethod
    def _compute_ks(reference: np.ndarray, current: np.ndarray) -> float:
        """
        计算 KS (Kolmogorov-Smirnov) 统计量
        
        Returns:
            KS 统计量（0~1，越大说明分布差异越大）
        """
        if len(reference) < 5 or len(current) < 5:
            return 0.0
        
        # 合并排序
        all_values = np.sort(np.concatenate([reference, current]))
        
        # 计算经验 CDF
        ref_cdf = np.searchsorted(np.sort(reference), all_values, side='right') / len(reference)
        cur_cdf = np.searchsorted(np.sort(current), all_values, side='right') / len(current)
        
        ks = float(np.max(np.abs(ref_cdf - cur_cdf)))
        return ks
    
    def check_drift(
        self,
        dataset: str,
        horizon: int,
        train_features: np.ndarray,
        test_features: np.ndarray,
        feature_names: List[str],
        train_residuals: np.ndarray = None,
        test_residuals: np.ndarray = None,
    ) -> List[DriftEvent]:
        """
        检测特征漂移和残差漂移
        
        Args:
            dataset: 数据集名称
            horizon: 预测步长
            train_features: 训练集特征矩阵 [n_train, n_features]
            test_features: 测试集特征矩阵 [n_test, n_features]
            feature_names: 特征名列表
            train_residuals: 训练集残差（可选）
            test_residuals: 测试集残差（可选）
        
        Returns:
            DriftEvent 列表
        """
        config = DRIFT_CONFIG.get(dataset, {"enabled": False})
        if not config["enabled"]:
            return []
        
        if len(train_features) < config["min_samples"]:
            return []
        
        events = []
        now = datetime.now().isoformat()
        window = config["window_size"]
        
        # 特征 PSI 检测（用训练集最后 window 作为参考）
        n_train = len(train_features)
        ref_start = max(0, n_train - window)
        
        for i, fname in enumerate(feature_names):
            if i >= train_features.shape[1] or i >= test_features.shape[1]:
                break
            
            ref = train_features[ref_start:, i]
            cur = test_features[:, i]
            
            # 过滤 NaN
            ref = ref[~np.isnan(ref)]
            cur = cur[~np.isnan(cur)]
            
            if len(ref) < 10 or len(cur) < 10:
                continue
            
            psi = self._compute_psi(ref, cur, self.n_bins)
            if psi > self.psi_threshold:
                events.append(DriftEvent(
                    dataset=dataset,
                    horizon=horizon,
                    window_start=ref_start,
                    window_end=n_train,
                    metric_type="psi",
                    feature_name=fname,
                    value=round(psi, 4),
                    threshold=self.psi_threshold,
                    timestamp=now,
                ))
        
        # 残差 KS 检测
        if train_residuals is not None and test_residuals is not None:
            r_train = train_residuals[~np.isnan(train_residuals)]
            r_test = test_residuals[~np.isnan(test_residuals)]
            
            if len(r_train) >= 10 and len(r_test) >= 10:
                ks = self._compute_ks(r_train, r_test)
                if ks > self.ks_threshold:
                    events.append(DriftEvent(
                        dataset=dataset,
                        horizon=horizon,
                        window_start=0,
                        window_end=len(train_residuals),
                        metric_type="ks",
                        feature_name="residuals",
                        value=round(ks, 4),
                        threshold=self.ks_threshold,
                        timestamp=now,
                    ))
        
        return events


def save_drift_events(events: List[DriftEvent], output_path: Path):
    """保存漂移事件到 JSON"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in events], f, ensure_ascii=False, indent=2)


def compute_drift_penalty(
    events: List[DriftEvent],
    val_test_gap_pct: float = 0.0,
    strategy_name: str = "",
    base_penalty_per_event: float = SCORE_PENALTY,
) -> float:
    """
    动态漂移惩罚：
    1) 先按事件严重度累计惩罚（value/threshold 越高罚越重）；
    2) 再按 |val-test gap| 放大或缩小，gap 小表示更鲁棒。
    """
    if not events:
        return 0.0

    severity_penalty = 0.0
    for e in events:
        threshold = max(float(getattr(e, "threshold", 0.0)), 1e-6)
        value = float(getattr(e, "value", 0.0))
        severity = value / threshold
        if severity > 5.0:
            severity_penalty += base_penalty_per_event * 2.0
        elif severity > 2.0:
            severity_penalty += base_penalty_per_event * 1.5
        else:
            severity_penalty += base_penalty_per_event

    gap_abs = abs(float(val_test_gap_pct)) if np.isfinite(val_test_gap_pct) else 0.0
    if gap_abs <= 5.0:
        gap_multiplier = 0.5
    elif gap_abs <= 15.0:
        gap_multiplier = 1.0
    elif gap_abs <= 30.0:
        gap_multiplier = 1.5
    else:
        gap_multiplier = 2.0

    return float(severity_penalty * gap_multiplier)
