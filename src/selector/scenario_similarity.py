from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import ks_2samp, pearsonr
from scipy.spatial.distance import euclidean, cosine
import warnings

# [Fix] 不再全局屏蔽警告，改为在特定计算中局部屏蔽
# warnings.filterwarnings('ignore')  # 已移除


class PowerScenarioAnalyzer:
    """电力场景分析器，专门用于电力需求预测的场景相似度计算"""
    
    def __init__(self):
        self.scenario_cache = {}
        
    def extract_scenario_signature(self, df: pd.DataFrame, region_type: str = None) -> Dict[str, float]:
        """提取电力场景特征签名
        
        ========== 场景编码: 手工统计特征 (技术限制) ==========
        【当前实现】纯统计特征提取
          - 负荷统计: mean/std/cv/peak_valley_ratio
          - 时间模式: day_night_ratio/weekend_weekday_ratio
          - 相关性: load_temp_corr/load_humidity_corr
        【局限性】
          - 无深度嵌入: 未使用 Transformer/LSTM 自动特征学习
          - 缺失事件特征: extreme_weather_alert/anomaly_detected
          - 表达能力受限: 无法捕捉复杂非线性模式
        【改进方向】见 docs/technical_limitations.md 第3节
        ========================================================
        """
        signature = {}
        
        # 基础统计特征
        signature["mean_load"] = float(df["load"].mean())
        signature["std_load"] = float(df["load"].std())
        signature["max_load"] = float(df["load"].max())
        signature["min_load"] = float(df["load"].min())
        signature["load_range"] = signature["max_load"] - signature["min_load"]
        signature["cv_load"] = signature["std_load"] / (signature["mean_load"] + 1e-6)  # 变异系数
        
        # 时间模式特征
        if "hour" in df.columns:
            signature.update(self._extract_temporal_patterns(df))
            
        # 区域特征
        if region_type:
            signature["region_type"] = self._encode_region_type(region_type)
            
        # 季节性特征
        if "timestamp" in df.columns:
            signature.update(self._extract_seasonal_patterns(df))
            
        # 天气相关特征
        weather_cols = [col for col in df.columns if col in ['temp', 'humidity', 'wind', 'weather_comfort']]
        if weather_cols:
            signature.update(self._extract_weather_correlation(df, weather_cols))
            
        # 负荷分布特征
        signature.update(self._extract_load_distribution(df))
        
        return signature
    
    def _extract_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """提取时间模式特征"""
        patterns = {}
        
        # 日内模式
        if "hour" in df.columns:
            hourly_mean = df.groupby("hour")["load"].mean()
            patterns["peak_hour"] = float(hourly_mean.idxmax())
            patterns["valley_hour"] = float(hourly_mean.idxmin())
            patterns["peak_valley_ratio"] = float(hourly_mean.max() / (hourly_mean.min() + 1e-6))
            
            # 白天夜间比例
            day_mask = df["hour"].between(8, 20)
            night_mask = ~day_mask
            patterns["day_night_ratio"] = float(
                (df[day_mask]["load"].mean() + 1e-6) / (df[night_mask]["load"].mean() + 1e-6)
            )
            
            # 工作时间模式
            work_mask = df["hour"].between(9, 17)
            patterns["work_hour_intensity"] = float(df[work_mask]["load"].mean() / (df["load"].mean() + 1e-6))
        
        # 周模式
        if "is_weekend" in df.columns:
            weekend_mask = df["is_weekend"] == 1
            weekday_mask = df["is_weekend"] == 0
            if weekend_mask.sum() > 0 and weekday_mask.sum() > 0:
                patterns["weekend_weekday_ratio"] = float(
                    (df[weekend_mask]["load"].mean() + 1e-6) / (df[weekday_mask]["load"].mean() + 1e-6)
                )
            else:
                patterns["weekend_weekday_ratio"] = 1.0
                
        return patterns
    
    def _extract_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """提取季节性模式特征"""
        patterns = {}
        
        if "timestamp" in df.columns:
            df_temp = df.copy()
            df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
            df_temp["month"] = df_temp["timestamp"].dt.month
            df_temp["day_of_year"] = df_temp["timestamp"].dt.dayofyear
            
            # 月度变化
            monthly_mean = df_temp.groupby("month")["load"].mean()
            patterns["seasonal_amplitude"] = float((monthly_mean.max() - monthly_mean.min()) / (monthly_mean.mean() + 1e-6))
            
            # 趋势分析
            if len(df_temp) > 24:  # 至少一天的数据
                x = np.arange(len(df_temp))
                load_values = df_temp["load"].fillna(df_temp["load"].mean())
                if not load_values.isna().all() and load_values.std() > 0:
                    # [Fix] 局部屏蔽 pearsonr 在常数输入时的警告
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        correlation, _ = pearsonr(x, load_values)
                    if not np.isnan(correlation):
                        patterns["trend_strength"] = float(abs(correlation))
                        patterns["trend_direction"] = float(np.sign(correlation))
                    else:
                        patterns["trend_strength"] = 0.0
                        patterns["trend_direction"] = 0.0
                else:
                    patterns["trend_strength"] = 0.0
                    patterns["trend_direction"] = 0.0
                
        return patterns
    
    def _extract_weather_correlation(self, df: pd.DataFrame, weather_cols: List[str]) -> Dict[str, float]:
        """提取天气相关特征"""
        correlations = {}
        
        for col in weather_cols:
            if col in df.columns and not df[col].isna().all():
                # [Fix] 局部屏蔽 pearsonr 在常数输入时的警告
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    corr, _ = pearsonr(df["load"].fillna(df["load"].mean()), 
                                     df[col].fillna(df[col].mean()))
                correlations[f"load_{col}_corr"] = float(abs(corr)) if not np.isnan(corr) else 0.0
                
        return correlations
    
    def _extract_load_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """提取负荷分布特征"""
        distribution = {}
        
        load_values = df["load"].dropna()
        if len(load_values) > 0:
            # 分位数特征
            distribution["load_q25"] = float(load_values.quantile(0.25))
            distribution["load_q50"] = float(load_values.quantile(0.50))
            distribution["load_q75"] = float(load_values.quantile(0.75))
            distribution["load_iqr"] = distribution["load_q75"] - distribution["load_q25"]
            
            # 偏度和峰度
            distribution["load_skewness"] = float(load_values.skew())
            distribution["load_kurtosis"] = float(load_values.kurtosis())
            
            # 负荷因子（平均负荷/最大负荷）
            distribution["load_factor"] = float(load_values.mean() / (load_values.max() + 1e-6))
            
        return distribution
    
    def _encode_region_type(self, region_type: str) -> float:
        """编码区域类型"""
        type_mapping = {
            "residential": 1.0,
            "charging": 2.0, 
            "service_area": 3.0,
            "industrial": 4.0,
            "commercial": 5.0
        }
        return type_mapping.get(region_type.lower(), 0.0)
    
    def calculate_scenario_similarity(self, sig_a: Dict[str, float], sig_b: Dict[str, float], 
                                    method: str = "combined") -> float:
        """计算场景相似度"""
        if method == "weighted_euclidean":
            return self._weighted_euclidean_similarity(sig_a, sig_b)
        elif method == "cosine":
            return self._cosine_similarity(sig_a, sig_b)
        elif method == "manhattan":
            return self._manhattan_similarity(sig_a, sig_b)
        elif method == "combined":
            return self._combined_similarity(sig_a, sig_b)
        else:
            return self._normalized_euclidean_similarity(sig_a, sig_b)

    def _combined_similarity(self, sig_a: Dict[str, float], sig_b: Dict[str, float]) -> float:
        """组合相似度 (Region Type 50% + Euclidean 50%)"""
        # 1. 区域类型匹配 (50%)
        region_match = 1.0 if sig_a.get("region_type") == sig_b.get("region_type") else 0.0
        
        # 2. 欧氏距离相似度 (50%)
        sim_euclidean = self._weighted_euclidean_similarity(sig_a, sig_b)
        
        return 0.5 * region_match + 0.5 * sim_euclidean
    
    def _weighted_euclidean_similarity(self, sig_a: Dict[str, float], sig_b: Dict[str, float]) -> float:
        """加权欧几里得相似度 (纯统计特征，无事件/深度嵌入)"""
        # 基础特征权重
        base_weights = {
            "mean_load": 0.2,
            "peak_valley_ratio": 0.15,
            "day_night_ratio": 0.15,
            "weekend_weekday_ratio": 0.1,
            "load_factor": 0.1,
            "seasonal_amplitude": 0.1,
            "region_type": 0.05,
            "cv_load": 0.05,
            "trend_strength": 0.05,
            "work_hour_intensity": 0.05
        }
        
        # 仅使用基础统计特征权重
        attention_weights = base_weights.copy()
                
        common_keys = set(sig_a.keys()) & set(sig_b.keys())
        if not common_keys:
            return 0.0
            
        weighted_distance = 0.0
        total_weight = 0.0
        
        for key in common_keys:
            weight = attention_weights.get(key, 0.01)
            diff = abs(sig_a[key] - sig_b[key])
            
            # 归一化差异
            avg_val = (abs(sig_a[key]) + abs(sig_b[key])) / 2.0
            normalized_diff = diff / (avg_val + 1e-6)
            
            weighted_distance += weight * normalized_diff ** 2
            total_weight += weight
            
        if total_weight == 0:
            return 0.0
            
        # 转换为相似度（0-1，1表示完全相似）
        distance = np.sqrt(weighted_distance / total_weight)
        similarity = 1.0 / (1.0 + distance)
        
        return float(similarity)
    
    def _cosine_similarity(self, sig_a: Dict[str, float], sig_b: Dict[str, float]) -> float:
        """余弦相似度"""
        common_keys = sorted(set(sig_a.keys()) & set(sig_b.keys()))
        if not common_keys:
            return 0.0
            
        vec_a = np.array([sig_a[k] for k in common_keys])
        vec_b = np.array([sig_b[k] for k in common_keys])
        
        # 避免零向量
        if np.linalg.norm(vec_a) == 0 or np.linalg.norm(vec_b) == 0:
            return 0.0
            
        similarity = 1.0 - cosine(vec_a, vec_b)
        return float(max(0.0, similarity))
    
    def _manhattan_similarity(self, sig_a: Dict[str, float], sig_b: Dict[str, float]) -> float:
        """曼哈顿相似度"""
        common_keys = set(sig_a.keys()) & set(sig_b.keys())
        if not common_keys:
            return 0.0
            
        total_diff = 0.0
        for key in common_keys:
            diff = abs(sig_a[key] - sig_b[key])
            avg_val = (abs(sig_a[key]) + abs(sig_b[key])) / 2.0
            normalized_diff = diff / (avg_val + 1e-6)
            total_diff += normalized_diff
            
        # 转换为相似度
        similarity = 1.0 / (1.0 + total_diff / len(common_keys))
        return float(similarity)
    
    def _normalized_euclidean_similarity(self, sig_a: Dict[str, float], sig_b: Dict[str, float]) -> float:
        """归一化欧几里得相似度"""
        common_keys = set(sig_a.keys()) & set(sig_b.keys())
        if not common_keys:
            return 0.0
            
        distances = []
        for key in common_keys:
            diff = abs(sig_a[key] - sig_b[key])
            avg_val = (abs(sig_a[key]) + abs(sig_b[key])) / 2.0
            normalized_diff = diff / (avg_val + 1e-6)
            distances.append(normalized_diff ** 2)
            
        distance = np.sqrt(np.mean(distances))
        similarity = 1.0 / (1.0 + distance)
        
        return float(similarity)
    
    def find_similar_scenarios(self, target_signature: Dict[str, float], 
                             historical_signatures: List[Tuple[str, Dict[str, float]]], 
                             top_k: int = 5, similarity_threshold: float = 0.3) -> List[Tuple[str, float]]:
        """找到最相似的历史场景"""
        similarities = []
        
        for scenario_id, hist_signature in historical_signatures:
            # 使用组合相似度
            similarity = self.calculate_scenario_similarity(target_signature, hist_signature, method="combined")
            if similarity >= similarity_threshold:
                similarities.append((scenario_id, similarity))
                
        # 按相似度排序，返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# 保持向后兼容的函数
def scenario_signature(df: pd.DataFrame) -> Dict[str, float]:
    """向后兼容的场景签名函数"""
    analyzer = PowerScenarioAnalyzer()
    return analyzer.extract_scenario_signature(df)


def signature_distance(sig_a: Dict[str, float], sig_b: Dict[str, float]) -> float:
    """向后兼容的距离计算函数"""
    analyzer = PowerScenarioAnalyzer()
    similarity = analyzer.calculate_scenario_similarity(sig_a, sig_b)
    return 1.0 - similarity  # 转换为距离
