"""
自适应权重管理模块
实现基于场景和历史性能的动态权重调整
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class AdaptiveWeightManager:
    """自适应权重管理器
    
    根据场景特征和历史性能动态调整多目标优化权重
    支持三种模式：
    1. 规则驱动：基于场景类型的预设权重
    2. 性能驱动：基于历史最佳权重的学习
    3. 混合模式：规则与学习的加权组合
    """
    
    def __init__(self, mode: str = "hybrid"):
        """
        Args:
            mode: 权重调整模式 ("rule", "learning", "hybrid")
        """
        self.mode = mode
        self.weight_history = []  # [(scenario_signature, weights, performance)]
        self.scaler = StandardScaler()
        self.weight_model = None
        
        # 预设规则权重（基于场景类型）
        self.rule_weights = {
            "residential": {"error": 0.7, "latency": 0.2, "resource": 0.1},  # 住宅区重误差
            "charging": {"error": 0.5, "latency": 0.3, "resource": 0.2},    # 充电桩平衡型
            "service_area": {"error": 0.6, "latency": 0.25, "resource": 0.15},  # 服务区略重误差
            "default": {"error": 0.6, "latency": 0.2, "resource": 0.2}  # 默认权重
        }
    
    def get_adaptive_weights(self, scenario_signature: Dict[str, float], 
                            scenario_type: str = None) -> Dict[str, float]:
        """获取自适应权重
        
        Args:
            scenario_signature: 场景特征签名
            scenario_type: 场景类型 (residential/charging/service_area)
            
        Returns:
            权重字典 {"error": float, "latency": float, "resource": float}
        """
        if self.mode == "rule":
            return self._get_rule_based_weights(scenario_type)
        elif self.mode == "learning":
            return self._get_learning_based_weights(scenario_signature)
        else:  # hybrid
            rule_weights = self._get_rule_based_weights(scenario_type)
            if len(self.weight_history) < 10:
                # 样本不足时使用规则
                return rule_weights
            learning_weights = self._get_learning_based_weights(scenario_signature)
            # 混合：70% 学习 + 30% 规则
            return self._blend_weights(learning_weights, rule_weights, alpha=0.7)
    
    def _get_rule_based_weights(self, scenario_type: str = None) -> Dict[str, float]:
        """基于规则的权重"""
        if scenario_type and scenario_type in self.rule_weights:
            return self.rule_weights[scenario_type].copy()
        return self.rule_weights["default"].copy()
    
    def _get_learning_based_weights(self, scenario_signature: Dict[str, float]) -> Dict[str, float]:
        """基于学习的权重"""
        if not self.weight_history or len(self.weight_history) < 5:
            return self.rule_weights["default"].copy()
        
        # 训练轻量级权重预测模型（如果尚未训练）
        if self.weight_model is None:
            self._train_weight_model()
        
        # 提取特征向量
        feature_vector = self._extract_feature_vector(scenario_signature)
        
        # 预测权重
        try:
            predicted_weights = self.weight_model.predict([feature_vector])[0]
            # 归一化确保权重和为1
            total = sum(predicted_weights)
            return {
                "error": max(0.1, predicted_weights[0] / total),
                "latency": max(0.05, predicted_weights[1] / total),
                "resource": max(0.05, predicted_weights[2] / total)
            }
        except Exception as e:
            print(f"权重预测失败: {e}，回退到规则权重")
            return self.rule_weights["default"].copy()
    
    def _train_weight_model(self):
        """训练轻量级权重预测模型"""
        if len(self.weight_history) < 5:
            return
        
        # 提取训练数据
        X = []
        y_error, y_latency, y_resource = [], [], []
        
        for sig, weights, perf in self.weight_history:
            feature_vec = self._extract_feature_vector(sig)
            X.append(feature_vec)
            # weights 是 tuple: (error, latency, resource)
            y_error.append(weights[0])
            y_latency.append(weights[1])
            y_resource.append(weights[2])
        
        X = np.array(X)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练三个独立的回归器（岭回归，轻量级）
        from sklearn.multioutput import MultiOutputRegressor
        self.weight_model = MultiOutputRegressor(Ridge(alpha=1.0))
        y = np.column_stack([y_error, y_latency, y_resource])
        self.weight_model.fit(X_scaled, y)
        
        print(f"权重预测模型已训练，样本数: {len(X)}")
    
    def _extract_feature_vector(self, scenario_signature: Dict[str, float]) -> np.ndarray:
        """从场景签名提取特征向量"""
        # 选择关键特征
        key_features = [
            "mean_load", "cv_load", "peak_valley_ratio", 
            "day_night_ratio", "seasonal_amplitude", 
            "trend_strength", "region_type"
        ]
        
        feature_vec = []
        for feat in key_features:
            feature_vec.append(scenario_signature.get(feat, 0.0))
        
        return np.array(feature_vec)
    
    def _blend_weights(self, w1: Dict[str, float], w2: Dict[str, float], 
                      alpha: float = 0.5) -> Dict[str, float]:
        """混合两组权重"""
        blended = {}
        for key in w1.keys():
            blended[key] = alpha * w1[key] + (1 - alpha) * w2[key]
        
        # 归一化
        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()}
    
    def update_history(self, scenario_signature: Dict[str, float], 
                      weights: Dict[str, float], 
                      performance: Dict[str, float]):
        """更新历史记录
        
        Args:
            scenario_signature: 场景签名（字典格式）
            weights: 使用的权重（字典格式：{'error': w1, 'latency': w2, 'resource': w3}）
            performance: 实际性能（字典格式：{'error': e, 'latency': l, 'resource': r}）
        """
        # 转换为 tuple 格式存储（兼容现有代码）
        weights_tuple = (weights.get('error', 0.6), weights.get('latency', 0.2), weights.get('resource', 0.2))
        performance_tuple = (performance.get('error', 0), performance.get('latency', 0), performance.get('resource', 0))
        
        self.weight_history.append((scenario_signature, weights_tuple, performance_tuple))
        
        # 定期重训练模型（每10个样本）
        if len(self.weight_history) % 10 == 0:
            self.weight_model = None  # 标记需要重训练
    
    def get_weight_statistics(self) -> Dict[str, any]:
        """获取权重统计信息"""
        if not self.weight_history:
            return {"count": 0}
        
        errors = [w["error"] for _, w, _ in self.weight_history]
        latencies = [w["latency"] for _, w, _ in self.weight_history]
        resources = [w["resource"] for _, w, _ in self.weight_history]
        
        return {
            "count": len(self.weight_history),
            "error_weight": {"mean": np.mean(errors), "std": np.std(errors)},
            "latency_weight": {"mean": np.mean(latencies), "std": np.std(latencies)},
            "resource_weight": {"mean": np.mean(resources), "std": np.std(resources)}
        }
