"""
资源消耗预测模块
基于轻量级机器学习模型预测模型资源消耗
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os


class ResourcePredictor:
    """资源消耗预测器
    
    使用轻量级梯度提升树预测模型的资源消耗
    特征：数据规模、特征维度、模型类型、历史性能
    目标：延迟(ms)、内存(MB)、CPU(%)
    """
    
    def __init__(self):
        self.latency_model = None
        self.memory_model = None
        self.cpu_model = None
        self.scaler = StandardScaler()
        self.training_data = []  # [(features, targets)]
        self.fitted = False
        
        # 模型类型编码
        self.model_type_encoding = {
            "arima": 1, "prophet": 2,
            "xgboost_reg": 3, "lgbm_reg": 4, "catboost_reg": 5,
            "power_difference": 6, "multimodal_fusion": 7,
            "unknown": 0
        }
    
    def fit(self, historical_data: List[Tuple[str, int, int, Dict[str, float]]]) -> None:
        """训练资源预测模型
        
        Args:
            historical_data: [(model_id, data_size, num_features, metrics), ...]
        """
        # 统一阈值：至少5条样本才训练
        if len(historical_data) < 5:
            print(f"  [ResourcePredictor] 样本不足 ({len(historical_data)}<5)，跳过训练")
            return
            
        X = []
        y_latency = []
        y_memory = []
        y_cpu = []
        
        normalized_training = []
        for model_id, size, feats, metrics in historical_data:
            # 提取特征
            feature_vec = self._extract_features(model_id, size, feats)
            X.append(feature_vec)
            
            # 提取目标
            lat = metrics.get('latency_ms', 100.0)
            mem = metrics.get('memory_mb', 50.0)
            cpu = metrics.get('cpu_percent', 10.0)
            y_latency.append(lat)
            y_memory.append(mem)
            y_cpu.append(cpu)

            # 规范化训练记录，保持 (features, targets) 结构
            normalized_training.append((feature_vec, np.array([lat, mem, cpu])))
            
        X = np.array(X)
        
        # 训练模型
        try:
            # 标准化
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # 初始化模型 (如果尚未初始化)
            if self.latency_model is None:
                self.latency_model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
                self.memory_model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
                self.cpu_model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
            
            self.latency_model.fit(X_scaled, y_latency)
            self.memory_model.fit(X_scaled, y_memory)
            self.cpu_model.fit(X_scaled, y_cpu)
            
            # 更新训练数据记录 (用于 predict_resources 的检查)
            self.training_data = normalized_training
            self.fitted = True
            print(f"  [ResourcePredictor] 模型已训练 (样本数: {len(X)})")
            
        except Exception as e:
            print(f"  [ResourcePredictor] 训练失败: {e}")

    def predict_resources(self, model_id: str, data_size: int, 
                         num_features: int) -> Dict[str, float]:
        """预测资源消耗
        
        Args:
            model_id: 模型ID
            data_size: 数据样本数
            num_features: 特征维度
            
        Returns:
            {"latency_ms": float, "memory_mb": float, "cpu_percent": float}
        """
        # 统一阈值检查：与 fit 的阈值保持一致
        if not self.fitted:
            # 回退到基于规则的估计
            return self._rule_based_estimation(model_id, data_size, num_features)
        
        # 提取特征
        feature_vec = self._extract_features(model_id, data_size, num_features)
        feature_scaled = self.scaler.transform([feature_vec])
        
        try:
            latency = max(10.0, self.latency_model.predict(feature_scaled)[0])
            memory = max(10.0, self.memory_model.predict(feature_scaled)[0])
            cpu = max(5.0, min(100.0, self.cpu_model.predict(feature_scaled)[0]))
            
            return {
                "latency_ms": latency,
                "memory_mb": memory,
                "cpu_percent": cpu
            }
        except Exception as e:
            print(f"资源预测失败: {e}，回退到规则估计")
            return self._rule_based_estimation(model_id, data_size, num_features)
    
    def _extract_features(self, model_id: str, data_size: int, 
                         num_features: int) -> np.ndarray:
        """提取特征向量"""
        # 基础特征
        features = [
            np.log1p(data_size),  # 对数变换数据规模
            num_features,
            num_features / (data_size + 1),  # 特征密度
        ]
        
        # 模型类型 one-hot 编码
        model_type = self.model_type_encoding.get(model_id, 0)
        for i in range(len(self.model_type_encoding)):
            features.append(1.0 if i == model_type else 0.0)
        
        # 交互特征
        features.append(np.log1p(data_size) * num_features)
        
        return np.array(features)
    
    def _rule_based_estimation(self, model_id: str, data_size: int, 
                               num_features: int) -> Dict[str, float]:
        """基于规则的资源估计（回退策略）"""
        # 基础消耗（取决于模型类型）
        base_costs = {
            "arima": {"latency": 100, "memory": 50, "cpu": 30},
            "prophet": {"latency": 150, "memory": 80, "cpu": 40},
            "xgboost_reg": {"latency": 80, "memory": 100, "cpu": 60},
            "lgbm_reg": {"latency": 60, "memory": 80, "cpu": 50},
            "catboost_reg": {"latency": 90, "memory": 120, "cpu": 55},
            "power_difference": {"latency": 50, "memory": 40, "cpu": 25},
            "multimodal_fusion": {"latency": 200, "memory": 150, "cpu": 70}
        }
        
        base = base_costs.get(model_id, {"latency": 100, "memory": 80, "cpu": 40})
        
        # 根据数据规模调整
        size_factor = np.log1p(data_size) / 10.0
        feature_factor = num_features / 50.0
        
        return {
            "latency_ms": base["latency"] * (1 + 0.3 * size_factor),
            "memory_mb": base["memory"] * (1 + 0.5 * size_factor + 0.2 * feature_factor),
            "cpu_percent": base["cpu"] * (1 + 0.2 * size_factor)
        }
    
    def add_observation(self, model_id: str, data_size: int, num_features: int,
                       actual_latency: float, actual_memory: float, 
                       actual_cpu: float):
        """添加真实观测数据
        
        Args:
            model_id: 模型ID
            data_size: 数据样本数
            num_features: 特征维度
            actual_latency: 实际延迟(ms)
            actual_memory: 实际内存(MB)
            actual_cpu: 实际CPU使用率(%)
        """
        features = self._extract_features(model_id, data_size, num_features)
        targets = np.array([actual_latency, actual_memory, actual_cpu])
        self.training_data.append((features, targets))
        
        # 首次达到5个样本时立即训练，之后每10个样本重训练
        n_samples = len(self.training_data)
        if n_samples == 5 or (n_samples > 5 and n_samples % 10 == 0):
            self._train_models()
            print(f"  [资源预测] 模型已更新，当前样本数: {n_samples}")
    
    def _train_models(self):
        """训练资源预测模型"""
        if len(self.training_data) < 5:
            return
        
        print(f"训练资源预测模型，样本数: {len(self.training_data)}")
        
        # 兼容历史格式：将 (model_id, size, feats, metrics) 转换为 (features, targets)
        normalized_data = []
        for item in self.training_data:
            if len(item) == 2:
                normalized_data.append(item)
            elif len(item) == 4:
                model_id, size, feats, metrics = item
                feat_vec = self._extract_features(model_id, size, feats)
                tgt = np.array([
                    metrics.get('latency_ms', 100.0),
                    metrics.get('memory_mb', 50.0),
                    metrics.get('cpu_percent', 10.0)
                ])
                normalized_data.append((feat_vec, tgt))
            else:
                # 跳过不合规记录
                continue
        self.training_data = normalized_data
        
        # 准备训练数据
        X = np.array([feat for feat, _ in self.training_data])
        y_latency = np.array([tgt[0] for _, tgt in self.training_data])
        y_memory = np.array([tgt[1] for _, tgt in self.training_data])
        y_cpu = np.array([tgt[2] for _, tgt in self.training_data])
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练三个独立的回归器（轻量级梯度提升）
        self.latency_model = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        self.memory_model = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        self.cpu_model = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        
        self.latency_model.fit(X_scaled, y_latency)
        self.memory_model.fit(X_scaled, y_memory)
        self.cpu_model.fit(X_scaled, y_cpu)
        
        self.fitted = True
        
        # 计算训练集性能指标
        y_pred_latency = self.latency_model.predict(X_scaled)
        y_pred_memory = self.memory_model.predict(X_scaled)
        y_pred_cpu = self.cpu_model.predict(X_scaled)
        
        mae_latency = np.mean(np.abs(y_latency - y_pred_latency))
        mae_memory = np.mean(np.abs(y_memory - y_pred_memory))
        mae_cpu = np.mean(np.abs(y_cpu - y_pred_cpu))
        
        # 计算 R2 分数
        from sklearn.metrics import r2_score
        r2_latency = r2_score(y_latency, y_pred_latency)
        r2_memory = r2_score(y_memory, y_pred_memory)
        r2_cpu = r2_score(y_cpu, y_pred_cpu)
        
        print(f"  [资源预测训练] 延迟MAE: {mae_latency:.2f}ms (R²={r2_latency:.3f}), "
              f"内存MAE: {mae_memory:.2f}MB (R²={r2_memory:.3f}), "
              f"CPU MAE: {mae_cpu:.2f}% (R²={r2_cpu:.3f})")
    
    def save_models(self, directory: str):
        """保存模型到目录（包括训练数据和模型）"""
        model_path = os.path.join(directory, "resource_models.pkl")
        model_data = {
            "latency_model": self.latency_model if self.fitted else None,
            "memory_model": self.memory_model if self.fitted else None,
            "cpu_model": self.cpu_model if self.fitted else None,
            "scaler": self.scaler,
            "training_data": self.training_data,
            "model_type_encoding": self.model_type_encoding,
            "fitted": self.fitted
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        if self.fitted:
            print(f"资源预测模型已保存到: {model_path} (包含 {len(self.training_data)} 个样本)")
        else:
            print(f"资源预测器训练数据已保存: {len(self.training_data)} 个样本（尚未训练）")
    
    def load_models(self, directory: str):
        """从目录加载模型"""
        model_path = os.path.join(directory, "resource_models.pkl")
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.latency_model = model_data.get("latency_model")
        self.memory_model = model_data.get("memory_model")
        self.cpu_model = model_data.get("cpu_model")
        self.scaler = model_data["scaler"]
        self.training_data = model_data.get("training_data", [])
        self.model_type_encoding = model_data.get("model_type_encoding", self.model_type_encoding)
        self.fitted = model_data.get("fitted", False)
        
        if self.fitted:
            print(f"资源预测模型已加载，样本数: {len(self.training_data)}")
        else:
            print(f"资源预测器训练数据已加载: {len(self.training_data)} 个样本（尚未训练）")
    
    def get_model_statistics(self) -> Dict[str, any]:
        """获取模型统计信息"""
        if not self.training_data:
            return {"count": 0, "fitted": False}
        
        latencies = [tgt[0] for _, tgt in self.training_data]
        memories = [tgt[1] for _, tgt in self.training_data]
        cpus = [tgt[2] for _, tgt in self.training_data]
        
        return {
            "count": len(self.training_data),
            "fitted": self.fitted,
            "latency_stats": {"mean": np.mean(latencies), "std": np.std(latencies)},
            "memory_stats": {"mean": np.mean(memories), "std": np.std(memories)},
            "cpu_stats": {"mean": np.mean(cpus), "std": np.std(cpus)}
        }
