"""
深度场景表征
使用深度学习模型学习场景的低维嵌入表示，替代手工特征
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
import pickle
from pathlib import Path
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import holidays
import datetime


class ScenarioEncoder:
    """场景编码器（基于非线性核方法的场景表征学习）"""
    
    def __init__(self, encoding_dim: int = 16):
        """
        Args:
            encoding_dim: 编码维度
        """
        self.encoding_dim = encoding_dim
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        
        # 使用 KernelPCA 替代简单的 SVD，捕捉非线性特征
        self.encoder = KernelPCA(
            n_components=encoding_dim, 
            kernel='rbf', 
            fit_inverse_transform=True,
            n_jobs=-1
        )
        
    def fit(self, scenarios: List[Dict[str, any]]) -> None:
        """训练场景编码器
        
        Args:
            scenarios: 历史场景列表（每个场景包含特征字典）
        """
        if len(scenarios) < 3:
            return
        
        # 提取特征矩阵
        X = self._extract_features(scenarios)
        if X is None or len(X) < 3:
            return
        
        # 标准化
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0) + 1e-8
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        
        # 训练非线性编码器
        try:
            self.encoder.fit(X_scaled)
            self.is_fitted = True
            print(f"  [SceneEncoder] 已训练 KernelPCA (dim={self.encoding_dim}, samples={len(X)})")
        except Exception as e:
            print(f"  [SceneEncoder] 训练失败: {e}")
    
    def encode(self, scenario: Dict[str, any]) -> np.ndarray:
        """将场景编码为低维向量
        
        Args:
            scenario: 场景字典
            
        Returns:
            编码向量 (encoding_dim,)
        """
        if not self.is_fitted:
            # 未训练时返回零向量
            return np.zeros(self.encoding_dim)
        
        # 提取特征
        features = self._extract_single_scenario_features(scenario)
        if features is None:
            return np.zeros(self.encoding_dim)
        
        # 标准化
        features_scaled = (features - self.scaler_mean) / self.scaler_std
        
        # 编码
        encoded = self.encoder.transform([features_scaled])[0]
        
        return encoded
    
    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """解码（重构场景特征）
        
        Args:
            encoded: 编码向量
            
        Returns:
            重构的特征向量
        """
        if not self.is_fitted:
            return np.zeros(len(self.feature_names))
        
        # 解码 (KernelPCA 支持 inverse_transform)
        decoded_scaled = self.encoder.inverse_transform([encoded])[0]
        
        # 反标准化
        decoded = decoded_scaled * self.scaler_std + self.scaler_mean
        
        return decoded
    
    def compute_similarity(self, scenario1: Dict[str, any], 
                          scenario2: Dict[str, any]) -> float:
        """计算两个场景的相似度（基于编码的余弦相似度）
        
        Args:
            scenario1: 场景1
            scenario2: 场景2
            
        Returns:
            相似度分数 [0, 1]
        """
        enc1 = self.encode(scenario1)
        enc2 = self.encode(scenario2)
        
        # 余弦相似度
        norm1 = np.linalg.norm(enc1)
        norm2 = np.linalg.norm(enc2)
        
        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0
        
        similarity = np.dot(enc1, enc2) / (norm1 * norm2)
        
        # 映射到 [0, 1]
        similarity = (similarity + 1.0) / 2.0
        
        return float(similarity)
    
    def _extract_features(self, scenarios: List[Dict[str, any]]) -> Optional[np.ndarray]:
        """从场景列表提取特征矩阵
        
        Args:
            scenarios: 场景列表
            
        Returns:
            特征矩阵 (n_scenarios, n_features)
        """
        feature_list = []
        expected_len = None
        
        for scenario in scenarios:
            features = self._extract_single_scenario_features(scenario)
            if features is not None:
                # 确保所有特征向量长度一致
                if expected_len is None:
                    expected_len = len(features)
                    feature_list.append(features)
                elif len(features) == expected_len:
                    feature_list.append(features)
                # 跳过长度不一致的场景（避免齐次数组错误）
        
        if len(feature_list) == 0:
            return None
        
        return np.array(feature_list)
    
    def _extract_single_scenario_features(self, scenario: Dict[str, any]) -> Optional[np.ndarray]:
        """从单个场景提取特征向量
        
        Args:
            scenario: 场景字典
            
        Returns:
            特征向量
        """
        # 提取场景中的所有数值型特征
        features = []
        feature_names = []
        
        # 基础统计特征 (兼容不同命名风格: load_mean vs mean_load)
        stat_keys = [
            ('load_mean', 'mean_load'), 
            ('load_std', 'std_load'), 
            ('load_max', 'max_load'), 
            ('load_min', 'min_load'),
            ('temp_mean', 'mean_temp'), 
            ('temp_std', 'std_temp'), 
            ('humidity_mean', 'mean_humidity')
        ]
        
        for k1, k2 in stat_keys:
            val = scenario.get(k1, scenario.get(k2))
            if val is not None:
                features.append(float(val))
                feature_names.append(k1) # 统一使用 k1 作为内部名
        
        # 时间特征
        time_keys = ['hour', 'day_of_week', 'month', 'is_weekend']
        for key in time_keys:
            if key in scenario:
                features.append(float(scenario[key]))
                feature_names.append(key)
        
        # 增强事件特征 (Holidays)
        is_holiday = 0.0
        # 使用配置的国别参数，默认中国节假日
        holiday_country = scenario.get('_holiday_country', 'CN')
        try:
            holiday_cal = holidays.country_holidays(holiday_country)
        except:
            holiday_cal = holidays.CN()
        
        # 优先使用 date 字段推断
        if 'date' in scenario:
            try:
                date_obj = pd.to_datetime(scenario['date'])
                if date_obj in holiday_cal:
                    is_holiday = 1.0
            except:
                pass
        # 其次尝试 timestamp
        elif 'timestamp' in scenario:
            try:
                date_obj = pd.to_datetime(scenario['timestamp'])
                if date_obj in holiday_cal:
                    is_holiday = 1.0
            except:
                pass
        # 尝试从 day_of_year + year 推断日期
        elif 'day_of_year' in scenario and 'year' in scenario:
            try:
                import datetime
                year = int(scenario['year'])
                doy = int(scenario['day_of_year'])
                date_obj = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1)
                if date_obj in holiday_cal:
                    is_holiday = 1.0
            except:
                pass
        # 最后尝试直接读取 is_holiday
        elif 'is_holiday' in scenario:
            is_holiday = float(scenario['is_holiday'])
            
        features.append(is_holiday)
        feature_names.append('is_holiday_enhanced')
        
        # 负荷特征 (兼容 load_trend vs trend_load 等)
        load_keys = [
            ('load_trend', 'trend_load'), 
            ('load_variability', 'variability_load'), 
            ('load_peak_ratio', 'peak_valley_ratio')
        ]
        for k1, k2 in load_keys:
            val = scenario.get(k1, scenario.get(k2))
            if val is not None:
                features.append(float(val))
                feature_names.append(k1)
        
        # 事件特征（如果存在）
        event_keys = ['num_change_points', 'num_anomalies', 'num_peaks',
                     'dominant_period', 'has_daily_pattern']
        for key in event_keys:
            if key in scenario:
                features.append(float(scenario[key]))
                feature_names.append(key)
        
        if len(features) == 0:
            return None
        
        # 首次提取时记录特征名
        if not self.feature_names:
            self.feature_names = feature_names
        
        # 确保特征数量一致 (如果后续场景特征缺失，补0或截断 - 简单起见这里假设一致或重新fit)
        # 在实际应用中应使用 DictVectorizer 或固定 schema
        
        return np.array(features)
    
    def get_encoding_info(self) -> Dict[str, any]:
        """获取编码器信息
        
        Returns:
            编码器状态信息
        """
        return {
            'is_fitted': self.is_fitted,
            'encoding_dim': self.encoding_dim,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names
        }
    
    def save(self, path: str) -> None:
        """保存编码器
        
        Args:
            path: 保存路径
        """
        # 使用 pickle 保存整个对象状态 (包括 KernelPCA)
        # 额外保存参数和形状，以便在 pickle 加载失败时尝试恢复
        state = {
            'encoding_dim': self.encoding_dim,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'encoder': self.encoder,
            'encoder_params': self.encoder.get_params() if hasattr(self.encoder, 'get_params') else {},
            'input_shape': self.scaler_mean.shape if self.scaler_mean is not None else None
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # 使用 .pkl 扩展名或直接写入二进制
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str) -> None:
        """加载编码器
        
        Args:
            path: 加载路径
        """
        if not Path(path).exists():
            return
        
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            self.encoding_dim = state['encoding_dim']
            self.is_fitted = state['is_fitted']
            self.feature_names = state['feature_names']
            self.scaler_mean = state['scaler_mean']
            self.scaler_std = state['scaler_std']
            self.encoder = state['encoder']
        except Exception as e:
            print(f"  [SceneEncoder] Pickle加载失败: {e}，尝试重置编码器")
            # 如果 pickle 加载失败（如版本不兼容），尝试重置为未训练状态
            # 这样至少不会导致程序崩溃，虽然需要重新训练
            self.is_fitted = False
            self.encoder = KernelPCA(
                n_components=self.encoding_dim, 
                kernel='rbf', 
                fit_inverse_transform=True,
                n_jobs=-1
            )


class DeepScenarioMatcher:
    """深度场景匹配器（结合编码器的场景相似度计算）"""
    
    def __init__(self, encoder: ScenarioEncoder):
        """
        Args:
            encoder: 场景编码器
        """
        self.encoder = encoder
        self.scenario_database: List[Dict[str, any]] = []
    
    def add_scenario(self, scenario: Dict[str, any]) -> None:
        """添加场景到数据库
        
        Args:
            scenario: 场景字典
        """
        # 编码并存储
        encoded = self.encoder.encode(scenario)
        scenario_with_encoding = scenario.copy()
        scenario_with_encoding['_encoding'] = encoded.tolist()
        self.scenario_database.append(scenario_with_encoding)
    
    def find_similar_scenarios(self, query_scenario: Dict[str, any], 
                              top_k: int = 5) -> List[Tuple[Dict, float]]:
        """查找最相似的历史场景
        
        Args:
            query_scenario: 查询场景
            top_k: 返回前 k 个最相似场景
            
        Returns:
            [(场景, 相似度), ...] 列表
        """
        if not self.scenario_database:
            return []
        
        query_encoded = self.encoder.encode(query_scenario)
        
        similarities = []
        for stored_scenario in self.scenario_database:
            if '_encoding' in stored_scenario:
                stored_encoded = np.array(stored_scenario['_encoding'])
                
                # 余弦相似度
                norm_q = np.linalg.norm(query_encoded)
                norm_s = np.linalg.norm(stored_encoded)
                
                if norm_q > 1e-9 and norm_s > 1e-9:
                    sim = np.dot(query_encoded, stored_encoded) / (norm_q * norm_s)
                    sim = (sim + 1.0) / 2.0  # 映射到 [0, 1]
                    similarities.append((stored_scenario, float(sim)))
        
        # 排序并返回 top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_database_size(self) -> int:
        """获取场景数据库大小
        
        Returns:
            场景数量
        """
        return len(self.scenario_database)
