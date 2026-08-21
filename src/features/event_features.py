"""
事件特征工程
检测和提取电力需求数据中的事件特征（突变、异常、周期性事件等）
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import zscore


class EventFeatureExtractor:
    """事件特征提取器"""
    
    def __init__(self):
        self.event_history: List[Dict] = []
    
    def extract_events(self, df: pd.DataFrame, 
                      value_col: str = 'load') -> Dict[str, any]:
        """提取时间序列中的事件特征
        
        Args:
            df: 数据框（必须包含 timestamp 和 value_col）
            value_col: 数值列名
            
        Returns:
            事件特征字典
        """
        if value_col not in df.columns:
            return {}
        
        values = df[value_col].values
        timestamps = pd.to_datetime(df['timestamp'])
        
        features = {}
        
        # 1. 突变检测（Change Point Detection）
        change_points = self._detect_change_points(values)
        features['num_change_points'] = len(change_points)
        features['change_point_intensity'] = np.std(change_points) if len(change_points) > 0 else 0.0
        
        # 2. 异常检测（Anomaly Detection）
        anomalies = self._detect_anomalies(values)
        features['num_anomalies'] = len(anomalies)
        features['anomaly_rate'] = len(anomalies) / len(values) if len(values) > 0 else 0.0
        
        # 3. 峰值事件检测（Peak Detection）
        peaks, troughs = self._detect_peaks_troughs(values)
        features['num_peaks'] = len(peaks)
        features['num_troughs'] = len(troughs)
        features['peak_trough_ratio'] = len(peaks) / (len(troughs) + 1)
        
        # 4. 周期性事件检测（Periodic Events）
        dominant_period = self._detect_periodicity(values)
        features['dominant_period'] = dominant_period
        features['has_daily_pattern'] = 1 if 20 <= dominant_period <= 28 else 0  # 24±4小时
        features['has_weekly_pattern'] = 1 if 160 <= dominant_period <= 176 else 0  # 168±8小时
        
        # 5. 趋势事件（Trend Events）
        trend_changes = self._detect_trend_changes(values)
        features['num_trend_changes'] = len(trend_changes)
        
        # 6. 波动事件（Volatility Events）
        volatility_events = self._detect_volatility_events(values)
        features['num_volatility_spikes'] = len(volatility_events)
        
        # 7. 时段事件统计（Time-based Events）
        if 'hour' in df.columns:
            time_features = self._extract_time_events(df, value_col)
            features.update(time_features)
        
        return features
    
    def _detect_change_points(self, values: np.ndarray, 
                             threshold: float = 2.0) -> List[int]:
        """检测突变点（基于一阶差分）
        
        Args:
            values: 时间序列值
            threshold: Z-score 阈值
            
        Returns:
            突变点索引列表
        """
        if len(values) < 3:
            return []
        
        # 计算一阶差分
        diff = np.diff(values)
        
        # 标准化差分
        diff_zscore = zscore(diff)
        
        # 找到超过阈值的点
        change_points = np.where(np.abs(diff_zscore) > threshold)[0]
        
        return change_points.tolist()
    
    def _detect_anomalies(self, values: np.ndarray, 
                         threshold: float = 3.0) -> List[int]:
        """检测异常值（基于 Z-score）
        
        Args:
            values: 时间序列值
            threshold: Z-score 阈值
            
        Returns:
            异常点索引列表
        """
        if len(values) < 3:
            return []
        
        z_scores = zscore(values)
        anomalies = np.where(np.abs(z_scores) > threshold)[0]
        
        return anomalies.tolist()
    
    def _detect_peaks_troughs(self, values: np.ndarray, 
                             prominence: float = None) -> Tuple[List[int], List[int]]:
        """检测峰值和谷值
        
        Args:
            values: 时间序列值
            prominence: 峰值显著性阈值
            
        Returns:
            (峰值索引列表, 谷值索引列表)
        """
        if len(values) < 3:
            return [], []
        
        # 自适应显著性阈值
        if prominence is None:
            prominence = np.std(values) * 0.5
        
        # 检测峰值
        peaks, _ = signal.find_peaks(values, prominence=prominence)
        
        # 检测谷值（反转序列）
        troughs, _ = signal.find_peaks(-values, prominence=prominence)
        
        return peaks.tolist(), troughs.tolist()
    
    def _detect_periodicity(self, values: np.ndarray) -> float:
        """检测主导周期（基于 FFT）
        
        Args:
            values: 时间序列值
            
        Returns:
            主导周期长度（样本数）
        """
        if len(values) < 10:
            return 0.0
        
        # 去除趋势
        detrended = signal.detrend(values)
        
        # FFT
        fft_vals = np.fft.fft(detrended)
        power = np.abs(fft_vals) ** 2
        
        # 只考虑正频率
        freqs = np.fft.fftfreq(len(values))
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        
        if len(positive_power) == 0:
            return 0.0
        
        # 找到最大功率对应的频率
        max_power_idx = np.argmax(positive_power[1:]) + 1  # 跳过直流分量
        dominant_freq = positive_freqs[max_power_idx]
        
        # 转换为周期
        if dominant_freq > 0:
            dominant_period = 1.0 / dominant_freq
        else:
            dominant_period = 0.0
        
        return float(dominant_period)
    
    def _detect_trend_changes(self, values: np.ndarray, 
                             window: int = 24) -> List[int]:
        """检测趋势变化点（上升/下降转换）
        
        Args:
            values: 时间序列值
            window: 滑动窗口大小
            
        Returns:
            趋势变化点索引列表
        """
        if len(values) < window * 2:
            return []
        
        # 计算滑动平均斜率
        slopes = []
        for i in range(window, len(values)):
            window_vals = values[i-window:i]
            x = np.arange(window)
            slope = np.polyfit(x, window_vals, 1)[0]
            slopes.append(slope)
        
        slopes = np.array(slopes)
        
        # 检测斜率符号变化
        slope_signs = np.sign(slopes)
        sign_changes = np.where(np.diff(slope_signs) != 0)[0]
        
        return (sign_changes + window).tolist()
    
    def _detect_volatility_events(self, values: np.ndarray, 
                                  window: int = 24, 
                                  threshold: float = 2.0) -> List[int]:
        """检测波动率突增事件
        
        Args:
            values: 时间序列值
            window: 滑动窗口大小
            threshold: 波动率阈值（相对于平均波动率）
            
        Returns:
            波动突增点索引列表
        """
        if len(values) < window * 2:
            return []
        
        # 计算滑动标准差
        rolling_std = pd.Series(values).rolling(window=window).std().values
        rolling_std = rolling_std[window:]  # 去除前面的 NaN
        
        # 计算平均波动率
        mean_volatility = np.nanmean(rolling_std)
        
        # 检测超过阈值的点
        volatility_events = np.where(rolling_std > mean_volatility * threshold)[0]
        
        return (volatility_events + window).tolist()
    
    def _extract_time_events(self, df: pd.DataFrame, 
                            value_col: str) -> Dict[str, float]:
        """提取基于时间的事件特征
        
        Args:
            df: 数据框
            value_col: 数值列名
            
        Returns:
            时间事件特征字典
        """
        features = {}
        
        if 'hour' not in df.columns:
            return features
        
        # 计算不同时段的负荷变化
        hourly_mean = df.groupby('hour')[value_col].mean()
        
        # 峰谷时段事件
        peak_hours = hourly_mean.nlargest(3).index.tolist()
        valley_hours = hourly_mean.nsmallest(3).index.tolist()
        
        features['peak_hour_concentration'] = len(set(peak_hours) & {7, 8, 18, 19, 20}) / 3.0
        features['valley_hour_concentration'] = len(set(valley_hours) & {0, 1, 2, 3, 4, 5}) / 3.0
        
        # 工作日/周末事件差异
        if 'is_weekend' in df.columns:
            weekend_mean = df[df['is_weekend'] == 1][value_col].mean()
            weekday_mean = df[df['is_weekend'] == 0][value_col].mean()
            features['weekend_event_intensity'] = abs(weekend_mean - weekday_mean) / (weekday_mean + 1e-9)
        
        return features
    
    def extract_event_sequences(self, df: pd.DataFrame, 
                               value_col: str = 'load') -> List[Dict]:
        """提取事件序列（带时间戳的事件列表）
        
        Args:
            df: 数据框
            value_col: 数值列名
            
        Returns:
            事件序列列表
        """
        values = df[value_col].values
        timestamps = pd.to_datetime(df['timestamp'])
        
        events = []
        
        # 检测各类事件
        change_points = self._detect_change_points(values)
        anomalies = self._detect_anomalies(values)
        peaks, troughs = self._detect_peaks_troughs(values)
        
        # 构建事件列表
        for idx in change_points:
            events.append({
                'type': 'change_point',
                'timestamp': timestamps.iloc[idx],
                'value': values[idx],
                'magnitude': abs(values[idx] - values[idx-1]) if idx > 0 else 0
            })
        
        for idx in anomalies:
            events.append({
                'type': 'anomaly',
                'timestamp': timestamps.iloc[idx],
                'value': values[idx],
                'z_score': (values[idx] - np.mean(values)) / (np.std(values) + 1e-9)
            })
        
        for idx in peaks:
            events.append({
                'type': 'peak',
                'timestamp': timestamps.iloc[idx],
                'value': values[idx]
            })
        
        for idx in troughs:
            events.append({
                'type': 'trough',
                'timestamp': timestamps.iloc[idx],
                'value': values[idx]
            })
        
        # 按时间排序
        events.sort(key=lambda x: x['timestamp'])
        
        return events
