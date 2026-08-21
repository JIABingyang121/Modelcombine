"""
场景推理优化模块

实现 0205.md 中的优化方案：
- 方案 G: 时间序列滚动验证 (RollingValidator)
- 方案 H: 小样本场景降级机制 (AdaptiveStrategySelector)
- 方案 B: 增强场景特征表达 (build_enhanced_ctx_features)
- 方案 C: 直接预测场景权重 (DirectWeightGatingNetwork)
- 方案 D: 自适应场景分桶 (AdaptiveBucketSelector)
- 方案 A: 场景相似度增强 (ScenarioSimilarityEnhancer)
"""

from typing import Dict, List, Tuple, Any, Callable
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================================
# 方案 G: 时间序列滚动验证
# ============================================================================

class RollingValidator:
    """
    时间序列滚动验证器
    
    用于评估场景推理策略在多个时间窗口上的稳定性，
    避免对单一验证集的过拟合。
    """
    
    def __init__(self, n_splits: int = 5, gap: int = 0, min_train_size: int = 500):
        """
        Args:
            n_splits: 滚动窗口数量
            gap: 训练和验证之间的间隔样本数（防止数据泄露）
            min_train_size: 最小训练集大小
        """
        self.n_splits = n_splits
        self.gap = gap
        self.min_train_size = min_train_size
    
    def split(self, n_samples: int):
        """
        生成滚动窗口切分的索引
        
        Yields:
            (train_idx, val_idx): 训练集和验证集的索引数组
        """
        # 计算每个折的大小，小样本时动态调整
        # 对于小样本数据集，降低 fold_size 要求
        min_fold_size = 20 if n_samples < 500 else 50 if n_samples < 1000 else 100
        fold_size = max(min_fold_size, n_samples // (self.n_splits + 2))
        
        # 小样本时同样降低 min_train_size
        effective_min_train = min(self.min_train_size, n_samples // 3) if n_samples < self.min_train_size * 2 else self.min_train_size
        
        for i in range(self.n_splits):
            train_end = effective_min_train + fold_size * i
            val_start = train_end + self.gap
            val_end = val_start + fold_size
            
            if val_end > n_samples:
                break
            
            if train_end < effective_min_train:
                continue
            
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            yield train_idx, val_idx
    
    def evaluate_strategy(
        self,
        strategy_fn: Callable,
        df: pd.DataFrame,
        y_col: str = "y",
        model_cols: List[str] = None
    ) -> Dict[str, float]:
        """
        评估策略在多个滚动窗口上的表现
        
        Args:
            strategy_fn: 策略函数，接受 (train_df, test_df, model_cols) 返回预测值
            df: 完整数据集（包含模型预测和真实值）
            y_col: 真实值列名
            model_cols: 模型预测列名列表
            
        Returns:
            包含 mean_rmse, std_rmse, stability 的字典
        """
        scores = []
        n_samples = len(df)
        
        for train_idx, val_idx in self.split(n_samples):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()
            
            try:
                preds = strategy_fn(train_df, val_df, model_cols)
                y_true = val_df[y_col].values
                rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
                scores.append(rmse)
            except Exception as e:
                print(f"    [RollingValidator] 评估失败: {e}")
                continue
        
        if not scores:
            return {
                'mean_rmse': float('inf'),
                'std_rmse': float('inf'),
                'stability': float('inf'),
                'n_folds': 0
            }
        
        mean_rmse = float(np.mean(scores))
        std_rmse = float(np.std(scores))
        stability = std_rmse / (mean_rmse + 1e-8)  # 变异系数
        
        return {
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'stability': stability,
            'n_folds': len(scores)
        }


def select_best_strategy_by_stability(
    strategies: Dict[str, Callable],
    df: pd.DataFrame,
    model_cols: List[str],
    y_col: str = "y",
    stability_weight: float = 0.5
) -> Tuple[str, Dict[str, Dict]]:
    """
    基于稳定性选择最优策略
    
    Args:
        strategies: 策略名称到策略函数的映射
        df: 数据集
        model_cols: 模型列名
        y_col: 真实值列名
        stability_weight: 稳定性权重（0-1）
        
    Returns:
        (最优策略名称, 所有策略的评估结果)
    """
    validator = RollingValidator(n_splits=5)
    results = {}
    
    for name, strategy_fn in strategies.items():
        metrics = validator.evaluate_strategy(strategy_fn, df, y_col, model_cols)
        # 综合得分：RMSE × (1 + stability_weight × 稳定性)
        metrics['score'] = metrics['mean_rmse'] * (1 + stability_weight * metrics['stability'])
        results[name] = metrics
    
    # 按综合得分排序
    best_name = min(results.keys(), key=lambda k: results[k]['score'])
    return best_name, results


# ============================================================================
# 方案 H: 小样本场景降级机制
# ============================================================================

class AdaptiveStrategySelector:
    """
    自适应策略选择器
    
    根据场景覆盖度（样本量/特征数）自动选择合适复杂度的策略，
    避免在小样本上使用过于复杂的场景推理策略。
    """
    
    # 策略复杂度分级
    STRATEGY_TIERS = {
        'tier1_simple': ['simple_avg', 'static_weight_safe', 'stacking_safe'],
        'tier2_moderate': ['dynamic_avg', 'dynamic_weight', 'dynamic_stacking'],
        'tier3_complex': ['gating_network', 'scenario_bucket', 'gating_network_v2', 
                          'adaptive_bucket', 'scenario_similarity']
    }
    
    # 场景覆盖度阈值（有效样本量）
    SAMPLE_THRESHOLDS = {
        'tier3_complex': 3000,   # 需要充分的场景覆盖
        'tier2_moderate': 1000,  # 需要基本的场景覆盖
        'tier1_simple': 0        # 任意样本量
    }
    
    def __init__(self, custom_thresholds: Dict[str, int] = None):
        """
        Args:
            custom_thresholds: 自定义阈值，覆盖默认值
        """
        self.thresholds = self.SAMPLE_THRESHOLDS.copy()
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)
    
    def select_tier(self, n_samples: int, n_ctx_features: int) -> str:
        """
        根据场景覆盖度选择策略层级
        
        Args:
            n_samples: 样本数量
            n_ctx_features: 场景特征数量
            
        Returns:
            策略层级名称
        """
        # 有效场景数 ≈ 样本数 / (场景特征维度的某种函数)
        # 场景特征越多，需要的样本越多
        effective_samples = n_samples / max(1, np.sqrt(n_ctx_features))
        
        if effective_samples >= self.thresholds['tier3_complex']:
            return 'tier3_complex'
        elif effective_samples >= self.thresholds['tier2_moderate']:
            return 'tier2_moderate'
        else:
            return 'tier1_simple'
    
    def get_allowed_strategies(self, n_samples: int, n_ctx_features: int) -> List[str]:
        """
        获取当前场景覆盖度允许使用的策略列表
        
        Returns:
            允许的策略名称列表（包含当前层级及以下的所有策略）
        """
        tier = self.select_tier(n_samples, n_ctx_features)
        
        allowed = []
        if tier == 'tier3_complex':
            allowed.extend(self.STRATEGY_TIERS['tier3_complex'])
        if tier in ['tier2_moderate', 'tier3_complex']:
            allowed.extend(self.STRATEGY_TIERS['tier2_moderate'])
        allowed.extend(self.STRATEGY_TIERS['tier1_simple'])
        
        return allowed
    
    def should_use_complex_strategy(self, n_samples: int, n_ctx_features: int) -> bool:
        """检查是否应该使用复杂的场景推理策略"""
        tier = self.select_tier(n_samples, n_ctx_features)
        return tier == 'tier3_complex'
    
    def get_fallback_strategy(self, n_samples: int, n_ctx_features: int) -> str:
        """获取推荐的回退策略"""
        tier = self.select_tier(n_samples, n_ctx_features)
        if tier == 'tier1_simple':
            return 'stacking_safe'  # 最稳健的简单策略
        elif tier == 'tier2_moderate':
            return 'dynamic_stacking'
        else:
            return 'gating_network_v2'


# ============================================================================
# 方案 B: 增强场景特征表达
# ============================================================================

def build_enhanced_ctx_features(
    df: pd.DataFrame,
    model_cols: List[str],
    existing_ctx_cols: List[str] = None,
    top_k: int = 3
) -> pd.DataFrame:
    """
    增强场景特征（加入模型分歧度信息）
    
    Args:
        df: 包含模型预测的数据框
        model_cols: 模型预测列名
        existing_ctx_cols: 已有的场景特征列名
        top_k: 计算 Top-K 差异时的 K 值
        
    Returns:
        增强后的场景特征数据框
    """
    features = {}
    
    # 1. 保留原有场景特征
    if existing_ctx_cols:
        for col in existing_ctx_cols:
            if col in df.columns:
                features[col] = df[col].values
    
    # 2. 获取模型预测矩阵
    available_models = [m for m in model_cols if m in df.columns]
    if not available_models:
        return pd.DataFrame(features, index=df.index)
    
    preds_matrix = df[available_models].values  # (n_samples, n_models)
    
    # 3. 模型预测统计特征（场景不确定性指标）
    features['ctx_pred_mean'] = np.mean(preds_matrix, axis=1)
    features['ctx_pred_std'] = np.std(preds_matrix, axis=1)
    features['ctx_pred_range'] = np.ptp(preds_matrix, axis=1)
    features['ctx_pred_cv'] = features['ctx_pred_std'] / (np.abs(features['ctx_pred_mean']) + 1e-8)
    
    # 4. Top-K 模型差异特征（控制规模）
    if len(available_models) >= top_k:
        sorted_preds = np.sort(preds_matrix, axis=1)
        features['ctx_top_k_range'] = sorted_preds[:, -1] - sorted_preds[:, -top_k]
        features['ctx_top_k_std'] = np.std(sorted_preds[:, -top_k:], axis=1)
        features['ctx_bottom_k_range'] = sorted_preds[:, top_k-1] - sorted_preds[:, 0]
        features['ctx_top_bottom_gap'] = sorted_preds[:, -1] - sorted_preds[:, 0]
    
    # 5. 各模型相对偏差（用于识别异常模型）
    pred_mean = features['ctx_pred_mean']
    for m in available_models:
        features[f'ctx_dev_{m}'] = df[m].values - pred_mean
    
    return pd.DataFrame(features, index=df.index)


def get_ctx_cols(df: pd.DataFrame) -> List[str]:
    """获取所有以 ctx_ 开头的场景特征列"""
    return [c for c in df.columns if c.startswith("ctx_")]


# ============================================================================
# 方案 C: 直接预测场景权重
# ============================================================================

class DirectWeightGatingNetwork:
    """
    场景门控网络（直接预测权重版）
    
    改进点：
    1. 直接学习 场景特征 → 模型权重 的映射，简化推理链
    2. 使用软标签（高温度 softmax）避免极端权重
    3. 交叉验证选择正则化强度，防止场景过拟合
    4. 支持增强场景特征
    """
    
    def __init__(
        self,
        model_cols: List[str],
        temperature: float = 2.0,
        cv_folds: int = 3,
        use_enhanced_features: bool = True,
        active_models: List[str] = None
    ):
        """
        Args:
            model_cols: 基础模型列名
            temperature: softmax 温度参数，越高越平滑
            cv_folds: 交叉验证折数
            use_enhanced_features: 是否使用增强特征
            active_models: 活跃模型列表（候选池约束）
        """
        self.model_cols = model_cols
        self.temperature = temperature
        self.cv_folds = cv_folds
        self.use_enhanced_features = use_enhanced_features
        self.active_models = active_models
        
        self.weight_predictor = None
        self.scaler = StandardScaler()
        self.ctx_cols = []
        self._effective_models = None
        self._fallback_weights = None
        self._best_method = "ridge"
        self._best_alpha = 1.0
        self._best_l1_ratio = None
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """计算 softmax，处理数值稳定性"""
        x = np.asarray(x)
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / (np.sum(exp_x) + 1e-8)
    
    def _get_effective_models(self, df: pd.DataFrame) -> List[str]:
        """确定有效模型列表"""
        if self.active_models is not None:
            effective = [m for m in self.active_models if m in self.model_cols and m in df.columns]
        else:
            effective = [m for m in self.model_cols if m in df.columns]
        # [Fix] 不再回退到 self.model_cols，返回空列表让调用方处理
        return effective
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """准备特征矩阵"""
        # 获取原有场景特征
        original_ctx = get_ctx_cols(df)
        
        if self.use_enhanced_features:
            # 增强特征
            enhanced_df = build_enhanced_ctx_features(
                df, self._effective_models, original_ctx
            )
            self.ctx_cols = list(enhanced_df.columns)
            features = enhanced_df.fillna(0).values.astype(float)
        else:
            self.ctx_cols = original_ctx
            if not self.ctx_cols:
                return np.zeros((len(df), 0))
            features = df[self.ctx_cols].fillna(0).values.astype(float)
        
        return features
    
    def _build_weight_regressor(self, method: str, alpha: float, l1_ratio: float = None):
        method = str(method).lower()
        if method == "ridge":
            return MultiOutputRegressor(Ridge(alpha=float(alpha), positive=True))
        if method == "lasso":
            return MultiOutputRegressor(Lasso(alpha=float(alpha), positive=True, max_iter=10000))
        if method == "elasticnet":
            l1 = 0.5 if l1_ratio is None else float(l1_ratio)
            return MultiOutputRegressor(
                ElasticNet(alpha=float(alpha), l1_ratio=l1, positive=True, max_iter=10000)
            )
        raise ValueError(f"Unsupported gating method: {method}")

    def _cv_select_regression_config(self, X: np.ndarray, y: np.ndarray) -> Tuple[str, float, float | None]:
        """时间序列交叉验证选择回归器类型与正则化强度"""
        candidate_cfgs: List[Tuple[str, float, float | None]] = []
        candidate_cfgs.extend([("ridge", a, None) for a in [0.1, 1.0, 10.0, 100.0, 1000.0]])
        candidate_cfgs.extend([("lasso", a, None) for a in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]])
        for a in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            for l1 in [0.2, 0.5, 0.8]:
                candidate_cfgs.append(("elasticnet", a, l1))

        best_method, best_alpha, best_l1_ratio = "ridge", 100.0, None
        best_score = float('inf')
        
        n_samples = len(X)
        if n_samples < self.cv_folds * 50:
            # 样本太少，使用较强正则化
            return "ridge", 100.0, None
        
        # 时间序列 CV（不打乱）
        fold_size = n_samples // (self.cv_folds + 1)
        
        for method, alpha, l1_ratio in candidate_cfgs:
            scores = []
            for i in range(self.cv_folds):
                train_end = fold_size * (i + 1)
                val_start = train_end
                val_end = min(val_start + fold_size, n_samples)
                
                if val_end <= val_start:
                    continue
                
                X_train, y_train = X[:train_end], y[:train_end]
                X_val, y_val = X[val_start:val_end], y[val_start:val_end]
                
                try:
                    model = self._build_weight_regressor(method, alpha, l1_ratio)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    score = float(np.mean(np.abs(pred - y_val)))
                    scores.append(score)
                except Exception:
                    continue
            
            if scores and np.mean(scores) < best_score:
                best_score = np.mean(scores)
                best_method = method
                best_alpha = alpha
                best_l1_ratio = l1_ratio
        
        return best_method, float(best_alpha), best_l1_ratio
    
    def fit(self, val_df: pd.DataFrame) -> bool:
        """
        在验证集上训练权重预测器
        
        Args:
            val_df: 验证集数据框，需包含模型预测列和 "y" 列
            
        Returns:
            是否训练成功
        """
        self._effective_models = self._get_effective_models(val_df)
        
        # [Fix] 无有效模型时显式报错
        if not self._effective_models:
            print("    [DirectWeightGating] 无有效模型列，跳过训练")
            return False
        
        n_models = len(self._effective_models)
        
        if "y" not in val_df.columns:
            print("    [DirectWeightGating] 缺少 y 列")
            return False
        
        y_true = val_df["y"].values
        n_samples = len(val_df)
        
        # 计算回退权重
        model_maes = {}
        for m in self._effective_models:
            if m in val_df.columns:
                model_maes[m] = float(mean_absolute_error(y_true, val_df[m].values))
        
        if model_maes:
            inv_maes = {m: 1.0 / (mae + 1e-6) for m, mae in model_maes.items()}
            inv_sum = sum(inv_maes.values())
            self._fallback_weights = {m: inv_maes[m] / inv_sum for m in inv_maes}
        
        # 准备特征
        X = self._prepare_features(val_df)
        
        if X.shape[1] == 0:
            print("    [DirectWeightGating] 无场景特征，使用回退权重")
            return False
        
        # 计算"场景最优权重"标签（软标签）
        target_weights = []
        for i in range(n_samples):
            preds = np.array([val_df[m].iloc[i] for m in self._effective_models])
            errors = np.abs(preds - y_true[i])
            # 高温度 softmax 生成软标签
            w = self._softmax(-errors / self.temperature)
            target_weights.append(w)
        
        target_weights = np.array(target_weights)  # (n_samples, n_models)
        
        # 标准化特征
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # 交叉验证选择回归器与正则化强度
        self._best_method, self._best_alpha, self._best_l1_ratio = self._cv_select_regression_config(
            X_scaled, target_weights
        )
        
        # 训练权重预测器
        self.weight_predictor = self._build_weight_regressor(
            self._best_method,
            self._best_alpha,
            self._best_l1_ratio,
        )
        
        try:
            self.weight_predictor.fit(X_scaled, target_weights)
            msg = (
                f"    [DirectWeightGating] 训练成功: {n_models} 个模型, "
                f"{X.shape[1]} 特征, method={self._best_method}, alpha={self._best_alpha}"
            )
            if self._best_l1_ratio is not None:
                msg += f", l1_ratio={self._best_l1_ratio}"
            print(msg)
            return True
        except Exception as e:
            print(f"    [DirectWeightGating] 训练失败: {e}")
            return False
    
    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        预测测试集
        
        Returns:
            (预测值数组, 平均使用模型数)
        """
        n_samples = len(test_df)
        train_models = self._effective_models or self.model_cols
        
        # [Fix] 检查有效模型是否存在于 test_df 中
        available_models = [m for m in train_models if m in test_df.columns]
        if not available_models:
            print("    [DirectWeightGating] 警告：test_df 中无有效模型列，返回零预测")
            return np.zeros(n_samples), 0.0
        
        # [Fix] 如果测试集缺少部分模型，回退到简单均值（避免维度不一致）
        if len(available_models) < len(train_models):
            missing = set(train_models) - set(available_models)
            print(f"    [DirectWeightGating] 警告：test_df 缺少模型 {missing}，回退到均值策略")
            return test_df[available_models].mean(axis=1).values, float(len(available_models))
        
        # 辅助函数：使用 fallback_weights 进行预测（带重归一化）
        def _fallback_predict():
            if self._fallback_weights:
                predictions = np.zeros(n_samples)
                total_weight = 0.0
                for m, w in self._fallback_weights.items():
                    if m in test_df.columns:
                        predictions += test_df[m].values * w
                        total_weight += w
                # [Fix] 重归一化
                if total_weight > 0:
                    predictions = predictions / total_weight
                return predictions, float(len([m for m in self._fallback_weights if m in test_df.columns]))
            return test_df[train_models].mean(axis=1).values, float(len(train_models))
        
        # 无预测器时使用回退
        if self.weight_predictor is None:
            return _fallback_predict()
        
        # 准备特征
        X = self._prepare_features(test_df)
        
        if X.shape[1] == 0:
            return _fallback_predict()
        
        X_scaled = self.scaler.transform(X)
        
        # 预测权重
        raw_weights = self.weight_predictor.predict(X_scaled)
        
        # 强制非负并归一化
        weights = np.clip(raw_weights, 0, None)
        weight_sums = weights.sum(axis=1, keepdims=True) + 1e-8
        weights = weights / weight_sums
        
        # 加权预测（此时 train_models 与 weights 维度一致）
        preds_matrix = np.column_stack([test_df[m].values for m in train_models])
        predictions = (preds_matrix * weights).sum(axis=1)
        
        # 计算平均有效模型数（权重 > 0.05 的模型数）
        avg_used = float(np.mean(np.sum(weights > 0.05, axis=1)))
        
        return predictions, avg_used


# ============================================================================
# 方案 D: 自适应场景分桶
# ============================================================================

class AdaptiveBucketSelector:
    """
    自适应场景分桶选择器
    
    改进点：
    1. 用决策树自动学习场景边界，而非硬编码
    2. 使用误差排名作为目标，减少噪声
    3. 全局回退机制
    4. 支持增强场景特征
    """
    
    def __init__(
        self,
        model_cols: List[str],
        max_depth: int = 4,
        min_bucket_size: int = 50,
        use_enhanced_features: bool = True,
        active_models: List[str] = None
    ):
        """
        Args:
            model_cols: 基础模型列名
            max_depth: 决策树最大深度（控制桶数量）
            min_bucket_size: 最小桶大小
            use_enhanced_features: 是否使用增强特征
            active_models: 活跃模型列表
        """
        self.model_cols = model_cols
        self.max_depth = max_depth
        self.min_bucket_size = min_bucket_size
        self.use_enhanced_features = use_enhanced_features
        self.active_models = active_models
        
        self.tree = None
        self.scaler = StandardScaler()
        self.bucket_weights = {}
        self.global_weights = {}
        self.ctx_cols = []
        self._effective_models = None
    
    def _get_effective_models(self, df: pd.DataFrame) -> List[str]:
        """确定有效模型列表"""
        if self.active_models is not None:
            effective = [m for m in self.active_models if m in self.model_cols and m in df.columns]
        else:
            effective = [m for m in self.model_cols if m in df.columns]
        # [Fix] 不再回退到 self.model_cols，返回空列表让调用方处理
        return effective
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """准备特征矩阵"""
        original_ctx = get_ctx_cols(df)
        
        if self.use_enhanced_features:
            enhanced_df = build_enhanced_ctx_features(
                df, self._effective_models, original_ctx
            )
            self.ctx_cols = list(enhanced_df.columns)
            features = enhanced_df.fillna(0).values.astype(float)
        else:
            self.ctx_cols = original_ctx
            if not self.ctx_cols:
                return np.zeros((len(df), 0))
            features = df[self.ctx_cols].fillna(0).values.astype(float)
        
        return features
    
    def fit(self, val_df: pd.DataFrame) -> bool:
        """
        在验证集上训练自适应分桶器
        
        Args:
            val_df: 验证集数据框
            
        Returns:
            是否训练成功
        """
        self._effective_models = self._get_effective_models(val_df)
        
        # [Fix] 无有效模型时显式报错
        if not self._effective_models:
            print("    [AdaptiveBucket] 无有效模型列，跳过训练")
            return False
        
        n_models = len(self._effective_models)
        
        if "y" not in val_df.columns:
            print("    [AdaptiveBucket] 缺少 y 列")
            return False
        
        y_true = val_df["y"].values
        n_samples = len(val_df)
        
        # 准备特征
        X = self._prepare_features(val_df)
        
        if X.shape[1] == 0:
            print("    [AdaptiveBucket] 无场景特征")
            return False
        
        # 计算各模型误差
        errors = np.column_stack([
            np.abs(val_df[m].values - y_true) for m in self._effective_models
        ])
        
        # 计算全局权重
        global_errors = errors.mean(axis=0)
        global_inv = 1.0 / (global_errors + 1e-8)
        global_inv_sum = global_inv.sum()
        self.global_weights = {
            m: global_inv[i] / global_inv_sum 
            for i, m in enumerate(self._effective_models)
        }
        
        # 使用最佳模型索引作为目标（分类问题）
        # argmin 返回每行误差最小的模型索引
        best_model_idx = np.argmin(errors, axis=1)
        target = best_model_idx  # 分类标签，整数类型
        
        # 标准化特征
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # 训练决策树分类器（分类问题更合适）
        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_bucket_size,
            random_state=42
        )
        
        try:
            self.tree.fit(X_scaled, target)
        except Exception as e:
            print(f"    [AdaptiveBucket] 决策树训练失败: {e}")
            return False
        
        # 提取桶 ID
        bucket_ids = self.tree.apply(X_scaled)
        unique_buckets = np.unique(bucket_ids)
        
        # 每个桶内学习权重
        for bid in unique_buckets:
            mask = bucket_ids == bid
            bucket_errors = errors[mask]
            
            if mask.sum() < 10:
                # 样本太少，使用全局权重
                self.bucket_weights[bid] = self.global_weights.copy()
                continue
            
            avg_errors = bucket_errors.mean(axis=0)
            inv_errors = 1.0 / (avg_errors + 1e-8)
            inv_sum = inv_errors.sum()
            
            self.bucket_weights[bid] = {
                m: inv_errors[i] / inv_sum
                for i, m in enumerate(self._effective_models)
            }
        
        print(f"    [AdaptiveBucket] 训练成功: {len(unique_buckets)} 个桶, "
              f"{n_models} 个模型, {X.shape[1]} 特征")
        return True
    
    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        预测测试集
        
        Returns:
            (预测值数组, 平均使用模型数)
        """
        n_samples = len(test_df)
        train_models = self._effective_models or self.model_cols
        
        # [Fix] 检查有效模型是否存在于 test_df 中
        available_models = [m for m in train_models if m in test_df.columns]
        if not available_models:
            print("    [AdaptiveBucket] 警告：test_df 中无有效模型列，返回零预测")
            return np.zeros(n_samples), 0.0
        
        # [Fix] 如果测试集缺少部分模型，回退到简单均值（避免维度不一致）
        if len(available_models) < len(train_models):
            missing = set(train_models) - set(available_models)
            print(f"    [AdaptiveBucket] 警告：test_df 缺少模型 {missing}，回退到均值策略")
            return test_df[available_models].mean(axis=1).values, float(len(available_models))
        
        # 辅助函数：使用全局权重回退
        def _fallback_predict():
            if self.global_weights:
                predictions = np.zeros(n_samples)
                total_weight = sum(w for m, w in self.global_weights.items() if m in test_df.columns)
                if total_weight > 0:
                    for m, w in self.global_weights.items():
                        if m in test_df.columns:
                            predictions += test_df[m].values * (w / total_weight)
                return predictions, float(len([m for m in self.global_weights if m in test_df.columns]))
            return test_df[train_models].mean(axis=1).values, float(len(train_models))
        
        if self.tree is None:
            return _fallback_predict()
        
        # 准备特征
        X = self._prepare_features(test_df)
        
        if X.shape[1] == 0:
            return _fallback_predict()
        
        X_scaled = self.scaler.transform(X)
        bucket_ids = self.tree.apply(X_scaled)
        
        # 逐样本预测（带权重重归一化）
        predictions = np.zeros(n_samples)
        for i in range(n_samples):
            bid = bucket_ids[i]
            weights = self.bucket_weights.get(bid, self.global_weights)
            
            # [Fix] 收集可用模型的权重并重归一化
            total_weight = 0.0
            for m, w in weights.items():
                if m in test_df.columns:
                    total_weight += w
            
            if total_weight > 0:
                for m, w in weights.items():
                    if m in test_df.columns:
                        predictions[i] += test_df[m].iloc[i] * (w / total_weight)
            else:
                # 无有效权重，使用均值
                predictions[i] = test_df[train_models].iloc[i].mean()
        
        avg_used = float(len(train_models))
        return predictions, avg_used


# ============================================================================
# 方案 A: 场景相似度增强
# ============================================================================

class ScenarioSimilarityEnhancer:
    """
    场景相似度增强器（优化版 - 无泄露 + 时间衰减 + 自适应回退）
    
    通过 kNN 找到历史相似场景，借鉴它们的权重，实现跨场景的知识迁移。
    
    优化点：
    1. 索引仅在训练集上构建（防止数据泄露）
    2. 支持多种样本权重模式（rank / softmax / error_softmax）
    3. 时间衰减：最近的训练样本权重更高（减少分布漂移影响）
    4. 自适应回退：邻居距离过大时回退到全局权重（提高鲁棒性）
    """
    
    def __init__(
        self,
        model_cols: List[str],
        n_neighbors: int = 10,
        temperature: float = 2.0,
        use_enhanced_features: bool = True,
        active_models: List[str] = None,
        time_aware_mode: str = 'none',
        time_gap: int = 0,
        # 优化参数
        use_rank_weights: bool = True,      # 兼容旧参数：True->rank, False->softmax
        weight_mode: str = None,            # rank / softmax / error_softmax
        distance_threshold: float = 2.0,    # 距离阈值（标准差倍数）
        fallback_blend: float = 0.3,        # 全局权重混合比例
        time_decay: float = 0.5,            # 时间衰减强度 (0=不衰减, 1=强衰减)
        recent_ratio: float = 0.3,          # 只使用最近 N% 的训练样本构建索引
    ):
        """
        Args:
            model_cols: 基础模型列名
            n_neighbors: 相似场景数量
            temperature: softmax 温度
            use_enhanced_features: 是否使用增强特征
            active_models: 活跃模型列表
            use_rank_weights: 兼容旧参数，映射到 rank/softmax
            weight_mode: 样本权重模式（rank / softmax / error_softmax）
            distance_threshold: 邻居距离阈值（超过则降低权重）
            fallback_blend: 与全局权重的混合比例（0=纯邻居, 1=纯全局）
            time_decay: 时间衰减强度，越大越偏向最近样本
            recent_ratio: 只使用最近 N% 的训练样本 (0=全部, 0.3=最近30%)
        """
        self.model_cols = model_cols
        self.n_neighbors = n_neighbors
        self.temperature = temperature
        self.use_enhanced_features = use_enhanced_features
        self.active_models = active_models
        self.time_aware_mode = time_aware_mode
        self.time_gap = time_gap
        
        # 优化参数
        self.use_rank_weights = use_rank_weights  # 保留兼容性
        if weight_mode is None:
            self.weight_mode = "rank" if use_rank_weights else "softmax"
        else:
            _wm = str(weight_mode).strip().lower()
            if _wm not in {"rank", "softmax", "error_softmax"}:
                _wm = "rank" if use_rank_weights else "softmax"
            self.weight_mode = _wm
        self.distance_threshold = distance_threshold
        self.fallback_blend = fallback_blend
        self.time_decay = time_decay
        self.recent_ratio = recent_ratio
        
        self.index = None
        self.scaler = StandardScaler()
        self.train_features = None
        self.train_weights = None
        self.train_indices = None
        self.ctx_cols = []
        self._effective_models = None
        self._fallback_weights = None
        self._n_train_samples = 0
        self._mean_distance = None
        self._time_weights = None  # 时间衰减权重
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / (np.sum(exp_x) + 1e-8)

    def _compute_sample_weights(self, errors: np.ndarray) -> np.ndarray:
        """根据配置模式计算单样本模型权重。"""
        errors = np.asarray(errors, dtype=float)
        safe_temperature = max(float(self.temperature), 1e-6)

        if self.weight_mode == "error_softmax":
            # 保留误差幅度信息，但压缩极端值，避免单模型过度主导。
            log_inv_errors = -np.log1p(np.maximum(errors, 0.0)) / safe_temperature
            return self._softmax(log_inv_errors)

        if self.weight_mode == "rank":
            ranks = np.argsort(np.argsort(errors)) + 1  # 1-based rank
            inv_ranks = 1.0 / ranks
            return inv_ranks / (inv_ranks.sum() + 1e-8)

        # "softmax"
        return self._softmax(-errors / safe_temperature)
    
    def _get_effective_models(self, df: pd.DataFrame) -> List[str]:
        """获取数据框中实际存在的模型列"""
        if self.active_models is not None:
            effective = [m for m in self.active_models if m in self.model_cols and m in df.columns]
        else:
            effective = [m for m in self.model_cols if m in df.columns]
        # [Fix] 不再回退到 self.model_cols，返回空列表让调用方处理
        return effective
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        original_ctx = get_ctx_cols(df)
        
        if self.use_enhanced_features:
            enhanced_df = build_enhanced_ctx_features(
                df, self._effective_models, original_ctx
            )
            self.ctx_cols = list(enhanced_df.columns)
            features = enhanced_df.fillna(0).values.astype(float)
        else:
            self.ctx_cols = original_ctx
            if not self.ctx_cols:
                return np.zeros((len(df), 0))
            features = df[self.ctx_cols].fillna(0).values.astype(float)
        
        return features
    
    def fit(self, train_df: pd.DataFrame) -> bool:
        """
        在训练集上构建场景索引（防止数据泄露）
        
        优化：
        1. 可只使用最近 N% 的训练样本（recent_ratio）
        2. 计算时间衰减权重（time_decay）
        
        注意：假设 train_df 已按时间升序排列（行索引越大越新）。
        如有 timestamp 列会自动排序，否则使用原始行顺序。
        
        Args:
            train_df: 训练数据（时间上在验证/测试之前，需按时间排序）
            
        Returns:
            是否成功构建索引
        """
        n_total = len(train_df)
        if n_total < 2:
            print(f"    [ScenarioSimilarity] 样本数过少 ({n_total} < 2)，跳过训练")
            return False
        
        # [Fix] 确保按时间排序（如有 timestamp 列），使用 to_datetime 避免字典序
        time_cols = [c for c in train_df.columns if c in ['timestamp', 'ts', 'datetime', 'date', 'time']]
        if time_cols:
            try:
                train_df = train_df.copy()
                train_df[time_cols[0]] = pd.to_datetime(train_df[time_cols[0]], errors='coerce')
                train_df = train_df.sort_values(time_cols[0]).reset_index(drop=True)
            except Exception:
                pass  # 排序失败则保持原序（假设已按时间排序）
        
        # 优化：只使用最近 N% 的训练样本
        # 注意：当 time_aware_mode != 'none' 时禁用 recent_ratio 截断，原因如下：
        #   1. time_aware_mode 主要用于泄露对照实验（fit=val, predict=val）
        #   2. leave_one_out/strict_history 依赖 query_idx 与 neighbor_indices 在同一索引空间
        #   3. 如果需要"截断 + 时间感知"的组合，应使用 fit(train_df) + predict(..., is_val_prediction=False)
        #      此时邻居自然来自更早时间，无需 time_aware_mode
        if self.time_aware_mode != 'none':
            # 泄露对照模式：需要完整索引空间以确保索引过滤正确
            n_samples = n_total
        elif self.recent_ratio > 0 and self.recent_ratio < 1.0:
            # [Fix] 边界处理：当 n_total < 100 时，取 min(100, n_total)
            n_recent = min(n_total, max(100, int(n_total * self.recent_ratio)))
            train_df = train_df.tail(n_recent).reset_index(drop=True)
            n_samples = len(train_df)
        else:
            n_samples = n_total
        
        self._effective_models = self._get_effective_models(train_df)
        
        # [Fix] 无有效模型时显式报错
        if not self._effective_models:
            print("    [ScenarioSimilarity] 无有效模型列，跳过训练")
            return False
        
        self._n_train_samples = n_samples
        
        if "y" not in train_df.columns:
            print("    [ScenarioSimilarity] 缺少 y 列")
            return False
        
        y_true = train_df["y"].values
        
        # 计算回退权重（基于训练集整体表现）
        model_maes = {}
        for m in self._effective_models:
            if m in train_df.columns:
                model_maes[m] = float(mean_absolute_error(y_true, train_df[m].values))
        
        if model_maes:
            inv_maes = {m: 1.0 / (mae + 1e-6) for m, mae in model_maes.items()}
            inv_sum = sum(inv_maes.values())
            self._fallback_weights = {m: inv_maes[m] / inv_sum for m in inv_maes}
        
        # 准备特征
        X = self._prepare_features(train_df)
        
        if X.shape[1] == 0:
            print("    [ScenarioSimilarity] 无场景特征")
            return False
        
        # 标准化
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # 构建 kNN 索引
        k_for_index = max(1, min(self.n_neighbors * 2, n_samples - 1))
        self.index = NearestNeighbors(n_neighbors=k_for_index, metric='euclidean')
        self.index.fit(X_scaled)
        self.train_features = X_scaled
        
        # 存储训练样本的原始索引
        self.train_indices = np.arange(n_samples)
        
        # 计算时间衰减权重：索引越大（越靠后/越新）权重越高
        if self.time_decay > 0:
            # 使用指数衰减：最新样本权重=1，最老样本权重=exp(-time_decay)
            time_positions = np.linspace(0, 1, n_samples)  # 0=最老, 1=最新
            self._time_weights = np.exp(self.time_decay * (time_positions - 1))
            # 归一化使平均值为 1
            self._time_weights = self._time_weights / self._time_weights.mean()
        else:
            self._time_weights = np.ones(n_samples)
        
        # 计算平均邻居距离（用于自适应阈值）
        # [Fix] 使用固定种子确保复现性
        if n_samples > self.n_neighbors + 1:
            rng = np.random.RandomState(42)
            sample_indices = rng.choice(n_samples, min(1000, n_samples), replace=False)
            sample_distances = []
            for idx in sample_indices:
                dists, _ = self.index.kneighbors(X_scaled[idx:idx+1])
                sample_distances.append(dists[0].mean())
            self._mean_distance = np.mean(sample_distances)
        else:
            self._mean_distance = 1.0
        
        # 存储每个训练场景的权重
        self.train_weights = []
        for i in range(n_samples):
            preds = np.array([train_df[m].iloc[i] for m in self._effective_models])
            errors = np.abs(preds - y_true[i])
            
            weights = self._compute_sample_weights(errors)
            self.train_weights.append(weights)
        
        self.train_weights = np.array(self.train_weights)
        
        print(
            f"    [ScenarioSimilarity] 索引构建成功: {n_samples} 个训练场景, "
            f"k={self.n_neighbors}, {X.shape[1]} 特征, weights={self.weight_mode}"
        )
        return True
    
    def predict(self, test_df: pd.DataFrame, is_val_prediction: bool = False) -> Tuple[np.ndarray, float]:
        """
        预测测试/验证集
        
        两种使用模式：
        1. 无泄露模式（推荐）：fit(train_df) + predict(val/test_df, is_val_prediction=False)
           - 邻居全部来自 train，自然无泄露
           - time_aware_mode 不生效
           
        2. 泄露对照模式：fit(val_df) + predict(val_df, is_val_prediction=True)
           - 用于 A1 对照实验
           - time_aware_mode='leave_one_out'/'strict_history' 会过滤邻居防止自匹配
        
        Args:
            test_df: 测试/验证数据框
            is_val_prediction: True 时启用 time_aware_mode 邻居过滤
            
        Returns:
            (预测值数组, 平均使用模型数) - 顺序与输入 test_df 一致
        """
        n_samples = len(test_df)
        train_models = self._effective_models or self.model_cols
        
        # [Fix] 检查有效模型是否存在于 test_df 中
        available_models = [m for m in train_models if m in test_df.columns]
        if not available_models:
            print("    [ScenarioSimilarity] 警告：test_df 中无有效模型列，返回零预测")
            return np.zeros(n_samples), 0.0
        
        # [Fix] 如果测试集缺少部分模型，回退到简单均值（避免维度不一致）
        # 因为 train_weights 和 fallback_w_vec 的维度是基于训练时的模型数
        if len(available_models) < len(train_models):
            missing = set(train_models) - set(available_models)
            print(f"    [ScenarioSimilarity] 警告：test_df 缺少模型 {missing}，回退到均值策略")
            return test_df[available_models].mean(axis=1).values, float(len(available_models))
        
        # 此时 available_models == train_models，维度一致
        effective_models = train_models
        
        # [Fix] 在所有逻辑之前做排序，记录排序映射以便最后恢复顺序
        sort_order = None  # 排序后的原始位置索引
        time_cols = [c for c in test_df.columns if c in ['timestamp', 'ts', 'datetime', 'date', 'time']]
        if time_cols:
            try:
                test_df = test_df.copy()
                # [Fix] 使用带 UUID 的临时列名避免覆盖用户数据
                _tmp_col = '__scenario_sim_orig_idx_7f3a9b__'
                test_df[_tmp_col] = np.arange(n_samples)
                test_df[time_cols[0]] = pd.to_datetime(test_df[time_cols[0]], errors='coerce')
                test_df = test_df.sort_values(time_cols[0]).reset_index(drop=True)
                sort_order = test_df[_tmp_col].values  # 排序后第 i 行对应原始第 sort_order[i] 行
                test_df = test_df.drop(columns=[_tmp_col])
            except Exception:
                pass  # 排序失败则保持原序
        
        def _restore_order(preds: np.ndarray) -> np.ndarray:
            """将排序后的预测恢复到原始顺序"""
            if sort_order is None:
                return preds
            restored = np.empty_like(preds)
            restored[sort_order] = preds  # 逆映射
            return restored
        
        # 回退逻辑：索引未构建时使用全局权重
        if self.index is None or self.train_weights is None:
            if self._fallback_weights:
                predictions = np.zeros(n_samples)
                for m, w in self._fallback_weights.items():
                    if m in test_df.columns:
                        predictions += test_df[m].values * w
                return _restore_order(predictions), float(len(self._fallback_weights))
            preds = test_df[effective_models].mean(axis=1).values
            return _restore_order(preds), float(len(effective_models))
        
        X = self._prepare_features(test_df)
        
        if X.shape[1] == 0:
            if self._fallback_weights:
                predictions = np.zeros(n_samples)
                for m, w in self._fallback_weights.items():
                    if m in test_df.columns:
                        predictions += test_df[m].values * w
                return _restore_order(predictions), float(len(self._fallback_weights))
            preds = test_df[effective_models].mean(axis=1).values
            return _restore_order(preds), float(len(effective_models))
        
        X_scaled = self.scaler.transform(X)
        
        # [Fix] 计算全局回退权重向量，处理 _fallback_weights 为 None 的情况
        if self._fallback_weights is not None:
            fallback_w_vec = np.array([self._fallback_weights.get(m, 1.0/len(effective_models)) 
                                       for m in effective_models])
        else:
            fallback_w_vec = np.ones(len(effective_models)) / len(effective_models)
        fallback_w_vec = fallback_w_vec / (fallback_w_vec.sum() + 1e-8)
        
        predictions = np.zeros(n_samples)
        for i in range(n_samples):
            # 找邻居
            distances, indices = self.index.kneighbors(X_scaled[i:i+1])
            distances = distances[0]
            indices = indices[0]
            
            # 限制邻居数量
            k_effective = min(self.n_neighbors, len(indices))
            distances = distances[:k_effective]
            indices = indices[:k_effective]
            
            # [Fix] time_aware_mode 邻居过滤（仅在 is_val_prediction=True 时生效）
            use_fallback_only = False
            if self.time_aware_mode != 'none' and is_val_prediction and len(indices) > 0:
                valid_mask = self._get_valid_neighbor_mask(
                    query_idx=i,
                    neighbor_indices=indices,
                    n_query_samples=n_samples,
                    is_val_prediction=True
                )
                if valid_mask.sum() > 0:
                    distances = distances[valid_mask]
                    indices = indices[valid_mask]
                else:
                    # [Fix] 无有效邻居时必须使用全局权重，不能回退到原始邻居（会重新引入泄露）
                    use_fallback_only = True
            
            preds = np.array([test_df[m].iloc[i] for m in effective_models])
            
            # [Fix] 无有效邻居或强制回退时，使用全局权重
            if len(indices) == 0 or use_fallback_only:
                predictions[i] = (preds * fallback_w_vec).sum()
                continue
            
            # 优化1: 距离阈值过滤（忽略过远的邻居）
            if self._mean_distance is not None and self.distance_threshold > 0:
                threshold = self._mean_distance * self.distance_threshold
                valid_mask = distances < threshold
                if valid_mask.sum() > 0:
                    distances = distances[valid_mask]
                    indices = indices[valid_mask]
            
            # 距离加权融合邻居权重
            neighbor_weights = self.train_weights[indices]
            distance_weights = 1.0 / (distances + 1e-8)
            
            # 时间衰减：最新样本权重更高
            if self._time_weights is not None:
                time_w = self._time_weights[indices]
                distance_weights = distance_weights * time_w
            
            distance_weights = distance_weights / (distance_weights.sum() + 1e-8)
            
            neighbor_final = (neighbor_weights * distance_weights[:, np.newaxis]).sum(axis=0)
            neighbor_final = np.clip(neighbor_final, 0, None)
            neighbor_final = neighbor_final / (neighbor_final.sum() + 1e-8)
            
            # 优化2: 自适应回退混合
            # 邻居距离越大，越倾向于使用全局权重
            mean_dist = distances.mean()
            if self._mean_distance is not None and self._mean_distance > 0:
                # 距离比率：>1 说明邻居较远，应增加全局权重比例
                dist_ratio = min(mean_dist / self._mean_distance, 2.0)
                blend = min(self.fallback_blend * dist_ratio, 0.8)
            else:
                blend = self.fallback_blend
            
            # 混合邻居权重和全局权重
            final_weights = (1 - blend) * neighbor_final + blend * fallback_w_vec
            final_weights = final_weights / (final_weights.sum() + 1e-8)
            
            # 加权预测
            preds = np.array([test_df[m].iloc[i] for m in effective_models])
            predictions[i] = (preds * final_weights).sum()
        
        avg_used = float(len(effective_models))
        return _restore_order(predictions), avg_used
    
    def _get_valid_neighbor_mask(
        self,
        query_idx: int,
        neighbor_indices: np.ndarray,
        n_query_samples: int,
        is_val_prediction: bool
    ) -> np.ndarray:
        """
        根据时间感知模式获取有效邻居掩码
        
        Args:
            query_idx: 当前查询样本在测试集中的索引
            neighbor_indices: kNN 返回的邻居在训练集中的索引
            n_query_samples: 查询集（测试集/验证集）的样本数
            is_val_prediction: 是否在验证集上预测
            
        Returns:
            布尔掩码数组
        """
        n_neighbors = len(neighbor_indices)
        
        if self.time_aware_mode == 'none':
            # 不做限制
            return np.ones(n_neighbors, dtype=bool)
        
        elif self.time_aware_mode == 'leave_one_out':
            # 排除自身（仅在验证集上预测时有意义）
            if is_val_prediction:
                return neighbor_indices != query_idx
            else:
                return np.ones(n_neighbors, dtype=bool)
        
        elif self.time_aware_mode == 'strict_history':
            # 仅使用时间严格在前的样本
            if is_val_prediction:
                # 验证集预测：只能使用索引小于当前样本的邻居
                return neighbor_indices < query_idx
            else:
                # 测试集预测：可以使用所有训练样本（它们时间上都在前）
                return np.ones(n_neighbors, dtype=bool)
        
        elif self.time_aware_mode == 'sliding_window':
            # 使用 timestamp < current - gap 的样本
            if is_val_prediction:
                return neighbor_indices < max(0, query_idx - self.time_gap)
            else:
                return np.ones(n_neighbors, dtype=bool)
        
        else:
            # 默认不做限制
            return np.ones(n_neighbors, dtype=bool)


# ============================================================================
# 工具函数
# ============================================================================

def create_optimized_strategy(
    strategy_name: str,
    model_cols: List[str],
    active_models: List[str] = None,
    **kwargs
) -> Any:
    """
    创建优化后的策略实例
    
    Args:
        strategy_name: 策略名称
        model_cols: 模型列名
        active_models: 活跃模型列表
        **kwargs: 额外参数
        
    Returns:
        策略实例
    """
    if strategy_name in ['gating_network_v2', 'direct_weight_gating']:
        return DirectWeightGatingNetwork(
            model_cols=model_cols,
            active_models=active_models,
            **kwargs
        )
    elif strategy_name in ['adaptive_bucket', 'scenario_bucket_v2']:
        return AdaptiveBucketSelector(
            model_cols=model_cols,
            active_models=active_models,
            **kwargs
        )
    elif strategy_name in ['scenario_similarity', 'similarity_enhancer']:
        return ScenarioSimilarityEnhancer(
            model_cols=model_cols,
            active_models=active_models,
            **kwargs
        )
    else:
        raise ValueError(f"未知策略: {strategy_name}")


DEFAULT_KG_COMPONENT_STRATEGIES = [
    "gating_network",
    "soft_gating",
    "scenario_bucket",
    "gating_network_v2",
    "adaptive_bucket",
    "scenario_similarity",
]

SMALL_SAMPLE_KG_COMPONENT_STRATEGIES = [
    "gating_network",
    "soft_gating",
    "scenario_bucket",
    "scenario_similarity",
]

def should_use_optimized_strategy(
    n_samples: int,
    n_ctx_features: int,
    dataset_name: str = None
) -> Tuple[bool, str, List[str]]:
    """
    判断是否应该使用优化策略，并返回允许启用的 kg_component 子策略列表。

    Returns:
        (是否使用, 推荐回退策略, 允许的 kg_component 策略列表)
    """
    selector = AdaptiveStrategySelector()

    # 特征不足时，不启用复杂策略
    if n_ctx_features < 2:
        return False, 'stacking_safe', []

    # 小样本兜底：100-500 样本仅启用低复杂度 KG component，抑制过拟合风险。
    if n_samples >= 100 and n_samples < 500:
        return True, 'stacking_safe', SMALL_SAMPLE_KG_COMPONENT_STRATEGIES.copy()

    if selector.should_use_complex_strategy(n_samples, n_ctx_features):
        return True, 'gating_network_v2', DEFAULT_KG_COMPONENT_STRATEGIES.copy()

    fallback = selector.get_fallback_strategy(n_samples, n_ctx_features)
    return False, fallback, []
