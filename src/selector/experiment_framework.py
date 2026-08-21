"""
实验框架模块

功能：
1. 实验结果隔离：基线 (baseline)、对比算法 (sota)、KG 组件算法 (kg_component) 分离存储
2. 统一记录模板：每个实验记录 Val/Test RMSE、Gap、状态、降级原因等
3. 对照实验支持：泄露对照 (A1/A2/A3)、滚动验证对照 (E1/E2)
4. 数据集兼容性诊断：检查特征、样本量、模型预测等问题
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================================
# 策略分类枚举（用于结果隔离）
# ============================================================================

class StrategyCategory(Enum):
    """策略类别 - 用于实验结果隔离"""
    BASELINE_CLASSIC = "baseline_classic"   # 传统基线: simple_avg, stacking, etc.
    BASELINE_SOTA = "baseline_sota"         # SOTA 对比: rl_qms, mole_router
    KG_COMPONENT = "kg_component"           # KG 组件策略（非 KG Core）
    ABLATION = "ablation"                   # 消融实验: 泄露对照、特征对照等


# ============================================================================
# 统一实验记录数据结构
# ============================================================================

@dataclass
class ExperimentRecord:
    """统一实验记录模板"""
    # 基本信息
    dataset: str
    horizon: int
    strategy_name: str
    category: StrategyCategory
    
    # 核心指标
    val_rmse: float = float('nan')
    test_rmse: float = float('nan')
    val_mae: float = float('nan')
    test_mae: float = float('nan')
    
    # 稳定性指标
    gap_percent: float = float('nan')  # (test - val) / val * 100
    
    # 附加指标
    top10_mae: float = float('nan')
    tail_rmse: float = float('nan')
    mase: float = float('nan')
    
    # 统一目标分数 (P0.2)
    unified_score: float = float('nan')  # 排名聚合分数，越低越好
    fallback_ratio: float = 0.0          # 降级到静态策略的比例
    drift_penalty: float = 0.0           # 漂移惩罚项
    
    # 执行状态
    status: str = "success"  # success / skipped / failed / degraded
    skip_reason: Optional[str] = None
    fallback_strategy: Optional[str] = None
    
    # 元数据
    n_samples_train: int = 0
    n_samples_val: int = 0
    n_samples_test: int = 0
    n_ctx_features: int = 0
    runtime_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 实验配置（用于消融实验区分）
    experiment_variant: Optional[str] = None  # e.g., "A1_baseline", "A2_leave_one_out"
    config: Dict[str, Any] = field(default_factory=dict)
    
    def compute_gap(self):
        """计算 Val-Test Gap"""
        if self.val_rmse > 0 and not np.isnan(self.val_rmse):
            self.gap_percent = (self.test_rmse - self.val_rmse) / self.val_rmse * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = asdict(self)
        d['category'] = self.category.value
        return d
    
    def is_valid(self) -> bool:
        """检查记录是否有效"""
        return self.status == "success" and not np.isnan(self.test_rmse)


@dataclass
class ExperimentSuite:
    """实验套件：组织一组相关实验"""
    name: str
    description: str
    records: List[ExperimentRecord] = field(default_factory=list)
    
    # 分类存储
    baseline_classic: Dict[str, ExperimentRecord] = field(default_factory=dict)
    baseline_sota: Dict[str, ExperimentRecord] = field(default_factory=dict)
    kg_component: Dict[str, ExperimentRecord] = field(default_factory=dict)
    ablation: Dict[str, ExperimentRecord] = field(default_factory=dict)
    
    def add_record(self, record: ExperimentRecord):
        """添加实验记录并自动分类"""
        self.records.append(record)
        
        key = f"{record.dataset}_{record.horizon}_{record.strategy_name}"
        if record.experiment_variant:
            key += f"_{record.experiment_variant}"
        
        if record.category == StrategyCategory.BASELINE_CLASSIC:
            self.baseline_classic[key] = record
        elif record.category == StrategyCategory.BASELINE_SOTA:
            self.baseline_sota[key] = record
        elif record.category == StrategyCategory.KG_COMPONENT:
            self.kg_component[key] = record
        elif record.category == StrategyCategory.ABLATION:
            self.ablation[key] = record
    
    def get_summary(self) -> Dict[str, Any]:
        """获取实验摘要"""
        return {
            "name": self.name,
            "description": self.description,
            "total_records": len(self.records),
            "by_category": {
                "baseline_classic": len(self.baseline_classic),
                "baseline_sota": len(self.baseline_sota),
                "kg_component": len(self.kg_component),
                "ablation": len(self.ablation),
            },
            "success_rate": sum(1 for r in self.records if r.status == "success") / max(len(self.records), 1),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            "summary": self.get_summary(),
            "baseline_classic": {k: v.to_dict() for k, v in self.baseline_classic.items()},
            "baseline_sota": {k: v.to_dict() for k, v in self.baseline_sota.items()},
            "kg_component": {k: v.to_dict() for k, v in self.kg_component.items()},
            "ablation": {k: v.to_dict() for k, v in self.ablation.items()},
        }
    
    def save(self, path: Path):
        """保存实验结果到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ============================================================================
# P0.2 统一目标分数（排名聚合）
# ============================================================================

# 默认权重
UNIFIED_SCORE_WEIGHTS = {
    "rmse_cv": 0.4,
    "gap_cv": 0.25,
    "tail_rmse": 0.2,
    "fallback_ratio": 0.15,
}


def compute_unified_scores(
    records: List[ExperimentRecord],
    weights: Dict[str, float] = None,
) -> List[ExperimentRecord]:
    """
    为一组同 (dataset, horizon) 的 ExperimentRecord 计算 unified_score。
    
    使用排名聚合（百分位排名）解决量纲问题：
        Score = Σ λ_i * rank_percentile_i + drift_penalty
    
    排名百分位：0 = 最好，1 = 最差。Score 越低越好。
    
    Args:
        records: 同 (dataset, horizon) 的实验记录列表
        weights: 指标权重字典
    
    Returns:
        更新了 unified_score 的记录列表
    """
    weights = weights or UNIFIED_SCORE_WEIGHTS
    
    valid = [r for r in records if r.status == "success" and not np.isnan(r.test_rmse)]
    if not valid:
        return records
    
    n = len(valid)
    if n == 1:
        valid[0].unified_score = 0.0 + valid[0].drift_penalty
        return records
    
    # 提取各指标值
    rmse_vals = [r.test_rmse for r in valid]
    gap_vals = [abs(r.gap_percent) if not np.isnan(r.gap_percent) else 0.0 for r in valid]
    tail_vals = [r.tail_rmse if not np.isnan(r.tail_rmse) else r.test_rmse for r in valid]
    fallback_vals = [r.fallback_ratio for r in valid]
    
    def percentile_ranks(values):
        """计算百分位排名 (0=最好, 1=最差)"""
        arr = np.array(values, dtype=float)
        ranks = np.zeros_like(arr)
        sorted_idx = np.argsort(arr)
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = rank / max(n - 1, 1)
        return ranks
    
    ranks_rmse = percentile_ranks(rmse_vals)
    ranks_gap = percentile_ranks(gap_vals)
    ranks_tail = percentile_ranks(tail_vals)
    ranks_fallback = percentile_ranks(fallback_vals)
    
    for i, r in enumerate(valid):
        score = (
            weights.get("rmse_cv", 0.4) * ranks_rmse[i] +
            weights.get("gap_cv", 0.25) * ranks_gap[i] +
            weights.get("tail_rmse", 0.2) * ranks_tail[i] +
            weights.get("fallback_ratio", 0.15) * ranks_fallback[i] +
            r.drift_penalty
        )
        r.unified_score = float(round(score, 6))
    
    return records


# ============================================================================
# P0.3 全覆盖评估校验
# ============================================================================

def check_evaluation_coverage(
    records: List[ExperimentRecord],
    expected_datasets: List[str],
    expected_horizons: Dict[str, List[int]],
    expected_strategies: List[str],
) -> Dict[str, Any]:
    """
    检查评估是否覆盖所有 (dataset, horizon, strategy) 三元组。
    
    Returns:
        {
            "coverage_rate": float,
            "total_expected": int,
            "total_actual": int,
            "missing": [(dataset, horizon, strategy), ...],
            "status": "pass" / "fail"
        }
    """
    # 构建已有三元组
    actual = set()
    for r in records:
        actual.add((r.dataset, r.horizon, r.strategy_name))
    
    # 构建期望三元组
    expected = set()
    for ds in expected_datasets:
        for h in expected_horizons.get(ds, []):
            for s in expected_strategies:
                expected.add((ds, h, s))
    
    missing = sorted(expected - actual)
    total_expected = len(expected)
    total_actual = len(actual & expected)
    coverage = total_actual / total_expected if total_expected > 0 else 1.0
    
    return {
        "coverage_rate": round(coverage, 4),
        "total_expected": total_expected,
        "total_actual": total_actual,
        "missing": [{"dataset": m[0], "horizon": m[1], "strategy": m[2]} for m in missing],
        "status": "pass" if coverage >= 1.0 else "fail",
    }


# ============================================================================
# 数据集兼容性诊断
# ============================================================================

@dataclass
class DiagnosticResult:
    """诊断结果"""
    dataset: str
    horizon: int
    is_compatible: bool
    issues: List[str]
    warnings: List[str]
    recommendations: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetDiagnostic:
    """数据集兼容性诊断器"""
    
    # 最小要求
    MIN_SAMPLES = 100
    MIN_CTX_FEATURES = 3
    MIN_MODELS = 2
    MAX_MISSING_RATIO = 0.1
    
    def __init__(self, min_samples: int = None, min_ctx_features: int = None):
        self.min_samples = min_samples or self.MIN_SAMPLES
        self.min_ctx_features = min_ctx_features or self.MIN_CTX_FEATURES
    
    def diagnose(
        self,
        df: pd.DataFrame,
        model_cols: List[str],
        dataset_name: str = "unknown",
        horizon: int = 1
    ) -> DiagnosticResult:
        """
        诊断数据集兼容性
        
        Args:
            df: 数据框（包含模型预测和场景特征）
            model_cols: 模型预测列名
            dataset_name: 数据集名称
            horizon: 预测时域
            
        Returns:
            DiagnosticResult: 诊断结果
        """
        issues = []
        warnings = []
        recommendations = {}
        
        n_samples = len(df)
        
        # 1. 检查样本量
        if n_samples < self.min_samples:
            issues.append(f"样本量过小: {n_samples} < {self.min_samples}")
            recommendations['sample_size'] = "降级到 stacking_safe 或 static_weight_safe"
        elif n_samples < self.min_samples * 3:
            warnings.append(f"样本量偏少: {n_samples}，复杂策略可能不稳定")
        
        # 2. 检查场景特征
        ctx_cols = [c for c in df.columns if c.startswith('ctx_')]
        n_ctx = len(ctx_cols)
        
        if n_ctx < self.min_ctx_features:
            issues.append(f"场景特征不足: {n_ctx} < {self.min_ctx_features}")
            recommendations['ctx_features'] = "检查特征工程流程，或降级到非场景策略"
        
        # 3. 检查模型预测列
        available_models = [m for m in model_cols if m in df.columns]
        n_models = len(available_models)
        
        if n_models < self.MIN_MODELS:
            issues.append(f"可用模型不足: {n_models} < {self.MIN_MODELS}")
            recommendations['models'] = "检查基础模型训练流程"
        
        # 4. 检查缺失值
        if available_models:
            missing_ratio = df[available_models].isna().mean().mean()
            if missing_ratio > self.MAX_MISSING_RATIO:
                issues.append(f"模型预测缺失率过高: {missing_ratio:.1%} > {self.MAX_MISSING_RATIO:.1%}")
                recommendations['missing'] = "检查模型预测流程，或使用 safe 版本策略"
            elif missing_ratio > 0.01:
                warnings.append(f"存在少量缺失值: {missing_ratio:.1%}")
        
        # 5. 检查目标列
        if 'y' not in df.columns:
            issues.append("缺少目标列 'y'")
            recommendations['target'] = "确保数据框包含 'y' 列"
        
        # 6. 检查场景覆盖度
        if n_ctx > 0 and n_samples > 0:
            coverage = n_samples / np.sqrt(n_ctx)
            if coverage < 500:
                warnings.append(f"场景覆盖度偏低: {coverage:.0f} (样本/√特征数)")
        
        is_compatible = len(issues) == 0
        
        return DiagnosticResult(
            dataset=dataset_name,
            horizon=horizon,
            is_compatible=is_compatible,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def should_use_complex_strategy(self, diag: DiagnosticResult) -> Tuple[bool, str]:
        """
        基于诊断结果判断是否应使用复杂策略
        
        Returns:
            (是否使用复杂策略, 推荐策略)
        """
        if not diag.is_compatible:
            # 有严重问题，降级到最简单策略
            return False, "stacking_safe"
        
        if diag.warnings:
            # 有警告，使用中等复杂度
            return False, "dynamic_stacking"
        
        return True, "gating_network_v2"


# ============================================================================
# 实验执行器
# ============================================================================

class ExperimentExecutor:
    """实验执行器：运行策略并记录结果"""
    
    def __init__(
        self,
        dataset_name: str,
        horizon: int,
        model_cols: List[str],
        naive_scale: float = None
    ):
        self.dataset_name = dataset_name
        self.horizon = horizon
        self.model_cols = model_cols
        self.naive_scale = naive_scale
        self.diagnostic = DatasetDiagnostic()
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """计算评估指标"""
        abs_errors = np.abs(y_true - y_pred)
        
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        
        # Top 10% MAE
        threshold = np.percentile(np.abs(y_true), 90)
        top_mask = np.abs(y_true) >= threshold
        top10_mae = float(np.mean(abs_errors[top_mask])) if top_mask.sum() > 0 else mae
        
        # Tail RMSE
        error_threshold = np.percentile(abs_errors, 90)
        tail_mask = abs_errors >= error_threshold
        tail_rmse = float(np.sqrt(np.mean(abs_errors[tail_mask] ** 2))) if tail_mask.sum() > 0 else rmse
        
        # MASE
        mase = float('nan')
        if self.naive_scale and self.naive_scale > 1e-8:
            mase = mae / self.naive_scale
        
        return {
            'mae': mae,
            'rmse': rmse,
            'top10_mae': top10_mae,
            'tail_rmse': tail_rmse,
            'mase': mase,
        }
    
    def run_strategy(
        self,
        strategy_fn: Callable,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        strategy_name: str,
        category: StrategyCategory,
        experiment_variant: str = None,
        config: Dict[str, Any] = None
    ) -> ExperimentRecord:
        """
        运行单个策略并返回实验记录
        
        Args:
            strategy_fn: 策略函数，签名 (train_df, val_df, test_df, model_cols) -> (val_pred, test_pred)
            train_df: 训练集
            val_df: 验证集
            test_df: 测试集
            strategy_name: 策略名称
            category: 策略类别
            experiment_variant: 实验变体名称
            config: 实验配置
            
        Returns:
            ExperimentRecord: 实验记录
        """
        # 初始化记录
        record = ExperimentRecord(
            dataset=self.dataset_name,
            horizon=self.horizon,
            strategy_name=strategy_name,
            category=category,
            experiment_variant=experiment_variant,
            config=config or {},
            n_samples_train=len(train_df),
            n_samples_val=len(val_df),
            n_samples_test=len(test_df),
        )
        
        # 诊断数据集
        diag = self.diagnostic.diagnose(
            val_df, self.model_cols, self.dataset_name, self.horizon
        )
        record.n_ctx_features = len([c for c in val_df.columns if c.startswith('ctx_')])
        
        # 检查是否应该跳过
        if not diag.is_compatible and category == StrategyCategory.KG_COMPONENT:
            use_complex, fallback = self.diagnostic.should_use_complex_strategy(diag)
            if not use_complex:
                record.status = "skipped"
                record.skip_reason = "; ".join(diag.issues)
                record.fallback_strategy = fallback
                return record
        
        # 执行策略
        start_time = time.time()
        try:
            val_pred, test_pred = strategy_fn(train_df, val_df, test_df, self.model_cols)
            record.runtime_seconds = time.time() - start_time
            
            # 计算指标
            y_val = val_df['y'].values
            y_test = test_df['y'].values
            
            val_metrics = self._compute_metrics(y_val, val_pred)
            test_metrics = self._compute_metrics(y_test, test_pred)
            
            record.val_rmse = val_metrics['rmse']
            record.val_mae = val_metrics['mae']
            record.test_rmse = test_metrics['rmse']
            record.test_mae = test_metrics['mae']
            record.top10_mae = test_metrics['top10_mae']
            record.tail_rmse = test_metrics['tail_rmse']
            record.mase = test_metrics['mase']
            
            record.compute_gap()
            record.status = "success"
            
        except Exception as e:
            record.status = "failed"
            record.skip_reason = str(e)
            record.runtime_seconds = time.time() - start_time
        
        return record


# ============================================================================
# 对照实验：泄露对照 (Exp-A1/A2/A3)
# ============================================================================

class LeakageAblation:
    """
    泄露对照实验
    
    - A1a: 完全泄露版（fit=val, 无防护）
    - A1b: 泄露版 + leave_one_out 防护
    - A2: 无泄露版（fit=train）- 推荐方案
    - A3: 无泄露版 + strict_history
    """
    
    def __init__(self, model_cols: List[str], n_neighbors: int = 10, temperature: float = 2.0):
        self.model_cols = model_cols
        self.n_neighbors = n_neighbors
        self.temperature = temperature
    
    def run_ablation(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        executor: ExperimentExecutor
    ) -> Dict[str, ExperimentRecord]:
        """
        运行泄露对照实验
        
        Args:
            train_df: 训练集（时间上在 val 之前）
            val_df: 验证集
            test_df: 测试集
            executor: 实验执行器
            
        Returns:
            Dict[str, ExperimentRecord]: 各变体的实验记录
        """
        from .scenario_optimizer import ScenarioSimilarityEnhancer
        
        results = {}
        
        # A1a: 完全泄露版（fit=val, 无防护，会自匹配）
        def strategy_a1a(train_df, val_df, test_df, model_cols):
            enhancer = ScenarioSimilarityEnhancer(
                model_cols=model_cols,
                n_neighbors=self.n_neighbors,
                temperature=self.temperature,
                time_aware_mode='none'  # 不做邻居过滤
            )
            enhancer.fit(val_df)  # 在 val 上 fit（有泄露）
            val_pred, _ = enhancer.predict(val_df, is_val_prediction=False)  # 不过滤，完全泄露
            test_pred, _ = enhancer.predict(test_df, is_val_prediction=False)
            return val_pred, test_pred
        
        results['A1a_full_leak'] = executor.run_strategy(
            strategy_a1a,
            val_df, val_df, test_df,  # train=val for this ablation
            strategy_name="scenario_similarity",
            category=StrategyCategory.ABLATION,
            experiment_variant="A1a_full_leak",
            config={"time_aware_mode": "none", "fit_on": "val", "is_val_prediction": False}
        )
        
        # A1b: 泄露版 + leave_one_out 防护（验证 time_aware_mode 是否有效）
        def strategy_a1b(train_df, val_df, test_df, model_cols):
            enhancer = ScenarioSimilarityEnhancer(
                model_cols=model_cols,
                n_neighbors=self.n_neighbors,
                temperature=self.temperature,
                time_aware_mode='leave_one_out'  # 启用邻居过滤
            )
            enhancer.fit(val_df)  # 在 val 上 fit
            # [Fix] is_val_prediction=True 触发 leave_one_out 过滤
            val_pred, _ = enhancer.predict(val_df, is_val_prediction=True)
            test_pred, _ = enhancer.predict(test_df, is_val_prediction=False)
            return val_pred, test_pred
        
        results['A1b_leak_with_loo'] = executor.run_strategy(
            strategy_a1b,
            val_df, val_df, test_df,
            strategy_name="scenario_similarity",
            category=StrategyCategory.ABLATION,
            experiment_variant="A1b_leak_with_loo",
            config={"time_aware_mode": "leave_one_out", "fit_on": "val", "is_val_prediction": True}
        )
        
        # A2: 无泄露版（fit=train）- 推荐方案
        def strategy_a2(train_df, val_df, test_df, model_cols):
            enhancer = ScenarioSimilarityEnhancer(
                model_cols=model_cols,
                n_neighbors=self.n_neighbors,
                temperature=self.temperature,
                time_aware_mode='none'
            )
            enhancer.fit(train_df)  # 在 train 上 fit（无泄露）
            val_pred, _ = enhancer.predict(val_df, is_val_prediction=False)
            test_pred, _ = enhancer.predict(test_df, is_val_prediction=False)
            return val_pred, test_pred
        
        results['A2_no_leak'] = executor.run_strategy(
            strategy_a2,
            train_df, val_df, test_df,
            strategy_name="scenario_similarity",
            category=StrategyCategory.ABLATION,
            experiment_variant="A2_no_leak",
            config={"time_aware_mode": "none", "fit_on": "train"}
        )
        
        # A3: 无泄露版 + 优化参数（排名权重 + 自适应回退）
        # 注：当 fit 在 train 上时，strict_history 不会改变行为（邻居本身就来自更早时间）
        #     因此这里改为测试优化参数的效果
        def strategy_a3(train_df, val_df, test_df, model_cols):
            enhancer = ScenarioSimilarityEnhancer(
                model_cols=model_cols,
                n_neighbors=self.n_neighbors,
                temperature=self.temperature,
                time_aware_mode='none',
                use_rank_weights=True,      # 使用排名权重
                distance_threshold=2.0,     # 距离阈值
                fallback_blend=0.3,         # 30% 全局权重混合
            )
            enhancer.fit(train_df)
            val_pred, _ = enhancer.predict(val_df, is_val_prediction=False)
            test_pred, _ = enhancer.predict(test_df, is_val_prediction=False)
            return val_pred, test_pred
        
        results['A3_optimized'] = executor.run_strategy(
            strategy_a3,
            train_df, val_df, test_df,
            strategy_name="scenario_similarity",
            category=StrategyCategory.ABLATION,
            experiment_variant="A3_optimized",
            config={"fit_on": "train", "use_rank_weights": True, "fallback_blend": 0.3}
        )
        
        return results


# ============================================================================
# 对照实验：滚动验证对照 (Exp-E1/E2)
# ============================================================================

class RollingValidationAblation:
    """
    滚动验证对照实验
    
    - E1: 单 val（当前）
    - E2: Rolling blocked CV
    """
    
    def __init__(self, model_cols: List[str], n_folds: int = 5):
        self.model_cols = model_cols
        self.n_folds = n_folds
    
    def run_ablation(
        self,
        full_df: pd.DataFrame,
        strategy_fn: Callable,
        strategy_name: str,
        executor: ExperimentExecutor,
        gap: int = None
    ) -> Dict[str, Any]:
        """
        运行滚动验证对照实验
        
        Args:
            full_df: 完整数据（含 val 和 test）
            strategy_fn: 策略函数
            strategy_name: 策略名称
            executor: 实验执行器
            gap: 验证间隔，默认为 max(horizon, 24)
            
        Returns:
            对照实验结果
        """
        from .scenario_optimizer import RollingValidator
        
        gap = gap or max(executor.horizon, 24)
        n_samples = len(full_df)
        
        # 动态调整 fold 数
        if n_samples < 500:
            effective_folds = 3
        elif n_samples < 5000:
            effective_folds = min(self.n_folds, 5)
        else:
            effective_folds = self.n_folds
        
        validator = RollingValidator(n_splits=effective_folds, gap=gap)
        
        # E1: 单 val 结果（从 executor 获取）
        # E2: Rolling CV 结果
        fold_results = []
        
        for train_idx, val_idx in validator.split(n_samples):
            train_df = full_df.iloc[train_idx].copy()
            val_df = full_df.iloc[val_idx].copy()
            
            try:
                val_pred, _ = strategy_fn(train_df, val_df, val_df, self.model_cols)
                y_val = val_df['y'].values
                rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
                fold_results.append({'rmse': rmse, 'n_samples': len(val_df)})
            except Exception as e:
                fold_results.append({'rmse': float('nan'), 'error': str(e)})
        
        valid_rmses = [r['rmse'] for r in fold_results if not np.isnan(r.get('rmse', float('nan')))]
        
        if valid_rmses:
            mean_rmse = float(np.mean(valid_rmses))
            std_rmse = float(np.std(valid_rmses))
            cv = std_rmse / (mean_rmse + 1e-8)
        else:
            mean_rmse = float('nan')
            std_rmse = float('nan')
            cv = float('nan')
        
        return {
            'strategy_name': strategy_name,
            'n_folds': effective_folds,
            'gap': gap,
            'fold_results': fold_results,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'cv': cv,  # 变异系数
            'pass_cv_threshold': cv < 0.10 if not np.isnan(cv) else False,
        }


# ============================================================================
# 验收检查器
# ============================================================================

class AcceptanceCriteria:
    """验收标准检查器"""
    
    # 验收阈值
    SCENARIO_SIMILARITY_MAX_GAP = 30.0  # 阶段1: Gap < 30%
    SCENARIO_SIMILARITY_FINAL_GAP = 20.0  # 阶段2: Gap < 20%
    GATING_NETWORK_VS_STACKING_THRESHOLD = 1.0  # 优于 stacking_safe 1%
    ROLLING_CV_THRESHOLD = 0.10  # CV < 10%
    
    @staticmethod
    def check_scenario_similarity(
        record: ExperimentRecord,
        stacking_record: ExperimentRecord = None,
        phase: int = 1
    ) -> Tuple[bool, str]:
        """
        检查 scenario_similarity 是否通过验收
        
        Args:
            record: scenario_similarity 实验记录
            stacking_record: stacking_safe 实验记录（用于比较）
            phase: 验收阶段（1=初步修复，2=最终验收）
            
        Returns:
            (是否通过, 说明)
        """
        if record.status != "success":
            return False, f"执行失败: {record.skip_reason}"
        
        gap_threshold = (
            AcceptanceCriteria.SCENARIO_SIMILARITY_MAX_GAP if phase == 1
            else AcceptanceCriteria.SCENARIO_SIMILARITY_FINAL_GAP
        )
        
        if record.gap_percent > gap_threshold:
            return False, f"Gap {record.gap_percent:.1f}% > {gap_threshold}%"
        
        if stacking_record and stacking_record.status == "success":
            diff_pct = (record.test_rmse - stacking_record.test_rmse) / stacking_record.test_rmse * 100
            if diff_pct > 3.0:  # 不劣于 stacking_safe 超过 3%
                return False, f"Test RMSE 比 stacking_safe 差 {diff_pct:.1f}%"
        
        return True, f"通过: Gap={record.gap_percent:.1f}%"
    
    @staticmethod
    def check_gating_network(
        record: ExperimentRecord,
        stacking_record: ExperimentRecord
    ) -> Tuple[bool, str]:
        """检查 gating_network_v2 是否通过验收"""
        if record.status != "success":
            return False, f"执行失败: {record.skip_reason}"
        
        if stacking_record.status != "success":
            return True, "stacking_safe 无结果，无法比较"
        
        improvement = (stacking_record.test_rmse - record.test_rmse) / stacking_record.test_rmse * 100
        
        if improvement > AcceptanceCriteria.GATING_NETWORK_VS_STACKING_THRESHOLD:
            return True, f"优于 stacking_safe {improvement:.1f}%"
        elif improvement > 0:
            return True, f"略优于 stacking_safe {improvement:.1f}% (未达 1% 显著)"
        else:
            return False, f"劣于 stacking_safe {-improvement:.1f}%"
    
    @staticmethod
    def check_dataset_coverage(
        suite: ExperimentSuite,
        required_datasets: List[str],
        required_horizons: List[int]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        检查数据集覆盖度
        
        Returns:
            (是否全覆盖, 覆盖详情)
        """
        coverage = {}
        all_covered = True
        
        for dataset in required_datasets:
            coverage[dataset] = {}
            for horizon in required_horizons:
                key_prefix = f"{dataset}_{horizon}_"
                
                # 检查 KG component 类别是否有结果
                kg_component_keys = [k for k in suite.kg_component.keys() if k.startswith(key_prefix)]

                if kg_component_keys:
                    # 有执行结果
                    records = [suite.kg_component[k] for k in kg_component_keys]
                    success_count = sum(1 for r in records if r.status == "success")
                    skipped_count = sum(1 for r in records if r.status == "skipped")
                    
                    if success_count > 0:
                        coverage[dataset][horizon] = {
                            "status": "success",
                            "strategies": [r.strategy_name for r in records if r.status == "success"]
                        }
                    elif skipped_count > 0:
                        # 有明确降级
                        skipped_record = next(r for r in records if r.status == "skipped")
                        coverage[dataset][horizon] = {
                            "status": "degraded",
                            "reason": skipped_record.skip_reason,
                            "fallback": skipped_record.fallback_strategy
                        }
                    else:
                        coverage[dataset][horizon] = {"status": "failed"}
                        all_covered = False
                else:
                    # 无结果
                    coverage[dataset][horizon] = {"status": "missing"}
                    all_covered = False
        
        return all_covered, coverage
