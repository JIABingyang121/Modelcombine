"""
对照实验运行脚本

功能：
1. 运行泄露对照实验 (Exp-A1/A2/A3)
2. 运行滚动验证对照实验 (Exp-E1/E2)
3. 运行多数据集兼容性检查
4. 生成隔离的实验结果文件

结果输出结构：
result/ablation/
├── leakage_ablation.json      # 泄露对照结果
├── rolling_validation.json    # 滚动验证结果
├── dataset_diagnostic.json    # 数据集诊断结果
└── acceptance_report.json     # 验收报告

注意：本脚本复用 modelcombine_eval.py 中的数据加载和预处理函数，
确保与主评估脚本的数据处理逻辑一致。
"""

import os
# 限制 OpenBLAS 线程数，防止内存错误
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 复用 modelcombine_eval.py 中的数据加载和预处理函数
from scripts.modelcombine_eval import (
    load_split,
    prepare_supervised,
    make_aligned_stack,
    compute_naive_scale,
    train_base_models,
    predict_models,
)

from src.selector import (
    StrategyCategory,
    ExperimentRecord,
    ExperimentSuite,
    DatasetDiagnostic,
    ExperimentExecutor,
    LeakageAblation,
    RollingValidationAblation,
    AcceptanceCriteria,
    ScenarioSimilarityEnhancer,
    DirectWeightGatingNetwork,
    AdaptiveBucketSelector,
)


# ============================================================================
# 数据准备（复用 modelcombine_eval.py 的函数）
# ============================================================================

def prepare_ablation_data(
    feature_dir: Path,
    target_col: str,
    horizon: int,
    baseline_root: Path = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    准备对照实验数据，复用 modelcombine_eval.py 的数据处理逻辑
    
    Args:
        feature_dir: 特征数据目录
        target_col: 目标列名
        horizon: 预测时域
        baseline_root: 外部模型预测目录（可选）
        
    Returns:
        (train_df, val_df, test_df, model_cols)
        - 各 df 包含：y, timestamp, ctx_* 场景特征, 模型预测列
    """
    from src.models.registry import model_registry
    
    # 加载原始数据
    train_raw = load_split(feature_dir, "train")
    val_raw = load_split(feature_dir, "val")
    test_raw = load_split(feature_dir, "test")
    
    # 使用 prepare_supervised 处理各切分
    X_train, y_train, ts_train, ctx_train = prepare_supervised(train_raw, target_col, horizon)
    X_val, y_val, ts_val, ctx_val = prepare_supervised(val_raw, target_col, horizon)
    X_test, y_test, ts_test, ctx_test = prepare_supervised(test_raw, target_col, horizon)
    
    # 训练基础模型
    tree_models = ["xgboost_reg", "lgbm_reg", "catboost_reg"]
    fitted = train_base_models(tree_models, X_train, y_train)
    
    # 预测
    val_preds = predict_models(fitted, X_val)
    test_preds = predict_models(fitted, X_test)
    
    # 加载外部模型预测（prophet/arima 等）
    external_models = ["prophet", "arima", "power_difference", "multimodal_fusion"]
    dataset_name = feature_dir.parent.name if feature_dir.parent else "unknown"
    
    if baseline_root is not None:
        for ext_m in external_models:
            val_pred_file = baseline_root / dataset_name / f"val_pred_h{horizon}_{ext_m}.csv"
            test_pred_file = baseline_root / dataset_name / f"test_pred_h{horizon}_{ext_m}.csv"
            if val_pred_file.exists() and test_pred_file.exists():
                try:
                    val_ext = pd.read_csv(val_pred_file)
                    test_ext = pd.read_csv(test_pred_file)
                    val_ext["timestamp"] = pd.to_datetime(val_ext["timestamp"])
                    test_ext["timestamp"] = pd.to_datetime(test_ext["timestamp"])
                    val_preds[ext_m] = val_ext.set_index("timestamp")["pred"].reindex(ts_val.values).values
                    test_preds[ext_m] = test_ext.set_index("timestamp")["pred"].reindex(ts_test.values).values
                    
                    # 检查 NaN 并填充
                    if np.isnan(val_preds[ext_m]).sum() > len(val_preds[ext_m]) * 0.5:
                        del val_preds[ext_m]
                        del test_preds[ext_m]
                    else:
                        val_mean = np.nanmean(val_preds[ext_m])
                        val_preds[ext_m] = np.nan_to_num(val_preds[ext_m], nan=val_mean)
                        test_preds[ext_m] = np.nan_to_num(test_preds[ext_m], nan=val_mean)
                except Exception as e:
                    print(f"    加载 {ext_m} 失败: {e}")
    
    # 确定可用模型列
    all_models = tree_models + external_models
    model_cols = [m for m in all_models if m in val_preds and m in test_preds]
    
    # 使用 make_aligned_stack 构建对齐的数据框
    train_df = make_aligned_stack(
        {m: fitted[m].predict(X_train) for m in tree_models if m in fitted},
        y_train, ts_train, [m for m in tree_models if m in fitted], 
        dataset_name, "train", context_features=ctx_train
    )
    val_df = make_aligned_stack(val_preds, y_val, ts_val, model_cols, dataset_name, "val", context_features=ctx_val)
    test_df = make_aligned_stack(test_preds, y_test, ts_test, model_cols, dataset_name, "test", context_features=ctx_test)
    
    return train_df, val_df, test_df, model_cols


# ============================================================================
# 对照实验执行器
# ============================================================================

class AblationRunner:
    """对照实验运行器"""
    
    def __init__(
        self,
        feature_root: Path,
        output_root: Path,
        datasets: List[str] = None,
        horizons: List[int] = None,
        baseline_root: Path = None,
        max_samples: int = None,
        n_folds: int = 3,
        quick_mode: bool = False
    ):
        self.feature_root = Path(feature_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 基线预测根目录（用于加载外部模型预测）
        self.baseline_root = Path(baseline_root) if baseline_root else None
        
        # 默认数据集和时域配置
        self.datasets = datasets or ["pjm", "aemo_vic", "aemo_nsw"]
        self.horizons = horizons or [1, 24]
        
        # 实验规模控制
        self.max_samples = max_samples or (5000 if quick_mode else None)
        self.n_folds = n_folds if not quick_mode else 3
        self.quick_mode = quick_mode
        
        # 数据集配置（与实际数据列名一致）
        self.dataset_config = {
            "pjm": {"target_col": "load", "seasonality": 24},
            "aemo_vic": {"target_col": "load", "seasonality": 24},
            "aemo_nsw": {"target_col": "load", "seasonality": 24},
        }
        
        # 结果存储
        self.leakage_results = {}
        self.rolling_results = {}
        self.diagnostic_results = {}
        self.acceptance_results = {}
    
    def run_leakage_ablation(self, dataset: str, horizon: int) -> Dict[str, Any]:
        """
        运行泄露对照实验 (Exp-A1/A2/A3)
        """
        print(f"\n{'='*60}")
        print(f"[泄露对照] {dataset} h={horizon}")
        print(f"{'='*60}")
        
        # 准备数据
        feature_dir = self.feature_root / dataset
        if not feature_dir.exists():
            print(f"  数据目录不存在: {feature_dir}")
            return {"status": "skipped", "reason": "数据目录不存在"}
        
        config = self.dataset_config.get(dataset, {"target_col": "value", "seasonality": 24})
        
        try:
            train_df, val_df, test_df, model_cols = prepare_ablation_data(
                feature_dir, config["target_col"], horizon, self.baseline_root
            )
        except Exception as e:
            print(f"  数据加载失败: {e}")
            return {"status": "failed", "reason": str(e)}
        
        # 应用采样限制（保留最近的数据）
        if self.max_samples:
            if len(val_df) > self.max_samples // 2:
                val_df = val_df.tail(self.max_samples // 2).reset_index(drop=True)
            if len(test_df) > self.max_samples // 2:
                test_df = test_df.tail(self.max_samples // 2).reset_index(drop=True)
            print(f"  [采样] val={len(val_df)}, test={len(test_df)}")
        
        naive_scale = compute_naive_scale(train_df['y'].values, config["seasonality"])
        
        # 创建执行器
        executor = ExperimentExecutor(
            dataset_name=dataset,
            horizon=horizon,
            model_cols=model_cols,
            naive_scale=naive_scale
        )
        
        results = {"dataset": dataset, "horizon": horizon, "variants": {}}
        
        # =====================================================================
        # 对照组：有泄露的实现（用于对比）
        # =====================================================================
        
        # A1a: 完全泄露版（fit 在 val 上，无任何防护）
        print(f"  [A1a] 完全泄露版（fit=val, 无防护）...")
        try:
            enhancer_a1a = ScenarioSimilarityEnhancer(
                model_cols=model_cols,
                n_neighbors=10,
                temperature=2.0,
                time_aware_mode='none'  # 不启用邻居过滤
            )
            enhancer_a1a.fit(val_df)
            # is_val_prediction=False，不触发任何过滤
            val_pred_a1a, _ = enhancer_a1a.predict(val_df, is_val_prediction=False)
            test_pred_a1a, _ = enhancer_a1a.predict(test_df, is_val_prediction=False)
            
            val_rmse_a1a = float(np.sqrt(mean_squared_error(val_df['y'].values, val_pred_a1a)))
            test_rmse_a1a = float(np.sqrt(mean_squared_error(test_df['y'].values, test_pred_a1a)))
            gap_a1a = (test_rmse_a1a - val_rmse_a1a) / val_rmse_a1a * 100
            
            results["variants"]["A1a_full_leak"] = {
                "val_rmse": val_rmse_a1a,
                "test_rmse": test_rmse_a1a,
                "gap_percent": gap_a1a,
                "status": "success",
                "note": "完全泄露：fit=val + 无防护（自匹配）"
            }
            print(f"    Val RMSE: {val_rmse_a1a:.2f}, Test RMSE: {test_rmse_a1a:.2f}, Gap: {gap_a1a:.1f}%")
        except Exception as e:
            results["variants"]["A1a_full_leak"] = {"status": "failed", "reason": str(e)}
            print(f"    失败: {e}")
        
        # A1b: 泄露版 + leave_one_out 防护（验证 time_aware_mode 是否有效）
        print(f"  [A1b] 泄露版 + leave_one_out 防护...")
        try:
            enhancer_a1b = ScenarioSimilarityEnhancer(
                model_cols=model_cols,
                n_neighbors=10,
                temperature=2.0,
                time_aware_mode='leave_one_out'  # 启用邻居过滤
            )
            enhancer_a1b.fit(val_df)
            # is_val_prediction=True，触发 leave_one_out 过滤
            val_pred_a1b, _ = enhancer_a1b.predict(val_df, is_val_prediction=True)
            test_pred_a1b, _ = enhancer_a1b.predict(test_df, is_val_prediction=False)
            
            val_rmse_a1b = float(np.sqrt(mean_squared_error(val_df['y'].values, val_pred_a1b)))
            test_rmse_a1b = float(np.sqrt(mean_squared_error(test_df['y'].values, test_pred_a1b)))
            gap_a1b = (test_rmse_a1b - val_rmse_a1b) / val_rmse_a1b * 100
            
            results["variants"]["A1b_leak_with_loo"] = {
                "val_rmse": val_rmse_a1b,
                "test_rmse": test_rmse_a1b,
                "gap_percent": gap_a1b,
                "status": "success",
                "note": "泄露版 + leave_one_out 防护"
            }
            print(f"    Val RMSE: {val_rmse_a1b:.2f}, Test RMSE: {test_rmse_a1b:.2f}, Gap: {gap_a1b:.1f}%")
        except Exception as e:
            results["variants"]["A1b_leak_with_loo"] = {"status": "failed", "reason": str(e)}
            print(f"    失败: {e}")
        
        # =====================================================================
        # 修复版：无泄露的实现
        # =====================================================================
        
        # A2: 修复版（fit 在 train 上）
        print(f"  [A2] 修复版（fit 在 train 上，无泄露）...")
        try:
            enhancer_a2 = ScenarioSimilarityEnhancer(
                model_cols=model_cols,
                n_neighbors=10,
                temperature=2.0,
                time_aware_mode='none'
            )
            # 无泄露：在 train 上 fit，然后在 val/test 上预测
            enhancer_a2.fit(train_df)
            val_pred_a2, _ = enhancer_a2.predict(val_df)
            test_pred_a2, _ = enhancer_a2.predict(test_df)
            
            val_rmse_a2 = float(np.sqrt(mean_squared_error(val_df['y'].values, val_pred_a2)))
            test_rmse_a2 = float(np.sqrt(mean_squared_error(test_df['y'].values, test_pred_a2)))
            gap_a2 = (test_rmse_a2 - val_rmse_a2) / val_rmse_a2 * 100
            
            results["variants"]["A2_no_leak"] = {
                "val_rmse": val_rmse_a2,
                "test_rmse": test_rmse_a2,
                "gap_percent": gap_a2,
                "status": "success",
                "note": "无泄露：fit 在 train 上"
            }
            print(f"    Val RMSE: {val_rmse_a2:.2f}, Test RMSE: {test_rmse_a2:.2f}, Gap: {gap_a2:.1f}%")
        except Exception as e:
            results["variants"]["A2_no_leak"] = {"status": "failed", "reason": str(e)}
            print(f"    失败: {e}")
        
        # A3: 优化版（排名权重 + 自适应回退）
        print(f"  [A3] 优化版（排名权重 + 自适应回退）...")
        try:
            enhancer_a3 = ScenarioSimilarityEnhancer(
                model_cols=model_cols,
                n_neighbors=10,
                temperature=2.0,
                use_rank_weights=True,      # 使用排名权重
                distance_threshold=2.0,     # 距离阈值
                fallback_blend=0.3,         # 30% 全局权重混合
            )
            enhancer_a3.fit(train_df)
            val_pred_a3, _ = enhancer_a3.predict(val_df)
            test_pred_a3, _ = enhancer_a3.predict(test_df)
            
            val_rmse_a3 = float(np.sqrt(mean_squared_error(val_df['y'].values, val_pred_a3)))
            test_rmse_a3 = float(np.sqrt(mean_squared_error(test_df['y'].values, test_pred_a3)))
            gap_a3 = (test_rmse_a3 - val_rmse_a3) / val_rmse_a3 * 100
            
            results["variants"]["A3_optimized"] = {
                "val_rmse": val_rmse_a3,
                "test_rmse": test_rmse_a3,
                "gap_percent": gap_a3,
                "status": "success",
                "note": "优化版：排名权重 + 自适应回退"
            }
            print(f"    Val RMSE: {val_rmse_a3:.2f}, Test RMSE: {test_rmse_a3:.2f}, Gap: {gap_a3:.1f}%")
        except Exception as e:
            results["variants"]["A3_optimized"] = {"status": "failed", "reason": str(e)}
            print(f"    失败: {e}")
        
        # A4: 优化版 + 更强回退
        print(f"  [A4] 优化版 + 50%回退...")
        try:
            enhancer_a4 = ScenarioSimilarityEnhancer(
                model_cols=model_cols,
                n_neighbors=15,
                temperature=2.0,
                use_rank_weights=True,
                distance_threshold=1.5,     # 更严格的距离阈值
                fallback_blend=0.5,         # 50% 全局权重混合
            )
            enhancer_a4.fit(train_df)
            val_pred_a4, _ = enhancer_a4.predict(val_df)
            test_pred_a4, _ = enhancer_a4.predict(test_df)
            
            val_rmse_a4 = float(np.sqrt(mean_squared_error(val_df['y'].values, val_pred_a4)))
            test_rmse_a4 = float(np.sqrt(mean_squared_error(test_df['y'].values, test_pred_a4)))
            gap_a4 = (test_rmse_a4 - val_rmse_a4) / val_rmse_a4 * 100
            
            results["variants"]["A4_optimized_strong"] = {
                "val_rmse": val_rmse_a4,
                "test_rmse": test_rmse_a4,
                "gap_percent": gap_a4,
                "status": "success",
                "note": "优化版：排名权重 + 50%回退 + 严格距离阈值"
            }
            print(f"    Val RMSE: {val_rmse_a4:.2f}, Test RMSE: {test_rmse_a4:.2f}, Gap: {gap_a4:.1f}%")
        except Exception as e:
            results["variants"]["A4_optimized_strong"] = {"status": "failed", "reason": str(e)}
            print(f"    失败: {e}")
        
        # A5: 时间衰减版（只使用最近30%训练数据）
        print(f"  [A5] 时间衰减版（recent=30% + decay=1.0）...")
        try:
            enhancer_a5 = ScenarioSimilarityEnhancer(
                model_cols=model_cols,
                n_neighbors=10,
                temperature=2.0,
                use_rank_weights=True,
                distance_threshold=2.0,
                fallback_blend=0.3,
                time_decay=1.0,             # 中等强度时间衰减
                recent_ratio=0.3,           # 只用最近30%训练数据
            )
            enhancer_a5.fit(train_df)
            val_pred_a5, _ = enhancer_a5.predict(val_df)
            test_pred_a5, _ = enhancer_a5.predict(test_df)
            
            val_rmse_a5 = float(np.sqrt(mean_squared_error(val_df['y'].values, val_pred_a5)))
            test_rmse_a5 = float(np.sqrt(mean_squared_error(test_df['y'].values, test_pred_a5)))
            gap_a5 = (test_rmse_a5 - val_rmse_a5) / val_rmse_a5 * 100
            
            results["variants"]["A5_time_decay"] = {
                "val_rmse": val_rmse_a5,
                "test_rmse": test_rmse_a5,
                "gap_percent": gap_a5,
                "status": "success",
                "note": "时间衰减版：recent=30% + decay=1.0"
            }
            print(f"    Val RMSE: {val_rmse_a5:.2f}, Test RMSE: {test_rmse_a5:.2f}, Gap: {gap_a5:.1f}%")
        except Exception as e:
            results["variants"]["A5_time_decay"] = {"status": "failed", "reason": str(e)}
            print(f"    失败: {e}")
        
        # A6: 强时间衰减版（只使用最近20%训练数据 + 强衰减）
        print(f"  [A6] 强时间衰减版（recent=20% + decay=2.0）...")
        try:
            enhancer_a6 = ScenarioSimilarityEnhancer(
                model_cols=model_cols,
                n_neighbors=8,
                temperature=2.0,
                use_rank_weights=True,
                distance_threshold=1.5,
                fallback_blend=0.4,
                time_decay=2.0,             # 强时间衰减
                recent_ratio=0.2,           # 只用最近20%训练数据
            )
            enhancer_a6.fit(train_df)
            val_pred_a6, _ = enhancer_a6.predict(val_df)
            test_pred_a6, _ = enhancer_a6.predict(test_df)
            
            val_rmse_a6 = float(np.sqrt(mean_squared_error(val_df['y'].values, val_pred_a6)))
            test_rmse_a6 = float(np.sqrt(mean_squared_error(test_df['y'].values, test_pred_a6)))
            gap_a6 = (test_rmse_a6 - val_rmse_a6) / val_rmse_a6 * 100
            
            results["variants"]["A6_strong_decay"] = {
                "val_rmse": val_rmse_a6,
                "test_rmse": test_rmse_a6,
                "gap_percent": gap_a6,
                "status": "success",
                "note": "强时间衰减版：recent=20% + decay=2.0"
            }
            print(f"    Val RMSE: {val_rmse_a6:.2f}, Test RMSE: {test_rmse_a6:.2f}, Gap: {gap_a6:.1f}%")
        except Exception as e:
            results["variants"]["A6_strong_decay"] = {"status": "failed", "reason": str(e)}
            print(f"    失败: {e}")
        
        # 计算基线对比（stacking_safe 近似）
        print(f"  [Baseline] 计算 simple_avg 作为参考...")
        try:
            available_models = [m for m in model_cols if m in test_df.columns]
            simple_avg_test = test_df[available_models].mean(axis=1).values
            simple_avg_rmse = float(np.sqrt(mean_squared_error(test_df['y'].values, simple_avg_test)))
            results["baseline_simple_avg_test_rmse"] = simple_avg_rmse
            print(f"    Simple Avg Test RMSE: {simple_avg_rmse:.2f}")
        except Exception as e:
            print(f"    计算失败: {e}")
        
        return results
    
    def run_rolling_validation(self, dataset: str, horizon: int) -> Dict[str, Any]:
        """
        运行滚动验证对照实验 (Exp-E1/E2)
        """
        print(f"\n{'='*60}")
        print(f"[滚动验证] {dataset} h={horizon}")
        print(f"{'='*60}")
        
        # 准备数据
        feature_dir = self.feature_root / dataset
        if not feature_dir.exists():
            print(f"  数据目录不存在: {feature_dir}")
            return {"status": "skipped", "reason": "数据目录不存在"}
        
        config = self.dataset_config.get(dataset, {"target_col": "value", "seasonality": 24})
        
        try:
            train_df, val_df, test_df, model_cols = prepare_ablation_data(
                feature_dir, config["target_col"], horizon, self.baseline_root
            )
        except Exception as e:
            print(f"  数据加载失败: {e}")
            return {"status": "failed", "reason": str(e)}
        
        # 合并 train + val 用于滚动验证
        full_df = pd.concat([train_df, val_df], ignore_index=True)
        
        # 应用采样限制
        if self.max_samples and len(full_df) > self.max_samples:
            full_df = full_df.tail(self.max_samples).reset_index(drop=True)
            print(f"  [采样] 限制为最近 {self.max_samples} 条数据")
        
        n_samples = len(full_df)
        
        # 使用配置的 fold 数量
        n_folds = min(self.n_folds, max(2, n_samples // 200))
        
        gap = max(horizon, config["seasonality"])
        
        results = {
            "dataset": dataset,
            "horizon": horizon,
            "n_samples": n_samples,
            "n_folds": n_folds,
            "gap": gap,
            "strategies": {}
        }
        
        # 测试 gating_network_v2
        print(f"  [gating_network_v2] 滚动验证 ({n_folds} folds)...")
        try:
            fold_rmses = []
            fold_size = n_samples // (n_folds + 2)
            min_train = max(100, n_samples // 4)
            
            for i in range(n_folds):
                train_end = min_train + fold_size * i
                val_start = train_end + gap
                val_end = min(val_start + fold_size, n_samples)
                
                if val_end <= val_start:
                    break
                
                fold_train = full_df.iloc[:train_end].copy()
                fold_val = full_df.iloc[val_start:val_end].copy()
                
                gating = DirectWeightGatingNetwork(
                    model_cols=model_cols,
                    temperature=2.0,
                    cv_folds=3,
                    use_enhanced_features=True
                )
                gating.fit(fold_train)
                val_pred, _ = gating.predict(fold_val)
                rmse = float(np.sqrt(mean_squared_error(fold_val['y'].values, val_pred)))
                fold_rmses.append(rmse)
                print(f"    Fold {i+1}: RMSE={rmse:.2f}")
            
            if fold_rmses:
                mean_rmse = float(np.mean(fold_rmses))
                std_rmse = float(np.std(fold_rmses))
                cv = std_rmse / (mean_rmse + 1e-8)
                results["strategies"]["gating_network_v2"] = {
                    "fold_rmses": fold_rmses,
                    "mean_rmse": mean_rmse,
                    "std_rmse": std_rmse,
                    "cv": cv,
                    "pass_cv_threshold": cv < 0.10,
                    "status": "success"
                }
                print(f"    Mean RMSE: {mean_rmse:.2f}, Std: {std_rmse:.2f}, CV: {cv:.3f}")
        except Exception as e:
            results["strategies"]["gating_network_v2"] = {"status": "failed", "reason": str(e)}
            print(f"    失败: {e}")
        
        # 测试 scenario_similarity (strict_history)
        print(f"  [scenario_similarity] 滚动验证...")
        try:
            import gc
            fold_rmses = []
            
            for i in range(n_folds):
                train_end = min_train + fold_size * i
                val_start = train_end + gap
                val_end = min(val_start + fold_size, n_samples)
                
                if val_end <= val_start:
                    break
                
                fold_train = full_df.iloc[:train_end].copy()
                fold_val = full_df.iloc[val_start:val_end].copy()
                
                # 小样本时减少邻居数
                n_neighbors = min(10, max(3, len(fold_train) // 10))
                
                try:
                    similarity = ScenarioSimilarityEnhancer(
                        model_cols=model_cols,
                        n_neighbors=n_neighbors,
                        temperature=2.0,
                        time_aware_mode='strict_history'
                    )
                    similarity.fit(fold_train)
                    val_pred, _ = similarity.predict(fold_val, is_val_prediction=False)
                    rmse = float(np.sqrt(mean_squared_error(fold_val['y'].values, val_pred)))
                    fold_rmses.append(rmse)
                    print(f"    Fold {i+1}: RMSE={rmse:.2f}")
                except Exception as fold_e:
                    print(f"    Fold {i+1}: 失败 - {fold_e}")
                finally:
                    # 清理内存
                    gc.collect()
            
            if fold_rmses:
                mean_rmse = float(np.mean(fold_rmses))
                std_rmse = float(np.std(fold_rmses))
                cv = std_rmse / (mean_rmse + 1e-8)
                results["strategies"]["scenario_similarity"] = {
                    "fold_rmses": fold_rmses,
                    "mean_rmse": mean_rmse,
                    "std_rmse": std_rmse,
                    "cv": cv,
                    "pass_cv_threshold": cv < 0.10,
                    "status": "success"
                }
                print(f"    Mean RMSE: {mean_rmse:.2f}, Std: {std_rmse:.2f}, CV: {cv:.3f}")
        except Exception as e:
            results["strategies"]["scenario_similarity"] = {"status": "failed", "reason": str(e)}
            print(f"    失败: {e}")
        
        return results
    
    def run_diagnostic(self, dataset: str, horizon: int) -> Dict[str, Any]:
        """
        运行数据集兼容性诊断
        """
        print(f"\n{'='*60}")
        print(f"[数据诊断] {dataset} h={horizon}")
        print(f"{'='*60}")
        
        feature_dir = self.feature_root / dataset
        if not feature_dir.exists():
            return {"status": "skipped", "reason": "数据目录不存在"}
        
        config = self.dataset_config.get(dataset, {"target_col": "value", "seasonality": 24})
        
        try:
            train_df, val_df, test_df, model_cols = prepare_ablation_data(
                feature_dir, config["target_col"], horizon, self.baseline_root
            )
        except Exception as e:
            return {"status": "failed", "reason": str(e)}
        
        diagnostic = DatasetDiagnostic()
        diag_result = diagnostic.diagnose(val_df, model_cols, dataset, horizon)
        
        print(f"  兼容性: {'✓' if diag_result.is_compatible else '✗'}")
        if diag_result.issues:
            print(f"  问题: {diag_result.issues}")
        if diag_result.warnings:
            print(f"  警告: {diag_result.warnings}")
        if diag_result.recommendations:
            print(f"  建议: {diag_result.recommendations}")
        
        use_complex, fallback = diagnostic.should_use_complex_strategy(diag_result)
        print(f"  推荐: {'使用复杂策略' if use_complex else f'降级到 {fallback}'}")
        
        return {
            "dataset": dataset,
            "horizon": horizon,
            "is_compatible": diag_result.is_compatible,
            "issues": diag_result.issues,
            "warnings": diag_result.warnings,
            "recommendations": diag_result.recommendations,
            "use_complex_strategy": use_complex,
            "fallback_strategy": fallback,
            "n_samples_val": len(val_df),
            "n_samples_test": len(test_df),
            "n_ctx_features": len([c for c in val_df.columns if c.startswith('ctx_')]),
            "n_models": len(model_cols)
        }
    
    def run_all(self):
        """运行所有对照实验"""
        print("\n" + "="*70)
        print("开始运行对照实验")
        print("="*70)
        
        for dataset in self.datasets:
            for horizon in self.horizons:
                key = f"{dataset}_h{horizon}"
                
                try:
                    # 1. 数据诊断
                    self.diagnostic_results[key] = self.run_diagnostic(dataset, horizon)
                except Exception as e:
                    print(f"  [ERROR] 数据诊断失败: {e}")
                    self.diagnostic_results[key] = {"status": "error", "reason": str(e)}
                
                try:
                    # 2. 泄露对照
                    self.leakage_results[key] = self.run_leakage_ablation(dataset, horizon)
                except Exception as e:
                    print(f"  [ERROR] 泄露对照失败: {e}")
                    self.leakage_results[key] = {"status": "error", "reason": str(e)}
                
                try:
                    # 3. 滚动验证
                    self.rolling_results[key] = self.run_rolling_validation(dataset, horizon)
                except Exception as e:
                    print(f"  [ERROR] 滚动验证失败: {e}")
                    self.rolling_results[key] = {"status": "error", "reason": str(e)}
                
                # 增量保存（每完成一个 dataset/horizon 就保存）
                self._save_results()
                print(f"  [已保存] {key} 结果")
        
        # 生成验收报告
        self._generate_acceptance_report()
        
        # 最终保存
        self._save_results()
        
        print("\n" + "="*70)
        print("对照实验完成")
        print(f"结果已保存到: {self.output_root}")
        print("="*70)
    
    def _generate_acceptance_report(self):
        """生成验收报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "details": {}
        }
        
        for key in self.leakage_results:
            leakage = self.leakage_results[key]
            rolling = self.rolling_results.get(key, {})
            diagnostic = self.diagnostic_results.get(key, {})
            
            # 检查 scenario_similarity 泄露对照
            if "variants" in leakage:
                # 验证1: A1a vs A1b 证明 leave_one_out 有效
                a1a = leakage["variants"].get("A1a_full_leak", {})
                a1b = leakage["variants"].get("A1b_leak_with_loo", {})
                if a1a.get("status") == "success" and a1b.get("status") == "success":
                    # A1b 的 Val RMSE 应该显著高于 A1a（证明 leave_one_out 生效）
                    loo_effective = a1b.get("val_rmse", 0) > a1a.get("val_rmse", float('inf')) * 1.1
                    report["details"][f"{key}_loo_effectiveness"] = {
                        "a1a_val_rmse": a1a.get("val_rmse"),
                        "a1b_val_rmse": a1b.get("val_rmse"),
                        "loo_effective": loo_effective
                    }
                
                # 验证2: A2 (无泄露版) Gap < 30%
                a2 = leakage["variants"].get("A2_no_leak", {})
                if a2.get("status") == "success":
                    gap = a2.get("gap_percent", float('inf'))
                    passed = gap < 30  # 阶段 1 标准
                    report["details"][f"{key}_scenario_similarity"] = {
                        "gap_percent": gap,
                        "passed_phase1": passed,
                        "test_rmse": a2.get("test_rmse"),
                        "note": "A2_no_leak: fit=train, 无泄露"
                    }
                
                # 验证3: 优化版（A3-A6）中最佳 Gap
                best_opt_gap = float('inf')
                best_opt_name = None
                for opt_name in ["A3_optimized", "A4_optimized_strong", "A5_time_decay", "A6_strong_decay"]:
                    opt = leakage["variants"].get(opt_name, {})
                    if opt.get("status") == "success":
                        opt_gap = opt.get("gap_percent", float('inf'))
                        if opt_gap < best_opt_gap:
                            best_opt_gap = opt_gap
                            best_opt_name = opt_name
                if best_opt_name:
                    report["details"][f"{key}_best_optimization"] = {
                        "variant": best_opt_name,
                        "gap_percent": best_opt_gap,
                        "improvement_vs_a2": (a2.get("gap_percent", 0) - best_opt_gap) if a2.get("status") == "success" else None
                    }
            
            # 检查 gating_network_v2
            if "strategies" in rolling:
                gating = rolling["strategies"].get("gating_network_v2", {})
                if gating.get("status") == "success":
                    cv = gating.get("cv", float('inf'))
                    passed_cv = cv < 0.10
                    report["details"][f"{key}_gating_network_v2"] = {
                        "cv": cv,
                        "passed_cv_threshold": passed_cv,
                        "mean_rmse": gating.get("mean_rmse")
                    }
            
            # 数据集覆盖
            if diagnostic.get("is_compatible") is not None:
                report["details"][f"{key}_compatibility"] = {
                    "is_compatible": diagnostic.get("is_compatible"),
                    "use_complex": diagnostic.get("use_complex_strategy"),
                    "fallback": diagnostic.get("fallback_strategy")
                }
        
        # 汇总
        total_checks = len(report["details"])
        passed_checks = sum(
            1 for d in report["details"].values()
            if d.get("passed_phase1") or d.get("passed_cv_threshold") or d.get("is_compatible")
        )
        report["summary"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "pass_rate": passed_checks / max(total_checks, 1)
        }
        
        self.acceptance_results = report
    
    def _save_results(self):
        """保存所有结果"""
        # 泄露对照
        with open(self.output_root / "leakage_ablation.json", 'w', encoding='utf-8') as f:
            json.dump(self.leakage_results, f, indent=2, ensure_ascii=False)
        
        # 滚动验证
        with open(self.output_root / "rolling_validation.json", 'w', encoding='utf-8') as f:
            json.dump(self.rolling_results, f, indent=2, ensure_ascii=False)
        
        # 数据诊断
        with open(self.output_root / "dataset_diagnostic.json", 'w', encoding='utf-8') as f:
            json.dump(self.diagnostic_results, f, indent=2, ensure_ascii=False)
        
        # 验收报告
        with open(self.output_root / "acceptance_report.json", 'w', encoding='utf-8') as f:
            json.dump(self.acceptance_results, f, indent=2, ensure_ascii=False)


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="运行对照实验")
    parser.add_argument(
        "--feature-root",
        type=str,
        default="data/features",
        help="特征数据根目录"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="result/ablation",
        help="输出目录"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["pjm", "aemo_vic", "aemo_nsw"],
        help="要测试的数据集"
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 24],
        help="要测试的预测时域"
    )
    parser.add_argument(
        "--baseline-root",
        type=str,
        default=None,
        help="基线预测根目录（用于加载外部模型预测，如 Prophet/ARIMA）"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大样本数限制（默认不限制，--quick 模式下为 5000）"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="滚动验证 fold 数量（默认 3）"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式：限制样本 5000，fold 3"
    )
    
    args = parser.parse_args()
    
    runner = AblationRunner(
        feature_root=Path(args.feature_root),
        output_root=Path(args.output_root),
        datasets=args.datasets,
        horizons=args.horizons,
        baseline_root=Path(args.baseline_root) if args.baseline_root else None,
        max_samples=args.max_samples,
        n_folds=args.n_folds,
        quick_mode=args.quick
    )
    
    runner.run_all()


if __name__ == "__main__":
    main()
