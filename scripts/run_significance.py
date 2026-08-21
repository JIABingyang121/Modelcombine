"""
显著性检验 - 修复版

修复内容:
1. DM 检验：移除错误去均值，添加 Harvey 修正
2. 完全兼容原 significance(1).json schema
3. 按稳定键对齐预测
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.combination_utils import generate_stable_key, DATASET_HORIZONS, MODELS


# ============================================================
# 核心比较定义（双向匹配）
# ============================================================

# 使用 frozenset 存储策略对，支持双向匹配
CORE_COMPARISON_PAIRS = {
    frozenset({"stacking", "static_weight"}),
    frozenset({"stacking", "simple_avg"}),
    frozenset({"static_weight", "simple_avg"}),
    frozenset({"smart_weighted", "static_weight"}),
    frozenset({"smart_weighted", "simple_avg"}),
    frozenset({"smart_weighted", "stacking"}),
}

# 涉及 best_single 的比较（动态）
CORE_VS_BEST_SINGLE = {"stacking", "static_weight", "simple_avg", "smart_weighted"}

EXTRA_STRATEGIES = {"constrained_opt"}


def is_core_comparison(comp_name: str, best_single: str) -> bool:
    """判断是否为核心比较（支持双向）"""
    parts = comp_name.split("_vs_")
    if len(parts) != 2:
        return False
    
    s1, s2 = parts
    
    # 排除 extra 策略
    if s1 in EXTRA_STRATEGIES or s2 in EXTRA_STRATEGIES:
        return False
    
    # 检查固定策略对（双向）
    pair = frozenset({s1, s2})
    if pair in CORE_COMPARISON_PAIRS:
        return True
    
    # 检查 vs best_single（双向）
    if best_single:
        if s1 == best_single and s2 in CORE_VS_BEST_SINGLE:
            return True
        if s2 == best_single and s1 in CORE_VS_BEST_SINGLE:
            return True
    
    return False


def normalize_comparison_name(s1: str, s2: str) -> str:
    """标准化比较名（保证一致顺序）"""
    # 定义优先顺序
    priority = {
        "stacking": 0,
        "static_weight": 1,
        "smart_weighted": 2,
        "simple_avg": 3,
        "constrained_opt": 4,
    }
    
    p1 = priority.get(s1, 100)
    p2 = priority.get(s2, 100)
    
    if p1 <= p2:
        return f"{s1}_vs_{s2}"
    else:
        return f"{s2}_vs_{s1}"


# ============================================================
# 稳定键辅助（复用 combination_utils）
# ============================================================

def add_stable_key(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """为 DataFrame 添加稳定键（包装 combination_utils.generate_stable_key）"""
    stable_key, sorted_df = generate_stable_key(df, ts_col)
    sorted_df["_stable_key"] = stable_key
    return sorted_df


# ============================================================
# DM 检验（修复版）
# ============================================================

def dm_test_safe(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int) -> Dict[str, float]:
    """
    Diebold-Mariano 检验 - 修复版
    
    修复:
    1. 移除错误的去均值
    2. 添加 Harvey 修正
    3. horizon=1 时 max_lag=0
    4. 使用 t 分布
    """
    d = loss_a - loss_b
    T = len(d)
    
    if T < 10:
        return {"stat": float("nan"), "p_value": float("nan")}
    
    d_mean = np.mean(d)
    
    # HAC 带宽
    if horizon == 1:
        max_lag = 0
    else:
        max_lag = min(horizon - 1, int(np.floor(T ** (1/3))))
        max_lag = max(0, max_lag)
    
    # Newey-West HAC 方差估计
    d_centered = d - d_mean
    gamma_0 = np.var(d_centered, ddof=1)
    
    var_long_run = gamma_0
    for k in range(1, max_lag + 1):
        if k >= T:
            break
        gamma_k = np.mean(d_centered[k:] * d_centered[:-k])
        weight = 1 - k / (max_lag + 1)  # Bartlett 核
        var_long_run += 2 * weight * gamma_k
    
    if var_long_run <= 1e-10:
        return {"stat": float("nan"), "p_value": float("nan")}
    
    # Harvey 修正
    harvey_term = T + 1 - 2 * horizon + horizon * (horizon - 1) / T
    
    if harvey_term <= 0:
        stat = d_mean / np.sqrt(var_long_run / T)
    else:
        stat_raw = d_mean / np.sqrt(var_long_run / T)
        stat = stat_raw * np.sqrt(harvey_term / T)
    
    # t 分布 p 值
    df = max(T - 1, 1)
    p = 2 * (1 - stats.t.cdf(abs(stat), df))
    
    return {"stat": float(stat), "p_value": float(p)}


# ============================================================
# Wilcoxon 检验
# ============================================================

def wilcoxon_test(loss_a: np.ndarray, loss_b: np.ndarray) -> Dict[str, float]:
    """Wilcoxon 符号秩检验"""
    try:
        res = stats.wilcoxon(loss_a, loss_b, zero_method="wilcox", alternative='two-sided')
        return {"stat": float(res.statistic), "p_value": float(res.pvalue)}
    except Exception:
        return {"stat": float("nan"), "p_value": float("nan")}


# ============================================================
# Block Bootstrap
# ============================================================

def block_bootstrap_single(loss_a: np.ndarray, loss_b: np.ndarray,
                           block_size: int, n_bootstrap: int = 1000,
                           seed: int = 42) -> Dict:
    """单次 block bootstrap"""
    np.random.seed(seed)
    
    n = len(loss_a)
    d = loss_a - loss_b
    observed_diff = np.mean(d)
    
    if n == 0:
        return {
            "mean_diff": float("nan"),
            "p_value": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "block_size": block_size
        }
    
    n_blocks = int(np.ceil(n / block_size))
    boot_diffs = []
    
    for _ in range(n_bootstrap):
        block_starts = np.random.randint(0, max(1, n - block_size + 1), n_blocks)
        indices = []
        for start in block_starts:
            indices.extend(range(start, min(start + block_size, n)))
        indices = indices[:n]
        boot_diffs.append(np.mean(d[indices]))
    
    boot_diffs = np.array(boot_diffs)
    
    # 双侧 p 值
    if observed_diff >= 0:
        p_value = np.mean(boot_diffs <= 0) * 2
    else:
        p_value = np.mean(boot_diffs >= 0) * 2
    p_value = min(p_value, 1.0)
    
    return {
        "mean_diff": float(observed_diff),
        "p_value": float(p_value),
        "ci_low": float(np.percentile(boot_diffs, 2.5)),
        "ci_high": float(np.percentile(boot_diffs, 97.5)),
        "block_size": block_size
    }


def multi_seed_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray,
                         block_size: int, n_bootstrap: int = 1000,
                         seeds: List[int] = None) -> Dict:
    """多种子 bootstrap（使用中位 p 值）"""
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]
    
    results = [block_bootstrap_single(loss_a, loss_b, block_size, n_bootstrap, s) for s in seeds]
    p_values = [r["p_value"] for r in results]
    
    return {
        "mean_diff": float(np.median([r["mean_diff"] for r in results])),
        "p_value_median": float(np.median(p_values)),
        "p_value_min": float(np.min(p_values)),
        "p_value_max": float(np.max(p_values)),
        "ci_low": float(np.median([r["ci_low"] for r in results])),
        "ci_high": float(np.median([r["ci_high"] for r in results])),
        "n_seeds": len(seeds),
        "block_size": block_size
    }


# ============================================================
# Cohen's D
# ============================================================

def cohens_d(loss_a: np.ndarray, loss_b: np.ndarray) -> float:
    """计算 Cohen's D 效应量"""
    if len(loss_a) == 0 or len(loss_b) == 0:
        return float("nan")
    diff = loss_a - loss_b
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-8))


# ============================================================
# Holm-Bonferroni 校正
# ============================================================

def holm_correction_original_format(comparisons: Dict[str, float]) -> Dict:
    """
    Holm-Bonferroni 校正 - 兼容原 schema
    
    原格式: {comparison_name: {raw_p, adjusted_p, holm_alpha, significant}}
    significant 判定使用 raw_p < holm_alpha
    """
    items = [(k, v) for k, v in comparisons.items() if not np.isnan(v)]
    items.sort(key=lambda x: x[1])
    
    n = len(items)
    results = {}
    
    for i, (name, p) in enumerate(items):
        holm_alpha = 0.05 / (n - i)
        adjusted_p = min(p * (n - i), 1.0)
        
        results[name] = {
            "raw_p": float(p),
            "adjusted_p": float(adjusted_p),
            "holm_alpha": float(holm_alpha),
            "significant": p < holm_alpha  # 使用 raw_p < holm_alpha
        }
    
    # 添加 nan 的
    for k, v in comparisons.items():
        if np.isnan(v) and k not in results:
            results[k] = {
                "raw_p": float("nan"),
                "adjusted_p": float("nan"),
                "holm_alpha": float("nan"),
                "significant": False
            }
    
    return results


# ============================================================
# 数据加载（修复版 - 严格校验）
# ============================================================

def load_base_preds_safe(root: Path, dataset: str, horizon: int, 
                         models: List[str], split: str,
                         strict: bool = True) -> pd.DataFrame:
    """
    安全加载单模型预测
    
    修复：
    1. 每个文件独立添加稳定键
    2. 一致性问题抛出异常（strict=True）
    3. 最终强制校验行数
    """
    frames = []
    y_ref = None
    expected_len = None
    
    for mid in models:
        p = root / dataset / f"{split}_pred_h{horizon}_{mid}.csv"
        if not p.exists():
            continue
        
        df = pd.read_csv(p)
        
        # 每个文件独立添加稳定键
        df = add_stable_key(df, "timestamp")
        
        # 键唯一性校验
        if df["_stable_key"].duplicated().any():
            msg = f"模型 {mid} 存在重复稳定键"
            if strict:
                raise ValueError(msg)
            warnings.warn(msg + "，跳过")
            continue
        
        df = df.rename(columns={"pred": mid})
        
        # y 一致性校验
        if y_ref is None:
            y_ref = df[["_stable_key", "timestamp", "y"]].copy()
            expected_len = len(y_ref)
        else:
            check = y_ref.merge(
                df[["_stable_key", "y"]], 
                on="_stable_key", 
                suffixes=("_ref", "_new"),
                how="inner"
            )
            
            if len(check) != expected_len:
                msg = f"模型 {mid} merge 后行数变化: {expected_len} -> {len(check)}"
                if strict:
                    raise ValueError(msg)
                warnings.warn(msg)
                continue
            
            y_diff = np.abs(check["y_ref"] - check["y_new"])
            if y_diff.max() > 1e-6:
                msg = f"模型 {mid} 的 y 值不一致 (max_diff={y_diff.max():.6f})"
                if strict:
                    raise ValueError(msg)
                warnings.warn(msg)
                continue
        
        frames.append(df[["_stable_key", mid]])
    
    if not frames or y_ref is None:
        raise FileNotFoundError(f"no base preds for {dataset} h={horizon} split={split}")
    
    # Merge
    merged = y_ref.copy()
    for f in frames:
        merged = merged.merge(f, on="_stable_key", how="inner")
    
    # 强制校验最终行数
    if len(merged) != expected_len:
        msg = f"最终 merge 后行数变化: {expected_len} -> {len(merged)}"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)
    
    # 排序并清理
    merged["_ts_dt"] = pd.to_datetime(merged["timestamp"])
    merged = merged.sort_values("_ts_dt").reset_index(drop=True)
    merged = merged.drop(columns=["_stable_key", "_ts_dt"], errors="ignore")
    
    return merged


def load_combo_preds(root: Path, dataset: str, horizon: int, split: str) -> pd.DataFrame:
    """加载组合预测"""
    p = root / dataset / f"{split}_combos_h{horizon}.csv"
    if not p.exists():
        raise FileNotFoundError(f"no combo preds for {dataset} h={horizon} split={split}")
    return pd.read_csv(p)


def select_best_single(val_df: pd.DataFrame, base_cols: List[str]) -> str:
    """选择验证集上最佳单模型"""
    best = None
    best_mae = float("inf")
    y = val_df["y"].values.astype(float)
    for col in base_cols:
        mae = float(np.mean(np.abs(y - val_df[col].values.astype(float))))
        if mae < best_mae:
            best_mae = mae
            best = col
    return best


# ============================================================
# 主函数
# ============================================================

def run_significance(base_root: Path, combo_root: Path, out_path: Path,
                     n_bootstrap: int = 1000,
                     seeds: List[int] = None,
                     block_size: int = None) -> Dict:
    """
    运行显著性检验 - 完全兼容原 schema
    
    修复:
    1. 使用稳定键对齐
    2. 标准化比较名顺序
    3. Holm 校正仅对核心比较
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]
    
    results: Dict[str, Dict] = {}
    
    for dataset, horizons in DATASET_HORIZONS.items():
        results[dataset] = {}
        
        for h in horizons:
            print(f"  {dataset} h={h}...")
            
            try:
                # 使用安全加载（稳定键对齐）
                val_df = load_base_preds_safe(base_root, dataset, h, MODELS, split="val")
                test_base = load_base_preds_safe(base_root, dataset, h, MODELS, split="test")
                test_combos = load_combo_preds(combo_root, dataset, h, split="test")
            except FileNotFoundError as e:
                print(f"    跳过 (文件缺失): {e}")
                continue
            except (ValueError, KeyError, TypeError) as e:
                print(f"    跳过 (数据异常): {e}")
                continue
            
            # 对齐 test_base 和 test_combos（使用稳定键）
            try:
                # 为两者独立生成稳定键
                test_base = add_stable_key(test_base, "timestamp")
                test_combos = add_stable_key(test_combos, "timestamp")
                
                # 校验行数一致
                if len(test_base) != len(test_combos):
                    raise ValueError(f"test_base ({len(test_base)}) 与 test_combos ({len(test_combos)}) 行数不一致")
                
                # 使用稳定键合并
                test_merged = test_base.merge(
                    test_combos, on="_stable_key", how="inner", 
                    suffixes=("_base", "_combo")
                )
                
                if len(test_merged) != len(test_base):
                    raise ValueError(f"稳定键合并后行数变化: {len(test_base)} -> {len(test_merged)}，可能存在键不匹配")
                
                # 校验 y_base 与 y_combo 一致性
                if "y_base" in test_merged.columns and "y_combo" in test_merged.columns:
                    y_diff = np.abs(test_merged["y_base"].values - test_merged["y_combo"].values)
                    if y_diff.max() > 1e-6:
                        raise ValueError(f"y_base 与 y_combo 不一致 (max_diff={y_diff.max():.6f})，数据可能被污染")
                
            except (ValueError, KeyError, TypeError) as e:
                print(f"    跳过 (对齐失败): {e}")
                continue
            
            # 清理列
            if "y_base" in test_merged.columns:
                test_merged = test_merged.rename(columns={"y_base": "y"})
            drop_cols = ["y_combo", "_stable_key", "_row_id"]
            test_merged = test_merged.drop(columns=[c for c in drop_cols if c in test_merged.columns])
            
            base_cols = [c for c in MODELS if c in val_df.columns]
            combo_cols = [c for c in test_combos.columns if c not in ("timestamp", "y", "row_id", "_row_id")]
            
            best_single = select_best_single(val_df, base_cols)
            if best_single is None:
                print(f"    跳过: 无法确定 best_single")
                continue
            
            y_true = test_merged["y"].values.astype(float)
            
            # 计算 block_size
            current_block_size = block_size or max(2 * h, 12)
            current_block_size = min(current_block_size, len(y_true))
            
            res_h = {
                "best_single": best_single,  # 字符串
                "comparisons": {},
                "_extra_comparisons": {},
                "holm_correction_dm": {},
                "holm_correction_bootstrap": {}
            }
            
            # 收集 p 值（分开收集核心和额外）
            core_dm_p_values = {}
            core_bootstrap_p_values = {}
            
            # 构建所有策略
            all_strategies = list(combo_cols) + [best_single]
            all_strategies = [s for s in all_strategies if s in test_merged.columns]
            
            # 两两比较
            for i, s1 in enumerate(all_strategies):
                for s2 in all_strategies[i+1:]:
                    # 标准化比较名
                    comp_name = normalize_comparison_name(s1, s2)
                    
                    # 确定 loss 顺序与比较名一致
                    if comp_name.startswith(s1):
                        loss_a = np.abs(y_true - test_merged[s1].values.astype(float))
                        loss_b = np.abs(y_true - test_merged[s2].values.astype(float))
                    else:
                        loss_a = np.abs(y_true - test_merged[s2].values.astype(float))
                        loss_b = np.abs(y_true - test_merged[s1].values.astype(float))
                    
                    # DM 检验（修复版）
                    dm = dm_test_safe(loss_a, loss_b, horizon=h)
                    
                    # Wilcoxon
                    wilcox = wilcoxon_test(loss_a, loss_b)
                    
                    # Bootstrap
                    boot = block_bootstrap_single(loss_a, loss_b, current_block_size, 
                                                  n_bootstrap, seed=seeds[0])
                    
                    # 多种子 bootstrap
                    multi_boot = multi_seed_bootstrap(loss_a, loss_b, current_block_size, 
                                                      n_bootstrap, seeds)
                    
                    # Cohen's D
                    cd = cohens_d(loss_a, loss_b)
                    
                    comp_result = {
                        "dm": dm,
                        "wilcoxon": wilcox,
                        "bootstrap": boot,
                        "multi_seed_bootstrap": multi_boot,
                        "cohens_d": cd
                    }
                    
                    # 区分核心比较和额外比较
                    is_core = is_core_comparison(comp_name, best_single)
                    
                    if is_core:
                        res_h["comparisons"][comp_name] = comp_result
                        # 仅核心比较参与 Holm 校正
                        core_dm_p_values[comp_name] = dm["p_value"]
                        core_bootstrap_p_values[comp_name] = multi_boot["p_value_median"]
                    else:
                        res_h["_extra_comparisons"][comp_name] = comp_result
            
            # Holm 校正（仅核心比较）
            res_h["holm_correction_dm"] = holm_correction_original_format(core_dm_p_values)
            res_h["holm_correction_bootstrap"] = holm_correction_original_format(core_bootstrap_p_values)
            
            results[dataset][str(h)] = res_h
    
    # 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {out_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="显著性检验（修复版）")
    parser.add_argument("--base-root", type=Path, default=Path("reports/baselines"),
                        help="单模型预测根目录")
    parser.add_argument("--combo-root", type=Path, default=Path("reports/combos_fixed"),
                        help="组合模型预测根目录")
    parser.add_argument("--out", type=Path, default=Path("reports/analysis/significance.json"),
                        help="输出 JSON")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="bootstrap 轮数")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456, 789, 1024],
                        help="多种子 bootstrap 的种子列表")
    parser.add_argument("--block-size", type=int, default=None,
                        help="block bootstrap 的块大小")
    args = parser.parse_args()
    
    # 支持相对路径
    project_root = Path(__file__).parent.parent
    base_root = args.base_root if args.base_root.is_absolute() else project_root / args.base_root
    combo_root = args.combo_root if args.combo_root.is_absolute() else project_root / args.combo_root
    out_path = args.out if args.out.is_absolute() else project_root / args.out
    
    run_significance(base_root, combo_root, out_path, 
                     args.n_bootstrap, args.seeds, args.block_size)


if __name__ == "__main__":
    main()
