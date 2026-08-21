"""统计显著性检验模块 — DM test, MBB, Wilcoxon, Friedman.

Usage:
    python scripts/stats_significance.py --kg-results reports/combos_kg/kg_results.json
    python scripts/stats_significance.py --pred-root reports/baselines --strategies kg_protocol_a kg_protocol_b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _normalize_alignment_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "row_id" in out.columns:
        out["row_id"] = out["row_id"].astype(str)
    if "timestamp" in out.columns:
        out["timestamp"] = out["timestamp"].astype(str)
    return out


def _detect_alignment_key(dfs: Dict[str, pd.DataFrame]) -> Optional[str]:
    if dfs and all("row_id" in df.columns for df in dfs.values()):
        return "row_id"
    if dfs and all("timestamp" in df.columns for df in dfs.values()):
        return "timestamp"
    return None


def _align_pair_predictions(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    key: Optional[str],
    y_col: str = "y",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "alignment_key": key,
        "aligned_rows": 0,
        "y_mismatch_dropped": 0,
    }
    if key:
        left = df1[[key, y_col, "pred"]].rename(columns={y_col: "y_1", "pred": "pred_1"})
        right = df2[[key, y_col, "pred"]].rename(columns={y_col: "y_2", "pred": "pred_2"})
        merged = left.merge(right, on=key, how="inner")
        if len(merged) == 0:
            meta["skip_reason"] = "no_overlap_after_key_merge"
            return None, None, None, meta
        y1 = np.asarray(merged["y_1"].values, dtype=float)
        y2 = np.asarray(merged["y_2"].values, dtype=float)
        y_match = np.isfinite(y1) & np.isfinite(y2) & (np.abs(y1 - y2) <= 1e-6)
        meta["y_mismatch_dropped"] = int(len(merged) - int(y_match.sum()))
        if int(y_match.sum()) < 10:
            meta["skip_reason"] = "too_few_rows_after_y_consistency_filter"
            return None, None, None, meta
        y = y1[y_match]
        p1 = np.asarray(merged.loc[y_match, "pred_1"].values, dtype=float)
        p2 = np.asarray(merged.loc[y_match, "pred_2"].values, dtype=float)
    else:
        if len(df1) != len(df2):
            meta["skip_reason"] = "no_key_and_length_mismatch"
            return None, None, None, meta
        y1 = np.asarray(df1[y_col].values, dtype=float)
        y2 = np.asarray(df2[y_col].values, dtype=float)
        y_match = np.isfinite(y1) & np.isfinite(y2) & (np.abs(y1 - y2) <= 1e-6)
        meta["y_mismatch_dropped"] = int(len(df1) - int(y_match.sum()))
        if int(y_match.sum()) < 10:
            meta["skip_reason"] = "too_few_rows_after_y_consistency_filter"
            return None, None, None, meta
        y = y1[y_match]
        p1 = np.asarray(df1.loc[y_match, "pred"].values, dtype=float)
        p2 = np.asarray(df2.loc[y_match, "pred"].values, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p1) & np.isfinite(p2)
    if int(mask.sum()) < 10:
        meta["skip_reason"] = "too_few_finite_rows"
        return None, None, None, meta
    meta["aligned_rows"] = int(mask.sum())
    return y[mask], p1[mask], p2[mask], meta


def _build_aligned_panel(
    dfs: Dict[str, pd.DataFrame],
    strategy_names: List[str],
    *,
    key: Optional[str],
    y_col: str = "y",
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    meta: Dict[str, Any] = {"alignment_key": key, "y_mismatch_dropped": 0}
    if not strategy_names:
        meta["skip_reason"] = "empty_strategy_names"
        return None, meta
    base = strategy_names[0]
    base_df = dfs[base]
    if key:
        panel = base_df[[key, y_col, "pred"]].rename(columns={y_col: "y", "pred": base}).copy()
        for s in strategy_names[1:]:
            tmp = dfs[s][[key, y_col, "pred"]].rename(columns={y_col: f"y_{s}", "pred": s})
            panel = panel.merge(tmp, on=key, how="inner")
            if len(panel) == 0:
                meta["skip_reason"] = "no_overlap_after_panel_merge"
                return None, meta
            y_base = np.asarray(panel["y"].values, dtype=float)
            y_cur = np.asarray(panel[f"y_{s}"].values, dtype=float)
            y_match = np.isfinite(y_base) & np.isfinite(y_cur) & (np.abs(y_base - y_cur) <= 1e-6)
            meta["y_mismatch_dropped"] += int(len(panel) - int(y_match.sum()))
            panel = panel.loc[y_match].drop(columns=[f"y_{s}"]).copy()
            if len(panel) == 0:
                meta["skip_reason"] = "all_rows_dropped_by_y_consistency"
                return None, meta
    else:
        n = len(base_df)
        if not all(len(dfs[s]) == n for s in strategy_names):
            meta["skip_reason"] = "no_key_and_length_mismatch"
            return None, meta
        panel = pd.DataFrame({"y": np.asarray(base_df[y_col].values, dtype=float)})
        for s in strategy_names:
            y_cur = np.asarray(dfs[s][y_col].values, dtype=float)
            y_match = np.isfinite(panel["y"].values) & np.isfinite(y_cur) & (np.abs(panel["y"].values - y_cur) <= 1e-6)
            meta["y_mismatch_dropped"] += int(n - int(y_match.sum()))
            if not bool(np.all(y_match)):
                meta["skip_reason"] = "no_key_and_y_mismatch"
                return None, meta
            panel[s] = np.asarray(dfs[s]["pred"].values, dtype=float)
    finite_cols = ["y"] + strategy_names
    finite_mask = np.all(np.isfinite(panel[finite_cols].values), axis=1)
    panel = panel.loc[finite_mask].copy()
    if len(panel) < 10:
        meta["skip_reason"] = "too_few_finite_panel_rows"
        return None, meta
    meta["aligned_rows"] = int(len(panel))
    return panel, meta


# ---------------------------------------------------------------------------
# 1. Diebold-Mariano (DM) test
# ---------------------------------------------------------------------------

def dm_test(
    e1: np.ndarray,
    e2: np.ndarray,
    horizon: int = 1,
    power: int = 1,
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """Diebold-Mariano test for predictive accuracy.

    Args:
        e1, e2: Forecast error arrays (y - pred) of equal length.
        horizon: Forecast horizon h (for Newey-West bandwidth).
        power: Loss power (1 = absolute loss, 2 = squared loss).
        alternative: 'two-sided', 'less' (e1 < e2), 'greater' (e1 > e2).

    Returns:
        dict with 'dm_stat', 'p_value', 'mean_loss_diff', 'n_obs', 'alternative'.
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    assert len(e1) == len(e2), "e1 and e2 must have equal length"
    n = len(e1)

    loss1 = np.abs(e1) ** power
    loss2 = np.abs(e2) ** power
    d = loss1 - loss2

    d_mean = float(np.mean(d))
    # Newey-West HAC variance estimator
    max_lag = max(1, horizon - 1)
    gamma = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        gamma[k] = float(np.mean((d[k:] - d_mean) * (d[: n - k] - d_mean)))
    var_d = gamma[0] + 2.0 * sum(
        (1.0 - k / (max_lag + 1)) * gamma[k] for k in range(1, max_lag + 1)
    )
    var_d = max(var_d, 1e-15)
    dm_stat = float(d_mean / np.sqrt(var_d / n))

    # p-value from standard normal
    from scipy import stats as sp_stats

    if alternative == "two-sided":
        p_value = float(2.0 * sp_stats.norm.sf(abs(dm_stat)))
    elif alternative == "less":
        p_value = float(sp_stats.norm.cdf(dm_stat))
    elif alternative == "greater":
        p_value = float(sp_stats.norm.sf(dm_stat))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return {
        "dm_stat": dm_stat,
        "p_value": p_value,
        "mean_loss_diff": d_mean,
        "n_obs": int(n),
        "horizon": int(horizon),
        "power": int(power),
        "alternative": alternative,
        "significant_0.05": bool(p_value < 0.05),
        "significant_0.01": bool(p_value < 0.01),
    }


# ---------------------------------------------------------------------------
# 2. Moving Block Bootstrap (MBB)
# ---------------------------------------------------------------------------

def moving_block_bootstrap(
    e1: np.ndarray,
    e2: np.ndarray,
    n_bootstrap: int = 5000,
    block_size: Optional[int] = None,
    power: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Moving Block Bootstrap for loss differential confidence interval.

    Args:
        e1, e2: Forecast error arrays.
        n_bootstrap: Number of bootstrap replications.
        block_size: Block length for MBB. Default = ceil(n^(1/3)).
        power: Loss power (1 = MAE, 2 = MSE).
        seed: Random seed.

    Returns:
        dict with 'mean_diff', 'ci_lower', 'ci_upper', 'bootstrap_p_value'.
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    n = len(e1)
    assert n == len(e2)

    loss1 = np.abs(e1) ** power
    loss2 = np.abs(e2) ** power
    d = loss1 - loss2
    observed_mean = float(np.mean(d))

    if block_size is None:
        block_size = max(1, int(np.ceil(n ** (1.0 / 3.0))))
    block_size = min(block_size, n)
    n_blocks = int(np.ceil(n / block_size))

    rng = np.random.RandomState(seed)
    boot_means = np.zeros(n_bootstrap)
    max_start = n - block_size

    for b in range(n_bootstrap):
        starts = rng.randint(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([d[s: s + block_size] for s in starts])[:n]
        boot_means[b] = float(np.mean(sample))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    # Two-sided p-value: proportion of bootstrap means on the opposite side of 0
    if observed_mean >= 0:
        boot_p = float(np.mean(boot_means <= 0)) * 2.0
    else:
        boot_p = float(np.mean(boot_means >= 0)) * 2.0
    boot_p = min(boot_p, 1.0)

    return {
        "mean_diff": observed_mean,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "bootstrap_p_value": boot_p,
        "n_bootstrap": int(n_bootstrap),
        "block_size": int(block_size),
        "n_obs": int(n),
        "significant_0.05": bool(boot_p < 0.05),
        "zero_in_ci": bool(ci_lower <= 0 <= ci_upper),
    }


# ---------------------------------------------------------------------------
# 3. Wilcoxon signed-rank test
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank(
    mae1: np.ndarray,
    mae2: np.ndarray,
) -> Dict[str, Any]:
    """Wilcoxon signed-rank test on paired MAE differences.

    Tests H0: median(|mae1| - |mae2|) = 0.
    """
    from scipy import stats as sp_stats

    mae1 = np.asarray(mae1, dtype=float)
    mae2 = np.asarray(mae2, dtype=float)
    diff = mae1 - mae2
    # Remove zeros
    nonzero_mask = diff != 0
    if int(nonzero_mask.sum()) < 5:
        return {
            "statistic": None,
            "p_value": None,
            "n_nonzero": int(nonzero_mask.sum()),
            "significant_0.05": False,
            "skip_reason": "too_few_nonzero_differences",
        }
    try:
        stat, p_val = sp_stats.wilcoxon(diff[nonzero_mask], alternative="two-sided")
        return {
            "statistic": float(stat),
            "p_value": float(p_val),
            "n_nonzero": int(nonzero_mask.sum()),
            "median_diff": float(np.median(diff)),
            "significant_0.05": bool(p_val < 0.05),
            "significant_0.01": bool(p_val < 0.01),
        }
    except Exception as e:
        return {
            "statistic": None,
            "p_value": None,
            "error": str(e),
            "significant_0.05": False,
        }


# ---------------------------------------------------------------------------
# 4. Friedman test (multi-strategy comparison)
# ---------------------------------------------------------------------------

def friedman_test(
    mae_matrix: np.ndarray,
    strategy_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Friedman test for multi-strategy comparison.

    Args:
        mae_matrix: (n_observations, n_strategies) matrix of MAE values.
        strategy_names: Optional names for each strategy.

    Returns:
        dict with 'statistic', 'p_value', 'mean_ranks', 'n_obs', 'n_strategies'.
    """
    from scipy import stats as sp_stats

    mae_matrix = np.asarray(mae_matrix, dtype=float)
    n_obs, n_strat = mae_matrix.shape
    if n_strat < 3:
        return {
            "statistic": None,
            "p_value": None,
            "skip_reason": "need_at_least_3_strategies",
        }
    if n_obs < 5:
        return {
            "statistic": None,
            "p_value": None,
            "skip_reason": "too_few_observations",
        }

    # Compute ranks per row
    from scipy.stats import rankdata

    ranks = np.zeros_like(mae_matrix)
    for i in range(n_obs):
        ranks[i] = rankdata(mae_matrix[i])
    mean_ranks = ranks.mean(axis=0).tolist()

    try:
        stat, p_val = sp_stats.friedmanchisquare(
            *[mae_matrix[:, j] for j in range(n_strat)]
        )
    except Exception as e:
        return {
            "statistic": None,
            "p_value": None,
            "error": str(e),
        }

    result: Dict[str, Any] = {
        "statistic": float(stat),
        "p_value": float(p_val),
        "n_obs": int(n_obs),
        "n_strategies": int(n_strat),
        "mean_ranks": {
            (strategy_names[j] if strategy_names else f"strategy_{j}"): float(mean_ranks[j])
            for j in range(n_strat)
        },
        "significant_0.05": bool(p_val < 0.05),
    }
    return result


# ---------------------------------------------------------------------------
# 5. 汇总 — 从 KG 结果跑完整显著性分析
# ---------------------------------------------------------------------------

def run_significance_analysis(
    kg_results_path: Path,
    output_path: Optional[Path] = None,
    split: str = "test",
) -> Dict[str, Any]:
    """从 kg_results.json 抽取预测误差并运行全套显著性检验。"""
    with kg_results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    report: Dict[str, Any] = {"tests": {}}

    for ds, ds_data in results.items():
        if ds.startswith("_") or not isinstance(ds_data, dict):
            continue
        report["tests"][ds] = {}
        for h_key, h_data in ds_data.items():
            if not isinstance(h_data, dict):
                continue
            horizon = int(h_key)
            # Extract Protocol A and B MAEs
            a_payload = h_data.get("kg_protocol_a", h_data.get("protocol_A", {}))
            b_payload = h_data.get("kg_protocol_b", h_data.get("protocol_B", {}))
            if not isinstance(a_payload, dict) or not isinstance(b_payload, dict):
                continue

            split_a = a_payload.get(split, {})
            split_b = b_payload.get(split, {})
            if not isinstance(split_a, dict) or not isinstance(split_b, dict):
                continue

            mae_a = split_a.get("mae")
            mae_b = split_b.get("mae")
            if mae_a is None or mae_b is None:
                continue

            meta = h_data.get("_meta", {})
            safe_models = meta.get("safe_models", [])

            task_report: Dict[str, Any] = {
                "mae_a": float(mae_a),
                "mae_b": float(mae_b),
                "winner": "B" if float(mae_b) < float(mae_a) else ("A" if float(mae_a) < float(mae_b) else "tie"),
                "rel_improve_b_vs_a": (float(mae_a) - float(mae_b)) / max(float(mae_a), 1e-10),
                "n_safe_models": len(safe_models),
            }

            # Note: DM/MBB/Wilcoxon need per-sample errors, not aggregated MAE.
            # When per-sample predictions are available (val/test split CSVs),
            # these tests can be computed directly. For now, we record the
            # aggregated metrics and mark per-sample tests as requiring raw data.
            task_report["per_sample_tests_available"] = False
            task_report["note"] = (
                "Per-sample DM/MBB/Wilcoxon tests require val/test prediction arrays. "
                "Use --pred-root to load raw predictions for full statistical analysis."
            )

            report["tests"][ds][h_key] = task_report

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        print(f"显著性分析已保存: {output_path}")

    return report


def run_per_sample_significance(
    pred_root: Path,
    strategies: List[str],
    dataset: str,
    horizon: int,
    split: str = "test",
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """从预测文件加载 per-sample errors 并运行 DM/MBB/Wilcoxon 检验。"""
    y_col = "y"
    task_name = f"{dataset}_h{horizon}_{split}"

    # Load predictions
    dfs: Dict[str, pd.DataFrame] = {}
    for s in strategies:
        pred_path = pred_root / dataset / f"{split}_pred_h{horizon}_{s}.csv"
        if not pred_path.exists():
            print(f"  跳过 {s}: 预测文件不存在 {pred_path}")
            continue
        try:
            df = _normalize_alignment_columns(pd.read_csv(pred_path))
            if "pred" in df.columns and y_col in df.columns:
                dfs[s] = df
        except Exception as e:
            print(f"  加载失败 {s}: {e}")

    if len(dfs) < 2:
        return {"task": task_name, "skip_reason": f"需要至少2个策略，仅找到 {len(dfs)}"}

    align_key = _detect_alignment_key(dfs)
    report: Dict[str, Any] = {"task": task_name, "pairwise": {}, "friedman": None, "alignment_key": align_key}

    # Pairwise DM + MBB + Wilcoxon
    strat_names = sorted(dfs.keys())
    for i, s1 in enumerate(strat_names):
        for s2 in strat_names[i + 1 :]:
            df1 = dfs[s1]
            df2 = dfs[s2]
            y, p1, p2, align_meta = _align_pair_predictions(df1, df2, key=align_key, y_col=y_col)
            if y is None or p1 is None or p2 is None:
                continue
            e1_m = y - p1
            e2_m = y - p2
            pair_key = f"{s1}_vs_{s2}"
            report["pairwise"][pair_key] = {
                "align_meta": align_meta,
                "dm": dm_test(e1_m, e2_m, horizon=horizon, power=1),
                "mbb": moving_block_bootstrap(e1_m, e2_m, power=1),
                "wilcoxon": wilcoxon_signed_rank(np.abs(e1_m), np.abs(e2_m)),
            }

    # Friedman test (if >=3 strategies with common rows)
    if len(strat_names) >= 3:
        panel, panel_meta = _build_aligned_panel(
            dfs=dfs,
            strategy_names=strat_names,
            key=align_key,
            y_col=y_col,
        )
        if panel is not None:
            y = np.asarray(panel["y"].values, dtype=float)
            mae_cols = []
            for s in strat_names:
                ae = np.abs(y - np.asarray(panel[s].values, dtype=float))
                mae_cols.append(ae)
            mae_matrix = np.column_stack(mae_cols)
            report["friedman"] = friedman_test(
                mae_matrix, strategy_names=strat_names
            )
            report["friedman_align_meta"] = panel_meta

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        print(f"per-sample 显著性分析已保存: {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="统计显著性检验模块")
    parser.add_argument(
        "--kg-results", type=Path, default=None,
        help="KG 结果 JSON 路径（聚合级分析）",
    )
    parser.add_argument(
        "--pred-root", type=Path, default=None,
        help="预测文件根目录（per-sample 分析）",
    )
    parser.add_argument(
        "--strategies", nargs="*", default=None,
        help="策略名（per-sample 模式下需指定）",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="数据集名",
    )
    parser.add_argument(
        "--horizons", nargs="*", type=int, default=None,
        help="预测步长",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="使用 val 还是 test 集",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("reports/significance"),
        help="输出路径",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    if args.kg_results:
        kg_path = args.kg_results if args.kg_results.is_absolute() else project_root / args.kg_results
        out_path = args.out if args.out.is_absolute() else project_root / args.out
        run_significance_analysis(
            kg_path,
            output_path=out_path / "significance_aggregated.json",
            split=args.split,
        )

    if args.pred_root and args.strategies and args.datasets and args.horizons:
        pred_root = args.pred_root if args.pred_root.is_absolute() else project_root / args.pred_root
        out_path = args.out if args.out.is_absolute() else project_root / args.out
        for ds in args.datasets:
            for h in args.horizons:
                print(f"\n分析 {ds} h={h} ...")
                run_per_sample_significance(
                    pred_root=pred_root,
                    strategies=args.strategies,
                    dataset=ds,
                    horizon=h,
                    split=args.split,
                    output_path=out_path / f"significance_{ds}_h{h}.json",
                )

    if not args.kg_results and not (args.pred_root and args.strategies):
        parser.print_help()
        print("\n请指定 --kg-results 或 --pred-root + --strategies")


if __name__ == "__main__":
    main()
