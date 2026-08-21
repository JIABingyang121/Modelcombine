import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def mase_scale(train_target: pd.Series, horizon: int) -> float:
    arr = train_target.astype(float).values
    if len(arr) <= horizon:
        return np.nan
    diffs = np.abs(arr[horizon:] - arr[:-horizon])
    scale = np.mean(diffs)
    return float(scale) if scale > 0 else np.nan


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> Dict[str, float]:
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "MASE": np.nan}
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))))
    mase = float(mae / scale) if scale and not np.isnan(scale) else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "MASE": mase}


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int) -> Dict[str, float]:
    d = loss_a - loss_b
    d = d - np.mean(d)
    T = len(d)
    if T == 0:
        return {"stat": np.nan, "p_value": np.nan}
    max_lag = min(horizon - 1, T - 1)
    gamma0 = np.mean(d * d)
    cov_sum = gamma0
    for k in range(1, max_lag + 1):
        cov = np.mean(d[k:] * d[:-k])
        cov_sum += 2 * cov
    denom = cov_sum / T
    if denom <= 0:
        return {"stat": np.nan, "p_value": np.nan}
    stat = np.mean(d) / np.sqrt(denom)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return {"stat": float(stat), "p_value": float(p)}


def wilcoxon_test(loss_a: np.ndarray, loss_b: np.ndarray) -> Dict[str, float]:
    try:
        res = stats.wilcoxon(loss_a, loss_b, zero_method="wilcox")
        return {"stat": float(res.statistic), "p_value": float(res.pvalue)}
    except ValueError:
        return {"stat": np.nan, "p_value": np.nan}


def build_slices(df: pd.DataFrame) -> Dict[str, pd.Series]:
    ts = pd.to_datetime(df["timestamp"])
    weekday = ts.dt.weekday < 5
    weekend = ~weekday
    high_load_cut = df["y"].quantile(0.8)
    high_load = df["y"] >= high_load_cut
    mid_low_load = df["y"] < high_load_cut
    ordered = df.sort_values("timestamp").reset_index()
    ordered["vol"] = ordered["y"].diff().abs()
    vol_cut = ordered["vol"].quantile(0.8)
    high_vol_idx = ordered.loc[ordered["vol"] >= vol_cut, "index"]
    low_vol_idx = ordered.loc[ordered["vol"] < vol_cut, "index"]
    high_vol = df.index.isin(high_vol_idx)
    low_vol = df.index.isin(low_vol_idx)
    return {
        "weekday": weekday,
        "weekend": weekend,
        "high_load": high_load,
        "mid_low_load": mid_low_load,
        "high_vol": high_vol,
        "low_vol": low_vol,
        "all": pd.Series(True, index=df.index),
    }


def eval_models(df: pd.DataFrame, model_cols: List[str], scale: float, mask: pd.Series) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    y = df.loc[mask, "y"].values.astype(float)
    for col in model_cols:
        preds = df.loc[mask, col].values.astype(float)
        out[col] = compute_metrics(y, preds, scale)
    return out


def select_best_single(val_df: pd.DataFrame, base_cols: List[str]) -> str:
    best = None
    best_mae = float("inf")
    y = val_df["y"].values.astype(float)
    exclude = {"seasonal_naive"}
    for col in base_cols:
        if col in exclude:
            continue
        mae = float(np.mean(np.abs(y - val_df[col].values.astype(float))))
        if mae < best_mae:
            best_mae = mae
            best = col
    return best


def load_train_target(features_root: Path, dataset: str, target_col: str) -> pd.Series:
    train_path = features_root / dataset / "train.csv"
    df = pd.read_csv(train_path, usecols=[target_col])
    return df[target_col]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features", help="特征根目录")
    parser.add_argument("--pred-root", default="reports/modelcombine", help="预测结果目录")
    parser.add_argument("--out", default="reports/analysis/slices_stats.json", help="输出 JSON 文件")
    parser.add_argument("--target-model", default="stacking", help="与最佳单模型对比的目标模型")
    args = parser.parse_args()

    features_root = Path(args.features)
    pred_root = Path(args.pred_root)
    out_path = Path(args.out)
    ensure_parent(out_path)

    horizons = {
        "pjm": [1, 6, 24],
        "aemo_vic": [1, 6, 24],
        "aemo_nsw": [1, 6, 24],
    }
    targets = {
        "pjm": "load",
        "aemo_vic": "load",
        "aemo_nsw": "load",
    }

    results: Dict[str, Dict] = {}

    for dataset, hs in horizons.items():
        ds_root = pred_root / dataset
        if not ds_root.exists():
            print(f"skip {dataset}, predictions not found")
            continue

        target_col = targets[dataset]
        train_target = load_train_target(features_root, dataset, target_col)

        results[dataset] = {}
        for h in hs:
            val_path = ds_root / f"val_base_h{h}.csv"
            test_base_path = ds_root / f"test_base_h{h}.csv"
            test_combo_path = ds_root / f"test_combos_h{h}.csv"
            if not (val_path.exists() and test_base_path.exists() and test_combo_path.exists()):
                print(f"missing files for {dataset} h={h}")
                continue

            val_df = pd.read_csv(val_path)
            test_base = pd.read_csv(test_base_path)
            test_combos = pd.read_csv(test_combo_path)

            base_cols = [c for c in val_df.columns if c not in ("y", "timestamp")]
            combo_cols = [c for c in test_combos.columns if c not in ("y", "timestamp")]
            for c in combo_cols:
                test_base[c] = test_combos[c]

            best_single = select_best_single(val_df, base_cols)
            model_cols = base_cols + combo_cols

            scale = mase_scale(train_target, h)
            slices = build_slices(test_base)

            slice_metrics = {}
            for slice_name, mask in slices.items():
                slice_metrics[slice_name] = eval_models(test_base, model_cols, scale, mask)

            if args.target_model in model_cols and best_single in model_cols:
                y = test_base["y"].values.astype(float)
                loss_best = np.abs(y - test_base[best_single].values.astype(float))
                loss_target = np.abs(y - test_base[args.target_model].values.astype(float))
                sig = {
                    "target_model": args.target_model,
                    "best_single": best_single,
                    "dm": dm_test(loss_target, loss_best, h),
                    "wilcoxon": wilcoxon_test(loss_target, loss_best),
                }
            else:
                sig = {
                    "target_model": args.target_model,
                    "best_single": best_single,
                    "dm": {"stat": np.nan, "p_value": np.nan},
                    "wilcoxon": {"stat": np.nan, "p_value": np.nan},
                }

            results[dataset][h] = {
                "best_single": best_single,
                "metrics": slice_metrics,
                "significance": sig,
            }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"saved slice metrics to {out_path}")


if __name__ == "__main__":
    main()
