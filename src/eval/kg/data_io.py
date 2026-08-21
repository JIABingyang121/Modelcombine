"""KG data loading and alignment helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.blocked_cv import blocked_cv_splits, resolve_blocked_cv_config

def _resolve_cv_config(n_samples: int, horizon: int) -> Tuple[int, int, int]:
    """返回 (n_folds, min_train, gap)。统一复用 blocked_cv 的口径。"""
    return resolve_blocked_cv_config(n_samples=n_samples, horizon=horizon)


def _valid_pair_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, int]:
    """NaN 安全的 MAE 计算，仅使用双端有限值的样本对。

    Returns:
        (mae, valid_ratio, valid_count)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n_valid = int(mask.sum())
    n_total = len(y_true)
    valid_ratio = float(n_valid / n_total) if n_total > 0 else 0.0
    if n_valid <= 0:
        return float("inf"), valid_ratio, n_valid
    mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
    if not np.isfinite(mae):
        return float("inf"), valid_ratio, n_valid
    return mae, valid_ratio, n_valid

def _blocked_cv_mae_from_pred(y: np.ndarray, pred: np.ndarray, horizon: int) -> Optional[float]:
    n_folds, min_train, gap = _resolve_cv_config(len(y), horizon)
    splits = blocked_cv_splits(len(y), n_folds=n_folds, min_train=min_train, gap=gap)
    if not splits:
        return None
    scores = []
    for _, val_idx in splits:
        if len(val_idx) == 0:
            continue
        scores.append(float(np.mean(np.abs(y[val_idx] - pred[val_idx]))))
    return float(np.mean(scores)) if scores else None

def _ensure_row_id(df: pd.DataFrame, ts_col: str = "timestamp", strict: bool = False) -> pd.DataFrame:
    out = df.copy()
    if "row_id" in out.columns:
        out["row_id"] = out["row_id"].astype(str)
    if "row_id" in out.columns and not out["row_id"].duplicated().any():
        return out
    if ts_col not in out.columns:
        if strict:
            raise ValueError(f"missing_{ts_col}_for_row_id")
        out = out.reset_index(drop=True)
        out["row_id"] = [f"row_{i}" for i in range(len(out))]
        return out
    try:
        ts = pd.to_datetime(out[ts_col], errors="coerce")
        tmp = pd.DataFrame({"_ts": ts, "_orig_idx": np.arange(len(out))})
        tmp = tmp.sort_values(["_ts", "_orig_idx"]).reset_index(drop=True)
        tmp["_k"] = tmp.groupby("_ts", dropna=False).cumcount().astype(str)
        stable = tmp["_ts"].astype(str) + "_" + tmp["_k"]
        aligned = out.iloc[tmp["_orig_idx"].values].reset_index(drop=True).copy()
        aligned["row_id"] = stable.astype(str).values
        return aligned
    except Exception:
        if strict:
            raise
        out = out.reset_index(drop=True)
        out["row_id"] = [f"row_{i}" for i in range(len(out))]
        return out

def _align_raw_to_pred(df_raw: pd.DataFrame, df_pred: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    将原始特征按 row_id 或稳定键对齐到预测 DataFrame。
    """
    raw = _ensure_row_id(df_raw.copy(), ts_col=ts_col)
    pred = _ensure_row_id(df_pred.copy(), ts_col=ts_col)
    raw["row_id"] = raw["row_id"].astype(str)
    pred["row_id"] = pred["row_id"].astype(str)
    merged = pred[["row_id"]].merge(raw, on="row_id", how="left")
    return merged

def _load_extended_predictions_file(path: Path, strategy_name: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, dtype={"row_id": "string"})
    except Exception:
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
    if "pred" not in df.columns:
        return None
    if "timestamp" not in df.columns:
        return None
    keep_cols = [c for c in ["timestamp", "y", "row_id", "pred"] if c in df.columns]
    out = df[keep_cols].copy()
    if "row_id" in out.columns:
        out["row_id"] = out["row_id"].astype(str)
    out = out.rename(columns={"pred": strategy_name})
    return out

def _merge_extended_predictions(
    base_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    strategy_name: str,
) -> pd.DataFrame:
    n_before = len(base_df)
    merged = _ensure_row_id(base_df.copy(), ts_col="timestamp")
    ext = _ensure_row_id(ext_df.copy(), ts_col="timestamp")
    merged["row_id"] = merged["row_id"].astype(str)
    ext["row_id"] = ext["row_id"].astype(str)
    merged = merged.merge(ext[["row_id", strategy_name]], on="row_id", how="inner")
    if len(merged) != n_before:
        raise ValueError(f"row_count_changed:{n_before}->{len(merged)}")
    return merged

def _load_extended_pool_for_split(
    split_df: pd.DataFrame,
    *,
    split: str,
    dataset: str,
    horizon: int,
    pred_root: Path,
    combo_root: Optional[Path],
    strategies: List[str],
    allow_in_sample: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    从 pred_root 的标准预测文件或 combo_root 的组合表加载扩展候选。
    """
    merged = split_df.copy()
    loaded = []
    skipped = {}
    strict_block = {
        "gating_network",
        "soft_gating",
        "scenario_bucket",
        "gating_network_v2",
        "adaptive_bucket",
        "scenario_similarity",
    }

    combo_table = None
    if combo_root is not None:
        combo_path = combo_root / dataset / f"{split}_combos_h{horizon}.csv"
        if combo_path.exists():
            try:
                combo_table = pd.read_csv(combo_path, dtype={"row_id": "string"})
            except Exception:
                try:
                    combo_table = pd.read_csv(combo_path)
                except Exception:
                    combo_table = None

    for s in strategies:
        val_mode = "unknown"
        meta_path = pred_root / dataset / f"val_pred_h{horizon}_{s}.meta.json"
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta_obj = json.load(f)
                val_mode = str(meta_obj.get("val_eval_mode", "unknown"))
            except Exception:
                val_mode = "unknown"
        if split == "val" and not allow_in_sample:
            safe_modes = {"oof", "blocked_cv", "blocked_cv_blend", "online_rolling", "deterministic"}
            if val_mode not in safe_modes:
                blocked_reason = "strict_mode_block_in_sample_or_unknown_val_mode"
                if s in strict_block:
                    blocked_reason = "strict_mode_block_in_sample_val_prediction"
                skipped[s] = f"{blocked_reason}:{val_mode}"
                continue

        ext_df = _load_extended_predictions_file(
            pred_root / dataset / f"{split}_pred_h{horizon}_{s}.csv",
            s,
        )

        if ext_df is None and combo_table is not None and s in combo_table.columns:
            cols = ["timestamp", "y", s]
            if "row_id" in combo_table.columns:
                cols.insert(0, "row_id")
            ext_df = combo_table[cols].copy()

        if ext_df is None:
            skipped[s] = "missing_prediction_file"
            continue

        try:
            merged = _merge_extended_predictions(merged, ext_df, s)
            loaded.append(s)
        except Exception as e:
            skipped[s] = f"merge_failed:{e}"

    return merged, {
        "loaded": loaded,
        "skipped": skipped,
        "strict_allow_in_sample": bool(allow_in_sample),
    }

def load_health_enabled_map(health_config_path: Optional[Path]) -> Dict[str, Dict[str, List[str]]]:
    """从 routing_config.json 读取模型健康白名单（enabled）。"""
    if health_config_path is None:
        return {}
    if not health_config_path.exists():
        print(f"警告: 健康配置文件不存在，跳过白名单过滤: {health_config_path}")
        return {}
    try:
        with health_config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        summary = cfg.get("model_health_summary", {})
        enabled_map: Dict[str, Dict[str, List[str]]] = {}
        for ds, ds_info in summary.items():
            if not isinstance(ds_info, dict):
                continue
            enabled_map[ds] = {}
            for h, h_info in ds_info.items():
                if not isinstance(h_info, dict):
                    continue
                enabled = h_info.get("enabled", [])
                if isinstance(enabled, list):
                    enabled_map[ds][str(h)] = [m for m in enabled if isinstance(m, str)]
        return enabled_map
    except Exception as e:
        print(f"警告: 解析健康配置失败，跳过白名单过滤: {e}")
        return {}

def load_candidate_audit_map(audit_path: Optional[Path]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """加载离线候选审计结果：dataset -> horizon(str) -> task_audit。"""
    if audit_path is None:
        return {}
    if not audit_path.exists():
        print(f"警告: candidate audit 文件不存在，跳过审计过滤: {audit_path}")
        return {}
    try:
        with audit_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        tasks = payload.get("tasks", {})
        if not isinstance(tasks, dict):
            return {}
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for ds, ds_data in tasks.items():
            if not isinstance(ds_data, dict):
                continue
            out[ds] = {}
            for h, task_data in ds_data.items():
                if isinstance(task_data, dict):
                    out[ds][str(h)] = task_data
        return out
    except Exception as e:
        print(f"警告: candidate audit 解析失败，跳过审计过滤: {e}")
        return {}

