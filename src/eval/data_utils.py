import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# 向后兼容导出：策略超参数迁移到 strategy_config
from src.eval.strategy_config import (
    SOFTGATING_ALPHA_CANDIDATES,
    KG_COMPONENT_MODEL_POOL_MODE,
    DIRECT_WEIGHT_GATING_TEMPERATURE,
    SCENARIO_SIMILARITY_WEIGHT_MODE,
    SCENARIO_SIMILARITY_TEMPERATURE,
    DRIFT_DECAY_BASE,
    DRIFT_DECAY_SLOPE,
    DRIFT_DECAY_MAX,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 从 pipeline.yaml 读取数据集配置（如 min_samples_for_dynamic / lgbm_overrides）
_PIPELINE_YAML = PROJECT_ROOT / "configs" / "pipeline.yaml"


def _load_dataset_configs() -> tuple:
    """从 pipeline.yaml 读取各数据集 LGBM 覆盖配置和动态策略最小样本阈值。"""
    _default_min_samples = {
        "pjm": 500, "aemo_vic": 500,
        "aemo_nsw": 500,
    }
    try:
        import yaml
        with open(_PIPELINE_YAML, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        datasets = cfg.get("datasets", {})
        lgbm_overrides_map = {}
        for ds_name, ds_cfg in datasets.items():
            if isinstance(ds_cfg, dict) and isinstance(ds_cfg.get("lgbm_overrides"), dict):
                lgbm_overrides_map[ds_name] = ds_cfg["lgbm_overrides"]
        min_samples = {}
        for ds_name, ds_cfg in datasets.items():
            if isinstance(ds_cfg, dict) and "min_samples_for_dynamic" in ds_cfg:
                min_samples[ds_name] = ds_cfg["min_samples_for_dynamic"]
            else:
                min_samples[ds_name] = _default_min_samples.get(ds_name, 500)
        for k, v in _default_min_samples.items():
            min_samples.setdefault(k, v)
        return lgbm_overrides_map, min_samples
    except Exception:
        return {}, _default_min_samples


DATASET_LGBM_OVERRIDES, MIN_SAMPLES_FOR_DYNAMIC = _load_dataset_configs()

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_split(feature_root: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(feature_root / f"{split}.csv")


def _debug_print_split(path: Path, df: pd.DataFrame) -> None:
    """用于核对数据规模与时间范围（仅打印三行关键信息）"""
    print(f"[debug] path={path}")
    print(f"[debug] shape={df.shape}")
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.notna().any():
            print(f"[debug] time_range={ts.min()} -> {ts.max()}")
        else:
            print("[debug] time_range=NA(timestamp parse failed)")
    else:
        print("[debug] time_range=NA(no timestamp)")


def _build_row_id_from_timestamp(ts: pd.Series) -> pd.Series:
    """
    生成稳定 row_id：timestamp + 同时刻序号。
    """
    ts_dt = pd.to_datetime(ts, errors="coerce")
    tmp = pd.DataFrame({"_ts_dt": ts_dt, "_orig_idx": np.arange(len(ts_dt), dtype=int)})
    tmp = tmp.sort_values(["_ts_dt", "_orig_idx"]).reset_index(drop=True)
    tmp["row_id"] = (
        tmp["_ts_dt"].astype(str)
        + "_"
        + tmp.groupby("_ts_dt").cumcount().astype(str)
    )
    tmp = tmp.sort_values("_orig_idx").reset_index(drop=True)
    return tmp["row_id"]


def prepare_supervised(df: pd.DataFrame, target_col: str, horizon: int, time_col: str = "timestamp") -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """返回 X, y, ts, 以及用于场景分析的 context_features"""
    df = df.copy()
    time_candidates = [time_col, 'timestamp', 'ts', 'datetime', 'date', 'time']
    actual_time_col = None
    for tc in time_candidates:
        if tc in df.columns:
            actual_time_col = tc
            break

    if actual_time_col is None:
        raise ValueError(f"数据缺少时间列，尝试了: {time_candidates}。请确保数据包含时间信息。")

    df[actual_time_col] = pd.to_datetime(df[actual_time_col], errors='coerce')

    import warnings
    nat_count = df[actual_time_col].isna().sum()
    if nat_count > 0:
        warnings.warn(
            f"[prepare_supervised] 时间列 '{actual_time_col}' 包含 {nat_count} 个无效时间值 (NaT)，这些行将被丢弃。"
            f"请检查数据质量。",
            UserWarning
        )
        df = df.dropna(subset=[actual_time_col])

    df = df.sort_values(actual_time_col)
    df[target_col] = df[target_col].astype(float)
    df["target_h"] = df[target_col].shift(-horizon)
    df["ts_target"] = df[actual_time_col].shift(-horizon)
    df = df.dropna(subset=["target_h", "ts_target"])

    exclude = {"timestamp", actual_time_col, target_col, "target_h", "ts_target"}
    X = df.drop(columns=[c for c in exclude if c in df.columns]).select_dtypes(include=[np.number])
    y = df["target_h"]
    ts = pd.to_datetime(df["ts_target"])

    if len(ts) == 0:
        raise ValueError(
            f"prepare_supervised 返回空序列。可能原因：\n"
            f"  1. 时间列 '{actual_time_col}' 全部为无效值 (NaT)\n"
            f"  2. horizon={horizon} 过大，导致所有样本被 shift 后丢弃\n"
            f"  请检查数据质量和 horizon 参数。"
        )

    context_cols = ["hour", "dayofweek", "month", "is_weekend", "is_holiday"]
    for c in df.columns:
        if c.startswith("lag_") or c.startswith("roll"):
            context_cols.append(c)
    context_cols = [c for c in context_cols if c in df.columns]
    context_features = df[context_cols].copy() if context_cols else pd.DataFrame(index=df.index)

    return X, y, ts, context_features


def make_aligned_stack(preds: Dict[str, np.ndarray], y: pd.Series, ts: pd.Series,
                       model_cols: List[str], name: str, split: str,
                       context_features: pd.DataFrame = None) -> pd.DataFrame:
    """Align prediction arrays with targets by timestamp; guard length mismatches and duplicates."""
    n_y = len(y)
    bad = {m: len(preds[m]) for m in model_cols if len(preds[m]) != n_y}
    if bad:
        raise ValueError(f"{name} {split} length mismatch: y={n_y}, preds={bad}")

    df = pd.DataFrame({m: preds[m] for m in model_cols})
    df["y"] = y.values
    df["timestamp"] = ts.values

    if context_features is not None and len(context_features) == n_y:
        for c in context_features.columns:
            df[f"ctx_{c}"] = context_features[c].values

    # 保留重复时间戳，使用 row_id 做严格对齐键，避免 DST/重复时刻被静默删行。
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["row_id"] = _build_row_id_from_timestamp(df["timestamp"])
    return df
