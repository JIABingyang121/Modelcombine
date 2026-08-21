"""KG graph conflict checks and correlation utilities."""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from src.eval.combination_utils import generate_stable_key
from src.graph.model_graph import ModelGraph

def check_conflict(mg: ModelGraph, models: List[str]) -> List[Tuple[str, str]]:
    """检查模型对之间的冲突（双向）"""
    conflicts = []
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            # 正向
            if mg.G.has_edge(m1, m2) and mg.G[m1][m2].get("edge_type") == "conflict":
                conflicts.append((m1, m2))
                continue
            # 反向
            if mg.G.has_edge(m2, m1) and mg.G[m2][m1].get("edge_type") == "conflict":
                conflicts.append((m1, m2))
    return conflicts

def filter_conflicting_models(
    mg: ModelGraph,
    models: List[str],
    maes: Optional[Dict[str, float]] = None,
) -> List[str]:
    """移除冲突的模型"""
    conflicts = check_conflict(mg, models)
    if not conflicts:
        return models

    excluded = set()
    for m1, m2 in conflicts:
        if m1 in excluded or m2 in excluded:
            continue
        drop = m2
        if maes is not None:
            mae1 = float(maes.get(m1, float("inf")))
            mae2 = float(maes.get(m2, float("inf")))
            if np.isfinite(mae1) and np.isfinite(mae2):
                # 显式按验证集 MAE 保留更优者
                drop = m1 if mae1 > mae2 else m2
        excluded.add(drop)

    return [m for m in models if m not in excluded]

def compute_error_correlations(df: pd.DataFrame, model_cols: List[str]) -> Dict[Tuple[str, str], float]:
    """计算模型对之间的误差相关性"""
    y = pd.to_numeric(df["y"], errors="coerce").values
    errors = {}
    for m in model_cols:
        pred = pd.to_numeric(df[m], errors="coerce").values
        errors[m] = pred - y

    correlations = {}
    for i, m1 in enumerate(model_cols):
        for m2 in model_cols[i+1:]:
            e1 = errors[m1]
            e2 = errors[m2]
            valid = np.isfinite(e1) & np.isfinite(e2)
            if int(np.sum(valid)) < 20:
                corr = 1.0
            else:
                e1v = e1[valid]
                e2v = e2[valid]
                if np.std(e1v) < 1e-10 or np.std(e2v) < 1e-10:
                    corr = 1.0
                else:
                    corr = float(np.corrcoef(e1v, e2v)[0, 1])
                    if not np.isfinite(corr):
                        corr = 1.0
            correlations[(m1, m2)] = corr

    return correlations

def compute_feature_model_correlation_safe(df_raw: pd.DataFrame, 
                                            df_pred: pd.DataFrame,
                                            model_cols: List[str],
                                            feature_cols: List[str],
                                            ts_col: str = "timestamp") -> Dict[Tuple[str, str], float]:
    """安全地计算特征-误差相关性"""
    # 对齐
    if "row_id" in df_raw.columns and "row_id" in df_pred.columns:
        df_raw = df_raw.copy()
        df_pred = df_pred.copy()
        df_raw["row_id"] = df_raw["row_id"].astype(str)
        df_pred["row_id"] = df_pred["row_id"].astype(str)
        merged = df_raw.merge(df_pred, on="row_id", how="inner", suffixes=("_raw", "_pred"))
    else:
        stable_key_raw, df_raw = generate_stable_key(df_raw, ts_col)
        df_raw["_stable_key"] = stable_key_raw.astype(str)
        stable_key_pred, df_pred = generate_stable_key(df_pred, ts_col)
        df_pred["_stable_key"] = stable_key_pred.astype(str)
        merged = df_raw.merge(df_pred, on="_stable_key", how="inner", suffixes=("_raw", "_pred"))
    
    if len(merged) == 0:
        warnings.warn("特征与预测对齐后无样本")
        return {}
    
    y_col = "y_pred" if "y_pred" in merged.columns else "y"
    y = pd.to_numeric(merged[y_col], errors="coerce").values
    
    correlations = {}
    for feat in feature_cols:
        if feat not in merged.columns:
            continue
        
        feat_values = pd.to_numeric(merged[feat], errors="coerce").values
        feat_values = feat_values[~np.isnan(feat_values)]
        if len(feat_values) == 0:
            continue
        if np.std(feat_values) < 1e-10:
            continue
        
        for m in model_cols:
            m_col = f"{m}_pred" if f"{m}_pred" in merged.columns else m
            if m_col not in merged.columns:
                continue
            
            error = (pd.to_numeric(merged[m_col], errors="coerce").values - y)
            feat_raw = pd.to_numeric(merged[feat], errors="coerce").values
            valid = (~np.isnan(error)) & (~np.isnan(feat_raw))
            if valid.sum() == 0:
                continue
            feat_vals_aligned = feat_raw[valid]
            error = error[valid]
            if len(feat_vals_aligned) == 0:
                continue
            if np.std(feat_vals_aligned) < 1e-10:
                continue
            if np.std(error) < 1e-10:
                continue
            
            corr = np.corrcoef(feat_vals_aligned, error)[0, 1]
            if not np.isnan(corr):
                correlations[(feat, m)] = abs(corr)
    
    return correlations

def _lookup_pair_corr(
    corr_map: Dict[Tuple[str, str], float],
    m1: str,
    m2: str,
    default: float = 0.0,
) -> float:
    if (m1, m2) in corr_map:
        return float(corr_map[(m1, m2)])
    if (m2, m1) in corr_map:
        return float(corr_map[(m2, m1)])
    return default

