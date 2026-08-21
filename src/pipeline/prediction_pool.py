"""无泄漏候选预测矩阵（System A/B 合一 Task 2）。

System A 原有数据流是"先选模型、再预测"（`PowerModelCombinator` 在没有任何
预测的情况下选路径），而 Protocol B 要求"先有候选预测矩阵、再做组合"。本模块
提供两者之间缺失的那一层：把单区域训练集按时间切成 fit/validation，产出
Protocol B 可直接消费的 `df_val` / `df_test` / `df_raw_val` / `df_raw_test`。

泄漏防线（本模块存在的主要理由）：

- validation 取自 **训练集时间尾部**，不碰 test；
- 第一轮只用 fit 段训练，用于产出 validation 预测；
- 第二轮在 **完整 train**（fit+validation）上重训，用于产出 test 预测；
- 任何一轮都不会把 test 标签喂进 `fit`。

本模块只负责生成矩阵，不做任何组合决策，也不调用旧 `PowerModelCombinator`。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..features.build_features import build_matrix

# build_matrix 的目标列；raw frame 需要额外保留 timestamp 供 Protocol B 对齐。
_TARGET_COL = "load"


def split_fit_validation(
    region_train: pd.DataFrame,
    validation_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """把单区域训练集按时间切成 (fit, validation)。

    validation 取训练集最后 `validation_days` 天；fit 是其之前的全部数据。
    切分严格按时间，不打乱、不重叠——validation 用于评估候选在"未来"的表现，
    随机切分会让滞后特征跨越切点造成泄漏。

    Raises:
        ValueError: 输入为空、`validation_days` 非正，或历史长度不足以在
            切分后同时留下非空的 fit 与 validation。
    """
    if validation_days <= 0:
        raise ValueError(
            f"split_fit_validation: validation_days must be positive, got {validation_days}"
        )
    if region_train is None or len(region_train) == 0:
        raise ValueError("split_fit_validation: region_train is empty")

    frame = region_train.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp")

    last_ts = frame["timestamp"].max()
    cutoff = last_ts - pd.Timedelta(days=validation_days)
    fit_df = frame[frame["timestamp"] <= cutoff]
    val_df = frame[frame["timestamp"] > cutoff]

    if len(val_df) == 0:
        raise ValueError(
            "split_fit_validation: validation window is empty; "
            f"validation_days={validation_days} exceeds available history"
        )
    if len(fit_df) == 0:
        raise ValueError(
            "split_fit_validation: fit window is empty; history is shorter than "
            f"validation_days={validation_days}"
        )
    return fit_df, val_df


@dataclass
class RegionPredictionBundle:
    """单区域候选预测矩阵，字段与 Protocol B 上下文构造器一一对应。"""

    df_val: pd.DataFrame
    df_test: pd.DataFrame
    df_raw_val: pd.DataFrame
    df_raw_test: pd.DataFrame
    model_cols: List[str]
    base_model_cols: List[str]
    fitted_test_models: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _raw_frame(features: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    """原始特征表：数值特征 + timestamp（Protocol B 的对齐键）。"""
    raw = features.reset_index(drop=True).copy()
    raw.insert(0, "timestamp", pd.to_datetime(timestamps).reset_index(drop=True))
    return raw


def build_region_prediction_bundle(
    *,
    region: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate_models: Sequence[str],
    validation_days: int,
    model_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    registry: Any = None,
    base_model_cols: Optional[Sequence[str]] = None,
) -> RegionPredictionBundle:
    """为单个区域生成无泄漏的候选预测矩阵。

    Args:
        region: 区域名；train/test 会按该值过滤（若含 `region` 列）。
        train: 该区域可见的全部训练数据（validation 从其尾部切出）。
        test: 测试数据；其标签只用于最终评估，绝不进入任何 `fit`。
        candidate_models: 候选模型 id 列表。
        validation_days: 训练集尾部划作 validation 的天数。
        model_params: 每个模型的构造参数，缺省为空。
        registry: 模型注册表，需实现 `create(key, **params)`；缺省用全局
            `model_registry`（测试可注入 fake 以记录 fit 调用）。
        base_model_cols: 基础模型子集；缺省为全部存活候选。

    Raises:
        ValueError: 候选为空、train/test 过滤后为空，或全部候选训练失败。
    """
    if not candidate_models:
        raise ValueError("build_region_prediction_bundle: candidate_models must not be empty")

    if registry is None:
        from ..models.registry import model_registry as registry

    params_map = dict(model_params or {})

    train_r = train[train["region"] == region].copy() if "region" in train.columns else train.copy()
    test_r = test[test["region"] == region].copy() if "region" in test.columns else test.copy()
    if len(train_r) == 0:
        raise ValueError(f"build_region_prediction_bundle: no train rows for region {region!r}")
    if len(test_r) == 0:
        raise ValueError(f"build_region_prediction_bundle: no test rows for region {region!r}")

    fit_df, val_df = split_fit_validation(train_r, validation_days)

    X_fit, y_fit = build_matrix(fit_df, target_col=_TARGET_COL)
    X_val, y_val = build_matrix(val_df, target_col=_TARGET_COL)
    X_train, y_train = build_matrix(train_r, target_col=_TARGET_COL)
    X_test, y_test = build_matrix(test_r, target_col=_TARGET_COL)

    val_preds: Dict[str, np.ndarray] = {}
    test_preds: Dict[str, np.ndarray] = {}
    fitted_test_models: Dict[str, Any] = {}
    failed_models: Dict[str, str] = {}

    for model_id in candidate_models:
        params = dict(params_map.get(model_id, {}))
        try:
            # 第一轮：只用 fit 段训练 -> validation 预测（validation 对该模型是"未来"）。
            val_model = registry.create(model_id, **params)
            val_model.fit(X_fit, y_fit)
            val_pred = np.asarray(val_model.predict(X_val), dtype=float)

            # 第二轮：在完整 train 上重训 -> test 预测；test 标签始终不参与。
            test_model = registry.create(model_id, **params)
            test_model.fit(X_train, y_train)
            test_pred = np.asarray(test_model.predict(X_test), dtype=float)
        except Exception as exc:  # 单个候选失败不应中断整池
            # 从 val/test 两侧同时移除，避免出现只有一侧存在的半截候选列。
            failed_models[model_id] = f"{type(exc).__name__}: {exc}"
            continue

        if len(val_pred) != len(val_df) or len(test_pred) != len(test_r):
            failed_models[model_id] = (
                f"prediction length mismatch: val={len(val_pred)}/{len(val_df)}, "
                f"test={len(test_pred)}/{len(test_r)}"
            )
            continue

        val_preds[model_id] = val_pred
        test_preds[model_id] = test_pred
        fitted_test_models[model_id] = test_model

    surviving = [m for m in candidate_models if m in val_preds]
    if not surviving:
        raise ValueError(
            "build_region_prediction_bundle: no candidate model survived training for "
            f"region {region!r}; failures={failed_models}"
        )

    df_val = pd.DataFrame({"timestamp": pd.to_datetime(val_df["timestamp"]).values})
    df_val["y"] = np.asarray(y_val.values, dtype=float)
    for model_id in surviving:
        df_val[model_id] = val_preds[model_id]

    df_test = pd.DataFrame({"timestamp": pd.to_datetime(test_r["timestamp"]).values})
    df_test["y"] = np.asarray(y_test.values, dtype=float)
    for model_id in surviving:
        df_test[model_id] = test_preds[model_id]

    resolved_base = list(base_model_cols) if base_model_cols is not None else list(surviving)
    resolved_base = [m for m in resolved_base if m in surviving]

    metadata = {
        "region": region,
        "validation_days": int(validation_days),
        "n_fit": int(len(fit_df)),
        "n_val": int(len(val_df)),
        "n_train": int(len(train_r)),
        "n_test": int(len(test_r)),
        "requested_models": list(candidate_models),
        "failed_models": failed_models,
        "fit_end": str(pd.to_datetime(fit_df["timestamp"]).max()),
        "val_start": str(pd.to_datetime(val_df["timestamp"]).min()),
    }

    return RegionPredictionBundle(
        df_val=df_val,
        df_test=df_test,
        df_raw_val=_raw_frame(X_val, val_df["timestamp"]),
        df_raw_test=_raw_frame(X_test, test_r["timestamp"]),
        model_cols=list(surviving),
        base_model_cols=resolved_base,
        fitted_test_models=fitted_test_models,
        metadata=metadata,
    )
