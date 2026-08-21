"""Protocol B 求解上下文的统一构造器（System A/B 合一 Task 1）。

原实现位于 `scripts/train_combinations_kg.py::_build_protocol_b_solve_context`，
只服务 System B 的真实数据实验脚本。System A 的 demo 入口要改为“先生成候选
预测矩阵、再调用 Protocol B”，两侧必须共用同一份上下文构造逻辑，否则会出现
第二份语义相近但细节漂移的实现（历史上 System A/B 正是这样分叉的）。

本模块只做上下文构造与输入校验，不做任何决策；决策仍在 `ProtocolBBackend`
内部完成。相对旧实现唯一的行为增强是把原先静默的退化路径改为显式报错
（见 `_validate_frame`），避免问题被推迟到 Protocol B 内部才暴露。
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..enums import TaskType
from ..schema import DataContract, ScenarioDefinition
from .context import SolveContext

# 非特征列：y 是标签，timestamp 是对齐键，都不参与 available_features。
_NON_FEATURE_COLUMNS = {"y", "timestamp"}

_BUSINESS_DOMAIN = "load_forecast"


def _validate_frame(frame: Any, *, name: str, model_cols: Sequence[str]) -> None:
    """校验 val/test 预测矩阵满足 Protocol B 的输入前提。"""
    if frame is None:
        raise ValueError(f"build_protocol_b_context: {name} is required, got None")
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(
            f"build_protocol_b_context: {name} must be a DataFrame, got {type(frame).__name__}"
        )
    if len(frame) == 0:
        raise ValueError(f"build_protocol_b_context: {name} is empty")

    columns = set(frame.columns)
    if "y" not in columns:
        raise ValueError(f"build_protocol_b_context: {name} is missing the label column 'y'")
    # Protocol B 内部（如 conflict.generate_stable_key）依赖 timestamp 对齐，
    # 缺失时旧路径会以 KeyError 或静默跳过的形式失败，这里提前拦下。
    if "timestamp" not in columns:
        raise ValueError(f"build_protocol_b_context: {name} is missing the 'timestamp' column")

    missing_models = [col for col in model_cols if col not in columns]
    if missing_models:
        raise ValueError(
            f"build_protocol_b_context: {name} is missing candidate columns {missing_models}"
        )


def _scenario_signature(
    *,
    horizon: int,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    model_cols: Sequence[str],
) -> dict:
    """场景签名：确定性、可 JSON 序列化，供 compute_scenario_id 生成稳定 id。"""
    y_val = np.asarray(df_val["y"].values, dtype=float)
    finite_y = y_val[np.isfinite(y_val)]
    return {
        "horizon": int(horizon),
        "n_val": int(len(df_val)),
        "n_test": int(len(df_test)),
        "n_models": int(len(model_cols)),
        "mean_y": float(np.mean(finite_y)) if finite_y.size else 0.0,
        "std_y": float(np.std(finite_y)) if finite_y.size else 0.0,
    }


def build_protocol_b_context(
    *,
    dataset: str,
    horizon: int,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    df_raw_val: Optional[pd.DataFrame],
    df_raw_test: Optional[pd.DataFrame],
    model_cols: List[str],
    base_model_cols: List[str],
    feedback_store: Any,
    return_predictions: bool = False,
) -> SolveContext:
    """构造 `ProtocolBBackend` 可直接消费的 `SolveContext`。

    Args:
        dataset: 数据集/区域名，同时作为场景 id 前缀。
        horizon: 预测步长。
        df_val / df_test: 候选预测矩阵，必须含 `timestamp`、`y` 和全部候选模型列。
        df_raw_val / df_raw_test: 原始特征表，可为 None；仅用于汇总可用特征。
        model_cols: 候选模型列（Protocol B 的组合对象）。
        base_model_cols: 基础模型列子集。
        feedback_store: `KGFeedbackStore` 或 None。
        return_predictions: 是否要求 Protocol B 交出真实 pred_val/pred_test。
            默认 False，实验脚本行为不变。

    Raises:
        ValueError: 候选列为空，或 val/test 缺 y/timestamp/候选列，或行数为空。
    """
    if not model_cols:
        raise ValueError("build_protocol_b_context: model_cols must not be empty")

    _validate_frame(df_val, name="df_val", model_cols=model_cols)
    _validate_frame(df_test, name="df_test", model_cols=model_cols)

    signature = _scenario_signature(
        horizon=horizon,
        df_val=df_val,
        df_test=df_test,
        model_cols=model_cols,
    )
    data_contract = DataContract(
        required_columns={"y": "float"},
        freq="H",
        min_samples=max(1, int(horizon)),
        business_domain=_BUSINESS_DOMAIN,
    )
    scenario = ScenarioDefinition(
        task_type=TaskType.FORECASTING,
        business_domain=_BUSINESS_DOMAIN,
        data_contract=data_contract,
        target_schema={"yhat": "float"},
        primary_metric="MAE",
        signature_features=list(signature.keys()),
        signature=signature,
        region=dataset,
    )

    raw_features: set = set()
    for raw_df in (df_raw_val, df_raw_test):
        if raw_df is not None:
            raw_features.update(str(c) for c in raw_df.columns)
    available_features = (
        set(df_val.columns) | set(df_test.columns) | raw_features
    ) - _NON_FEATURE_COLUMNS

    return SolveContext(
        scenario=scenario,
        available_features=available_features,
        model_cols=list(model_cols),
        df_val=df_val,
        df_test=df_test,
        df_raw_val=df_raw_val,
        df_raw_test=df_raw_test,
        horizon=int(horizon),
        dataset_name=dataset,
        base_model_cols=list(base_model_cols),
        feedback_store=feedback_store,
        return_predictions=bool(return_predictions),
    )
