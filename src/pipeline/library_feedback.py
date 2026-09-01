"""在线反馈：用后来返回的真实值更新对应关系的实际表现统计。

有 row_id 时按 row_id 一对一对齐，否则按 timestamp 对齐；重复、缺失或不一致直接报错。同一个
prediction run 只允许反馈一次（由 model_store 的 SQL 条件更新保证）。
本模块不调整 Protocol B 关系权重、不把一次 MAE 转成新的"关系强度"。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.storage.model_store import ModelStore


class LibraryFeedbackError(RuntimeError):
    """反馈入口的用户输入 / 状态边界错误。"""


def apply_feedback(
    *,
    database: str,
    prediction_run_id: int,
    actual_path: str,
) -> Dict[str, Any]:
    store = ModelStore(str(database))
    try:
        return _apply_feedback(store, int(prediction_run_id), actual_path)
    finally:
        store.close()


def _apply_feedback(
    store: ModelStore,
    prediction_run_id: int,
    actual_path: str,
) -> Dict[str, Any]:
    run = store.get_prediction_run(prediction_run_id)
    if run is None:
        raise LibraryFeedbackError(f"prediction run {prediction_run_id} 不存在")

    prediction_ref = Path(run["prediction_ref"])
    if not prediction_ref.exists():
        raise LibraryFeedbackError(f"原预测文件不存在: {prediction_ref}")
    preds = pd.read_csv(prediction_ref)
    if "timestamp" not in preds.columns or "yhat" not in preds.columns:
        raise LibraryFeedbackError("原预测文件缺少 timestamp / yhat 列")

    actual = pd.read_csv(actual_path)
    if "timestamp" not in actual.columns or "y" not in actual.columns:
        raise LibraryFeedbackError("actual 文件必须包含 timestamp 和 y 列")

    if "row_id" in preds.columns and "row_id" in actual.columns:
        key = "row_id"
    else:
        key = "timestamp"
        preds = preds.assign(timestamp=pd.to_datetime(preds["timestamp"], errors="raise"))
        actual = actual.assign(timestamp=pd.to_datetime(actual["timestamp"], errors="raise"))
    if preds[key].duplicated().any():
        raise LibraryFeedbackError(f"原预测文件存在重复 {key}")
    if actual[key].duplicated().any():
        raise LibraryFeedbackError(f"actual 文件存在重复 {key}")

    merged = preds.merge(actual[[key, "y"]], on=key, how="inner")
    if len(merged) != len(preds) or len(merged) != len(actual):
        raise LibraryFeedbackError(
            f"预测与真实值 {key} 未一一对齐: preds={len(preds)}, "
            f"actual={len(actual)}, matched={len(merged)}"
        )

    actual_mae = float(
        np.mean(np.abs(merged["yhat"].to_numpy(dtype=float) - merged["y"].to_numpy(dtype=float)))
    )
    store.record_feedback(prediction_run_id, actual_mae)

    relation = store.get_relation(int(run["relation_id"]))
    return {
        "prediction_run_id": prediction_run_id,
        "relation_id": int(run["relation_id"]),
        "actual_mae": actual_mae,
        "feedback_count": relation["feedback_count"],
        "mean_actual_mae": relation["mean_actual_mae"],
        "n_rows": int(len(merged)),
    }
