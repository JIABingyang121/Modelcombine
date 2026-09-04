"""在线模型库预测：数据库匹配 -> 产物加载 -> 预测 -> trace -> 使用记录。

模型集合只能来自已保存的 scenario-data-combination 关系；本模块绝不调用候选
选择器或 Protocol B 求解器，也不在用户未来数据上重新挑选或训练模型。
没有兼容场景或必要产物缺失时直接失败，不回退旧引擎。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.index import ScenarioIndex
from src.models.artifacts import load_artifact
from src.storage.model_store import ModelStore

_REQUIRED_SCENARIO_FIELDS = (
    "task_type",
    "business_domain",
    "region",
    "horizon",
    "freq",
    "signature",
)

_HISTORY_SCENARIO_FIELDS = ("task_type", "business_domain", "region", "freq")
_COUNTRY_BY_REGION = {"pjm": "US", "aemo_vic": "AU", "aemo_nsw": "AU"}


class LibraryPredictionError(RuntimeError):
    """在线模型库预测的用户输入 / 产物边界错误。"""


def _load_scenario(path: str, *, history_mode: bool = False) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required_fields = _HISTORY_SCENARIO_FIELDS if history_mode else _REQUIRED_SCENARIO_FIELDS
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise LibraryPredictionError(f"scenario 文件缺少必填字段: {missing}")
    if history_mode:
        return payload
    if not isinstance(payload["signature"], dict) or not payload["signature"]:
        raise LibraryPredictionError(
            "scenario 的 signature 必须是非空对象（调用方按用户历史数据生成的场景/数据特征）"
        )
    return payload


def _match_scenario(store: ModelStore, scenario: Dict[str, Any]) -> Tuple[str, float]:
    candidates = store.list_scenarios(
        task_type=scenario["task_type"],
        business_domain=scenario["business_domain"],
        region=scenario["region"],
        horizon=int(scenario["horizon"]),
        freq=scenario["freq"],
    )
    if not candidates:
        raise LibraryPredictionError(
            "no compatible scenario: 数据库中没有 "
            "task_type/business_domain/horizon/freq 全部匹配的历史场景"
        )
    index = ScenarioIndex()
    for candidate in candidates:
        index.add(
            {
                "scenario_id": candidate["scenario_id"],
                "signature": candidate["signature"],
                "business_domain": candidate["business_domain"],
            }
        )
    # signature 由 _load_scenario 保证存在且非空；每条 candidate 产出一条排序结果，
    # 无需数据库首条记录回退。
    best = index.query(signature=scenario["signature"])[0]
    return best["scenario_id"], float(best["_score"])


def predict(
    *,
    database: str,
    scenario_path: str,
    features_path: Optional[str] = None,
    history_path: Optional[str] = None,
    output_path: str,
) -> Dict[str, Any]:
    store = ModelStore(str(database))
    try:
        if history_path is not None:
            return _predict_from_history(store, scenario_path, history_path, output_path)
        return _predict(store, scenario_path, features_path, output_path)
    finally:
        store.close()


def _predict(
    store: ModelStore,
    scenario_path: str,
    features_path: Optional[str],
    output_path: str,
) -> Dict[str, Any]:
    scenario = _load_scenario(scenario_path)
    features = pd.read_csv(features_path)
    if "timestamp" not in features.columns:
        raise LibraryPredictionError("features 文件缺少 timestamp 列")

    matched = _load_matched_combination(store, scenario)
    scenario_id, similarity = matched["scenario_id"], matched["similarity"]
    relation = matched["relation"]
    combination = matched["combination"]
    combination_predictor = matched["predictor"]
    for model_row in matched["models"]:
        missing = [column for column in model_row["required_features"] if column not in features.columns]
        if missing:
            raise LibraryPredictionError(
                f"模型 {model_row['model_id']} 需要的特征在输入中缺失: {missing}"
            )

    base_predictions = _base_predictions(matched, features)

    for feature_name in combination_predictor.required_feature_names:
        if feature_name not in features.columns:
            raise LibraryPredictionError(
                f"组合器 interaction 需要的特征在输入中缺失: {feature_name}"
            )

    yhat = np.asarray(combination_predictor.predict(base_predictions, features), dtype=float)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {"timestamp": features["timestamp"], "yhat": yhat}
    if "row_id" in features.columns:
        output_data = {"row_id": features["row_id"], **output_data}
    pd.DataFrame(output_data).to_csv(output, index=False)

    prediction_run_id = store.record_prediction_run(relation["relation_id"], str(output))

    trace = {
        "mode": "model_library",
        "model_selection_source": "saved_relation",
        "selector_invoked": False,
        "scenario_id": scenario_id,
        "scenario_similarity": similarity,
        "relation_id": relation["relation_id"],
        "combination_id": combination["combination_id"],
        "data_profile_id": relation["data_profile_id"],
        "strategy": combination["strategy"],
        "model_ids": matched["model_ids"],
        "member_weights": {
            member["model_id"]: float(member["weight"])
            for member in combination["members"]
        },
        "member_types": matched["member_types"],
        "artifact_paths": {
            "combination": matched["combo_path"],
            "models": matched["model_artifact_paths"],
        },
        "has_interaction": combination_predictor.interaction is not None,
        "validation_mae": relation["validation_mae"],
        "mean_actual_mae": relation["mean_actual_mae"],
        "feedback_count": relation["feedback_count"],
        "n_rows": int(len(features)),
        "prediction_run_id": prediction_run_id,
        "output": str(output),
    }
    trace_path = output.with_suffix(".trace.json")
    trace_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    trace["trace_path"] = str(trace_path)
    return trace


def _load_matched_combination(store: ModelStore, scenario: Dict[str, Any]) -> Dict[str, Any]:
    scenario_id, similarity = _match_scenario(store, scenario)
    relations = store.list_relations_for_scenario(scenario_id)
    if not relations:
        raise LibraryPredictionError(f"scenario {scenario_id} 没有已保存的组合关系")
    relation = relations[0]
    combination = store.get_combination(relation["combination_id"])
    if combination is None:
        raise LibraryPredictionError(f"combination {relation['combination_id']} 不存在")
    combo_path = Path(combination["artifact_path"])
    if not combo_path.exists():
        raise LibraryPredictionError(f"组合器产物不存在: {combo_path}")
    predictor = load_artifact(combo_path)
    member_types = list(predictor.member_ids)
    if len(combination["members"]) != len(member_types):
        raise LibraryPredictionError(
            f"组合器成员数与关系成员数不一致: {len(member_types)} vs {len(combination['members'])}"
        )
    models = []
    model_ids = []
    model_artifact_paths = {}
    for member in combination["members"]:
        model_row = store.get_model(member["model_id"])
        if model_row is None:
            raise LibraryPredictionError(f"模型 {member['model_id']} 未登记")
        artifact_path = Path(model_row["artifact_path"])
        if not artifact_path.exists():
            raise LibraryPredictionError(f"模型产物不存在: {artifact_path}")
        model_row["predictor"] = load_artifact(artifact_path)
        models.append(model_row)
        model_ids.append(member["model_id"])
        model_artifact_paths[member["model_id"]] = str(artifact_path)
    return {
        "scenario_id": scenario_id,
        "similarity": similarity,
        "relation": relation,
        "combination": combination,
        "combo_path": str(combo_path),
        "predictor": predictor,
        "member_types": member_types,
        "models": models,
        "model_ids": model_ids,
        "model_artifact_paths": model_artifact_paths,
    }


def _base_predictions(matched: Dict[str, Any], features: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        member_type: np.asarray(model_row["predictor"].predict(features[model_row["required_features"]]), dtype=float)
        for model_row, member_type in zip(matched["models"], matched["member_types"])
    }


def _history_signature(history: pd.DataFrame, freq: str) -> Dict[str, float]:
    from scripts.train_combinations_kg import _scenario_signature

    target_frame = history.rename(columns={"load": "y"})
    return _scenario_signature(target_frame, history, horizon=1, freq=freq)


def _forecast_origin_feature_row(
    history: pd.DataFrame,
    feature_names: set[str],
    country: str,
) -> pd.DataFrame:
    from scripts.generate_features import add_holiday, add_lag_roll_grouped, add_time_features

    lags = sorted({int(value) for value in re.findall(r"lag_(\d+)", " ".join(feature_names))})
    windows = sorted({int(value) for value in re.findall(r"roll(\d+)_(?:mean|std)", " ".join(feature_names))})
    frame = history.copy()
    frame = add_time_features(frame, "timestamp")
    frame = add_holiday(frame, "timestamp", country)
    frame = add_lag_roll_grouped(frame, [], "timestamp", "load", lags, windows)
    return frame.tail(1)


def _predict_from_history(
    store: ModelStore,
    scenario_path: str,
    history_path: str,
    output_path: str,
) -> Dict[str, Any]:
    scenario = _load_scenario(scenario_path, history_mode=True)
    history = pd.read_csv(history_path)
    missing = [column for column in ("timestamp", "load") if column not in history.columns]
    if missing:
        raise LibraryPredictionError(f"history 文件缺少必填列: {missing}")
    history = history[["timestamp", "load"]].copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    history["load"] = pd.to_numeric(history["load"])
    history = history.sort_values("timestamp").reset_index(drop=True)
    scenario["horizon"] = 1
    scenario["signature"] = _history_signature(history, scenario["freq"])
    matched = _load_matched_combination(store, scenario)
    predictor = matched["predictor"]
    feature_names = set(predictor.required_feature_names)
    for model_row in matched["models"]:
        feature_names.update(model_row["required_features"])
    country = _COUNTRY_BY_REGION[scenario["region"]]
    future = []
    for _ in range(720):
        origin = history["timestamp"].iloc[-1]
        timestamp = origin + pd.Timedelta(hours=1)
        features = _forecast_origin_feature_row(history, feature_names, country)
        base_predictions = _base_predictions(matched, features)
        yhat = float(predictor.predict(base_predictions, features)[0])
        future.append({"timestamp": timestamp, "yhat": yhat})
        history.loc[len(history)] = [timestamp, yhat]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    forecast = pd.DataFrame(future)
    forecast.to_csv(output, index=False)
    prediction_run_id = store.record_prediction_run(matched["relation"]["relation_id"], str(output))
    trace = {
        "mode": "model_library",
        "model_selection_source": "saved_relation",
        "selector_invoked": False,
        "scenario_id": matched["scenario_id"],
        "scenario_similarity": matched["similarity"],
        "relation_id": matched["relation"]["relation_id"],
        "combination_id": matched["combination"]["combination_id"],
        "data_profile_id": matched["relation"]["data_profile_id"],
        "strategy": matched["combination"]["strategy"],
        "model_ids": matched["model_ids"],
        "member_weights": {
            member["model_id"]: float(member["weight"])
            for member in matched["combination"]["members"]
        },
        "member_types": matched["member_types"],
        "artifact_paths": {
            "combination": matched["combo_path"],
            "models": matched["model_artifact_paths"],
        },
        "has_interaction": predictor.interaction is not None,
        "validation_mae": matched["relation"]["validation_mae"],
        "mean_actual_mae": matched["relation"]["mean_actual_mae"],
        "feedback_count": matched["relation"]["feedback_count"],
        "n_rows": 720,
        "forecast_steps": 720,
        "signature_source": "history",
        "prediction_run_id": prediction_run_id,
        "output": str(output),
    }
    trace_path = output.with_suffix(".trace.json")
    trace_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    trace["trace_path"] = str(trace_path)
    return trace
