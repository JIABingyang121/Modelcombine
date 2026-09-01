"""在线模型库预测：数据库匹配 -> 产物加载 -> 预测 -> trace -> 使用记录。

模型集合只能来自已保存的 scenario-data-combination 关系；本模块绝不调用候选
选择器或 Protocol B 求解器，也不在用户未来数据上重新挑选或训练模型。
没有兼容场景或必要产物缺失时直接失败，不回退旧引擎。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

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


class LibraryPredictionError(RuntimeError):
    """在线模型库预测的用户输入 / 产物边界错误。"""


def _load_scenario(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [field for field in _REQUIRED_SCENARIO_FIELDS if field not in payload]
    if missing:
        raise LibraryPredictionError(f"scenario 文件缺少必填字段: {missing}")
    if not isinstance(payload["signature"], dict) or not payload["signature"]:
        raise LibraryPredictionError(
            "scenario 的 signature 必须是非空对象（调用方按用户历史数据生成的场景/数据特征）"
        )
    return payload


def _match_scenario(store: ModelStore, scenario: Dict[str, Any]) -> Tuple[str, float]:
    candidates = store.list_scenarios(
        task_type=scenario["task_type"],
        business_domain=scenario["business_domain"],
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
    features_path: str,
    output_path: str,
) -> Dict[str, Any]:
    store = ModelStore(str(database))
    try:
        return _predict(store, scenario_path, features_path, output_path)
    finally:
        store.close()


def _predict(
    store: ModelStore,
    scenario_path: str,
    features_path: str,
    output_path: str,
) -> Dict[str, Any]:
    scenario = _load_scenario(scenario_path)
    features = pd.read_csv(features_path)
    if "timestamp" not in features.columns:
        raise LibraryPredictionError("features 文件缺少 timestamp 列")

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
    combination_predictor = load_artifact(combo_path)

    member_types = list(combination_predictor.member_ids)
    if len(combination["members"]) != len(member_types):
        raise LibraryPredictionError(
            f"组合器成员数与关系成员数不一致: {len(member_types)} vs {len(combination['members'])}"
        )

    base_predictions: Dict[str, np.ndarray] = {}
    model_ids = []
    model_artifact_paths = {}
    for member, member_type in zip(combination["members"], member_types):
        model_row = store.get_model(member["model_id"])
        if model_row is None:
            raise LibraryPredictionError(f"模型 {member['model_id']} 未登记")
        model_ids.append(member["model_id"])
        required = list(model_row["required_features"])
        missing = [column for column in required if column not in features.columns]
        if missing:
            raise LibraryPredictionError(
                f"模型 {member['model_id']} 需要的特征在输入中缺失: {missing}"
            )
        artifact_path = Path(model_row["artifact_path"])
        if not artifact_path.exists():
            raise LibraryPredictionError(f"模型产物不存在: {artifact_path}")
        model_artifact_paths[member["model_id"]] = str(artifact_path)
        model = load_artifact(artifact_path)
        base_predictions[member_type] = np.asarray(
            model.predict(features[required]), dtype=float
        )

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
        "model_ids": model_ids,
        "member_weights": {
            member["model_id"]: float(member["weight"])
            for member in combination["members"]
        },
        "member_types": member_types,
        "artifact_paths": {
            "combination": str(combo_path),
            "models": model_artifact_paths,
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
