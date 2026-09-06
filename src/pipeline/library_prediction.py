"""在线模型库预测：数据库匹配 -> 产物加载 -> 预测 -> trace -> 使用记录。

模型集合只能来自已保存的 scenario-data-combination 关系；本模块绝不调用候选
选择器或 Protocol B 求解器，也不在用户未来数据上重新挑选或训练模型。
没有兼容场景或必要产物缺失时直接失败，不回退旧引擎。

在线步骤固定为（方案 §3.4）：

```text
用户场景 + 历史 timestamp,load + forecast_steps
  -> 生成 signature
  -> 精确过滤 task/business/region/freq/base_horizon/forecast_steps
  -> 对剩余的每一条"场景—数据—组合"关系，按其 data_profile 的数据特征算相似度
  -> 取最相似的那条完整关系
  -> 加载该关系已保存的组合及成员模型
  -> 每个成员生成完整轨迹
  -> 用已保存权重融合
  -> 输出和 trace
```

匹配单位是**整条关系**，不是"先挑场景、再在该场景下挑一条误差较小的关系"。同一个场景
下会积累多段历史数据，各自关联不同的已验证组合；用户提交自己的历史负荷时要拿到的是
**数据画像与之最相似**的那条关系。命中之后不再按 validation MAE 或反馈统计改选。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.core.index import ScenarioIndex
from src.models.artifacts import load_artifact
from src.models.trajectory_forecast import (
    TrajectoryForecastError,
    calendar_frame,
    future_timestamps,
    generate_member_trajectory,
)
from src.storage.model_store import (
    ModelStore,
    UnsupportedForecastSteps,
    validate_forecast_steps,
)

#: 基础预测器的单步语义：当前候选池全部按 X(t) -> y(t+1) 训练。
BASE_HORIZON = 1

_REQUIRED_SCENARIO_FIELDS = (
    "task_type",
    "business_domain",
    "region",
    "horizon",
    "forecast_steps",
    "freq",
    "signature",
)

_HISTORY_SCENARIO_FIELDS = (
    "task_type",
    "business_domain",
    "region",
    "forecast_steps",
    "freq",
)
_COUNTRY_BY_REGION = {"pjm": "US", "aemo_vic": "AU", "aemo_nsw": "AU"}


class LibraryPredictionError(RuntimeError):
    """在线模型库预测的用户输入 / 产物边界错误。"""


def _load_scenario(path: str, *, history_mode: bool = False) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required_fields = _HISTORY_SCENARIO_FIELDS if history_mode else _REQUIRED_SCENARIO_FIELDS
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise LibraryPredictionError(f"scenario 文件缺少必填字段: {missing}")
    try:
        payload["forecast_steps"] = validate_forecast_steps(payload["forecast_steps"])
    except UnsupportedForecastSteps as exc:
        raise LibraryPredictionError(str(exc)) from exc
    if history_mode:
        return payload
    if not isinstance(payload["signature"], dict) or not payload["signature"]:
        raise LibraryPredictionError(
            "scenario 的 signature 必须是非空对象（调用方按用户历史数据生成的场景/数据特征）"
        )
    return payload


def _country_for(region: str) -> str:
    try:
        return _COUNTRY_BY_REGION[region]
    except KeyError:
        raise LibraryPredictionError(
            f"未知 region={region!r}，无法确定日历特征所用节假日日历"
        ) from None


def _match_relation(store: ModelStore, scenario: Dict[str, Any]) -> Dict[str, Any]:
    """先按硬契约精确过滤，再在剩余的**每一条关系**上按其数据画像算相似度。

    ``forecast_steps`` 是不能混用的任务契约，必须在相似度之前过滤：168 步请求
    不得匹配 24 或 720 步关系。过滤之后按 ``data_profile`` 的 signature 排序，
    **不看** validation MAE——那是历史表现，不是"这段数据像不像"。

    只有相似度完全并列时（两段历史数据的画像无法区分），才退回 ``ModelStore``
    既有的关系排序做 tie-break：有真实反馈的优先、按 ``mean_actual_mae`` 升序，
    其次按 ``validation_mae`` 升序、再按 ``relation_id`` 升序。``ScenarioIndex.query``
    的排序是稳定的，因此按该顺序插入即可让它成为 tie-break。
    """
    forecast_steps = validate_forecast_steps(scenario["forecast_steps"])
    scenarios = store.list_scenarios(
        task_type=scenario["task_type"],
        business_domain=scenario["business_domain"],
        region=scenario["region"],
        horizon=int(scenario["horizon"]),
        forecast_steps=forecast_steps,
        freq=scenario["freq"],
    )
    if not scenarios:
        raise LibraryPredictionError(
            "no compatible scenario: 数据库中没有 task_type/business_domain/region/"
            f"horizon/forecast_steps={forecast_steps}/freq 全部匹配的历史场景"
        )

    # 保持 store 给出的关系顺序：它就是并列时的 tie-break 依据
    candidates: Dict[str, Dict[str, Any]] = {}
    for candidate in scenarios:
        for relation in store.list_relations_for_scenario(candidate["scenario_id"]):
            profile = store.get_data_profile(relation["data_profile_id"])
            candidates[str(relation["relation_id"])] = {
                "scenario": candidate, "relation": relation, "profile": profile,
            }
    if not candidates:
        raise LibraryPredictionError(
            "no compatible relation: 匹配到的历史场景下没有任何已保存的组合关系"
        )

    index = ScenarioIndex()
    for key, entry in candidates.items():
        index.add({
            "scenario_id": key,
            "signature": entry["profile"]["signature"],
            "business_domain": entry["scenario"]["business_domain"],
        })
    best = index.query(signature=scenario["signature"])[0]
    chosen = candidates[best["scenario_id"]]
    chosen["data_similarity"] = float(best["_score"])
    return chosen


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
    forecast_steps = scenario["forecast_steps"]
    features = pd.read_csv(features_path)
    if "timestamp" not in features.columns:
        raise LibraryPredictionError("features 文件缺少 timestamp 列")
    if len(features) != forecast_steps:
        raise LibraryPredictionError(
            f"features 行数 {len(features)} 与请求的 forecast_steps={forecast_steps} 不一致"
        )

    matched = _load_matched_combination(store, scenario)
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

    trace = _base_trace(
        matched,
        # --features 入口可以命中 V1 的 horizon=24 关系，基础预测器语义由匹配到的
        # 场景决定，不是恒为 1。
        base_horizon=int(scenario["horizon"]),
        forecast_steps=forecast_steps,
        n_rows=int(len(features)),
        prediction_run_id=prediction_run_id,
        output=output,
    )
    trace["signature_source"] = "caller"
    return _write_trace(trace, output)


def _load_matched_combination(store: ModelStore, scenario: Dict[str, Any]) -> Dict[str, Any]:
    matched = _match_relation(store, scenario)
    relation = matched["relation"]
    profile = matched["profile"]
    scenario_id = matched["scenario"]["scenario_id"]
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
        "data_similarity": matched["data_similarity"],
        "data_profile": profile,
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


def _base_trace(
    matched: Dict[str, Any],
    *,
    base_horizon: int,
    forecast_steps: int,
    n_rows: int,
    prediction_run_id: int,
    output: Path,
) -> Dict[str, Any]:
    relation = matched["relation"]
    predictor = matched["predictor"]
    return {
        "mode": "model_library",
        "model_selection_source": "saved_relation",
        "selector_invoked": False,
        "scenario_id": matched["scenario_id"],
        "relation_id": relation["relation_id"],
        "combination_id": matched["combination"]["combination_id"],
        # 命中的是哪一段历史数据——trace 要能回答"这次为什么选了这个组合"
        "data_profile_id": relation["data_profile_id"],
        "data_ref": matched["data_profile"]["data_ref"],
        "data_start_at": matched["data_profile"]["start_at"],
        "data_end_at": matched["data_profile"]["end_at"],
        "data_similarity": matched["data_similarity"],
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
        "validation_mae": relation["validation_mae"],
        "mean_actual_mae": relation["mean_actual_mae"],
        "feedback_count": relation["feedback_count"],
        "base_horizon": int(base_horizon),
        "forecast_steps": int(forecast_steps),
        "n_rows": int(n_rows),
        "prediction_run_id": prediction_run_id,
        "output": str(output),
    }


def _write_trace(trace: Dict[str, Any], output: Path) -> Dict[str, Any]:
    trace_path = output.with_suffix(".trace.json")
    trace_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    trace["trace_path"] = str(trace_path)
    return trace


def _history_signature(history: pd.DataFrame, freq: str) -> Dict[str, float]:
    """与离线建库共用同一个签名实现，保证两侧 key 集与数值口径一致。"""
    from scripts.train_combinations_kg import history_window_signature

    try:
        return history_window_signature(history, freq=freq, base_horizon=BASE_HORIZON)
    except ValueError as exc:
        raise LibraryPredictionError(str(exc)) from exc


def _read_history(history_path: str) -> pd.DataFrame:
    history = pd.read_csv(history_path)
    missing = [column for column in ("timestamp", "load") if column not in history.columns]
    if missing:
        raise LibraryPredictionError(f"history 文件缺少必填列: {missing}")
    history = history[["timestamp", "load"]].copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    history["load"] = pd.to_numeric(history["load"])
    return history.sort_values("timestamp").reset_index(drop=True)


def _predict_from_history(
    store: ModelStore,
    scenario_path: str,
    history_path: str,
    output_path: str,
) -> Dict[str, Any]:
    scenario = _load_scenario(scenario_path, history_mode=True)
    forecast_steps = scenario["forecast_steps"]
    history = _read_history(history_path)
    country = _country_for(scenario["region"])

    # 基础预测器是 h=1；用户请求的是长度 forecast_steps 的完整轨迹，两者不混用。
    scenario["horizon"] = BASE_HORIZON
    scenario["signature"] = _history_signature(history, scenario["freq"])
    matched = _load_matched_combination(store, scenario)
    predictor = matched["predictor"]

    timestamps = future_timestamps(history, forecast_steps)
    base_predictions: Dict[str, np.ndarray] = {}
    for model_row, member_type in zip(matched["models"], matched["member_types"]):
        try:
            # 每个成员独立递归：只把自己的预测写回自己的 lag/rolling 特征。
            base_predictions[member_type] = generate_member_trajectory(
                model=model_row["predictor"],
                model_type=model_row["model_type"],
                required_features=model_row["required_features"],
                history=history,
                forecast_steps=forecast_steps,
                country=country,
            )
        except TrajectoryForecastError as exc:
            raise LibraryPredictionError(
                f"成员 {model_row['model_id']} 无法生成 {forecast_steps} 步轨迹: {exc}"
            ) from exc

    raw_features = calendar_frame(timestamps, country)
    for feature_name in predictor.required_feature_names:
        if feature_name not in raw_features.columns:
            raise LibraryPredictionError(
                f"组合器 interaction 需要的特征无法由未来时间戳生成: {feature_name}"
            )
    yhat = np.asarray(predictor.predict(base_predictions, raw_features), dtype=float)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": timestamps, "yhat": yhat}).to_csv(output, index=False)

    prediction_run_id = store.record_prediction_run(matched["relation"]["relation_id"], str(output))
    trace = _base_trace(
        matched,
        base_horizon=BASE_HORIZON,
        forecast_steps=forecast_steps,
        n_rows=int(len(yhat)),
        prediction_run_id=prediction_run_id,
        output=output,
    )
    trace["signature_source"] = "history"
    return _write_trace(trace, output)
