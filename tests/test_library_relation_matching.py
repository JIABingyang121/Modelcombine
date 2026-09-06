"""在线匹配单位是完整的"场景—数据—组合"关系，不是"先挑场景再挑误差最小的关系"。

模型库的一条记录是：

```text
场景 S + 数据画像 D + 预测长度 forecast_steps -> 组合 C -> 成员 M1, M2, ...
```

同一个场景下可以积累多段用户历史数据，各自关联不同的组合。用户提交自己的历史负荷时，
系统必须挑**数据画像与之最相似**的那条关系，而不是挑 validation MAE 最小或排在最前的
那条——否则"按场景相似度检索已验证组合"这句话就名不副实。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from scripts.train_combinations_kg import history_window_signature
from src.models.artifacts import save_artifact
from src.models.combination_predictor import CombinationPredictor
from src.storage.model_store import ModelStore
from tests.forecast_steps_fixtures import (
    DATASET,
    WEEKLY_FEATURES,
    _supervised,
    fit_weekly,
    make_series,
    register_models,
    run_predict,
    write_scenario,
)

STEPS = 168
SCENARIO_ID = "pjm_shared_scenario"
#: 两段历史数据的量级差一个数量级，相似度必须能分辨
NEAR_LEVEL_SEED = 11
FAR_SCALE = 8.0


def _series(rows: int, *, seed: int, scale: float = 1.0, start: str = "2025-01-01"):
    frame = make_series(rows, start=start, seed=seed)[["timestamp", "load"]].copy()
    frame["load"] = frame["load"] * scale
    return frame


def _build_two_data_profiles(tmp_path: Path) -> dict:
    """同一场景、同一预测长度下的两条关系，分别绑定两段差异明显的历史数据。

    S1（远）绑定 catboost_reg，且 validation MAE 更小；
    S2（近）绑定 lgbm_reg，validation MAE 更大。
    用户数据与 S2 同量级，因此必须命中 S2 那条关系。
    """
    db = tmp_path / "lib.sqlite3"
    artifacts = tmp_path / "artifacts"
    train = make_series(2000, start="2024-01-01", seed=3)
    daily = ["hour", "dayofweek", "lag_24"]
    register_models(db, artifacts, [
        ("catboost_reg", fit_weekly(train), WEEKLY_FEATURES),
        ("lgbm_reg", Ridge(alpha=1.0).fit(*_supervised(train, daily)), daily),
    ])

    store = ModelStore(str(db))
    store.add_scenario(
        scenario_id=SCENARIO_ID, task_type="load_forecast", business_domain="power_load",
        region=DATASET, horizon=1, forecast_steps=STEPS, freq="h",
        signature={"horizon": 1.0, "y_mean": 1000.0, "y_std": 60.0},
    )

    relations = {}
    specs = [
        # (样例名, 量级, 成员, validation MAE) —— 远的那条误差更小，用来验证不被误差偷换
        ("S1_far", FAR_SCALE, "catboost_reg", 0.5),
        ("S2_near", 1.0, "lgbm_reg", 5.0),
    ]
    for sample, scale, member, validation_mae in specs:
        frame = _series(900, seed=NEAR_LEVEL_SEED, scale=scale, start="2025-01-01")
        signature = history_window_signature(frame, freq="h", base_horizon=1)
        data_ref = f"data/{DATASET}/{sample}.csv"
        profile_id = store.add_data_profile(
            scenario_id=SCENARIO_ID, data_ref=data_ref, target_column="load",
            features=["timestamp", "load"], sample_count=len(frame),
            start_at=str(frame["timestamp"].iloc[0]), end_at=str(frame["timestamp"].iloc[-1]),
            signature=signature,
        )
        predictor = CombinationPredictor(
            member_ids=[member], linear_weights=[1.0],
            strategy="protocol_b_combination", interaction=None,
        )
        combo_path = save_artifact(predictor, artifacts / f"{sample}__combo.pkl")
        combo_id = store.add_combination(
            "protocol_b_combination", str(combo_path),
            [(f"{DATASET}__h1__{member}", 0, 1.0)],
        )
        relation_id = store.add_relation(
            SCENARIO_ID, profile_id, combo_id, validation_mae=validation_mae
        )
        relations[sample] = {
            "relation_id": relation_id, "data_profile_id": profile_id,
            "combination_id": combo_id, "member": member, "data_ref": data_ref,
            "start_at": str(frame["timestamp"].iloc[0]),
            "end_at": str(frame["timestamp"].iloc[-1]),
        }
    store.close()
    return {"db": db, "relations": relations}


def _predict_with_user_history(tmp_path: Path, db: Path, *, scale: float):
    history = _series(900, seed=NEAR_LEVEL_SEED, scale=scale, start="2026-01-01")
    history_path = tmp_path / f"history_{scale}.csv"
    history.to_csv(history_path, index=False)
    scenario = write_scenario(tmp_path, forecast_steps=STEPS, name=f"scenario_{scale}.json")
    output = tmp_path / f"forecast_{scale}.csv"
    proc = run_predict(db, scenario, history_path, output)
    return proc, output


def test_user_data_selects_the_most_similar_data_profile_not_the_lowest_val_mae(tmp_path):
    built = _build_two_data_profiles(tmp_path)
    near = built["relations"]["S2_near"]
    far = built["relations"]["S1_far"]

    proc, output = _predict_with_user_history(tmp_path, built["db"], scale=1.0)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    trace = json.loads(output.with_suffix(".trace.json").read_text())

    # 必须命中"数据最相似"的那条，而不是 validation MAE 更小的那条
    assert trace["relation_id"] == near["relation_id"]
    assert trace["data_profile_id"] == near["data_profile_id"]
    assert trace["combination_id"] == near["combination_id"]
    assert trace["member_types"] == [near["member"]]
    assert trace["relation_id"] != far["relation_id"]
    assert trace["validation_mae"] > 1.0  # 选中的正是误差更大的那条


def test_a_user_matching_the_other_sample_selects_the_other_relation(tmp_path):
    """把用户数据换成与远样例同量级：必须改选另一条关系。

    没有这一半，上一条用例可能只是"恰好选中了排在后面的那条"。
    """
    built = _build_two_data_profiles(tmp_path)
    far = built["relations"]["S1_far"]

    proc, output = _predict_with_user_history(tmp_path, built["db"], scale=FAR_SCALE)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    trace = json.loads(output.with_suffix(".trace.json").read_text())

    assert trace["relation_id"] == far["relation_id"]
    assert trace["data_profile_id"] == far["data_profile_id"]
    assert trace["member_types"] == [far["member"]]


def test_trace_explains_which_data_and_which_combination_were_chosen(tmp_path):
    """trace 要能回答"这次为什么选了这个模型组合"。"""
    built = _build_two_data_profiles(tmp_path)
    near = built["relations"]["S2_near"]

    proc, output = _predict_with_user_history(tmp_path, built["db"], scale=1.0)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    trace = json.loads(output.with_suffix(".trace.json").read_text())

    assert trace["scenario_id"] == SCENARIO_ID
    assert trace["data_profile_id"] == near["data_profile_id"]
    assert trace["data_ref"] == near["data_ref"]
    assert trace["data_start_at"] == near["start_at"]
    assert trace["data_end_at"] == near["end_at"]
    assert 0.0 <= trace["data_similarity"] <= 1.0
    assert trace["relation_id"] == near["relation_id"]
    assert trace["combination_id"] == near["combination_id"]
    assert trace["model_ids"] == [f"{DATASET}__h1__{near['member']}"]
    assert trace["forecast_steps"] == STEPS
    assert trace["selector_invoked"] is False


def _seed_identical_profile_relation(
    store: ModelStore, artifacts: Path, *, scenario_id: str, member: str,
    signature: dict, validation_mae: float,
) -> int:
    """在一个新场景下建一条关系，数据画像与其他场景**完全相同**。"""
    store.add_scenario(
        scenario_id=scenario_id, task_type="load_forecast", business_domain="power_load",
        region=DATASET, horizon=1, forecast_steps=STEPS, freq="h", signature=signature,
    )
    profile_id = store.add_data_profile(
        scenario_id=scenario_id, data_ref=f"data/{scenario_id}.csv", target_column="load",
        features=["timestamp", "load"], sample_count=900,
        start_at="2025-01-01 00:00:00", end_at="2025-02-07 11:00:00", signature=signature,
    )
    predictor = CombinationPredictor(
        member_ids=[member], linear_weights=[1.0],
        strategy="protocol_b_combination", interaction=None,
    )
    combo_path = save_artifact(predictor, artifacts / f"{scenario_id}__combo.pkl")
    combo_id = store.add_combination(
        "protocol_b_combination", str(combo_path), [(f"{DATASET}__h1__{member}", 0, 1.0)]
    )
    return store.add_relation(scenario_id, profile_id, combo_id, validation_mae=validation_mae)


def test_feedback_tiebreak_applies_across_scenarios_not_only_within_one(tmp_path):
    """数据画像完全相同但分属不同 scenario 时，并列判据必须**全局**按反馈排序。

    三条关系画像一模一样，相似度必然相等：

    - scenario_a：无反馈，validation_mae 最小（0.1）
    - scenario_b：有反馈，mean_actual_mae = 9.0
    - scenario_c：有反馈，mean_actual_mae = 1.0

    必须选 scenario_c——它既不是 scenario_id 最小的，也不是 relation_id 最小的，
    更不是 validation_mae 最小的，只有"有反馈优先 + 实际 MAE 最小"能解释。
    """
    db = tmp_path / "lib.sqlite3"
    artifacts = tmp_path / "artifacts"
    train = make_series(2000, start="2024-01-01", seed=3)
    daily = ["hour", "dayofweek", "lag_24"]
    register_models(db, artifacts, [
        ("catboost_reg", fit_weekly(train), WEEKLY_FEATURES),
        ("lgbm_reg", Ridge(alpha=1.0).fit(*_supervised(train, daily)), daily),
    ])

    shared = history_window_signature(
        _series(900, seed=NEAR_LEVEL_SEED, start="2025-01-01"), freq="h", base_horizon=1
    )
    store = ModelStore(str(db))
    rel_a = _seed_identical_profile_relation(
        store, artifacts, scenario_id="pjm_scenario_a", member="catboost_reg",
        signature=shared, validation_mae=0.1,
    )
    rel_b = _seed_identical_profile_relation(
        store, artifacts, scenario_id="pjm_scenario_b", member="catboost_reg",
        signature=shared, validation_mae=5.0,
    )
    rel_c = _seed_identical_profile_relation(
        store, artifacts, scenario_id="pjm_scenario_c", member="lgbm_reg",
        signature=shared, validation_mae=5.0,
    )
    store.record_feedback(store.record_prediction_run(rel_b, "b.csv"), actual_mae=9.0)
    store.record_feedback(store.record_prediction_run(rel_c, "c.csv"), actual_mae=1.0)
    store.close()

    proc, output = _predict_with_user_history(tmp_path, db, scale=1.0)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    trace = json.loads(output.with_suffix(".trace.json").read_text())

    assert trace["relation_id"] == rel_c
    assert trace["scenario_id"] == "pjm_scenario_c"
    assert trace["member_types"] == ["lgbm_reg"]
    # 排除三种"看起来也对"的解释
    assert trace["relation_id"] != rel_a  # 不是 scenario_id / relation_id 最小的
    assert trace["relation_id"] != rel_b  # 不是"有反馈里排在前面"的
