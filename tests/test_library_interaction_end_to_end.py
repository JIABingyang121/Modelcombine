"""带 interaction 的组合关系在在线入口上的逐值重放（真实 run.py predict）。

方案 §3.3：第一版只允许使用可由未来时间戳生成的日历特征做 interaction。要保证的是
**在线路径**能把这样一条关系原样重放出来——日历特征表取自未来目标时间戳、居中用的是
保存下来的 feature_means、设计矩阵按保存的列顺序拼装。

这里直接构造一条带 interaction 的关系，而不是指望离线建库在合成数据上恰好拟合出
interaction：Protocol B 的近零权重清理在本装置上会把组合收敛成单成员，而单成员关系
按约定就不带 interaction（见 tests/test_offline_model_library_wiring.py 的单成员用例）。
把用例建立在"引擎恰好选了什么"上会让它随装置漂移，覆盖的却不是本文件要守的东西。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.models.artifacts import load_artifact, save_artifact
from src.models.combination_predictor import CombinationPredictor, InteractionResidual
from src.models.trajectory_forecast import calendar_frame, generate_member_trajectory
from src.storage.model_store import ModelStore
from tests.forecast_steps_fixtures import (
    DATASET,
    REPO_ROOT,
    WEEKLY_FEATURES,
    _supervised,
    fit_weekly,
    make_series,
    register_models,
    run_predict,
    write_scenario,
)

STEPS = 168
MEMBERS = ("catboost_reg", "lgbm_reg")
#: interaction 只用日历特征；hour 在 168 步窗口上取值 0—23，居中均值固定为 11.5
INTERACTION_FEATURE = "hour"
INTERACTION_MEAN = 11.5
INTERACTION_COEF = 2.0e-4


def _hour_interaction(member_ids) -> InteractionResidual:
    """真实的 InteractionResidual：设计矩阵每列是 成员预测 ×（hour - 11.5）。"""
    regressor = Ridge(alpha=1.0, fit_intercept=False)
    regressor.coef_ = np.full(len(member_ids), INTERACTION_COEF, dtype=float)
    regressor.intercept_ = 0.0
    regressor.n_features_in_ = len(member_ids)
    return InteractionResidual(
        columns=[(member_id, INTERACTION_FEATURE) for member_id in member_ids],
        feature_means={INTERACTION_FEATURE: INTERACTION_MEAN},
        regressor=regressor,
    )


def _build_library_with_interaction(tmp_path: Path):
    db = tmp_path / "lib.sqlite3"
    artifacts = tmp_path / "artifacts"
    train = make_series(2000, start="2025-01-01", seed=1)
    daily_features = ["hour", "dayofweek", "lag_24"]
    register_models(db, artifacts, [
        (MEMBERS[0], fit_weekly(train), WEEKLY_FEATURES),
        (MEMBERS[1], Ridge(alpha=1.0).fit(*_supervised(train, daily_features)),
         daily_features),
    ])

    predictor = CombinationPredictor(
        member_ids=list(MEMBERS),
        linear_weights=[0.6, 0.4],
        strategy="protocol_b_combination",
        interaction=_hour_interaction(MEMBERS),
    )
    combo_path = save_artifact(predictor, artifacts / "combo_with_interaction.pkl")

    store = ModelStore(str(db))
    combo_id = store.add_combination(
        "protocol_b_combination", str(combo_path),
        [(f"{DATASET}__h1__{m}", order, w)
         for order, (m, w) in enumerate(zip(MEMBERS, predictor.linear_weights))],
    )
    signature = {"horizon": 1.0, "y_mean": 1000.0, "y_std": 60.0, "y_cv": 0.06}
    store.add_scenario(
        scenario_id="pjm_interaction", task_type="load_forecast",
        business_domain="power_load", region=DATASET, horizon=1,
        forecast_steps=STEPS, freq="h", signature=signature,
    )
    profile_id = store.add_data_profile(
        scenario_id="pjm_interaction", data_ref="data/pjm", target_column="load",
        features=["timestamp", "load"], sample_count=720,
        start_at="2025-01-01T00:00:00", end_at="2025-02-01T00:00:00", signature=signature,
    )
    store.add_relation("pjm_interaction", profile_id, combo_id, validation_mae=1.0)
    store.close()
    return db, predictor


def test_interaction_combo_replays_through_run_py_predict(tmp_path):
    db, predictor = _build_library_with_interaction(tmp_path)
    assert predictor.required_feature_names == [INTERACTION_FEATURE]

    history_frame = make_series(900, start="2026-01-01", seed=7)[["timestamp", "load"]]
    history_path = tmp_path / "history.csv"
    history_frame.to_csv(history_path, index=False)
    scenario = write_scenario(tmp_path, forecast_steps=STEPS)
    output = tmp_path / "online.csv"

    proc = run_predict(db, scenario, history_path, output)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    trace = json.loads(output.with_suffix(".trace.json").read_text())
    assert trace["has_interaction"] is True
    assert trace["forecast_steps"] == STEPS
    online = pd.read_csv(output)
    assert len(online) == STEPS

    # 独立重算：成员轨迹 + 由未来时间戳生成的日历特征
    store = ModelStore(str(db))
    base = {}
    for member_id, member_type in zip(trace["model_ids"], trace["member_types"]):
        row = store.get_model(member_id)
        base[member_type] = generate_member_trajectory(
            model=load_artifact(row["artifact_path"]), model_type=row["model_type"],
            required_features=row["required_features"], history=history_frame,
            forecast_steps=STEPS, country="US",
        )
    store.close()
    calendar = calendar_frame(pd.to_datetime(online["timestamp"]), "US")
    expected = predictor.predict(base, calendar)
    np.testing.assert_allclose(
        online["yhat"].to_numpy(dtype=float), expected, rtol=0, atol=1e-8
    )

    # interaction 必须真的参与了：去掉它结果就不同，否则这条用例什么也没验证
    linear_only = CombinationPredictor(
        member_ids=list(predictor.member_ids),
        linear_weights=list(predictor.linear_weights),
        strategy=predictor.strategy, interaction=None,
    ).predict(base)
    assert not np.allclose(expected, linear_only, rtol=0, atol=1e-6)
    # 且贡献随 hour 变化，不是一个常数偏置
    contribution = expected - linear_only
    assert float(np.std(contribution)) > 1e-6
    hours = pd.to_datetime(online["timestamp"]).dt.hour.to_numpy(dtype=float)
    assert abs(float(np.corrcoef(contribution, hours - INTERACTION_MEAN)[0, 1])) > 0.9
