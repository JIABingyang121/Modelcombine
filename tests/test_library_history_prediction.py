"""V2：用户只提交历史负荷，模型库递归生成未来 720 小时预测。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.models.artifacts import save_artifact
from src.models.combination_predictor import CombinationPredictor
from src.pipeline.library_prediction import _match_scenario
from src.storage.model_store import ModelStore


REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_COLUMNS = [
    "hour", "dayofweek", "month", "is_weekend", "is_holiday",
    "lag_1", "lag_2", "lag_6", "lag_12", "lag_24",
    "roll3_mean", "roll3_std", "roll6_mean", "roll6_std",
    "roll12_mean", "roll12_std", "roll24_mean", "roll24_std",
]


def _training_frame() -> tuple[pd.DataFrame, pd.Series]:
    ts = pd.date_range("2025-01-01", periods=240, freq="h")
    load = 100 + 15 * np.sin(np.arange(len(ts)) * 2 * np.pi / 24)
    frame = pd.DataFrame({"timestamp": ts, "load": load})
    frame["hour"] = ts.hour
    frame["dayofweek"] = ts.dayofweek
    frame["month"] = ts.month
    frame["is_weekend"] = ts.dayofweek.isin([5, 6]).astype(int)
    frame["is_holiday"] = 0
    for lag in (1, 2, 6, 12, 24):
        frame[f"lag_{lag}"] = frame["load"].shift(lag)
    for window in (3, 6, 12, 24):
        frame[f"roll{window}_mean"] = frame["load"].shift(1).rolling(window).mean()
        frame[f"roll{window}_std"] = frame["load"].shift(1).rolling(window).std()
    frame = frame.dropna().reset_index(drop=True)
    return frame[FEATURE_COLUMNS], frame["load"]


def _build_h1_library(tmp_path: Path) -> tuple[Path, int]:
    db = tmp_path / "library.sqlite3"
    store = ModelStore(str(db))
    store.create_schema()
    X, y = _training_frame()
    artifacts = tmp_path / "artifacts"
    members = []
    for index, alpha in enumerate((0.1, 1.0), 1):
        model = Ridge(alpha=alpha).fit(X, y)
        model_id = f"pjm__h1__m{index}"
        path = save_artifact(model, artifacts / f"{model_id}.pkl")
        store.add_model(
            model_id=model_id,
            model_type="ridge",
            task_type="load_forecast",
            artifact_path=str(path),
            required_features=FEATURE_COLUMNS,
            model_params={"alpha": alpha},
            lifecycle_stage="active",
        )
        members.append((model_id, index - 1, 0.5))
    predictor = CombinationPredictor(
        member_ids=["m1", "m2"], linear_weights=[0.5, 0.5], strategy="linear",
    )
    combo_path = save_artifact(predictor, artifacts / "pjm__h1__combo.pkl")
    combo_id = store.add_combination("protocol_b_combination", str(combo_path), members)
    store.add_scenario(
        scenario_id="pjm_h1_reference",
        task_type="load_forecast",
        business_domain="power_load",
        region="pjm",
        horizon=1,
        freq="h",
        signature={"horizon": 1.0, "y_mean": 100.0, "y_std": 15.0},
    )
    profile_id = store.add_data_profile(
        scenario_id="pjm_h1_reference",
        data_ref="data/pjm",
        target_column="load",
        features=FEATURE_COLUMNS,
        sample_count=216,
        start_at="2025-01-01T00:00:00",
        end_at="2025-01-10T00:00:00",
        signature={"horizon": 1.0, "y_mean": 100.0, "y_std": 15.0},
    )
    relation_id = store.add_relation(
        "pjm_h1_reference", profile_id, combo_id, validation_mae=1.0,
    )
    store.close()
    return db, relation_id


def test_history_input_generates_30_day_forecast_and_accepts_feedback(tmp_path):
    db, relation_id = _build_h1_library(tmp_path)
    ts = pd.date_range("2026-01-01", periods=72, freq="h")
    history = tmp_path / "history.csv"
    pd.DataFrame({
        "timestamp": ts,
        "load": 100 + 15 * np.sin(np.arange(len(ts)) * 2 * np.pi / 24),
    }).to_csv(history, index=False)
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({
        "task_type": "load_forecast",
        "business_domain": "power_load",
        "region": "pjm",
        "freq": "h",
    }), encoding="utf-8")
    output = tmp_path / "forecast.csv"

    predict = subprocess.run(
        [
            sys.executable, "run.py", "predict", "--database", str(db),
            "--scenario", str(scenario), "--history", str(history),
            "--output", str(output),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )

    assert predict.returncode == 0, predict.stdout + predict.stderr
    forecast = pd.read_csv(output)
    assert list(forecast.columns) == ["timestamp", "yhat"]
    assert len(forecast) == 720
    assert forecast["yhat"].notna().all()
    trace = json.loads(output.with_suffix(".trace.json").read_text())
    assert trace["scenario_id"] == "pjm_h1_reference"
    assert trace["relation_id"] == relation_id
    assert trace["selector_invoked"] is False
    assert trace["forecast_steps"] == 720
    assert trace["signature_source"] == "history"

    actual = tmp_path / "actual.csv"
    pd.DataFrame({"timestamp": forecast["timestamp"], "y": forecast["yhat"]}).to_csv(actual, index=False)
    feedback = subprocess.run(
        [
            sys.executable, "run.py", "feedback", "--database", str(db),
            "--prediction-run-id", str(trace["prediction_run_id"]), "--actual", str(actual),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert feedback.returncode == 0, feedback.stdout + feedback.stderr


def test_matching_keeps_user_region_as_a_hard_constraint(tmp_path):
    db, _ = _build_h1_library(tmp_path)
    store = ModelStore(str(db))
    store.add_scenario(
        scenario_id="aemo_h1_reference",
        task_type="load_forecast",
        business_domain="power_load",
        region="aemo_vic",
        horizon=1,
        freq="h",
        signature={"horizon": 1.0, "y_mean": 100.0, "y_std": 15.0},
    )

    scenario_id, _ = _match_scenario(store, {
        "task_type": "load_forecast",
        "business_domain": "power_load",
        "region": "pjm",
        "horizon": 1,
        "freq": "h",
        "signature": {"horizon": 1.0, "y_mean": 100.0, "y_std": 15.0},
    })

    store.close()
    assert scenario_id == "pjm_h1_reference"
