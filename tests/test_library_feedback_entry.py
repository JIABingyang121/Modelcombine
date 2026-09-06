"""在线反馈入口 `run.py feedback`（SQLite 模型库 Task 6）。

- 真实 run.py predict 产生 prediction run，再 run.py feedback 用真实值更新；
- 核对真实 MAE、反馈次数与累计平均；
- 同一 run 第二次反馈必须失败；
- 多个历史关系时，后续匹配读取更新后的实际表现，而不是仅看计数 / use_count。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.storage.model_store import ModelStore
from tests.test_library_prediction_entry import (
    REPO_ROOT,
    _build_library,
    _future_features,
    _scenario_json,
)


def _predict(db, scenario, features, output):
    return subprocess.run(
        [sys.executable, "run.py", "predict", "--database", str(db),
         "--scenario", str(scenario), "--features", str(features), "--output", str(output)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def _feedback(db, run_id, actual):
    return subprocess.run(
        [sys.executable, "run.py", "feedback", "--database", str(db),
         "--prediction-run-id", str(run_id), "--actual", str(actual)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def _actual_csv(tmp_path, timestamps, y, name="actual.csv"):
    path = tmp_path / name
    pd.DataFrame({"timestamp": timestamps, "y": np.asarray(y, dtype=float)}).to_csv(path, index=False)
    return path


def test_feedback_records_real_mae_and_running_mean(tmp_path):
    lib = _build_library(tmp_path)
    features = _future_features(tmp_path)
    scenario = _scenario_json(tmp_path)
    out = tmp_path / "predictions.csv"

    proc = _predict(lib["db"], scenario, features, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    trace = json.loads((tmp_path / "predictions.trace.json").read_text())
    run_id = trace["prediction_run_id"]

    preds = pd.read_csv(out)
    actual = _actual_csv(tmp_path, preds["timestamp"], preds["yhat"].to_numpy() + 4.0)

    fb = _feedback(lib["db"], run_id, actual)
    assert fb.returncode == 0, fb.stdout + fb.stderr

    store = ModelStore(str(lib["db"]))
    run_row = store.get_prediction_run(run_id)
    assert run_row["actual_mae"] == pytest.approx(4.0, abs=1e-6)
    assert run_row["feedback_at"] is not None
    rel = store.get_relation(trace["relation_id"])
    assert rel["feedback_count"] == 1
    assert rel["mean_actual_mae"] == pytest.approx(4.0, abs=1e-6)

    # 第二轮 predict + feedback -> 精确累计平均
    out2 = tmp_path / "predictions2.csv"
    assert _predict(lib["db"], scenario, features, out2).returncode == 0
    trace2 = json.loads((tmp_path / "predictions2.trace.json").read_text())
    preds2 = pd.read_csv(out2)
    actual2 = _actual_csv(tmp_path, preds2["timestamp"], preds2["yhat"].to_numpy() + 10.0, "actual2.csv")
    assert _feedback(lib["db"], trace2["prediction_run_id"], actual2).returncode == 0
    rel = store.get_relation(trace["relation_id"])
    assert rel["feedback_count"] == 2
    assert rel["mean_actual_mae"] == pytest.approx(7.0, abs=1e-6)
    store.close()


def test_feedback_aligns_duplicate_timestamps_by_row_id(tmp_path):
    lib = _build_library(tmp_path)
    features_path = _future_features(tmp_path)
    features = pd.read_csv(features_path)
    features.loc[1, "timestamp"] = features.loc[0, "timestamp"]
    features.insert(0, "row_id", [f"row_{i}" for i in range(len(features))])
    features.to_csv(features_path, index=False)
    scenario = _scenario_json(tmp_path)
    out = tmp_path / "predictions.csv"

    proc = _predict(lib["db"], scenario, features_path, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    trace = json.loads((tmp_path / "predictions.trace.json").read_text())
    preds = pd.read_csv(out)
    assert preds.columns.tolist() == ["row_id", "timestamp", "yhat"]
    assert preds["timestamp"].duplicated().any()

    actual = tmp_path / "actual.csv"
    pd.DataFrame(
        {
            "row_id": preds["row_id"],
            "timestamp": preds["timestamp"],
            "y": preds["yhat"].to_numpy(float) + 4.0,
        }
    ).to_csv(actual, index=False)
    fb = _feedback(lib["db"], trace["prediction_run_id"], actual)

    assert fb.returncode == 0, fb.stdout + fb.stderr
    store = ModelStore(str(lib["db"]))
    assert store.get_prediction_run(trace["prediction_run_id"])["actual_mae"] == pytest.approx(4.0)
    store.close()


def test_second_feedback_for_same_run_fails(tmp_path):
    lib = _build_library(tmp_path)
    features = _future_features(tmp_path)
    scenario = _scenario_json(tmp_path)
    out = tmp_path / "predictions.csv"
    assert _predict(lib["db"], scenario, features, out).returncode == 0
    trace = json.loads((tmp_path / "predictions.trace.json").read_text())
    run_id = trace["prediction_run_id"]
    preds = pd.read_csv(out)
    actual = _actual_csv(tmp_path, preds["timestamp"], preds["yhat"].to_numpy())

    assert _feedback(lib["db"], run_id, actual).returncode == 0
    second = _feedback(lib["db"], run_id, actual)
    assert second.returncode != 0

    store = ModelStore(str(lib["db"]))
    assert store.get_relation(trace["relation_id"])["feedback_count"] == 1
    store.close()


def test_timestamp_misalignment_is_rejected(tmp_path):
    lib = _build_library(tmp_path)
    features = _future_features(tmp_path)
    scenario = _scenario_json(tmp_path)
    out = tmp_path / "predictions.csv"
    assert _predict(lib["db"], scenario, features, out).returncode == 0
    trace = json.loads((tmp_path / "predictions.trace.json").read_text())
    preds = pd.read_csv(out)
    short = _actual_csv(tmp_path, preds["timestamp"].iloc[:-5], preds["yhat"].to_numpy()[:-5])

    proc = _feedback(lib["db"], trace["prediction_run_id"], short)
    assert proc.returncode != 0
    store = ModelStore(str(lib["db"]))
    assert store.get_relation(trace["relation_id"])["feedback_count"] == 0
    store.close()


def _seed_second_relation(db, scenario_id, *, validation_mae, signature=None):
    """在同一 scenario 下追加第二个组合关系，复用已有 combination。

    ``signature`` 默认与既有关系的数据画像**完全相同**——只有画像无法区分时，
    反馈统计才会成为 tie-break，这正是本文件要覆盖的情形。
    """
    store = ModelStore(str(db))
    combo_id = store.connection.execute(
        "SELECT combination_id FROM combinations LIMIT 1"
    ).fetchone()[0]
    profile_id = store.add_data_profile(
        scenario_id=scenario_id, data_ref="data/alt", target_column="y",
        features=["hour", "dow", "temp"], sample_count=200,
        start_at="2024-01-01T00:00:00+00:00", end_at="2024-06-01T00:00:00+00:00",
        signature=signature or {
            "y_mean": 100.0, "y_std": 15.0, "y_cv": 0.15,
            "mean_load": 100.0, "cv_load": 0.15,
        },
    )
    rel_id = store.add_relation(scenario_id, profile_id, combo_id, validation_mae=validation_mae)
    store.close()
    return rel_id


def test_feedback_breaks_ties_only_when_data_profiles_are_indistinguishable(tmp_path):
    """在线匹配的第一顺位是数据画像相似度；画像完全并列时才看反馈统计。

    这里两条关系的数据画像**一模一样**，用户数据对二者的相似度必然相等，因此
    tie-break 生效：有反馈的按 mean_actual_mae 升序，use_count 不参与。反馈通过
    run.py feedback 改善后，后续匹配随之改变。
    """
    lib = _build_library(tmp_path)
    features = _future_features(tmp_path)
    scenario = _scenario_json(tmp_path)

    rel_hi = lib["rel_a"]          # validation_mae = 5.0
    rel_lo = _seed_second_relation(lib["db"], "pjm_h24_A", validation_mae=2.0)

    store = ModelStore(str(lib["db"]))
    # 历史累计：validation 更优的 rel_lo 实际表现反而更差
    run_hi = store.record_prediction_run(rel_hi, "seed_hi.csv")
    store.record_feedback(run_hi, actual_mae=8.0)
    run_lo = store.record_prediction_run(rel_lo, "seed_lo.csv")
    store.record_feedback(run_lo, actual_mae=9.0)
    # use_count 不是质量分：把 rel_hi 的使用次数拉高
    for _ in range(5):
        store.record_prediction_run(rel_hi, "noise.csv")
    store.close()

    out1 = tmp_path / "p1.csv"
    assert _predict(lib["db"], scenario, features, out1).returncode == 0
    t1 = json.loads((tmp_path / "p1.trace.json").read_text())
    assert t1["relation_id"] == rel_hi  # mean_actual_mae 8 < 9，不看 validation_mae / use_count

    # 通过 run.py feedback 把 rel_lo 的实际表现显著改善
    good_pred = tmp_path / "good_pred.csv"
    ts = pd.read_csv(out1)["timestamp"]
    pd.DataFrame({"timestamp": ts, "yhat": np.full(len(ts), 100.0)}).to_csv(good_pred, index=False)
    store = ModelStore(str(lib["db"]))
    run_lo2 = store.record_prediction_run(rel_lo, str(good_pred))
    store.close()
    good_actual = _actual_csv(tmp_path, ts, np.full(len(ts), 100.5), "good_actual.csv")
    assert _feedback(lib["db"], run_lo2, good_actual).returncode == 0

    store = ModelStore(str(lib["db"]))
    assert store.get_relation(rel_lo)["mean_actual_mae"] == pytest.approx((9.0 + 0.5) / 2, abs=1e-6)
    store.close()

    out2 = tmp_path / "p2.csv"
    assert _predict(lib["db"], scenario, features, out2).returncode == 0
    t2 = json.loads((tmp_path / "p2.trace.json").read_text())
    assert t2["relation_id"] == rel_lo  # 4.75 < 8，匹配随反馈更新而改变


def test_data_similarity_outranks_feedback_when_profiles_differ(tmp_path):
    """数据画像能区分时，相似度说了算，反馈不得把匹配拉回另一条关系。

    给第二条关系一个与用户数据明显不同量级的画像，并让它的反馈表现远好于第一条；
    匹配仍必须落在画像更接近的第一条上。
    """
    lib = _build_library(tmp_path)
    features = _future_features(tmp_path)
    scenario = _scenario_json(tmp_path)

    rel_near = lib["rel_a"]
    rel_far = _seed_second_relation(
        lib["db"], "pjm_h24_A", validation_mae=0.1,
        signature={"y_mean": 5000.0, "y_std": 900.0, "y_cv": 0.18,
                   "mean_load": 5000.0, "cv_load": 0.18},
    )

    store = ModelStore(str(lib["db"]))
    run_near = store.record_prediction_run(rel_near, "near.csv")
    store.record_feedback(run_near, actual_mae=50.0)     # 近的那条反馈很差
    run_far = store.record_prediction_run(rel_far, "far.csv")
    store.record_feedback(run_far, actual_mae=0.1)       # 远的那条反馈极好
    store.close()

    out = tmp_path / "p.csv"
    assert _predict(lib["db"], scenario, features, out).returncode == 0
    trace = json.loads((tmp_path / "p.trace.json").read_text())

    assert trace["relation_id"] == rel_near
    assert trace["data_similarity"] > 0.0
