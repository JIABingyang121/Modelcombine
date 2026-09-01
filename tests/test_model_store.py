import sqlite3

import pytest

from src.storage.model_store import FeedbackAlreadyRecorded, ModelStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "modelcombine.sqlite3"
    s = ModelStore(str(db_path))
    s.create_schema()
    try:
        yield s
    finally:
        s.close()


def _seed_models(store, model_ids):
    for mid in model_ids:
        store.add_model(
            model_id=mid,
            model_type="xgboost_reg",
            task_type="load_forecast",
            artifact_path=f"/artifacts/{mid}.pkl",
            required_features=["hour", "dow", "lag_24"],
            model_params={"n_estimators": 200, "max_depth": 6},
            lifecycle_stage="active",
        )


def _seed_scenario_and_profile(store, scenario_id="pjm_h1"):
    store.add_scenario(
        scenario_id=scenario_id,
        task_type="load_forecast",
        business_domain="power_load",
        region="pjm",
        horizon=1,
        freq="h",
        signature={"mean_load": 100.0, "cv_load": 0.2},
    )
    profile_id = store.add_data_profile(
        scenario_id=scenario_id,
        data_ref="data/features/pjm",
        target_column="load",
        features=["hour", "dow", "lag_24"],
        sample_count=21780,
        start_at="2024-01-01T00:00:00+00:00",
        end_at="2024-12-31T23:00:00+00:00",
        signature={"mean_load": 100.0},
    )
    return scenario_id, profile_id


def test_schema_creates_all_planned_tables(store):
    names = {
        row[0]
        for row in store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert {
        "models",
        "scenarios",
        "data_profiles",
        "combinations",
        "combination_members",
        "scenario_data_combinations",
        "prediction_runs",
    } <= names


def test_create_schema_is_idempotent(store):
    store.create_schema()  # second call must not raise
    _seed_models(store, ["m_a"])
    assert store.get_model("m_a")["model_params"]["max_depth"] == 6


def test_three_member_combination_round_trips_in_member_order(store):
    _seed_models(store, ["m_a", "m_b", "m_c"])
    combo_id = store.add_combination(
        strategy="protocol_b_ridge",
        artifact_path="/artifacts/combo_1.pkl",
        # 传入顺序与 member_order 不一致，读回必须按 member_order 排序
        members=[("m_a", 1, 0.3), ("m_c", 0, 0.5), ("m_b", 2, 0.2)],
    )
    combo = store.get_combination(combo_id)
    assert combo["strategy"] == "protocol_b_ridge"
    assert combo["artifact_path"] == "/artifacts/combo_1.pkl"
    assert [m["model_id"] for m in combo["members"]] == ["m_c", "m_a", "m_b"]
    assert [m["member_order"] for m in combo["members"]] == [0, 1, 2]
    assert combo["members"][0]["weight"] == pytest.approx(0.5)


def test_single_member_combination_is_allowed(store):
    _seed_models(store, ["m_solo"])
    combo_id = store.add_combination("best_single", "/c.pkl", [("m_solo", 0, 1.0)])
    assert [m["model_id"] for m in store.get_combination(combo_id)["members"]] == ["m_solo"]


def test_foreign_key_enforced_on_combination_member_and_parent_rolls_back(store):
    _seed_models(store, ["m_a"])
    with pytest.raises(sqlite3.IntegrityError):
        store.add_combination(
            strategy="s",
            artifact_path="/x.pkl",
            members=[("m_a", 0, 0.6), ("missing_model", 1, 0.4)],
        )
    assert (
        store.connection.execute("SELECT COUNT(*) FROM combinations").fetchone()[0] == 0
    )
    assert (
        store.connection.execute(
            "SELECT COUNT(*) FROM combination_members"
        ).fetchone()[0]
        == 0
    )


def test_relation_unique_constraint_rejects_duplicate_triple(store):
    _seed_models(store, ["m_a", "m_b"])
    combo_id = store.add_combination("s", "/c.pkl", [("m_a", 0, 0.5), ("m_b", 1, 0.5)])
    sid, pid = _seed_scenario_and_profile(store)
    store.add_relation(sid, pid, combo_id, validation_mae=12.5, test_mae=13.0)
    with pytest.raises(sqlite3.IntegrityError):
        store.add_relation(sid, pid, combo_id, validation_mae=99.0)


def test_relation_transaction_failure_leaves_no_partial_row(store):
    sid, pid = _seed_scenario_and_profile(store)
    with pytest.raises(sqlite3.IntegrityError):
        store.add_relation(sid, pid, combination_id=999_999, validation_mae=1.0)
    assert (
        store.connection.execute(
            "SELECT COUNT(*) FROM scenario_data_combinations"
        ).fetchone()[0]
        == 0
    )


def test_prediction_run_bumps_use_count_and_last_used(store):
    _seed_models(store, ["m_a", "m_b"])
    combo_id = store.add_combination("s", "/c.pkl", [("m_a", 0, 0.5), ("m_b", 1, 0.5)])
    sid, pid = _seed_scenario_and_profile(store)
    rel_id = store.add_relation(sid, pid, combo_id, validation_mae=10.0)
    assert store.get_relation(rel_id)["use_count"] == 0

    store.record_prediction_run(rel_id, prediction_ref="out/pred_1.csv")
    store.record_prediction_run(rel_id, prediction_ref="out/pred_2.csv")

    rel = store.get_relation(rel_id)
    assert rel["use_count"] == 2
    assert rel["last_used_at"] is not None


def test_feedback_running_mean_is_exact(store):
    _seed_models(store, ["m_a", "m_b"])
    combo_id = store.add_combination("s", "/c.pkl", [("m_a", 0, 0.5), ("m_b", 1, 0.5)])
    sid, pid = _seed_scenario_and_profile(store)
    rel_id = store.add_relation(sid, pid, combo_id, validation_mae=10.0)
    run1 = store.record_prediction_run(rel_id, prediction_ref="p1.csv")
    run2 = store.record_prediction_run(rel_id, prediction_ref="p2.csv")
    run3 = store.record_prediction_run(rel_id, prediction_ref="p3.csv")

    store.record_feedback(run1, actual_mae=12.0)
    rel = store.get_relation(rel_id)
    assert rel["feedback_count"] == 1
    assert rel["mean_actual_mae"] == pytest.approx(12.0)

    store.record_feedback(run2, actual_mae=18.0)
    rel = store.get_relation(rel_id)
    assert rel["feedback_count"] == 2
    assert rel["mean_actual_mae"] == pytest.approx(15.0)

    store.record_feedback(run3, actual_mae=6.0)
    rel = store.get_relation(rel_id)
    assert rel["feedback_count"] == 3
    assert rel["mean_actual_mae"] == pytest.approx(12.0)

    assert store.get_prediction_run(run1)["actual_mae"] == pytest.approx(12.0)
    assert store.get_prediction_run(run1)["feedback_at"] is not None


def test_second_feedback_for_same_run_fails_and_does_not_change_stats(store):
    _seed_models(store, ["m_a", "m_b"])
    combo_id = store.add_combination("s", "/c.pkl", [("m_a", 0, 0.5), ("m_b", 1, 0.5)])
    sid, pid = _seed_scenario_and_profile(store)
    rel_id = store.add_relation(sid, pid, combo_id, validation_mae=10.0)
    run1 = store.record_prediction_run(rel_id, prediction_ref="p1.csv")
    store.record_feedback(run1, actual_mae=12.0)

    with pytest.raises(FeedbackAlreadyRecorded):
        store.record_feedback(run1, actual_mae=5.0)

    rel = store.get_relation(rel_id)
    assert rel["feedback_count"] == 1
    assert rel["mean_actual_mae"] == pytest.approx(12.0)


def test_json_columns_round_trip_with_sorted_keys(store):
    _seed_models(store, ["m_a"])
    model = store.get_model("m_a")
    assert model["required_features"] == ["hour", "dow", "lag_24"]
    assert model["model_params"] == {"n_estimators": 200, "max_depth": 6}
    raw = store.connection.execute(
        "SELECT model_params_json FROM models WHERE model_id = 'm_a'"
    ).fetchone()[0]
    assert raw == '{"max_depth": 6, "n_estimators": 200}'
