from src.core.evidence import EvidenceRecord, EvidenceStore


def test_evidence_record_generates_stable_id():
    record = EvidenceRecord(
        scenario_id="scenario_1",
        data_slice_ref="data/pjm/load.csv#2020-01..2020-03",
        residual_summary={"rmse": 12.3},
        drift_events=["psi_high"],
    )

    same = EvidenceRecord(
        scenario_id="scenario_1",
        data_slice_ref="data/pjm/load.csv#2020-01..2020-03",
        residual_summary={"rmse": 12.3},
        drift_events=["psi_high"],
    )

    assert record.evidence_id == same.evidence_id
    assert record.evidence_id.startswith("ev_")


def test_evidence_store_add_query_and_json_roundtrip(tmp_path):
    store = EvidenceStore()
    record = EvidenceRecord(
        scenario_id="scenario_1",
        data_slice_ref="data/pjm/load.csv#2020-01..2020-03",
        feature_snapshot_ref="reports/features/scenario_1.parquet",
        training_log_ref="reports/logs/scenario_1.log",
        residual_summary={"rmse": 12.3},
    )

    evidence_id = store.add(record)
    assert store.get(evidence_id) == record
    assert store.query_by_scenario("scenario_1") == [record]
    assert store.query_by_scenario("missing") == []

    path = tmp_path / "evidence.json"
    store.save_json(path)
    restored = EvidenceStore.load_json(path)

    assert restored.get(evidence_id) == record


def test_evidence_store_query_by_drift_event_and_data_slice_prefix():
    store = EvidenceStore([
        EvidenceRecord(
            scenario_id="s1",
            data_slice_ref="data/pjm/load.csv#2020-01..2020-03",
            drift_events=["psi_high"],
        ),
        EvidenceRecord(
            scenario_id="s2",
            data_slice_ref="data/aemo/load.csv#2020-01..2020-03",
            drift_events=["ks_high"],
        ),
    ])

    assert [r.scenario_id for r in store.query_by_drift_event("psi_high")] == ["s1"]
    assert [r.scenario_id for r in store.query_by_data_slice("data/pjm/load.csv")] == ["s1"]


def test_evidence_store_load_missing_returns_empty(tmp_path):
    missing = tmp_path / "missing.json"
    store = EvidenceStore.load_json(missing)

    assert store.to_list() == []
