from src.core.enums import ModelLifecycleStage, TaskType
from src.core.index import (
    CapabilityIndex,
    ConflictIndex,
    DriftIndex,
    EvidenceIndex,
    IndexManager,
    LatencyIndex,
    PerformanceIndex,
    ScenarioIndex,
)
from src.core.evidence import EvidenceRecord, EvidenceStore
from src.core.schema import ModelManifest


def test_index_manager_registers_and_queries_named_indices():
    manager = IndexManager()
    manager.register("performance", PerformanceIndex([
        {"path_id": "p1", "score": 0.8},
        {"path_id": "p2", "score": 0.6},
    ]))

    results = manager.query("performance", top_k=1)

    assert results == [{"path_id": "p1", "score": 0.8}]


def test_index_manager_has_phase2_default_indices():
    manager = IndexManager.with_defaults(
        scenario_records=[{"scenario_id": "s1", "business_domain": "load_forecast", "signature": {"mean_load": 1.0}}],
        performance_records=[{"path_id": "p1", "score": 1.0}],
        conflict_edges=[{"source": "a", "target": "b", "weight": 0.9, "type": "conflict"}],
        evidence_store=EvidenceStore([EvidenceRecord(scenario_id="s1", data_slice_ref="data/pjm/load.csv#2020")]),
    )

    assert manager.names() == [
        "capability",
        "conflict",
        "drift",
        "evidence",
        "latency",
        "performance",
        "scenario",
    ]
    assert isinstance(manager.get("latency"), LatencyIndex)
    assert isinstance(manager.get("drift"), DriftIndex)
    assert manager.query("scenario", signature={"mean_load": 1.0})[0]["scenario_id"] == "s1"
    assert manager.query("performance")[0]["path_id"] == "p1"
    assert manager.query("conflict", model_id="a")[0]["target"] == "b"
    assert manager.query("evidence", scenario_id="s1")[0].data_slice_ref.startswith("data/pjm")
    assert manager.query("latency") == []
    assert manager.query("drift") == []


def test_scenario_index_filters_domain_and_sorts_by_similarity():
    idx = ScenarioIndex([
        {"scenario_id": "s1", "business_domain": "load_forecast", "signature": {"mean_load": 100.0}},
        {"scenario_id": "s2", "business_domain": "load_forecast", "signature": {"mean_load": 200.0}},
        {"scenario_id": "s3", "business_domain": "risk", "signature": {"mean_load": 100.0}},
    ])

    results = idx.query(
        signature={"mean_load": 110.0},
        business_domain="load_forecast",
        top_k=2,
    )

    assert [r["scenario_id"] for r in results] == ["s1", "s2"]
    assert all(r["business_domain"] == "load_forecast" for r in results)


def test_scenario_index_can_use_power_scenario_analyzer_style_similarity():
    idx = ScenarioIndex([
        {"scenario_id": "same_region", "business_domain": "load_forecast", "signature": {"mean_load": 100.0, "region_type": 1.0}},
        {"scenario_id": "different_region", "business_domain": "load_forecast", "signature": {"mean_load": 100.0, "region_type": 2.0}},
    ])

    results = idx.query(
        signature={"mean_load": 100.0, "region_type": 1.0},
        business_domain="load_forecast",
    )

    assert [r["scenario_id"] for r in results] == ["same_region", "different_region"]
    assert results[0]["_score"] > results[1]["_score"]


def test_scenario_index_can_load_historical_scenarios_tuple_shape():
    idx = ScenarioIndex.from_historical_scenarios([
        ("s1", {"mean_load": 10.0}, {"score": 0.9}),
        ("s2", {"mean_load": 20.0}, {"score": 0.8}),
    ], business_domain="load_forecast")

    results = idx.query(signature={"mean_load": 10.0}, business_domain="load_forecast")

    assert results[0]["scenario_id"] == "s1"


def test_capability_index_matches_task_domain_and_lifecycle():
    manifests = {
        "active_forecast": ModelManifest(
            model_id="active_forecast",
            task_types=[TaskType.FORECASTING],
            business_domains=["load_forecast"],
            lifecycle_stage=ModelLifecycleStage.ACTIVE,
        ),
        "shadow_forecast": ModelManifest(
            model_id="shadow_forecast",
            task_types=[TaskType.FORECASTING],
            business_domains=["load_forecast"],
            lifecycle_stage=ModelLifecycleStage.SHADOW,
        ),
        "risk_model": ModelManifest(
            model_id="risk_model",
            task_types=[TaskType.CLASSIFICATION],
            business_domains=["fee_recovery_risk"],
            lifecycle_stage=ModelLifecycleStage.ACTIVE,
        ),
    }
    idx = CapabilityIndex(manifests)

    results = idx.query(task_type=TaskType.FORECASTING, business_domain="load_forecast")

    assert [m.model_id for m in results] == ["active_forecast"]


def test_capability_index_can_filter_by_available_features():
    manifests = {
        "needs_weather": ModelManifest(
            model_id="needs_weather",
            task_types=[TaskType.FORECASTING],
            business_domains=["load_forecast"],
            input_constraints={"features": ["load", "weather"]},
            lifecycle_stage=ModelLifecycleStage.ACTIVE,
        ),
        "needs_load": ModelManifest(
            model_id="needs_load",
            task_types=[TaskType.FORECASTING],
            business_domains=["load_forecast"],
            input_constraints={"features": ["load"]},
            lifecycle_stage=ModelLifecycleStage.ACTIVE,
        ),
    }
    idx = CapabilityIndex(manifests)

    results = idx.query(
        task_type=TaskType.FORECASTING,
        business_domain="load_forecast",
        available_features={"load"},
    )

    assert [m.model_id for m in results] == ["needs_load"]


def test_conflict_index_filters_model_and_weight():
    idx = ConflictIndex([
        {"source": "xgboost_reg", "target": "lgbm_reg", "weight": 0.91, "type": "conflict"},
        {"source": "xgboost_reg", "target": "prophet", "weight": 0.2, "type": "conflict"},
    ])

    results = idx.query(model_id="xgboost_reg", min_weight=0.9)

    assert results == [{"source": "xgboost_reg", "target": "lgbm_reg", "weight": 0.91, "type": "conflict"}]


def test_performance_index_can_load_historical_scenarios_tuple_shape():
    idx = PerformanceIndex.from_historical_scenarios([
        ("s1", {"mean_load": 10.0}, {"path_id": "p1", "score": 0.9}),
        ("s2", {"mean_load": 20.0}, {"path_id": "p2", "score": 0.8}),
    ])

    assert idx.query(scenario_id="s1")[0]["path_id"] == "p1"


def test_evidence_index_wraps_evidence_store():
    store = EvidenceStore([
        EvidenceRecord(scenario_id="s1", data_slice_ref="data/pjm/load.csv#2020-01", drift_events=["psi_high"]),
        EvidenceRecord(scenario_id="s2", data_slice_ref="data/aemo/load.csv#2020-01", drift_events=[]),
    ])
    idx = EvidenceIndex(store)

    results = idx.query(scenario_id="s1")
    assert len(results) == 1
    assert results[0].scenario_id == "s1"

    drift_results = idx.query(drift_event="psi_high")
    assert [r.scenario_id for r in drift_results] == ["s1"]

    combined_results = idx.query(scenario_id="s1", drift_event="psi_high", data_slice_prefix="data/pjm")
    assert [r.scenario_id for r in combined_results] == ["s1"]


def test_latency_index_filters_and_sorts_by_latency():
    idx = LatencyIndex([
        {"model_id": "m1", "scenario_id": "s1", "latency_ms": 80.0},
        {"model_id": "m2", "scenario_id": "s1", "latency_ms": 40.0},
        {"model_id": "m3", "scenario_id": "s2", "latency_ms": 20.0},
    ])

    results = idx.query(scenario_id="s1", max_latency_ms=100.0)

    assert [r["model_id"] for r in results] == ["m2", "m1"]


def test_drift_index_filters_by_level_and_psi():
    idx = DriftIndex([
        {"model_id": "m1", "scenario_id": "s1", "drift_level": "low", "median_psi": 0.05},
        {"model_id": "m2", "scenario_id": "s1", "drift_level": "high", "median_psi": 0.32},
        {"model_id": "m3", "scenario_id": "s2", "drift_level": "high", "median_psi": 0.42},
    ])

    results = idx.query(scenario_id="s1", drift_level="high", min_psi=0.2)

    assert [r["model_id"] for r in results] == ["m2"]
