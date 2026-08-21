import json

from src.core.index import DriftIndex, IndexManager, LatencyIndex
from src.core.index.signal_loaders import (
    build_signal_records_from_reports,
    load_drift_records_from_kg_results,
    load_latency_records_from_ablation_profile,
    load_latency_records_from_kg_results,
)


def test_load_latency_records_from_ablation_profile(tmp_path):
    path = tmp_path / "ablation_profile.json"
    path.write_text(
        json.dumps(
            {
                "pjm": {
                    "1": {
                        "resources": {
                            "simple_avg": {
                                "predict_time_ms": 2.0,
                                "train_time_ms": 1.0,
                                "rss_mb": 123.0,
                                "models_used": 3,
                            },
                            "top2_avg": {
                                "predict_time_ms": 1.0,
                                "train_time_ms": 0.5,
                                "rss_mb": 120.0,
                                "models_used": 2,
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    records = load_latency_records_from_ablation_profile(path)

    assert len(records) == 2
    assert records[0]["scenario_id"] == "pjm_h1"
    assert records[0]["model_id"] == "simple_avg"
    assert records[0]["latency_ms"] == 2.0
    assert records[0]["train_time_ms"] == 1.0
    assert records[0]["memory_mb"] == 123.0
    assert [r["model_id"] for r in LatencyIndex(records).query(scenario_id="pjm_h1")] == [
        "top2_avg",
        "simple_avg",
    ]


def test_load_latency_and_drift_records_from_kg_results(tmp_path):
    path = tmp_path / "kg_results.json"
    path.write_text(
        json.dumps(
            {
                "pjm": {
                    "1": {
                        "_meta": {
                            "runtime_protocol_b_sec": 1.25,
                            "protocol_b_solver": {
                                "trace_path": "reports/x/trace.json",
                                "trace_stages": ["CapabilityMatch", "ProtocolBBackend"],
                            },
                            "protocol_b": {
                                "guard_config": {
                                    "drift_level_effective": "medium",
                                    "drift_level_raw": "low",
                                    "drift_level_source": "psi_deadband",
                                    "drift_median_psi": 0.29,
                                }
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    latency_records = load_latency_records_from_kg_results(path)
    drift_records = load_drift_records_from_kg_results(path)

    assert latency_records == [
        {
            "scenario_id": "pjm_h1",
            "dataset": "pjm",
            "horizon": 1,
            "model_id": "kg_protocol_b",
            "strategy_id": "kg_protocol_b",
            "latency_ms": 1250.0,
            "runtime_sec": 1.25,
            "source": str(path),
            "trace_path": "reports/x/trace.json",
            "trace_stages": ["CapabilityMatch", "ProtocolBBackend"],
        }
    ]
    assert DriftIndex(drift_records).query(scenario_id="pjm_h1")[0]["drift_level"] == "medium"
    assert DriftIndex(drift_records).query(min_psi=0.2)[0]["median_psi"] == 0.29


def test_build_signal_records_from_reports_feeds_index_manager(tmp_path):
    kg_path = tmp_path / "kg_results.json"
    kg_path.write_text(
        json.dumps(
            {
                "pjm": {
                    "1": {
                        "_meta": {
                            "runtime_protocol_b_sec": 0.5,
                            "protocol_b": {
                                "guard_config": {
                                    "drift_level_effective": "high",
                                    "drift_median_psi": 0.4,
                                }
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    signals = build_signal_records_from_reports(kg_result_paths=[kg_path])
    manager = IndexManager.with_defaults(
        latency_records=signals["latency"],
        drift_records=signals["drift"],
    )

    assert manager.query("latency", scenario_id="pjm_h1")[0]["latency_ms"] == 500.0
    assert manager.query("drift", drift_level="high")[0]["median_psi"] == 0.4


def test_index_manager_can_build_with_report_signals(tmp_path):
    kg_path = tmp_path / "kg_results.json"
    kg_path.write_text(
        json.dumps(
            {
                "pjm": {
                    "1": {
                        "_meta": {
                            "runtime_protocol_b_sec": 0.75,
                            "protocol_b": {
                                "guard_config": {
                                    "drift_level_effective": "medium",
                                    "drift_median_psi": 0.25,
                                }
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manager = IndexManager.with_report_signals(kg_result_paths=[kg_path])

    assert isinstance(manager.get("latency"), LatencyIndex)
    assert isinstance(manager.get("drift"), DriftIndex)
    assert manager.query("latency", scenario_id="pjm_h1")[0]["latency_ms"] == 750.0
    assert manager.query("drift", scenario_id="pjm_h1")[0]["drift_level"] == "medium"
