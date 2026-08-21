from src.core.trace import SelectionTrace


def test_selection_trace_records_stages_rejections_and_final_choice():
    trace = SelectionTrace(scenario_id="scenario_1", timestamp="2026-07-01T00:00:00Z")

    trace.add_stage(
        "SimilarScenarioRetrieve",
        inputs={"top_k": 3},
        outputs={"candidates": ["s0", "s1"]},
        duration_ms=12.5,
    )
    trace.consider(["path_good", "path_bad"])
    trace.reject("path_bad", "max_latency exceeded")
    trace.set_final(["xgboost_reg", "lgbm_reg"], {"xgboost_reg": 0.6, "lgbm_reg": 0.4})
    trace.add_evidence_ref("evidence_1")

    assert trace.stages[0]["stage"] == "SimilarScenarioRetrieve"
    assert trace.candidates_considered == ["path_good", "path_bad"]
    assert trace.candidates_rejected == {"path_bad": "max_latency exceeded"}
    assert trace.final_selection == ["xgboost_reg", "lgbm_reg"]
    assert trace.evidence_refs == ["evidence_1"]


def test_selection_trace_roundtrip_dict():
    trace = SelectionTrace(scenario_id="scenario_1", timestamp="2026-07-01T00:00:00Z")
    trace.add_stage("TaskMatch", outputs={"task_type": "forecasting"})
    trace.set_final(["seasonal_naive"], {"seasonal_naive": 1.0})

    restored = SelectionTrace.from_dict(trace.to_dict())

    assert restored.scenario_id == trace.scenario_id
    assert restored.stages == trace.stages
    assert restored.final_selection == ["seasonal_naive"]


def test_selection_trace_json_roundtrip(tmp_path):
    trace = SelectionTrace(scenario_id="scenario_1", timestamp="2026-07-01T00:00:00Z")
    trace.consider(["p1"])
    trace.reject("p2", "conflict edge")
    trace.set_final(["p1"], {"p1": 1.0})

    path = tmp_path / "trace.json"
    trace.save_json(path)
    restored = SelectionTrace.load_json(path)

    assert restored.to_dict() == trace.to_dict()
