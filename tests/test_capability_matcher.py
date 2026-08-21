from src.core.capability_matcher import CapabilityMatcher, MatchResult
from src.core.enums import ModelLifecycleStage, TaskType
from src.core.index import IndexManager
from src.core.schema import DataContract, ModelManifest, ScenarioDefinition
from src.core.trace import SelectionTrace
from src.graph.model_graph import ModelGraph
from src.graph.relations import add_substitute_relation


def _dc():
    return DataContract(
        required_columns={"load": "float"},
        freq="H",
        min_samples=50,
        business_domain="load_forecast",
    )


def _scenario(features):
    return ScenarioDefinition(
        task_type=TaskType.FORECASTING,
        business_domain="load_forecast",
        data_contract=_dc(),
        target_schema={"yhat": "float"},
        primary_metric="MAE",
        signature_features=list(features),
        signature={k: 1.0 for k in features},
        region="PJME",
    )


def _manifest(
    mid,
    task=TaskType.FORECASTING,
    domain="load_forecast",
    feats=None,
    stage=ModelLifecycleStage.ACTIVE,
    out=None,
):
    return ModelManifest(
        model_id=mid,
        task_types=[task],
        business_domains=[domain],
        input_constraints={"features": list(feats or [])},
        output_schema=out if out is not None else {"yhat": "float"},
        resource_cost={},
        lifecycle_stage=stage,
    )


def test_eligible_model_passes():
    manifests = {"m_ok": _manifest("m_ok", feats=["load"])}
    cm = CapabilityMatcher(manifests)

    res = cm.match(_scenario({"load"}), available_features={"load"})

    assert isinstance(res, MatchResult)
    assert res.eligible == ["m_ok"]
    assert res.rejected == {}
    assert res.manifests["m_ok"].model_id == "m_ok"


def test_reject_task_type_mismatch():
    manifests = {"m_clf": _manifest("m_clf", task=TaskType.CLASSIFICATION, feats=["load"])}
    cm = CapabilityMatcher(manifests)

    res = cm.match(_scenario({"load"}), available_features={"load"})

    assert "m_clf" not in res.eligible
    assert "task_type" in res.rejected["m_clf"]


def test_reject_missing_features():
    manifests = {"m_wx": _manifest("m_wx", feats=["load", "temp"])}
    cm = CapabilityMatcher(manifests)

    res = cm.match(_scenario({"load"}), available_features={"load"})

    assert "m_wx" not in res.eligible
    assert "feature" in res.rejected["m_wx"].lower()
    assert "temp" in res.rejected["m_wx"]


def test_reject_non_active():
    manifests = {
        "m_shadow": _manifest("m_shadow", feats=["load"], stage=ModelLifecycleStage.SHADOW)
    }
    cm = CapabilityMatcher(manifests)

    res = cm.match(_scenario({"load"}), available_features={"load"})

    assert "m_shadow" not in res.eligible
    assert "lifecycle" in res.rejected["m_shadow"].lower()


def test_reject_business_domain_mismatch():
    manifests = {"m_other": _manifest("m_other", domain="fee_recovery_risk", feats=["load"])}
    cm = CapabilityMatcher(manifests)

    res = cm.match(_scenario({"load"}), available_features={"load"})

    assert "m_other" not in res.eligible
    assert "business_domain" in res.rejected["m_other"]


def test_reject_output_schema_incompatible():
    sc = ScenarioDefinition(
        task_type=TaskType.FORECASTING,
        business_domain="load_forecast",
        data_contract=_dc(),
        target_schema={"proba": "float"},
        primary_metric="MAE",
        signature_features=["load"],
        signature={"load": 1.0},
        region="R",
    )
    manifests = {"m_reg": _manifest("m_reg", feats=["load"], out={"yhat": "float"})}

    res = CapabilityMatcher(manifests).match(sc, available_features={"load"})

    assert "m_reg" not in res.eligible
    assert "output" in res.rejected["m_reg"].lower()


def test_empty_output_schema_is_compatible_for_legacy_models():
    manifests = {"legacy": _manifest("legacy", feats=["load"], out={})}

    res = CapabilityMatcher(manifests).match(_scenario({"load"}), available_features={"load"})

    assert res.eligible == ["legacy"]


def test_match_writes_selection_trace():
    manifests = {
        "m_ok": _manifest("m_ok", feats=["load"]),
        "m_clf": _manifest("m_clf", task=TaskType.CLASSIFICATION, feats=["load"]),
    }
    trace = SelectionTrace(scenario_id="PJME_x")

    res = CapabilityMatcher(manifests).match(_scenario({"load"}), {"load"}, trace=trace)

    assert res.eligible == ["m_ok"]
    assert set(trace.candidates_considered) == {"m_ok", "m_clf"}
    assert "m_clf" in trace.candidates_rejected
    stage = next(s for s in trace.stages if s["stage"] == "CapabilityMatch")
    assert stage["outputs"]["eligible"] == ["m_ok"]
    assert stage["outputs"]["n_rejected"] == 1


def test_from_index_manager_and_substitute_hook_noop():
    manifests = {"m_ok": _manifest("m_ok", feats=["load"])}
    manager = IndexManager.with_defaults(manifests=manifests)
    cm = CapabilityMatcher.from_index_manager(manager)

    res = cm.match(_scenario({"load"}), {"load"}, allow_substitute=True)

    assert "m_ok" in res.eligible
    assert res.substitutions == {}


def test_substitute_maps_rejected_model_to_eligible_neighbor():
    graph = ModelGraph()
    add_substitute_relation(graph, "needs_weather", "load_only", weight=0.9)
    manifests = {
        "needs_weather": _manifest("needs_weather", feats=["load", "weather"]),
        "load_only": _manifest("load_only", feats=["load"]),
    }

    res = CapabilityMatcher(manifests, substitute_graph=graph).match(
        _scenario({"load"}),
        {"load"},
        allow_substitute=True,
    )

    assert res.eligible == ["load_only"]
    assert res.substitutions == {"needs_weather": "load_only"}


def test_substitute_is_traced_without_clearing_original_rejection():
    graph = ModelGraph()
    add_substitute_relation(graph, "needs_weather", "load_only", weight=0.9)
    manifests = {
        "needs_weather": _manifest("needs_weather", feats=["load", "weather"]),
        "load_only": _manifest("load_only", feats=["load"]),
    }
    trace = SelectionTrace(scenario_id="PJME_x")

    CapabilityMatcher(manifests, substitute_graph=graph).match(
        _scenario({"load"}),
        {"load"},
        trace=trace,
        allow_substitute=True,
    )

    assert "needs_weather" in trace.candidates_rejected
    stage = next(s for s in trace.stages if s["stage"] == "CapabilityMatch")
    assert stage["outputs"]["substitutions"] == {"needs_weather": "load_only"}
