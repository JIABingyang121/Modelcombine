from datetime import datetime, timedelta, timezone

from src.core.enums import TaskType
from src.core.schema import DataContract, ScenarioDefinition
from src.core.solver.context import SolveContext
from src.core.trace import SelectionTrace
from src.graph.model_graph import ModelGraph
from src.graph.temporal_relations import (
    HawkesRelationUpdater,
    RelationEvent,
    events_from_selection_trace,
    events_from_solver_result,
    make_temporal_relation_stage,
    record_temporal_update,
)


NOW = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)


def _scenario():
    return ScenarioDefinition(
        task_type=TaskType.FORECASTING,
        business_domain="load_forecast",
        data_contract=DataContract(
            required_columns={"load": "float"},
            freq="H",
            min_samples=50,
            business_domain="load_forecast",
        ),
        target_schema={"yhat": "float"},
        primary_metric="MAE",
        signature_features=["load"],
        signature={"load": 1.0},
        region="R",
    )


def test_hawkes_score_uses_positive_negative_events_and_time_decay():
    updater = HawkesRelationUpdater(
        base_strength=0.5,
        alpha=0.4,
        beta=1.0,
        time_unit_seconds=1.0,
    )
    recent_success = RelationEvent(
        source="scenario_pjm_h1",
        target="path_a",
        relation_type="selected_for",
        timestamp=NOW,
        polarity="positive",
        magnitude=1.0,
    )
    recent_failure = RelationEvent(
        source="scenario_pjm_h1",
        target="path_a",
        relation_type="selected_for",
        timestamp=NOW,
        polarity="negative",
        magnitude=1.0,
    )
    old_success = RelationEvent(
        source="scenario_pjm_h1",
        target="path_a",
        relation_type="selected_for",
        timestamp=NOW - timedelta(seconds=10),
        polarity="positive",
        magnitude=1.0,
    )

    single_success_score = updater.score([recent_success], now=NOW)
    repeated_success_score = updater.score([recent_success, recent_success], now=NOW)
    old_success_score = updater.score([old_success], now=NOW)

    assert single_success_score > 0.5
    assert repeated_success_score > single_success_score
    assert updater.score([recent_failure], now=NOW) < 0.5
    assert old_success_score < single_success_score
    assert abs(old_success_score - 0.5) < 0.001


def test_hawkes_updater_writes_dynamic_strength_to_model_graph():
    graph = ModelGraph()
    graph.add_scenario_node("scenario_pjm_h1", {"signature": {}})
    graph.add_path_node("path_a", ["m1", "m2"], "weighted_mean")
    graph.add_scenario_path_edge("scenario_pjm_h1", "path_a", performance_score=0.5)
    event = RelationEvent(
        source="scenario_pjm_h1",
        target="path_a",
        relation_type="recommended_for",
        timestamp=NOW,
        polarity="positive",
        magnitude=1.0,
        evidence_ref="trace:p1",
    )

    summary = HawkesRelationUpdater(
        base_strength=0.5,
        alpha=0.3,
        beta=0.0,
    ).apply_to_graph(graph, [event], now=NOW)

    edge = graph.G["scenario_pjm_h1"]["path_a"]
    assert summary["updated_edges"] == 1
    assert edge["weight"] == edge["dynamic_strength"] == 0.8
    assert edge["event_count"] == 1
    assert edge["last_event_at"] == "2026-07-02T12:00:00Z"
    assert edge["event_evidence_refs"] == ["trace:p1"]


def test_model_graph_exposes_temporal_relation_update_entrypoint():
    graph = ModelGraph()
    graph.add_scenario_node("scenario_pjm_h1", {"signature": {}})
    graph.add_path_node("path_a", ["m1", "m2"], "weighted_mean")
    graph.add_scenario_path_edge("scenario_pjm_h1", "path_a", performance_score=0.5)
    event = RelationEvent(
        source="scenario_pjm_h1",
        target="path_a",
        relation_type="recommended_for",
        timestamp=NOW,
        polarity="negative",
        magnitude=1.0,
    )

    summary = graph.update_temporal_relation_strength(
        [event],
        now=NOW,
        updater=HawkesRelationUpdater(base_strength=0.5, alpha=0.2, beta=0.0),
    )

    assert summary["updated_edges"] == 1
    assert graph.G["scenario_pjm_h1"]["path_a"]["dynamic_strength"] == 0.3


def test_events_from_selection_trace_and_record_update_summary():
    trace = SelectionTrace(
        scenario_id="scenario_pjm_h1",
        timestamp="2026-07-02T12:00:00Z",
    )
    trace.set_final(["m1", "m2"], {"m1": 0.7, "m2": 0.3})
    trace.reject("m_bad", "uncertainty_bypass: score=0.9 exceeds threshold")

    events = events_from_selection_trace(
        trace,
        selected_target="path_m1_m2",
        evidence_ref="trace:p1",
    )

    # §11#7 起：除 scenario->path 边外，**始终**同时产出 scenario->model 边，
    # 且关系类型与 selected_relation_type 一致——候选评分消费的正是后者。
    assert [(e.source, e.target, e.relation_type, e.polarity) for e in events] == [
        ("scenario_pjm_h1", "path_m1_m2", "selected_for", "positive"),
        ("scenario_pjm_h1", "m1", "selected_for", "positive"),
        ("scenario_pjm_h1", "m2", "selected_for", "positive"),
        ("scenario_pjm_h1", "m_bad", "rejected_candidate", "negative"),
    ]
    assert events[0].magnitude == 1.0
    # 拒绝事件现在排在逐模型边之后，按关系类型定位而不是按固定下标
    rejected = next(e for e in events if e.relation_type == "rejected_candidate")
    assert rejected.metadata["reason"].startswith("uncertainty_bypass")

    record_temporal_update(
        trace,
        {
            "updated_edges": 1,
            "edge_updates": [
                {
                    "source": "scenario_pjm_h1",
                    "target": "path_m1_m2",
                    "dynamic_strength": 0.8,
                }
            ],
        },
        evidence_ref="trace:p1",
    )

    stage = next(s for s in trace.stages if s["stage"] == "TemporalRelationUpdate")
    assert stage["outputs"]["updated_edges"] == 1
    assert trace.evidence_refs == ["trace:p1"]


def test_events_from_solver_result_do_not_require_final_trace_to_be_set():
    trace = SelectionTrace(
        scenario_id="scenario_pjm_h1",
        timestamp="2026-07-02T12:00:00Z",
    )
    trace.reject("m_bad", "drift: psi exceeds threshold")

    events = events_from_solver_result(
        trace,
        {
            "models": ["m1", "m2"],
            "weights": {"m1": 0.6, "m2": 0.4},
        },
        selected_target="path_m1_m2",
        selected_relation_type="recommended_for",
        evidence_ref="trace:p1",
    )

    assert trace.final_selection == []
    # 逐模型 recommended_for 边一并产出（§11#7 闭环所需）
    assert [(e.target, e.relation_type, e.polarity) for e in events] == [
        ("path_m1_m2", "recommended_for", "positive"),
        ("m1", "recommended_for", "positive"),
        ("m2", "recommended_for", "positive"),
        ("m_bad", "rejected_candidate", "negative"),
    ]
    assert events[-1].magnitude == 1.0


def test_temporal_relation_stage_updates_graph_and_trace_from_solver_result():
    graph = ModelGraph()
    graph.add_scenario_node("scenario_pjm_h1", {"signature": {}})
    graph.add_model_node("m1", {})
    graph.add_model_node("m2", {})
    graph.add_path_node("path_a", ["m1", "m2"], "weighted_mean")
    graph.add_scenario_path_edge("scenario_pjm_h1", "path_a", performance_score=0.5)
    ctx = SolveContext(
        scenario=_scenario(),
        available_features={"load"},
        model_cols=["m1", "m2", "m_bad"],
    )
    ctx.extras["result"] = {
        "models": ["m1", "m2"],
        "weights": {"m1": 0.6, "m2": 0.4},
        "strategy": "weighted_mean",
        "path_id": "path_a",
    }
    trace = SelectionTrace(
        scenario_id="scenario_pjm_h1",
        timestamp="2026-07-02T12:00:00Z",
    )
    trace.reject("m_bad", "uncertainty_bypass: score=0.9 exceeds threshold")
    stage = make_temporal_relation_stage(
        graph,
        updater=HawkesRelationUpdater(base_strength=0.5, alpha=0.3, beta=0.0),
        now=NOW,
    )

    stage(ctx, trace)

    assert graph.G["scenario_pjm_h1"]["path_a"]["dynamic_strength"] == 0.8
    assert graph.G["m1"]["path_a"]["dynamic_strength"] == 0.65
    assert graph.G["m2"]["path_a"]["dynamic_strength"] == 0.65
    assert ctx.extras["temporal_relation_update"]["updated_edges"] == 1
    assert ctx.extras["temporal_relation_update"]["hyperpath"]["member_event_count"] == 2
    assert any(s["stage"] == "TemporalRelationUpdate" for s in trace.stages)
    assert trace.evidence_refs == ["trace:scenario_pjm_h1:2026-07-02T12:00:00Z"]


def test_apply_to_graph_accumulates_event_history_across_rounds():
    """多事件衰减累积：第二轮 apply 必须叠加第一轮的历史事件，而非只看当次事件。"""
    from src.graph.model_graph import ModelGraph
    from src.graph.temporal_relations import HawkesRelationUpdater, RelationEvent

    graph = ModelGraph()
    graph.G.add_edge("s1", "p1", edge_type="recommended_for", weight=0.5)
    updater = HawkesRelationUpdater(alpha=0.2, beta=0.0)  # beta=0：无衰减，纯叠加

    def _event(ts, ref):
        return RelationEvent(
            source="s1", target="p1", relation_type="recommended_for",
            timestamp=ts, polarity="positive", magnitude=1.0, evidence_ref=ref,
        )

    updater.apply_to_graph(graph, [_event("2026-01-01T00:00:00", "ev1")],
                           now="2026-01-01T00:00:00", accumulate=True)
    s1 = graph.G["s1"]["p1"]["dynamic_strength"]

    updater.apply_to_graph(graph, [_event("2026-01-02T00:00:00", "ev2")],
                           now="2026-01-02T00:00:00", accumulate=True)
    edge = graph.G["s1"]["p1"]

    assert edge["dynamic_strength"] > s1, "第二轮正事件应在第一轮基础上继续增强"
    assert edge["event_count"] == 2
    assert len(edge["event_history"]) == 2


def test_accumulated_history_decays_when_events_age():
    """长时间无强化后，旧事件按指数衰减，强度回落到 base 附近。"""
    from src.graph.model_graph import ModelGraph
    from src.graph.temporal_relations import HawkesRelationUpdater, RelationEvent

    graph = ModelGraph()
    graph.G.add_edge("s1", "p1", edge_type="recommended_for", weight=0.5)
    updater = HawkesRelationUpdater(alpha=0.2, beta=1.0 / 86400.0)  # 1天时间常数

    def _event(ts, mag):
        return RelationEvent(
            source="s1", target="p1", relation_type="recommended_for",
            timestamp=ts, polarity="positive", magnitude=mag, evidence_ref="",
        )

    updater.apply_to_graph(graph, [_event("2026-01-01T00:00:00", 1.0)],
                           now="2026-01-01T00:00:00", accumulate=True)
    fresh = graph.G["s1"]["p1"]["dynamic_strength"]

    # 100 天后一个微小事件触发重算：旧事件几乎衰减殆尽
    updater.apply_to_graph(graph, [_event("2026-04-11T00:00:00", 0.001)],
                           now="2026-04-11T00:00:00", accumulate=True)
    aged = graph.G["s1"]["p1"]["dynamic_strength"]

    assert aged < fresh
    assert abs(aged - 0.5) < 0.01, "旧事件衰减后强度应回落到 base_strength 附近"
