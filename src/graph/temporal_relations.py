"""Temporal relation strength updates for dynamic model graphs."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from src.core.trace import SelectionTrace

if TYPE_CHECKING:
    from src.core.solver.context import SolveContext

    from .model_graph import ModelGraph


_POSITIVE_POLARITIES = {"positive", "success", "boost", "+"}
_NEGATIVE_POLARITIES = {"negative", "failure", "penalty", "drift", "uncertainty", "-"}


@dataclass(frozen=True)
class RelationEvent:
    """A signed event that changes one semantic graph relation over time."""

    source: str
    target: str
    relation_type: str
    timestamp: datetime | str | int | float
    polarity: str = "positive"
    magnitude: float = 1.0
    evidence_ref: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def timestamp_dt(self) -> datetime:
        return _coerce_datetime(self.timestamp)

    @property
    def signed_magnitude(self) -> float:
        polarity = self.polarity.lower()
        if polarity in _POSITIVE_POLARITIES:
            sign = 1.0
        elif polarity in _NEGATIVE_POLARITIES:
            sign = -1.0
        else:
            raise ValueError(f"Unsupported relation event polarity: {self.polarity}")
        return sign * max(float(self.magnitude), 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "timestamp": _format_datetime(self.timestamp_dt),
            "polarity": self.polarity,
            "magnitude": float(self.magnitude),
            "evidence_ref": self.evidence_ref,
            "metadata": dict(self.metadata),
        }


class HawkesRelationUpdater:
    """Score relation strength with exponentially decayed signed events."""

    def __init__(
        self,
        *,
        base_strength: float = 0.5,
        alpha: float = 0.2,
        beta: float = 1.0 / (7 * 24 * 3600),
        time_unit_seconds: float = 1.0,
        min_strength: float = 0.0,
        max_strength: float = 1.0,
    ):
        self.base_strength = float(base_strength)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.time_unit_seconds = max(float(time_unit_seconds), 1e-12)
        self.min_strength = float(min_strength)
        self.max_strength = float(max_strength)
        if self.min_strength > self.max_strength:
            raise ValueError("min_strength must be <= max_strength")

    def score(
        self,
        events: Iterable[RelationEvent],
        *,
        now: datetime | str | int | float,
    ) -> float:
        now_dt = _coerce_datetime(now)
        strength = self.base_strength
        for event in events:
            age = max((now_dt - event.timestamp_dt).total_seconds(), 0.0)
            decay = math.exp(-self.beta * (age / self.time_unit_seconds))
            strength += self.alpha * event.signed_magnitude * decay
        return _clip(strength, self.min_strength, self.max_strength)

    def apply_to_graph(
        self,
        graph: "ModelGraph",
        events: Sequence[RelationEvent],
        *,
        now: datetime | str | int | float,
        create_missing: bool = False,
        accumulate: bool = False,
        max_history: int = 50,
    ) -> Dict[str, Any]:
        grouped = _group_events(events)
        edge_updates: List[Dict[str, Any]] = []
        skipped_edges = 0
        now_dt = _coerce_datetime(now)

        for (source, target, relation_type), group in grouped.items():
            if not graph.G.has_edge(source, target):
                if create_missing:
                    graph.G.add_edge(source, target, edge_type=relation_type, weight=self.base_strength)
                else:
                    skipped_edges += 1
                    continue

            edge = graph.G[source][target]
            edge_type = edge.get("edge_type")
            if edge_type is not None and edge_type != relation_type:
                skipped_edges += 1
                continue

            scoring_events: List[RelationEvent] = list(group)
            history: List[Dict[str, Any]] = []
            if accumulate:
                # 跨轮事件历史参与重算：旧事件按 exp(-beta*age) 自然衰减
                history = list(edge.get("event_history") or [])
                scoring_events = [
                    RelationEvent(
                        source=source,
                        target=target,
                        relation_type=relation_type,
                        timestamp=item["timestamp"],
                        polarity=item.get("polarity", "positive"),
                        magnitude=float(item.get("magnitude", 1.0)),
                        evidence_ref=item.get("evidence_ref", ""),
                    )
                    for item in history
                ] + scoring_events

            dynamic_strength = self.score(scoring_events, now=now_dt)
            evidence_refs = [
                event.evidence_ref
                for event in group
                if event.evidence_ref
            ]
            if accumulate:
                history.extend(
                    {
                        "timestamp": _format_datetime(event.timestamp_dt),
                        "polarity": event.polarity,
                        "magnitude": float(event.magnitude),
                        "evidence_ref": event.evidence_ref,
                    }
                    for event in group
                )
                edge["event_history"] = history[-max(1, int(max_history)):]
            edge["edge_type"] = relation_type
            edge["weight"] = dynamic_strength
            edge["dynamic_strength"] = dynamic_strength
            edge["last_event_at"] = _format_datetime(now_dt)
            edge["event_count"] = int(edge.get("event_count", 0)) + len(group)
            edge["event_evidence_refs"] = _merge_refs(
                edge.get("event_evidence_refs", []),
                evidence_refs,
            )
            edge_updates.append({
                "source": source,
                "target": target,
                "relation_type": relation_type,
                "dynamic_strength": dynamic_strength,
                "event_count": len(group),
                "evidence_refs": evidence_refs,
            })

        return {
            "updated_edges": len(edge_updates),
            "skipped_edges": skipped_edges,
            "edge_updates": edge_updates,
            "now": _format_datetime(now_dt),
        }


def events_from_selection_trace(
    trace: SelectionTrace,
    *,
    selected_target: Optional[str] = None,
    source: Optional[str] = None,
    evidence_ref: Optional[str] = None,
    timestamp: Optional[datetime | str | int | float] = None,
    selected_relation_type: str = "selected_for",
    rejected_relation_type: str = "rejected_candidate",
) -> List[RelationEvent]:
    return events_from_solver_result(
        trace,
        {
            "models": list(trace.final_selection),
            "weights": dict(trace.final_weights),
        },
        selected_target=selected_target,
        source=source,
        evidence_ref=evidence_ref,
        timestamp=timestamp,
        selected_relation_type=selected_relation_type,
        rejected_relation_type=rejected_relation_type,
    )


def events_from_solver_result(
    trace: SelectionTrace,
    result: Mapping[str, Any],
    *,
    selected_target: Optional[str] = None,
    source: Optional[str] = None,
    evidence_ref: Optional[str] = None,
    timestamp: Optional[datetime | str | int | float] = None,
    selected_relation_type: str = "selected_for",
    rejected_relation_type: str = "rejected_candidate",
) -> List[RelationEvent]:
    event_source = source or trace.scenario_id
    event_timestamp = timestamp or trace.timestamp
    event_ref = evidence_ref or f"trace:{trace.scenario_id}:{trace.timestamp}"
    final_selection = list(result.get("models") or [])
    final_weights = dict(result.get("weights") or {})
    events: List[RelationEvent] = []

    if selected_target is not None:
        events.append(RelationEvent(
            source=event_source,
            target=selected_target,
            relation_type=selected_relation_type,
            timestamp=event_timestamp,
            polarity="positive",
            magnitude=_selection_magnitude(final_weights),
            evidence_ref=event_ref,
            metadata={"selected": final_selection},
        ))
    # 逐模型的 scenario->model 边**始终**产出，且关系类型与 selected_target 分支
    # 保持一致（§11#7）：候选评分消费的正是 scenario->model 的 recommended_for。
    # 原实现只在没有 path_id 时才建逐模型边，且把关系类型硬编码成
    # "selected_model"，与消费者不匹配，闭环因此断裂。
    for model_id in final_selection:
        events.append(RelationEvent(
            source=event_source,
            target=model_id,
            relation_type=selected_relation_type,
            timestamp=event_timestamp,
            polarity="positive",
            magnitude=abs(float(final_weights.get(model_id, 1.0))),
            evidence_ref=event_ref,
        ))

    for candidate_id, reason in trace.candidates_rejected.items():
        events.append(RelationEvent(
            source=event_source,
            target=candidate_id,
            relation_type=rejected_relation_type,
            timestamp=event_timestamp,
            polarity="negative",
            magnitude=_rejection_magnitude(reason),
            evidence_ref=event_ref,
            metadata={"reason": reason},
        ))

    return events


def make_temporal_relation_stage(
    graph: "ModelGraph",
    *,
    updater: Optional[HawkesRelationUpdater] = None,
    now: Optional[
        datetime
        | str
        | int
        | float
        | Callable[[], datetime | str | int | float]
    ] = None,
    selected_relation_type: str = "recommended_for",
    rejected_relation_type: str = "rejected_candidate",
    create_missing: bool = False,
    accumulate: bool = True,
) -> Callable[["SolveContext", SelectionTrace], None]:
    """Build a solver post-stage that updates temporal relation strengths.

    accumulate=True（默认）时，边上保留有界事件历史并参与强度重算，
    使跨轮反馈真正驱动 Hawkes 衰减累积；首轮行为与不累积完全一致。
    """

    relation_updater = updater or HawkesRelationUpdater()

    def temporal_relation_stage(ctx: "SolveContext", trace: SelectionTrace) -> None:
        event_time = _resolve_now(now, trace.timestamp)
        result = dict(ctx.extras.get("result") or {})
        selected_target = result.get("path_id") or _path_id_from_result(result)
        evidence_ref = f"trace:{trace.scenario_id}:{trace.timestamp}"
        events = events_from_solver_result(
            trace,
            result,
            selected_target=selected_target,
            evidence_ref=evidence_ref,
            timestamp=event_time,
            selected_relation_type=selected_relation_type,
            rejected_relation_type=rejected_relation_type,
        )
        if selected_target is not None and _is_path_target(graph, selected_target):
            from .hyperpath import apply_hyperpath_temporal_update

            hyper_summary = apply_hyperpath_temporal_update(
                graph,
                events,
                now=event_time,
                updater=relation_updater,
                create_missing=create_missing,
                accumulate=accumulate,
            )
            summary = {
                **dict(hyper_summary["path_relation_update"]),
                "hyperpath": hyper_summary,
            }
        else:
            summary = relation_updater.apply_to_graph(
                graph,
                events,
                now=event_time,
                create_missing=create_missing,
                accumulate=accumulate,
            )
        ctx.extras["temporal_relation_update"] = summary
        record_temporal_update(trace, summary, evidence_ref=evidence_ref)

    temporal_relation_stage.__name__ = "temporal_relation_stage"
    return temporal_relation_stage


def record_temporal_update(
    trace: SelectionTrace,
    summary: Mapping[str, Any],
    *,
    evidence_ref: Optional[str] = None,
) -> None:
    trace.add_stage(
        "TemporalRelationUpdate",
        outputs=dict(summary),
        metadata={"method": "hawkes_relation_update"},
    )
    if evidence_ref:
        trace.add_evidence_ref(evidence_ref)


def _group_events(
    events: Iterable[RelationEvent],
) -> Dict[tuple[str, str, str], List[RelationEvent]]:
    grouped: Dict[tuple[str, str, str], List[RelationEvent]] = {}
    for event in events:
        key = (event.source, event.target, event.relation_type)
        grouped.setdefault(key, []).append(event)
    return grouped


def _selection_magnitude(weights: Mapping[str, float]) -> float:
    if not weights:
        return 1.0
    total = sum(abs(float(value)) for value in weights.values())
    return max(total, 1e-12)


def _rejection_magnitude(reason: str) -> float:
    reason_l = reason.lower()
    if "drift" in reason_l or "uncertainty" in reason_l or "cascade" in reason_l:
        return 1.0
    return 0.5


def _resolve_now(
    now: Optional[
        datetime
        | str
        | int
        | float
        | Callable[[], datetime | str | int | float]
    ],
    fallback: datetime | str | int | float,
) -> datetime | str | int | float:
    if now is None:
        return fallback
    if callable(now):
        return now()
    return now


def _path_id_from_result(result: Mapping[str, Any]) -> Optional[str]:
    models = list(result.get("models") or [])
    if not models:
        return None
    strategy = result.get("strategy") or "unknown"
    return f"path::{strategy}::{'__'.join(models)}"


def _is_path_target(graph: "ModelGraph", target: str) -> bool:
    return (
        graph.G.has_node(target)
        and graph.G.nodes[target].get("node_type") == "path"
    )


def _merge_refs(existing: Iterable[str], new: Iterable[str]) -> List[str]:
    refs: List[str] = []
    seen = set()
    for ref in list(existing or []) + list(new or []):
        if ref and ref not in seen:
            refs.append(ref)
            seen.add(ref)
    return refs


def _coerce_datetime(value: datetime | str | int | float) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_datetime(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)
