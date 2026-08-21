"""Path-as-hyperedge helpers for model-combination graph reasoning."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.core.trace import SelectionTrace

from .temporal_relations import HawkesRelationUpdater, RelationEvent

if TYPE_CHECKING:
    from .model_graph import ModelGraph


_ORDERED_STRATEGIES = {"stacking", "serial", "pipeline", "chain", "ordered"}
_METRIC_FIELDS = {
    "mae",
    "rmse",
    "mape",
    "latency_ms",
    "drift_level",
    "uncertainty_score",
    "dynamic_strength",
}


def instantiate_hyperpath(
    graph: "ModelGraph",
    *,
    models: Sequence[str],
    strategy: str = "weighted_mean",
    path_id: Optional[str] = None,
    created_from: str = "",
    ordered: Optional[bool] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    evidence_ref: str = "",
    create_missing_models: bool = False,
) -> str:
    """Create or update a Path node as an engineering hyperedge."""
    members = list(normalize_path_members(models, strategy=strategy, ordered=ordered))
    if not members:
        raise ValueError("hyperpath requires at least one model")

    resolved_ordered = is_order_sensitive(strategy, ordered=ordered)
    resolved_path_id = path_id or build_hyperpath_id(
        members,
        strategy=strategy,
        ordered=resolved_ordered,
    )
    attrs = build_hyperpath_node_attrs(
        members,
        strategy=strategy,
        created_from=created_from,
        ordered=resolved_ordered,
        metrics=metrics,
        evidence_ref=evidence_ref,
    )

    if graph.G.has_node(resolved_path_id):
        existing = graph.G.nodes[resolved_path_id]
        existing_refs = existing.get("evidence_refs", [])
        attrs["evidence_refs"] = _merge_refs(existing_refs, attrs.get("evidence_refs", []))
        if metrics is None:
            attrs.pop("metrics", None)
        else:
            attrs["metrics"] = {
                **dict(existing.get("metrics") or {}),
                **dict(attrs.get("metrics") or {}),
            }
        if not created_from and existing.get("created_from"):
            attrs["created_from"] = existing["created_from"]
        graph.G.nodes[resolved_path_id].update(attrs)
    else:
        graph.G.add_node(resolved_path_id, **attrs)

    member_count = len(members)
    for order, model_id in enumerate(members):
        if not graph.G.has_node(model_id):
            if create_missing_models:
                graph.G.add_node(model_id, node_type="model")
            else:
                continue
        graph.G.add_edge(
            model_id,
            resolved_path_id,
            edge_type="part_of",
            order=order,
            membership_weight=1.0 / member_count,
        )

    return resolved_path_id


def build_hyperpath_node_attrs(
    members: Sequence[str],
    *,
    strategy: str,
    created_from: str = "",
    ordered: bool = False,
    metrics: Optional[Mapping[str, Any]] = None,
    evidence_ref: str = "",
) -> Dict[str, Any]:
    metrics = dict(metrics or {})
    attrs: Dict[str, Any] = {
        "node_type": "path",
        "is_hyperedge": True,
        "members": list(members),
        "composition": list(members),
        "member_count": len(members),
        "strategy": strategy,
        "ordered": bool(ordered),
        "canonical_key": canonical_hyperpath_key(
            members,
            strategy=strategy,
            ordered=ordered,
        ),
        "created_from": created_from,
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key in _METRIC_FIELDS
        },
        "evidence_refs": [evidence_ref] if evidence_ref else [],
    }
    for key, value in metrics.items():
        if key in _METRIC_FIELDS:
            attrs[key] = value
    return attrs


def update_hyperpath_metrics(
    graph: "ModelGraph",
    path_id: str,
    metrics: Mapping[str, Any],
    *,
    evidence_ref: str = "",
) -> Dict[str, Any]:
    """Update aggregate performance/cost/drift/uncertainty fields on a Path node."""
    _require_path_node(graph, path_id)
    node = graph.G.nodes[path_id]
    node_metrics = dict(node.get("metrics") or {})
    changed: Dict[str, Any] = {}
    for key, value in dict(metrics).items():
        if key not in _METRIC_FIELDS:
            continue
        node[key] = value
        node_metrics[key] = value
        changed[key] = value
    node["metrics"] = node_metrics
    if evidence_ref:
        node["evidence_refs"] = _merge_refs(node.get("evidence_refs", []), [evidence_ref])
    return changed


def add_scenario_hyperpath_edge(
    graph: "ModelGraph",
    scenario_id: str,
    path_id: str,
    *,
    relation_type: str = "recommended_for",
    weight: float = 1.0,
    evidence_ref: str = "",
    **attrs: Any,
) -> None:
    """Connect a scenario to a Path hyperedge with evidence-aware edge metadata."""
    _require_path_node(graph, path_id)
    existing_refs: Iterable[str] = []
    if graph.G.has_edge(scenario_id, path_id):
        existing_refs = graph.G[scenario_id][path_id].get("evidence_refs", [])
    graph.G.add_edge(
        scenario_id,
        path_id,
        edge_type=relation_type,
        weight=float(weight),
        evidence_refs=_merge_refs(existing_refs, [evidence_ref] if evidence_ref else []),
        **attrs,
    )


def get_top_hyperpaths_for_scenario(
    graph: "ModelGraph",
    scenario_id: str,
    *,
    relation_types: Sequence[str] = ("recommended_for", "selected_for"),
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Return scenario-linked Path hyperedges sorted by current relation strength."""
    if not graph.G.has_node(scenario_id):
        return []
    allowed = set(relation_types)
    results: List[Dict[str, Any]] = []
    for _, path_id, edge in graph.G.out_edges(scenario_id, data=True):
        if edge.get("edge_type") not in allowed:
            continue
        if not _is_path_node(graph, path_id):
            continue
        node = dict(graph.G.nodes[path_id])
        score = float(edge.get("dynamic_strength", edge.get("weight", 0.0)))
        results.append({
            "scenario_id": scenario_id,
            "path_id": path_id,
            "score": score,
            "relation_type": edge.get("edge_type"),
            "edge": dict(edge),
            "path": node,
        })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def get_hyperpaths_for_model(
    graph: "ModelGraph",
    model_id: str,
    *,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return Path hyperedges that contain the given model."""
    if not graph.G.has_node(model_id):
        return []
    results: List[Dict[str, Any]] = []
    for _, path_id, edge in graph.G.out_edges(model_id, data=True):
        if edge.get("edge_type") != "part_of":
            continue
        if not _is_path_node(graph, path_id):
            continue
        node = dict(graph.G.nodes[path_id])
        score = float(node.get("dynamic_strength", edge.get("dynamic_strength", 0.0)))
        results.append({
            "model_id": model_id,
            "path_id": path_id,
            "score": score,
            "edge": dict(edge),
            "path": node,
        })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results if top_k is None else results[:top_k]


def apply_hyperpath_temporal_update(
    graph: "ModelGraph",
    events: Sequence[RelationEvent],
    *,
    now: datetime | str | int | float,
    updater: Optional[HawkesRelationUpdater] = None,
    create_missing: bool = False,
    accumulate: bool = False,
) -> Dict[str, Any]:
    """Apply temporal relation updates to scenario->Path and member->Path edges."""
    relation_updater = updater or HawkesRelationUpdater()
    path_summary = relation_updater.apply_to_graph(
        graph,
        events,
        now=now,
        create_missing=create_missing,
        accumulate=accumulate,
    )
    path_updates = _sync_path_nodes_from_edge_updates(graph, path_summary)
    member_events = _member_events_from_path_events(graph, events)
    member_summary = relation_updater.apply_to_graph(
        graph,
        member_events,
        now=now,
        create_missing=False,
        accumulate=accumulate,
    )
    return {
        "path_relation_update": path_summary,
        "member_relation_update": member_summary,
        "path_updates": path_updates,
        "member_event_count": len(member_events),
    }


def record_hyperpath_selection(
    trace: SelectionTrace,
    path_id: str,
    *,
    score: Optional[float] = None,
    evidence_ref: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    outputs: Dict[str, Any] = {"path_id": path_id}
    if score is not None:
        outputs["score"] = float(score)
    trace.add_stage(
        "HyperPathSelect",
        outputs=outputs,
        metadata=dict(metadata or {}),
    )
    if evidence_ref:
        trace.add_evidence_ref(evidence_ref)


def normalize_path_members(
    models: Sequence[str],
    *,
    strategy: str = "weighted_mean",
    ordered: Optional[bool] = None,
) -> tuple[str, ...]:
    """Normalize members for order-insensitive paths; preserve sequence for ordered paths."""
    seen = set()
    unique = []
    for model_id in models:
        model_key = str(model_id)
        if model_key in seen:
            continue
        unique.append(model_key)
        seen.add(model_key)
    if is_order_sensitive(strategy, ordered=ordered):
        return tuple(unique)
    return tuple(sorted(unique))


def is_order_sensitive(strategy: str, *, ordered: Optional[bool] = None) -> bool:
    if ordered is not None:
        return bool(ordered)
    return strategy.lower() in _ORDERED_STRATEGIES


def canonical_hyperpath_key(
    members: Sequence[str],
    *,
    strategy: str,
    ordered: bool,
) -> str:
    separator = ">" if ordered else "+"
    return f"{strategy}|{separator.join(members)}"


def build_hyperpath_id(
    members: Sequence[str],
    *,
    strategy: str,
    ordered: bool,
) -> str:
    separator = "__then__" if ordered else "__"
    return f"path::{strategy}::{separator.join(members)}"


def _sync_path_nodes_from_edge_updates(
    graph: "ModelGraph",
    summary: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    updates: List[Dict[str, Any]] = []
    for update in summary.get("edge_updates", []):
        path_id = update.get("target")
        if not path_id or not _is_path_node(graph, path_id):
            continue
        strength = float(update.get("dynamic_strength", 0.0))
        refs = update.get("evidence_refs", [])
        node = graph.G.nodes[path_id]
        node["dynamic_strength"] = strength
        node["event_count"] = int(node.get("event_count", 0)) + int(update.get("event_count", 0))
        node["event_evidence_refs"] = _merge_refs(node.get("event_evidence_refs", []), refs)
        updates.append({
            "path_id": path_id,
            "dynamic_strength": strength,
            "evidence_refs": list(refs),
        })
    return updates


def _member_events_from_path_events(
    graph: "ModelGraph",
    events: Sequence[RelationEvent],
) -> List[RelationEvent]:
    member_events: List[RelationEvent] = []
    for event in events:
        if not _is_path_node(graph, event.target):
            continue
        members = list(graph.G.nodes[event.target].get("members") or [])
        if not members:
            continue
        magnitude = float(event.magnitude) / len(members)
        for model_id in members:
            if graph.G.has_edge(model_id, event.target):
                member_events.append(RelationEvent(
                    source=model_id,
                    target=event.target,
                    relation_type="part_of",
                    timestamp=event.timestamp,
                    polarity=event.polarity,
                    magnitude=magnitude,
                    evidence_ref=event.evidence_ref,
                    metadata={
                        **dict(event.metadata),
                        "source_event": event.relation_type,
                    },
                ))
    return member_events


def _require_path_node(graph: "ModelGraph", path_id: str) -> None:
    if not _is_path_node(graph, path_id):
        raise ValueError(f"Path node does not exist or is not a path hyperedge: {path_id}")


def _is_path_node(graph: "ModelGraph", path_id: str) -> bool:
    if not graph.G.has_node(path_id):
        return False
    node = graph.G.nodes[path_id]
    return node.get("node_type") == "path"


def _merge_refs(existing: Iterable[str], new: Iterable[str]) -> List[str]:
    refs: List[str] = []
    seen = set()
    for ref in list(existing or []) + list(new or []):
        if ref and ref not in seen:
            refs.append(ref)
            seen.add(ref)
    return refs
