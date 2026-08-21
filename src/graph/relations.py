"""Semantic graph relation helpers for Phase 3 scheduling."""
from __future__ import annotations

from typing import Any, Dict, List

from .model_graph import ModelGraph


def add_substitute_relation(
    graph: ModelGraph,
    source: str,
    target: str,
    weight: float = 1.0,
    reason: str = "",
    bidirectional: bool = True,
    **attrs: Any,
) -> None:
    """Add an interchangeable-model relation; bidirectional by default."""
    edge_attrs = {
        "edge_type": "substitute",
        "weight": float(weight),
        "reason": reason,
        **attrs,
    }
    graph.G.add_edge(source, target, **edge_attrs)
    if bidirectional:
        graph.G.add_edge(target, source, **edge_attrs)


def add_cascade_relation(
    graph: ModelGraph,
    source: str,
    target: str,
    cost_delta: float,
    uncertainty_gain: float,
    weight: float = 1.0,
    **attrs: Any,
) -> None:
    """Add a directional cascade relation: cheap model -> stronger model."""
    graph.G.add_edge(
        source,
        target,
        edge_type="cascade",
        weight=float(weight),
        cost_delta=float(cost_delta),
        uncertainty_gain=float(uncertainty_gain),
        **attrs,
    )


def get_relations_by_type(
    graph: ModelGraph,
    model_id: str,
    rel_type: str,
    min_weight: float = 0.0,
) -> List[Dict[str, Any]]:
    """Query outgoing relations by semantic edge type, sorted by weight descending."""
    relations = []
    for _, target, data in graph.G.out_edges(model_id, data=True):
        if data.get("edge_type") != rel_type:
            continue
        weight = float(data.get("weight", 0.0))
        if weight < min_weight:
            continue
        relations.append({
            "source": model_id,
            "target": target,
            **dict(data),
            "weight": weight,
        })
    relations.sort(key=lambda item: item.get("weight", 0.0), reverse=True)
    return relations


def get_substitutes(
    graph: ModelGraph,
    model_id: str,
    min_weight: float = 0.0,
) -> List[str]:
    """Return substitute model ids."""
    return [
        item["target"]
        for item in get_relations_by_type(graph, model_id, "substitute", min_weight)
    ]


def get_cascade_options(
    graph: ModelGraph,
    model_id: str,
    min_weight: float = 0.0,
) -> List[Dict[str, Any]]:
    """Return candidate cascade upgrades from the given model."""
    return get_relations_by_type(graph, model_id, "cascade", min_weight)
