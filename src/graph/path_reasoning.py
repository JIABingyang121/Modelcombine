"""Lightweight NBFNet-style path reasoning scores (explainable multi-hop chains).

不训练神经网络：借鉴"关系路径本身可评分、可解释"的思想，把
Scenario -> Feature -> Model -> Path 推理链拆成可审计的分项 hop 得分。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

if TYPE_CHECKING:
    from .model_graph import ModelGraph

# 各 hop 在加权平均中的权重；缺失的 hop 不参与归一化
_DEFAULT_HOP_WEIGHTS: Dict[str, float] = {
    "feature_coverage": 0.3,
    "historical_performance": 0.4,
    "relation_strength": 0.3,
    "latency": 0.1,
    "drift": 0.1,
    "uncertainty": 0.1,
}


@dataclass
class ReasoningPath:
    scenario_id: str
    path_id: str
    nodes: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)
    hop_scores: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    evidence_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "path_id": self.path_id,
            "nodes": list(self.nodes),
            "edges": [list(edge) for edge in self.edges],
            "relation_types": list(self.relation_types),
            "hop_scores": dict(self.hop_scores),
            "final_score": self.final_score,
            "evidence_refs": list(self.evidence_refs),
        }


class PathReasoningScorer:
    def __init__(self, hop_weights: Optional[Dict[str, float]] = None):
        self.hop_weights = dict(hop_weights or _DEFAULT_HOP_WEIGHTS)

    def score(
        self,
        graph: "ModelGraph",
        scenario_id: str,
        path_id: str,
        *,
        available_features: Optional[Set[str]] = None,
    ) -> ReasoningPath:
        node = dict(graph.G.nodes[path_id]) if graph.G.has_node(path_id) else {}
        members = [m for m in (node.get("members") or node.get("composition") or [])]
        if not members:
            members = [
                u for u, _, d in graph.G.in_edges(path_id, data=True)
                if d.get("edge_type") == "part_of"
            ]

        hop_scores: Dict[str, float] = {}
        nodes: List[str] = [scenario_id]
        edges: List[Tuple[str, str]] = []
        relation_types: List[str] = []
        evidence_refs: List[str] = []

        # hop 1: feature coverage over members' required features
        required: Set[str] = set()
        for model_id in members:
            constraints = {}
            if graph.G.has_node(model_id):
                constraints = graph.G.nodes[model_id].get("input_constraints") or {}
            required |= set(constraints.get("features") or [])
        if available_features is None or not required:
            hop_scores["feature_coverage"] = 1.0
            covered: Set[str] = set()
        else:
            covered = required & set(available_features)
            hop_scores["feature_coverage"] = len(covered) / len(required)
        for feat in sorted(covered):
            nodes.append(feat)
            for model_id in members:
                if graph.G.has_edge(feat, model_id):
                    edges.append((feat, model_id))
                    relation_types.append("input_to")

        # hop 2: historical performance (existing lightweight lookup, 0-1)
        try:
            hop_scores["historical_performance"] = float(
                graph._get_path_historical_score(path_id, scenario_id)
            )
        except Exception:
            hop_scores["historical_performance"] = 0.5

        # hop 3: scenario -> path relation strength
        if graph.G.has_edge(scenario_id, path_id):
            edge = graph.G[scenario_id][path_id]
            strength = float(edge.get("dynamic_strength", edge.get("weight", 0.5)))
            hop_scores["relation_strength"] = min(1.0, max(0.0, strength))
            edges.append((scenario_id, path_id))
            relation_types.append(str(edge.get("edge_type", "recommended_for")))
            evidence_refs.extend(edge.get("evidence_refs", []) or [])
        else:
            hop_scores["relation_strength"] = 0.5

        # optional hops: path-level latency / drift / uncertainty metrics
        metrics = dict(node.get("metrics") or {})
        if "latency_ms" in metrics:
            hop_scores["latency"] = 1.0 / (1.0 + max(0.0, float(metrics["latency_ms"])) / 1000.0)
        if "drift_level" in metrics:
            hop_scores["drift"] = 1.0 - min(1.0, max(0.0, float(metrics["drift_level"])))
        if "uncertainty_score" in metrics:
            hop_scores["uncertainty"] = 1.0 - min(1.0, max(0.0, float(metrics["uncertainty_score"])))

        for model_id in members:
            nodes.append(model_id)
            if graph.G.has_edge(model_id, path_id):
                edges.append((model_id, path_id))
                relation_types.append("part_of")
        nodes.append(path_id)

        evidence_refs.extend(node.get("evidence_refs", []) or [])
        evidence_refs.extend(node.get("event_evidence_refs", []) or [])
        deduped_refs = list(dict.fromkeys(ref for ref in evidence_refs if ref))

        weight_sum = sum(self.hop_weights.get(name, 0.1) for name in hop_scores)
        final_score = 0.0
        if weight_sum > 0:
            final_score = sum(
                score * self.hop_weights.get(name, 0.1)
                for name, score in hop_scores.items()
            ) / weight_sum
        final_score = min(1.0, max(0.0, final_score))

        return ReasoningPath(
            scenario_id=scenario_id,
            path_id=path_id,
            nodes=nodes,
            edges=edges,
            relation_types=relation_types,
            hop_scores=hop_scores,
            final_score=final_score,
            evidence_refs=deduped_refs,
        )


def top_reasoning_paths(
    graph: "ModelGraph",
    scenario_id: str,
    *,
    available_features: Optional[Set[str]] = None,
    top_k: int = 3,
    include_path_ids: Optional[Sequence[str]] = None,
    scorer: Optional[PathReasoningScorer] = None,
) -> List[ReasoningPath]:
    """Score scenario-linked Path hyperedges (plus explicit extras), best first."""
    scorer = scorer or PathReasoningScorer()
    candidates: List[str] = []
    if graph.G.has_node(scenario_id):
        for _, path_id, data in graph.G.out_edges(scenario_id, data=True):
            if data.get("edge_type") not in ("recommended_for", "selected_for"):
                continue
            if graph.G.nodes[path_id].get("node_type") == "path":
                candidates.append(path_id)
    for path_id in include_path_ids or []:
        if path_id and graph.G.has_node(path_id) and graph.G.nodes[path_id].get("node_type") == "path":
            candidates.append(path_id)

    results = [
        scorer.score(graph, scenario_id, path_id, available_features=available_features)
        for path_id in dict.fromkeys(candidates)
    ]
    results.sort(key=lambda item: item.final_score, reverse=True)
    return results[:top_k]
