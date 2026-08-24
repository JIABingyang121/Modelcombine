"""把有来源的路径推理结果转换为逐模型评分贡献（Task 8.3 Task 2）。

Protocol B 的 reasoning 不再在 stepwise 之后覆盖模型集合，而是把
``infer_optimal_path_by_reasoning`` 返回的带证据路径转换为逐模型先验，
在 base_scores 形成之前交给唯一候选选择器。

只有带 ``evidence_refs`` 的历史路径才产生非中性贡献；冷启动路径返回中性
得分 0.5，贡献恰好为 0，不参与模型取舍。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set

import numpy as np

from src.graph.model_graph import ModelGraph

# 冷启动路径的中性历史得分（与 `_get_path_historical_score` 的缺省一致）。
REASONING_NEUTRAL_SCORE = 0.5
# 单模型贡献的截断区间，避免历史证据主导候选评分。
REASONING_CONTRIBUTION_CLAMP = 0.5


@dataclass(frozen=True)
class ReasoningEvidence:
    contribution_by_model: Dict[str, float] = field(default_factory=dict)
    paths: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "disabled"
    cold_start_no_evidence: bool = True


def _members_in_scope(graph: ModelGraph, path_id: str, model_cols: Sequence[str]) -> List[str]:
    members = (
        graph.get_node_attr(path_id, "members", default=None)
        or graph.get_node_attr(path_id, "composition", default=None)
        or []
    )
    return [m for m in members if m in model_cols]


def _neutral(
    model_cols: Sequence[str],
    *,
    source: str,
    graph: ModelGraph | None = None,
    paths: Sequence[Any] | None = None,
) -> ReasoningEvidence:
    records: List[Dict[str, Any]] = []
    if graph is not None and paths:
        for p in paths:
            records.append({
                "path_id": p.path_id,
                "final_score": float(p.final_score),
                "evidence_refs": list(p.evidence_refs),
                "models": _members_in_scope(graph, p.path_id, model_cols),
                "gain": 0.0,
            })
    return ReasoningEvidence(
        contribution_by_model={m: 0.0 for m in model_cols},
        paths=records,
        source=source,
        cold_start_no_evidence=True,
    )


def build_reasoning_evidence(
    *,
    graph: ModelGraph,
    scenario_id: str,
    available_features: Set[str],
    model_cols: Sequence[str],
    max_models: int,
    mode: str,
) -> ReasoningEvidence:
    """把路径推理结果转换为逐模型评分贡献。

    只有带 ``evidence_refs`` 的路径产生 ``final_score - neutral`` 的非中性贡献；
    一个模型出现在多条证据路径时取均值。无证据时输出零贡献并标记冷启动。
    """
    model_cols = list(model_cols)
    if mode == "off":
        return _neutral(model_cols, source="disabled")

    details = graph.infer_optimal_path_by_reasoning(
        scenario_id,
        set(available_features),
        constraints={"max_models": max_models},
        return_details=True,
    )
    # 稳定排序：final_score 降序，同分用 path_id 作次级键。
    details = sorted(details, key=lambda p: (-p.final_score, p.path_id))
    evidenced = [p for p in details if p.evidence_refs]
    if not evidenced:
        return _neutral(model_cols, source="cold_start_no_evidence", graph=graph, paths=details)

    per_model_gains: Dict[str, List[float]] = {}
    records: List[Dict[str, Any]] = []
    for p in details:
        members = _members_in_scope(graph, p.path_id, model_cols)
        gain = float(p.final_score) - REASONING_NEUTRAL_SCORE if p.evidence_refs else 0.0
        if p.evidence_refs:
            for m in members:
                per_model_gains.setdefault(m, []).append(gain)
        records.append({
            "path_id": p.path_id,
            "final_score": float(p.final_score),
            "evidence_refs": list(p.evidence_refs),
            "models": list(members),
            "gain": gain,
        })

    contribution_by_model: Dict[str, float] = {m: 0.0 for m in model_cols}
    for m, gains in per_model_gains.items():
        contribution_by_model[m] = float(
            np.clip(float(np.mean(gains)), -REASONING_CONTRIBUTION_CLAMP, REASONING_CONTRIBUTION_CLAMP)
        )

    return ReasoningEvidence(
        contribution_by_model=contribution_by_model,
        paths=records,
        source="historical_evidence",
        cold_start_no_evidence=False,
    )
