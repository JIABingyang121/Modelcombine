"""Reusable solver stages."""
from __future__ import annotations

from typing import Any, Callable, Mapping

from ..capability_matcher import CapabilityMatcher
from ..index import IndexManager
from ..schema import ModelManifest
from ..trace import SelectionTrace
from .context import SolveContext

Stage = Callable[[SolveContext, SelectionTrace], None]


def make_capability_match_stage(
    manifests: Mapping[str, ModelManifest],
    *,
    allow_substitute: bool = False,
    substitute_graph: Any = None,
    min_substitute_weight: float = 0.0,
) -> Stage:
    matcher = CapabilityMatcher(
        manifests,
        substitute_graph=substitute_graph,
        min_substitute_weight=min_substitute_weight,
    )

    def stage(ctx: SolveContext, trace: SelectionTrace) -> None:
        result = matcher.match(
            ctx.scenario,
            ctx.available_features,
            trace=trace,
            allow_substitute=allow_substitute,
        )
        if ctx.model_cols:
            matched_models = []
            for model in ctx.model_cols:
                if model in result.eligible:
                    matched_models.append(model)
                elif model in result.substitutions:
                    matched_models.append(result.substitutions[model])
            ctx.model_cols = list(dict.fromkeys(matched_models))
        else:
            ctx.model_cols = list(result.eligible)
        ctx.extras["capability_match"] = result

    return stage


def make_similar_scenario_stage(index_manager: IndexManager, top_k: int = 3) -> Stage:
    def stage(ctx: SolveContext, trace: SelectionTrace) -> None:
        similar = index_manager.query(
            "scenario",
            signature=ctx.scenario.signature,
            business_domain=ctx.scenario.business_domain,
            top_k=top_k,
        )
        ctx.similar = similar
        trace.add_stage(
            "SimilarScenarioRetrieve",
            inputs={
                "top_k": top_k,
                "business_domain": ctx.scenario.business_domain,
                "signature_keys": sorted(ctx.scenario.signature.keys()),
            },
            outputs={
                "n_similar": len(similar),
                "scenario_ids": [record.get("scenario_id") for record in similar],
            },
        )

    return stage
