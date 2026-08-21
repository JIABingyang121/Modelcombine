"""Cascade decision stage for Phase 3 scheduling."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

from src.graph.model_graph import ModelGraph
from src.graph.relations import get_cascade_options

from ..trace import SelectionTrace
from .context import SolveContext


class CascadeDecider:
    """Apply auditable model-level cascade upgrades after uncertainty estimation."""

    def __init__(
        self,
        graph: ModelGraph,
        uncertainty_threshold: float,
        *,
        min_relation_weight: float = 0.0,
        scores: Optional[Mapping[str, float]] = None,
    ):
        self.graph = graph
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.min_relation_weight = float(min_relation_weight)
        self.scores = dict(scores or {})

    def apply(self, ctx: SolveContext, trace: SelectionTrace) -> None:
        result = dict(ctx.extras.get("result") or {})
        selected = list(result.get("models") or ctx.model_cols)
        weights = dict(result.get("weights") or {})
        scores = self._resolve_scores(ctx, selected)
        available_targets = set(ctx.model_cols)

        cascades: Dict[str, str] = {}
        final_models: list[str] = []
        final_weights: Dict[str, float] = {}
        for model_id in selected:
            target = self._choose_target(model_id, scores, available_targets)
            final_id = target or model_id
            if target is not None:
                cascades[model_id] = target
                trace.reject(
                    model_id,
                    "cascade_bypass: "
                    f"uncertainty={scores.get(model_id, 0.0):.6g} exceeds "
                    f"threshold={self.uncertainty_threshold:.6g}; upgraded to {target}",
                )
            if final_id not in final_models:
                final_models.append(final_id)
            final_weights[final_id] = final_weights.get(final_id, 0.0) + float(
                weights.get(model_id, 0.0)
            )

        if cascades:
            result["models"] = final_models
            result["weights"] = self._renormalize(final_models, final_weights)
            ctx.extras["result"] = result

        trace.add_stage(
            "CascadeDecide",
            inputs={
                "uncertainty_threshold": self.uncertainty_threshold,
                "min_relation_weight": self.min_relation_weight,
                "models": selected,
            },
            outputs={
                "cascades": dict(cascades),
                "models": list(result.get("models", selected)),
                "weights": dict(result.get("weights", weights)),
            },
            metadata={
                "bypass_applied": bool(cascades),
                "decision_authority": "cascade_bypass" if cascades else "audit_only",
            },
        )

    def _resolve_scores(
        self,
        ctx: SolveContext,
        selected: Sequence[str],
    ) -> Dict[str, float]:
        result = ctx.extras.get("result") or {}
        scores = {model_id: float(ctx.uncertainty.get(model_id, 0.0)) for model_id in selected}
        for source in (
            result.get("uncertainty", {}),
            ctx.extras.get("uncertainty", {}),
            self.scores,
        ):
            for model_id, score in dict(source or {}).items():
                if model_id in selected:
                    scores[model_id] = float(score)
        return scores

    def _choose_target(
        self,
        model_id: str,
        scores: Mapping[str, float],
        available_targets: set[str],
    ) -> Optional[str]:
        if scores.get(model_id, 0.0) <= self.uncertainty_threshold:
            return None
        for option in get_cascade_options(
            self.graph,
            model_id,
            min_weight=self.min_relation_weight,
        ):
            target = option.get("target")
            if target in available_targets:
                return target
        return None

    @staticmethod
    def _renormalize(
        models: Sequence[str],
        weights: Mapping[str, float],
    ) -> Dict[str, float]:
        if not models:
            return {}
        total = sum(float(weights.get(model_id, 0.0)) for model_id in models)
        if total <= 0.0:
            equal = 1.0 / len(models)
            return {model_id: equal for model_id in models}
        return {model_id: float(weights.get(model_id, 0.0)) / total for model_id in models}
