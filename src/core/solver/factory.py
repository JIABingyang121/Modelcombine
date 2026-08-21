"""Factory entry point for Phase 3 solver wiring."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from ...graph.temporal_relations import make_temporal_relation_stage
from ..index import IndexManager
from ..schema import ModelManifest
from .backends import CombinatorBackend, ProtocolBBackend
from .context import CombinationBackend
from .solver import LogicalModelSelectionSolver, Stage
from .stages import make_capability_match_stage, make_similar_scenario_stage


_SUPPORTED_MODES = {"protocol_b", "combinator"}


def build_solver(
    mode: str,
    *,
    manifests: Optional[Mapping[str, ModelManifest]] = None,
    index_manager: Optional[IndexManager] = None,
    combinator: Any = None,
    backend: Optional[CombinationBackend] = None,
    uncertainty_gate: Any = None,
    cascade_decider: Any = None,
    temporal_relation_graph: Any = None,
    temporal_relation_updater: Any = None,
    temporal_relation_now: Any = None,
    temporal_relation_type: str = "recommended_for",
    temporal_relation_create_missing: bool = False,
    allow_substitute: bool = False,
    substitute_graph: Any = None,
    min_substitute_weight: float = 0.0,
    pre_stages: Optional[Iterable[Stage]] = None,
    post_stages: Optional[Iterable[Stage]] = None,
    similar_top_k: int = 3,
) -> LogicalModelSelectionSolver:
    """Build a LogicalModelSelectionSolver without switching existing app entrypoints."""
    normalized_mode = mode.strip().lower().replace("-", "_")
    if normalized_mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported solver mode '{mode}'. "
            f"Expected one of: {', '.join(sorted(_SUPPORTED_MODES))}"
        )

    default_pre_stages: list[Stage] = []
    if manifests is not None:
        default_pre_stages.append(
            make_capability_match_stage(
                manifests,
                allow_substitute=allow_substitute,
                substitute_graph=substitute_graph,
                min_substitute_weight=min_substitute_weight,
            )
        )
    if index_manager is not None:
        default_pre_stages.append(
            make_similar_scenario_stage(index_manager, top_k=similar_top_k)
        )

    selected_backend = backend or _make_backend(normalized_mode, combinator=combinator)
    default_post_stages: list[Stage] = []
    if uncertainty_gate is not None:
        apply_gate = getattr(uncertainty_gate, "apply", None)
        if not callable(apply_gate):
            raise TypeError("uncertainty_gate must expose a callable apply(ctx, trace)")
        default_post_stages.append(apply_gate)
    if cascade_decider is not None:
        apply_cascade = getattr(cascade_decider, "apply", None)
        if not callable(apply_cascade):
            raise TypeError("cascade_decider must expose a callable apply(ctx, trace)")
        default_post_stages.append(apply_cascade)
    if temporal_relation_graph is not None:
        default_post_stages.append(
            make_temporal_relation_stage(
                temporal_relation_graph,
                updater=temporal_relation_updater,
                now=temporal_relation_now,
                selected_relation_type=temporal_relation_type,
                create_missing=temporal_relation_create_missing,
            )
        )

    return LogicalModelSelectionSolver(
        pre_stages=default_pre_stages + list(pre_stages or []),
        backend=selected_backend,
        post_stages=default_post_stages + list(post_stages or []),
    )


def _make_backend(mode: str, *, combinator: Any = None) -> CombinationBackend:
    if mode == "protocol_b":
        return ProtocolBBackend()
    if mode == "combinator":
        return CombinatorBackend(combinator=combinator)
    raise ValueError(f"Unsupported solver mode '{mode}'")
