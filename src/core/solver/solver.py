"""LogicalModelSelectionSolver: orchestrates shared stages and a combination backend."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..trace import SelectionTrace
from .context import CombinationBackend, SolveContext

Stage = Callable[[SolveContext, SelectionTrace], None]


class LogicalModelSelectionSolver:
    def __init__(
        self,
        pre_stages: List[Stage],
        backend: CombinationBackend,
        post_stages: List[Stage],
    ):
        self.pre_stages = list(pre_stages)
        self.backend = backend
        self.post_stages = list(post_stages)

    def solve(
        self,
        ctx: SolveContext,
        trace_path: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], SelectionTrace]:
        trace = SelectionTrace(scenario_id=ctx.scenario.scenario_id)

        for stage in self.pre_stages:
            stage(ctx, trace)

        result = self.backend.combine(ctx, trace)
        ctx.extras["result"] = result

        for stage in self.post_stages:
            stage(ctx, trace)

        result = ctx.extras.get("result", result)
        trace.set_final(result.get("models", []), result.get("weights", {}))

        if trace_path:
            trace.save_json(trace_path)

        return result, trace
