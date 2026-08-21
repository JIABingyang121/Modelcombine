"""Shared context and backend contract for solver orchestration."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..schema import ScenarioDefinition
from ..trace import SelectionTrace


@dataclass
class SolveContext:
    scenario: ScenarioDefinition
    available_features: Set[str]
    model_cols: List[str] = field(default_factory=list)

    # ProtocolBBackend payload.
    df_val: Any = None
    df_test: Any = None
    df_raw_val: Any = None
    df_raw_test: Any = None
    horizon: int = 1
    dataset_name: Optional[str] = None
    base_model_cols: Optional[List[str]] = None
    feedback_store: Any = None
    # 要求 Protocol B 交出真实 pred_val/pred_test（Task 3.1）。默认关闭：
    # 实验脚本的 JSON 结构与体积保持不变。
    return_predictions: bool = False

    # CombinatorBackend bridge payload.
    constraints: Dict[str, float] = field(default_factory=dict)
    model_graph: Any = None
    similar_scenarios: List[Any] = field(default_factory=list)
    historical_scenarios: List[Any] = field(default_factory=list)

    # Shared stage outputs.
    similar: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty: Dict[str, float] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


class CombinationBackend(ABC):
    name: str = "base"

    @abstractmethod
    def combine(self, ctx: SolveContext, trace: SelectionTrace) -> Dict[str, Any]:
        """Return a normalized result: {models, weights, strategy, path_id, ...}."""
