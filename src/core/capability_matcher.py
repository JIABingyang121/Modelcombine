"""CapabilityMatcher：solver 的能力匹配阶段。

它薄封装 Phase 2 的 CapabilityIndex 思路，补齐逐模型拒绝原因、输出 schema 校验、
SelectionTrace 记录，以及后续 substitute 边兜底的接口钩子。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Set, TYPE_CHECKING

from .enums import ModelLifecycleStage
from .schema import ModelManifest, ScenarioDefinition

if TYPE_CHECKING:
    from .index import IndexManager
    from .trace import SelectionTrace


@dataclass
class MatchResult:
    eligible: list[str] = field(default_factory=list)
    rejected: Dict[str, str] = field(default_factory=dict)
    manifests: Dict[str, ModelManifest] = field(default_factory=dict)
    substitutions: Dict[str, str] = field(default_factory=dict)


class CapabilityMatcher:
    def __init__(
        self,
        manifests: Mapping[str, ModelManifest],
        *,
        substitute_graph: Any = None,
        min_substitute_weight: float = 0.0,
    ):
        self.manifests = dict(manifests)
        self.substitute_graph = substitute_graph
        self.min_substitute_weight = float(min_substitute_weight)

    @classmethod
    def from_index_manager(cls, manager: "IndexManager") -> "CapabilityMatcher":
        capability_index = manager.get("capability")
        manifests = getattr(capability_index, "manifests", None)
        if manifests is not None:
            return cls(manifests)

        indexed = manager.query("capability", active_only=False)
        return cls({m.model_id: m for m in indexed})

    def match(
        self,
        scenario: ScenarioDefinition,
        available_features: Set[str],
        trace: Optional["SelectionTrace"] = None,
        allow_substitute: bool = False,
    ) -> MatchResult:
        result = MatchResult()
        all_model_ids = list(self.manifests.keys())

        for model_id, manifest in self.manifests.items():
            reason = self._reject_reason(manifest, scenario, available_features)
            if reason is None:
                result.eligible.append(model_id)
                result.manifests[model_id] = manifest
            else:
                result.rejected[model_id] = reason

        if allow_substitute and self.substitute_graph is not None:
            self._apply_substitutes(result)

        if trace is not None:
            trace.consider(all_model_ids)
            for model_id, reason in result.rejected.items():
                trace.reject(model_id, reason)
            trace.add_stage(
                "CapabilityMatch",
                inputs={
                    "n_manifests": len(self.manifests),
                    "task_type": scenario.task_type.value,
                    "business_domain": scenario.business_domain,
                    "available_features": sorted(available_features),
                    "allow_substitute": allow_substitute,
                },
                outputs={
                    "eligible": list(result.eligible),
                    "n_rejected": len(result.rejected),
                    "substitutions": dict(result.substitutions),
                },
            )

        return result

    def _apply_substitutes(self, result: MatchResult) -> None:
        try:
            from src.graph.relations import get_substitutes
        except ImportError:
            return

        eligible_set = set(result.eligible)
        for rejected_id in result.rejected:
            substitutes = get_substitutes(
                self.substitute_graph,
                rejected_id,
                min_weight=self.min_substitute_weight,
            )
            replacement = next((mid for mid in substitutes if mid in eligible_set), None)
            if replacement is not None:
                result.substitutions[rejected_id] = replacement

    def _reject_reason(
        self,
        manifest: ModelManifest,
        scenario: ScenarioDefinition,
        available_features: Set[str],
    ) -> Optional[str]:
        if manifest.lifecycle_stage is not ModelLifecycleStage.ACTIVE:
            return f"lifecycle: stage={manifest.lifecycle_stage.value} 非 active，不可调度"

        if scenario.task_type not in manifest.task_types:
            return (
                f"task_type: 模型支持 {[t.value for t in manifest.task_types]}，"
                f"场景需要 {scenario.task_type.value}"
            )

        if scenario.business_domain and scenario.business_domain not in manifest.business_domains:
            return (
                f"business_domain: 模型域 {manifest.business_domains}，"
                f"场景域 {scenario.business_domain}"
            )

        required_features = set(manifest.input_constraints.get("features", []))
        missing_features = required_features - set(available_features)
        if missing_features:
            return f"feature: 缺失必需特征 {sorted(missing_features)}"

        target_keys = set(scenario.target_schema.keys())
        output_keys = set(manifest.output_schema.keys())
        if output_keys and not target_keys.issubset(output_keys):
            return (
                f"output_schema: 模型产出 {sorted(output_keys)}，"
                f"无法满足场景所需 {sorted(target_keys)}"
            )

        return None
