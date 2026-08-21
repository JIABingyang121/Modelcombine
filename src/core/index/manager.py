"""IndexManager：按名称管理多类型索引。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..evidence import EvidenceStore
from ..schema import ModelManifest
from .base import BaseIndex
from .signal_loaders import build_signal_records_from_reports
from .indices import (
    CapabilityIndex,
    ConflictIndex,
    DriftIndex,
    EvidenceIndex,
    LatencyIndex,
    PerformanceIndex,
    ScenarioIndex,
)


class IndexManager:
    def __init__(self):
        self._indices: Dict[str, BaseIndex] = {}

    @classmethod
    def with_defaults(
        cls,
        scenario_records: Optional[Iterable[Dict[str, Any]]] = None,
        manifests: Optional[Mapping[str, ModelManifest]] = None,
        performance_records: Optional[Iterable[Dict[str, Any]]] = None,
        conflict_edges: Optional[Iterable[Dict[str, Any]]] = None,
        evidence_store: Optional[EvidenceStore] = None,
        latency_records: Optional[Iterable[Dict[str, Any]]] = None,
        drift_records: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> "IndexManager":
        manager = cls()
        manager.register("scenario", ScenarioIndex(scenario_records))
        manager.register("capability", CapabilityIndex(manifests))
        manager.register("performance", PerformanceIndex(performance_records))
        manager.register("conflict", ConflictIndex(conflict_edges))
        manager.register("evidence", EvidenceIndex(evidence_store))
        manager.register("latency", LatencyIndex(latency_records))
        manager.register("drift", DriftIndex(drift_records))
        return manager

    @classmethod
    def with_report_signals(
        cls,
        scenario_records: Optional[Iterable[Dict[str, Any]]] = None,
        manifests: Optional[Mapping[str, ModelManifest]] = None,
        performance_records: Optional[Iterable[Dict[str, Any]]] = None,
        conflict_edges: Optional[Iterable[Dict[str, Any]]] = None,
        evidence_store: Optional[EvidenceStore] = None,
        latency_records: Optional[Iterable[Dict[str, Any]]] = None,
        drift_records: Optional[Iterable[Dict[str, Any]]] = None,
        ablation_profile_paths: Optional[Iterable[str | Path]] = None,
        kg_result_paths: Optional[Iterable[str | Path]] = None,
    ) -> "IndexManager":
        signals = build_signal_records_from_reports(
            ablation_profile_paths=ablation_profile_paths,
            kg_result_paths=kg_result_paths,
        )
        merged_latency = list(latency_records or []) + signals["latency"]
        merged_drift = list(drift_records or []) + signals["drift"]
        return cls.with_defaults(
            scenario_records=scenario_records,
            manifests=manifests,
            performance_records=performance_records,
            conflict_edges=conflict_edges,
            evidence_store=evidence_store,
            latency_records=merged_latency,
            drift_records=merged_drift,
        )

    def register(self, name: str, index: BaseIndex) -> None:
        if not name:
            raise ValueError("index name must be non-empty")
        self._indices[name] = index

    def get(self, name: str) -> BaseIndex:
        if name not in self._indices:
            raise KeyError(f"Index '{name}' is not registered. Available: {sorted(self._indices)}")
        return self._indices[name]

    def query(self, name: str, **kwargs: Any) -> List[Any]:
        return self.get(name).query(**kwargs)

    def names(self) -> List[str]:
        return sorted(self._indices)
