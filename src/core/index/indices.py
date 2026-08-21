"""默认轻量索引实现。"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..evidence import EvidenceRecord, EvidenceStore
from ..enums import ModelLifecycleStage, TaskType
from ..schema import ModelManifest
from .base import BaseIndex


def _coerce_task_type(task_type: TaskType | str | None) -> TaskType | None:
    if task_type is None or isinstance(task_type, TaskType):
        return task_type
    return TaskType(task_type)


def _signature_similarity(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    common = [k for k in set(a.keys()) & set(b.keys()) if isinstance(a[k], (int, float)) and isinstance(b[k], (int, float))]
    if not common:
        return 0.0
    normalized_diffs = []
    for key in common:
        denom = (abs(float(a[key])) + abs(float(b[key]))) / 2.0 + 1e-9
        normalized_diffs.append(abs(float(a[key]) - float(b[key])) / denom)
    mean_diff = sum(normalized_diffs) / len(normalized_diffs)
    return 1.0 / (1.0 + mean_diff)


class ScenarioIndex(BaseIndex):
    def __init__(self, records: Optional[Iterable[Dict[str, Any]]] = None, similarity_method: str = "combined"):
        self.records = list(records or [])
        self.similarity_method = similarity_method
        self._analyzer = None

    def add(self, record: Dict[str, Any]) -> None:
        self.records.append(record)

    @classmethod
    def from_historical_scenarios(
        cls,
        historical_scenarios: Sequence[Tuple[str, Dict[str, Any], Dict[str, Any]]],
        business_domain: str = "",
    ) -> "ScenarioIndex":
        records = []
        for scenario_id, signature, performance in historical_scenarios:
            records.append({
                "scenario_id": scenario_id,
                "signature": signature,
                "business_domain": business_domain or performance.get("business_domain", ""),
                "performance": performance,
            })
        return cls(records)

    def _score(self, signature: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
        try:
            if self._analyzer is None:
                from ...selector.scenario_similarity import PowerScenarioAnalyzer

                self._analyzer = PowerScenarioAnalyzer()
            return float(self._analyzer.calculate_scenario_similarity(
                dict(signature), dict(candidate), method=self.similarity_method
            ))
        except (ImportError, KeyError, TypeError, ValueError):
            return _signature_similarity(signature, candidate)

    def query(
        self,
        signature: Optional[Mapping[str, Any]] = None,
        business_domain: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for record in self.records:
            if business_domain and record.get("business_domain") != business_domain:
                continue
            item = dict(record)
            if signature is not None:
                item["_score"] = self._score(signature, record.get("signature", {}))
            else:
                item["_score"] = float(record.get("_score", record.get("score", 0.0)))
            if min_score is not None and item["_score"] < min_score:
                continue
            results.append(item)
        results.sort(key=lambda r: r.get("_score", 0.0), reverse=True)
        return results[:top_k] if top_k is not None else results


class CapabilityIndex(BaseIndex):
    def __init__(self, manifests: Optional[Mapping[str, ModelManifest]] = None):
        self.manifests = dict(manifests or {})

    def query(
        self,
        task_type: TaskType | str | None = None,
        business_domain: Optional[str] = None,
        available_features: Optional[Set[str]] = None,
        active_only: bool = True,
        **_: Any,
    ) -> List[ModelManifest]:
        task = _coerce_task_type(task_type)
        results: List[ModelManifest] = []
        for manifest in self.manifests.values():
            if active_only and manifest.lifecycle_stage is not ModelLifecycleStage.ACTIVE:
                continue
            if task is not None and task not in manifest.task_types:
                continue
            if business_domain and business_domain not in manifest.business_domains:
                continue
            required_features = set(manifest.input_constraints.get("features", []))
            if available_features is not None and not required_features.issubset(set(available_features)):
                continue
            results.append(manifest)
        return results


class PerformanceIndex(BaseIndex):
    def __init__(self, records: Optional[Iterable[Dict[str, Any]]] = None):
        self.records = list(records or [])

    @classmethod
    def from_historical_scenarios(
        cls,
        historical_scenarios: Sequence[Tuple[str, Dict[str, Any], Dict[str, Any]]],
    ) -> "PerformanceIndex":
        records = []
        for scenario_id, _, performance in historical_scenarios:
            record = {"scenario_id": scenario_id, **dict(performance)}
            records.append(record)
        return cls(records)

    def query(
        self,
        scenario_id: Optional[str] = None,
        model_id: Optional[str] = None,
        path_id: Optional[str] = None,
        top_k: Optional[int] = None,
        sort_key: str = "score",
        descending: bool = True,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        results = []
        for record in self.records:
            if scenario_id and record.get("scenario_id") != scenario_id:
                continue
            if model_id and record.get("model_id") != model_id:
                continue
            if path_id and record.get("path_id") != path_id:
                continue
            results.append(dict(record))
        results.sort(key=lambda r: r.get(sort_key, 0.0), reverse=descending)
        return results[:top_k] if top_k is not None else results


class EvidenceIndex(BaseIndex):
    def __init__(self, store: Optional[EvidenceStore] = None):
        self.store = store or EvidenceStore()

    def add(self, record: EvidenceRecord) -> str:
        return self.store.add(record)

    def query(
        self,
        scenario_id: Optional[str] = None,
        drift_event: Optional[str] = None,
        data_slice_prefix: Optional[str] = None,
        **_: Any,
    ) -> List[EvidenceRecord]:
        records = [EvidenceRecord.from_dict(item) for item in self.store.to_list()]
        if scenario_id:
            records = [r for r in records if r.scenario_id == scenario_id]
        if drift_event:
            records = [r for r in records if drift_event in r.drift_events]
        if data_slice_prefix:
            records = [r for r in records if r.data_slice_ref.startswith(data_slice_prefix)]
        return records


def _first_float(record: Mapping[str, Any], keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        if key in record and record[key] is not None:
            try:
                return float(record[key])
            except (TypeError, ValueError):
                continue
    return float(default)


class LatencyIndex(BaseIndex):
    """轻量资源/延迟索引，Phase 3-D 替换 DeferredIndex。"""

    def __init__(self, records: Optional[Iterable[Dict[str, Any]]] = None):
        self.records = list(records or [])

    def add(self, record: Dict[str, Any]) -> None:
        self.records.append(dict(record))

    def query(
        self,
        model_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        max_latency_ms: Optional[float] = None,
        top_k: Optional[int] = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        results = []
        for record in self.records:
            if model_id and record.get("model_id") != model_id:
                continue
            if scenario_id and record.get("scenario_id") != scenario_id:
                continue
            latency_ms = _first_float(
                record,
                ("latency_ms", "p50_latency_ms", "p95_latency_ms"),
            )
            if max_latency_ms is not None and latency_ms > float(max_latency_ms):
                continue
            item = dict(record)
            item["_latency_ms"] = latency_ms
            results.append(item)
        results.sort(key=lambda r: r.get("_latency_ms", 0.0))
        return results[:top_k] if top_k is not None else results


class DriftIndex(BaseIndex):
    """轻量漂移索引，承接 PSI/KS 等统计信号。"""

    def __init__(self, records: Optional[Iterable[Dict[str, Any]]] = None):
        self.records = list(records or [])

    def add(self, record: Dict[str, Any]) -> None:
        self.records.append(dict(record))

    def query(
        self,
        model_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        drift_level: Optional[str] = None,
        min_psi: Optional[float] = None,
        feature: Optional[str] = None,
        top_k: Optional[int] = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        results = []
        for record in self.records:
            if model_id and record.get("model_id") != model_id:
                continue
            if scenario_id and record.get("scenario_id") != scenario_id:
                continue
            if drift_level and record.get("drift_level") != drift_level:
                continue
            if feature and record.get("feature") != feature:
                continue
            psi = _first_float(record, ("psi", "median_psi", "max_psi"))
            if min_psi is not None and psi < float(min_psi):
                continue
            item = dict(record)
            item["_psi"] = psi
            results.append(item)
        results.sort(key=lambda r: r.get("_psi", 0.0), reverse=True)
        return results[:top_k] if top_k is not None else results


class DeferredIndex(BaseIndex):
    """Phase 2 占位索引：接口先固定，内部逻辑 Phase 3 接入。"""

    def __init__(self, name: str, reason: str):
        self.name = name
        self.reason = reason

    def query(self, **kwargs: Any) -> List[Any]:
        raise NotImplementedError(f"Index '{self.name}' is intentionally deferred. {self.reason}")


class ConflictIndex(BaseIndex):
    def __init__(self, edges: Optional[Iterable[Dict[str, Any]]] = None):
        self.edges = list(edges or [])

    def query(
        self,
        model_id: Optional[str] = None,
        min_weight: float = 0.0,
        relation_type: str = "conflict",
        **_: Any,
    ) -> List[Dict[str, Any]]:
        results = []
        for edge in self.edges:
            if relation_type and edge.get("type") != relation_type:
                continue
            if model_id and edge.get("source") != model_id and edge.get("target") != model_id:
                continue
            if float(edge.get("weight", 0.0)) < min_weight:
                continue
            results.append(dict(edge))
        return results
