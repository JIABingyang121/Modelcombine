"""轻量索引接口骨架。

Phase 2 先稳定调用接口；内部目前使用 list 过滤/排序，未来可替换为向量库或倒排索引。
"""
from .base import BaseIndex
from .indices import (
    CapabilityIndex,
    ConflictIndex,
    DeferredIndex,
    DriftIndex,
    EvidenceIndex,
    LatencyIndex,
    PerformanceIndex,
    ScenarioIndex,
)
from .manager import IndexManager
from .signal_loaders import (
    build_signal_records_from_reports,
    load_drift_records_from_kg_results,
    load_latency_records_from_ablation_profile,
    load_latency_records_from_kg_results,
)

__all__ = [
    "BaseIndex",
    "CapabilityIndex",
    "ConflictIndex",
    "DeferredIndex",
    "DriftIndex",
    "EvidenceIndex",
    "IndexManager",
    "LatencyIndex",
    "PerformanceIndex",
    "ScenarioIndex",
    "build_signal_records_from_reports",
    "load_drift_records_from_kg_results",
    "load_latency_records_from_ablation_profile",
    "load_latency_records_from_kg_results",
]
