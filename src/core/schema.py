"""跨系统共享的契约对象（ADR-001 ④）。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple
from .enums import TaskType, ModelLifecycleStage
from .scenario_id import compute_scenario_id


# 各任务类型允许的主指标（防止指标与任务类型错配）
_ALLOWED_METRICS = {
    TaskType.FORECASTING: {"MAE", "RMSE", "MAPE", "MASE", "R2"},
    TaskType.CLASSIFICATION: {"AUC", "F1", "PRECISION", "RECALL", "ACCURACY"},
    TaskType.RANKING: {"NDCG", "MAP", "MRR"},
    TaskType.ANOMALY_DETECTION: {"AUC", "F1", "PRECISION", "RECALL"},
}


@dataclass
class DataContract:
    required_columns: Dict[str, str]     # {列名: 类型}
    freq: str                            # "H" | "D" | ...
    min_samples: int
    business_domain: str

    def validate_columns(self, actual_columns: Set[str]) -> Tuple[bool, Set[str]]:
        """校验实际数据列是否覆盖契约要求。返回 (是否通过, 缺失列集合)。"""
        missing = set(self.required_columns.keys()) - set(actual_columns)
        return (len(missing) == 0, missing)


@dataclass
class ScenarioDefinition:
    task_type: TaskType
    business_domain: str
    data_contract: DataContract
    target_schema: Dict[str, str]
    primary_metric: str
    signature_features: List[str]
    signature: Dict[str, Any]
    region: str = ""
    scenario_id: str = ""

    def __post_init__(self):
        if self.primary_metric.upper() not in _ALLOWED_METRICS.get(self.task_type, set()):
            raise ValueError(
                f"primary_metric={self.primary_metric} 与 task_type={self.task_type.value} 不匹配；"
                f"允许: {sorted(_ALLOWED_METRICS.get(self.task_type, set()))}"
            )
        if not self.scenario_id:
            self.scenario_id = self.compute_id()

    def compute_id(self) -> str:
        return compute_scenario_id(self.signature, prefix=self.region)


@dataclass
class ModelManifest:
    model_id: str
    task_types: List[TaskType] = field(default_factory=lambda: [TaskType.FORECASTING])
    business_domains: List[str] = field(default_factory=lambda: ["general"])
    input_constraints: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, str] = field(default_factory=dict)
    resource_cost: Dict[str, str] = field(default_factory=dict)
    lifecycle_stage: ModelLifecycleStage = ModelLifecycleStage.REGISTERED

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelManifest":
        raw_tasks = d.get("task_types") or ["forecasting"]
        task_types = [TaskType(t) for t in raw_tasks]
        raw_stage = d.get("lifecycle_stage", "registered")
        return cls(
            model_id=d["id"],
            task_types=task_types,
            business_domains=d.get("business_dims") or d.get("business_domains") or ["general"],
            input_constraints=d.get("input_constraints", {}),
            output_schema=d.get("output_schema_struct") or {},
            resource_cost=d.get("resource_cost", {}),
            lifecycle_stage=ModelLifecycleStage(raw_stage),
        )
