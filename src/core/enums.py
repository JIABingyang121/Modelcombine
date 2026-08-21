"""跨系统共享的枚举与模型生命周期状态机（ADR-001 ③）。"""
from __future__ import annotations
from enum import Enum


class TaskType(str, Enum):
    FORECASTING = "forecasting"
    CLASSIFICATION = "classification"
    RANKING = "ranking"
    ANOMALY_DETECTION = "anomaly_detection"


class ModelLifecycleStage(str, Enum):
    REGISTERED = "registered"
    VALIDATED = "validated"
    BENCHMARKED = "benchmarked"
    SHADOW = "shadow"
    PROBATION = "probation"
    ACTIVE = "active"
    DISABLED = "disabled"
    DEPRECATED = "deprecated"


# 允许的状态转移（ADR-001 ③ 的编码）
_ALLOWED_TRANSITIONS = {
    ModelLifecycleStage.REGISTERED: {ModelLifecycleStage.VALIDATED, ModelLifecycleStage.DISABLED},
    ModelLifecycleStage.VALIDATED: {ModelLifecycleStage.BENCHMARKED, ModelLifecycleStage.DISABLED},
    ModelLifecycleStage.BENCHMARKED: {ModelLifecycleStage.SHADOW, ModelLifecycleStage.DISABLED},
    ModelLifecycleStage.SHADOW: {ModelLifecycleStage.PROBATION, ModelLifecycleStage.ACTIVE, ModelLifecycleStage.DISABLED},
    ModelLifecycleStage.PROBATION: {ModelLifecycleStage.ACTIVE, ModelLifecycleStage.SHADOW, ModelLifecycleStage.DISABLED},
    ModelLifecycleStage.ACTIVE: {ModelLifecycleStage.DEPRECATED, ModelLifecycleStage.PROBATION, ModelLifecycleStage.DISABLED},
    ModelLifecycleStage.DISABLED: {ModelLifecycleStage.REGISTERED},
    ModelLifecycleStage.DEPRECATED: set(),  # 终态
}


def can_transition(src: ModelLifecycleStage, dst: ModelLifecycleStage) -> bool:
    """判断状态转移是否合法。"""
    return dst in _ALLOWED_TRANSITIONS.get(src, set())


def is_schedulable(stage: ModelLifecycleStage) -> bool:
    """只有 active 的模型允许参与真实调度（进入候选池）。"""
    return stage is ModelLifecycleStage.ACTIVE
