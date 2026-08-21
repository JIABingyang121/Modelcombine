from src.core.enums import TaskType, ModelLifecycleStage, can_transition, is_schedulable


def test_task_type_values():
    assert TaskType.FORECASTING.value == "forecasting"
    assert {t.value for t in TaskType} == {
        "forecasting", "classification", "ranking", "anomaly_detection",
    }


def test_only_active_is_schedulable():
    assert is_schedulable(ModelLifecycleStage.ACTIVE) is True
    for stage in ModelLifecycleStage:
        if stage is not ModelLifecycleStage.ACTIVE:
            assert is_schedulable(stage) is False


def test_valid_transition():
    assert can_transition(ModelLifecycleStage.SHADOW, ModelLifecycleStage.ACTIVE) is True
    assert can_transition(ModelLifecycleStage.REGISTERED, ModelLifecycleStage.VALIDATED) is True


def test_invalid_transition():
    # 不能从 registered 直接跳到 active
    assert can_transition(ModelLifecycleStage.REGISTERED, ModelLifecycleStage.ACTIVE) is False
    # deprecated 是终态
    assert can_transition(ModelLifecycleStage.DEPRECATED, ModelLifecycleStage.ACTIVE) is False
