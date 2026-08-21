import pytest
from src.core.enums import TaskType, ModelLifecycleStage
from src.core.schema import DataContract, ScenarioDefinition


def test_data_contract_validate_ok():
    dc = DataContract(
        required_columns={"timestamp": "datetime", "load": "float"},
        freq="H", min_samples=100, business_domain="load_forecast",
    )
    ok, missing = dc.validate_columns({"timestamp", "load", "region"})
    assert ok is True
    assert missing == set()


def test_data_contract_validate_missing():
    dc = DataContract(
        required_columns={"timestamp": "datetime", "load": "float"},
        freq="H", min_samples=100, business_domain="load_forecast",
    )
    ok, missing = dc.validate_columns({"timestamp"})
    assert ok is False
    assert missing == {"load"}


def test_scenario_definition_auto_id():
    dc = DataContract(required_columns={"load": "float"}, freq="H",
                      min_samples=50, business_domain="load_forecast")
    sd = ScenarioDefinition(
        task_type=TaskType.FORECASTING, business_domain="load_forecast",
        data_contract=dc, target_schema={"load": "float"},
        primary_metric="MAE", signature_features=["mean_load", "cv_load"],
        signature={"mean_load": 100.0, "cv_load": 0.2}, region="PJME",
    )
    # scenario_id 由 signature 自动派生且稳定，带 region 前缀
    assert sd.scenario_id.startswith("PJME_")
    assert sd.scenario_id == sd.compute_id()


def test_primary_metric_must_match_task_type():
    dc = DataContract(required_columns={"load": "float"}, freq="H",
                      min_samples=50, business_domain="load_forecast")
    with pytest.raises(ValueError):
        ScenarioDefinition(
            task_type=TaskType.FORECASTING, business_domain="load_forecast",
            data_contract=dc, target_schema={"load": "float"},
            primary_metric="AUC",  # 分类指标用在回归任务 → 报错
            signature_features=["mean_load"], signature={"mean_load": 1.0},
        )
