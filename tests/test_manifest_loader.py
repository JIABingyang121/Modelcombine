from src.core.enums import TaskType, ModelLifecycleStage
from src.core.schema import ModelManifest
from src.core.manifest_loader import load_manifests, active_model_ids


def test_load_manifests_from_yaml():
    manifests = load_manifests()  # 默认读 configs/model_assets.yaml
    assert isinstance(manifests, dict)
    # yaml 里声明的模型都应被加载
    assert "xgboost_reg" in manifests
    m = manifests["xgboost_reg"]
    assert isinstance(m, ModelManifest)
    assert TaskType.FORECASTING in m.task_types


def test_active_models_filtered_by_lifecycle():
    manifests = load_manifests()
    ids = active_model_ids(manifests)
    # active_model_ids 只返回 lifecycle_stage == active 的模型
    for mid in ids:
        assert manifests[mid].lifecycle_stage is ModelLifecycleStage.ACTIVE


def test_manifest_defaults_when_fields_absent():
    # 缺 lifecycle_stage 的模型默认 registered（不可调度），缺 task_types 默认 forecasting
    m = ModelManifest.from_dict({"id": "foo"})
    assert m.lifecycle_stage is ModelLifecycleStage.REGISTERED
    assert m.task_types == [TaskType.FORECASTING]


def test_combination_models_derived_from_manifest():
    from src.eval import combination_utils
    from src.core.manifest_loader import load_manifests, active_model_ids
    expected = set(active_model_ids(load_manifests()))
    assert set(combination_utils.MODELS) == expected
