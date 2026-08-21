from src.models.registry import model_registry
from src.core.manifest_loader import load_manifests


def test_every_active_manifest_model_has_constructor():
    """yaml 里 active 的模型必须有对应构造器，否则调度时会崩。"""
    manifests = load_manifests()
    registered = set(model_registry.get_available_models())
    for mid, m in manifests.items():
        if m.lifecycle_stage.value == "active":
            assert mid in registered, f"active 模型 {mid} 缺少构造器注册"


def test_check_manifest_sync_returns_report():
    report = model_registry.check_manifest_sync()
    # 报告包含两类漂移：在 registry 但不在 manifest / 在 manifest 但不在 registry
    assert "registered_not_in_manifest" in report
    assert "manifest_not_registered" in report
    # informer/autoformer/powergpt 在 registry 但 yaml 无声明
    assert "informer" in report["registered_not_in_manifest"]
