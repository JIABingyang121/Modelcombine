"""模型清单单一真源：从 configs/model_assets.yaml 加载 ModelManifest。"""
from __future__ import annotations
import os
from typing import Dict, List
import yaml
from .schema import ModelManifest
from .enums import ModelLifecycleStage

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_YAML = os.path.join(_PROJECT_ROOT, "configs", "model_assets.yaml")


def load_manifests(path: str = _DEFAULT_YAML) -> Dict[str, ModelManifest]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    manifests: Dict[str, ModelManifest] = {}
    for m in cfg.get("models", []):
        manifests[m["id"]] = ModelManifest.from_dict(m)
    return manifests


def active_model_ids(manifests: Dict[str, ModelManifest]) -> List[str]:
    """只返回 lifecycle_stage == active 的模型 id（可调度池）。"""
    return [mid for mid, m in manifests.items()
            if m.lifecycle_stage is ModelLifecycleStage.ACTIVE]
