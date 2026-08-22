from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Callable
import yaml

from .implementations import (
    XGBModel, LGBMModel, CatBoostModel, 
    ProphetModel, ARIMAModel, SeasonalNaiveModel, PowerLoadDifferenceModel, 
    MultiModalFusionModel, WeightedBlender, StackingBlender
)
from .deep_learning import InformerModel, AutoformerModel


@dataclass
class ModelSpec:
    id: str
    type: str
    family: str
    params: Dict[str, Any]


class ModelRegistry:
    def __init__(self):
        self._constructors: Dict[str, Callable[..., Any]] = {}
        self._register_default_models()

    def register(self, key: str, ctor: Callable[..., Any]):
        self._constructors[key] = ctor

    def create(self, key: str, **kwargs) -> Any:
        if key not in self._constructors:
            raise KeyError(f"Model '{key}' is not registered. Available models: {list(self._constructors.keys())}")
        return self._constructors[key](**kwargs)
    
    def _register_default_models(self):
        """注册默认的电力预测模型"""
        # 基础回归模型
        self.register("xgboost_reg", XGBModel)
        self.register("lgbm_reg", LGBMModel)
        self.register("catboost_reg", CatBoostModel)
        
        # 时序预测模型
        self.register("prophet", ProphetModel)
        self.register("arima", ARIMAModel)
        self.register("seasonal_naive", SeasonalNaiveModel)
        
        # 电力专用模型
        self.register("power_difference", PowerLoadDifferenceModel)
        self.register("multimodal_fusion", MultiModalFusionModel)
        
        # 集成模型
        self.register("weighted_blender", WeightedBlender)
        self.register("stacking_blender", StackingBlender)

        # 深度学习/大模型
        self.register("informer", InformerModel)
        self.register("autoformer", AutoformerModel)
        # powergpt 未注册：其实现为占位（fit 不训练、predict 返回全零），
        # 注册即会作为候选进入预测矩阵。实现完成后再恢复注册。
    
    def get_available_models(self) -> list[str]:
        """获取所有可用模型列表"""
        return list(self._constructors.keys())

    def check_manifest_sync(self) -> Dict[str, list]:
        """对比 registry 注册的构造器与 model_assets.yaml manifest，返回漂移报告。"""
        from ..core.manifest_loader import load_manifests
        manifest_ids = set(load_manifests().keys())
        registered = set(self._constructors.keys())
        # 排除集成器（blender 不是可调度模型）
        registered_models = {k for k in registered if "blender" not in k}
        return {
            "registered_not_in_manifest": sorted(registered_models - manifest_ids),
            "manifest_not_registered": sorted(manifest_ids - registered_models),
        }

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """获取模型信息"""
        model_info = {
            "xgboost_reg": {
                "type": "regression",
                "family": "tree_boosting", 
                "description": "XGBoost回归模型，适合特征丰富的场景",
                "best_for": ["特征工程后的数据", "中等规模数据集"]
            },
            "lgbm_reg": {
                "type": "regression",
                "family": "tree_boosting",
                "description": "LightGBM回归模型，训练速度快",
                "best_for": ["大规模数据集", "快速原型开发"]
            },
            "catboost_reg": {
                "type": "regression", 
                "family": "tree_boosting",
                "description": "CatBoost回归模型，对类别特征友好",
                "best_for": ["包含类别特征的数据", "自动特征处理"]
            },
            "prophet": {
                "type": "ts_forecast",
                "family": "bayesian_additive",
                "description": "Prophet时序预测，强季节性处理",
                "best_for": ["强季节性数据", "节假日效应", "趋势变化"]
            },
            "arima": {
                "type": "ts_forecast", 
                "family": "stats",
                "description": "ARIMA时序预测，经典统计方法",
                "best_for": ["短期预测", "平稳时序", "基线模型"]
            },
            "seasonal_naive": {
                "type": "ts_forecast",
                "family": "baseline",
                "description": "Seasonal Naive基线模型，使用上一个季节周期的观测值预测",
                "best_for": ["强季节性", "快速基线对比", "稳定周期"]
            },
            "power_difference": {
                "type": "regression",
                "family": "power_specific",
                "description": "电力负荷差异分析模型",
                "best_for": ["多区域用电差异", "峰谷电价分析", "区域特征建模"]
            },
            "multimodal_fusion": {
                "type": "fusion",
                "family": "power_specific", 
                "description": "多模态数据融合模型",
                "best_for": ["多数据源融合", "用电+天气+时间", "跨模态预测"]
            },
            "informer": {
                "type": "deep_learning",
                "family": "transformer",
                "description": "Informer长序列预测模型",
                "best_for": ["长序列预测", "复杂时序依赖", "高精度要求"]
            },
            "autoformer": {
                "type": "deep_learning",
                "family": "transformer",
                "description": "Autoformer自相关分解模型",
                "best_for": ["长序列预测", "趋势分解", "周期性明显数据"]
            },
            "powergpt": {
                "type": "deep_learning",
                "family": "llm",
                "description": "PowerGPT电力领域大模型",
                "best_for": ["少样本学习", "复杂场景推理", "多任务处理"]
            }
        }
        return model_info.get(model_id, {"type": "unknown", "family": "unknown", "description": "未知模型"})


def load_asset_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 全局模型注册表实例
model_registry = ModelRegistry()

# 启动时对 manifest 做一致性检查（仅告警，不阻断）
try:
    _sync = model_registry.check_manifest_sync()
    if _sync["registered_not_in_manifest"]:
        print(f"[registry][WARN] 已注册但 model_assets.yaml 无声明: {_sync['registered_not_in_manifest']}")
    if _sync["manifest_not_registered"]:
        print(f"[registry][WARN] manifest 声明但未注册构造器: {_sync['manifest_not_registered']}")
except Exception as _e:  # 校验失败不应阻断主流程
    print(f"[registry][WARN] manifest 一致性检查跳过: {_e}")
