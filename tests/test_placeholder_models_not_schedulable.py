"""占位实现的模型不得可调度。

`PowerGPTModel` 是未完成的占位实现：`fit()` 只打印一行并置 `is_fitted_=True`
（真正的模型加载与微调是注释掉的），`predict()` 直接 `return np.zeros(len(X))`。

它却被注册进 `model_registry`，因此会作为候选进入预测矩阵。Task 5 的实测已经
留下痕迹：`powergpt` 的 `val_pred_std=0.0`（常数预测）、`val_mae=39401`
（≈负荷均值，正是拿全零去比的结果）——即它确实以全零预测参与过候选竞争。

零预测不是"效果差的模型"，而是**无效数据**：它会进入误差相关性、稳定性统计和
drift 判定，污染的是调度决策本身，不只是自己那一列。因此要求：不可注册、
不可静默产出零预测。
"""
import numpy as np
import pandas as pd
import pytest

from src.models.registry import model_registry


def test_powergpt_is_not_registered_as_schedulable_model():
    """占位实现不得出现在可用模型列表里，否则会自动进入候选池。"""
    assert "powergpt" not in model_registry.get_available_models()


def test_powergpt_cannot_be_constructed_via_registry():
    with pytest.raises(KeyError):
        model_registry.create("powergpt")


def test_powergpt_fit_fails_loudly_instead_of_returning_zeros():
    """即使直接构造，也必须显式报错，不能让全零预测流到下游。"""
    from src.models.deep_learning import PowerGPTModel

    X = pd.DataFrame({"f0": np.arange(10, dtype=float)})
    y = pd.Series(np.arange(10, dtype=float))

    with pytest.raises(NotImplementedError, match="placeholder"):
        PowerGPTModel().fit(X, y)


def test_powergpt_predict_fails_loudly():
    from src.models.deep_learning import PowerGPTModel

    X = pd.DataFrame({"f0": np.arange(10, dtype=float)})

    with pytest.raises(NotImplementedError, match="placeholder"):
        PowerGPTModel().predict(X)
