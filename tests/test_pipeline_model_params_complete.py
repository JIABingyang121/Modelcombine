"""`configs/pipeline.yaml` 的 `models:` 段必须覆盖被引用的模型。

`seasonal_naive` 此前只出现在 `selection.weights` 中、`models:` 段没有它，
因此参数表里取不到它——Task 7 v4 九任务中它 9/9 全部记为
`missing_pred_file_val_or_test`，等于基准里根本没有 seasonal naive。

**本模块只覆盖"配置是否声明"，不能证明"训练时真的会跑"。** 早期版本的
docstring 曾断言 `load_pipeline_model_params` 决定训练集合——**这是错的**：
该函数只取参数，训练集合当时由 `run_dataset` 内的硬编码列表决定，所以本文件
6 个用例全绿时 `seasonal_naive` 依然没被训练。真正的集成断言见
`tests/test_train_baselines_model_set.py`。

这对结论口径有直接影响：CLAUDE.md 明确要求区分"含 naive"与"不含 seasonal_naive"
两种基准，不能混用。
"""
import numpy as np
import pandas as pd

from src.models.registry import model_registry
from src.utils.io import load_yaml

CFG = load_yaml("configs/pipeline.yaml")


def test_every_weighted_model_has_params_entry():
    """凡是 selection.weights 引用的模型，都必须能被基线训练取到参数。"""
    weighted = set(CFG["selection"]["weights"])
    declared = set(CFG["models"])

    missing = sorted(weighted - declared)

    assert not missing, (
        f"这些模型被 selection.weights 引用但 models: 段未声明，"
        f"train_baselines 不会训练它们: {missing}"
    )


def test_seasonal_naive_is_trainable_with_declared_params():
    """seasonal_naive 必须能用声明的参数真实训练并产出非常数预测。"""
    params = CFG["models"]["seasonal_naive"]
    model = model_registry.create("seasonal_naive", **params)

    n = 240
    ts = pd.date_range("2026-01-01", periods=n, freq="h")
    y = pd.Series(100 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24))
    X = pd.DataFrame({"hour": ts.hour})

    model.fit(X, y)
    pred = np.asarray(model.predict(X), dtype=float)

    assert len(pred) == n
    assert np.isfinite(pred).all()
    # 季节性基线不应退化成常数——那说明周期参数没生效
    assert float(np.std(pred)) > 0.0
