"""默认发布路径的可重复性（合一计划 Task 8.1）。

**背景**：Task 8 把 `run.py` 默认决策路径切到 Protocol B 后，两轮完整运行的结果
不一致——MAE 306.8151 vs 306.8043、RMSE 419.8455 vs 419.8429，远超计划验收条件②
要求的 `1e-8`。已定位根因：

1. 主流程未设任何全局种子；
2. `informer/autoformer/powergpt` 完全没有种子；
3. `lgbm_reg` 即使 `random_state=42` 仍非确定性——`configs/pipeline.yaml` 里
   `n_jobs: -1` 直接进模型且**优先于**环境变量（`LGBMModel.__init__` 只在
   `"n_jobs" not in lgbm_params` 时才读 `MODELCOMBINE_TREE_N_JOBS`），
   而 `MODELCOMBINE_USE_GPU` 默认 True 会让它请求 `device_type="gpu"`。
   LightGBM 官方说明 `deterministic=true` **仅适用于 CPU**，GPU 训练不保证重复。

因此修复不能只是 `n_jobs=1`：必须固定 CPU + `deterministic` + 逐候选重置随机源。
本模块覆盖本机可验证的树模型与预测矩阵契约；深度模型的真实重复训练属于服务器
实验，在完成前被明确排除出严格默认候选池，而不是在本机测试中长时间训练。

注：树模型用例使用**真实模型库**训练。项目历史上出现过"手造假对象测试全绿、真实
路径空转"的教训（见 §6.1 树模型不确定性），可重复性尤其不能只用假对象验证。
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.models.registry import model_registry
from src.utils.io import load_yaml

PIPELINE_CFG = load_yaml("configs/pipeline.yaml")


def _hash(arr) -> str:
    return hashlib.sha256(np.asarray(arr, dtype=float).tobytes()).hexdigest()[:16]


def _dataset(n: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(6)})
    y = pd.Series(3.0 * X["f0"] - 2.0 * X["f1"] + rng.normal(0, 1.0, n))
    return X, y


def _train_twice(model_id: str, params: dict, X, y):
    """同参数、同数据训练两次，返回两次预测的哈希。"""
    from src.utils.determinism import seed_for_candidate

    hashes = []
    for _ in range(2):
        # 每轮训练前按固定派生种子重置随机源，避免上一候选的随机消耗影响本候选
        seed_for_candidate(model_id, stage="val")
        model = model_registry.create(model_id, **params)
        model.fit(X, y)
        hashes.append(_hash(model.predict(X)))
    return hashes


# --- 随机源重置工具 ---------------------------------------------------------


def test_seed_helper_resets_python_numpy_and_torch():
    """逐候选派生种子必须同时覆盖 Python / NumPy / PyTorch 三个随机源。"""
    import random

    from src.utils.determinism import seed_for_candidate

    seed_for_candidate("m", stage="val")
    first = (random.random(), float(np.random.rand()))
    seed_for_candidate("m", stage="val")
    second = (random.random(), float(np.random.rand()))
    assert first == second, "同一 (模型, 阶段) 必须复现同一随机流"

    seed_for_candidate("m", stage="test")
    other = (random.random(), float(np.random.rand()))
    assert other != first, "不同阶段应派生不同种子，避免 val/test 用同一随机流"

    torch = pytest.importorskip("torch")
    seed_for_candidate("m", stage="val")
    t1 = torch.rand(4).tolist()
    seed_for_candidate("m", stage="val")
    t2 = torch.rand(4).tolist()
    assert t1 == t2, "PyTorch 随机源未被重置"


def test_seed_is_isolated_across_candidates():
    """前一个候选消耗随机数，不得改变后一个候选的随机流。"""
    from src.utils.determinism import seed_for_candidate

    seed_for_candidate("b", stage="val")
    baseline = float(np.random.rand())

    seed_for_candidate("a", stage="val")
    _ = np.random.rand(37)  # a 消耗随机数
    seed_for_candidate("b", stage="val")
    after = float(np.random.rand())

    assert baseline == after


# --- 真实模型可重复性 -------------------------------------------------------


def test_real_lightgbm_is_deterministic_under_release_config():
    """真实 LightGBM 在默认发布配置下两次训练必须逐值一致。

    红灯基线（Task 8 实测）：两次预测哈希 bf92e225 / 5beaa97a 不同。
    """
    X, y = _dataset()
    params = dict(PIPELINE_CFG["models"].get("lgbm_reg", {}))

    h1, h2 = _train_twice("lgbm_reg", params, X, y)

    assert h1 == h2, f"lgbm_reg 非确定性：{h1} != {h2}"


@pytest.mark.parametrize("model_id", ["xgboost_reg", "catboost_reg"])
def test_other_tree_models_stay_deterministic(model_id):
    """xgboost/catboost 原本即确定，修复不得破坏它们。"""
    X, y = _dataset()
    params = dict(PIPELINE_CFG["models"].get(model_id, {}))

    h1, h2 = _train_twice(model_id, params, X, y)

    assert h1 == h2, f"{model_id} 非确定性：{h1} != {h2}"


@pytest.mark.parametrize("model_id", ["informer", "autoformer", "powergpt"])
def test_unverified_deep_models_are_explicitly_excluded_from_default_pool(model_id):
    """未在服务器完成可重复性验收的深度候选不能进入本机默认池。

    这里不训练深度模型：真实重复训练属于服务器实验，不得把长耗时 GPU 实验
    伪装成本机单元测试。服务器验收通过后，才可以修改候选策略并补回对应证据。
    """
    from src.utils.determinism import default_candidate_models, default_exclusion_reason

    assert model_id not in default_candidate_models()
    assert default_exclusion_reason(model_id)


def test_release_lightgbm_config_uses_cpu_deterministic_mode():
    """逐值验收使用服务器基准验证过的 CPU 单线程确定性模式。"""
    params = PIPELINE_CFG["models"]["lgbm_reg"]

    assert params["device_type"] == "cpu"
    assert params["deterministic"] is True
    assert params["force_col_wise"] is True
    assert params["n_jobs"] == 1


def test_default_candidate_pool_excludes_every_declared_nonrelease_model():
    """凡被标记为不允许默认发布的候选，都不得残留在默认候选池。"""
    from src.utils.determinism import default_candidate_models, default_exclusion_reason

    pool = default_candidate_models()
    assert pool, "默认候选池不得为空"
    for model_id in pool:
        assert not default_exclusion_reason(model_id), (
            f"{model_id} 被标记为不允许默认发布却仍在默认池中："
            f"{default_exclusion_reason(model_id)}"
        )


# --- 预测矩阵级可重复性 -----------------------------------------------------


def test_prediction_bundle_is_reproducible():
    """同一输入两次构建候选预测矩阵，各候选列必须逐值一致。"""
    from src.pipeline.prediction_pool import build_region_prediction_bundle
    from src.utils.determinism import default_candidate_models

    ts = pd.date_range("2026-01-01", periods=720, freq="h")
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({
        "timestamp": ts,
        "region": "R1",
        "region_type": "residential",
        "load": 100 + 20 * np.sin(np.arange(720) * 2 * np.pi / 24) + rng.normal(0, 2, 720),
        "hour": ts.hour,
        "dow": ts.dayofweek,
        "is_weekend": ts.dayofweek.isin([5, 6]).astype(int),
        "lag_1": np.linspace(99, 199, 720),
    })
    train, test = frame.iloc[:648], frame.iloc[648:]
    pool = [m for m in default_candidate_models() if m in {"lgbm_reg", "xgboost_reg", "catboost_reg"}]

    sigs = []
    determinism_meta = []
    for _ in range(2):
        bundle = build_region_prediction_bundle(
            region="R1", train=train, test=test,
            candidate_models=pool, validation_days=5,
            model_params=PIPELINE_CFG.get("models", {}),
        )
        sigs.append({m: _hash(bundle.df_test[m].values) for m in bundle.model_cols})
        determinism_meta.append(bundle.metadata.get("determinism"))

    assert sigs[0] == sigs[1], f"预测矩阵不可重复：{sigs[0]} != {sigs[1]}"
    assert determinism_meta == [
        {"global_seed": 42, "candidate_seed_strategy": "sha256(global_seed|model_id|stage)"},
        {"global_seed": 42, "candidate_seed_strategy": "sha256(global_seed|model_id|stage)"},
    ]
