"""默认发布路径的可重复性支持（合一计划 Task 8.1）。

Task 8 把 `run.py` 默认切到 Protocol B 后，两轮完整运行结果不一致
（MAE 306.8151 vs 306.8043），未达计划验收条件②要求的 `1e-8`。根因有三层，
本模块只解决"随机源"这一层，另两层分别由 `configs/pipeline.yaml` 的 LightGBM
发布参数与默认候选池收敛处理：

1. 主流程未设全局种子；
2. 候选之间共用同一随机流——前一个候选消耗多少随机数会改变后一个候选的结果；
3. `lgbm_reg` 默认请求 GPU 且 `n_jobs=-1`，而 LightGBM 官方说明
   `deterministic=true` **仅适用于 CPU**，GPU 训练不保证重复运行一致。

因此这里提供**按 (模型, 阶段) 派生的确定性种子**，在每个候选的每个训练阶段前
重置 Python / NumPy / PyTorch 随机源，使候选之间互不干扰。
"""
from __future__ import annotations

import hashlib
import os
import random
from typing import Dict, List, Optional

import numpy as np

#: 全局基准种子；可用环境变量覆盖以做敏感性检查，但默认发布路径固定。
GLOBAL_SEED_ENV = "MODELCOMBINE_GLOBAL_SEED"
DEFAULT_GLOBAL_SEED = 42

#: 不允许进入默认发布路径的候选 -> 可审计原因。
#:
#: 这不等同于宣称每个候选“已被证明随机”：其中深度模型必须完成服务器重复运行
#: 验收才可纳入默认池。原因必须是代码可直接核实的事实，或指向待完成的服务器
#: 验收，不能把尚未跑过的实验伪写成既成结论。
DEFAULT_CANDIDATE_EXCLUSIONS: Dict[str, str] = {
    "informer": (
        "server release-environment repeatability validation is pending; "
        "not admitted to the strict default pool"
    ),
    "autoformer": (
        "Task 5 observed CUDA OOM on small/medium windows; server repeatability and "
        "resource validation are pending, so it is not admitted to the strict default pool"
    ),
    "powergpt": (
        "the registered implementation is a placeholder: fit performs no training and "
        "predict returns zeros; it is not a valid default forecasting candidate"
    ),
}

#: 非候选模型（集成器等），始终不进候选池。
_NON_CANDIDATE_MARKERS = ("blender",)


def global_seed() -> int:
    raw = os.environ.get(GLOBAL_SEED_ENV, "").strip()
    if not raw:
        return DEFAULT_GLOBAL_SEED
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_GLOBAL_SEED


def derive_seed(model_id: str, stage: str = "") -> int:
    """由 (全局种子, 模型, 阶段) 派生稳定种子。

    用 sha256 而非内置 `hash()`：后者受 PYTHONHASHSEED 影响，跨进程不稳定
    （项目此前已因内置 hash 踩过坑，见 `src/core/scenario_id.py`）。
    """
    key = f"{global_seed()}|{model_id}|{stage}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def seed_everything(seed: int) -> None:
    """重置 Python / NumPy / PyTorch 随机源。torch 缺失时静默跳过。"""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_for_candidate(model_id: str, stage: str = "") -> int:
    """在训练某候选的某阶段之前调用，重置随机源并返回所用种子。"""
    seed = derive_seed(model_id, stage)
    seed_everything(seed)
    return seed


def default_exclusion_reason(model_id: str) -> Optional[str]:
    """返回模型未获准进入严格默认候选池的原因；获准则返回 ``None``。"""
    return DEFAULT_CANDIDATE_EXCLUSIONS.get(model_id)


def default_candidate_models(all_models: Optional[List[str]] = None) -> List[str]:
    """默认发布路径的候选池：剔除集成器与未获准发布的候选。

    被剔除的模型不会静默消失——原因记录在
    :data:`DEFAULT_CANDIDATE_EXCLUSIONS`，调用方可经
    :func:`default_exclusion_reason` 取出并写进报告。
    """
    if all_models is None:
        from src.models.registry import model_registry

        all_models = model_registry.get_available_models()
    return [
        m
        for m in all_models
        if not any(marker in m.lower() for marker in _NON_CANDIDATE_MARKERS)
        and m not in DEFAULT_CANDIDATE_EXCLUSIONS
    ]
