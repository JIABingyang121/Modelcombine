"""训练模型与组合预测器产物的 pickle 保存与加载。

只用标准库 ``pickle``（项目既有的模型序列化方式）。保存后立即真实反序列化一次
以暴露不可 pickle 的对象；不建立多套序列化回退，不吞异常。
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Union

_PathLike = Union[str, Path]


def save_artifact(obj: Any, path: _PathLike) -> Path:
    """把 ``obj`` pickle 到 ``path``，随后立即重新加载一次做真实性校验。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with path.open("rb") as handle:
        pickle.load(handle)
    return path


def load_artifact(path: _PathLike) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
