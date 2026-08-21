"""稳定的场景 id 生成（替代不稳定的 Python 内置 hash()）。

修复 src/pipeline/main.py 的活跃 bug：内置 hash() 因字符串哈希随机化
（PYTHONHASHSEED）跨进程不稳定，导致图谱持久化与历史反馈的跨运行匹配失效。
"""
from __future__ import annotations
import hashlib
import json
from typing import Any, Mapping


def compute_scenario_id(fields: Mapping[str, Any], prefix: str = "") -> str:
    """根据场景字段计算确定性 id。

    Args:
        fields: 场景字段（如场景签名 dict）。必须是可 JSON 序列化的确定性内容。
        prefix: 可选前缀（如 region 名）。保留以兼容 main.py 的 `region in sid` 模糊匹配。

    Returns:
        稳定 id：`{prefix}_{sha256前16位}` 或（无前缀时）`{sha256前16位}`。
    """
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}" if prefix else digest
