"""索引基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class BaseIndex(ABC):
    @abstractmethod
    def query(self, **kwargs: Any) -> List[Any]:
        """返回与查询条件匹配的记录。"""
