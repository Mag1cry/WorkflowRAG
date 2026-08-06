"""索引层抽象基类。"""

from abc import ABC, abstractmethod

import numpy as np


class WorkflowIndexBase(ABC):
    """Workflow 向量索引接口。"""

    @abstractmethod
    def add(self, workflow_id: str, vec: np.ndarray) -> None:
        ...

    @abstractmethod
    def update(self, workflow_id: str, vec: np.ndarray) -> None:
        ...

    @abstractmethod
    def remove(self, workflow_id: str) -> None:
        ...

    @abstractmethod
    def search(self, vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """搜索最相似的 top_k 个 Workflow。"""
        ...

    @abstractmethod
    def get_vector(self, workflow_id: str) -> np.ndarray | None:
        ...

    @abstractmethod
    def size(self) -> int:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...