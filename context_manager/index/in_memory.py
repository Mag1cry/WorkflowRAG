"""内存 Workflow 索引实现（暴力余弦相似度）。"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base import WorkflowIndexBase


class MemoryWorkflowIndex(WorkflowIndexBase):
    """基于内存 dict 的 Workflow 向量索引。"""

    def __init__(self):
        self._vectors: dict[str, np.ndarray] = {}

    def add(self, workflow_id: str, vec: np.ndarray) -> None:
        self._vectors[workflow_id] = vec.astype(np.float32).copy()

    def update(self, workflow_id: str, vec: np.ndarray) -> None:
        self._vectors[workflow_id] = vec.astype(np.float32).copy()

    def remove(self, workflow_id: str) -> None:
        self._vectors.pop(workflow_id, None)

    def search(self, vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if not self._vectors:
            return []
        ids = list(self._vectors.keys())
        vecs = np.stack([self._vectors[tid] for tid in ids])
        scores = cosine_similarity(vec.reshape(1, -1), vecs)[0]
        sorted_idx = np.argsort(scores)[::-1][:top_k]
        return [(ids[i], float(scores[i])) for i in sorted_idx]

    def get_vector(self, workflow_id: str) -> np.ndarray | None:
        return self._vectors.get(workflow_id)

    def size(self) -> int:
        return len(self._vectors)

    def clear(self) -> None:
        self._vectors.clear()