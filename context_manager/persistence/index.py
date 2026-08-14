"""Workflow 向量索引抽象基类与实现。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from abc import ABC, abstractmethod


class WorkflowIndexBase(ABC):
    """Workflow 向量索引接口。"""

    @abstractmethod
    def add(self, workflow_id: str, vec: np.ndarray) -> None: ...

    @abstractmethod
    def update(self, workflow_id: str, vec: np.ndarray) -> None: ...

    @abstractmethod
    def remove(self, workflow_id: str) -> None: ...

    @abstractmethod
    def search(self, vec: np.ndarray, top_k: int) -> list[tuple[str, float]]: ...

    @abstractmethod
    def get_vector(self, workflow_id: str) -> np.ndarray | None: ...

    @abstractmethod
    def size(self) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...


class FaissWorkflowIndex(WorkflowIndexBase):
    """基于 FAISS IndexFlatIP 的 Workflow 向量索引。"""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}
        self._next_int_id = 0
        self._vec_cache: dict[str, np.ndarray] = {}

    def add(self, workflow_id: str, vec: np.ndarray) -> None:
        normalized = self._normalize(vec)
        int_id = self._next_int_id
        self._next_int_id += 1
        self._index.add_with_ids(
            normalized.reshape(1, -1).astype(np.float32),
            np.array([int_id], dtype=np.int64),
        )
        self._id_to_int[workflow_id] = int_id
        self._int_to_id[int_id] = workflow_id
        self._vec_cache[workflow_id] = normalized

    def update(self, workflow_id: str, vec: np.ndarray) -> None:
        if workflow_id in self._id_to_int:
            old_int_id = self._id_to_int[workflow_id]
            self._index.remove_ids(np.array([old_int_id], dtype=np.int64))
            del self._int_to_id[old_int_id]
        self.add(workflow_id, vec)

    def remove(self, workflow_id: str) -> None:
        if workflow_id not in self._id_to_int:
            return
        int_id = self._id_to_int.pop(workflow_id)
        self._index.remove_ids(np.array([int_id], dtype=np.int64))
        self._int_to_id.pop(int_id, None)
        self._vec_cache.pop(workflow_id, None)

    def search(self, vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self.size() == 0:
            return []
        k = min(top_k, self.size())
        normalized = self._normalize(vec)
        distances, indices = self._index.search(
            normalized.reshape(1, -1).astype(np.float32), k
        )
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            workflow_id = self._int_to_id.get(int(idx))
            if workflow_id:
                results.append((workflow_id, float(dist)))
        return results

    def get_vector(self, workflow_id: str) -> np.ndarray | None:
        return self._vec_cache.get(workflow_id)

    def size(self) -> int:
        return self._index.ntotal

    def clear(self) -> None:
        self._index.reset()
        self._id_to_int.clear()
        self._int_to_id.clear()
        self._vec_cache.clear()
        self._next_int_id = 0

    def save(self, path: str) -> None:
        faiss.write_index(
            faiss.downcast_index(self._index.index),
            str(path),
        )

    def load(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            flat_index = faiss.read_index(str(p))
            self._index = faiss.IndexIDMap(flat_index)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec


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
