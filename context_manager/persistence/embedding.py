"""M3E 文本嵌入引擎。"""

from __future__ import annotations

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from ..config import Settings


class M3EEmbedding:
    """M3E 中文文本嵌入模型封装。

    使用 mean pooling 将任意长度文本编码为 768 维向量。
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._device = self._resolve_device()

        print(f"[Embedding] 加载 M3E → {self._device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_path)
        self.model = AutoModel.from_pretrained(self.settings.model_path).to(
            self._device
        )
        self.model.eval()

    def _resolve_device(self) -> str:
        if self.settings.device:
            return self.settings.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device(self) -> str:
        return self._device

    @property
    def dimension(self) -> int:
        return self.settings.index_dimension

    def embed(self, text: str) -> np.ndarray:
        """单条文本 → (768,) 向量。"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.settings.max_length,
        ).to(self._device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            vec = outputs.last_hidden_state.mean(dim=1)
        return vec.cpu().numpy().squeeze().astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """批量文本 → (N, 768) 向量。"""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        inputs = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.settings.max_length,
        ).to(self._device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            vecs = outputs.last_hidden_state.mean(dim=1)
        return vecs.cpu().numpy().astype(np.float32)
