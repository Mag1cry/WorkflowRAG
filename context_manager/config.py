"""可配置参数。"""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    """ContextManager 全局配置。

    环境变量:
        M3E_MODEL_PATH: 覆盖默认的 M3E 模型路径。
    """

    # ── 模型 ──
    model_path: str = "C:/003Codes/models/m3e-base"
    device: str = ""                     # 空 = 自动检测 CUDA/CPU
    max_length: int = 512

    # ── 存储 ──
    storage_path: str = "context_manager.db"
    index_dimension: int = 768

    # ── 检索 ──
    faiss_search_top_k: int = 20

    def __post_init__(self) -> None:
        env_model_path = os.environ.get("M3E_MODEL_PATH")
        if env_model_path:
            self.model_path = env_model_path
