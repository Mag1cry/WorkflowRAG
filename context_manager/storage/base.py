"""Workflow + Step 元数据存储抽象基类。"""

from abc import ABC, abstractmethod


class WorkflowStoreBase(ABC):
    """Workflow 元数据存储接口。"""

    @abstractmethod
    def create_workflow(self, workflow_id: str, name: str,
                        description: str = "", source_thread_id: str = "",
                        tags: str = "") -> None:
        ...

    @abstractmethod
    def get_workflow(self, workflow_id: str) -> dict | None:
        ...

    @abstractmethod
    def list_workflows(self, status: str | None = None) -> list[dict]:
        ...

    @abstractmethod
    def update_status(self, workflow_id: str, status: str) -> None:
        ...

    @abstractmethod
    def update_description(self, workflow_id: str, description: str) -> None:
        ...

    @abstractmethod
    def delete_workflow(self, workflow_id: str) -> None:
        ...

    @abstractmethod
    def add_step(self, step_id: str, workflow_id: str, step_index: int,
                 type: str, name: str, arguments: str = "", result: str = "",
                 status: str = "success", duration_ms: int = 0,
                 error_message: str = "", timestamp: str = "") -> None:
        ...

    @abstractmethod
    def get_steps(self, workflow_id: str) -> list[dict]:
        """获取 Workflow 的所有 Step，按 step_index 排序。"""
        ...

    @abstractmethod
    def update_step_pruned(self, step_id: str, is_pruned: bool) -> None:
        """更新 Step 的剪枝标记。"""
        ...

    @abstractmethod
    def close(self) -> None:
        ...