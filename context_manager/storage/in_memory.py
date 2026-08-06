"""内存 Workflow + Step 元数据存储（测试用）。"""

import datetime

from .base import WorkflowStoreBase


class MemoryWorkflowStore(WorkflowStoreBase):
    """基于内存 dict 的 Workflow 元数据存储。"""

    def __init__(self):
        self._workflows: dict[str, dict] = {}
        self._steps: dict[str, list[dict]] = {}

    # ── Workflow CRUD ────────────────────────────────

    def create_workflow(self, workflow_id: str, name: str,
                        description: str = "", source_thread_id: str = "",
                        tags: str = "") -> None:
        now = datetime.datetime.now().isoformat()
        self._workflows[workflow_id] = {
            "workflow_id": workflow_id,
            "name": name,
            "description": description,
            "status": "RAW",
            "source_thread_id": source_thread_id,
            "tags": tags,
            "created_at": now,
            "updated_at": now,
        }
        self._steps[workflow_id] = []

    def get_workflow(self, workflow_id: str) -> dict | None:
        return self._workflows.get(workflow_id)

    def list_workflows(self, status: str | None = None) -> list[dict]:
        result = list(self._workflows.values())
        if status:
            result = [w for w in result if w["status"] == status]
        return result

    def update_status(self, workflow_id: str, status: str) -> None:
        wf = self._workflows.get(workflow_id)
        if wf:
            wf["status"] = status
            wf["updated_at"] = datetime.datetime.now().isoformat()

    def update_description(self, workflow_id: str, description: str) -> None:
        wf = self._workflows.get(workflow_id)
        if wf:
            wf["description"] = description
            wf["updated_at"] = datetime.datetime.now().isoformat()

    def delete_workflow(self, workflow_id: str) -> None:
        self._workflows.pop(workflow_id, None)
        self._steps.pop(workflow_id, None)

    # ── Step CRUD ────────────────────────────────────

    def add_step(self, step_id: str, workflow_id: str, step_index: int,
                 type: str, name: str, arguments: str = "", result: str = "",
                 status: str = "success", duration_ms: int = 0,
                 error_message: str = "", timestamp: str = "") -> None:
        if workflow_id not in self._steps:
            self._steps[workflow_id] = []
        self._steps[workflow_id].append({
            "step_id": step_id,
            "workflow_id": workflow_id,
            "step_index": step_index,
            "type": type,
            "name": name,
            "arguments": arguments,
            "result": result,
            "status": status,
            "duration_ms": duration_ms,
            "error_message": error_message,
            "is_pruned": False,
            "timestamp": timestamp,
        })

    def get_steps(self, workflow_id: str) -> list[dict]:
        steps = self._steps.get(workflow_id, [])
        return sorted(steps, key=lambda s: s["step_index"])

    def update_step_pruned(self, step_id: str, is_pruned: bool) -> None:
        for steps in self._steps.values():
            for s in steps:
                if s["step_id"] == step_id:
                    s["is_pruned"] = is_pruned
                    return

    def close(self) -> None:
        pass