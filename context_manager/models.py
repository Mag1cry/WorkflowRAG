"""Workflow + Step 数据类（顶层共享模型）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Step:
    """步骤：Agent 的一次工具调用或命令执行。"""

    step_id: str
    workflow_id: str
    step_index: int
    type: Literal["toolcall", "bashcall"]
    name: str
    arguments: str = ""
    result: str = ""
    status: Literal["success", "failure"] = "success"
    duration_ms: int = 0
    error_message: str = ""
    is_pruned: bool = False
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "workflow_id": self.workflow_id,
            "step_index": self.step_index,
            "type": self.type,
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "is_pruned": self.is_pruned,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Step:
        return cls(
            step_id=d["step_id"],
            workflow_id=d.get("workflow_id", ""),
            step_index=d.get("step_index", 0),
            type=d.get("type", "toolcall"),
            name=d.get("name", ""),
            arguments=d.get("arguments", ""),
            result=d.get("result", ""),
            status=d.get("status", "success"),
            duration_ms=d.get("duration_ms", 0),
            error_message=d.get("error_message", ""),
            is_pruned=bool(d.get("is_pruned", False)),
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class Workflow:
    """工作流：Agent 完成某项任务的有序步骤序列。"""

    workflow_id: str
    name: str
    description: str = ""
    status: Literal["RAW", "SOLIDIFIED"] = "RAW"
    source_thread_id: str = ""
    tags: str = ""
    created_at: str = ""
    updated_at: str = ""
    steps: list[Step] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "source_thread_id": self.source_thread_id,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Workflow:
        steps_data = d.pop("steps", []) if isinstance(d, dict) else []
        steps = [Step.from_dict(s) for s in steps_data]
        return cls(
            workflow_id=d.get("workflow_id", ""),
            name=d.get("name", ""),
            description=d.get("description", ""),
            status=d.get("status", "RAW"),
            source_thread_id=d.get("source_thread_id", ""),
            tags=d.get("tags", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            steps=steps,
        )