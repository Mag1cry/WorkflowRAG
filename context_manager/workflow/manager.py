"""WorkflowManager — 基于 LangGraph 的 Workflow 提取与管理引擎。

核心思想：context = workflow。
任务完成后，从 LangGraph Checkpoints 中提取 toolcall/bashcall 步骤序列，
经过剪枝优化后固化，作为未来任务的优秀案例注入上下文。

使用:
    from context_manager import WorkflowManager

    wfm = WorkflowManager()
    wf_id = wfm.extract_workflow("some_thread_id")
    wfm.solidify(wf_id)
    results = wfm.retrieve("如何修复导入错误")
"""

from __future__ import annotations

import uuid
import datetime

import numpy as np

from ..config import Settings
from ..models import Workflow, Step
from ..persistence import (
    WorkflowStoreBase, SQLiteWorkflowStore, MemoryWorkflowStore,
    WorkflowIndexBase, FaissWorkflowIndex, MemoryWorkflowIndex,
    M3EEmbedding,
)
from . import pruner as pruner_mod
from .injector import format_context


class WorkflowManager:
    """Workflow 管理引擎。

    依赖 LangGraph Checkpointer 管理 Thread 状态，
    WorkflowManager 自身负责步骤提取、剪枝、存储和检索。

    所有公开方法也可作为审查 LLM 的 function call 使用。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        checkpointer=None,
        workflow_store: WorkflowStoreBase | None = None,
        embedding: M3EEmbedding | None = None,
        index: WorkflowIndexBase | None = None,
    ):
        self.settings = settings or Settings()

        if checkpointer is None:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
        self.checkpointer = checkpointer

        self.workflow_store = workflow_store or SQLiteWorkflowStore(
            self.settings.storage_path
        )

        self.embedding = embedding or M3EEmbedding(self.settings)
        self.index = index or FaissWorkflowIndex(self.settings.index_dimension)

        self._rebuild_index()

    # ── 索引重建 ────────────────────────────────────

    def _rebuild_index(self) -> None:
        """从 SQLite 重建 FAISS 索引。"""
        workflows = self.workflow_store.list_workflows(status="SOLIDIFIED")
        if not workflows:
            return
        restored = 0
        for wf in workflows:
            desc = wf.description or wf.name
            if desc:
                vec = self.embedding.embed(desc)
                self.index.add(wf.workflow_id, vec)
                restored += 1
        if restored > 0:
            print(f"[WorkflowManager] 从存储恢复 {restored} 个 Workflow 索引")

    # ── 步骤提取 ────────────────────────────────────

    def extract_workflow(self, thread_id: str, name: str = "") -> str:
        """从 LangGraph Thread 中事后提取步骤，创建 RAW Workflow。

        Args:
            thread_id: LangGraph Thread ID。
            name: 可选的 Workflow 名称，为空则自动生成。

        Returns:
            workflow_id: 新创建的 Workflow ID。
        """
        messages = self._get_thread_messages(thread_id)
        steps = self._messages_to_steps(messages)

        workflow_id = uuid.uuid4().hex[:12]
        wf_name = name or f"workflow_{workflow_id}"

        self.workflow_store.create_workflow(
            workflow_id=workflow_id,
            name=wf_name,
            source_thread_id=thread_id,
        )

        for step in steps:
            self.workflow_store.add_step(
                step_id=step["step_id"],
                workflow_id=workflow_id,
                step_index=step["step_index"],
                type=step["type"],
                name=step["name"],
                arguments=step.get("arguments", ""),
                result=step.get("result", ""),
                status=step.get("status", "success"),
                duration_ms=step.get("duration_ms", 0),
                error_message=step.get("error_message", ""),
                timestamp=step.get("timestamp", ""),
            )

        print(f"  [EXTRACT] {workflow_id} | {len(steps)} steps from thread {thread_id[:12]}...")
        return workflow_id

    def _get_thread_messages(self, thread_id: str) -> list:
        """从 LangGraph Checkpointer 获取 Thread 的所有消息。"""
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        result = self.checkpointer.get_tuple(config)
        if result is None:
            return []
        channel_values = result.checkpoint.get("channel_values", {})
        return list(channel_values.get("messages", []))

    def _messages_to_steps(self, messages: list) -> list[dict]:
        """将 LangChain 消息列表转换为 Step 字典列表。"""
        from langchain_core.messages import AIMessage, ToolMessage

        steps = []
        tool_call_map: dict[str, dict] = {}

        for msg in messages:
            if isinstance(msg, AIMessage):
                tc_list = getattr(msg, "tool_calls", []) or []
                for tc in tc_list:
                    step_id = uuid.uuid4().hex[:12]
                    tool_name = tc.get("name", "")
                    args = tc.get("args", {})

                    step_type = "bashcall" if tool_name in ("bash", "terminal") else "toolcall"
                    now = datetime.datetime.now().isoformat()

                    step = {
                        "step_id": step_id,
                        "step_index": len(steps),
                        "type": step_type,
                        "name": tool_name,
                        "arguments": str(args),
                        "result": "",
                        "status": "success",
                        "duration_ms": 0,
                        "error_message": "",
                        "timestamp": now,
                        "is_pruned": False,
                    }
                    tool_call_map[tc.get("id", step_id)] = step
                    steps.append(step)

            elif isinstance(msg, ToolMessage):
                tc_id = getattr(msg, "tool_call_id", None)
                if tc_id and tc_id in tool_call_map:
                    matched = tool_call_map[tc_id]
                    matched["result"] = str(msg.content) if msg.content else ""
                    if hasattr(msg, "status") and msg.status == "error":
                        matched["status"] = "failure"
                        matched["error_message"] = str(msg.content)

        return steps

    # ── 固化（剪枝 + 索引）───────────────────────────

    def solidify(self, workflow_id: str) -> None:
        """对 RAW Workflow 执行剪枝，生成 SOLIDIFIED Workflow。

        流程：
        1. 读取 Workflow 的所有 Step
        2. 执行规则剪枝
        3. 更新剪枝标记到存储
        4. 生成 description
        5. 生成 Embedding → 写入索引
        6. 更新状态为 SOLIDIFIED
        """
        wf = self.workflow_store.get_workflow(workflow_id)
        if wf is None:
            raise ValueError(f"Workflow not found: {workflow_id}")

        steps = self.workflow_store.get_steps(workflow_id)
        if not steps:
            print(f"  [SOLIDIFY] {workflow_id} — no steps, skipping")
            return

        step_dicts = [s.to_dict() for s in steps]
        pruner_mod.prune(step_dicts)

        for sd in step_dicts:
            self.workflow_store.update_step_pruned(sd["step_id"], sd.get("is_pruned", False))

        description = pruner_mod.generate_description(step_dicts)

        self.workflow_store.update_description(workflow_id, description)

        if description and description != "(empty workflow)":
            vec = self.embedding.embed(description).astype(np.float32)
            self.index.add(workflow_id, vec)

        self.workflow_store.update_status(workflow_id, "SOLIDIFIED")

        kept = sum(1 for s in step_dicts if not s.get("is_pruned"))
        total = len(step_dicts)
        print(f"  [SOLIDIFY] {workflow_id} | {kept}/{total} steps retained | {description[:60]}...")

    # ── 检索 ────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> list[Workflow]:
        """根据 query 检索最相关的 SOLIDIFIED Workflow。"""
        if self.index.size() == 0:
            return []

        vec = self.embedding.embed(query).astype(np.float32)
        results = self.index.search(vec, top_k)

        output = []
        for workflow_id, score in results:
            wf = self.workflow_store.get_workflow(workflow_id)
            if wf is None:
                continue
            wf.steps = [s for s in wf.steps if not s.is_pruned]
            output.append(wf)
        return output

    # ── 上下文注入 ──────────────────────────────────

    def format_context(self, workflow: Workflow) -> str:
        """将 Workflow 格式化为 Agent 上下文注入文本。"""
        return format_context(workflow)

    # ── 审查 LLM 工具 ────────────────────────────────

    def get_workflow(self, workflow_id: str) -> Workflow | None:
        """获取 Workflow 及其 Steps。"""
        return self.workflow_store.get_workflow(workflow_id)

    def list_workflows(self, status: str | None = None) -> list[Workflow]:
        """列出 Workflow。可选按 status 过滤。"""
        return self.workflow_store.list_workflows(status)

    def delete_workflow(self, workflow_id: str) -> None:
        """物理删除 Workflow。"""
        self.workflow_store.delete_workflow(workflow_id)
        self.index.remove(workflow_id)
        print(f"  [DELETE] {workflow_id}")

    def update_workflow_name(self, workflow_id: str, name: str) -> bool:
        """更新 Workflow 名称。"""
        wf = self.workflow_store.get_workflow(workflow_id)
        if wf is None:
            return False
        self.workflow_store.update_description(workflow_id, name)
        return True

    def update_workflow_description(self, workflow_id: str, description: str) -> bool:
        """更新 Workflow 描述。"""
        wf = self.workflow_store.get_workflow(workflow_id)
        if wf is None:
            return False
        self.workflow_store.update_description(workflow_id, description)
        return True

    def prune_step(self, step_id: str, is_pruned: bool) -> bool:
        """标记/取消标记 Step 为已剪枝。"""
        self.workflow_store.update_step_pruned(step_id, is_pruned)
        return True

    def update_step(self, step_id: str, **kwargs) -> bool:
        """更新 Step 的字段（name, arguments, result 等）。
        
        当前仅支持更新 is_pruned 标记，完整字段更新需后续扩展。
        """
        if "is_pruned" in kwargs:
            self.workflow_store.update_step_pruned(step_id, kwargs["is_pruned"])
        return True

    # ── 生命周期 ────────────────────────────────────

    def close(self) -> None:
        """关闭所有连接。"""
        self.workflow_store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ── 便捷工厂 ────────────────────────────────────────


def create_memory_manager() -> WorkflowManager:
    """创建纯内存版 WorkflowManager（测试用）。"""
    from langgraph.checkpoint.memory import MemorySaver

    return WorkflowManager(
        settings=Settings(),
        checkpointer=MemorySaver(),
        workflow_store=MemoryWorkflowStore(),
        index=MemoryWorkflowIndex(),
    )