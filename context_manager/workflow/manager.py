"""WorkflowManager — Workflow 生命周期管理引擎。

职责（生命周期）：
- 提取:     extract_workflow（从 LangGraph Checkpoints 事后提取步骤）
- 检索:     retrieve（向量检索 SOLIDIFIED Workflow）
- 注入:     format_context（Workflow → Agent 上下文文本）
- 展示:     visualize_comparison / run_case_study

审查工具（get_step / prune_step / solidify 等）由 ReviewToolsMixin 提供，
见 workflow/tools.py。LLM 剪枝见 workflow/judge.py。

用法:
    from context_manager import WorkflowManager
    from context_manager.workflow.judge import WorkflowJudge

    wfm = WorkflowManager()
    wf_id = wfm.extract_workflow("some_thread_id")
    WorkflowJudge(wfm, llm).judge(wf_id)   # LLM 剪枝
    wfm.solidify(wf_id)                    # 固化
    results = wfm.retrieve("如何修复导入错误")
"""

from __future__ import annotations

import datetime
import uuid

import numpy as np

from ..config import Settings
from ..models import Workflow
from ..persistence import (
    WorkflowStoreBase,
    SQLiteWorkflowStore,
    MemoryWorkflowStore,
    WorkflowIndexBase,
    FaissWorkflowIndex,
    MemoryWorkflowIndex,
    M3EEmbedding,
)
from .injector import format_context as _format_context
from .tools import ReviewToolsMixin
from .visualizer import (
    visualize_comparison as _compare,
    build_case_study as _case_study,
)


def _generate_description(steps: list[dict]) -> str:
    """从步骤序列生成描述文本（跳过已剪枝步骤）。"""
    kept = [s for s in steps if not s.get("is_pruned")]
    parts = []
    for s in kept:
        name = s.get("name", "?")
        args = s.get("arguments", "")
        if args and len(args) > 60:
            args = args[:60] + "..."
        parts.append(f"{name}({args})" if args else name)
    return " → ".join(parts) if parts else "(empty workflow)"


class WorkflowManager(ReviewToolsMixin):
    """Workflow 管理引擎（生命周期 + 审查工具）。

    依赖 LangGraph Checkpointer 管理 Thread 状态，
    WorkflowManager 自身负责步骤提取、固化、存储和检索。

    公开方法：
    - 生命周期: extract_workflow / retrieve / format_context / visualize_comparison
    - 审查工具: get_workflow / list_workflows / get_step / prune_step / solidify ...
      （见 tools.py）
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
        """从存储重建向量索引（启动时执行）。"""
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

        print(
            f"  [EXTRACT] {workflow_id} | {len(steps)} steps from thread "
            f"{thread_id[:12]}..."
        )
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

                    step_type = (
                        "bashcall" if tool_name in ("bash", "terminal") else "toolcall"
                    )
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

    # ── 检索 ────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> list[Workflow]:
        """根据 query 检索最相关的 SOLIDIFIED Workflow（已过滤剪枝步骤）。"""
        if self.index.size() == 0:
            return []

        vec = self.embedding.embed(query).astype(np.float32)
        results = self.index.search(vec, top_k)

        output = []
        for workflow_id, _score in results:
            wf = self.workflow_store.get_workflow(workflow_id)
            if wf is None:
                continue
            wf.steps = [s for s in wf.steps if not s.is_pruned]
            output.append(wf)
        return output

    # ── 上下文注入 ──────────────────────────────────

    def format_context(self, workflow: Workflow) -> str:
        """将 Workflow 格式化为 Agent 上下文注入文本（详细格式）。"""
        return _format_context(workflow)

    # ── 展示（非 LLM 工具）───────────────────────────

    def visualize_comparison(self, workflow_id: str) -> str:
        """对比 RAW 与剪枝后的效果（需先运行 judge/solidify）。"""
        wf = self.workflow_store.get_workflow(workflow_id)
        if wf is None:
            return f"Workflow not found: {workflow_id}"

        raw = Workflow(
            workflow_id=wf.workflow_id,
            name=wf.name,
            status="RAW",
            steps=wf.steps,
        )

        kept = [s for s in wf.steps if not s.is_pruned]
        solidified = Workflow(
            workflow_id=wf.workflow_id,
            name=wf.name,
            status="SOLIDIFIED",
            description=wf.description,
            steps=kept,
        )
        return _compare(raw, solidified)

    @staticmethod
    def run_case_study() -> str:
        """运行内置案例，展示剪枝效果。"""
        raw, solidified = _case_study()
        return _compare(raw, solidified)

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
