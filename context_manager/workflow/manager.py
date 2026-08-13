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
from .visualizer import visualize_workflow as _visualize, visualize_comparison as _compare, build_case_study as _case_study


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
        self.workflow_store.update_name(workflow_id, name)
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
        """更新 Step 的字段（name, arguments, result, status, error_message, type, timestamp, is_pruned 等）。

        LLM 审查工具：必须通过此工具修改步骤，禁止直接输出修改后的内容。
        返回简短确认，不返回被修改的数据。
        """
        updatable = {"name", "arguments", "result", "status", "error_message",
                     "duration_ms", "type", "timestamp", "is_pruned"}
        fields = {k: v for k, v in kwargs.items() if k in updatable}
        if fields:
            self.workflow_store.update_step_fields(step_id, **fields)
        return True

    # ── 可视化与案例 ────────────────────────────────

    def visualize(self, workflow_id: str, show_pruned: bool = True) -> str:
        """可视化指定 Workflow 的步骤序列。"""
        wf = self.workflow_store.get_workflow(workflow_id)
        if wf is None:
            return f"Workflow not found: {workflow_id}"
        return _visualize(wf, show_pruned=show_pruned)

    def visualize_comparison(self, workflow_id: str) -> str:
        """对比 RAW 和剪枝后的效果（需先运行 solidify）。"""
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

    # ── 审查 LLM 工具（新增/增强） ──────────────────

    def get_step(self, workflow_id: str, step_index: int) -> str:
        """获取指定 Workflow 中第 N 步的详细信息。"""
        wf = self.workflow_store.get_workflow(workflow_id)
        if wf is None:
            return f"Workflow not found: {workflow_id}"
        for s in wf.steps:
            if s.step_index == step_index:
                return (f"Step {s.step_index}: [{s.type}] {s.name}\n"
                        f"  args: {s.arguments}\n"
                        f"  result: {s.result}\n"
                        f"  status: {s.status} | pruned: {s.is_pruned}")
        return f"Step index {step_index} not found in {workflow_id}"

    def add_step(self, workflow_id: str, after_index: int, type: str, name: str,
                 arguments: str = "", result: str = "", status: str = "success") -> str:
        """在指定位置后插入一个新步骤。返回新 step_id，不返回步骤内容。"""
        import uuid
        wf = self.workflow_store.get_workflow(workflow_id)
        if wf is None:
            return f"Workflow not found: {workflow_id}"

        existing = sorted(wf.steps, key=lambda s: s.step_index)
        new_index = after_index + 1
        for s in existing:
            if s.step_index >= new_index:
                s.step_index += 1

        step_id = uuid.uuid4().hex[:12]
        self.workflow_store.add_step(
            step_id=step_id, workflow_id=workflow_id, step_index=new_index,
            type=type, name=name, arguments=arguments,
            result=result, status=status,
        )
        return f"ok:{step_id}"

    def remove_step(self, step_id: str) -> str:
        """删除指定步骤。返回简短确认。"""
        self.workflow_store.delete_step(step_id)
        return "ok"

    def reorder_steps(self, workflow_id: str, step_id_order: list[str]) -> str:
        """重新排序步骤。step_id_order 是 step_id 列表，按新顺序排列。"""
        step_index_map = {sid: i for i, sid in enumerate(step_id_order)}
        self.workflow_store.reorder_steps(workflow_id, step_index_map)
        return "ok"

    def batch_prune(self, workflow_id: str, step_ids: list[str]) -> str:
        """批量标记步骤为已剪枝。"""
        for sid in step_ids:
            self.workflow_store.update_step_pruned(sid, True)
        return f"ok: {len(step_ids)} steps pruned"

    def review_summary(self, workflow_id: str) -> str:
        """生成审查摘要：步骤总数、保留/剪枝数、类型分布。"""
        wf = self.workflow_store.get_workflow(workflow_id)
        if wf is None:
            return f"Workflow not found: {workflow_id}"
        total = len(wf.steps)
        if total == 0:
            return f"Workflow {workflow_id}: (empty)"
        kept = sum(1 for s in wf.steps if not s.is_pruned)
        pruned = total - kept
        tc = sum(1 for s in wf.steps if s.type == "toolcall")
        bc = sum(1 for s in wf.steps if s.type == "bashcall")
        failed = sum(1 for s in wf.steps if s.status == "failure")
        return (f"Workflow: {wf.name} ({wf.status})\n"
                f"  步骤: {total}  | 保留: {kept}  | 剪枝: {pruned}\n"
                f"  toolcall: {tc}  | bashcall: {bc}  | 失败: {failed}")

    # ── Function Call Schema ─────────────────────────

    def get_tool_schemas(self) -> list[dict]:
        """返回所有可用工具的 OpenAI-compatible function call schema。"""
        return [
            {
                "name": "get_workflow",
                "description": "获取 Workflow 完整信息（含所有步骤）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string", "description": "Workflow ID"}
                    },
                    "required": ["workflow_id"],
                },
            },
            {
                "name": "list_workflows",
                "description": "列出所有 Workflow，可按状态过滤",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["RAW", "SOLIDIFIED"], "description": "过滤状态（可选）"}
                    },
                },
            },
            {
                "name": "visualize",
                "description": "可视化 Workflow 的步骤序列（带颜色分类）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string", "description": "Workflow ID"},
                        "show_pruned": {"type": "boolean", "description": "是否显示已剪枝步骤"},
                    },
                    "required": ["workflow_id"],
                },
            },
            {
                "name": "review_summary",
                "description": "审查摘要：步骤总数、保留/剪枝数、类型分布",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string", "description": "Workflow ID"}
                    },
                    "required": ["workflow_id"],
                },
            },
            {
                "name": "get_step",
                "description": "获取 Workflow 中某一步的详细信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string", "description": "Workflow ID"},
                        "step_index": {"type": "integer", "description": "步骤索引（从0开始）"},
                    },
                    "required": ["workflow_id", "step_index"],
                },
            },
            {
                "name": "prune_step",
                "description": "标记/取消标记步骤为已剪枝（反复调用幂等）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "step_id": {"type": "string", "description": "Step ID"},
                        "is_pruned": {"type": "boolean", "description": "true=剪枝，false=取消剪枝"},
                    },
                    "required": ["step_id", "is_pruned"],
                },
            },
            {
                "name": "batch_prune",
                "description": "批量标记多个步骤为已剪枝",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string", "description": "Workflow ID"},
                        "step_ids": {"type": "array", "items": {"type": "string"}, "description": "要剪枝的 step_id 列表"},
                    },
                    "required": ["workflow_id", "step_ids"],
                },
            },
            {
                "name": "update_step",
                "description": "更新步骤的字段（name/arguments/result/status/type/is_pruned 等）。LLM 必须通过此工具修改，禁止直接输出修改后的内容。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "step_id": {"type": "string", "description": "Step ID"},
                        "name": {"type": "string", "description": "步骤名称"},
                        "arguments": {"type": "string", "description": "参数字符串"},
                        "result": {"type": "string", "description": "结果字符串"},
                        "status": {"type": "string", "enum": ["success", "failure"]},
                        "is_pruned": {"type": "boolean"},
                        "type": {"type": "string", "enum": ["toolcall", "bashcall"]},
                        "error_message": {"type": "string"},
                    },
                },
            },
            {
                "name": "add_step",
                "description": "在指定位置后插入新步骤。返回新 step_id。禁止直接输出步骤内容。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string", "description": "Workflow ID"},
                        "after_index": {"type": "integer", "description": "在此索引后插入（从0开始，-1=插入到开头）"},
                        "type": {"type": "string", "enum": ["toolcall", "bashcall"]},
                        "name": {"type": "string", "description": "步骤名称"},
                        "arguments": {"type": "string", "description": "参数"},
                        "result": {"type": "string", "description": "结果"},
                        "status": {"type": "string", "enum": ["success", "failure"]},
                    },
                    "required": ["workflow_id", "after_index", "type", "name"],
                },
            },
            {
                "name": "remove_step",
                "description": "删除指定步骤。返回简短确认。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "step_id": {"type": "string", "description": "要删除的 Step ID"}
                    },
                    "required": ["step_id"],
                },
            },
            {
                "name": "reorder_steps",
                "description": "重新排序步骤",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string", "description": "Workflow ID"},
                        "step_id_order": {"type": "array", "items": {"type": "string"}, "description": "按新顺序排列的 step_id 列表"},
                    },
                    "required": ["workflow_id", "step_id_order"],
                },
            },
            {
                "name": "update_workflow_description",
                "description": "更新 Workflow 描述",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string", "description": "Workflow ID"},
                        "description": {"type": "string", "description": "新描述"},
                    },
                    "required": ["workflow_id", "description"],
                },
            },
            {
                "name": "solidify",
                "description": "对 RAW Workflow 执行剪枝，生成 SOLIDIFIED",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string", "description": "Workflow ID"}
                    },
                    "required": ["workflow_id"],
                },
            },
            {
                "name": "delete_workflow",
                "description": "物理删除 Workflow",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string", "description": "Workflow ID"}
                    },
                    "required": ["workflow_id"],
                },
            },
        ]

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