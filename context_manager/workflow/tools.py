"""审查工具集 — WorkflowManager 的 function-call 工具层。

设计：
- 所有工具方法定义在 `ReviewToolsMixin`，WorkflowManager 继承后对外 API 不变。
- `get_tool_schemas()` 用 inspect 从方法签名自动生成 OpenAI 兼容 function call schema，
  替代手工维护（LLM 剪枝的 WorkflowJudge 直接复用同一套工具定义）。

工具分类：
- 查看:   get_workflow / list_workflows / get_step / list_steps / get_steps / visualize
- 操作:   prune_step / batch_prune / update_step / add_step / remove_step
          / reorder_steps
- 元数据: update_workflow_name / update_workflow_description
- 生命周期: solidify / delete_workflow / review_summary

原则：LLM 只能通过工具修改 Workflow，禁止直接输出修改后的内容（防止幻觉）。
"""

from __future__ import annotations

import inspect
import typing
from typing import Any, get_origin, get_args

from ..models import Workflow
from ..persistence import M3EEmbedding, WorkflowIndexBase, WorkflowStoreBase

# 工具方法白名单（get_tool_schemas 的输出顺序）
TOOL_METHODS: tuple[str, ...] = (
    "get_workflow",
    "list_workflows",
    "get_step",
    "list_steps",
    "get_steps",
    "visualize",
    "review_summary",
    "prune_step",
    "batch_prune",
    "update_step",
    "add_step",
    "remove_step",
    "reorder_steps",
    "update_workflow_name",
    "update_workflow_description",
    "solidify",
    "delete_workflow",
)


def _type_schema(annotation: Any, default: Any = inspect.Parameter.empty) -> dict:
    """Python 类型 → JSON Schema 类型（含枚举/默认值）。"""
    if annotation is inspect.Parameter.empty:
        t = {"type": "string"}
    elif annotation is str:
        t = {"type": "string"}
    elif annotation is bool:
        t = {"type": "boolean"}
    elif annotation is int:
        t = {"type": "integer"}
    elif annotation is list or (get_origin(annotation) is list):
        inner = get_args(annotation)[0] if get_args(annotation) else str
        t = {"type": "array", "items": _type_schema(inner)}
    elif annotation is dict or (get_origin(annotation) is dict):
        t = {"type": "object"}
    elif annotation is Workflow:
        t = {"type": "object"}
    else:
        t = {"type": "string"}
    if default is not inspect.Parameter.empty and default is not None:
        t["default"] = default
    return t


class ReviewToolsMixin:
    """Workflow 审查工具（供 LLM 通过 function call 调用）。

    依赖宿主（WorkflowManager）提供的属性: workflow_store / embedding / index。
    以下为宿主注入的属性声明（类型检查用，运行时由 WorkflowManager.__init__ 赋值）：
    """

    # ── 宿主注入属性（声明，不赋值）────────────────────
    workflow_store: WorkflowStoreBase
    embedding: M3EEmbedding
    index: WorkflowIndexBase
    settings: Any
    checkpointer: Any

    # ── 查看 ──────────────────────────────────────────

    def get_workflow(self, workflow_id: str) -> Workflow | None:
        """获取 Workflow 完整信息（含所有步骤）。"""
        return self.workflow_store.get_workflow(workflow_id)

    def list_workflows(self, status: str | None = None) -> list[Workflow]:
        """列出所有 Workflow，可按状态过滤（RAW/SOLIDIFIED）。"""
        return self.workflow_store.list_workflows(status)

    def get_step(self, workflow_id: str, step_index: int) -> str:
        """获取指定 Workflow 中第 N 步的详细信息（参数、结果、状态）。"""
        wf = self.workflow_store.get_workflow(workflow_id)
        if wf is None:
            return f"Workflow not found: {workflow_id}"
        for s in wf.steps:
            if s.step_index == step_index:
                return (
                    f"Step {s.step_index}: [{s.type}] {s.name}\n"
                    f"  args: {s.arguments}\n"
                    f"  result: {s.result}\n"
                    f"  status: {s.status} | pruned: {s.is_pruned}"
                )
        return f"Step index {step_index} not found in {workflow_id}"

    def list_steps(self, workflow_id: str) -> str:
        """列出 Workflow 所有步骤（step_id/索引/类型/名称/状态/参数与结果摘要）。"""
        wf = self.workflow_store.get_workflow(workflow_id)
        if wf is None:
            return f"Workflow not found: {workflow_id}"
        lines = [f"Workflow: {wf.name} ({wf.status}) 共 {len(wf.steps)} 步"]
        for s in wf.steps:
            args = s.arguments
            if len(args) > 50:
                args = args[:50] + "..."
            result = (s.result or "")[:60].replace("\n", " ")
            if len(s.result or "") > 60:
                result += "..."
            mark = "[PRUNED]" if s.is_pruned else "[keep]"
            lines.append(
                f"{mark} step_id={s.step_id} index={s.step_index} "
                f"type={s.type} name={s.name} status={s.status} "
                f"args={args!r} result={result!r}"
            )
        return "\n".join(lines)

    def get_steps(self, workflow_id: str, indices: list[int]) -> str:
        """批量查看指定索引步骤的详情（一次最多 5 个）。"""
        wf = self.workflow_store.get_workflow(workflow_id)
        if wf is None:
            return f"Workflow not found: {workflow_id}"
        out = []
        for idx in indices[:5]:
            for s in wf.steps:
                if s.step_index == idx:
                    out.append(
                        f"--- index={idx} step_id={s.step_id} [{s.type}] {s.name} "
                        f"status={s.status} pruned={s.is_pruned}\n"
                        f"    args: {s.arguments}\n"
                        f"    result: {s.result[:200]}\n"
                        f"    error: {s.error_message[:100]}"
                    )
                    break
        return "\n".join(out) if out else f"indices not found: {indices}"

    def visualize(self, workflow_id: str, show_pruned: bool = True) -> str:
        """可视化 Workflow 的步骤序列（带颜色分类）。"""
        from .visualizer import visualize_workflow

        wf = self.workflow_store.get_workflow(workflow_id)
        if wf is None:
            return f"Workflow not found: {workflow_id}"
        return visualize_workflow(wf, show_pruned=show_pruned)

    def review_summary(self, workflow_id: str) -> str:
        """审查摘要：步骤总数、保留/剪枝数、类型分布。"""
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
        return (
            f"Workflow: {wf.name} ({wf.status})\n"
            f"  步骤: {total}  | 保留: {kept}  | 剪枝: {pruned}\n"
            f"  toolcall: {tc}  | bashcall: {bc}  | 失败: {failed}"
        )

    # ── 操作 ──────────────────────────────────────────

    def prune_step(self, step_id: str, is_pruned: bool) -> bool:
        """标记/取消标记步骤为已剪枝（反复调用幂等）。"""
        self.workflow_store.update_step_pruned(step_id, is_pruned)
        return True

    def batch_prune(self, workflow_id: str, step_ids: list[str]) -> str:
        """批量标记多个步骤为已剪枝（step_ids 是 step_id 列表）。"""
        # 防御：LLM 可能把列表传成逗号分隔字符串
        if isinstance(step_ids, str):
            step_ids = [s.strip() for s in step_ids.split(",") if s.strip()]
        for sid in step_ids:
            self.workflow_store.update_step_pruned(sid, True)
        return f"ok: {len(step_ids)} steps pruned"

    def update_step(
        self,
        step_id: str,
        name: str | None = None,
        arguments: str | None = None,
        result: str | None = None,
        status: str | None = None,
        is_pruned: bool | None = None,
        type: str | None = None,
        error_message: str | None = None,
        duration_ms: int | None = None,
        timestamp: str | None = None,
    ) -> bool:
        """更新步骤的字段（只传需要修改的字段）。

        LLM 审查工具：必须通过此工具修改步骤，禁止直接输出修改后的内容。
        """
        fields = {
            "name": name,
            "arguments": arguments,
            "result": result,
            "status": status,
            "is_pruned": is_pruned,
            "type": type,
            "error_message": error_message,
            "duration_ms": duration_ms,
            "timestamp": timestamp,
        }
        fields = {k: v for k, v in fields.items() if v is not None}
        if fields:
            self.workflow_store.update_step_fields(step_id, **fields)
        return True

    def add_step(
        self,
        workflow_id: str,
        after_index: int,
        type: str,
        name: str,
        arguments: str = "",
        result: str = "",
        status: str = "success",
    ) -> str:
        """在指定位置后插入一个新步骤。返回新 step_id。"""
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
            step_id=step_id,
            workflow_id=workflow_id,
            step_index=new_index,
            type=type,
            name=name,
            arguments=arguments,
            result=result,
            status=status,
        )
        return f"ok:{step_id}"

    def remove_step(self, step_id: str) -> str:
        """删除指定步骤。"""
        self.workflow_store.delete_step(step_id)
        return "ok"

    def reorder_steps(self, workflow_id: str, step_id_order: list[str]) -> str:
        """重新排序步骤（step_id_order 是按新顺序排列的 step_id 列表）。"""
        step_index_map = {sid: i for i, sid in enumerate(step_id_order)}
        self.workflow_store.reorder_steps(workflow_id, step_index_map)
        return "ok"

    # ── 元数据 ────────────────────────────────────────

    def update_workflow_name(self, workflow_id: str, name: str) -> bool:
        """更新 Workflow 名称。"""
        wf = self.workflow_store.get_workflow(workflow_id)
        if wf is None:
            return False
        self.workflow_store.update_name(workflow_id, name)
        return True

    def update_workflow_description(self, workflow_id: str, description: str) -> bool:
        """更新 Workflow 描述（用于检索索引）。"""
        wf = self.workflow_store.get_workflow(workflow_id)
        if wf is None:
            return False
        self.workflow_store.update_description(workflow_id, description)
        return True

    # ── 生命周期 ──────────────────────────────────────

    def solidify(self, workflow_id: str) -> None:
        """固化 Workflow：生成描述、写入索引、标记 SOLIDIFIED。

        剪枝由 WorkflowJudge 负责，solidify 只读取现有 is_pruned 标记。
        """
        from .manager import _generate_description

        wf = self.workflow_store.get_workflow(workflow_id)
        if wf is None:
            raise ValueError(f"Workflow not found: {workflow_id}")

        steps = self.workflow_store.get_steps(workflow_id)
        if not steps:
            print(f"  [SOLIDIFY] {workflow_id} — no steps, skipping")
            return

        step_dicts = [s.to_dict() for s in steps]
        description = _generate_description(step_dicts)
        self.workflow_store.update_description(workflow_id, description)

        if description and description != "(empty workflow)":
            import numpy as np

            vec = self.embedding.embed(description).astype(np.float32)
            self.index.add(workflow_id, vec)

        self.workflow_store.update_status(workflow_id, "SOLIDIFIED")

        kept = sum(1 for s in step_dicts if not s.get("is_pruned"))
        total = len(step_dicts)
        print(
            f"  [SOLIDIFY] {workflow_id} | {kept}/{total} steps retained | "
            f"{description[:60]}..."
        )

    def delete_workflow(self, workflow_id: str) -> None:
        """物理删除 Workflow（含步骤与索引）。"""
        self.workflow_store.delete_workflow(workflow_id)
        self.index.remove(workflow_id)
        print(f"  [DELETE] {workflow_id}")

    # ── Schema ────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict]:
        """自动生成 OpenAI 兼容 function call schema（基于方法签名与 docstring）。"""
        schemas = []
        for name in TOOL_METHODS:
            fn = getattr(type(self), name)
            doc = inspect.getdoc(fn) or ""
            description = (
                doc.split("\n\n")[0].strip().replace("\n", " ") if doc else name
            )

            # `from __future__ import annotations` 下注解是字符串，
            # 用 get_type_hints 解析为真实类型再做 schema 推断
            try:
                hints = typing.get_type_hints(fn)
            except Exception:
                hints = {}

            properties: dict[str, Any] = {}
            required: list[str] = []
            for pname, param in inspect.signature(fn).parameters.items():
                if pname == "self":
                    continue
                annotation = hints.get(pname, param.annotation)
                schema = _type_schema(annotation, param.default)
                if param.default is inspect.Parameter.empty:
                    required.append(pname)
                properties[pname] = schema

            schemas.append(
                {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                }
            )
        return schemas
