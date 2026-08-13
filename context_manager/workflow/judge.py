"""LLM 剪枝 — WorkflowJudge：审查 LLM 通过 function call 操作 Workflow 步骤。

设计初衷（见 docs/Design.md "审查 LLM 工具集"）：
审查 LLM 只能通过 toolcall 修改 Workflow，不能直接输出修改后的内容（防止幻觉）。

工具集（全部来自 WorkflowManager 的公开方法）:
- 查看:   review_summary / list_steps / get_step
- 操作:   prune_step / batch_prune / update_step / add_step / remove_step
          / reorder_steps / update_workflow_description
- 结束:   judge_done(report)

用法:
    from context_manager.workflow.judge import WorkflowJudge

    judge = WorkflowJudge(manager, llm)
    result = judge.judge("workflow_id")
    # result = {"tool_calls": [...], "total_tokens": int, "report": str, ...}
"""

from __future__ import annotations

import re

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from .manager import WorkflowManager

# ANSI 转义清理（visualize 输出带颜色，喂给 LLM 前去除）
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

SYSTEM_PROMPT = """你是 Workflow 审查员，负责把 RAW Workflow 修剪为干净、可复用的 SOLIDIFIED Workflow。

你的目标：标记对任务完成没有贡献的步骤（is_pruned=true），保留核心修改与验证步骤。

剪枝标准：
1. 探索性调用：ls/dir/cat/type/read_file 读无关文件、pwd 等导航/查看命令（含藏在 bash 参数里的，如 bash("cd && pwd")）
2. 出错但无关：步骤执行失败，但后续通过其他方式解决了问题
3. 结果被覆盖：同一文件/目标被多次修改，只保留最后一次有效修改
4. 重复验证：同一验证命令多次成功运行，只保留最后一次
5. 保留：有效的写操作（write_file/edit_file）、成功且关键的验证运行（python xxx.py / pytest）

工作流程：
1. 先调用 review_summary 和 list_steps 了解全貌（list_steps 已含参数与结果摘要，通常足够判断）
2. 需要完整细节时用 get_steps 批量查看多个步骤（一次最多 5 个），避免逐个 get_step
3. 用 prune_step 或 batch_prune 标记剪枝（用 step_id 操作）
4. 可选：update_step 修正步骤参数/结果的摘要（删除路径噪音如 cd /sandbox）、update_workflow_description 完善描述
5. 全部完成后，调用 judge_done 提交审查报告（说明剪了什么、为什么、保留了哪些关键步骤）

效率要求：尽可能少调用工具——list_steps 一次能看完全部步骤，不要对每个步骤单独 get_step。

注意：所有修改必须通过工具完成，禁止在对话中输出修改后的步骤内容。
"""


class WorkflowJudge:
    """让 LLM 通过工具审查并修剪 Workflow 的剪枝器。"""

    def __init__(self, manager: WorkflowManager, llm):
        self.manager = manager
        self.llm = llm
        self.tools = self._build_tools()

    # ── 工具构建 ────────────────────────────────────

    def _build_tools(self) -> list:
        m = self.manager

        @tool
        def review_summary(workflow_id: str) -> str:
            """查看 Workflow 审查概况（步骤总数、保留/剪枝数、类型分布）。"""
            return m.review_summary(workflow_id)

        @tool
        def list_steps(workflow_id: str) -> str:
            """列出 Workflow 所有步骤的 step_id/索引/类型/名称/状态（每行一个，含参数与结果摘要）。"""
            wf = m.get_workflow(workflow_id)
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

        @tool
        def get_steps(workflow_id: str, indices: list[int]) -> str:
            """批量查看指定索引步骤的详情（参数、结果、错误）。一次最多 5 个。"""
            wf = m.get_workflow(workflow_id)
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

        @tool
        def get_step(workflow_id: str, step_index: int) -> str:
            """查看指定索引步骤的完整信息（参数、结果、错误）。

            注意参数名是 workflow_id（Workflow 的 ID），step_index 是步骤索引（从 0 开始）。
            """
            return m.get_step(workflow_id, step_index)

        @tool
        def visualize(workflow_id: str, show_pruned: bool = True) -> str:
            """可视化 Workflow 步骤序列（含剪枝原因标注，已去除颜色码）。"""
            text = m.visualize(workflow_id, show_pruned=show_pruned)
            return _ANSI_RE.sub("", text)

        @tool
        def prune_step(step_id: str, is_pruned: bool) -> str:
            """标记/取消标记某个步骤为已剪枝（用 step_id 操作）。"""
            ok = m.prune_step(step_id, is_pruned)
            return f"ok: step {step_id} pruned={is_pruned}" if ok else f"failed: {step_id}"

        @tool
        def batch_prune(workflow_id: str, step_ids: list[str]) -> str:
            """批量标记多个步骤为已剪枝。"""
            return m.batch_prune(workflow_id, step_ids)

        @tool
        def update_step(step_id: str, name: str = None, arguments: str = None,
                        result: str = None, status: str = None,
                        is_pruned: bool = None, type: str = None,
                        error_message: str = None) -> str:
            """更新某个步骤的字段（只传需要修改的字段）。可用于修正参数摘要、状态等。"""
            kwargs = {k: v for k, v in {
                "name": name, "arguments": arguments, "result": result,
                "status": status, "is_pruned": is_pruned, "type": type,
                "error_message": error_message,
            }.items() if v is not None}
            if not kwargs:
                return "no fields to update"
            m.update_step(step_id, **kwargs)
            return f"ok: step {step_id} updated"

        @tool
        def add_step(workflow_id: str, after_index: int, type: str, name: str,
                     arguments: str = "", result: str = "", status: str = "success") -> str:
            """在指定位置后插入一个新步骤。"""
            return m.add_step(workflow_id, after_index, type, name,
                              arguments, result, status)

        @tool
        def remove_step(step_id: str) -> str:
            """删除某个步骤。"""
            return m.remove_step(step_id)

        @tool
        def reorder_steps(workflow_id: str, step_id_order: list[str]) -> str:
            """按给定顺序重新排列步骤（step_id 列表）。"""
            return m.reorder_steps(workflow_id, step_id_order)

        @tool
        def update_workflow_description(workflow_id: str, description: str) -> str:
            """更新 Workflow 描述（用于检索索引）。"""
            ok = m.update_workflow_description(workflow_id, description)
            return f"ok: description updated" if ok else f"failed: {workflow_id}"

        @tool
        def judge_done(report: str) -> str:
            """审查完成，提交最终报告（说明剪了什么、为什么、保留了什么）。"""
            return f"JUDGE_DONE: {report}"

        return [review_summary, list_steps, get_steps, get_step, visualize,
                prune_step, batch_prune, update_step, add_step, remove_step,
                reorder_steps, update_workflow_description, judge_done]

    # ── 执行 ────────────────────────────────────────

    def judge(self, workflow_id: str, max_rounds: int = 25) -> dict:
        """对 Workflow 执行 LLM 剪枝。

        Returns:
            {
                "tool_calls": [{"name", "args"}, ...],
                "input_tokens": int, "output_tokens": int, "total_tokens": int,
                "rounds": int, "report": str, "done": bool,
            }
        """
        llm = self.llm.bind_tools(self.tools)
        messages = [
            ("system", SYSTEM_PROMPT.format(workflow_id=workflow_id)),
            ("user", f"请审查并修剪 Workflow: {workflow_id}。"
                     f"先看概况与步骤清单，再逐项剪枝，最后调用 judge_done 提交报告。"),
        ]
        tool_calls: list[dict] = []
        input_tokens = output_tokens = 0
        done = False
        report = ""
        rounds = 0

        for _ in range(max_rounds):
            rounds += 1
            resp = llm.invoke(messages)
            um = getattr(resp, "usage_metadata", None) or {}
            input_tokens += um.get("input_tokens", 0)
            output_tokens += um.get("output_tokens", 0)
            messages.append(resp)

            tcs = getattr(resp, "tool_calls", None) or []
            if not tcs:
                break

            for tc in tcs:
                name = tc.get("name", "")
                args = tc.get("args") or {}
                tool_calls.append({"name": name, "args": args})
                if name == "judge_done":
                    done = True
                    report = str(args.get("report", ""))
                    break
                result = self._dispatch(name, args)
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            if done:
                break

        return {
            "tool_calls": tool_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "rounds": rounds,
            "report": report,
            "done": done,
        }

    def _dispatch(self, name: str, args: dict) -> str:
        """执行工具调用，捕获异常并返回错误信息给 LLM。"""
        for t in self.tools:
            if t.name == name:
                try:
                    return str(t.invoke(args))
                except Exception as e:  # noqa: BLE001
                    return f"Error: {type(e).__name__}: {e}"
        return f"Error: unknown tool {name}"
