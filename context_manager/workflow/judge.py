"""LLM 剪枝 — WorkflowJudge：审查 LLM 通过 function call 操作 Workflow 步骤。

设计：
- 工具集来自 WorkflowManager.get_tool_schemas()（workflow/tools.py 自动生成），
  审查 LLM 只能通过工具修改 Workflow，不能直接输出修改后的内容（防止幻觉）。
- judge() 驱动 LLM 循环：查看 → 剪枝 → 编辑 → judge_done 提交报告。

用法:
    from context_manager.workflow.judge import WorkflowJudge

    judge = WorkflowJudge(manager, llm)   # llm: 支持 function calling 的 ChatOpenAI
    result = judge.judge("workflow_id")
    # result = {"tool_calls": [...], "total_tokens": int, "report": str, ...}
"""

from __future__ import annotations

import re

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

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


def _args_model(parameters: dict) -> type[BaseModel]:
    """OpenAI schema 参数 → pydantic 模型（用于 StructuredTool 校验）。"""
    fields: dict[str, tuple] = {}
    required = set(parameters.get("required", []))
    for pname, spec in parameters.get("properties", {}).items():
        if spec.get("type") == "array":
            inner = spec.get("items", {}).get("type", "string")
            t = list[str] if inner == "string" else list
        else:
            t = {"string": str, "boolean": bool, "integer": int}.get(
                spec.get("type"), str
            )
        desc = spec.get("description", pname)
        if pname in required:
            fields[pname] = (t, Field(description=desc))
        else:
            fields[pname] = (t | None, Field(description=desc, default=None))
    return create_model("ToolArgs", **fields)


class WorkflowJudge:
    """让 LLM 通过工具审查并修剪 Workflow 的剪枝器。"""

    def __init__(self, manager: WorkflowManager, llm):
        self.manager = manager
        self.llm = llm
        self.tools = self._build_tools()

    # ── 工具构建（复用 manager.get_tool_schemas）──────

    def _build_tools(self) -> list:
        tools = []
        for schema in self.manager.get_tool_schemas():
            name = schema["name"]
            tools.append(
                StructuredTool.from_function(
                    name=name,
                    description=schema["description"],
                    args_schema=_args_model(schema["parameters"]),
                    func=self._make_wrapper(name),
                )
            )

        @StructuredTool.from_function
        def judge_done(report: str) -> str:
            """审查完成，提交最终报告（说明剪了什么、为什么、保留了什么）。"""
            return f"JUDGE_DONE: {report}"

        tools.append(judge_done)
        return tools

    def _make_wrapper(self, name: str):
        """包装 manager 方法：执行 + 异常捕获 + 结果字符串化 + ANSI 清理。"""
        fn = getattr(self.manager, name)

        def wrapper(**kwargs) -> str:
            try:
                result = fn(**kwargs)
            except Exception as e:  # noqa: BLE001
                return f"Error: {type(e).__name__}: {e}"
            return _ANSI_RE.sub("", str(result))

        wrapper.__name__ = name
        return wrapper

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
