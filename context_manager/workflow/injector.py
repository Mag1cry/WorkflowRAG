"""上下文注入：将 Workflow 格式化为 Agent 上下文。"""

from __future__ import annotations

from ..models import Workflow


def format_context(workflow: Workflow, max_args_length: int = 80) -> str:
    """将 Workflow 格式化为适合注入 Agent 上下文的文本。

    Args:
        workflow: 要格式化的 Workflow 对象。
        max_args_length: 参数显示最大长度，超长截断。

    Returns:
        格式化后的上下文文本。
    """
    kept = [s for s in workflow.steps if not s.is_pruned]
    if not kept:
        return f"[参考 Workflow: {workflow.name}] (无可用步骤)"

    lines = [f"[参考 Workflow: {workflow.name}]"]
    if workflow.description:
        lines.append(f"描述: {workflow.description}")

    for i, step in enumerate(kept):
        args = step.arguments
        if args and len(args) > max_args_length:
            args = args[:max_args_length] + "..."

        result = step.result
        if result and len(result) > 60:
            result = result[:60] + "..."

        step_line = f"Step {i + 1}:  {step.name}({args})"
        if result:
            step_line += f"  → {result}"
        lines.append(step_line)

    return "\n".join(lines)


def format_context_compact(workflow: Workflow) -> str:
    """紧凑格式：仅保留步骤名称和参数摘要。"""
    kept = [s for s in workflow.steps if not s.is_pruned]
    if not kept:
        return f"[Workflow: {workflow.name}] (empty)"

    parts = []
    for s in kept:
        name = s.name
        args = s.arguments
        if args and len(args) > 40:
            args = args[:40] + "..."
        parts.append(f"{name}({args})" if args else name)
    return f"[Workflow: {workflow.name}] {' → '.join(parts)}"
