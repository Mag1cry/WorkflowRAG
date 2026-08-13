"""上下文可视化：将 Workflow 以结构化方式展示，区分不同工具调用类型。"""

from __future__ import annotations

from ..models import Workflow, Step


# ANSI 颜色
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_BLUE = "\033[94m"
_RED = "\033[91m"
_GRAY = "\033[90m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _step_icon(step: Step) -> str:
    if step.is_pruned:
        return "·"
    if step.type == "bashcall":
        return "$"
    return "→"


def _step_color(step: Step) -> str:
    if step.is_pruned:
        return _GRAY
    if step.status == "failure":
        return _RED
    if step.name in ("read_file", "search_code", "grep", "list_directory"):
        return _BLUE
    if step.name in ("edit_file", "write_file", "create_file", "apply_diff", "insert_content"):
        return _CYAN
    if step.type == "bashcall" and step.name in ("python", "pytest", "npm_test"):
        return _GREEN
    if step.type == "bashcall":
        return _YELLOW
    return _RESET


def _prune_reason(step: Step, prev_steps: list[Step]) -> str:
    if not step.is_pruned:
        return ""
    name = step.name.lower()
    if step.type == "bashcall" and name in ("ls", "cat", "echo", "pwd", "which", "date", "print"):
        return "探索性"
    if name in ("read_file", "search_code", "grep", "list_directory"):
        if not any(s.name == step.name and not s.is_pruned for s in prev_steps):
            return "探索性"
    if step.status == "failure":
        return "出错无关"
    for ps in prev_steps:
        if ps.is_pruned:
            continue
        if ps.name == step.name and ps.type == step.type:
            return "被覆盖"
    return "已剪枝"


def visualize_workflow(wf: Workflow, show_pruned: bool = True) -> str:
    """将 Workflow 可视化为带颜色分类的文本块。

    Args:
        wf: 要展示的 Workflow 对象。
        show_pruned: 是否显示已剪枝的步骤（灰显）。

    Returns:
        格式化后的可视化文本。
    """
    lines = []
    sep = "─" * 58

    status_tag = f"{_GREEN}SOLIDIFIED{_RESET}" if wf.status == "SOLIDIFIED" else f"{_YELLOW}RAW{_RESET}"
    lines.append(f"{_BOLD}┌─ {wf.name}  [{status_tag}]{_RESET}")
    if wf.description and wf.description != "(empty workflow)":
        lines.append(f"│  描述: {_GRAY}{wf.description}{_RESET}")
    lines.append(f"│{_GRAY}{sep}{_RESET}")

    if not wf.steps:
        lines.append(f"│  {_GRAY}(无步骤){_RESET}")
    else:
        kept = sum(1 for s in wf.steps if not s.is_pruned)
        total = len(wf.steps)
        for i, step in enumerate(wf.steps):
            if step.is_pruned and not show_pruned:
                continue

            color = _step_color(step)
            icon = _step_icon(step)
            idx = f"{step.step_index + 1:02d}"

            name_display = step.name
            type_display = step.type

            if step.is_pruned:
                reason = _prune_reason(step, wf.steps[:i])
                tag = f"  {_RED}({reason}){_RESET}"
            else:
                tag = ""

            line = f"│  {color}{idx} {icon} [{type_display:8s}] {name_display}{tag}{_RESET}"

            args = step.arguments
            if args and len(args) > 50:
                args = args[:50] + "..."
            if args:
                line += f"\n│       args: {_GRAY}{args}{_RESET}"

            if step.result and not step.is_pruned:
                result = step.result
                if len(result) > 60:
                    result = result[:60] + "..."
                line += f"\n│       → {result}"

            if step.status == "failure":
                err = step.error_message or step.result
                if err and len(err) > 60:
                    err = err[:60] + "..."
                line += f"\n│       {_RED}✗ {err}{_RESET}"

            lines.append(line)

        lines.append(f"│{_GRAY}{sep}{_RESET}")
        status = f"{_GREEN}保留 {kept}{_RESET} | {_RED}剪枝 {total - kept}{_RESET}" if total > kept else f"{_GREEN}保留 {kept}{_RESET}"
        lines.append(f"│  步骤: {total}  |  {status}")

    lines.append(f"{_BOLD}└{_RESET}{'─' * 58}")
    return "\n".join(lines)


def visualize_comparison(raw: Workflow, solidified: Workflow) -> str:
    """并排对比 RAW 和 SOLIDIFIED 两个 Workflow。"""
    lines = []
    lines.append(f"{_BOLD}{'=' * 60}{_RESET}")
    lines.append(f"{_BOLD}  剪枝效果对比{_RESET}")
    lines.append(f"{_BOLD}{'=' * 60}{_RESET}")
    lines.append("")

    lines.append(f"{_BOLD}【剪枝前 RAW】{_RESET}")
    lines.append(visualize_workflow(raw))
    lines.append("")

    raw_steps = len(raw.steps)
    solid_steps = len([s for s in solidified.steps if not s.is_pruned])
    reduction = raw_steps - solid_steps
    pct = int(reduction / raw_steps * 100) if raw_steps > 0 else 0

    lines.append(f"{_BOLD}【剪枝后 SOLIDIFIED】{_RESET}")
    lines.append(visualize_workflow(solidified))
    lines.append("")

    lines.append(f"{_BOLD}{'=' * 60}{_RESET}")
    lines.append(f"  {_BOLD}结果:{_RESET} {raw_steps} 步 → {solid_steps} 步  ({_GREEN}-{pct}%{_RESET})")
    lines.append(f"  剪枝率: {_GREEN}{pct}%{_RESET} 的噪音步骤被移除")
    lines.append(f"{_BOLD}{'=' * 60}{_RESET}")

    return "\n".join(lines)


def build_case_study() -> tuple[Workflow, Workflow]:
    """构建一个真实的案例场景，用于演示剪枝效果。

    场景: Agent 修复一个 Python 包导入错误。
    """
    raw_steps = [
        Step(step_id="s1", workflow_id="case", step_index=0, type="bashcall",
             name="ls", arguments="{'dir': '.'}", result="src/ tests/"),
        Step(step_id="s2", workflow_id="case", step_index=1, type="toolcall",
             name="read_file", arguments="{'path': 'src/main.py'}", result="from utils import helper"),
        Step(step_id="s3", workflow_id="case", step_index=2, type="bashcall",
             name="python", arguments="{'script': 'src/main.py'}", result="ModuleNotFoundError: No module named 'utils'"),
        Step(step_id="s4", workflow_id="case", step_index=3, type="bashcall",
             name="ls", arguments="{'dir': 'src/'}", result="main.py"),
        Step(step_id="s5", workflow_id="case", step_index=4, type="bashcall",
             name="pip_install", arguments="{'package': 'utils'}", result="ERROR: Could not find a version"),
        Step(step_id="s6", workflow_id="case", step_index=5, type="toolcall",
             name="grep", arguments="{'pattern': 'utils', 'path': 'src/'}", result="src/main.py: from utils import helper"),
        Step(step_id="s7", workflow_id="case", step_index=6, type="toolcall",
             name="read_file", arguments="{'path': 'src/main.py'}", result="from utils import helper"),
        Step(step_id="s8", workflow_id="case", step_index=7, type="toolcall",
             name="edit_file", arguments="{'path': 'src/main.py', 'content': 'import helper'}"),
        Step(step_id="s9", workflow_id="case", step_index=8, type="bashcall",
             name="python", arguments="{'script': 'src/main.py'}", result="ModuleNotFoundError: No module named 'helper'"),
        Step(step_id="s10", workflow_id="case", step_index=9, type="toolcall",
             name="edit_file", arguments="{'path': 'src/main.py', 'content': 'from src.helper import helper'}"),
        Step(step_id="s11", workflow_id="case", step_index=10, type="bashcall",
             name="python", arguments="{'script': 'src/main.py'}", result="success"),
        Step(step_id="s12", workflow_id="case", step_index=11, type="bashcall",
             name="ls", arguments="{'dir': 'src/'}", result="main.py helper.py"),
    ]

    raw = Workflow(
        workflow_id="case_study",
        name="修复 Python 导入错误",
        status="RAW",
        steps=raw_steps,
    )

    # 模拟剪枝后的结果
    solidified_steps = [
        Step(step_id="s2", workflow_id="case", step_index=0, type="toolcall",
             name="read_file", arguments="{'path': 'src/main.py'}", result="from utils import helper"),
        Step(step_id="s3", workflow_id="case", step_index=1, type="bashcall",
             name="python", arguments="{'script': 'src/main.py'}", result="ModuleNotFoundError: No module named 'utils'"),
        Step(step_id="s6", workflow_id="case", step_index=2, type="toolcall",
             name="grep", arguments="{'pattern': 'utils', 'path': 'src/'}", result="src/main.py: from utils import helper"),
        Step(step_id="s10", workflow_id="case", step_index=3, type="toolcall",
             name="edit_file", arguments="{'path': 'src/main.py', 'content': 'from src.helper import helper'}", result="success"),
        Step(step_id="s11", workflow_id="case", step_index=4, type="bashcall",
             name="python", arguments="{'script': 'src/main.py'}", result="success"),
    ]

    solidified = Workflow(
        workflow_id="case_study",
        name="修复 Python 导入错误",
        status="SOLIDIFIED",
        description="read_file({'path': 'src/main.py'}) → python({'script': 'src/main.py'}) → grep(...) → edit_file(...) → python({'script': 'src/main.py'})",
        steps=solidified_steps,
    )

    return raw, solidified