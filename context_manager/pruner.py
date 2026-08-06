"""Pruner — 剪枝引擎，将 RAW Workflow 转化为 SOLIDIFIED。

剪枝策略（第一版实现规则剪枝）：
1. 结果被覆盖 — 同一目标被多次修改，只保留最后一次
2. 出错但无关 — 步骤失败但后续成功完成
3. 探索性调用 — 只读探索/调试操作
4. LLM 评判 — 保留接口，第一版不实现
"""

from typing import Callable


# 探索性命令关键词
_EXPLORATORY_KEYWORDS = {
    "ls", "cat", "less", "more", "head", "tail", "echo", "print",
    "pwd", "whoami", "date", "which", "find", "grep", "tree",
    "read_file", "list_directory", "search_code",
}

# 写操作命令关键词
_WRITE_KEYWORDS = {
    "edit_file", "write_file", "create_file", "delete_file",
    "rename_file", "mkdir", "touch", "mv", "cp", "rm",
    "pip_install", "conda_install", "npm_install",
    "apply_diff", "insert_content",
}


def _extract_target(name: str, arguments: str) -> str:
    """从 toolcall 参数中提取操作目标（文件名、包名等）。"""
    for kw in ["path=", "file_path=", "file=", "target=", "package="]:

        if kw in arguments:
            start = arguments.find(kw) + len(kw)
            end = arguments.find(",", start)
            if end == -1:
                end = arguments.find("}", start)
            if end == -1:
                end = len(arguments)
            return arguments[start:end].strip().strip('"').strip("'").strip("'")
    for kw in ["'path':", '"path":', "'file_path':", '"file_path":', "'file':", '"file":', "'target':", '"target":']:
        if kw in arguments:
            start = arguments.find(kw) + len(kw)
            start = arguments.find("'", start)
            if start == -1:
                start = arguments.find('"', arguments.find(kw) + len(kw))
            if start == -1:
                continue
            start += 1
            end = arguments.find("'", start)
            if end == -1:
                end = arguments.find('"', start)
            if end == -1:
                end = len(arguments)
            return arguments[start:end].strip()
    return name


def _is_exploratory(name: str) -> bool:
    """判断是否为探索性调用。"""
    return name.lower() in _EXPLORATORY_KEYWORDS


def _is_write_operation(name: str) -> bool:
    """判断是否为写操作。"""
    return name.lower() in _WRITE_KEYWORDS


def prune_result_overwritten(steps: list[dict]) -> list[dict]:
    """策略1：结果被覆盖 — 对同一目标的写操作只保留最后一次。"""
    if not steps:
        return steps

    last_write: dict[str, int] = {}
    for i, step in enumerate(steps):
        if step["type"] == "toolcall" and _is_write_operation(step["name"]):
            target = _extract_target(step["name"], step["arguments"])
            last_write[f"{step['name']}:{target}"] = i

    for i, step in enumerate(steps):
        if step["type"] == "toolcall" and _is_write_operation(step["name"]):
            target = _extract_target(step["name"], step["arguments"])
            key = f"{step['name']}:{target}"
            if key in last_write and i != last_write[key]:
                step["is_pruned"] = True
    return steps


def prune_exploratory(steps: list[dict]) -> list[dict]:
    """策略2：探索性调用 — 只读探索/调试操作。"""
    for step in steps:
        if step["type"] == "bashcall" and _is_exploratory(step["name"]):
            step["is_pruned"] = True
        elif step["type"] == "toolcall" and _is_exploratory(step["name"]):
            step["is_pruned"] = True
    return steps


def prune_failed_irrelevant(steps: list[dict]) -> list[dict]:
    """策略3：出错但无关 — 步骤失败但后续成功完成。"""
    failed_indices = []
    for i, step in enumerate(steps):
        if step["status"] == "failure":
            failed_indices.append(i)

    if not failed_indices:
        return steps

    last_success = -1
    for i in range(len(steps) - 1, -1, -1):
        if steps[i]["status"] == "success" and not steps[i].get("is_pruned"):
            last_success = i
            break

    for idx in failed_indices:
        if idx < last_success:
            steps[idx]["is_pruned"] = True
    return steps


def prune_llm_judgement(steps: list[dict], llm_judge: Callable[[list[dict]], list[bool]] | None = None) -> list[dict]:
    """策略4：LLM 评判（保留接口，第一版不实现）。"""
    if llm_judge is None:
        return steps
    results = llm_judge(steps)
    for step, should_prune in zip(steps, results):
        if should_prune:
            step["is_pruned"] = True
    return steps


def prune(steps: list[dict], llm_judge: Callable[[list[dict]], list[bool]] | None = None) -> list[dict]:
    """执行全部剪枝策略。

    Args:
        steps: Step 字典列表。
        llm_judge: 可选的 LLM 评判函数。

    Returns:
        剪枝后的 Step 列表（is_pruned 标记已更新）。
    """
    steps = prune_exploratory(steps)
    steps = prune_failed_irrelevant(steps)
    steps = prune_result_overwritten(steps)
    steps = prune_llm_judgement(steps, llm_judge)
    return steps


def generate_description(steps: list[dict]) -> str:
    """从已剪枝的步骤序列生成描述文本。"""
    kept = [s for s in steps if not s.get("is_pruned")]
    parts = []
    for s in kept:
        name = s.get("name", "?")
        args = s.get("arguments", "")
        if args and len(args) > 60:
            args = args[:60] + "..."
        parts.append(f"{name}({args})" if args else name)
    return " → ".join(parts) if parts else "(empty workflow)"