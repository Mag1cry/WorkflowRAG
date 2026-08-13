"""Demo agent — LangGraph ReAct agent，用于端到端省 Token 评测。

工具集刻意贴近真实 Agent（bash 真执行 + 文件工具），
任务限定在 sandbox 目录内执行，绝不触碰 sandbox 之外的文件。

用法（在 eval/ 目录下运行）:
    DEEPSEEK_API_KEY=sk-xxx python agent.py <sandbox_dir> "<任务描述>"
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# ── 模型 ──────────────────────────────────────────────


def _resolve_api_key() -> str | None:
    """读取 DeepSeek API key：进程环境 → Windows 注册表（系统/用户级）。

    DSH 等宿主进程可能早于 setx 启动而未继承系统环境变量，
    这里直接从注册表回退读取，保证 eval 脚本独立可跑。
    """
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    try:
        import winreg

        for hive, sub in (
            (winreg.HKEY_LOCAL_MACHINE,
             r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            (winreg.HKEY_CURRENT_USER, r"Environment"),
        ):
            try:
                with winreg.OpenKey(hive, sub) as k:
                    val, _ = winreg.QueryValueEx(k, "DEEPSEEK_API_KEY")
                    if val:
                        return val
            except OSError:
                continue
    except ImportError:
        pass
    return None


def build_model() -> ChatOpenAI:
    """DeepSeek API（OpenAI 兼容端点）。"""
    api_key = _resolve_api_key()
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未找到（进程环境变量 + Windows 注册表均无）。"
            "请执行: setx DEEPSEEK_API_KEY sk-xxx"
        )
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_retries=2,
        timeout=120,
    )


# ── 工具 ──────────────────────────────────────────────


def make_tools(sandbox: str) -> list:
    """构造绑定 sandbox 的工具集。所有路径校验必须在 sandbox 内。"""
    sb = Path(sandbox).resolve()
    sb.mkdir(parents=True, exist_ok=True)

    def _guard(path: str) -> Path:
        rp = Path(path)
        if not rp.is_absolute():
            rp = sb / rp
        rp = rp.resolve()
        if not (rp == sb or str(rp).startswith(str(sb) + os.sep)):
            raise ValueError(f"路径越界（禁止访问 sandbox 之外）: {path}")
        return rp

    @tool
    def bash(command: str) -> str:
        """在 sandbox 目录内执行 shell 命令（Windows cmd）。返回 stdout/stderr 与退出码。

        参数必须是单条命令；禁止访问 sandbox 目录之外的路径。
        python 解析为评测环境（agent conda env）的解释器。

        注意：输出通过临时文件重定向读取（不使用管道捕获），
        避免沙箱环境下长输出命令（如 pip install）的管道阻塞。
        """
        env = dict(os.environ)
        py_dir = str(Path(sys.executable).resolve().parent)
        env["PATH"] = py_dir + os.pathsep + env.get("PATH", "")
        # pip 等工具的临时目录可能落在沙箱不可写区（如 %TEMP%\dsh-*），
        # 统一重定向到 sandbox 内的可写目录
        tmp_dir = sb / ".cm_tmp"
        tmp_dir.mkdir(exist_ok=True)
        env["TEMP"] = env["TMP"] = env["TMPDIR"] = str(tmp_dir)
        out_file = sb / f"__cm_out_{os.getpid()}.tmp"
        err_file = sb / f"__cm_err_{os.getpid()}.tmp"
        try:
            with open(out_file, "w", encoding="utf-8", errors="replace") as fo, \
                 open(err_file, "w", encoding="utf-8", errors="replace") as fe:
                try:
                    r = subprocess.run(
                        command, shell=True, cwd=str(sb), stdout=fo, stderr=fe,
                        timeout=180, env=env,
                    )
                    code = r.returncode
                except subprocess.TimeoutExpired:
                    code = -1
            out = out_file.read_text(encoding="utf-8", errors="replace").strip()[-4000:]
            err = err_file.read_text(encoding="utf-8", errors="replace").strip()[-2000:]
        finally:
            for f in (out_file, err_file):
                try:
                    f.unlink()
                except OSError:
                    pass
        if code == -1:
            return "exit_code=-1\nError: 命令超时（>180s）"
        parts = [f"exit_code={code}"]
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr]\n{err}")
        return "\n".join(parts)

    @tool
    def read_file(path: str) -> str:
        """读取 sandbox 内的文本文件（最多返回前 4000 字符）。"""
        rp = _guard(path)
        if not rp.exists():
            return f"Error: {path} 不存在"
        if not rp.is_file():
            return f"Error: {path} 不是文件"
        content = rp.read_text(encoding="utf-8", errors="replace")
        if len(content) > 4000:
            content = content[:4000] + "\n...(截断)"
        return content

    @tool
    def write_file(path: str, content: str) -> str:
        """写入 sandbox 内的文本文件（自动创建父目录）。"""
        rp = _guard(path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(content, encoding="utf-8")
        return f"written: {path} ({len(content)} chars)"

    @tool
    def list_dir(path: str = ".") -> str:
        """列出 sandbox 内目录的内容。"""
        rp = _guard(path)
        if not rp.is_dir():
            return f"Error: {path} 不是目录"
        items = [f"{p.name}/" if p.is_dir() else p.name for p in sorted(rp.iterdir())]
        return "\n".join(items[:200]) if items else "(empty)"

    return [bash, read_file, write_file, list_dir]


# ── Agent ─────────────────────────────────────────────


def build_agent(sandbox: str, system_extra: str = ""):
    """构建 ReAct agent + MemorySaver checkpointer。

    Returns:
        (agent, checkpointer)
    """
    tools = make_tools(sandbox)
    model = build_model()
    prompt = (
        "你是一个在隔离 sandbox 目录中完成文件操作任务的 agent。\n"
        "规则：\n"
        "1. 只能操作 sandbox 目录内的文件，禁止访问其他路径\n"
        "2. 用 bash 执行 python/pytest 等命令验证结果\n"
        "3. 环境是 Windows：命令在 cmd 兼容 shell 中执行，python 命令可直接使用；"
        "文件浏览请优先使用 list_dir/read_file 工具，避免使用 ls/cat/chmod 等 Linux 命令\n"
        "4. 任务完成后，用最终回答简要说明你做了什么、结果如何\n"
        + (system_extra or "")
    )
    checkpointer = MemorySaver()
    agent = create_react_agent(
        model, tools, prompt=prompt, checkpointer=checkpointer,
    )
    return agent, checkpointer


# ── 运行与统计 ────────────────────────────────────────


def run_and_stats(agent, thread_id: str, task_prompt: str, max_steps: int = 50) -> dict:
    """运行 agent 一次，返回统计信息。

    Returns:
        {
            "tool_calls": [...],           # 每次工具调用的 (name, args) 
            "tool_call_count": int,
            "prompt_tokens": int,          # 累计
            "completion_tokens": int,
            "total_tokens": int,
            "final_answer": str,
            "steps": int,                  # 模型调用轮数
        }
    """
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max(100, max_steps * 2 + 10),
    }
    tool_calls: list[dict] = []
    prompt_tokens = completion_tokens = total_tokens = 0
    steps = 0
    final_answer = ""

    for chunk in agent.stream(
        {"messages": [("user", task_prompt)]},
        config=config,
        stream_mode="values",
    ):
        messages = chunk.get("messages", [])
        if not messages:
            continue
        msg = messages[-1]
        if getattr(msg, "type", "") == "ai":
            steps += 1
            um = getattr(msg, "usage_metadata", None) or {}
            prompt_tokens += um.get("input_tokens", 0)
            completion_tokens += um.get("output_tokens", 0)
            total_tokens += um.get("total_tokens", 0) or (
                um.get("input_tokens", 0) + um.get("output_tokens", 0)
            )
            for tc in getattr(msg, "tool_calls", []) or []:
                tool_calls.append({"name": tc.get("name"), "args": tc.get("args")})
        elif getattr(msg, "type", "") == "ai" or (
            getattr(msg, "content", "") and getattr(msg, "type", "") in ("ai", "human")
        ):
            pass
        if getattr(msg, "type", "") == "ai" and not getattr(msg, "tool_calls", []):
            final_answer = getattr(msg, "content", "") or ""
        if steps > max_steps:
            break

    return {
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "final_answer": final_answer[:500],
        "steps": steps,
    }


# ── CLI ───────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    sandbox = sys.argv[1]
    task_prompt = sys.argv[2]
    agent, _cp = build_agent(sandbox)
    stats = run_and_stats(agent, "cli_thread", task_prompt)
    print(f"\n=== 工具调用 {stats['tool_call_count']} 次 / "
          f"token {stats['total_tokens']} (prompt {stats['prompt_tokens']} + "
          f"completion {stats['completion_tokens']}) / {stats['steps']} 轮 ===")
    for tc in stats["tool_calls"]:
        args = str(tc["args"])[:80]
        print(f"  → {tc['name']}({args}...)")
    print(f"\n=== 最终回答 ===\n{stats['final_answer']}")


if __name__ == "__main__":
    main()
