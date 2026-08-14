"""完整样例 — WorkflowManager 端到端演示。

展示完整闭环：真实 Agent 执行任务 → 提取 → LLM 剪枝 → 固化 → 检索 → 注入复用 → 对比省 token。

用法:
    python demo.py                # 完整流程（需要 DEEPSEEK_API_KEY，真实调用 LLM）
    python demo.py --offline      # 离线演示（假数据，无需 key）

流程:
    1. Agent 在沙箱中完成"修复 Python 导入错误"任务（第一次执行，无参考）
    2. WorkflowManager 从事后 Checkpoints 提取 RAW Workflow
    3. WorkflowJudge（LLM 审查 Agent）通过工具剪枝 → 审查报告
    4. solidify 固化 → FAISS 索引 → 检索命中验证
    5. 注入剪枝后的工作流，Agent 再次完成同一任务（第二次执行）
    6. 对比两次执行：工具调用次数 / token 消耗 / 是否成功
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Windows 控制台默认 GBK，无法输出 emoji/部分中文 → 强制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from langchain_openai import ChatOpenAI  # noqa: E402

from context_manager import WorkflowManager  # noqa: E402
from context_manager.config import Settings  # noqa: E402
from context_manager.persistence.embedding import M3EEmbedding  # noqa: E402
from context_manager.workflow.injector import format_context_compact  # noqa: E402
from context_manager.workflow.judge import WorkflowJudge  # noqa: E402

from eval.agent import _resolve_api_key, build_agent, run_and_stats  # noqa: E402
from eval.tasks import T1_PROMPT, init_t1, verify_t1  # noqa: E402

DEMO_DIR = ROOT / "eval" / "tmp" / "demo"
DB_PATH = ROOT / "eval" / "tmp" / "demo.db"

SEP = "=" * 72


def _banner(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=_resolve_api_key(),
        base_url="https://api.deepseek.com",
        temperature=0,
        max_retries=2,
        timeout=120,
    )


def _fresh_sandbox(tag: str) -> Path:
    sb = DEMO_DIR / tag
    shutil.rmtree(sb, ignore_errors=True)
    sb.mkdir(parents=True)
    return sb


def _print_stats(label: str, stats: dict, elapsed_s: float) -> None:
    print(f"  {label}: 工具调用 {stats['tool_call_count']} 次 | "
          f"token {stats['total_tokens']} (prompt {stats['prompt_tokens']} + "
          f"completion {stats['completion_tokens']}) | {stats['steps']} 轮 | "
          f"耗时 {elapsed_s:.1f}s")
    for tc in stats["tool_calls"]:
        args = str(tc["args"])[:70]
        print(f"      → {tc['name']}({args}...)")


def _show_workflow(wfm: WorkflowManager, wf_id: str, title: str) -> None:
    wf = wfm.get_workflow(wf_id)
    print(f"\n  {title} ({wf.status}, 共 {len(wf.steps)} 步):")
    for s in wf.steps:
        mark = "✗ 剪枝" if s.is_pruned else "✓ 保留"
        args = (s.arguments or "")[:55]
        print(f"    [{mark}] #{s.step_index} {s.type:8s} {s.name:12s} {args}...")


def run_full() -> int:
    """完整流程（真实 LLM）。"""
    key = _resolve_api_key()
    if not key:
        print("错误: 完整样例需要 DEEPSEEK_API_KEY。\n"
              "  提示: 可用 `python demo.py --offline` 运行离线演示。")
        return 1

    llm = _build_llm()

    _banner("1. 第一次执行 — Agent 无参考完成『修复 Python 导入错误』")
    sb1 = _fresh_sandbox("run1")
    init_t1(sb1)
    agent1, checkpointer = build_agent(str(sb1))
    t0 = time.time()
    stats1 = run_and_stats(agent1, "demo_thread_1", T1_PROMPT)
    elapsed1 = time.time() - t0
    passed1, detail1 = verify_t1(sb1)
    _print_stats("第一次执行", stats1, elapsed1)
    print(f"  结果: {'✅ 成功' if passed1 else '❌ 失败'} | {detail1[:60]}")

    _banner("2. 提取 RAW Workflow（从 Checkpoints 事后提取）")
    settings = Settings(storage_path=str(DB_PATH))
    if DB_PATH.exists():
        DB_PATH.unlink()
    embedding = M3EEmbedding()
    wfm = WorkflowManager(settings=settings, checkpointer=checkpointer, embedding=embedding)
    wf_id = wfm.extract_workflow("demo_thread_1", name="修复 Python 导入错误")
    _show_workflow(wfm, wf_id, "RAW Workflow")

    _banner("3. WorkflowJudge — LLM 审查剪枝")
    judge = WorkflowJudge(wfm, llm)
    judge_result = judge.judge(wf_id)
    print(f"  审查过程: {len(judge_result['tool_calls'])} 次工具调用 | "
          f"{judge_result['total_tokens']} tokens | {judge_result['rounds']} 轮")
    for tc in judge_result["tool_calls"]:
        args = str(tc["args"])[:70]
        print(f"      → {tc['name']}({args}...)")
    print(f"\n  📋 审查报告:\n{judge_result['report']}")
    _show_workflow(wfm, wf_id, "剪枝后 Workflow")

    _banner("4. 固化 + 检索验证")
    wfm.solidify(wf_id)
    hits = wfm.retrieve(T1_PROMPT, top_k=1)
    print(f"  检索『修复导入错误』: 命中 {len(hits)} 个")
    if hits:
        print(f"    描述: {hits[0].description[:90]}")

    _banner("5. 第二次执行 — 注入 LLM 剪枝后的工作流")
    injection = format_context_compact(hits[0])
    print(f"  注入内容（{len(injection)} 字符）:\n  {injection}")
    sb2 = _fresh_sandbox("run2")
    init_t1(sb2)
    system_extra = (
        "\n以下是你上次完成该任务、经 LLM 审查剪枝后的工作流，"
        "请直接参考复用，避免重复探索：\n" + injection
    )
    agent2, _ = build_agent(str(sb2), system_extra=system_extra)
    t0 = time.time()
    stats2 = run_and_stats(agent2, "demo_thread_2", T1_PROMPT)
    elapsed2 = time.time() - t0
    passed2, detail2 = verify_t1(sb2)
    _print_stats("第二次执行", stats2, elapsed2)
    print(f"  结果: {'✅ 成功' if passed2 else '❌ 失败'} | {detail2[:60]}")

    _banner("6. 对比总结")
    save_calls = stats1["tool_call_count"] - stats2["tool_call_count"]
    save_tokens = stats1["total_tokens"] - stats2["total_tokens"]
    rate = save_tokens / stats1["total_tokens"] * 100 if stats1["total_tokens"] else 0
    print(f"  {'指标':<12}{'无参考':<12}{'注入复用':<12}{'节省':<10}")
    print(f"  {'-' * 46}")
    print(f"  {'工具调用':<10}{stats1['tool_call_count']:<12}"
          f"{stats2['tool_call_count']:<12}{save_calls:<10}")
    print(f"  {'token':<10}{stats1['total_tokens']:<12}"
          f"{stats2['total_tokens']:<12}{save_tokens} ({rate:.1f}%)")
    print(f"  {'成功率':<10}{'✅' if passed1 else '❌':<12}{'✅' if passed2 else '❌':<12}")
    print(f"\n  LLM 剪枝成本（一次性）: {judge_result['total_tokens']} tokens，"
          f"复用 {max(1, round(judge_result['total_tokens'] / max(save_tokens, 1)))} 次回本")
    if rate >= 10:
        verdict = f"注入显著省 token ✅（节省 {rate:.1f}%）"
    else:
        verdict = "本次任务注入收益有限——注入适用于固定流程/复杂任务（见 docs/EvalReport.md）"
    print(f"\n  结论: {verdict}")

    wfm.close()
    return 0


def run_offline() -> int:
    """离线演示：假数据展示 提取 → 工具剪枝 → 固化 → 检索 → 注入 全流程。"""
    from context_manager import create_memory_manager

    _banner("离线演示（假数据）— 提取 → 工具剪枝 → 固化 → 检索 → 注入")
    wfm = create_memory_manager()

    fake_steps = [
        {"step_id": "s1", "step_index": 0, "type": "toolcall", "name": "read_file",
         "arguments": "{'path': 'src/main.py'}", "result": "def main(): pass",
         "status": "success", "duration_ms": 100, "error_message": "", "timestamp": "2024-01-01"},
        {"step_id": "s2", "step_index": 1, "type": "bashcall", "name": "ls",
         "arguments": "{'dir': 'src/'}", "result": "main.py utils.py",
         "status": "success", "duration_ms": 50, "error_message": "", "timestamp": "2024-01-01"},
        {"step_id": "s3", "step_index": 2, "type": "toolcall", "name": "edit_file",
         "arguments": "{'path': 'src/main.py', 'content': 'def main(): return 42'}",
         "result": "success", "status": "success", "duration_ms": 200, "error_message": "", "timestamp": "2024-01-01"},
        {"step_id": "s4", "step_index": 3, "type": "toolcall", "name": "edit_file",
         "arguments": "{'path': 'src/main.py', 'content': 'def main(): return 99'}",
         "result": "success", "status": "success", "duration_ms": 150, "error_message": "", "timestamp": "2024-01-01"},
        {"step_id": "s5", "step_index": 4, "type": "bashcall", "name": "python",
         "arguments": "{'script': 'src/main.py'}", "result": "99",
         "status": "success", "duration_ms": 300, "error_message": "", "timestamp": "2024-01-01"},
    ]

    wf_id = "demo_workflow_001"
    wfm.workflow_store.create_workflow(
        workflow_id=wf_id, name="修复 main.py 返回值", source_thread_id="demo_thread")
    for s in fake_steps:
        wfm.workflow_store.add_step(**s, workflow_id=wf_id)

    print(f"\n  RAW Workflow: {wf_id}（{len(fake_steps)} 步，含探索/被覆盖噪音）")
    print("\n  LLM 审查工具操作（模拟 LLM 剪枝决策）:")
    print("    [x] read_file     → 探索性调用，剪枝")
    wfm.prune_step("s1", True)
    print("    [x] ls            → 探索性调用，剪枝")
    wfm.prune_step("s2", True)
    print("    [x] edit_file v1  → 结果被 v2 覆盖，剪枝")
    wfm.prune_step("s3", True)

    wfm.solidify(wf_id)
    wf = wfm.get_workflow(wf_id)
    kept = [s for s in wf.steps if not s.is_pruned]
    print(f"\n  固化完成: {len(kept)}/{len(wf.steps)} 步保留 → {wf.description[:70]}")

    results = wfm.retrieve("如何修改 Python 函数返回值", top_k=3)
    print(f"\n  检索『修改函数返回值』命中 {len(results)} 个:")
    for r in results:
        print(f"    {r.workflow_id} | {r.description[:60]}")

    print("\n  注入上下文（紧凑格式）:")
    print(f"    {format_context_compact(wf)}")
    print("\n  （完整样例请运行 `python demo.py`，需要 DEEPSEEK_API_KEY）")
    wfm.close()
    return 0


def main_with_args(offline: bool = False) -> int:
    """供 cli.py demo 命令调用。"""
    t0 = time.time()
    code = run_offline() if offline else run_full()
    print(f"\n完成，总耗时 {time.time() - t0:.1f}s")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WorkflowManager 完整样例：提取 → LLM 剪枝 → 固化 → 检索 → 注入复用")
    parser.add_argument("--offline", action="store_true",
                        help="离线演示（假数据，无需 API key）")
    args = parser.parse_args()
    return main_with_args(offline=args.offline)


if __name__ == "__main__":
    sys.exit(main())
