"""WorkflowManager CLI — 统一入口。

用法:
    python cli.py demo              → 运行 demo
    python cli.py review <thread>   → 一键审查：提取 → 展示 → 固化
"""

import argparse
import sys


def cmd_demo(args) -> None:
    """运行 demo：展示提取 → 剪枝 → 检索流程。"""
    from context_manager import create_memory_manager

    wfm = create_memory_manager()

    print("\n=== WorkflowManager Demo ===\n")

    # 模拟原始步骤
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
        workflow_id=wf_id,
        name="修复 main.py 返回值",
        source_thread_id="demo_thread",
    )
    for s in fake_steps:
        wfm.workflow_store.add_step(**s, workflow_id=wf_id)

    print(f"创建 RAW Workflow: {wf_id}")
    print(f"  原始步骤数: {len(fake_steps)}")

    print("\n=== 剪枝前 ===")
    for s in fake_steps:
        print(f"  [ ] [{s['type']:8s}] {s['name']}({s['arguments'][:40]}...)")

    print("\n=== LLM 剪枝（审查工具操作）===")
    print("  [x] read_file   → 探索性调用，剪枝")
    wfm.prune_step("s1", True)
    print("  [x] ls          → 探索性调用，剪枝")
    wfm.prune_step("s2", True)
    print("  [x] edit_file v1→ 结果被 v2 覆盖，剪枝")
    wfm.prune_step("s3", True)

    print("\n=== 执行固化 (solidify) ===")
    wfm.solidify(wf_id)

    wf = wfm.get_workflow(wf_id)
    if wf:
        print("\n=== 注入上下文 ===\n")
        print(wfm.format_context(wf))

    print("\n=== 检索: '如何修改 Python 函数返回值' ===\n")
    results = wfm.retrieve("如何修改 Python 函数返回值", top_k=3)
    for r in results:
        print(f"  [{r.workflow_id}] similarity=-- | {r.description[:60]}...")
        for s in r.steps:
            print(f"    Step {s.step_index}: {s.name}({s.arguments[:40]}...)")

    print("\n=== Workflow 列表 ===\n")
    for wf in wfm.list_workflows():
        print(f"  {wf.workflow_id} [{wf.status}] — {wf.name}")

    wfm.close()


def cmd_review(args) -> None:
    """一键审查：提取 RAW Workflow → 展示审查摘要 → 固化。"""
    from context_manager import WorkflowManager

    wfm = WorkflowManager()

    thread_id = args.thread
    print(f"\n=== 审查: thread {thread_id} ===\n")

    wf_id = wfm.extract_workflow(thread_id)
    wf = wfm.get_workflow(wf_id)
    if wf is None:
        print("提取失败")
        wfm.close()
        return

    print(f"Workflow: {wf.name} ({wf.workflow_id})")
    print(f"步骤数: {len(wf.steps)}")
    print()

    for s in wf.steps:
        marker = " "
        print(f"  [{marker}] [{s.type:8s}] {s.name}")

    print("\n执行固化 (solidify)...")
    wfm.solidify(wf_id)

    wf = wfm.get_workflow(wf_id)
    if wf:
        kept = [s for s in wf.steps if not s.is_pruned]
        pruned = [s for s in wf.steps if s.is_pruned]
        print(f"\n保留: {len(kept)} 步骤 | 剪枝: {len(pruned)} 步骤")
        print(f"描述: {wf.description}")

    wfm.close()


def cmd_case(args) -> None:
    """运行内置案例，展示剪枝效果。"""
    from context_manager.workflow.visualizer import build_case_study, visualize_comparison

    raw, solidified = build_case_study()
    print()
    print(visualize_comparison(raw, solidified))
    print()

    # 展示上下文注入对比
    print("=" * 60)
    print("  上下文注入（剪枝前 vs 剪枝后）")
    print("=" * 60)
    print()

    from context_manager.workflow.injector import format_context

    print("【剪枝前 - 注入 Agent 上下文】")
    print("-" * 40)
    print(format_context(raw))
    print()
    print("【剪枝后 - 注入 Agent 上下文】")
    print("-" * 40)
    print(format_context(solidified))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WorkflowManager CLI — AI Workflow 管理引擎",
    )
    parser.add_argument("--list-tools", action="store_true",
                        help="输出所有审查 LLM 工具的 function call schema（JSON）")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    subparsers.add_parser("demo", help="运行 demo")
    subparsers.add_parser("case", help="运行剪枝效果案例")

    review_parser = subparsers.add_parser("review", help="审查 Thread 的工作流")
    review_parser.add_argument("thread", help="LangGraph Thread ID")

    args = parser.parse_args()

    if args.list_tools:
        from context_manager import create_memory_manager
        wfm = create_memory_manager()
        import json
        print(json.dumps(wfm.get_tool_schemas(), indent=2, ensure_ascii=False))
        wfm.close()
        return

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "case":
        cmd_case(args)
    elif args.command == "review":
        cmd_review(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()