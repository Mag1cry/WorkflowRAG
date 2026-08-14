"""WorkflowManager CLI — 统一入口。

用法:
    python cli.py demo [--offline]  → 完整样例：真实 Agent 任务 → 提取 → LLM 剪枝
                                       → 固化 → 检索 → 注入复用 → 对比省 token
                                       （--offline 为假数据离线演示，无需 API key）
    python cli.py review <thread>   → 一键审查：提取 → 展示 → 固化
    python cli.py case              → 内置案例：剪枝效果对比
    python cli.py --list-tools      → 输出审查 LLM 工具的 function call schema
"""

import argparse
import sys


def cmd_demo(args) -> int:
    """完整样例：委托 demo.py（真实 LLM 闭环 / 离线演示）。"""
    import demo

    return demo.main_with_args(offline=args.offline)


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

    demo_parser = subparsers.add_parser("demo", help="完整样例（真实 LLM 闭环）")
    demo_parser.add_argument("--offline", action="store_true",
                             help="离线演示（假数据，无需 API key）")

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
        sys.exit(cmd_demo(args))
    elif args.command == "case":
        cmd_case(args)
    elif args.command == "review":
        cmd_review(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
