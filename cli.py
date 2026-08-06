"""WorkflowManager CLI — 统一入口。

用法:
    python cli.py demo      → 运行 demo（展示提取 → 剪枝 → 检索流程）
    python cli.py serve     → 启动 FastAPI 服务（后续版本）
"""

import argparse
import sys


def cmd_demo(args) -> None:
    """运行 demo：模拟提取 → 剪枝 → 检索流程。"""
    from context_manager import create_memory_manager

    wfm = create_memory_manager()

    print("\n=== WorkflowManager Demo ===\n")

    # 模拟从 LangGraph 消息中提取的步骤
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

    # 模拟创建 RAW Workflow
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

    # 展示剪枝效果
    print("\n=== 剪枝前 ===")
    for s in fake_steps:
        marker = " "
        print(f"  [{marker}] [{s['type']:8s}] {s['name']}({s['arguments'][:40]}...)")

    print("\n=== 执行剪枝 ===")
    wfm.solidify(wf_id)

    # 检索
    print("\n=== 检索: '如何修改 Python 函数返回值' ===\n")
    results = wfm.retrieve("如何修改 Python 函数返回值", top_k=3)
    for r in results:
        print(f"  [{r['workflow_id']}] similarity={r['similarity']} | {r['description'][:60]}...")
        for s in r["steps"]:
            print(f"    Step {s['step_index']}: {s['name']}({s['arguments'][:40]}...)")

    print("\n=== Workflow 列表 ===\n")
    for wf in wfm.list_workflows():
        print(f"  {wf['workflow_id']} [{wf['status']}] — {wf['name']}")

    wfm.close()


def cmd_serve(args) -> None:
    """启动 FastAPI 服务（后续版本实现）。"""
    print("[serve] FastAPI 服务将在后续版本中实现。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WorkflowManager CLI — AI Workflow 管理引擎",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    subparsers.add_parser("demo", help="运行 demo")
    serve_parser = subparsers.add_parser("serve", help="启动 FastAPI 服务")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()