"""runner — 端到端省 Token 评测主脚本。

基线模式:    agent 无参考直接跑任务 N 次
注入模式:    第 1 次跑 → extract_workflow → WorkflowJudge(LLM 剪枝) → solidify
             → retrieve → 注入上下文 → 再跑 N 次

用法（在项目根目录运行）:
    python eval/runner.py --task T1 --mode both --runs 3
    python eval/runner.py --task all --mode both --runs 3

输出: eval/results/<task>_<mode>.json + 终端对比表
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.agent import build_agent, run_and_stats, _resolve_api_key  # noqa: E402
from eval.tasks import TASKS_REGISTRY, TASKS  # noqa: E402

from context_manager import WorkflowManager  # noqa: E402
from context_manager.config import Settings  # noqa: E402
from context_manager.persistence.embedding import M3EEmbedding  # noqa: E402
from context_manager.workflow.injector import format_context_compact  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SANDBOX_ROOT = Path(__file__).resolve().parent / "tmp" / "sandboxes"


# ── 单次执行 ──────────────────────────────────────────


def run_once(
    task_key: str, thread_id: str, system_extra: str = "", max_steps: int = 50
) -> dict:
    """初始化 sandbox 并运行 agent 一次，返回统计 + verify 结果。

    sandbox 固定放在 eval/tmp/sandboxes/ 下（workspace 内，避免临时目录权限问题）。
    """
    spec = TASKS_REGISTRY[task_key]
    sandbox = SANDBOX_ROOT / f"{task_key}_{uuid.uuid4().hex[:6]}"
    sandbox.mkdir(parents=True, exist_ok=True)
    spec["init"](sandbox)

    agent, checkpointer = build_agent(str(sandbox), system_extra=system_extra)
    t0 = time.time()
    stats = run_and_stats(agent, thread_id, spec["prompt"], max_steps=max_steps)
    elapsed = time.time() - t0

    passed, detail = spec["verify"](sandbox)
    stats.update(
        {
            "passed": passed,
            "verify_detail": detail,
            "elapsed_s": round(elapsed, 1),
            "sandbox": str(sandbox),
        }
    )
    return stats, checkpointer


# ── Workflow 提取与注入 ───────────────────────────────


def build_injection(
    task_key: str, checkpointer, db_path: Path, embedding: M3EEmbedding, judge_llm
) -> tuple[str | None, dict | None]:
    """从 run 的 checkpointer 提取 → LLM 剪枝 → 固化 → 检索 → 生成注入文本。

    Returns:
        (injection_text, judge_stats)
    """
    spec = TASKS_REGISTRY[task_key]
    settings = Settings(storage_path=str(db_path))
    wfm = WorkflowManager(
        settings=settings, checkpointer=checkpointer, embedding=embedding
    )
    judge_stats = None
    try:
        wf_id = wfm.extract_workflow("thread_1", name=task_key)
        raw = wfm.get_workflow(wf_id)
        raw_count = len(raw.steps) if raw else 0

        from context_manager.workflow.judge import WorkflowJudge

        judge = WorkflowJudge(wfm, judge_llm)
        judge_stats = judge.judge(wf_id)
        print(
            f"  [LLM剪枝] {wf_id} | {len(judge_stats['tool_calls'])} 次工具调用 | "
            f"{judge_stats['total_tokens']} tokens | {judge_stats['rounds']} 轮 | "
            f"done={judge_stats['done']}"
        )

        wfm.solidify(wf_id)
        wf = wfm.get_workflow(wf_id)
        kept_count = sum(1 for s in wf.steps if not s.is_pruned) if wf else 0

        results = wfm.retrieve(spec["prompt"], top_k=1)
        if not results:
            print("  [注入] 检索无结果，跳过注入")
            return None, judge_stats
        top = results[0]
        injection = format_context_compact(top)
        print(
            f"  [注入] {wf_id} | RAW {raw_count} → SOLIDIFIED {kept_count} "
            f"(LLM剪枝) | 注入 {len(injection)} chars"
        )
        return injection, judge_stats
    finally:
        wfm.close()


# ── 主流程 ────────────────────────────────────────────


def run_mode(task_key: str, mode: str, runs: int, db_dir: Path) -> dict:
    results: dict = {"task": task_key, "mode": mode, "runs": []}
    embedding = None

    if mode == "baseline":
        for i in range(1, runs + 1):
            stats, _cp = run_once(task_key, f"thread_{i}")
            print(
                f"  [baseline] run {i}/{runs} | calls={stats['tool_call_count']} "
                f"tokens={stats['total_tokens']} passed={stats['passed']}"
            )
            results["runs"].append(stats)
    else:  # inject（固定 LLM 剪枝）
        db_path = db_dir / f"{task_key}_inject.db"
        if db_path.exists():
            db_path.unlink()
        # run 1: 无注入，产出 workflow
        stats1, cp1 = run_once(task_key, "thread_1")
        print(
            f"  [inject] run 1 (raw) | calls={stats1['tool_call_count']} "
            f"tokens={stats1['total_tokens']} passed={stats1['passed']}"
        )
        results["runs"].append({**stats1, "phase": "raw"})

        embedding = M3EEmbedding()
        from langchain_openai import ChatOpenAI

        judge_llm = ChatOpenAI(
            model=os.environ.get("DEEPSEEK_JUDGE_MODEL", "deepseek-chat"),
            api_key=_resolve_api_key(),
            base_url="https://api.deepseek.com",
            temperature=0,
            max_retries=2,
            timeout=120,
        )
        injection, judge_stats = build_injection(
            task_key, cp1, db_path, embedding, judge_llm
        )
        results["injection"] = injection
        results["judge"] = judge_stats

        # run 2..N: 注入后执行
        for i in range(2, runs + 1):
            stats, _cp = run_once(task_key, f"thread_{i}", system_extra=injection or "")
            print(
                f"  [inject] run {i}/{runs} (injected) | "
                f"calls={stats['tool_call_count']} "
                f"tokens={stats['total_tokens']} passed={stats['passed']}"
            )
            results["runs"].append({**stats, "phase": "injected"})

    if embedding is not None:
        del embedding
    return results


def summarize(results: dict) -> dict:
    """对 runs 求均值/汇总。"""
    runs = results["runs"]
    if results["mode"] == "baseline":
        key = [r for r in runs]
    else:
        key = [r for r in runs if r.get("phase") == "injected"]
        if not key:
            key = runs[1:]
    n = len(key)
    if n == 0:
        return {}
    avg = lambda f: round(sum(r[f] for r in key) / n, 1)  # noqa: E731
    return {
        "runs_used": n,
        "avg_tool_calls": avg("tool_call_count"),
        "avg_total_tokens": avg("total_tokens"),
        "avg_prompt_tokens": avg("prompt_tokens"),
        "avg_completion_tokens": avg("completion_tokens"),
        "avg_steps": avg("steps"),
        "success_rate": round(sum(1 for r in key if r["passed"]) / n * 100, 1),
        "avg_elapsed_s": avg("elapsed_s"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Workflow 省 Token 端到端评测（LLM 剪枝）"
    )
    parser.add_argument("--task", choices=list(TASKS) + ["all"], default="T1")
    parser.add_argument(
        "--mode", choices=["baseline", "inject", "both"], default="both"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="每个模式的有效 run 数（inject 模式额外 +1 次 raw run）",
    )
    parser.add_argument(
        "--db-dir", default=str(Path(__file__).resolve().parent / "tmp")
    )
    args = parser.parse_args()

    if not _resolve_api_key():
        print("错误: DEEPSEEK_API_KEY 未找到（环境变量 + 注册表均无）")
        sys.exit(1)

    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = list(TASKS) if args.task == "all" else [args.task]
    modes = ["baseline", "inject"] if args.mode == "both" else [args.mode]

    summary_rows = []
    for task_key in tasks:
        for mode in modes:
            print(f"\n=== {task_key} [{mode}] runs={args.runs} ===")
            results = run_mode(task_key, mode, args.runs, db_dir)
            out = RESULTS_DIR / f"{task_key}_{mode}.json"
            out.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            s = summarize(results)
            s["mode_tag"] = mode
            summary_rows.append({"task": task_key, "mode": mode, **s})
            print(f"  汇总: {s}")

    # 对比表
    print("\n" + "=" * 100)
    print(
        f"{'任务':<5}{'模式':<10}{'run数':<6}{'调用数':<8}{'总token':<12}"
        f"{'prompt':<12}{'completion':<12}{'步骤':<6}{'成功率':<9}{'耗时s':<8}"
    )
    print("-" * 100)
    for r in summary_rows:
        print(
            f"{r['task']:<5}{r['mode']:<10}{r.get('runs_used', 0):<6}"
            f"{r.get('avg_tool_calls', '-'):<8}{r.get('avg_total_tokens', '-'):<12}"
            f"{r.get('avg_prompt_tokens', '-'):<12}"
            f"{r.get('avg_completion_tokens', '-'):<12}"
            f"{r.get('avg_steps', '-'):<6}{r.get('success_rate', '-'):<9}"
            f"{r.get('avg_elapsed_s', '-'):<8}"
        )
    print("=" * 100)
    print(f"结果文件已保存到 {RESULTS_DIR}")


if __name__ == "__main__":
    main()
