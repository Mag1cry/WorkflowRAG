"""report.py — 从 eval/results/*.json 生成 Markdown 评测报告。

用法:
    python eval/report.py            # 汇总全部结果 → eval/tmp/EvalReport.md
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent / "tmp" / "EvalReport.md"

TASKS_CN = {
    "T1": "修复 Python 导入错误",
    "T2": "安装依赖并验证",
    "T3": "批量重命名文件",
    "T4": "运行测试并修复失败",
}


def load() -> dict:
    """解析 eval/results/*.json → {task: {mode_tag: data}}。

    文件名规则: <TASK>_<mode>[_[pruner]].json
    e.g. T1_baseline.json, T1_inject_rule.json, T1_inject_llm.json
    """
    out = {}
    for p in sorted(RESULTS_DIR.glob("*_*.json")):
        parts = p.stem.split("_")
        task = parts[0]
        mode = parts[1] if len(parts) > 1 else parts[0]
        pruner = parts[2] if len(parts) > 2 else ""
        tag = mode if not pruner else f"{mode}_{pruner}"
        out.setdefault(task, {})[tag] = json.loads(p.read_text(encoding="utf-8"))
    return out


def avg(items, key):
    vals = [r[key] for r in items if key in r]
    return round(sum(vals) / len(vals), 1) if vals else None


def collect(task_data: dict, mode_tag: str) -> list:
    runs = task_data.get(mode_tag, {}).get("runs", [])
    if mode_tag.startswith("inject"):
        runs = [r for r in runs if r.get("phase") == "injected"]
        if not runs and len(task_data.get(mode_tag, {}).get("runs", [])) > 1:
            runs = task_data[mode_tag]["runs"][1:]
    return runs


def render(data: dict) -> str:
    now = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [
        "# EvalReport — Workflow 复用省 Token 端到端评测结果",
        "",
        f"_生成时间: {now}_",
        "",
        "## 1. 汇总对比",
        "",
        "| 任务 | 模式 | run数 | 工具调用 | 总token | prompt | completion | "
        "步骤 | 成功率 | 耗时s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for task in ("T1", "T2", "T3", "T4"):
        td = data.get(task, {})
        for mode_tag in sorted(td):
            runs = collect(td, mode_tag)
            if not runs:
                continue
            ok = sum(1 for r in runs if r.get("passed"))
            lines.append(
                f"| {task} {TASKS_CN.get(task, '')} | {mode_tag} | {len(runs)} | "
                f"{avg(runs, 'tool_call_count')} | {avg(runs, 'total_tokens')} | "
                f"{avg(runs, 'prompt_tokens')} | {avg(runs, 'completion_tokens')} | "
                f"{avg(runs, 'steps')} | {round(ok / len(runs) * 100, 1)}% | "
                f"{avg(runs, 'elapsed_s')} |"
            )
    lines += ["", "## 2. 净收益分析（注入模式 vs 基线）", ""]
    lines += [
        "| 任务 | 基线token | 注入后token | 节省 | 节省率 | 基线调用 | "
        "注入后调用 | 调用节省 | 注入文本长度 | LLM剪枝成本 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for task in ("T1", "T2", "T3", "T4"):
        td = data.get(task, {})
        base = collect(td, "baseline")
        for mode_tag in sorted(td):
            if not mode_tag.startswith("inject"):
                continue
            inj = collect(td, mode_tag)
            if not base or not inj:
                continue
            bt, it = avg(base, "total_tokens"), avg(inj, "total_tokens")
            bc, ic = avg(base, "tool_call_count"), avg(inj, "tool_call_count")
            inj_text = td.get(mode_tag, {}).get("injection") or ""
            judge = td.get(mode_tag, {}).get("judge") or {}
            judge_tokens = judge.get("total_tokens", 0)
            saving = round(bt - it, 1) if bt and it else None
            rate = round((bt - it) / bt * 100, 1) if bt and it else None
            lines.append(
                f"| {task} [{mode_tag}] | {bt} | {it} | {saving} | {rate}% | "
                f"{bc} | {ic} | "
                f"{round(bc - ic, 1) if bc and ic else None} | "
                f"{len(inj_text)} chars | {judge_tokens} |"
            )
    lines += ["", "## 3. 注入内容样本", ""]
    for task in ("T1", "T2", "T3", "T4"):
        for mode_tag in sorted(data.get(task, {})):
            inj = data[task][mode_tag].get("injection")
            if inj:
                lines += [
                    f"### {task} [{mode_tag}] {TASKS_CN.get(task, '')}",
                    "",
                    "```",
                    inj,
                    "```",
                    "",
                ]
    lines += ["", "## 4. LLM 剪枝成本明细", ""]
    lines += [
        "| 任务 | 模式 | 工具调用 | input tokens | output tokens | 总tokens | "
        "轮数 | done | 报告摘要 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for task in ("T1", "T2", "T3", "T4"):
        for mode_tag in sorted(data.get(task, {})):
            judge = data[task][mode_tag].get("judge")
            if judge:
                report = (judge.get("report") or "").replace("\n", " ")[:80]
                lines.append(
                    f"| {task} [{mode_tag}] | {mode_tag} | "
                    f"{len(judge.get('tool_calls', []))} | "
                    f"{judge.get('input_tokens', 0)} | "
                    f"{judge.get('output_tokens', 0)} | "
                    f"{judge.get('total_tokens', 0)} | "
                    f"{judge.get('rounds', 0)} | "
                    f"{judge.get('done', False)} | {report} |"
                )
    lines += [
        "",
        "## 5. 判定",
        "",
        "> 结论待填：净收益 ≥30% 且成功率不降 → 继续；±30% → 调优；"
        "负收益 → 转型/收尾。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    data = load()
    if not data:
        print("eval/results/ 为空，先运行 eval_runner.py")
        sys.exit(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(data), encoding="utf-8")
    print(f"报告已生成: {OUT}")


if __name__ == "__main__":
    main()
