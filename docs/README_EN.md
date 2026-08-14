# WorkflowManager

**Extract, prune, and solidify LLM agent workflows from LangGraph checkpoints.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/langgraph-%E2%9C%93-green)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

**Core idea: `context = workflow`**

Instead of treating agent execution history as raw message dumps, WorkflowManager extracts structured **toolcall/bashcall step sequences** from LangGraph checkpoints, prunes out noise, and solidifies them into reusable workflows. When a similar task arises, the solidified workflow is injected as context — giving the agent a proven recipe to follow.

---

## Why Workflow?

| Aspect | Traditional Skill | Workflow |
| -------- | ------------------- | ---------- |
| **Nature** | Abstract description by the agent | Real, executable step sequence |
| **Granularity** | High-level summary | Step-by-step tool/bash calls |
| **Reference value** | Conceptual guide | Directly reusable pattern |
| **Example** | "Fix import errors by checking paths" | `read_file → grep → edit_file → python` |

An agent seeing a workflow knows **exactly what steps to take** — not just a vague idea.

---

## How It Works

```mermaid
flowchart TB
    Agent --> WM[WorkflowManager]
    WM --> Extract[extract_workflow<br/>from Checkpoints]
    WM --> Judge[WorkflowJudge<br/>LLM prunes via tools]
    WM --> Store[WorkflowStore<br/>SQLite]
    WM --> Index[WorkflowIndex<br/>FAISS]
    WM --> Embed[Embedding<br/>M3E]
    Store --> DB[(workflows + steps)]
    Index --> Vec[(workflow_id → vector)]
```

### Lifecycle

```
1. Agent completes a task (LangGraph Thread)
2. extract_workflow(thread_id)  →  parse messages into Step[], store as RAW
3. judge.judge(workflow_id)     →  LLM review agent prunes steps via function calls
4. solidify(workflow_id)        →  generate description → index via FAISS → status = SOLIDIFIED
5. retrieve(query)              →  embed query → FAISS search → return SOLIDIFIED workflow with steps
```

---

## Quick Start

```bash
pip install -e .
python cli.py demo                  # ★ Full end-to-end demo: real agent task → extract
                                    #   → LLM pruning → solidify → retrieve → inject reuse
                                    #   → compare token savings (needs DEEPSEEK_API_KEY)
python cli.py demo --offline        # Offline demo with fake data (no API key needed)
python demo.py                      # Same as `cli.py demo` (run the sample directly)
python cli.py review <thread_id>    # One-shot review: extract → show → solidify
python cli.py case                  # Built-in case: pruning effect comparison
python cli.py --list-tools          # Print function-call schemas for the review LLM
```

Full demo output preview (`python cli.py demo`):

```
1. First run — agent completes a task without reference  → 9 calls / 8992 tokens
2. Extract RAW workflow (9 steps incl. exploration noise)
3. WorkflowJudge LLM pruning → 3 steps pruned (navigation + failed verify) + review report
4. Solidify + retrieval hit
5. Inject pruned workflow → second run                      → 8 calls
6. Comparison: calls/tokens/success + pruning-cost payback analysis
```

**Dependencies**: `torch` `transformers` `faiss-cpu` `scikit-learn` `numpy` `langgraph` `langchain-core`

**Model**: Download [moka-ai/m3e-base](https://huggingface.co/moka-ai/m3e-base) (768-dim Chinese embedding) to `models/` directory.

---

## API

```python
from context_manager import WorkflowManager
from context_manager.workflow.judge import WorkflowJudge

wfm = WorkflowManager()

# 1. Extract workflow from a completed LangGraph thread
wf_id = wfm.extract_workflow("some_thread_id")

# 2. LLM pruning — review agent operates steps via function-call tools
judge = WorkflowJudge(wfm, llm)          # llm: function-calling ChatOpenAI
result = judge.judge(wf_id)              # returns stats + review report

# 3. Solidify (description → vector index → SOLIDIFIED)
wfm.solidify(wf_id)

# 4. Retrieve relevant workflows (pruned steps filtered out)
results = wfm.retrieve("how to fix import error", top_k=3)

# 5. Compact context injection (token-efficient)
from context_manager.workflow.injector import format_context_compact
context = format_context_compact(results[0])

# 6. Review tools (used by the LLM via function calls)
wfm.get_workflow(wf_id)
wfm.list_workflows()
wfm.prune_step("step_id", True)
wfm.update_workflow_description(wf_id, "new desc")

wfm.close()
```

For testing with in-memory backends:

```python
from context_manager import create_memory_manager

wfm = create_memory_manager()
```

---

## Core Concepts

| Concept | Description |
| --------- | ------------- |
| **Workflow** | Ordered sequence of toolcall/bashcall steps (replaces Episode) |
| **Step** | A single tool call or bash command execution |
| **RAW** | Raw step sequence extracted from checkpoints, including noise |
| **SOLIDIFIED** | Clean, pruned step sequence ready for context injection |

### Step Schema

```python
Step = {
  step_id: str,
  type: "toolcall" | "bashcall",
  name: str,              # e.g. "read_file", "edit_file", "python"
  arguments: str,         # full input parameters
  result: str,            # full output
  status: "success" | "failure",
  duration_ms: int,
  timestamp: datetime,
  step_index: int,
  is_pruned: bool,        # marked by WorkflowJudge (LLM)
  error_message: str | null,
}
```

---

## LLM Pruning (WorkflowJudge)

Pruning is done by an LLM review agent (`workflow/judge.py`) that operates the workflow **only through function-call tools** (preventing hallucinated edits). The tool set (17 tools) is defined once in `workflow/tools.py` (`ReviewToolsMixin`) and auto-generated as OpenAI-compatible schemas via `get_tool_schemas()`:

| Category | Tools |
| ---------- | ------- |
| View | `get_workflow` `list_workflows` `get_step` `list_steps` `get_steps` `visualize` |
| Edit | `prune_step` `batch_prune` `update_step` `add_step` `remove_step` `reorder_steps` |
| Metadata | `update_workflow_name` `update_workflow_description` |
| Lifecycle | `solidify` `delete_workflow` `review_summary` |
| Finish | `judge_done(report)` (WorkflowJudge built-in) |

Review criteria: exploratory calls (incl. inside `bash` args), failed-and-bypassed steps, overwritten results, repeated verifications are pruned; effective writes and successful verifications are kept.

---

## Project Structure

```bash
013ContextManager/
├── cli.py                         # CLI entry point (demo / review / case / list-tools)
├── demo.py                        # ★ Full end-to-end sample (real LLM / --offline)
├── context_manager/
│   ├── __init__.py                # Exports WorkflowManager, WorkflowJudge
│   ├── config.py                  # Settings dataclass
│   ├── models.py                  # Workflow + Step dataclasses
│   ├── persistence/
│   │   ├── store.py               # WorkflowStoreBase + SQLite + InMemory
│   │   ├── index.py               # WorkflowIndexBase + FAISS + InMemory
│   │   └── embedding.py           # M3EEmbedding
│   └── workflow/
│       ├── manager.py             # WorkflowManager (lifecycle: extract/retrieve/inject)
│       ├── tools.py               # ReviewToolsMixin (17 tools + auto-generated schemas)
│       ├── judge.py               # WorkflowJudge (LLM pruning agent, reuses tools)
│       ├── injector.py            # Context injection formatting
│       └── visualizer.py          # ANSI visualization
├── eval/                          # End-to-end token-saving evaluation
│   ├── agent.py                   # Demo agent (bash/read/write/list + stats)
│   ├── tasks.py                   # 4 benchmark tasks
│   ├── runner.py                  # Eval runner (baseline vs LLM-pruned injection)
│   └── report.py                  # Results → Markdown report
└── pyproject.toml                 # Packaging & dependencies
```

---

## Evaluation (eval/)

Real end-to-end benchmark (deepseek-chat + LangGraph ReAct + 4 tasks) measuring token savings:

```bash
python eval/runner.py --task all --mode both --runs 3   # baseline vs LLM-pruned injection
python eval/report.py                                    # generate Markdown report
```

Conclusion ([docs/EvalReport.md](./EvalReport.md)): reuse yields **-48% tokens / -52% calls** on fixed-procedure tasks (e.g. venv setup), little gain on trivial tasks; LLM pruning beats rule-based pruning (13.5k vs 16.7k tokens on the same task) but costs ~15-18k tokens once per task, paid back after ~4 reuses.

---

## Running Tests

```bash
pytest tests/ -v    # 36 tests
```

---

## Database Schema

```sql
CREATE TABLE workflows (
    workflow_id     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT DEFAULT '',
    status          TEXT DEFAULT 'RAW',
    source_thread_id TEXT,
    tags            TEXT DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE steps (
    step_id         TEXT PRIMARY KEY,
    workflow_id     TEXT NOT NULL,
    step_index      INTEGER NOT NULL,
    type            TEXT NOT NULL,
    name            TEXT NOT NULL,
    arguments       TEXT DEFAULT '',
    result          TEXT DEFAULT '',
    status          TEXT DEFAULT 'success',
    duration_ms     INTEGER DEFAULT 0,
    error_message   TEXT DEFAULT '',
    is_pruned       INTEGER DEFAULT 0,
    timestamp       TEXT NOT NULL,
    FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id)
);
```

---

## Out of Scope (v1)

- Multi-agent shared workflow library
- Workflow versioning
- Workflow merge/split
- Workflow visualization UI
- Auto tag generation (simple rules for now)

---

## Data Consistency

- **SQLite** is the source of truth for workflow metadata
- **LangGraph** is the source of truth for execution state
- **FAISS index** can be rebuilt from SQLite at any time
