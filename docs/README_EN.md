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
|--------|-------------------|----------|
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
    WM --> Pruner[Pruner<br/>prune useless steps]
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
3. solidify(workflow_id)        →  prune noise → generate description → index via FAISS → status = SOLIDIFIED
4. retrieve(query)              →  embed query → FAISS search → return SOLIDIFIED workflow with steps
```

---

## Quick Start

```bash
pip install -r requirements.txt
python cli.py demo
```

**Dependencies**: `torch` `transformers` `faiss-cpu` `scikit-learn` `numpy` `langgraph` `langchain-core`

**Model**: Download [moka-ai/m3e-base](https://huggingface.co/moka-ai/m3e-base) (768-dim Chinese embedding) to `models/` directory.

---

## API

```python
from context_manager import WorkflowManager

wfm = WorkflowManager()

# 1. Extract workflow from a completed LangGraph thread
wf_id = wfm.extract_workflow("some_thread_id")

# 2. Solidify (prune + index)
wfm.solidify(wf_id)

# 3. Retrieve relevant workflows (returns complete step sequences)
results = wfm.retrieve("how to fix import error", top_k=3)
# → [{"workflow_id", "name", "description", "similarity", "steps": [...]}]

# 4. Management
wfm.list_workflows()
wfm.get_workflow(wf_id)
wfm.delete_workflow(wf_id)

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
|---------|-------------|
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
  is_pruned: bool,        # marked by Pruner
  error_message: str | null,
}
```

---

## Pruning Strategies

Pruning is the core transformation — converting RAW workflows into high-quality SOLIDIFIED ones.

| Strategy | Heuristic | Example |
|----------|-----------|---------|
| **Result overwritten** | A later write to the same target supersedes earlier ones | `edit_file(a.py)` then `edit_file(a.py)` again — first is pruned |
| **Failed & irrelevant** | A step failed but the task was completed via another path | `pip install` fails, `conda install` succeeds |
| **Exploratory call** | Read-only exploration/debugging that doesn't contribute | `ls`, `cat`, `read_file` of unrelated files |
| **LLM judgement** | LLM evaluates step contribution | Interface reserved, not implemented in v1 |

---

## Project Structure

```bash
013ContextManager/
├── cli.py                         # CLI entry point
├── context_manager/
│   ├── __init__.py                # Exports WorkflowManager
│   ├── config.py                  # Settings dataclass
│   ├── embedding.py               # M3EEmbedding
│   ├── manager.py                 # WorkflowManager
│   ├── pruner.py                  # Pruner engine
│   ├── storage/
│   │   ├── base.py                # WorkflowStoreBase ABC
│   │   ├── in_memory.py           # MemoryWorkflowStore
│   │   └── sqlite.py              # SQLiteWorkflowStore
│   └── index/
│       ├── base.py                # WorkflowIndexBase ABC
│       ├── in_memory.py           # MemoryWorkflowIndex
│       └── faiss_index.py         # FaissWorkflowIndex
├── tests/                         # 22 tests
├── docs/
│   ├── DESIGN.md                  # Full design document
│   └── README_EN.md               # This file
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v    # 22 tests
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
- LLM-judged pruning (rule-based for now)

---

## Data Consistency

- **SQLite** is the source of truth for workflow metadata
- **LangGraph** is the source of truth for execution state
- **FAISS index** can be rebuilt from SQLite at any time