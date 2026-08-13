# WorkflowManager

基于 LangGraph 的 Workflow 提取与管理引擎。

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/langgraph-%E2%9C%93-green)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

[English](./docs/README_EN.md) · **简体中文**

**核心思想：`context = workflow`**

将 Agent 执行过程中的 toolcall/bashcall 步骤序列提取为结构化 Workflow，经过剪枝优化后固化，作为未来任务的优秀案例注入上下文。

## 与 Skill 的区别

Skill 是 Agent 自己总结的抽象描述，而 Workflow 是**真实的、可执行的步骤序列**——Agent 看到"上次解决这个问题用了这些步骤"，可以直接复用。

## 安装

```bash
pip install -r requirements.txt
```

依赖：`torch` `transformers` `faiss-cpu` `scikit-learn` `numpy` `langgraph` `langchain-core`
模型：需本地放置 [moka-ai/m3e-base](https://huggingface.co/moka-ai/m3e-base)（768维中文嵌入）

## 快速开始

```bash
python cli.py demo
python cli.py review <thread_id>   # 一键审查
```

## 核心概念

| 概念 | 说明 |
| --- | --- |
| **Workflow** | Agent 完成某项任务的有序 toolcall/bashcall 步骤序列 |
| **Step** | 一次工具调用或命令执行，Workflow 的最小组成单元 |
| **RAW** | 任务完成后从 Checkpoints 提取的原始步骤序列 |
| **SOLIDIFIED** | 经过 LLM 剪枝后的干净步骤序列，适合作为上下文注入 |

## 架构

```bash
context_manager/
├── models.py                    # Workflow + Step 数据类（顶层共享）
├── persistence/                 # 持久化层
│   ├── store.py                 # WorkflowStore (SQLite/InMemory)
│   ├── index.py                 # WorkflowIndex (FAISS/InMemory)
│   └── embedding.py             # M3EEmbedding
└── workflow/                    # Workflow 管理（RAG API）
    ├── manager.py               # WorkflowManager（提取、固化、检索、编辑）
    ├── judge.py                 # WorkflowJudge（LLM 剪枝审查 Agent）
    └── injector.py              # 上下文注入（格式化 Workflow → Agent 上下文）
```

## API

```python
from context_manager import WorkflowManager

wfm = WorkflowManager()

# 1. 提取 Workflow（任务完成后，从 LangGraph Thread 事后提取）
wf_id = wfm.extract_workflow("some_thread_id")

# 2. 固化（规则剪枝 + 索引）
wfm.solidify(wf_id)

# 3. 检索（返回 Workflow 对象）
results = wfm.retrieve("如何修复导入错误", top_k=3)
# → [Workflow(workflow_id, name, description, steps=[Step, ...])]

# 4. 上下文注入
context = wfm.format_context(results[0])

# 5. 审查 LLM 工具
wfm.get_workflow(wf_id)
wfm.list_workflows()
wfm.prune_step("step_id", True)
wfm.update_workflow_description(wf_id, "new desc")

wfm.close()
```

依赖注入（纯内存，测试用）：

```python
from context_manager import create_memory_manager
wfm = create_memory_manager()
```

## 项目结构

```bash
013ContextManager/
├── cli.py                         # 统一入口（demo / review）
├── context_manager/
│   ├── __init__.py                # 导出 WorkflowManager、WorkflowJudge、Workflow、Step
│   ├── config.py                  # Settings
│   ├── models.py                  # Workflow + Step 数据类
│   ├── persistence/               # 持久化层
│   │   ├── __init__.py
│   │   ├── store.py               # WorkflowStoreBase + SQLite + InMemory
│   │   ├── index.py               # WorkflowIndexBase + FAISS + InMemory
│   │   └── embedding.py           # M3EEmbedding
│   └── workflow/                  # Workflow 管理
│       ├── __init__.py
│       ├── manager.py             # WorkflowManager
│       ├── judge.py               # WorkflowJudge（LLM 剪枝审查 Agent）
│       └── injector.py            # 上下文注入
├── eval/                          # 端到端评测（agent + 4 任务 + runner）
├── tests/                         # 测试
├── docs/
│   ├── Design.md                  # 完整设计文档
│   └── EvalReport.md              # 省 Token 端到端评测报告
└── .gitignore
```

## 运行测试

```bash
pytest tests/ -v   # 28 个测试
```

## 审查流程

```bash
python cli.py review <langgraph_thread_id>
```

自动化流程：

1. 从 LangGraph Thread 提取 RAW Workflow
2. 展示步骤摘要
3. 执行 LLM 剪枝（WorkflowJudge 通过工具审查：探索性调用、失败尝试、结果被覆盖等）
4. 生成 SOLIDIFIED Workflow
5. 写入 FAISS 索引

## LLM 剪枝（WorkflowJudge）

剪枝由 LLM 审查 Agent 完成，它只能通过 **function call 工具** 操作 Workflow，不能直接输出修改内容（防止幻觉）：

| 类别 | 工具 |
| --- | --- |
| 查看 | `review_summary` `list_steps` `get_steps` `get_step` `visualize` |
| 操作 | `prune_step` `batch_prune` `update_step` `add_step` `remove_step` `reorder_steps` |
| 元数据 | `update_workflow_description` |
| 结束 | `judge_done(report)`（提交审查报告） |

```python
from context_manager.workflow.judge import WorkflowJudge

judge = WorkflowJudge(wfm, llm)   # llm: 支持 function calling 的 ChatOpenAI
result = judge.judge(wf_id)       # 剪枝标记写入 store，返回统计与审查报告
```

LLM 审查标准：

| 标准 | 说明 |
| --- | --- |
| **探索性调用** | ls/dir/cat/read_file 读无关文件、pwd 等导航命令（含藏在 bash 参数内的） |
| **出错但无关** | 步骤执行失败，但后续通过其他方式解决了问题 |
| **结果被覆盖** | 同一文件/目标被多次修改，只保留最后一次有效修改 |
| **重复验证** | 同一验证命令多次成功运行，只保留最后一次 |
| **保留** | 有效的写操作、成功且关键的验证运行 |
