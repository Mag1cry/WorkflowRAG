# 端到端验证计划：Workflow 复用到底省不省 Token？

> 状态：`待确认`（2026-02 定稿）
> 背景：项目现状检视后重新定位——原始动机是"复用 Agent 的 toolcall/bashcall 步骤序列以节约 token"，
> 而非泛化的长期记忆。本计划用 LangGraph 自建 demo agent 做端到端评测，用数据回答"值不值"。

## 0. 核心问题与判定标准

**核心问题**：将上次任务的 SOLIDIFIED Workflow 注入上下文，能否降低同类任务的 token 消耗与工具调用次数，且不降低成功率？

| 结论档位 | 判定标准 | 后续动作 |
|---|---|---|
| ✅ 显著省 | 净收益（省下的探索 token − 注入成本）≥ 30% 且成功率不降 | 继续投入打磨 |
| ⚠️ 持平 | 收益在 ±30% 内 | 调优注入格式 / 剪枝 / 检索阈值后再定 |
| ❌ 反而亏 | 注入成本 > 省下的探索成本 | 用数据说话，项目转型或收尾 |

## 1. 关键决策（已确认）

| 决策项 | 选择 | 备注 |
|---|---|---|
| 数据源 | **LangGraph 自建 demo agent** | `extract_workflow` 走原生 Checkpointer 路径，零 adapter |
| LLM 后端 | **DeepSeek API**（OpenAI 兼容端点） | 需要 API key（待配置） |
| Python 环境 | **conda `agent` 环境**（Python 3.11.13） | 依赖已全部就绪，无需新建 |
| 任务范围 | 临时 sandbox 目录（`%TEMP%\cm_eval_*`） | 真实执行但不碰真实代码 |

### 环境事实（已核实）

- `C:\003Codes\PythonEnvs\Miniconda3\envs\agent\python.exe`：Python 3.11.13
- langgraph 0.6.5 / langchain-core 0.3.74 / langchain-openai 0.3.32 / openai 1.101.0
- torch 2.5.1+cu121 / transformers 4.55.3 / faiss-cpu 1.14.3 / numpy 2.1.2 / sklearn 1.9.0 / pytest 8.4.1 / zstandard 0.24.0
- M3E 模型已存在：`C:\003Codes\models\m3e-base`

## 2. 阶段 A：环境复位

1. 用 agent 环境跑通 `pytest tests/ -v`（28 个测试）
2. 修复 `update_workflow_name()` 调用 `update_description()` 的 bug（manager.py）
3. 提交当前未提交的 14 个文件修改 + `visualizer.py`
4. 验证 `python cli.py demo` / `case` 可跑

## 3. 阶段 B：LangGraph Demo Agent

**工具集**（刻意贴近真实 Agent，检验规则剪枝对"不干净工具名"的鲁棒性）：

| 工具 | 实现 | 备注 |
|---|---|---|
| `bash` | subprocess 真执行（限定 cwd=sandbox） | 真实命令，含失败路径 |
| `read_file` | 读文件 | 结果截断 |
| `write_file` | 写文件 | |
| `list_dir` | 列目录 | |

**任务集**（4 个日常重复型任务，预埋问题，全部在 sandbox 目录内）：

| 任务 | 内容 | 预期探索量 |
|---|---|---|
| T1 | 修复 Python 导入错误 | 中 |
| T2 | 虚拟环境内安装依赖并验证 | 中高 |
| T3 | 批量重命名/整理文件 | 低 |
| T4 | 运行测试并修复失败 | 高 |

- Agent 结构：ReAct（`create_react_agent`）+ MemorySaver checkpointer + system prompt 可注入参考 workflow
- 每次运行记录：toolcall 列表（名称/参数/结果/成败/耗时）、token usage（prompt/completion）

## 4. 阶段 C：评测脚本 `eval_runner.py`

- **基线模式**：无参考跑任务 N 次（N≥3，消除随机性）→ 统计 toolcall 数、token、成功率
- **注入模式**：`extract_workflow(thread_id)` → `solidify()` → `retrieve(query)` → `format_context_compact()` 注入 system prompt → 再跑
- 对比指标：
  - toolcall 次数（Δ）
  - 总 token（Δ）
  - **净收益 = 省下 token − 注入 token**
  - 成功率（必须不降）
  - 剪枝率（RAW vs SOLIDIFIED 步骤数）
- 输出：对比报告（表格 + 结论判定）

## 5. 阶段 D：调优与结论

1. 根据 C 的结果调整：注入格式（compact vs 详细）、检索阈值、剪枝规则
2. 可选对比实验：**失败路径记忆（坑位地图）**——注入"不要试 X"一行 vs 完整流程注入
3. 产出 `docs/EvalReport.md`：给出"继续 / 转型 / 收尾"的明确结论

## 6. 风险与预案

| 风险 | 预案 |
|---|---|
| 规则剪枝对真实工具名失效（如 `pwsh` 不在探索性关键词表） | 阶段 B 起就记录剪枝率，D 阶段补关键词/LLM 评判 |
| DeepSeek API 无 key 或配额不足 | 阶段 A 前必须解决；备选：本地 4bit 量化（慢） |
| 评测随机性大 | N≥3 取均值；同任务同 seed 初始化 sandbox |
| demo agent 真执行命令破坏环境 | 所有任务 cwd 限定在 `%TEMP%\cm_eval_*`，任务结束即清理 |

## 7. 工作量预估

A（半天）→ B（1 天）→ C（1 天）→ D（半天），大部分由 AI 代理执行，人工负责提供 key 与复核。
