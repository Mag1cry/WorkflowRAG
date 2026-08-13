# EvalReport — Workflow 复用省 Token 端到端评测结果

_生成时间: 2026-08-14T03:36:27_

## 1. 汇总对比

| 任务 | 模式 | run数 | 工具调用 | 总token | prompt | completion | 步骤 | 成功率 | 耗时s |
|---|---|---|---|---|---|---|---|---|---|
| T1 修复 Python 导入错误 | baseline | 3 | 13.0 | 13860.3 | 12828.3 | 1032.0 | 10.7 | 100.0% | 12.2 |
| T1 修复 Python 导入错误 | inject_llm | 2 | 11.0 | 13537.5 | 12539.0 | 998.5 | 10.0 | 100.0% | 11.8 |
| T1 修复 Python 导入错误 | inject_rule | 2 | 12.5 | 16653.0 | 15606.0 | 1047.0 | 11.0 | 100.0% | 16.9 |
| T2 安装依赖并验证 | baseline | 3 | 6.3 | 8107.0 | 7540.0 | 567.0 | 7.3 | 100.0% | 8.6 |
| T2 安装依赖并验证 | inject_llm | 2 | 3.0 | 4215.0 | 3873.0 | 342.0 | 4.0 | 100.0% | 4.9 |
| T3 批量重命名文件 | baseline | 3 | 7.0 | 8870.0 | 8097.3 | 772.7 | 6.7 | 100.0% | 9.4 |
| T3 批量重命名文件 | inject_llm | 2 | 5.0 | 7561.0 | 6938.0 | 623.0 | 6.0 | 100.0% | 7.2 |
| T4 运行测试并修复失败 | baseline | 3 | 4.0 | 4730.0 | 4262.0 | 468.0 | 4.0 | 100.0% | 6.1 |
| T4 运行测试并修复失败 | inject_llm | 2 | 4.0 | 4889.0 | 4409.0 | 480.0 | 4.0 | 100.0% | 5.7 |

## 2. 净收益分析（注入模式 vs 基线）

| 任务 | 基线token | 注入后token | 节省 | 节省率 | 基线调用 | 注入后调用 | 调用节省 | 注入文本长度 | LLM剪枝成本 |
|---|---|---|---|---|---|---|---|---|---|
| T1 [inject_llm] | 13860.3 | 13537.5 | 322.8 | 2.3% | 13.0 | 11.0 | 2.0 | 175 chars | 16773 |
| T1 [inject_rule] | 13860.3 | 16653.0 | -2792.7 | -20.1% | 13.0 | 12.5 | 0.5 | 403 chars | 0 |
| T2 [inject_llm] | 8107.0 | 4215.0 | 3892.0 | 48.0% | 6.3 | 3.0 | 3.3 | 168 chars | 15084 |
| T3 [inject_llm] | 8870.0 | 7561.0 | 1309.0 | 14.8% | 7.0 | 5.0 | 2.0 | 191 chars | 17962 |
| T4 [inject_llm] | 4730.0 | 4889.0 | -159.0 | -3.4% | 4.0 | 4.0 | 0.0 | 122 chars | 14725 |

## 3. 注入内容样本

### T1 [inject_llm] 修复 Python 导入错误

```
[Workflow: T1] write_file({'path': 'app/utils.py', 'content': "def...) → write_file({'content': 'from utils import helper\n\...) → bash({'command': 'python app/main.py 2>&1'})
```

### T1 [inject_rule] 修复 Python 导入错误

```
[Workflow: T1] bash({'command': 'cd /sandbox && python app/m...) → list_dir({'path': '.'}) → list_dir({'path': 'app'}) → write_file({'path': 'app/utils.py', 'content': "def...) → bash({'command': 'cd /sandbox && python app/m...) → bash({'command': 'cd && dir'}) → bash({'command': 'python app/main.py'}) → write_file({'content': 'from utils import helper\n\...) → bash({'command': 'python app/main.py'})
```

### T2 [inject_llm] 安装依赖并验证

```
[Workflow: T2] bash({'command': 'python -m venv --without-pi...) → bash({'command': '.venv\\Scripts\\python.exe ...) → bash({'command': '.venv\\Scripts\\python.exe ...)
```

### T3 [inject_llm] 批量重命名文件

```
[Workflow: T3] write_file({'path': 'rename_images.py', 'content': ...) → bash({'command': 'python rename_images.py'}) → bash({'command': 'dir /b'}) → bash({'command': 'del rename_images.py'})
```

### T4 [inject_llm] 运行测试并修复失败

```
[Workflow: T4] write_file({'path': 'math_utils.py', 'content': "de...) → bash({'command': 'python -m pytest test_math....)
```


## 4. LLM 剪枝成本明细

| 任务 | 模式 | 工具调用 | input tokens | output tokens | 总tokens | 轮数 | done | 报告摘要 |
|---|---|---|---|---|---|---|---|
| T1 [inject_llm] | inject_llm | 6 | 15385 | 1388 | 16773 | 5 | True | 审查完成，共 12 步，剪枝 9 步，保留 3 步。  **剪枝的步骤（9 步）：** - index 0 (bash `cd /sandbox && pyth |
| T2 [inject_llm] | inject_llm | 7 | 13936 | 1148 | 15084 | 5 | True | ## 审查报告  ### 剪枝的步骤（4 个）  1. **index=0 (list_dir)** — 探索性调用，仅列出目录内容（demo.py, test |
| T3 [inject_llm] | inject_llm | 7 | 16660 | 1302 | 17962 | 5 | True | ## 审查报告  ### 剪枝情况（6 步剪枝，4 步保留）  **剪枝的步骤：** - **index=0** (`dir /b`)：探索性目录查看命令，对任 |
| T4 [inject_llm] | inject_llm | 6 | 13475 | 1250 | 14725 | 5 | True | 审查完成，共 4 步，剪枝 2 步，保留 2 步。  **剪枝的步骤：** - index=0 (read_file math_utils.py)：探索性读取， |

## 5. 判定

### 结论：方向成立（有条件继续）— "选择性复用"策略

**核心发现（用数据说话）：**

1. **Workflow 复用省 token 在"固定流程类任务"上显著成立**
   - T2（创建 venv 并验证）：token **-48%**、工具调用 -52%、成功率 100%
   - T3（批量重命名）：token -14.8%、调用 -28.6%、成功率 100%
   - 原因：workflow 提供了精确命令序列，agent 无需思考探索，直接照做

2. **在"简单/模型已高效"任务上不划算**
   - T1（修导入错误）：仅 -2.3%（deepseek-chat 本身就能高效完成）
   - T4（修测试）：**+3.4%**（4 步任务无探索可省，注入纯属负担）
   - 推论：需要**任务复杂度/检索阈值过滤**，简单任务不注入

3. **LLM 剪枝质量碾压规则剪枝**
   - T1 同任务对比：LLM 剪枝注入 13537 tokens vs 规则剪枝 16653 tokens（差 23%）
   - 注入文本 175 chars vs 403 chars；LLM 版无 `cd /sandbox` 有害信息
   - LLM 能识别规则剪枝做不到的：bash 参数内探索、路径失效失败、重复失败

4. **LLM 剪枝成本（一次性 ~15-18k tokens）需要摊薄**
   - T2：每次省 3892 tokens → **复用 4 次回本**
   - T3：每次省 1309 → 复用 12 次回本
   - T1：每次省 323 → 复用 52 次（几乎不划算）
   - 成本高是 LLM 剪枝的短板，但剪枝是低频一次性操作（同任务只剪一次）

### 建议

| 方向 | 建议 |
|---|---|
| 产品定位 | "固定流程任务的跨会话步骤复用"，不做通用记忆 |
| 选择性注入 | 检索命中 + 任务复杂度过滤（简单任务跳过注入） |
| 剪枝策略 | LLM 剪枝为主；可加"规则预剪 + LLM 复核"混合模式降低成本 |
| 后续实验 | 复用次数递增实验（第 2/5/10 次复用的边际收益） |
