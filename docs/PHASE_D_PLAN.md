# 阶段 D 方案：基于真实评测数据的剪枝/注入改进（草案）

> 依据：T1 真实 workflow 分析（14 步，规则剪枝只剪 5 个 read_file，
> 漏剪 list_dir ×2 + bash 内探索命令 ×1；注入文本含有害 `cd /sandbox`）。

## 1. 工具名别名与命令内容探索检测（pruner.py）

### 1.1 别名表
```python
_EXPLORATORY_ALIASES = {
    "list_dir": "list_directory",   # 现有表里有 list_directory 没有 list_dir
    "dir": "ls",
    "where": "which",
    "type": "cat",                  # Windows cmd 的 type = cat
    "dir /b": "ls",
}
```
实现：`_is_exploratory(name)` 时先查原名，再查别名映射。

### 1.2 bash 命令内容探索检测
`bash` 工具的 arguments 是 `{'command': '...'}`。解析 command：
1. 剥离前缀 `cd xxx && ` / `cd xxx; `（cd 本身是导航，不是有效操作）
2. 如果剩余命令以探索性命令开头（ls/dir/cat/type/pwd/where/echo/head/tail/find/date/whoami/print）→ 整条标记探索性剪枝
3. 例外：`python -m pytest` / `python app/main.py` 这类验证命令不剪

## 2. 重复验证精简（pruner.py）

相同 bash 验证命令（规范化后，如剥离 cd 前缀）多次 success 运行，只保留最后一次：
```python
# 类似 prune_result_overwritten，但对 bash 验证命令按 command 文本归组
```

## 3. 注入层有害参数过滤（injector.py）

- `cd /sandbox` 等不存在路径：注入时剥离 bash 参数里的 `cd <非存在路径> && ` 前缀
- 理由：注入的 workflow 来自"上一次会话"，路径/环境上下文可能失效，
  对模型来说有害信息比没有信息更糟（T1 实测 agent 重复 `cd /sandbox` 失败 2 次）

## 4. 实验设计（改后对比）

改完后重跑 T1（改动影响最典型的任务）：
- 改进前 vs 改进后：剪枝率、注入文本长度、注入后 token/调用数
- 如果 T1 改进显著，再跑 T2/T4 确认

## 5. 预期

- 剪枝率提升：T1 从 9/14 保留 → 预计 5-6/14（剪掉 list_dir×2、cd&&pwd、重复验证）
- 注入文本从 403 chars → 预计 ~200 chars
- 注入后 token 应低于基线（目标净省 ≥10-30%）
