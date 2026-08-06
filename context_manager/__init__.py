"""WorkflowManager — 基于 LangGraph 的 Workflow 提取与管理引擎。

核心思想：context = workflow。
任务完成后，从 LangGraph Checkpoints 中提取 toolcall/bashcall 步骤序列，
经过剪枝优化后固化，作为未来任务的优秀案例注入上下文。

用法:
    from context_manager import WorkflowManager

    wfm = WorkflowManager()
    wf_id = wfm.extract_workflow("some_thread_id")
    wfm.solidify(wf_id)
    results = wfm.retrieve("如何修复导入错误")
"""

from .config import Settings
from .manager import WorkflowManager, create_memory_manager

__version__ = "0.5.0"
__all__ = ["WorkflowManager", "Settings", "create_memory_manager"]