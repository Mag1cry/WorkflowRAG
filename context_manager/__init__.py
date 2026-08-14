"""WorkflowManager — 基于 LangGraph 的 Workflow 提取与管理引擎。

核心思想：context = workflow。
任务完成后，从 LangGraph Checkpoints 中提取 toolcall/bashcall 步骤序列，
由 WorkflowJudge（LLM 审查 Agent）剪枝后固化，作为未来任务的优秀案例注入上下文。

用法:
    from context_manager import WorkflowManager
    from context_manager.workflow.judge import WorkflowJudge

    wfm = WorkflowManager()
    wf_id = wfm.extract_workflow("some_thread_id")
    judge = WorkflowJudge(wfm, llm)   # LLM 剪枝
    judge.judge(wf_id)
    wfm.solidify(wf_id)
    results = wfm.retrieve("如何修复导入错误")
"""

from .config import Settings
from .models import Workflow, Step
from .workflow import WorkflowManager, WorkflowJudge, create_memory_manager

__version__ = "0.7.0"
__all__ = [
    "WorkflowManager",
    "WorkflowJudge",
    "Workflow",
    "Step",
    "Settings",
    "create_memory_manager",
]
