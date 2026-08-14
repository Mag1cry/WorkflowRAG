"""上下文注入 injector 单元测试。"""

from context_manager import Workflow, Step
from context_manager.workflow.injector import format_context, format_context_compact


class TestInjector:
    def test_format_context(self):
        steps = [
            Step(
                step_id="s1",
                workflow_id="w1",
                step_index=0,
                type="toolcall",
                name="read_file",
                arguments="{'path': 'test.py'}",
                result="content",
            ),
            Step(
                step_id="s2",
                workflow_id="w1",
                step_index=1,
                type="toolcall",
                name="edit_file",
                arguments="{'path': 'test.py'}",
                result="fixed",
            ),
        ]
        wf = Workflow(workflow_id="w1", name="修复测试", steps=steps)
        result = format_context(wf)
        assert "参考 Workflow: 修复测试" in result
        assert "Step 1:  read_file" in result
        assert "Step 2:  edit_file" in result

    def test_format_context_with_pruned(self):
        steps = [
            Step(
                step_id="s1",
                workflow_id="w1",
                step_index=0,
                type="bashcall",
                name="ls",
                arguments="",
                is_pruned=True,
            ),
            Step(
                step_id="s2",
                workflow_id="w1",
                step_index=1,
                type="toolcall",
                name="edit_file",
                arguments="{'path': 'test.py'}",
                result="fixed",
            ),
        ]
        wf = Workflow(workflow_id="w1", name="含剪枝", steps=steps)
        result = format_context(wf)
        assert "ls" not in result
        assert "edit_file" in result

    def test_format_context_empty(self):
        wf = Workflow(workflow_id="w1", name="空 Workflow")
        result = format_context(wf)
        assert "无可用步骤" in result

    def test_format_context_compact(self):
        steps = [
            Step(
                step_id="s1",
                workflow_id="w1",
                step_index=0,
                type="toolcall",
                name="read_file",
                arguments="{'path': 'test.py'}",
            ),
            Step(
                step_id="s2",
                workflow_id="w1",
                step_index=1,
                type="toolcall",
                name="edit_file",
                arguments="{'path': 'test.py'}",
            ),
        ]
        wf = Workflow(workflow_id="w1", name="紧凑测试", steps=steps)
        result = format_context_compact(wf)
        assert "read_file" in result
        assert "edit_file" in result
        assert "→" in result

    def test_format_context_compact_empty(self):
        wf = Workflow(workflow_id="w1", name="空")
        result = format_context_compact(wf)
        assert "empty" in result
