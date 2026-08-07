"""WorkflowManager 集成测试。"""

from context_manager import create_memory_manager, Workflow, Step


class TestWorkflowManager:
    def setup_method(self):
        self.wfm = create_memory_manager()

    def teardown_method(self):
        self.wfm.close()

    def test_create_workflow(self):
        self.wfm.workflow_store.create_workflow(
            workflow_id="test_wf_001",
            name="测试 Workflow",
            source_thread_id="thread_001",
        )
        wf = self.wfm.workflow_store.get_workflow("test_wf_001")
        assert wf is not None
        assert isinstance(wf, Workflow)
        assert wf.name == "测试 Workflow"
        assert wf.status == "RAW"
        assert wf.source_thread_id == "thread_001"

    def test_store_step(self):
        wf_id = "test_wf_steps"
        self.wfm.workflow_store.create_workflow(wf_id, "带步骤的 Workflow")
        self.wfm.workflow_store.add_step(
            step_id="s1", workflow_id=wf_id, step_index=0,
            type="toolcall", name="read_file",
            arguments="{'path': 'test.py'}", result="content",
            timestamp="2024-01-01",
        )
        self.wfm.workflow_store.add_step(
            step_id="s2", workflow_id=wf_id, step_index=1,
            type="bashcall", name="python",
            arguments="{'script': 'test.py'}", result="OK",
            timestamp="2024-01-01",
        )
        steps = self.wfm.workflow_store.get_steps(wf_id)
        assert len(steps) == 2
        assert isinstance(steps[0], Step)
        assert steps[0].name == "read_file"
        assert steps[0].type == "toolcall"
        assert steps[1].name == "python"
        assert steps[1].type == "bashcall"

    def test_solidify_and_prune(self):
        wf_id = "test_prune"
        self.wfm.workflow_store.create_workflow(wf_id, "剪枝测试")

        steps_data = [
            ("s1", 0, "bashcall", "ls", "{'dir': '.'}", "files", "success"),
            ("s2", 1, "toolcall", "edit_file", "{'path': 'a.py', 'content': 'v1'}", "ok", "success"),
            ("s3", 2, "toolcall", "edit_file", "{'path': 'a.py', 'content': 'v2'}", "ok", "success"),
            ("s4", 3, "bashcall", "python", "{'script': 'a.py'}", "v2", "success"),
        ]
        for sid, idx, typ, name, args, result, status in steps_data:
            self.wfm.workflow_store.add_step(
                step_id=sid, workflow_id=wf_id, step_index=idx,
                type=typ, name=name, arguments=args, result=result,
                status=status, timestamp="2024-01-01",
            )
        self.wfm.solidify(wf_id)

        wf = self.wfm.workflow_store.get_workflow(wf_id)
        assert wf.status == "SOLIDIFIED"

        steps = self.wfm.workflow_store.get_steps(wf_id)
        pruned = [s for s in steps if s.is_pruned]
        kept = [s for s in steps if not s.is_pruned]
        assert len(pruned) == 2  # ls + edit_file v1
        assert len(kept) == 2    # edit_file v2 + python

    def test_retrieve(self):
        wf_id = "test_retrieve"
        self.wfm.workflow_store.create_workflow(wf_id, "检索测试")
        self.wfm.workflow_store.add_step(
            step_id="s1", workflow_id=wf_id, step_index=0,
            type="toolcall", name="read_file",
            arguments="{'path': 'bug.py'}", result="content",
            status="success", timestamp="2024-01-01",
        )
        self.wfm.workflow_store.add_step(
            step_id="s2", workflow_id=wf_id, step_index=1,
            type="toolcall", name="edit_file",
            arguments="{'path': 'bug.py'}", result="fixed",
            status="success", timestamp="2024-01-01",
        )
        self.wfm.solidify(wf_id)
        results = self.wfm.retrieve("修复 bug", top_k=3)
        assert len(results) >= 1
        assert isinstance(results[0], Workflow)

    def test_retrieve_empty(self):
        results = self.wfm.retrieve("test")
        assert results == []

    def test_delete_workflow(self):
        wf_id = "test_delete"
        self.wfm.workflow_store.create_workflow(wf_id, "待删除")
        assert self.wfm.workflow_store.get_workflow(wf_id) is not None
        self.wfm.delete_workflow(wf_id)
        assert self.wfm.workflow_store.get_workflow(wf_id) is None

    def test_list_workflows(self):
        self.wfm.workflow_store.create_workflow("wf1", "第一个")
        self.wfm.workflow_store.create_workflow("wf2", "第二个")
        assert len(self.wfm.list_workflows()) == 2
        assert len(self.wfm.list_workflows(status="RAW")) == 2

    def test_prune_step_tool(self):
        wf_id = "test_tool"
        self.wfm.workflow_store.create_workflow(wf_id, "工具测试")
        self.wfm.workflow_store.add_step(
            step_id="s1", workflow_id=wf_id, step_index=0,
            type="toolcall", name="read_file", timestamp="2024-01-01",
        )
        result = self.wfm.prune_step("s1", True)
        assert result is True
        steps = self.wfm.workflow_store.get_steps(wf_id)
        assert steps[0].is_pruned is True