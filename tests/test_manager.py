"""WorkflowManager 集成测试。"""

from context_manager import create_memory_manager


class TestWorkflowManager:
    """WorkflowManager 核心功能测试（纯内存后端）。"""

    def setup_method(self):
        self.wfm = create_memory_manager()

    def teardown_method(self):
        self.wfm.close()

    def test_create_workflow(self):
        wf_id = "test_wf_001"
        self.wfm.workflow_store.create_workflow(
            workflow_id=wf_id,
            name="测试 Workflow",
            source_thread_id="thread_001",
        )
        wf = self.wfm.workflow_store.get_workflow(wf_id)
        assert wf is not None
        assert wf["name"] == "测试 Workflow"
        assert wf["status"] == "RAW"
        assert wf["source_thread_id"] == "thread_001"

    def test_add_and_get_steps(self):
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
        assert steps[0]["name"] == "read_file"
        assert steps[1]["name"] == "python"

    def test_solidify_and_prune(self):
        wf_id = "test_prune"
        self.wfm.workflow_store.create_workflow(wf_id, "剪枝测试")

        # 按顺序添加步骤：探索性bash → 写文件v1 → 写文件v2（覆盖）→ 验证
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
        assert wf["status"] == "SOLIDIFIED"

        steps = self.wfm.workflow_store.get_steps(wf_id)
        # s1 (ls) 应该被剪枝（探索性）, s2 (edit_file v1) 应该被剪枝（结果被覆盖）
        pruned = [s for s in steps if s.get("is_pruned")]
        kept = [s for s in steps if not s.get("is_pruned")]
        assert len(pruned) == 2
        assert len(kept) == 2

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

        all_wf = self.wfm.list_workflows()
        assert len(all_wf) == 2

        raw_wf = self.wfm.list_workflows(status="RAW")
        assert len(raw_wf) == 2