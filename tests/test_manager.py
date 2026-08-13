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

    def test_solidify_keeps_all_steps(self):
        """solidify 不再自动剪枝（剪枝由 WorkflowJudge 负责），全部步骤保留。"""
        wf_id = "test_prune"
        self.wfm.workflow_store.create_workflow(wf_id, "固化测试")

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
        # 无 LLM 剪枝标记 → 全部保留
        assert all(not s.is_pruned for s in steps)
        assert wf.description != "(empty workflow)"

    def test_solidify_with_manual_prune(self):
        """手动剪枝标记（LLM 剪枝的工具操作）后 solidify，只固化保留步骤。"""
        wf_id = "test_prune_manual"
        self.wfm.workflow_store.create_workflow(wf_id, "手动剪枝测试")

        steps_data = [
            ("s1", 0, "bashcall", "ls", "{'dir': '.'}", "files", "success"),
            ("s2", 1, "toolcall", "edit_file", "{'path': 'a.py', 'content': 'v2'}", "ok", "success"),
        ]
        for sid, idx, typ, name, args, result, status in steps_data:
            self.wfm.workflow_store.add_step(
                step_id=sid, workflow_id=wf_id, step_index=idx,
                type=typ, name=name, arguments=args, result=result,
                status=status, timestamp="2024-01-01",
            )
        # 模拟 LLM 审查剪枝 s1
        self.wfm.prune_step("s1", True)
        self.wfm.solidify(wf_id)

        wf = self.wfm.workflow_store.get_workflow(wf_id)
        steps = self.wfm.workflow_store.get_steps(wf_id)
        assert steps[0].is_pruned is True
        assert steps[1].is_pruned is False
        assert "edit_file" in wf.description
        assert "ls" not in wf.description

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

    # ── 新工具测试 ──────────────────────────────────

    def test_get_step(self):
        wf_id = "test_get_step"
        self.wfm.workflow_store.create_workflow(wf_id, "获取步骤测试")
        self.wfm.workflow_store.add_step(
            step_id="s1", workflow_id=wf_id, step_index=0,
            type="toolcall", name="read_file",
            arguments="{'path': 'test.py'}", result="content",
            timestamp="2024-01-01",
        )
        result = self.wfm.get_step(wf_id, 0)
        assert "Step 0" in result
        assert "read_file" in result
        assert "test.py" in result

    def test_get_step_not_found(self):
        result = self.wfm.get_step("nonexistent", 0)
        assert "not found" in result

    def test_add_step(self):
        wf_id = "test_add_step"
        self.wfm.workflow_store.create_workflow(wf_id, "添加步骤测试")
        self.wfm.workflow_store.add_step(
            step_id="s1", workflow_id=wf_id, step_index=0,
            type="toolcall", name="read_file", timestamp="2024-01-01",
        )
        result = self.wfm.add_step(wf_id, 0, "bashcall", "python",
                                   arguments="{'script': 'test.py'}", result="OK")
        assert result.startswith("ok:")
        step_id = result.split(":")[1]
        assert len(step_id) == 12
        steps = self.wfm.workflow_store.get_steps(wf_id)
        assert len(steps) == 2
        assert steps[1].name == "python"

    def test_add_step_nonexistent_workflow(self):
        result = self.wfm.add_step("nonexistent", 0, "toolcall", "read_file")
        assert "not found" in result

    def test_remove_step(self):
        wf_id = "test_remove_step"
        self.wfm.workflow_store.create_workflow(wf_id, "删除步骤测试")
        self.wfm.workflow_store.add_step(
            step_id="s1", workflow_id=wf_id, step_index=0,
            type="toolcall", name="read_file", timestamp="2024-01-01",
        )
        self.wfm.workflow_store.add_step(
            step_id="s2", workflow_id=wf_id, step_index=1,
            type="bashcall", name="python", timestamp="2024-01-01",
        )
        result = self.wfm.remove_step("s1")
        assert result == "ok"
        steps = self.wfm.workflow_store.get_steps(wf_id)
        assert len(steps) == 1

    def test_reorder_steps(self):
        wf_id = "test_reorder"
        self.wfm.workflow_store.create_workflow(wf_id, "重排序测试")
        self.wfm.workflow_store.add_step(
            step_id="s1", workflow_id=wf_id, step_index=0,
            type="toolcall", name="step_a", timestamp="2024-01-01",
        )
        self.wfm.workflow_store.add_step(
            step_id="s2", workflow_id=wf_id, step_index=1,
            type="toolcall", name="step_b", timestamp="2024-01-01",
        )
        result = self.wfm.reorder_steps(wf_id, ["s2", "s1"])
        assert result == "ok"
        steps = self.wfm.workflow_store.get_steps(wf_id)
        assert steps[0].step_id == "s2"
        assert steps[1].step_id == "s1"

    def test_batch_prune(self):
        wf_id = "test_batch_prune"
        self.wfm.workflow_store.create_workflow(wf_id, "批量剪枝测试")
        self.wfm.workflow_store.add_step(
            step_id="s1", workflow_id=wf_id, step_index=0,
            type="toolcall", name="read_file", timestamp="2024-01-01",
        )
        self.wfm.workflow_store.add_step(
            step_id="s2", workflow_id=wf_id, step_index=1,
            type="bashcall", name="ls", timestamp="2024-01-01",
        )
        result = self.wfm.batch_prune(wf_id, ["s1", "s2"])
        assert "2 steps pruned" in result
        steps = self.wfm.workflow_store.get_steps(wf_id)
        assert all(s.is_pruned for s in steps)

    def test_review_summary(self):
        wf_id = "test_review_summary"
        self.wfm.workflow_store.create_workflow(wf_id, "摘要测试")
        self.wfm.workflow_store.add_step(
            step_id="s1", workflow_id=wf_id, step_index=0,
            type="toolcall", name="read_file", status="success",
            timestamp="2024-01-01",
        )
        self.wfm.workflow_store.add_step(
            step_id="s2", workflow_id=wf_id, step_index=1,
            type="bashcall", name="python", status="failure",
            timestamp="2024-01-01",
        )
        summary = self.wfm.review_summary(wf_id)
        assert "摘要测试" in summary
        assert "toolcall: 1" in summary
        assert "bashcall: 1" in summary
        assert "失败: 1" in summary

    def test_review_summary_empty(self):
        self.wfm.workflow_store.create_workflow("empty_wf", "空 Workflow")
        summary = self.wfm.review_summary("empty_wf")
        assert "(empty)" in summary

    def test_update_step_all_fields(self):
        wf_id = "test_update_fields"
        self.wfm.workflow_store.create_workflow(wf_id, "字段更新测试")
        self.wfm.workflow_store.add_step(
            step_id="s1", workflow_id=wf_id, step_index=0,
            type="toolcall", name="read_file",
            arguments="{'path': 'old.py'}", result="旧内容",
            timestamp="2024-01-01",
        )
        self.wfm.update_step("s1", name="edit_file", arguments="{'path': 'new.py'}", result="新内容")
        steps = self.wfm.workflow_store.get_steps(wf_id)
        assert steps[0].name == "edit_file"
        assert steps[0].arguments == "{'path': 'new.py'}"
        assert steps[0].result == "新内容"

    def test_get_tool_schemas(self):
        schemas = self.wfm.get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) >= 14
        names = [s["name"] for s in schemas]
        assert "get_workflow" in names
        assert "update_step" in names
        assert "add_step" in names
        assert "remove_step" in names
        assert "reorder_steps" in names
        assert "batch_prune" in names
        assert "review_summary" in names
        assert "get_step" in names
        # 验证每个 schema 都有必要字段
        for s in schemas:
            assert "name" in s
            assert "description" in s
            assert "parameters" in s
            assert "properties" in s["parameters"]