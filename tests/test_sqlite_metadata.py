"""SQLiteWorkflowStore 单元测试。"""

import os
import tempfile
import time

from context_manager.storage import SQLiteWorkflowStore


class TestSQLiteWorkflowStore:
    """SQLiteWorkflowStore 核心功能测试。"""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.store = SQLiteWorkflowStore(self.tmp.name)

    def teardown_method(self):
        self.store.close()
        for _ in range(5):
            try:
                os.unlink(self.tmp.name)
                break
            except PermissionError:
                time.sleep(0.1)

    def test_create_and_get_workflow(self):
        self.store.create_workflow("wf1", "测试 Workflow", source_thread_id="t1")
        wf = self.store.get_workflow("wf1")
        assert wf is not None
        assert wf["name"] == "测试 Workflow"
        assert wf["status"] == "RAW"
        assert wf["source_thread_id"] == "t1"

    def test_list_workflows(self):
        self.store.create_workflow("wf1", "第一个")
        self.store.create_workflow("wf2", "第二个")
        assert len(self.store.list_workflows()) == 2

    def test_list_workflows_by_status(self):
        self.store.create_workflow("wf1", "RAW 测试")
        self.store.create_workflow("wf2", "SOLIDIFIED 测试")
        self.store.update_status("wf2", "SOLIDIFIED")

        raw_list = self.store.list_workflows(status="RAW")
        solid_list = self.store.list_workflows(status="SOLIDIFIED")
        assert len(raw_list) == 1
        assert len(solid_list) == 1

    def test_update_status(self):
        self.store.create_workflow("wf1", "状态测试")
        self.store.update_status("wf1", "SOLIDIFIED")
        wf = self.store.get_workflow("wf1")
        assert wf["status"] == "SOLIDIFIED"

    def test_delete_workflow(self):
        self.store.create_workflow("wf1", "待删除")
        self.store.add_step("s1", "wf1", 0, "toolcall", "read_file")
        self.store.delete_workflow("wf1")
        assert self.store.get_workflow("wf1") is None
        assert self.store.get_steps("wf1") == []

    def test_add_and_get_steps(self):
        self.store.create_workflow("wf1", "步骤测试")
        self.store.add_step("s1", "wf1", 0, "toolcall", "read_file",
                            arguments="{'path': 'test.py'}", result="content")
        self.store.add_step("s2", "wf1", 1, "bashcall", "python",
                            arguments="{'script': 'test.py'}", result="OK")

        steps = self.store.get_steps("wf1")
        assert len(steps) == 2
        assert steps[0]["name"] == "read_file"
        assert steps[0]["type"] == "toolcall"
        assert steps[1]["name"] == "python"
        assert steps[1]["type"] == "bashcall"

    def test_update_step_pruned(self):
        self.store.create_workflow("wf1", "剪枝测试")
        self.store.add_step("s1", "wf1", 0, "toolcall", "read_file")
        self.store.update_step_pruned("s1", True)

        steps = self.store.get_steps("wf1")
        assert steps[0]["is_pruned"] is True

    def test_get_nonexistent(self):
        assert self.store.get_workflow("nonexistent") is None