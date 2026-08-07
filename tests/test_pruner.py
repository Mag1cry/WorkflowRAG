"""Pruner 剪枝引擎单元测试。"""

from context_manager.workflow import pruner


class TestPruner:
    def test_prune_exploratory(self):
        steps = [
            {"step_id": "s1", "type": "bashcall", "name": "ls", "arguments": "", "is_pruned": False},
            {"step_id": "s2", "type": "toolcall", "name": "read_file", "arguments": "", "is_pruned": False},
            {"step_id": "s3", "type": "toolcall", "name": "edit_file", "arguments": "", "is_pruned": False},
        ]
        result = pruner.prune_exploratory(steps)
        assert result[0]["is_pruned"] is True
        assert result[1]["is_pruned"] is True
        assert result[2]["is_pruned"] is False

    def test_prune_result_overwritten(self):
        steps = [
            {"step_id": "s1", "type": "toolcall", "name": "edit_file",
             "arguments": "{'path': 'a.py'}", "is_pruned": False},
            {"step_id": "s2", "type": "toolcall", "name": "edit_file",
             "arguments": "{'path': 'a.py'}", "is_pruned": False},
            {"step_id": "s3", "type": "toolcall", "name": "edit_file",
             "arguments": "{'path': 'b.py'}", "is_pruned": False},
        ]
        result = pruner.prune_result_overwritten(steps)
        assert result[0]["is_pruned"] is True
        assert result[1]["is_pruned"] is False
        assert result[2]["is_pruned"] is False

    def test_prune_failed_irrelevant(self):
        steps = [
            {"step_id": "s1", "type": "toolcall", "name": "pip_install",
             "arguments": "", "status": "failure", "is_pruned": False},
            {"step_id": "s2", "type": "toolcall", "name": "conda_install",
             "arguments": "", "status": "success", "is_pruned": False},
            {"step_id": "s3", "type": "bashcall", "name": "python",
             "arguments": "", "status": "success", "is_pruned": False},
        ]
        result = pruner.prune_failed_irrelevant(steps)
        assert result[0]["is_pruned"] is True
        assert result[1]["is_pruned"] is False
        assert result[2]["is_pruned"] is False

    def test_prune_all_strategies(self):
        steps = [
            {"step_id": "s1", "type": "bashcall", "name": "ls", "arguments": "",
             "status": "success", "is_pruned": False},
            {"step_id": "s2", "type": "toolcall", "name": "edit_file",
             "arguments": "{'path': 'x.py'}", "status": "success", "is_pruned": False},
            {"step_id": "s3", "type": "toolcall", "name": "edit_file",
             "arguments": "{'path': 'x.py'}", "status": "success", "is_pruned": False},
            {"step_id": "s4", "type": "bashcall", "name": "python",
             "arguments": "", "status": "success", "is_pruned": False},
        ]
        result = pruner.prune(steps)
        pruned = [s for s in result if s["is_pruned"]]
        kept = [s for s in result if not s["is_pruned"]]
        assert len(pruned) == 2
        assert len(kept) == 2

    def test_generate_description(self):
        steps = [
            {"step_id": "s1", "type": "toolcall", "name": "read_file",
             "arguments": "{'path': 'test.py'}", "is_pruned": False},
            {"step_id": "s2", "type": "toolcall", "name": "edit_file",
             "arguments": "{'path': 'test.py'}", "is_pruned": False},
        ]
        desc = pruner.generate_description(steps)
        assert "read_file" in desc
        assert "edit_file" in desc

    def test_generate_description_empty(self):
        desc = pruner.generate_description([])
        assert desc == "(empty workflow)"

    def test_generate_description_after_prune(self):
        steps = [
            {"step_id": "s1", "type": "bashcall", "name": "ls", "arguments": "",
             "is_pruned": True},
            {"step_id": "s2", "type": "toolcall", "name": "edit_file",
             "arguments": "{'path': 'a.py'}", "is_pruned": False},
        ]
        desc = pruner.generate_description(steps)
        assert "ls" not in desc
        assert "edit_file" in desc