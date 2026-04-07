import pytest
from pathlib import Path
from context import AgentContext
from tools.bash import BashTool
from tools.file_read import FileReadTool


class TestAgentContext:
    def setup_method(self, tmp_path=None):
        self.bash = BashTool()
        self.read = FileReadTool()

    def test_system_prompt_contains_cwd(self, tmp_path):
        ctx = AgentContext(cwd=str(tmp_path), tools=[self.bash])
        prompt = ctx.build_system_prompt()
        assert str(tmp_path) in prompt

    def test_system_prompt_contains_tool_names(self, tmp_path):
        ctx = AgentContext(cwd=str(tmp_path), tools=[self.bash, self.read])
        prompt = ctx.build_system_prompt()
        assert "Bash" in prompt
        assert "Read" in prompt

    def test_system_prompt_includes_coder_md(self, tmp_path):
        (tmp_path / "CODER.md").write_text("# Project rules\nNo monkey patching.")
        ctx = AgentContext(cwd=str(tmp_path), tools=[])
        prompt = ctx.build_system_prompt()
        assert "No monkey patching." in prompt

    def test_system_prompt_no_coder_md(self, tmp_path):
        ctx = AgentContext(cwd=str(tmp_path), tools=[])
        prompt = ctx.build_system_prompt()
        assert "<coder_md>" not in prompt

    def test_compact_removes_oldest_pair(self, tmp_path):
        ctx = AgentContext(cwd=str(tmp_path), tools=[])
        for i in range(10):
            ctx.messages.append({"role": "user", "content": f"msg {i}"})
            ctx.messages.append({"role": "assistant", "content": f"reply {i}"})

        ctx.compact_messages(max_messages=6)
        assert len(ctx.messages) <= 6
        # Newest messages should be preserved
        assert ctx.messages[-1]["content"] == "reply 9"
        assert ctx.messages[-2]["content"] == "msg 9"

    def test_compact_no_op_when_short(self, tmp_path):
        ctx = AgentContext(cwd=str(tmp_path), tools=[])
        ctx.messages = [{"role": "user", "content": "hi"}]
        ctx.compact_messages(max_messages=40)
        assert len(ctx.messages) == 1

    def test_compact_never_leaves_tool_result_as_head(self, tmp_path):
        """After compaction, history must not start with a tool_result message."""
        ctx = AgentContext(cwd=str(tmp_path), tools=[])
        # Simulate: user → assistant(tool_use) → user(tool_result) → assistant(text)
        ctx.messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
            {"role": "assistant", "content": "done"},
        ]
        # Force compaction to remove first message
        ctx.compact_messages(max_messages=2)
        # Head must never be a tool_result
        assert not AgentContext._is_tool_result(ctx.messages[0])
