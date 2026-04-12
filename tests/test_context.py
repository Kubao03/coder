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

