import pytest
from unittest.mock import patch
from pathlib import Path
from context import AgentContext, _detect_shell, _detect_os_version, _is_git_repo
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

    def test_system_prompt_contains_environment_section(self, tmp_path):
        ctx = AgentContext(cwd=str(tmp_path), tools=[])
        prompt = ctx.build_system_prompt()
        assert "# Environment" in prompt
        assert "Platform:" in prompt
        assert "Shell:" in prompt
        assert "OS Version:" in prompt
        assert "git repository:" in prompt

    def test_git_repo_detected(self, tmp_path):
        # tmp_path is not a git repo
        assert _is_git_repo(str(tmp_path)) is False

    def test_detect_shell(self):
        with patch.dict("os.environ", {"SHELL": "/bin/zsh"}):
            assert _detect_shell() == "zsh"
        with patch.dict("os.environ", {"SHELL": "/usr/bin/bash"}):
            assert _detect_shell() == "bash"

    def test_detect_os_version(self):
        version = _detect_os_version()
        # should be something like "Darwin 25.2.0" or "Linux 6.x"
        assert len(version) > 3
