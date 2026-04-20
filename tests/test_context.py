import pytest
import shutil
import subprocess
from unittest.mock import patch
from pathlib import Path
import context as context_module
from context import (
    AgentContext,
    _detect_shell,
    _detect_os_version,
    _is_git_repo,
    _current_git_branch,
    _today,
)
from tools.bash import BashTool
from tools.file_read import FileReadTool


def _git_available() -> bool:
    return shutil.which("git") is not None


@pytest.fixture
def git_repo(tmp_path):
    """Create a real git repo with one commit on branch 'main'."""
    if not _git_available():
        pytest.skip("git not installed")
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(*args):
        subprocess.run(args, cwd=repo, check=True, capture_output=True)

    run("git", "init", "-q", "-b", "main")
    run("git", "config", "user.email", "t@t.t")
    run("git", "config", "user.name", "t")
    (repo / "README").write_text("hi\n")
    run("git", "add", "README")
    run("git", "commit", "-q", "-m", "init")
    return repo


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

    def test_system_prompt_contains_today(self, tmp_path, monkeypatch):
        monkeypatch.setattr(context_module, "_today", lambda: "2026-04-20")
        ctx = AgentContext(cwd=str(tmp_path), tools=[])
        prompt = ctx.build_system_prompt()
        assert "Today's date: 2026-04-20" in prompt

    def test_system_prompt_shows_git_branch_in_repo(self, git_repo):
        ctx = AgentContext(cwd=str(git_repo), tools=[])
        prompt = ctx.build_system_prompt()
        assert "Git branch: main" in prompt

    def test_system_prompt_omits_git_branch_outside_repo(self, tmp_path):
        ctx = AgentContext(cwd=str(tmp_path), tools=[])
        prompt = ctx.build_system_prompt()
        assert "Git branch:" not in prompt

    def test_git_repo_detected(self, tmp_path):
        # tmp_path is not a git repo
        assert _is_git_repo(str(tmp_path)) is False

    def test_current_git_branch_in_repo(self, git_repo):
        assert _current_git_branch(str(git_repo)) == "main"

    def test_current_git_branch_outside_repo(self, tmp_path):
        if not _git_available():
            pytest.skip("git not installed")
        outside = tmp_path / "nope"
        outside.mkdir()
        assert _current_git_branch(str(outside)) is None

    def test_current_git_branch_detached_head_returns_none(self, git_repo):
        # Detach HEAD by checking out the commit directly.
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=git_repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "-c", "advice.detachedHead=false", "checkout", head],
            cwd=git_repo, check=True, capture_output=True,
        )
        assert _current_git_branch(str(git_repo)) is None

    def test_today_returns_iso_date_string(self):
        s = _today()
        # Basic shape check: YYYY-MM-DD
        assert len(s) == 10 and s[4] == "-" and s[7] == "-"

    def test_detect_shell(self):
        with patch.dict("os.environ", {"SHELL": "/bin/zsh"}):
            assert _detect_shell() == "zsh"
        with patch.dict("os.environ", {"SHELL": "/usr/bin/bash"}):
            assert _detect_shell() == "bash"

    def test_detect_os_version(self):
        version = _detect_os_version()
        # should be something like "Darwin 25.2.0" or "Linux 6.x"
        assert len(version) > 3
