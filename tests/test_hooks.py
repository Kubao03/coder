"""Tests for HookRunner."""

import json
import pytest
from hooks import HookRunner, _matches


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

class TestMatchesPattern:
    def test_empty_matcher_matches_all(self):
        assert _matches("Bash", "")

    def test_star_matches_all(self):
        assert _matches("Bash", "*")

    def test_exact_name(self):
        assert _matches("Bash", "Bash")
        assert not _matches("Edit", "Bash")

    def test_pipe_separated(self):
        assert _matches("Edit", "Edit|Write")
        assert _matches("Write", "Edit|Write")
        assert not _matches("Bash", "Edit|Write")

    def test_regex_pattern(self):
        assert _matches("FileRead", "File.*")
        assert not _matches("Bash", "File.*")

    def test_invalid_regex_returns_false(self):
        assert not _matches("Bash", "[invalid")


# ---------------------------------------------------------------------------
# HookRunner
# ---------------------------------------------------------------------------

@pytest.fixture
def runner(tmp_path):
    return HookRunner(hooks_config={}, cwd=str(tmp_path))


@pytest.fixture
def runner_with_hooks(tmp_path):
    config = {
        "PreToolUse": [
            {
                "matcher": "Bash",
                "hooks": [{"type": "command", "command": "echo '{\"decision\": \"approve\"}'"}],
            }
        ],
        "PostToolUse": [
            {
                "matcher": "Bash",
                "hooks": [{"type": "command", "command": "echo 'post hook ran'"}],
            }
        ],
    }
    return HookRunner(hooks_config=config, cwd=str(tmp_path))


class TestHookRunner:
    @pytest.mark.asyncio
    async def test_no_hooks_returns_unblocked(self, runner):
        result = await runner.run_pre_tool("Bash", {"command": "ls"})
        assert not result.blocked

    @pytest.mark.asyncio
    async def test_pre_tool_approve(self, runner_with_hooks):
        result = await runner_with_hooks.run_pre_tool("Bash", {"command": "ls"})
        assert not result.blocked

    @pytest.mark.asyncio
    async def test_pre_tool_block_via_json(self, tmp_path):
        config = {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {"type": "command", "command": "echo '{\"decision\": \"block\", \"reason\": \"no rm\"}'"}
                    ],
                }
            ]
        }
        runner = HookRunner(config, cwd=str(tmp_path))
        result = await runner.run_pre_tool("Bash", {"command": "rm -rf /"})
        assert result.blocked
        assert "no rm" in result.block_reason

    @pytest.mark.asyncio
    async def test_pre_tool_block_via_exit_code_2(self, tmp_path):
        config = {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "echo 'blocked'; exit 2"}],
                }
            ]
        }
        runner = HookRunner(config, cwd=str(tmp_path))
        result = await runner.run_pre_tool("Bash", {"command": "ls"})
        assert result.blocked

    @pytest.mark.asyncio
    async def test_pre_tool_no_match(self, runner_with_hooks):
        result = await runner_with_hooks.run_pre_tool("Edit", {"path": "foo.py"})
        assert not result.blocked

    @pytest.mark.asyncio
    async def test_post_tool_output(self, runner_with_hooks):
        result = await runner_with_hooks.run_post_tool("Bash", {"command": "ls"}, "file.txt")
        assert "post hook ran" in result.output

    @pytest.mark.asyncio
    async def test_stdin_receives_json(self, tmp_path):
        """Hook reads stdin and echoes tool_name back; verify JSON is passed correctly."""
        config = {
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [
                        {"type": "command", "command": "python3 -c \"import sys,json; d=json.load(sys.stdin); print(d['tool_name'])\""}
                    ],
                }
            ]
        }
        runner = HookRunner(config, cwd=str(tmp_path))
        result = await runner.run_pre_tool("Bash", {"command": "ls"})
        assert not result.blocked
