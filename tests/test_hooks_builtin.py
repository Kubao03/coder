"""Tests for built-in hooks: dangerous_bash_guard and file_write_audit."""

import pytest
from hooks import HookRunner, register_builtin_hooks
from hooks.builtin import dangerous_bash_guard, file_write_audit


# ---------------------------------------------------------------------------
# Dangerous Bash guard
# ---------------------------------------------------------------------------

class TestDangerousBashGuard:
    def test_ignores_non_bash_tools(self):
        result = dangerous_bash_guard("Edit", {"file_path": "x"}, "/tmp")
        assert result is None

    def test_safe_command_passes(self):
        result = dangerous_bash_guard("Bash", {"command": "ls -la"}, "/tmp")
        assert result is None

    def test_rm_rf_root_blocked(self):
        result = dangerous_bash_guard("Bash", {"command": "rm -rf /"}, "/tmp")
        assert result is not None and result.blocked

    def test_rm_rf_home_blocked(self):
        result = dangerous_bash_guard("Bash", {"command": "rm -rf $HOME"}, "/tmp")
        assert result is not None and result.blocked

    def test_rm_rf_tilde_blocked(self):
        result = dangerous_bash_guard("Bash", {"command": "rm -rf ~"}, "/tmp")
        assert result is not None and result.blocked

    def test_rm_rf_relative_path_passes(self):
        result = dangerous_bash_guard("Bash", {"command": "rm -rf ./build"}, "/tmp")
        assert result is None

    def test_dd_to_device_blocked(self):
        result = dangerous_bash_guard("Bash", {"command": "dd if=/dev/zero of=/dev/sda"}, "/tmp")
        assert result is not None and result.blocked

    def test_mkfs_blocked(self):
        result = dangerous_bash_guard("Bash", {"command": "mkfs.ext4 /dev/sda1"}, "/tmp")
        assert result is not None and result.blocked

    def test_fork_bomb_blocked(self):
        result = dangerous_bash_guard("Bash", {"command": ":(){ :|:& };:"}, "/tmp")
        assert result is not None and result.blocked

    def test_non_string_command_passes(self):
        result = dangerous_bash_guard("Bash", {"command": None}, "/tmp")
        assert result is None


# ---------------------------------------------------------------------------
# File write audit log
# ---------------------------------------------------------------------------

class TestFileWriteAudit:
    def test_ignores_non_write_tools(self, tmp_path):
        result = file_write_audit("Bash", {"command": "ls"}, "", str(tmp_path))
        assert result is None
        assert not (tmp_path / ".coder" / "audit.log").exists()

    def test_edit_logged(self, tmp_path):
        file_write_audit("Edit", {"file_path": "/foo/bar.py"}, "ok", str(tmp_path))
        log = (tmp_path / ".coder" / "audit.log").read_text()
        assert "Edit" in log
        assert "/foo/bar.py" in log

    def test_write_logged(self, tmp_path):
        file_write_audit("Write", {"file_path": "/a.txt"}, "ok", str(tmp_path))
        log = (tmp_path / ".coder" / "audit.log").read_text()
        assert "Write" in log
        assert "/a.txt" in log

    def test_appends_not_overwrites(self, tmp_path):
        file_write_audit("Write", {"file_path": "/a.txt"}, "", str(tmp_path))
        file_write_audit("Edit", {"file_path": "/b.txt"}, "", str(tmp_path))
        log = (tmp_path / ".coder" / "audit.log").read_text()
        assert "/a.txt" in log
        assert "/b.txt" in log
        assert len(log.strip().splitlines()) == 2

    def test_missing_file_path_uses_placeholder(self, tmp_path):
        file_write_audit("Write", {}, "", str(tmp_path))
        log = (tmp_path / ".coder" / "audit.log").read_text()
        assert "<unknown>" in log

    def test_returns_none(self, tmp_path):
        """Audit hook should not inject text into tool output."""
        result = file_write_audit("Write", {"file_path": "/a.txt"}, "ok", str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# Registration integration
# ---------------------------------------------------------------------------

class TestRegistration:
    @pytest.mark.asyncio
    async def test_guard_blocks_via_runner(self, tmp_path):
        runner = HookRunner({}, cwd=str(tmp_path))
        register_builtin_hooks(runner)
        result = await runner.run_pre_tool("Bash", {"command": "rm -rf /"})
        assert result.blocked

    @pytest.mark.asyncio
    async def test_safe_bash_passes_via_runner(self, tmp_path):
        runner = HookRunner({}, cwd=str(tmp_path))
        register_builtin_hooks(runner)
        result = await runner.run_pre_tool("Bash", {"command": "echo hi"})
        assert not result.blocked

    @pytest.mark.asyncio
    async def test_audit_written_via_runner(self, tmp_path):
        runner = HookRunner({}, cwd=str(tmp_path))
        register_builtin_hooks(runner)
        await runner.run_post_tool("Edit", {"file_path": "x.py"}, "ok")
        assert (tmp_path / ".coder" / "audit.log").exists()
