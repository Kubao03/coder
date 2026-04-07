import pytest
from unittest.mock import patch
from permissions import PermissionManager
from tools.bash import BashTool
from tools.file_read import FileReadTool
from tools.file_write import FileWriteTool


class TestPermissionManager:
    def setup_method(self):
        self.pm = PermissionManager()
        self.bash = BashTool()
        self.read = FileReadTool()
        self.write = FileWriteTool()

    # Layer 1: always deny
    def test_deny_dangerous_command(self):
        allowed, _ = self.pm.check(self.bash, {"command": "rm -rf / --no-preserve-root"})
        assert allowed is False

    # Layer 2: always allow
    def test_always_allow_read_tool(self):
        allowed, _ = self.pm.check(self.read, {"file_path": "/tmp/x"})
        assert allowed is True

    # Layer 3: read-only
    def test_allow_readonly_bash(self):
        allowed, _ = self.pm.check(self.bash, {"command": "ls -la"})
        assert allowed is True

    # Layer 4: session approval
    def test_session_approval(self):
        allowed, _ = self.pm.check(self.write, {"file_path": "/tmp/x", "content": "y"})
        assert allowed is None  # needs confirmation first

        self.pm._session_allowed.add("Write")
        allowed, _ = self.pm.check(self.write, {"file_path": "/tmp/x", "content": "y"})
        assert allowed is True

    # Layer 5: user prompt - y
    def test_ask_yes(self):
        with patch("builtins.input", return_value="y"):
            result = self.pm.ask(self.write, {"file_path": "/tmp/x", "content": "y"})
        assert result is True
        assert "Write" not in self.pm._session_allowed

    # Layer 5: user prompt - a (always)
    def test_ask_always(self):
        with patch("builtins.input", return_value="a"):
            result = self.pm.ask(self.write, {"file_path": "/tmp/x", "content": "y"})
        assert result is True
        assert "Write" in self.pm._session_allowed

    # Layer 5: user prompt - n
    def test_ask_no(self):
        with patch("builtins.input", return_value="n"):
            result = self.pm.ask(self.write, {"file_path": "/tmp/x", "content": "y"})
        assert result is False

    # Full flow: is_allowed
    def test_is_allowed_write_with_user_yes(self):
        with patch("builtins.input", return_value="y"):
            assert self.pm.is_allowed(self.write, {"file_path": "/tmp/x", "content": "y"}) is True

    def test_is_allowed_dangerous_denied(self):
        assert self.pm.is_allowed(self.bash, {"command": "rm -rf /"}) is False
