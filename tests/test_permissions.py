import json
import pytest
from unittest.mock import patch
from coder.persistence.settings import Settings
from coder.permissions import PermissionManager
from coder.tools.bash import BashTool
from coder.tools.file_read import FileReadTool
from coder.tools.file_write import FileWriteTool


def _make_pm(allow=None, deny=None):
    """Helper to create a PermissionManager with given rules."""
    settings = Settings(
        permissions={
            "allow": allow or [],
            "deny": deny or [],
        },
        _user_raw={"permissions": {"allow": allow or [], "deny": deny or []}},
        _project_raw={},
    )
    return PermissionManager(settings, cwd="/tmp")


class TestPermissionManager:
    def setup_method(self):
        self.bash = BashTool()
        self.read = FileReadTool()
        self.write = FileWriteTool()

    # Hard-coded deny
    def test_deny_dangerous_command(self):
        pm = _make_pm()
        allowed, _ = pm.check(self.bash, {"command": "rm -rf / --no-preserve-root"})
        assert allowed is False

    # Read-only tool auto-allowed
    def test_always_allow_read_tool(self):
        pm = _make_pm()
        allowed, _ = pm.check(self.read, {"file_path": "/tmp/x"})
        assert allowed is True

    # Read-only bash auto-allowed
    def test_allow_readonly_bash(self):
        pm = _make_pm()
        allowed, _ = pm.check(self.bash, {"command": "ls -la"})
        assert allowed is True

    # Settings deny rule
    def test_deny_rule(self):
        pm = _make_pm(deny=["Bash(npm *)"])
        allowed, reason = pm.check(self.bash, {"command": "npm install evil"})
        assert allowed is False
        assert "denied" in reason.lower()

    # Settings allow rule
    def test_allow_rule(self):
        pm = _make_pm(allow=["Write"])
        allowed, reason = pm.check(self.write, {"file_path": "/tmp/x", "content": "y"})
        assert allowed is True
        assert "allow" in reason.lower()

    # Deny overrides allow
    def test_deny_overrides_allow(self):
        pm = _make_pm(allow=["Bash"], deny=["Bash(rm *)"])
        allowed, _ = pm.check(self.bash, {"command": "rm foo"})
        assert allowed is False

    # Bash pattern matching
    def test_allow_bash_pattern(self):
        pm = _make_pm(allow=["Bash(git *)"])
        allowed, _ = pm.check(self.bash, {"command": "git status"})
        assert allowed is True
        allowed2, _ = pm.check(self.bash, {"command": "npm install"})
        assert allowed2 is None  # needs confirmation

    # Session approval
    def test_session_approval(self):
        pm = _make_pm()
        allowed, _ = pm.check(self.write, {"file_path": "/tmp/x", "content": "y"})
        assert allowed is None

        with patch("builtins.input", return_value="a"):
            result = pm.ask(self.write, {"file_path": "/tmp/x", "content": "y"})
        assert result is True

        # now session rule should apply
        allowed2, reason = pm.check(self.write, {"file_path": "/tmp/x", "content": "y"})
        assert allowed2 is True
        assert "session" in reason

    # User prompt - y
    def test_ask_yes(self):
        pm = _make_pm()
        with patch("builtins.input", return_value="y"):
            result = pm.ask(self.write, {"file_path": "/tmp/x", "content": "y"})
        assert result is True

    # User prompt - n
    def test_ask_no(self):
        pm = _make_pm()
        with patch("builtins.input", return_value="n"):
            result = pm.ask(self.write, {"file_path": "/tmp/x", "content": "y"})
        assert result is False

    # Persist rule — saves with generalized pattern
    def test_ask_persist(self, tmp_path):
        settings = Settings(
            permissions={"allow": [], "deny": []},
            _user_raw={},
            _project_raw={},
        )
        pm = PermissionManager(settings, cwd=str(tmp_path))

        with patch("builtins.input", return_value="p"):
            result = pm.ask(self.write, {"file_path": "/tmp/x", "content": "y"})
        assert result is True

        # rule should be persisted with parent directory glob
        settings_path = tmp_path / ".coder" / "settings.json"
        assert settings_path.exists()
        data = json.loads(settings_path.read_text())
        assert "Write(/tmp/*)" in data["permissions"]["allow"]

        # should match other files in the same directory
        allowed, _ = pm.check(self.write, {"file_path": "/tmp/other.py", "content": "y"})
        assert allowed is True

        # but NOT match a different directory
        allowed2, _ = pm.check(self.write, {"file_path": "/etc/passwd", "content": "y"})
        assert allowed2 is None

    # Persist rule for Bash — saves generalized command prefix
    def test_ask_persist_bash(self, tmp_path):
        settings = Settings(
            permissions={"allow": [], "deny": []},
            _user_raw={},
            _project_raw={},
        )
        pm = PermissionManager(settings, cwd=str(tmp_path))

        with patch("builtins.input", return_value="p"):
            result = pm.ask(self.bash, {"command": "git add main.py"})
        assert result is True

        settings_path = tmp_path / ".coder" / "settings.json"
        data = json.loads(settings_path.read_text())
        # should save 'git add *', not 'git add main.py'
        assert "Bash(git add *)" in data["permissions"]["allow"]

        # should match other git add commands
        allowed, _ = pm.check(self.bash, {"command": "git add other.py"})
        assert allowed is True

        # but not git push
        allowed2, _ = pm.check(self.bash, {"command": "git push"})
        assert allowed2 is None

    # Session always — also saves with generalized pattern
    def test_session_always_with_pattern(self):
        pm = _make_pm()
        with patch("builtins.input", return_value="a"):
            pm.ask(self.bash, {"command": "npm install foo"})

        # same subcommand, different args — should be allowed
        allowed, _ = pm.check(self.bash, {"command": "npm install bar"})
        assert allowed is True

        # different subcommand — should NOT be auto-allowed
        allowed2, _ = pm.check(self.bash, {"command": "npm run evil"})
        assert allowed2 is None

    # Full flow
    def test_is_allowed_dangerous_denied(self):
        pm = _make_pm()
        assert pm.is_allowed(self.bash, {"command": "rm -rf /"}) is False
