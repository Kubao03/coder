import json
import pytest
from pathlib import Path
from coder.settings import load_settings, update_settings, add_permission_rule, _merge_settings


class TestMergeSettings:
    def test_scalar_override(self):
        result = _merge_settings({"model": "a"}, {"model": "b"})
        assert result["model"] == "b"

    def test_list_dedup(self):
        result = _merge_settings(
            {"permissions": {"allow": ["Bash(git *)"]}},
            {"permissions": {"allow": ["Bash(git *)", "Read"]}},
        )
        assert result["permissions"]["allow"] == ["Bash(git *)", "Read"]

    def test_deep_merge(self):
        result = _merge_settings(
            {"permissions": {"allow": ["A"], "deny": ["B"]}},
            {"permissions": {"allow": ["C"]}},
        )
        assert result["permissions"]["allow"] == ["A", "C"]
        assert result["permissions"]["deny"] == ["B"]


class TestLoadSettings:
    def test_no_files(self, tmp_path):
        settings = load_settings(str(tmp_path))
        assert settings.permissions == {"allow": [], "deny": []}
        assert settings.model is None

    def test_user_settings(self, tmp_path, monkeypatch):
        user_dir = tmp_path / "user_home" / ".coder"
        user_dir.mkdir(parents=True)
        (user_dir / "settings.json").write_text(json.dumps({
            "permissions": {"allow": ["Read"]},
            "model": "claude-sonnet-4-6",
        }))
        monkeypatch.setattr("coder.settings.USER_SETTINGS_DIR", user_dir.parent / ".coder")
        # fix: point to the right dir
        monkeypatch.setattr("coder.settings._user_settings_path", lambda: user_dir / "settings.json")

        settings = load_settings(str(tmp_path))
        assert "Read" in settings.permissions["allow"]
        assert settings.model == "claude-sonnet-4-6"

    def test_project_overrides_user(self, tmp_path, monkeypatch):
        # user settings
        user_dir = tmp_path / "user" / ".coder"
        user_dir.mkdir(parents=True)
        (user_dir / "settings.json").write_text(json.dumps({
            "model": "user-model",
            "permissions": {"allow": ["Read"]},
        }))
        monkeypatch.setattr("coder.settings._user_settings_path", lambda: user_dir / "settings.json")

        # project settings
        proj_dir = tmp_path / "project" / ".coder"
        proj_dir.mkdir(parents=True)
        (proj_dir / "settings.json").write_text(json.dumps({
            "model": "project-model",
            "permissions": {"allow": ["Bash(git *)"]},
        }))

        settings = load_settings(str(tmp_path / "project"))
        assert settings.model == "project-model"
        assert "Read" in settings.permissions["allow"]
        assert "Bash(git *)" in settings.permissions["allow"]


class TestUpdateSettings:
    def test_update_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("coder.settings._user_settings_path", lambda: tmp_path / ".coder" / "settings.json")
        update_settings("user", str(tmp_path), {"model": "test-model"})

        path = tmp_path / ".coder" / "settings.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["model"] == "test-model"

    def test_update_merges(self, tmp_path, monkeypatch):
        settings_path = tmp_path / ".coder" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps({"model": "old", "permissions": {"allow": ["A"]}}))
        monkeypatch.setattr("coder.settings._user_settings_path", lambda: settings_path)

        update_settings("user", str(tmp_path), {"permissions": {"allow": ["B"]}})
        data = json.loads(settings_path.read_text())
        assert data["model"] == "old"
        assert "A" in data["permissions"]["allow"]
        assert "B" in data["permissions"]["allow"]


class TestAddPermissionRule:
    def test_add_rule(self, tmp_path):
        proj_dir = tmp_path / ".coder"
        proj_dir.mkdir()

        add_permission_rule("project", str(tmp_path), "allow", "Bash(git *)")
        data = json.loads((proj_dir / "settings.json").read_text())
        assert "Bash(git *)" in data["permissions"]["allow"]

    def test_no_duplicates(self, tmp_path):
        proj_dir = tmp_path / ".coder"
        proj_dir.mkdir()

        add_permission_rule("project", str(tmp_path), "allow", "Read")
        add_permission_rule("project", str(tmp_path), "allow", "Read")
        data = json.loads((proj_dir / "settings.json").read_text())
        assert data["permissions"]["allow"].count("Read") == 1
