"""Two-layer settings: user (~/.coder/) and project (.coder/)."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_SETTINGS_DIR = Path.home() / ".coder"
PROJECT_SETTINGS_DIRNAME = ".coder"
SETTINGS_FILENAME = "settings.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "permissions": {
        "allow": [],
        "deny": [],
    },
    "hooks": {},
    "model": None,
}


# ---------------------------------------------------------------------------
# Settings loading / merging / updating
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    """Merged settings from user + project layers."""

    permissions: dict[str, list[str]] = field(default_factory=lambda: {"allow": [], "deny": []})
    hooks: dict[str, Any] = field(default_factory=dict)
    model: str | None = None

    # raw per-source data for update targeting
    _user_raw: dict[str, Any] = field(default_factory=dict, repr=False)
    _project_raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file, returning empty dict if missing or invalid."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _merge_lists(base: list, override: list) -> list:
    """Concatenate and deduplicate, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in base + override:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _merge_settings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge override into base. Arrays deduplicate, dicts recurse, scalars override."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged:
            if isinstance(merged[key], list) and isinstance(value, list):
                merged[key] = _merge_lists(merged[key], value)
            elif isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = _merge_settings(merged[key], value)
            else:
                merged[key] = value
        else:
            merged[key] = value
    return merged


def _user_settings_path() -> Path:
    return USER_SETTINGS_DIR / SETTINGS_FILENAME


def _project_settings_path(cwd: str) -> Path:
    return Path(cwd) / PROJECT_SETTINGS_DIRNAME / SETTINGS_FILENAME


def load_settings(cwd: str) -> Settings:
    """Load and merge user → project settings."""
    user_raw = _read_json(_user_settings_path())
    project_raw = _read_json(_project_settings_path(cwd))

    base = dict(DEFAULT_SETTINGS)
    merged = _merge_settings(base, user_raw)
    merged = _merge_settings(merged, project_raw)

    perms = merged.get("permissions", {})

    return Settings(
        permissions={
            "allow": perms.get("allow", []),
            "deny": perms.get("deny", []),
        },
        hooks=merged.get("hooks", {}),
        model=merged.get("model"),
        _user_raw=user_raw,
        _project_raw=project_raw,
    )


def update_settings(source: str, cwd: str, patch: dict[str, Any]) -> None:
    """Update a specific settings source and write to disk.

    - source: "user" or "project"
    - patch: partial settings dict to merge in
    """
    if source == "user":
        path = _user_settings_path()
    elif source == "project":
        path = _project_settings_path(cwd)
    else:
        raise ValueError(f"Unknown source: {source}")

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_json(path)
    updated = _merge_settings(existing, patch)
    path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def add_permission_rule(source: str, cwd: str, behavior: str, rule: str) -> None:
    """Append a permission rule to a settings source, avoiding duplicates."""
    if source == "user":
        path = _user_settings_path()
    elif source == "project":
        path = _project_settings_path(cwd)
    else:
        raise ValueError(f"Unknown source: {source}")

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_json(path)
    perms = existing.setdefault("permissions", {})
    rules = perms.setdefault(behavior, [])
    if rule not in rules:
        rules.append(rule)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
