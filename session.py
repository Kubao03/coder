import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any

SESSION_DIR = Path.home() / ".coder" / "sessions"


class SessionManager:
    """Persist conversation to JSONL, support resume and listing."""

    def __init__(self, session_id: str | None = None, cwd: str | None = None):
        self.session_id = session_id or uuid4().hex[:12]
        self.cwd = cwd or os.getcwd()
        self._dir = self._project_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        self.path = self._dir / f"{self.session_id}.jsonl"

    def _project_dir(self) -> Path:
        """Session dir scoped to project (sanitized cwd)."""
        sanitized = self.cwd.replace("/", "_").replace("\\", "_").strip("_")
        return SESSION_DIR / sanitized

    def record(self, message: dict[str, Any]) -> None:
        """Append a message to the JSONL file."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "role": message["role"],
            "content": message["content"],
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load(self) -> list[dict[str, Any]]:
        """Load all messages from the JSONL file."""
        if not self.path.exists():
            return []
        messages = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            messages.append({"role": entry["role"], "content": entry["content"]})
        return messages

    @staticmethod
    def list_sessions(cwd: str | None = None) -> list[dict[str, Any]]:
        """List all sessions, optionally filtered by cwd. Most recent first."""
        if cwd:
            sanitized = cwd.replace("/", "_").replace("\\", "_").strip("_")
            dirs = [SESSION_DIR / sanitized]
        else:
            dirs = [d for d in SESSION_DIR.iterdir() if d.is_dir()] if SESSION_DIR.exists() else []

        sessions = []
        for d in dirs:
            if not d.exists():
                continue
            for f in d.glob("*.jsonl"):
                meta = _read_session_meta(f)
                if meta:
                    sessions.append(meta)

        sessions.sort(key=lambda s: s["modified"], reverse=True)
        return sessions

    @staticmethod
    def find_latest(cwd: str) -> "SessionManager | None":
        """Find the most recent session for this project."""
        sessions = SessionManager.list_sessions(cwd=cwd)
        if not sessions:
            return None
        latest = sessions[0]
        return SessionManager(session_id=latest["id"], cwd=cwd)


def _read_session_meta(path: Path) -> dict[str, Any] | None:
    """Read session metadata from JSONL file (first line + file stats)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if not first_line:
            return None
        entry = json.loads(first_line)
        preview = entry.get("content", "")
        if isinstance(preview, list):
            preview = "(tool result)"
        if len(preview) > 80:
            preview = preview[:80] + "..."
        return {
            "id": path.stem,
            "preview": preview,
            "modified": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
            "path": str(path),
        }
    except Exception:
        return None
