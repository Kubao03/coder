import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, TYPE_CHECKING
from tools.base import Tool

if TYPE_CHECKING:
    from agent_services import AgentServices


def _detect_shell() -> str:
    """Return the user's shell name (e.g. zsh, bash)."""
    shell = os.environ.get("SHELL", "")
    if shell:
        return Path(shell).name
    return "unknown"


def _detect_os_version() -> str:
    """Return OS type and release (e.g. Darwin 25.2.0)."""
    return f"{platform.system()} {platform.release()}"


def _is_git_repo(cwd: str) -> bool:
    """Check if cwd is inside a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd, capture_output=True, timeout=3,
        )
        return result.returncode == 0
    except Exception:
        return False


def _current_git_branch(cwd: str) -> str | None:
    """Return current branch name, or None if not in a repo or in detached HEAD."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return None
        branch = result.stdout.strip()
        # 'HEAD' means detached — not useful to show as a branch name.
        if not branch or branch == "HEAD":
            return None
        return branch
    except Exception:
        return None


def _today() -> str:
    """Today's date in ISO format (YYYY-MM-DD). Wrapped so tests can monkeypatch."""
    return date.today().isoformat()


@dataclass
class AgentContext:
    """Agent state: working directory, tools, and conversation history."""

    cwd: str
    tools: list[Tool]
    messages: list[dict[str, Any]] = field(default_factory=list)
    services: "AgentServices | None" = None

    def build_system_prompt(self) -> str:
        """Assemble the system prompt with environment info, CODER.md, and tool list."""
        parts = [
            "You are a coding agent. You help users with software engineering tasks.",
        ]

        # environment info
        is_repo = _is_git_repo(self.cwd)
        env_lines = [
            f" - Primary working directory: {self.cwd}",
            f" - Is a git repository: {'Yes' if is_repo else 'No'}",
        ]
        if is_repo:
            branch = _current_git_branch(self.cwd)
            if branch:
                env_lines.append(f" - Git branch: {branch}")
        env_lines += [
            f" - Platform: {platform.system().lower()}",
            f" - Shell: {_detect_shell()}",
            f" - OS Version: {_detect_os_version()}",
            f" - Today's date: {_today()}",
        ]
        parts.append("\n# Environment\n" + "\n".join(env_lines))

        # project instructions
        coder_md = Path(self.cwd) / "CODER.md"
        if coder_md.exists():
            parts.append(f"\n<coder_md>\n{coder_md.read_text()}\n</coder_md>")

        # available tools
        tool_names = ", ".join(t.name for t in self.tools)
        parts.append(f"\nAvailable tools: {tool_names}")

        return "\n".join(parts)
