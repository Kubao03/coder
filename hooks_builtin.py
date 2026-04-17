"""Built-in hooks registered at agent startup.

Two hooks:
- dangerous_bash_guard: PreToolUse on Bash, blocks obviously destructive commands
- file_write_audit: PostToolUse on Edit/Write, appends an entry to .coder/audit.log
"""

import datetime
import re
from pathlib import Path

from hooks import HookResult, HookRunner


# ---------------------------------------------------------------------------
# Dangerous Bash guard
# ---------------------------------------------------------------------------

# Each pattern is a regex + human-readable reason.
# Patterns intentionally conservative — false positives here break the agent.
_DANGEROUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f?|--recursive).*\s+(/|~|\$HOME)(\s|$)"),
     "rm -rf on filesystem root or $HOME"),
    (re.compile(r">\s*/dev/(sd[a-z]|nvme\d|disk\d)"),
     "redirect to block device"),
    (re.compile(r"\bdd\b.*\bof=/dev/(sd[a-z]|nvme\d|disk\d)"),
     "dd writing to block device"),
    (re.compile(r"\bmkfs\.[a-z0-9]+\b"),
     "filesystem format command"),
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}"),
     "fork bomb"),
    (re.compile(r"\bchmod\s+-R\s+[0-7]*777\s+/"),
     "recursive chmod 777 on root"),
]


def dangerous_bash_guard(tool_name: str, tool_input: dict, cwd: str) -> HookResult | None:
    """Block obviously destructive Bash commands before they run."""
    if tool_name != "Bash":
        return None
    command = tool_input.get("command", "")
    if not isinstance(command, str):
        return None
    for pattern, reason in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return HookResult(
                blocked=True,
                block_reason=(
                    f"Destructive command blocked by built-in safety hook: {reason}. "
                    f"If you really need to run this, ask the user to disable the guard."
                ),
            )
    return None


# ---------------------------------------------------------------------------
# File write audit log
# ---------------------------------------------------------------------------

def file_write_audit(
    tool_name: str, tool_input: dict, tool_output: str, cwd: str
) -> str | None:
    """Record Edit/Write operations to .coder/audit.log."""
    if tool_name not in ("Edit", "Write"):
        return None

    path = tool_input.get("file_path", "<unknown>")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    audit_dir = Path(cwd) / ".coder"
    try:
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_file = audit_dir / "audit.log"
        with audit_file.open("a", encoding="utf-8") as f:
            f.write(f"{ts}\t{tool_name}\t{path}\n")
    except OSError:
        # Never let an audit failure surface to the agent
        return None

    return None  # no message into the tool output


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_builtin_hooks(runner: HookRunner) -> None:
    """Attach the default built-in hooks to a HookRunner instance."""
    runner.register_pre_callback("Bash", dangerous_bash_guard)
    runner.register_post_callback("Edit|Write", file_write_audit)
