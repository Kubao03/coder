"""Hook runner: execute PreToolUse / PostToolUse shell hooks from settings."""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class HookResult:
    """Outcome of running a set of hooks for a single event."""
    blocked: bool = False
    block_reason: str = ""
    output: str = ""  # combined stdout from all hooks


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def _matches(tool_name: str, matcher: str) -> bool:
    """Return True if tool_name matches the hook matcher pattern.

    - empty / "*": matches all tools
    - pipe-separated names: "Edit|Write" matches either
    - otherwise treated as regex
    """
    if not matcher or matcher == "*":
        return True
    if re.fullmatch(r"[A-Za-z0-9_|]+", matcher):
        return tool_name in {m.strip() for m in matcher.split("|")}
    try:
        return bool(re.search(matcher, tool_name))
    except re.error:
        return False


# ---------------------------------------------------------------------------
# HookRunner
# ---------------------------------------------------------------------------

class HookRunner:
    """Run configured shell hooks for tool events."""

    def __init__(self, hooks_config: dict[str, Any], cwd: str, session_id: str = ""):
        self._config = hooks_config  # {"PreToolUse": [...], "PostToolUse": [...]}
        self._cwd = cwd
        self._session_id = session_id

    def _get_matching_commands(self, event: str, tool_name: str) -> list[str]:
        """Return ordered list of shell commands that match tool_name for event."""
        matchers = self._config.get(event, [])
        commands: list[str] = []
        for entry in matchers:
            matcher = entry.get("matcher", "")
            if not _matches(tool_name, matcher):
                continue
            for hook in entry.get("hooks", []):
                if hook.get("type") == "command" and hook.get("command"):
                    commands.append(hook["command"])
        return commands

    async def _run_command(
        self,
        command: str,
        stdin_data: dict[str, Any],
        timeout: float = 30.0,
    ) -> tuple[int, str, str]:
        """Run a shell command with JSON on stdin. Returns (returncode, stdout, stderr)."""
        stdin_bytes = json.dumps(stdin_data).encode()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(stdin_bytes), timeout=timeout
            )
            return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return 1, "", f"Hook timed out after {timeout}s"
        except Exception as e:
            return 1, "", str(e)

    async def run_pre_tool(self, tool_name: str, tool_input: dict[str, Any]) -> HookResult:
        """Run PreToolUse hooks. Returns HookResult with blocked=True if any hook blocks."""
        commands = self._get_matching_commands("PreToolUse", tool_name)
        if not commands:
            return HookResult()

        stdin_data = {
            "session_id": self._session_id,
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "cwd": self._cwd,
        }

        output_parts: list[str] = []
        for command in commands:
            rc, stdout, stderr = await self._run_command(command, stdin_data)

            if stderr.strip():
                output_parts.append(stderr.strip())

            # exit code 2 → block unconditionally
            if rc == 2:
                reason = stdout.strip() or stderr.strip() or f"Hook exited with code 2"
                return HookResult(blocked=True, block_reason=reason, output="\n".join(output_parts))

            # Try to parse JSON from stdout
            if stdout.strip():
                try:
                    resp = json.loads(stdout.strip())
                    if resp.get("decision") == "block":
                        reason = resp.get("reason", "Blocked by PreToolUse hook")
                        return HookResult(blocked=True, block_reason=reason, output="\n".join(output_parts))
                    if stdout.strip():
                        output_parts.append(stdout.strip())
                except json.JSONDecodeError:
                    # not JSON — treat as plain output
                    output_parts.append(stdout.strip())

        return HookResult(output="\n".join(output_parts))

    async def run_post_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
    ) -> HookResult:
        """Run PostToolUse hooks. Blocking is ignored; output is returned for display."""
        commands = self._get_matching_commands("PostToolUse", tool_name)
        if not commands:
            return HookResult()

        stdin_data = {
            "session_id": self._session_id,
            "hook_event_name": "PostToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_output,
            "cwd": self._cwd,
        }

        output_parts: list[str] = []
        for command in commands:
            rc, stdout, stderr = await self._run_command(command, stdin_data)
            if stdout.strip():
                output_parts.append(stdout.strip())
            if stderr.strip():
                output_parts.append(stderr.strip())

        return HookResult(output="\n".join(output_parts))
