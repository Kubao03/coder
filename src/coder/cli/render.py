"""ANSI rendering helpers, tool-input summarization, and sub-agent listener."""

import os
import shutil
from ..core.events import TextDelta, ToolUseStart, ToolExecResult, TurnComplete, UsageSummary

# ---------------------------------------------------------------------------
# ANSI styles
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[34m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
GRAY = "\033[90m"
MAGENTA = "\033[35m"


def terminal_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def separator() -> str:
    return f"{GRAY}{'─' * terminal_width()}{RESET}"


def truncate(text: str, max_lines: int = 15, max_chars: int = 800) -> str:
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    else:
        truncated = False
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    result = "\n".join(lines)
    if truncated:
        result += f"\n{DIM}... (truncated){RESET}"
    return result


def print_banner(session, resumed: bool = False):
    w = terminal_width()
    line = f"{GRAY}{'─' * w}{RESET}"
    print(line)
    print(f"  {BOLD}{CYAN}Coder Agent{RESET}  {DIM}— your AI coding assistant{RESET}")
    print(f"  {DIM}cwd: {os.getcwd()}{RESET}")
    print(f"  {DIM}session: {session.session_id}{' (resumed)' if resumed else ''}{RESET}")
    print(line)
    print(f"  {DIM}Type a message to start. 'exit' to quit.{RESET}")
    print()


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def summarize_tool_input(name: str, input: dict) -> str:
    """Short one-line param preview for a tool call."""
    if name == "Bash":
        return _trunc(input.get("command", ""), 80)
    if name in ("Read", "Edit", "Write"):
        return input.get("file_path", "")
    if name == "Glob":
        return input.get("pattern", "")
    if name == "Grep":
        p = input.get("pattern", "")
        path = input.get("path", "")
        return f"{p}  {DIM}{path}{RESET}" if path else p
    if name == "Agent":
        return f"{input.get('subagent_type','?')}: {_trunc(input.get('description',''), 60)}"
    return ""


def subagent_listener(preset: str, agent_id: str, kind: str, payload: dict) -> None:
    """Render sub-agent progress to the terminal.

    - kind="start":    sub-agent dispatched (payload: {"description": ...})
    - kind="worktree": isolated worktree created (payload: {"path": ..., "branch": ...})
    - kind="tool":     sub-agent called a tool (payload: {"name": ..., "input": ...})
    - kind="end":      sub-agent finished (payload: {"error": bool})
    """
    tag = f"{CYAN}[{preset}:{agent_id[-6:]}]{RESET}"
    if kind == "start":
        desc = payload.get("description", "")
        print(f"  {tag} {DIM}▸ {desc}{RESET}")
    elif kind == "worktree":
        branch = payload.get("branch", "?")
        print(f"  {tag} {DIM}⎇ worktree on {branch}{RESET}")
    elif kind == "tool":
        name = payload.get("name", "?")
        summary = summarize_tool_input(name, payload.get("input", {}))
        print(f"  {tag} {GRAY}↳ {name}{RESET} {DIM}{summary}{RESET}")
    elif kind == "end":
        marker = f"{RED}failed{RESET}" if payload.get("error") else f"{GREEN}done{RESET}"
        print(f"  {tag} {marker}")


async def render_stream(agent, user_input: str):
    """Stream agent events to the terminal."""
    in_text = False
    async for event in agent.run_stream(user_input):
        match event:
            case TextDelta(text=text):
                if not in_text:
                    in_text = True
                print(text, end="", flush=True)

            case ToolUseStart(name=name, input=tool_input):
                if in_text:
                    print()
                    in_text = False
                summary = summarize_tool_input(name, tool_input or {})
                if summary:
                    print(f"  {MAGENTA}{name}{RESET} {DIM}{summary}{RESET}")
                else:
                    print(f"  {MAGENTA}{name}{RESET}")

            case ToolExecResult(data=data, is_error=is_error):
                color = RED if is_error else DIM
                preview = truncate(data)
                for line in preview.splitlines():
                    print(f"  {color}{line}{RESET}")

            case UsageSummary(
                turn=turn, input_tokens=inp, output_tokens=out,
                cache_read_tokens=cr, cache_write_tokens=cw, cost_usd=cost,
            ):
                if in_text:
                    print()
                    in_text = False
                parts = [f"in={inp:,}", f"out={out:,}"]
                if cr:
                    parts.append(f"cache_read={cr:,}")
                if cw:
                    parts.append(f"cache_write={cw:,}")
                parts.append(f"~${cost:.4f}")
                print(f"  {GRAY}[turn {turn}] {' '.join(parts)}{RESET}")

            case TurnComplete(text=text):
                if text.startswith("[Permission denied"):
                    print(f"\n  {YELLOW}{text}{RESET}")
                elif in_text:
                    print()
