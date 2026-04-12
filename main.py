import argparse
import asyncio
import os
import shutil
from context import AgentContext
from permissions import PermissionManager
from agent_loop import AgentLoop
from agent_types import TextDelta, ToolUseStart, ToolExecResult, TurnComplete
from session import SessionManager
from services.compact import (
    auto_compact, estimate_tokens, compact_conversation,
    COMPACT_USER_PREFIX, _calculate_keep_index, _is_tool_result,
)
from tools.bash import BashTool
from tools.file_read import FileReadTool
from tools.file_edit import FileEditTool
from tools.file_write import FileWriteTool
from tools.glob_tool import GlobTool
from tools.grep_tool import GrepTool

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


def print_banner(session: SessionManager, resumed: bool = False):
    w = terminal_width()
    line = f"{GRAY}{'─' * w}{RESET}"
    print(line)
    print(f"  {BOLD}{CYAN}Coder Agent{RESET}  {DIM}— your AI coding assistant{RESET}")
    print(f"  {DIM}cwd: {os.getcwd()}{RESET}")
    print(f"  {DIM}session: {session.session_id}{' (resumed)' if resumed else ''}{RESET}")
    print(line)
    print(f"  {DIM}Type a message to start. 'exit' to quit.{RESET}")
    print()


def make_agent(cwd: str, session: SessionManager) -> AgentLoop:
    tools = [BashTool(), FileReadTool(), FileEditTool(), FileWriteTool(), GlobTool(), GrepTool()]
    ctx = AgentContext(cwd=cwd, tools=tools)
    pm = PermissionManager()
    return AgentLoop(ctx, pm, session=session)


async def handle_message(agent: AgentLoop, user_input: str):
    in_text = False
    async for event in agent.run_stream(user_input):
        match event:
            case TextDelta(text=text):
                if not in_text:
                    in_text = True
                print(text, end="", flush=True)

            case ToolUseStart(name=name):
                if in_text:
                    print()
                    in_text = False
                print(f"  {MAGENTA}{name}{RESET}")

            case ToolExecResult(data=data, is_error=is_error):
                color = RED if is_error else DIM
                preview = truncate(data)
                for line in preview.splitlines():
                    print(f"  {color}{line}{RESET}")

            case TurnComplete():
                if in_text:
                    print()


async def handle_slash(agent: AgentLoop, cmd: str):
    parts = cmd.strip().split()
    name = parts[0].lower()

    if name == "/compact":
        ctx = agent.context
        tokens = estimate_tokens(ctx.messages, ctx.build_system_prompt())
        print(f"  {DIM}Current tokens (est): {tokens}{RESET}")
        print(f"  {DIM}Compacting...{RESET}")
        try:
            keep_idx = _calculate_keep_index(ctx.messages)
            to_summarize = ctx.messages[:keep_idx]
            to_keep = ctx.messages[keep_idx:]
            if not to_summarize:
                to_summarize = ctx.messages
                to_keep = []
            summary = await compact_conversation(
                agent.client, agent.model, to_summarize, ctx.build_system_prompt(),
            )
            while to_keep and _is_tool_result(to_keep[0]):
                to_keep.pop(0)
            ctx.messages = [{"role": "user", "content": COMPACT_USER_PREFIX + summary}] + to_keep
            new_tokens = estimate_tokens(ctx.messages, ctx.build_system_prompt())
            print(f"  {GREEN}Compacted: {tokens} -> {new_tokens} tokens{RESET}")
        except Exception as e:
            print(f"  {RED}Compact failed: {e}{RESET}")

    elif name == "/tokens":
        ctx = agent.context
        tokens = estimate_tokens(ctx.messages, ctx.build_system_prompt())
        print(f"  {DIM}Messages: {len(ctx.messages)}, Tokens (est): {tokens}{RESET}")

    elif name == "/clear":
        agent.context.messages.clear()
        print(f"  {DIM}Conversation cleared.{RESET}")

    elif name == "/sessions":
        cwd = os.getcwd()
        sessions = SessionManager.list_sessions(cwd=cwd)
        if not sessions:
            print(f"  {DIM}No sessions found.{RESET}")
        else:
            for s in sessions[:10]:
                ts = s["modified"].strftime("%m-%d %H:%M")
                print(f"  {DIM}{s['id']}  {ts}  {s['preview']}{RESET}")

    elif name == "/help":
        print(f"  {DIM}/compact   — Compress conversation via LLM summary{RESET}")
        print(f"  {DIM}/tokens    — Show estimated token count{RESET}")
        print(f"  {DIM}/sessions  — List recent sessions{RESET}")
        print(f"  {DIM}/clear     — Clear conversation history{RESET}")
        print(f"  {DIM}/help      — Show this help{RESET}")

    else:
        print(f"  {YELLOW}Unknown command: {name}. Type /help for commands.{RESET}")


def read_input() -> str:
    """Read user input, supporting multi-line with trailing backslash."""
    line = input(f"{BOLD}{GREEN}>{RESET} ")
    lines = []
    while line.endswith("\\"):
        lines.append(line[:-1])
        line = input(f"{DIM}.{RESET} ")
    lines.append(line)
    return "\n".join(lines).strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Coder Agent")
    parser.add_argument("--resume", nargs="?", const="__latest__", default=None,
                        metavar="SESSION_ID",
                        help="Resume a session (latest if no ID given)")
    parser.add_argument("--list", action="store_true",
                        help="List recent sessions and exit")
    return parser.parse_args()


async def repl():
    args = parse_args()
    cwd = os.getcwd()

    if args.list:
        sessions = SessionManager.list_sessions(cwd=cwd)
        if not sessions:
            print(f"{DIM}No sessions found.{RESET}")
        else:
            for s in sessions[:20]:
                ts = s["modified"].strftime("%Y-%m-%d %H:%M")
                print(f"  {s['id']}  {ts}  {s['preview']}")
        return

    resumed = False
    if args.resume:
        if args.resume == "__latest__":
            session = SessionManager.find_latest(cwd)
            if not session:
                print(f"{YELLOW}No session to resume.{RESET}")
                return
        else:
            session = SessionManager(session_id=args.resume, cwd=cwd)
            if not session.path.exists():
                print(f"{RED}Session not found: {args.resume}{RESET}")
                return
        resumed = True
    else:
        session = SessionManager(cwd=cwd)

    agent = make_agent(cwd, session)

    if resumed:
        agent.context.messages = session.load()
        msg_count = len(agent.context.messages)
        print(f"  {GREEN}Restored {msg_count} messages from session {session.session_id}{RESET}")

    print_banner(session, resumed=resumed)

    while True:
        try:
            user_input = read_input()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print(f"{DIM}Goodbye.{RESET}")
            break

        if user_input.startswith("/"):
            await handle_slash(agent, user_input)
            print(separator())
            continue

        print()
        try:
            await handle_message(agent, user_input)
            print()
        except KeyboardInterrupt:
            print(f"\n{YELLOW}[interrupted]{RESET}\n")
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}\n")
        print(separator())


if __name__ == "__main__":
    asyncio.run(repl())
