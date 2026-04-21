"""Main REPL loop, agent factory, and CLI entry point."""

import argparse
import asyncio
import os
from dotenv import load_dotenv

from .render import (
    BOLD, DIM, GREEN, RED, YELLOW, RESET,
    print_banner, separator, subagent_listener, render_stream,
)
from .commands import handle_slash
from ..logging_config import setup_logging
from ..core.context import AgentContext
from ..core.services import AgentServices
from ..core.agent_loop import AgentLoop
from ..permissions import PermissionManager
from ..persistence.settings import load_settings
from ..persistence.session import SessionManager
from ..usage import UsageTracker
from ..hooks import HookRunner, register_builtin_hooks
from ..tools.bash import BashTool
from ..tools.file_read import FileReadTool
from ..tools.file_edit import FileEditTool
from ..tools.file_write import FileWriteTool
from ..tools.glob_tool import GlobTool
from ..tools.grep_tool import GrepTool
from ..tools.agent import AgentTool


def make_agent(cwd: str, session: SessionManager) -> AgentLoop:
    tools = [
        BashTool(), FileReadTool(), FileEditTool(), FileWriteTool(),
        GlobTool(), GrepTool(), AgentTool(),
    ]
    settings = load_settings(cwd)
    pm = PermissionManager(settings, cwd)
    hooks = HookRunner(settings.hooks, cwd=cwd, session_id=session.session_id)
    register_builtin_hooks(hooks)
    services = AgentServices(
        permissions=pm,
        hooks=hooks,
        settings=settings,
        subagent_listener=subagent_listener,
        usage=UsageTracker(),
    )
    ctx = AgentContext(cwd=cwd, tools=tools)
    return AgentLoop(ctx, services, session=session)


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
    load_dotenv()
    setup_logging()
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

    setup_logging(session.session_id)
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
            await render_stream(agent, user_input)
            print()
        except KeyboardInterrupt:
            print(f"\n{YELLOW}[interrupted]{RESET}\n")
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}\n")
        print(separator())


def main_entry() -> None:
    """Entry point for the `coder` CLI command (installed via pip/pipx)."""
    asyncio.run(repl())
