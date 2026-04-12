import asyncio
import os
import shutil
from context import AgentContext
from permissions import PermissionManager
from agent_loop import AgentLoop
from agent_types import TextDelta, ToolUseStart, ToolExecResult, TurnComplete
from tools.bash import BashTool
from tools.file_read import FileReadTool
from tools.file_edit import FileEditTool
from tools.file_write import FileWriteTool
from tools.glob_tool import GlobTool
from tools.grep_tool import GrepTool

# --- ANSI styles ---
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


def print_banner():
    w = terminal_width()
    line = f"{GRAY}{'─' * w}{RESET}"
    print(line)
    print(f"  {BOLD}{CYAN}Coder Agent{RESET}  {DIM}— your AI coding assistant{RESET}")
    print(f"  {DIM}cwd: {os.getcwd()}{RESET}")
    print(line)
    print(f"  {DIM}Type a message to start. 'exit' to quit.{RESET}")
    print()


def make_agent(cwd: str) -> AgentLoop:
    tools = [BashTool(), FileReadTool(), FileEditTool(), FileWriteTool(), GlobTool(), GrepTool()]
    ctx = AgentContext(cwd=cwd, tools=tools)
    pm = PermissionManager()
    return AgentLoop(ctx, pm)


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


async def repl():
    cwd = os.getcwd()
    agent = make_agent(cwd)
    print_banner()

    while True:
        try:
            user_input = input(f"{BOLD}{GREEN}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print(f"{DIM}Goodbye.{RESET}")
            break

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
