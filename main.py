import asyncio
import os
import sys
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

# --- ANSI colors ---
BLUE = "\033[34m"
GRAY = "\033[90m"
RED = "\033[31m"
RESET = "\033[0m"
BOLD = "\033[1m"


def make_agent(cwd: str) -> AgentLoop:
    tools = [BashTool(), FileReadTool(), FileEditTool(), FileWriteTool(), GlobTool(), GrepTool()]
    ctx = AgentContext(cwd=cwd, tools=tools)
    pm = PermissionManager()
    return AgentLoop(ctx, pm)


async def handle_message(agent: AgentLoop, user_input: str):
    """Stream agent response to terminal."""
    in_text = False
    async for event in agent.run_stream(user_input):
        match event:
            case TextDelta(text=text):
                if not in_text:
                    in_text = True
                print(text, end="", flush=True)

            case ToolUseStart(name=name, id=tid):
                if in_text:
                    print()
                    in_text = False
                print(f"{BLUE}{BOLD}[{name}]{RESET}", flush=True)

            case ToolExecResult(name=name, data=data, is_error=is_error):
                color = RED if is_error else GRAY
                # Show truncated result
                preview = data[:500] + ("..." if len(data) > 500 else "")
                print(f"{color}{preview}{RESET}", flush=True)

            case TurnComplete():
                if in_text:
                    print()


async def repl():
    cwd = os.getcwd()
    agent = make_agent(cwd)
    print(f"{BOLD}Code Agent ready.{RESET} cwd: {cwd}")
    print("Type your message, or 'exit' to quit.\n")

    while True:
        try:
            user_input = input(f"{BOLD}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye.")
            break

        try:
            await handle_message(agent, user_input)
            print()
        except KeyboardInterrupt:
            print("\n[interrupted]")
        except Exception as e:
            print(f"{RED}[error] {e}{RESET}")


if __name__ == "__main__":
    asyncio.run(repl())
