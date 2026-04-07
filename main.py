import asyncio
import os
from context import AgentContext
from permissions import PermissionManager
from agent_loop import AgentLoop
from tools.bash import BashTool
from tools.file_read import FileReadTool
from tools.file_edit import FileEditTool
from tools.file_write import FileWriteTool
from tools.glob_tool import GlobTool
from tools.grep_tool import GrepTool


def make_agent(cwd: str) -> AgentLoop:
    tools = [BashTool(), FileReadTool(), FileEditTool(), FileWriteTool(), GlobTool(), GrepTool()]
    ctx = AgentContext(cwd=cwd, tools=tools)
    pm = PermissionManager()
    return AgentLoop(ctx, pm)


async def repl():
    cwd = os.getcwd()
    agent = make_agent(cwd)
    print(f"Code Agent ready. cwd: {cwd}")
    print("Type your message, or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye.")
            break

        try:
            response = await agent.run(user_input)
            print(f"\n{response}\n")
        except KeyboardInterrupt:
            print("\n[interrupted]")
        except Exception as e:
            print(f"[error] {e}")


if __name__ == "__main__":
    asyncio.run(repl())
