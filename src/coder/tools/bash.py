import asyncio
from .base import Tool
from ..agent_types import ToolResult


class BashTool(Tool):
    name = "Bash"
    description = "Execute a bash command and return stdout+stderr."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to run"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)"},
        },
        "required": ["command"],
    }

    READ_ONLY_PREFIXES = (
        "ls", "cat", "head", "tail", "wc", "find", "grep",
        "which", "echo", "pwd", "date", "whoami",
        "git status", "git log", "git diff", "git branch",
    )

    def is_read_only(self, args: dict) -> bool:
        cmd = args.get("command", "").strip()
        return any(cmd.startswith(p) for p in self.READ_ONLY_PREFIXES)

    async def call(self, args: dict, context) -> ToolResult:
        command = args["command"]
        timeout = args.get("timeout", 120)
        cwd = context.cwd if context is not None else None
        proc = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = (stdout + stderr).decode(errors="replace").strip()
            if proc.returncode != 0:
                return ToolResult(data=output or f"Exit code {proc.returncode}", is_error=True)
            return ToolResult(data=output)
        except asyncio.TimeoutError:
            if proc is not None:
                proc.kill()
                await proc.wait()
            return ToolResult(data=f"Command timed out after {timeout}s", is_error=True)
        except Exception as e:
            return ToolResult(data=str(e), is_error=True)
