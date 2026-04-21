from pathlib import Path
from .base import Tool
from .base import ToolResult

# System directories that should never be written to by the agent.
_BLOCKED_PREFIXES = ("/etc", "/usr", "/bin", "/sbin", "/lib", "/boot", "/sys", "/proc")


class FileWriteTool(Tool):
    name = "Write"
    description = "Write content to a file, overwriting if it exists. Creates parent directories as needed."
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["file_path", "content"],
    }

    async def call(self, args: dict, context) -> ToolResult:
        path = Path(args["file_path"])
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path

        # Refuse writes to critical system directories regardless of permissions.
        resolved_str = str(resolved)
        for prefix in _BLOCKED_PREFIXES:
            if resolved_str == prefix or resolved_str.startswith(prefix + "/"):
                return ToolResult(
                    data=f"Write refused: {resolved} is inside a protected system directory.",
                    is_error=True,
                )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["content"])
            return ToolResult(data="OK")
        except Exception as e:
            return ToolResult(data=str(e), is_error=True)
