from pathlib import Path
from tools.base import Tool
from agent_types import ToolResult


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
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["content"])
            return ToolResult(data="OK")
        except Exception as e:
            return ToolResult(data=str(e), is_error=True)
