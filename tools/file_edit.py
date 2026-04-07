from pathlib import Path
from tools.base import Tool
from agent_types import ToolResult


class FileEditTool(Tool):
    name = "Edit"
    description = "Replace an exact string in a file. old_string must appear exactly once."
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    async def call(self, args: dict, context) -> ToolResult:
        path = Path(args["file_path"])
        if not path.exists():
            return ToolResult(data=f"File not found: {path}", is_error=True)

        content = path.read_text(errors="replace")
        old = args["old_string"]
        new = args["new_string"]

        count = content.count(old)
        if count == 0:
            return ToolResult(data=f"old_string not found in {path}", is_error=True)
        if count > 1:
            return ToolResult(data=f"old_string found {count} times, must be unique", is_error=True)

        path.write_text(content.replace(old, new, 1))
        return ToolResult(data="OK")
