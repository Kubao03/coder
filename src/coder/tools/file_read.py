from pathlib import Path
from .base import Tool
from .base import ToolResult


class FileReadTool(Tool):
    name = "Read"
    description = "Read a file. Optionally specify offset (start line) and limit (number of lines)."
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Absolute path to the file"},
            "offset": {"type": "integer", "description": "Line number to start from (1-indexed)"},
            "limit": {"type": "integer", "description": "Number of lines to read"},
        },
        "required": ["file_path"],
    }

    def is_read_only(self, args: dict) -> bool:
        return True

    async def call(self, args: dict, context) -> ToolResult:
        path = Path(args["file_path"])
        if not path.exists():
            return ToolResult(data=f"File not found: {path}", is_error=True)
        if not path.is_file():
            return ToolResult(data=f"Not a file: {path}", is_error=True)
        try:
            lines = path.read_text(errors="replace").splitlines()
            offset = args.get("offset", 1)
            limit = args.get("limit")
            start = max(0, offset - 1)
            end = (start + limit) if limit else len(lines)
            selected = lines[start:end]
            content = "\n".join(
                f"{start + i + 1}\t{line}" for i, line in enumerate(selected)
            )
            return ToolResult(data=content)
        except Exception as e:
            return ToolResult(data=str(e), is_error=True)
