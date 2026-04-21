import re
from pathlib import Path
from .base import Tool
from .base import ToolResult


class GrepTool(Tool):
    name = "Grep"
    description = "Search for a regex pattern in a file or directory."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "path": {"type": "string", "description": "File or directory to search"},
            "glob": {"type": "string", "description": "File glob filter when path is a directory, e.g. '*.py'"},
        },
        "required": ["pattern", "path"],
    }

    def is_read_only(self, args: dict) -> bool:
        return True

    async def call(self, args: dict, context) -> ToolResult:
        try:
            regex = re.compile(args["pattern"])
        except re.error as e:
            return ToolResult(data=f"Invalid regex: {e}", is_error=True)

        target = Path(args["path"])
        glob_filter = args.get("glob", "*")

        files = [target] if target.is_file() else sorted(target.rglob(glob_filter))
        matches = []
        for f in files:
            if not f.is_file():
                continue
            try:
                for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                    if regex.search(line):
                        matches.append(f"{f}:{i}: {line}")
            except Exception:
                continue

        if not matches:
            return ToolResult(data="No matches found")
        return ToolResult(data="\n".join(matches))
