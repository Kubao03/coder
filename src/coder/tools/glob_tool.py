from pathlib import Path
from .base import Tool
from .base import ToolResult


class GlobTool(Tool):
    name = "Glob"
    description = "Find files matching a glob pattern."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'"},
            "path": {"type": "string", "description": "Directory to search in (default: cwd)"},
        },
        "required": ["pattern"],
    }

    def is_read_only(self, args: dict) -> bool:
        return True

    async def call(self, args: dict, context) -> ToolResult:
        base = Path(args.get("path") or (context.cwd if context else "."))
        matches = sorted(str(p) for p in base.glob(args["pattern"]))
        if not matches:
            return ToolResult(data="No files found")
        return ToolResult(data="\n".join(matches))
