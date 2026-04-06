from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Result of a tool execution."""
    data: str
    is_error: bool = False


@dataclass
class ToolUseBlock:
    """A tool call requested by the LLM."""
    id: str
    name: str
    input: dict[str, Any]
