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


# ---------------------------------------------------------------------------
# Stream events yielded by AgentLoop.run_stream()
# ---------------------------------------------------------------------------

@dataclass
class TextDelta:
    """Incremental text from the LLM."""
    text: str


@dataclass
class ToolUseStart:
    """The LLM has started a tool call."""
    name: str
    id: str


@dataclass
class ToolExecResult:
    """A tool finished executing."""
    name: str
    id: str
    data: str
    is_error: bool = False


@dataclass
class TurnComplete:
    """Agent turn is complete (no more tool calls)."""
    text: str


StreamEvent = TextDelta | ToolUseStart | ToolExecResult | TurnComplete


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PermissionDeniedError(Exception):
    """Raised when the user denies a tool permission, ending the current turn."""
    pass
