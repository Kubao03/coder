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
    """The LLM has started a tool call. `input` is the fully-parsed tool input."""
    name: str
    id: str
    input: dict[str, Any] = field(default_factory=dict)


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


@dataclass
class UsageSummary:
    """Token/cost summary emitted after each completed LLM call."""
    turn: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float


StreamEvent = TextDelta | ToolUseStart | ToolExecResult | TurnComplete | UsageSummary


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PermissionDeniedError(Exception):
    """Raised when the user denies a tool permission, ending the current turn."""
    pass
