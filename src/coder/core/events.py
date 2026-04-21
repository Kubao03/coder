"""Stream events yielded by AgentLoop.run_stream()."""

from dataclasses import dataclass, field
from typing import Any


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
