"""Token/cost tracking for the agent session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Pricing table  (USD per 1M tokens)
# ---------------------------------------------------------------------------

_PRICING: dict[str, dict[str, float]] = {
    # input, output, cache_read, cache_write
    "claude-opus-4-7":   {"input": 5.00, "output": 25.00, "cache_read": 0.50,  "cache_write": 6.25},
    "claude-opus-4-6":   {"input": 5.00, "output": 25.00, "cache_read": 0.50,  "cache_write": 6.25},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_read": 0.30,  "cache_write": 3.75},
    "claude-haiku-4-5":  {"input": 1.00, "output":  5.00, "cache_read": 0.10,  "cache_write": 1.25},
    # DeepSeek (via Anthropic-compatible API)
    "deepseek-chat":     {"input": 0.27, "output":  1.10, "cache_read": 0.07,  "cache_write": 0.27},
    "deepseek-reasoner": {"input": 0.55, "output":  2.19, "cache_read": 0.14,  "cache_write": 0.55},
}

_DEFAULT_PRICING = {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25}


def _price(model: str, kind: str, tokens: int) -> float:
    """Return cost in USD for `tokens` tokens of `kind` for `model`."""
    rates = _PRICING.get(model, _DEFAULT_PRICING)
    return rates.get(kind, 0.0) * tokens / 1_000_000


# ---------------------------------------------------------------------------
# Per-turn snapshot
# ---------------------------------------------------------------------------

@dataclass
class TurnUsage:
    """Token counts and cost for a single LLM call."""
    turn: int
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return (
            _price(self.model, "input", self.input_tokens)
            + _price(self.model, "output", self.output_tokens)
            + _price(self.model, "cache_read", self.cache_read_tokens)
            + _price(self.model, "cache_write", self.cache_write_tokens)
        )


# ---------------------------------------------------------------------------
# Session-level accumulator
# ---------------------------------------------------------------------------

@dataclass
class UsageTracker:
    """Accumulates token usage and cost across all turns in a session."""

    turns: list[TurnUsage] = field(default_factory=list)

    def record(self, model: str, usage: Any) -> TurnUsage:
        """Record usage from an Anthropic API `usage` object; return a TurnUsage snapshot."""
        turn_num = len(self.turns) + 1
        t = TurnUsage(
            turn=turn_num,
            model=model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
        self.turns.append(t)
        return t

    # ------------------------------------------------------------------
    # Cumulative aggregates
    # ------------------------------------------------------------------

    @property
    def total_input(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def total_cache_read(self) -> int:
        return sum(t.cache_read_tokens for t in self.turns)

    @property
    def total_cache_write(self) -> int:
        return sum(t.cache_write_tokens for t in self.turns)

    @property
    def total_cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.turns)

    @property
    def turn_count(self) -> int:
        return len(self.turns)
