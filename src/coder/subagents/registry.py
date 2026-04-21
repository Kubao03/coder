"""Sub-agent registry: built-in agent definitions keyed by agent_type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AgentDefinition:
    """Definition of a sub-agent preset.

    - agent_type: unique key (e.g. "general-purpose")
    - when_to_use: one-line summary surfaced in the AgentTool enum description
    - system_prompt: the sub-agent's system prompt (overrides parent)
    - tools: allowed tool names, or ["*"] to inherit all parent tools
      (AgentTool itself is always filtered out to prevent recursion)
    - isolation: if "worktree", the sub-agent runs in a fresh git worktree on
      a new branch. Changes stay in that worktree unless the caller merges
      them back. None (default) means the sub-agent shares the parent's cwd.
    """

    agent_type: str
    when_to_use: str
    system_prompt: str
    tools: list[str]
    isolation: Literal["worktree"] | None = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AGENT_REGISTRY: dict[str, AgentDefinition] = {}


def register(definition: AgentDefinition) -> None:
    AGENT_REGISTRY[definition.agent_type] = definition


def get(agent_type: str) -> AgentDefinition | None:
    return AGENT_REGISTRY.get(agent_type)


def all_types() -> list[str]:
    return list(AGENT_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Load built-ins
# ---------------------------------------------------------------------------

def _load_builtins() -> None:
    from .general_purpose import GENERAL_PURPOSE
    from .explore import EXPLORE
    from .plan import PLAN
    register(GENERAL_PURPOSE)
    register(EXPLORE)
    register(PLAN)


_load_builtins()
