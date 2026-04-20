"""Sub-agent registry: built-in agent definitions keyed by agent_type."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentDefinition:
    """Definition of a sub-agent preset.

    - agent_type: unique key (e.g. "general-purpose")
    - when_to_use: one-line summary surfaced in the AgentTool enum description
    - system_prompt: the sub-agent's system prompt (overrides parent)
    - tools: allowed tool names, or ["*"] to inherit all parent tools
      (AgentTool itself is always filtered out to prevent recursion)
    """

    agent_type: str
    when_to_use: str
    system_prompt: str
    tools: list[str]


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
    from subagents.general_purpose import GENERAL_PURPOSE
    from subagents.explore import EXPLORE
    from subagents.plan import PLAN
    register(GENERAL_PURPOSE)
    register(EXPLORE)
    register(PLAN)


_load_builtins()
