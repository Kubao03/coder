"""AgentTool: dispatch work to a sub-agent with a scoped tool list and its own system prompt."""

from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from tools.base import Tool
from context import AgentContext
from agent_types import ToolResult, TurnComplete, ToolUseStart
from subagents.registry import AGENT_REGISTRY, AgentDefinition, all_types


class AgentTool(Tool):
    """Dispatch a task to a sub-agent. The sub-agent runs an independent conversation
    with its own system prompt and a filtered tool set, then returns a final report.

    MVP tradeoffs:
    - Permissions: shared with parent (same PermissionManager instance)
    - Hooks: shared with parent
    - Streaming: silent — sub-agent events are not forwarded to the terminal
    - No fork/resume, no memory, no worktree isolation (deferred to 3.2)
    - AgentTool is always filtered out of the sub-agent's tool list (prevents recursion)
    """

    name = "Agent"

    @property
    def description(self) -> str:
        lines = [
            "Dispatch a task to a specialized sub-agent. Use this for open-ended research, "
            "multi-step exploration, or work that doesn't need to stay in the main conversation.",
            "",
            "Available agent types:",
        ]
        for t in all_types():
            d = AGENT_REGISTRY[t]
            lines.append(f"- {t}: {d.when_to_use}")
        lines.append("")
        lines.append(
            "The sub-agent receives only the `prompt` you pass — it has no memory of this "
            "conversation, so include everything it needs: file paths, what you've tried, "
            "and what form the answer should take."
        )
        return "\n".join(lines)

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short (3-5 word) summary of the task, shown to the user.",
                },
                "prompt": {
                    "type": "string",
                    "description": "The task for the sub-agent. Must be self-contained.",
                },
                "subagent_type": {
                    "type": "string",
                    "description": f"Which sub-agent preset to use. One of: {', '.join(all_types())}.",
                    "enum": all_types(),
                },
            },
            "required": ["description", "prompt"],
        }

    def is_read_only(self, args: dict[str, Any]) -> bool:
        return False

    def is_concurrent_safe(self, args: dict[str, Any]) -> bool:
        # Two Agent calls dispatched in the same turn are independent asyncio
        # tasks with their own AgentLoop, messages, and executor — they don't
        # share mutable state at this level. Any concurrency risk (e.g. two
        # sub-agents editing the same file) is owned by the sub-agent's own
        # permission checks, not by this tool's scheduler hint. Parallelism
        # here is a ~Nx speedup for multi-agent fan-out, so we opt in.
        return True

    async def call(self, args: dict[str, Any], context: Any) -> ToolResult:
        prompt = args.get("prompt", "").strip()
        agent_type = args.get("subagent_type", "general-purpose")
        description = args.get("description", "")

        if not prompt:
            return ToolResult(data="AgentTool error: `prompt` is required.", is_error=True)

        definition = AGENT_REGISTRY.get(agent_type)
        if definition is None:
            available = ", ".join(all_types())
            return ToolResult(
                data=f"AgentTool error: unknown subagent_type '{agent_type}'. Available: {available}",
                is_error=True,
            )

        # Each AgentTool invocation gets its own id for display attribution
        # (multiple sub-agents running in parallel interleave their lines).
        invocation_id = uuid4().hex
        listener = getattr(context, "subagent_listener", None)
        _notify(listener, agent_type, invocation_id, "start", {"description": description})

        try:
            report = await _run_subagent(definition, prompt, context, invocation_id, listener)
        except Exception as e:
            _notify(listener, agent_type, invocation_id, "end", {"error": True})
            return ToolResult(data=f"Sub-agent failed: {e}", is_error=True)

        _notify(listener, agent_type, invocation_id, "end", {"error": False})
        return ToolResult(data=report)


# ---------------------------------------------------------------------------
# Sub-agent context
# ---------------------------------------------------------------------------

class _SubagentContext(AgentContext):
    """AgentContext variant that returns the sub-agent's preset system prompt.

    Subclasses AgentContext so it's a drop-in replacement for AgentLoop (which
    calls `build_system_prompt()`). The override keeps the sub-agent focused —
    no env details, no CODER.md, no tool list dump.
    """

    def __init__(self, *, system_prompt_override: str, **kwargs):
        super().__init__(**kwargs)
        self._system_prompt_override = system_prompt_override

    def build_system_prompt(self) -> str:
        return self._system_prompt_override


# ---------------------------------------------------------------------------
# Sub-agent execution
# ---------------------------------------------------------------------------

async def _run_subagent(
    definition: AgentDefinition,
    prompt: str,
    parent_context: Any,
    invocation_id: str = "",
    listener: Callable | None = None,
) -> str:
    """Build a child AgentContext + AgentLoop, run to completion, return final text.

    If `listener` is provided, the sub-agent's tool-use events are forwarded to
    it so the UI can render progress (preset tag + tool name + brief args).
    """
    # Local import: AgentLoop depends on tools indirectly via type hints at
    # runtime, but importing at module load time creates a cycle with main.py's
    # tool construction path. Local import keeps the graph clean.
    from agent_loop import AgentLoop

    child_tools = _filter_tools(parent_context.tools, definition.tools)

    child_ctx = _SubagentContext(
        cwd=parent_context.cwd,
        tools=child_tools,
        settings=parent_context.settings,
        messages=[],
        system_prompt_override=definition.system_prompt,
    )

    # Parent permission manager is attached to the context by AgentLoop as `_pm`.
    # Falls back to a fresh instance if not present (e.g. tests).
    pm = getattr(parent_context, "_pm", None)
    if pm is None:
        from permissions import PermissionManager
        pm = PermissionManager(parent_context.settings, parent_context.cwd)

    # Sub-agent runs without its own session file (MVP — transient).
    child_loop = AgentLoop(child_ctx, pm, session=None)

    final_text = ""
    async for event in child_loop.run_stream(prompt):
        if isinstance(event, TurnComplete):
            final_text = event.text
        elif isinstance(event, ToolUseStart) and listener is not None:
            _notify(
                listener, definition.agent_type, invocation_id, "tool",
                {"name": event.name, "input": event.input},
            )

    return final_text or "(sub-agent produced no final text)"


def _notify(
    listener: Callable | None, preset: str, invocation_id: str, kind: str, payload: dict,
) -> None:
    """Fire a sub-agent progress event; swallow listener errors so they don't crash execution."""
    if listener is None:
        return
    try:
        listener(preset, invocation_id, kind, payload)
    except Exception:
        pass


def _filter_tools(parent_tools: list[Tool], allowed: list[str]) -> list[Tool]:
    """Filter parent tools for the sub-agent.

    - AgentTool itself is always removed (prevents recursion).
    - If `allowed == ["*"]`, all remaining parent tools are inherited.
    - Otherwise, only tools whose `name` appears in `allowed` are kept.
    """
    no_recursion = [t for t in parent_tools if t.name != "Agent"]
    if allowed == ["*"]:
        return no_recursion
    allowed_set = set(allowed)
    return [t for t in no_recursion if t.name in allowed_set]
