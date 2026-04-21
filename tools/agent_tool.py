"""AgentTool: dispatch work to a sub-agent with a scoped tool list and its own system prompt."""

from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from tools.base import Tool
from context import AgentContext
from agent_services import AgentServices
from agent_types import ToolResult, TurnComplete, ToolUseStart
from subagents.registry import AGENT_REGISTRY, AgentDefinition, all_types
from services import worktree as wt


class AgentTool(Tool):
    """Dispatch a task to a sub-agent. The sub-agent runs an independent conversation
    with its own system prompt and a filtered tool set, then returns a final report.

    MVP tradeoffs:
    - Permissions: shared with parent (same PermissionManager instance)
    - Hooks: shared with parent
    - Streaming: sub-agent events are forwarded via context.subagent_listener
    - No fork/resume, no memory
    - Isolation: per-preset via AgentDefinition.isolation. "worktree" creates a
      sibling git worktree on a fresh branch. None means shared parent cwd.
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
        listener = context.services.subagent_listener if context.services else None
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
        # Strip any lingering settings= kwarg from older callers so we don't
        # pass an unknown field to AgentContext (which no longer has settings).
        kwargs.pop("settings", None)
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

    If the preset declares `isolation="worktree"`, the child runs in a separate
    git worktree. After completion, the worktree is removed if empty; otherwise
    its path + branch are appended to the report so the caller can review.
    """
    # Local import: AgentLoop depends on tools indirectly via type hints at
    # runtime, but importing at module load time creates a cycle with main.py's
    # tool construction path. Local import keeps the graph clean.
    from agent_loop import AgentLoop

    child_tools = _filter_tools(parent_context.tools, definition.tools)

    # Resolve cwd — default to parent, override with worktree path if isolated.
    child_cwd = parent_context.cwd
    worktree_obj: wt.Worktree | None = None
    if definition.isolation == "worktree":
        worktree_obj = _setup_worktree(parent_context.cwd, invocation_id or "run")
        child_cwd = str(worktree_obj.path)
        if listener is not None:
            _notify(
                listener, definition.agent_type, invocation_id, "worktree",
                {"path": child_cwd, "branch": worktree_obj.branch},
            )

    child_ctx = _SubagentContext(
        cwd=child_cwd,
        tools=child_tools,
        messages=[],
        system_prompt_override=definition.system_prompt,
    )

    child_services = _build_child_services(parent_context, child_cwd)

    # Sub-agent runs without its own session file (MVP — transient).
    child_loop = AgentLoop(child_ctx, child_services, session=None)

    final_text = ""
    try:
        async for event in child_loop.run_stream(prompt):
            if isinstance(event, TurnComplete):
                final_text = event.text
            elif isinstance(event, ToolUseStart) and listener is not None:
                _notify(
                    listener, definition.agent_type, invocation_id, "tool",
                    {"name": event.name, "input": event.input},
                )
    finally:
        if worktree_obj is not None:
            final_text = _finalize_worktree(worktree_obj, final_text)

    return final_text or "(sub-agent produced no final text)"


def _build_child_services(parent_context: Any, child_cwd: str) -> AgentServices:
    """Build AgentServices for a sub-agent.

    - Reuses parent's PermissionManager and settings.
    - Builds a fresh HookRunner scoped to the child cwd so shell hooks and the
      audit log land in the right directory.
    - Inherits the parent listener so the UI can track sub-agent progress.
    """
    from hooks import HookRunner, register_builtin_hooks

    parent_services = parent_context.services

    if parent_services is not None:
        hooks_config = parent_services.settings.hooks if parent_services.settings else {}
        child_hooks = HookRunner(hooks_config, cwd=child_cwd, session_id="")
        register_builtin_hooks(child_hooks)
        return AgentServices(
            permissions=parent_services.permissions,
            hooks=child_hooks,
            settings=parent_services.settings,
            subagent_listener=parent_services.subagent_listener,
        )

    # Fallback: no services on parent (unit tests, direct construction).
    from permissions import PermissionManager
    from settings import load_settings
    from hooks import HookRunner, register_builtin_hooks
    settings = load_settings(parent_context.cwd)
    pm = PermissionManager(settings, parent_context.cwd)
    hooks = HookRunner({}, cwd=child_cwd, session_id="")
    register_builtin_hooks(hooks)
    return AgentServices(permissions=pm, hooks=hooks, settings=settings)


def _setup_worktree(parent_cwd: str, tag: str) -> wt.Worktree:
    """Find git root from parent_cwd and create a worktree, or raise.

    Hard-errors when the parent cwd is not inside a git repo — isolation was
    requested by the preset, so silently falling back to shared cwd would
    defeat the whole point.
    """
    repo_root = wt.find_git_root(parent_cwd)
    if repo_root is None:
        raise wt.WorktreeError(
            f"worktree isolation requested but {parent_cwd} is not inside a git repository. "
            "Switch to a different sub-agent preset, or run inside a git repo."
        )
    return wt.create_worktree(repo_root, tag)


def _finalize_worktree(worktree_obj: wt.Worktree, report: str) -> str:
    """Remove empty worktree, or append path + branch to the report if it has changes."""
    if wt.has_changes(worktree_obj):
        note = (
            f"\n\n---\n"
            f"Sub-agent changes preserved in worktree:\n"
            f"  path:   {worktree_obj.path}\n"
            f"  branch: {worktree_obj.branch}\n"
            f"Review and merge back into the main workspace if desired."
        )
        return (report or "") + note
    wt.remove_worktree(worktree_obj)
    return report


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
