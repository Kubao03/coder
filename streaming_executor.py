import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator

from agent_types import ToolResult, ToolUseBlock, ToolExecResult, PermissionDeniedError
from tools.base import Tool
from permissions import PermissionManager


@dataclass
class TrackedTool:
    """Internal state for a single tool invocation."""
    block: ToolUseBlock
    is_concurrent_safe: bool
    status: str = "queued"  # queued → executing → completed
    result: ToolResult | None = None
    task: asyncio.Task | None = None


class StreamingToolExecutor:
    """Execute tools as they arrive during LLM streaming.

    Concurrent-safe tools run in parallel; non-safe tools block the queue.
    """

    def __init__(self, tool_map: dict[str, Tool], pm: PermissionManager, context):
        self._tool_map = tool_map
        self._pm = pm
        self._context = context
        self._tools: list[TrackedTool] = []
        self._done_event = asyncio.Event()
        self._permission_error: PermissionDeniedError | None = None

    def add_tool(self, block: ToolUseBlock):
        tool = self._tool_map.get(block.name)
        safe = tool.is_concurrent_safe(block.input) if tool else False
        tracked = TrackedTool(block=block, is_concurrent_safe=safe)
        self._tools.append(tracked)
        self._try_execute()

    def _can_execute(self, candidate: TrackedTool) -> bool:
        executing = [t for t in self._tools if t.status == "executing"]
        if not executing:
            return True
        if candidate.is_concurrent_safe and all(t.is_concurrent_safe for t in executing):
            return True
        return False

    def _try_execute(self):
        for tracked in self._tools:
            if tracked.status != "queued":
                continue
            if self._can_execute(tracked):
                tracked.status = "executing"
                tracked.task = asyncio.create_task(self._run(tracked))
            elif not tracked.is_concurrent_safe:
                break

    async def _run(self, tracked: TrackedTool):
        tool = self._tool_map.get(tracked.block.name)
        if tool is None:
            tracked.result = ToolResult(data=f"Unknown tool: {tracked.block.name}", is_error=True)
        elif not self._pm.is_allowed(tool, tracked.block.input):
            self._permission_error = PermissionDeniedError(tracked.block.name)
            tracked.status = "completed"
            self._done_event.set()
            return
        else:
            tracked.result = await tool.call(tracked.block.input, self._context)
        tracked.status = "completed"
        self._done_event.set()
        self._try_execute()

    async def get_results(self) -> AsyncGenerator[ToolExecResult, None]:
        """Wait for all tools to complete, yielding results in order."""
        idx = 0
        while idx < len(self._tools):
            tracked = self._tools[idx]
            if tracked.status == "completed":
                if self._permission_error is not None:
                    raise self._permission_error
                yield ToolExecResult(
                    name=tracked.block.name,
                    id=tracked.block.id,
                    data=tracked.result.data,
                    is_error=tracked.result.is_error,
                )
                idx += 1
            else:
                self._done_event.clear()
                await self._done_event.wait()

    def get_tool_results(self) -> list[tuple[ToolUseBlock, ToolResult]]:
        """Return ordered (block, result) pairs after all tools complete."""
        return [(t.block, t.result) for t in self._tools]
