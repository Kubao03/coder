import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from coder.core.streaming import StreamingToolExecutor, TrackedTool
from coder.agent_types import ToolResult, ToolUseBlock, ToolExecResult, PermissionDeniedError
from coder.tools.base import Tool


# --- Helpers ---

class FakeTool(Tool):
    def __init__(self, tool_name: str, read_only: bool = True, delay: float = 0):
        self._name = tool_name
        self._read_only = read_only
        self._delay = delay
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Fake {self._name}"

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    def is_read_only(self, args: dict) -> bool:
        return self._read_only

    async def call(self, args: dict, context) -> ToolResult:
        self.call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        return ToolResult(data=f"{self._name} result")


class FakePermissionManager:
    def __init__(self, deny_names: set[str] | None = None):
        self._deny = deny_names or set()

    def is_allowed(self, tool, args):
        return tool.name not in self._deny


class FakeServices:
    """Minimal services stub for executor tests — no real hooks or settings needed."""

    def __init__(self, deny_names: set[str] | None = None):
        self.permissions = FakePermissionManager(deny_names)
        self.hooks = None  # executor falls back gracefully when hooks is None


def make_block(name: str, tool_id: str, input_: dict | None = None) -> ToolUseBlock:
    return ToolUseBlock(id=tool_id, name=name, input=input_ or {})


# --- Tests ---

class TestStreamingExecutor:
    @pytest.mark.asyncio
    async def test_single_tool_executes(self):
        read = FakeTool("Read", read_only=True)
        executor = StreamingToolExecutor({"Read": read}, FakeServices(), None)

        executor.add_tool(make_block("Read", "t1", {"file_path": "/a"}))
        results = [r async for r in executor.get_results()]

        assert len(results) == 1
        assert results[0].name == "Read"
        assert results[0].data == "Read result"
        assert not results[0].is_error

    @pytest.mark.asyncio
    async def test_concurrent_safe_tools_run_in_parallel(self):
        order = []

        class TimedTool(FakeTool):
            async def call(self, args, context):
                order.append(f"{self._name}_start")
                await asyncio.sleep(0.05)
                order.append(f"{self._name}_end")
                return ToolResult(data=f"{self._name} done")

        r1 = TimedTool("Read1", read_only=True)
        r2 = TimedTool("Read2", read_only=True)
        tool_map = {"Read1": r1, "Read2": r2}
        executor = StreamingToolExecutor(tool_map, FakeServices(), None)

        executor.add_tool(make_block("Read1", "t1"))
        executor.add_tool(make_block("Read2", "t2"))
        results = [r async for r in executor.get_results()]

        assert len(results) == 2
        assert results[0].name == "Read1"
        assert results[1].name == "Read2"
        # Both should start before either ends (parallel)
        assert order.index("Read2_start") < order.index("Read1_end")

    @pytest.mark.asyncio
    async def test_non_safe_tool_blocks_queue(self):
        order = []

        class TimedTool(FakeTool):
            async def call(self, args, context):
                order.append(f"{self._name}_start")
                await asyncio.sleep(0.02)
                order.append(f"{self._name}_end")
                return ToolResult(data=f"{self._name} done")

        read = TimedTool("Read", read_only=True)
        bash = TimedTool("Bash", read_only=False)
        tool_map = {"Read": read, "Bash": bash}
        executor = StreamingToolExecutor(tool_map, FakeServices(), None)

        executor.add_tool(make_block("Read", "t1"))
        executor.add_tool(make_block("Bash", "t2"))
        results = [r async for r in executor.get_results()]

        assert len(results) == 2
        # Bash must start after Read ends (sequential)
        assert order.index("Bash_start") >= order.index("Read_end")

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        executor = StreamingToolExecutor({}, FakeServices(), None)

        executor.add_tool(make_block("NoSuchTool", "t1"))
        results = [r async for r in executor.get_results()]

        assert len(results) == 1
        assert results[0].is_error
        assert "Unknown tool" in results[0].data

    @pytest.mark.asyncio
    async def test_permission_denied_raises_error(self):
        bash = FakeTool("Bash", read_only=False)
        services = FakeServices(deny_names={"Bash"})
        executor = StreamingToolExecutor({"Bash": bash}, services, None)

        executor.add_tool(make_block("Bash", "t1", {"command": "rm -rf /"}))
        with pytest.raises(PermissionDeniedError):
            async for _ in executor.get_results():
                pass

    @pytest.mark.asyncio
    async def test_results_yielded_in_order(self):
        """Even if tool 2 finishes before tool 1, results come in submission order."""

        class SlowTool(FakeTool):
            async def call(self, args, context):
                await asyncio.sleep(0.05)
                return ToolResult(data="slow")

        class FastTool(FakeTool):
            async def call(self, args, context):
                return ToolResult(data="fast")

        slow = SlowTool("Slow", read_only=True)
        fast = FastTool("Fast", read_only=True)
        tool_map = {"Slow": slow, "Fast": fast}
        executor = StreamingToolExecutor(tool_map, FakeServices(), None)

        executor.add_tool(make_block("Slow", "t1"))
        executor.add_tool(make_block("Fast", "t2"))
        results = [r async for r in executor.get_results()]

        assert results[0].name == "Slow"
        assert results[1].name == "Fast"

    @pytest.mark.asyncio
    async def test_get_tool_results_returns_pairs(self):
        read = FakeTool("Read", read_only=True)
        executor = StreamingToolExecutor({"Read": read}, FakeServices(), None)

        block = make_block("Read", "t1")
        executor.add_tool(block)
        _ = [r async for r in executor.get_results()]

        pairs = executor.get_tool_results()
        assert len(pairs) == 1
        assert pairs[0][0] == block
        assert pairs[0][1].data == "Read result"

    @pytest.mark.asyncio
    async def test_mixed_safe_and_unsafe_batching(self):
        """Read, Read, Bash, Read → batch(Read,Read), then Bash, then Read."""
        order = []

        class TrackedFakeTool(FakeTool):
            async def call(self, args, context):
                order.append(self._name)
                return ToolResult(data=f"{self._name} done")

        read = TrackedFakeTool("Read", read_only=True)
        bash = TrackedFakeTool("Bash", read_only=False)
        tool_map = {"Read": read, "Bash": bash}
        executor = StreamingToolExecutor(tool_map, FakeServices(), None)

        executor.add_tool(make_block("Read", "t1"))
        executor.add_tool(make_block("Read", "t2"))
        executor.add_tool(make_block("Bash", "t3"))
        executor.add_tool(make_block("Read", "t4"))

        results = [r async for r in executor.get_results()]
        assert len(results) == 4
        assert [r.name for r in results] == ["Read", "Read", "Bash", "Read"]
