import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from coder.core.agent_loop import AgentLoop
from coder.core.services import AgentServices
from coder.core.context import AgentContext
from coder.permissions import PermissionManager
from coder.persistence.settings import Settings
from coder.hooks import HookRunner, register_builtin_hooks
from coder.agent_types import (
    ToolUseBlock, TextDelta, ToolUseStart, ToolExecResult, TurnComplete,
)
from coder.tools.bash import BashTool
from coder.tools.file_read import FileReadTool


# --- Helpers to build mock stream ---

def _make_block(type_: str, **kwargs):
    """Create a mock content block."""
    block = MagicMock()
    block.type = type_
    for k, v in kwargs.items():
        setattr(block, k, v)
    block.model_dump.return_value = {"type": type_, **kwargs}
    return block


def _make_final_message(blocks, stop_reason="end_turn"):
    msg = MagicMock()
    msg.stop_reason = stop_reason
    msg.content = blocks
    return msg


class MockStream:
    """Simulate async iteration of SDK stream events + get_final_message()."""

    def __init__(self, events, final_message):
        self._events = events
        self._final = final_message
        self.current_message_snapshot = final_message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._iter_events()

    async def _iter_events(self):
        for e in self._events:
            yield e

    async def get_final_message(self):
        return self._final


def make_text_stream(text: str):
    """Stream that yields text deltas and resolves to a text-only final message."""
    events = []
    for ch in text:
        ev = MagicMock()
        ev.type = "text"
        ev.text = ch
        events.append(ev)
    block = _make_block("text", text=text)
    return MockStream(events, _make_final_message([block]))


def make_tool_stream(tool_name: str, tool_id: str, tool_input: dict):
    """Stream that yields a tool_use start + stop event and resolves to a tool_use final message."""
    start_ev = MagicMock()
    start_ev.type = "content_block_start"
    start_ev.index = 0
    content_block = MagicMock()
    content_block.type = "tool_use"
    content_block.configure_mock(name=tool_name)
    content_block.id = tool_id
    start_ev.content_block = content_block

    stop_ev = MagicMock()
    stop_ev.type = "content_block_stop"
    stop_ev.index = 0

    block = _make_block("tool_use", id=tool_id, name=tool_name, input=tool_input)
    return MockStream([start_ev, stop_ev], _make_final_message([block], stop_reason="tool_use"))


# --- Fixtures ---

@pytest.fixture
def ctx(tmp_path):
    return AgentContext(cwd=str(tmp_path), tools=[BashTool(), FileReadTool()])


@pytest.fixture
def services(tmp_path):
    settings = Settings(
        permissions={"allow": [], "deny": []},
        _user_raw={},
        _project_raw={},
    )
    pm = PermissionManager(settings, cwd=str(tmp_path))
    hooks = HookRunner({}, cwd=str(tmp_path), session_id="test")
    register_builtin_hooks(hooks)
    return AgentServices(permissions=pm, hooks=hooks, settings=settings)


# --- Tests ---

class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_plain_text_response(self, ctx, services):
        loop = AgentLoop(ctx, services)
        with patch.object(loop.client.messages, "stream", return_value=make_text_stream("Hello!")):
            result = await loop.run("hi")
        assert result == "Hello!"
        assert ctx.messages[-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_stream_text_deltas(self, ctx, services):
        loop = AgentLoop(ctx, services)
        with patch.object(loop.client.messages, "stream", return_value=make_text_stream("AB")):
            events = [e async for e in loop.run_stream("hi")]
        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_deltas) == 2
        assert text_deltas[0].text == "A"
        assert text_deltas[1].text == "B"
        assert isinstance(events[-1], TurnComplete)
        assert events[-1].text == "AB"

    @pytest.mark.asyncio
    async def test_single_tool_call(self, ctx, services):
        loop = AgentLoop(ctx, services)
        tool_stream = make_tool_stream("Bash", "tu_1", {"command": "echo hi"})
        final_stream = make_text_stream("Done")
        with patch.object(loop.client.messages, "stream", side_effect=[tool_stream, final_stream]):
            result = await loop.run("run echo")
        assert result == "Done"
        # messages: user, assistant(tool_use), user(tool_result), assistant(text)
        assert len(ctx.messages) == 4

    @pytest.mark.asyncio
    async def test_stream_tool_events(self, ctx, services):
        loop = AgentLoop(ctx, services)
        tool_stream = make_tool_stream("Bash", "tu_1", {"command": "echo hi"})
        final_stream = make_text_stream("Done")
        with patch.object(loop.client.messages, "stream", side_effect=[tool_stream, final_stream]):
            events = [e async for e in loop.run_stream("run echo")]
        tool_starts = [e for e in events if isinstance(e, ToolUseStart)]
        tool_results = [e for e in events if isinstance(e, ToolExecResult)]
        assert len(tool_starts) == 1
        assert tool_starts[0].name == "Bash"
        assert len(tool_results) == 1
        assert not tool_results[0].is_error

    @pytest.mark.asyncio
    async def test_permission_denied(self, ctx, services):
        loop = AgentLoop(ctx, services)
        tool_stream = make_tool_stream("Bash", "tu_1", {"command": "rm -rf /"})
        with patch.object(loop.client.messages, "stream", side_effect=[tool_stream]):
            result = await loop.run("delete everything")
        # turn ends with error tool_result so message history stays API-valid
        assert "[Permission denied" in result
        assert len(ctx.messages) == 3  # user + assistant + tool_result
        tool_result_msg = ctx.messages[2]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["is_error"] is True
        assert "user denied" in tool_result_msg["content"][0]["content"].lower()

    @pytest.mark.asyncio
    async def test_unknown_tool(self, ctx, services):
        loop = AgentLoop(ctx, services)
        tool_stream = make_tool_stream("NonExistentTool", "tu_1", {})
        final_stream = make_text_stream("ok")
        with patch.object(loop.client.messages, "stream", side_effect=[tool_stream, final_stream]):
            await loop.run("use unknown tool")
        tool_result_msg = ctx.messages[2]
        assert "Unknown tool" in tool_result_msg["content"][0]["content"]

