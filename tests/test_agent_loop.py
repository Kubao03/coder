import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agent_loop import AgentLoop
from context import AgentContext
from permissions import PermissionManager
from tools.bash import BashTool
from tools.file_read import FileReadTool


def make_text_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    block.model_dump.return_value = {"type": "text", "text": text}
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


def make_tool_response(tool_name: str, tool_id: str, tool_input: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = tool_name
    block.input = tool_input
    block.model_dump.return_value = {"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input}
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


@pytest.fixture
def ctx(tmp_path):
    return AgentContext(cwd=str(tmp_path), tools=[BashTool(), FileReadTool()])


@pytest.fixture
def pm():
    return PermissionManager()


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_plain_text_response(self, ctx, pm):
        loop = AgentLoop(ctx, pm)
        with patch.object(loop.client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = make_text_response("Hello!")
            result = await loop.run("hi")
        assert result == "Hello!"
        assert ctx.messages[-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_single_tool_call(self, ctx, pm):
        loop = AgentLoop(ctx, pm)
        tool_response = make_tool_response("Bash", "tu_1", {"command": "echo hi"})
        final_response = make_text_response("Done")
        with patch.object(loop.client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [tool_response, final_response]
            result = await loop.run("run echo")
        assert result == "Done"
        # messages: user, assistant(tool_use), user(tool_result), assistant(text)
        assert len(ctx.messages) == 4

    @pytest.mark.asyncio
    async def test_permission_denied(self, ctx, pm):
        loop = AgentLoop(ctx, pm)
        # FileWriteTool not in ctx, use Bash with dangerous command
        tool_response = make_tool_response("Bash", "tu_1", {"command": "rm -rf /"})
        final_response = make_text_response("Blocked")
        with patch.object(loop.client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [tool_response, final_response]
            result = await loop.run("delete everything")
        # tool_result should contain permission denied error
        tool_result_msg = ctx.messages[2]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["is_error"] is True

    @pytest.mark.asyncio
    async def test_unknown_tool(self, ctx, pm):
        loop = AgentLoop(ctx, pm)
        tool_response = make_tool_response("NonExistentTool", "tu_1", {})
        final_response = make_text_response("ok")
        with patch.object(loop.client.messages, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [tool_response, final_response]
            await loop.run("use unknown tool")
        tool_result_msg = ctx.messages[2]
        assert "Unknown tool" in tool_result_msg["content"][0]["content"]

    def test_partition_safe_tools_batched(self, ctx, pm):
        loop = AgentLoop(ctx, pm)
        from agent_types import ToolUseBlock
        blocks = [
            ToolUseBlock(id="1", name="Read", input={"file_path": "/a"}),
            ToolUseBlock(id="2", name="Read", input={"file_path": "/b"}),
            ToolUseBlock(id="3", name="Bash", input={"command": "rm file"}),
            ToolUseBlock(id="4", name="Read", input={"file_path": "/c"}),
        ]
        batches = loop._partition(blocks)
        assert len(batches) == 3
        assert len(batches[0]) == 2  # two reads batched
        assert len(batches[1]) == 1  # unsafe bash alone
        assert len(batches[2]) == 1  # trailing read alone
