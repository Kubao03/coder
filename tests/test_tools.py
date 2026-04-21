import pytest
from coder.tools.base import ToolResult
from coder.tools.base import Tool


class ConcreteTool(Tool):
    name = "TestTool"
    description = "a test tool"
    input_schema = {"type": "object", "properties": {}}

    async def call(self, args, context):
        return ToolResult(data="ok")


class TestToolBase:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Tool()

    def test_concrete_tool_instantiates(self):
        t = ConcreteTool()
        assert t.name == "TestTool"

    def test_is_read_only_default_false(self):
        t = ConcreteTool()
        assert t.is_read_only({}) is False

    def test_concurrent_safe_follows_read_only(self):
        t = ConcreteTool()
        assert t.is_concurrent_safe({}) is False

    def test_to_api_schema(self):
        t = ConcreteTool()
        schema = t.to_api_schema()
        assert schema["name"] == "TestTool"
        assert schema["description"] == "a test tool"
        assert "input_schema" in schema
