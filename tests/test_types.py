from coder.tools.base import ToolResult, ToolUseBlock


class TestToolResult:
    def test_defaults(self):
        r = ToolResult(data="output")
        assert r.data == "output"
        assert r.is_error is False

    def test_error_flag(self):
        r = ToolResult(data="something went wrong", is_error=True)
        assert r.is_error is True

    def test_empty_data(self):
        r = ToolResult(data="")
        assert r.data == ""
        assert r.is_error is False


class TestToolUseBlock:
    def test_fields(self):
        block = ToolUseBlock(id="tool_123", name="Bash", input={"command": "ls"})
        assert block.id == "tool_123"
        assert block.name == "Bash"
        assert block.input == {"command": "ls"}

    def test_empty_input(self):
        block = ToolUseBlock(id="x", name="Read", input={})
        assert block.input == {}
