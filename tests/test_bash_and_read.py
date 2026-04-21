import pytest
import tempfile
from pathlib import Path
from coder.tools.bash import BashTool
from coder.tools.file_read import FileReadTool


class TestBashTool:
    @pytest.mark.asyncio
    async def test_simple_command(self):
        result = await BashTool().call({"command": "echo hello"}, None)
        assert result.data == "hello"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_nonzero_exit(self):
        result = await BashTool().call({"command": "exit 1"}, None)
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_timeout(self):
        result = await BashTool().call({"command": "sleep 10", "timeout": 1}, None)
        assert result.is_error is True
        assert "timed out" in result.data

    def test_is_read_only_true(self):
        assert BashTool().is_read_only({"command": "ls -la"}) is True
        assert BashTool().is_read_only({"command": "git status"}) is True

    def test_is_read_only_false(self):
        assert BashTool().is_read_only({"command": "rm file.txt"}) is False


class TestFileReadTool:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3")
        result = await FileReadTool().call({"file_path": str(f)}, None)
        assert result.is_error is False
        assert "line1" in result.data
        assert "line3" in result.data

    @pytest.mark.asyncio
    async def test_read_with_offset_and_limit(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("a\nb\nc\nd\ne")
        result = await FileReadTool().call({"file_path": str(f), "offset": 2, "limit": 2}, None)
        assert "b" in result.data
        assert "c" in result.data
        assert "d" not in result.data

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        result = await FileReadTool().call({"file_path": "/nonexistent/file.txt"}, None)
        assert result.is_error is True
        assert "not found" in result.data

    def test_is_read_only(self):
        assert FileReadTool().is_read_only({}) is True
