import pytest
from pathlib import Path
from coder.tools.file_edit import FileEditTool
from coder.tools.file_write import FileWriteTool
from coder.tools.glob_tool import GlobTool
from coder.tools.grep_tool import GrepTool


class TestFileEditTool:
    @pytest.mark.asyncio
    async def test_basic_edit(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = await FileEditTool().call(
            {"file_path": str(f), "old_string": "world", "new_string": "python"}, None
        )
        assert result.is_error is False
        assert f.read_text() == "hello python"

    @pytest.mark.asyncio
    async def test_old_string_not_found(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = await FileEditTool().call(
            {"file_path": str(f), "old_string": "missing", "new_string": "x"}, None
        )
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_old_string_not_unique(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("foo foo")
        result = await FileEditTool().call(
            {"file_path": str(f), "old_string": "foo", "new_string": "bar"}, None
        )
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        result = await FileEditTool().call(
            {"file_path": "/nonexistent.txt", "old_string": "x", "new_string": "y"}, None
        )
        assert result.is_error is True


class TestFileWriteTool:
    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path):
        f = tmp_path / "new.txt"
        result = await FileWriteTool().call({"file_path": str(f), "content": "hello"}, None)
        assert result.is_error is False
        assert f.read_text() == "hello"

    @pytest.mark.asyncio
    async def test_overwrite_existing(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old")
        await FileWriteTool().call({"file_path": str(f), "content": "new"}, None)
        assert f.read_text() == "new"

    @pytest.mark.asyncio
    async def test_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "a" / "b" / "c.txt"
        result = await FileWriteTool().call({"file_path": str(f), "content": "deep"}, None)
        assert result.is_error is False
        assert f.read_text() == "deep"


class TestGlobTool:
    @pytest.mark.asyncio
    async def test_find_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = await GlobTool().call({"pattern": "*.py", "path": str(tmp_path)}, None)
        assert "a.py" in result.data
        assert "b.py" in result.data
        assert "c.txt" not in result.data

    @pytest.mark.asyncio
    async def test_no_matches(self, tmp_path):
        result = await GlobTool().call({"pattern": "*.xyz", "path": str(tmp_path)}, None)
        assert "No files found" in result.data

    def test_is_read_only(self):
        assert GlobTool().is_read_only({}) is True


class TestGrepTool:
    @pytest.mark.asyncio
    async def test_find_in_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    pass\ndef bar():\n    pass")
        result = await GrepTool().call({"pattern": "def foo", "path": str(f)}, None)
        assert result.is_error is False
        assert "def foo" in result.data

    @pytest.mark.asyncio
    async def test_no_matches(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("nothing here")
        result = await GrepTool().call({"pattern": "xyz123", "path": str(f)}, None)
        assert "No matches" in result.data

    @pytest.mark.asyncio
    async def test_invalid_regex(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        result = await GrepTool().call({"pattern": "[invalid", "path": str(f)}, None)
        assert result.is_error is True

    def test_is_read_only(self):
        assert GrepTool().is_read_only({}) is True
