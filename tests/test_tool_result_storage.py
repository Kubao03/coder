import pytest
from pathlib import Path

from coder.services.tool_result_storage import (
    PERSISTED_OUTPUT_TAG,
    PERSISTED_OUTPUT_CLOSING_TAG,
    PREVIEW_SIZE_BYTES,
    build_large_result_message,
    empty_result_marker,
    generate_preview,
    is_empty_content,
    persist_tool_result,
    process_tool_result_content,
)


class TestEmptyContent:
    def test_empty_string_is_empty(self):
        assert is_empty_content("")

    def test_whitespace_only_is_empty(self):
        assert is_empty_content("   \n\t  ")

    def test_non_empty(self):
        assert not is_empty_content("hello")

    def test_marker_format(self):
        assert empty_result_marker("Bash") == "(Bash completed with no output)"


class TestPreview:
    def test_short_content_unchanged(self):
        preview, has_more = generate_preview("hello", 100)
        assert preview == "hello"
        assert has_more is False

    def test_long_content_truncated(self):
        content = "a" * 5000
        preview, has_more = generate_preview(content, 100)
        assert has_more is True
        assert len(preview) <= 100

    def test_truncates_at_newline(self):
        # Build content that has a newline within the limit.
        content = "line1\n" + "x" * 100 + "\n" + "y" * 1000
        preview, has_more = generate_preview(content, 150)
        assert has_more is True
        # Should cut at a newline boundary.
        assert preview.endswith("\n") or "\n" in preview


class TestPersist:
    def test_persist_writes_file(self, tmp_path):
        path = persist_tool_result("hello world", "t_abc", tmp_path)
        assert path is not None
        assert path.exists()
        assert path.read_text() == "hello world"
        assert path.name == "t_abc.txt"

    def test_persist_creates_dir(self, tmp_path):
        target = tmp_path / "nested" / "dir"
        path = persist_tool_result("x", "t1", target)
        assert path is not None
        assert target.exists()

    def test_persist_idempotent_same_id(self, tmp_path):
        persist_tool_result("first", "t1", tmp_path)
        # Second call with same id does not overwrite (tool_use_id is unique, content is deterministic).
        path = persist_tool_result("second", "t1", tmp_path)
        assert path is not None
        assert path.read_text() == "first"


class TestBuildMessage:
    def test_contains_tag_and_path(self, tmp_path):
        fp = tmp_path / "result.txt"
        msg = build_large_result_message(fp, 100_000, "preview text", True)
        assert msg.startswith(PERSISTED_OUTPUT_TAG)
        assert msg.endswith(PERSISTED_OUTPUT_CLOSING_TAG)
        assert str(fp) in msg
        assert "preview text" in msg


class TestProcess:
    def test_empty_content_becomes_marker(self, tmp_path):
        out = process_tool_result_content("", "Bash", "t1", tmp_path)
        assert out == "(Bash completed with no output)"

    def test_whitespace_content_becomes_marker(self, tmp_path):
        out = process_tool_result_content("   \n  ", "Read", "t1", tmp_path)
        assert out == "(Read completed with no output)"

    def test_small_content_unchanged(self, tmp_path):
        out = process_tool_result_content("hello", "Bash", "t1", tmp_path)
        assert out == "hello"
        # Nothing written to disk.
        assert not any(tmp_path.iterdir())

    def test_large_content_persisted(self, tmp_path):
        big = "x" * 60_000
        out = process_tool_result_content(big, "Bash", "t_big", tmp_path, threshold_chars=50_000)
        assert PERSISTED_OUTPUT_TAG in out
        assert "t_big.txt" in out
        # The file is actually written to disk.
        assert (tmp_path / "t_big.txt").exists()
        assert (tmp_path / "t_big.txt").read_text() == big

    def test_below_threshold_unchanged(self, tmp_path):
        content = "x" * 49_000
        out = process_tool_result_content(content, "Bash", "t1", tmp_path, threshold_chars=50_000)
        assert out == content

    def test_no_session_dir_skips_persist(self):
        # When no session dir is given, pass through even large results (better to spend tokens than to drop data).
        big = "x" * 60_000
        out = process_tool_result_content(big, "Bash", "t1", None, threshold_chars=50_000)
        assert out == big

    def test_non_string_passthrough(self, tmp_path):
        # result.data is str today; defensively pass through non-str values.
        out = process_tool_result_content([{"type": "text", "text": "x"}], "Bash", "t1", tmp_path)
        assert out == [{"type": "text", "text": "x"}]

    def test_large_result_preview_contains_head(self, tmp_path):
        content = "HEAD\n" + ("middle\n" * 10_000) + "TAIL"
        out = process_tool_result_content(content, "Bash", "t_p", tmp_path, threshold_chars=1_000)
        assert "HEAD" in out
        # TAIL must not appear in the preview.
        assert "TAIL" not in out
