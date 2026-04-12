import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.compact import (
    estimate_tokens,
    format_compact_summary,
    compact_conversation,
    auto_compact,
)


class TestEstimateTokens:
    def test_simple_messages(self):
        messages = [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"},
        ]
        tokens = estimate_tokens(messages)
        assert tokens > 0

    def test_with_system(self):
        t1 = estimate_tokens([], system="")
        t2 = estimate_tokens([], system="a" * 400)
        assert t2 > t1

    def test_list_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "some result data here"},
                ],
            }
        ]
        tokens = estimate_tokens(messages)
        assert tokens > 0

    def test_empty(self):
        assert estimate_tokens([]) == 0


class TestFormatCompactSummary:
    def test_strips_analysis(self):
        raw = "<analysis>thinking...</analysis>\n<summary>the summary</summary>"
        result = format_compact_summary(raw)
        assert "thinking" not in result
        assert "the summary" in result

    def test_extracts_summary(self):
        raw = "<summary>hello world</summary>"
        result = format_compact_summary(raw)
        assert result == "Summary:\nhello world"

    def test_no_tags(self):
        raw = "plain text response"
        result = format_compact_summary(raw)
        assert result == "plain text response"

    def test_collapses_whitespace(self):
        raw = "<summary>line1\n\n\n\nline2</summary>"
        result = format_compact_summary(raw)
        assert "\n\n\n" not in result


class TestCompactConversation:
    @pytest.mark.asyncio
    async def test_returns_formatted_summary(self):
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "<analysis>x</analysis>\n<summary>compressed</summary>"

        mock_final = MagicMock()
        mock_final.content = [mock_block]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.get_final_message = AsyncMock(return_value=mock_final)

        client = AsyncMock()
        client.messages.stream = MagicMock(return_value=mock_stream)

        result = await compact_conversation(
            client, "claude-opus-4-6",
            [{"role": "user", "content": "hi"}],
            "system prompt",
        )
        assert "compressed" in result
        assert "analysis" not in result


class TestAutoCompact:
    @pytest.mark.asyncio
    async def test_no_compact_under_threshold(self):
        messages = [{"role": "user", "content": "short"}]
        result = await auto_compact(
            AsyncMock(), "model", messages, "sys",
            context_window=200_000,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_compact_over_threshold(self):
        big_content = "x" * 100_000
        messages = [
            {"role": "user", "content": big_content},
            {"role": "assistant", "content": big_content},
            {"role": "user", "content": "latest question"},
            {"role": "assistant", "content": "latest answer"},
        ]

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "<summary>compressed</summary>"
        mock_final = MagicMock()
        mock_final.content = [mock_block]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.get_final_message = AsyncMock(return_value=mock_final)

        client = AsyncMock()
        client.messages.stream = MagicMock(return_value=mock_stream)

        result = await auto_compact(
            client, "model", messages, "sys",
            context_window=100_000,
            threshold_ratio=0.5,
        )
        assert result is not None
        assert result[0]["role"] == "user"
        assert "compressed" in result[0]["content"]
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_compact_skips_tool_result_head(self):
        big_content = "x" * 100_000
        messages = [
            {"role": "user", "content": big_content},
            {"role": "assistant", "content": big_content},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "1", "content": "r"}],
            },
            {"role": "assistant", "content": "after tool"},
        ]

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "<summary>s</summary>"
        mock_final = MagicMock()
        mock_final.content = [mock_block]

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.get_final_message = AsyncMock(return_value=mock_final)

        client = AsyncMock()
        client.messages.stream = MagicMock(return_value=mock_stream)

        result = await auto_compact(
            client, "model", messages, "sys",
            context_window=100_000,
            threshold_ratio=0.5,
        )
        assert result is not None
        for msg in result:
            if msg["role"] == "user":
                content = msg.get("content", "")
                if isinstance(content, list) and content:
                    assert content[0].get("type") != "tool_result"
