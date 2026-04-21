import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from coder.services.compact import (
    estimate_tokens,
    format_compact_summary,
    compact_conversation,
    auto_compact,
    micro_compact,
    _calculate_keep_index,
    _has_text_content,
    MAX_TOOL_RESULT_CHARS,
    STALE_TURN_THRESHOLD,
    STALE_TRIM_CHARS,
    CLEARED_MESSAGE,
    MIN_KEEP_TOKENS,
    MIN_KEEP_TEXT_MESSAGES,
    MAX_KEEP_TOKENS,
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
        big_old = "x" * 200_000  # 50K tokens - old content to be summarized
        # Recent messages collectively > MIN_KEEP_TOKENS so they won't pull in old ones
        medium = "y" * 20_000   # 5K tokens each, 4 of these = 20K > MIN_KEEP_TOKENS
        messages = [
            {"role": "user", "content": big_old},
            {"role": "assistant", "content": big_old},
            {"role": "user", "content": medium},
            {"role": "assistant", "content": medium},
            {"role": "user", "content": medium},
            {"role": "assistant", "content": medium},
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
            context_window=200_000,
            threshold_ratio=0.5,
        )
        assert result is not None
        # First message is the summary
        assert result[0]["role"] == "user"
        assert "compressed" in result[0]["content"]
        # Recent messages are kept (dynamic, not fixed 4)
        assert len(result) >= 2
        # The big old content messages should have been summarized away
        assert all("x" * 200_000 not in str(m.get("content", "")) for m in result)

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
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
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
        # First kept message must not be a tool_result
        assert result[0]["role"] == "user"
        content = result[0].get("content", "")
        if isinstance(content, list) and content:
            assert content[0].get("type") != "tool_result"


class TestCalculateKeepIndex:
    def test_empty_messages(self):
        assert _calculate_keep_index([]) == 0

    def test_short_conversation_keeps_all(self):
        """If total tokens < MIN_KEEP_TOKENS, keep everything (index=0)."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        ]
        idx = _calculate_keep_index(messages)
        assert idx == 0  # all kept, nothing to summarize

    def test_long_conversation_splits(self):
        """With enough content, should split: old stuff summarized, recent kept."""
        # 50K chars ≈ 12.5K tokens each, so two big messages = 25K tokens > MIN_KEEP_TOKENS
        big = "x" * 50_000
        messages = [
            {"role": "user", "content": big},       # old, should be summarized
            {"role": "assistant", "content": big},   # old, should be summarized
            {"role": "user", "content": big},        # kept (tokens expand backwards)
            {"role": "assistant", "content": big},   # kept
            {"role": "user", "content": "recent q"},
            {"role": "assistant", "content": "recent a"},
        ]
        idx = _calculate_keep_index(messages)
        # Should keep enough to meet MIN_KEEP_TOKENS but not everything
        assert 0 < idx < len(messages)

    def test_respects_max_keep_tokens(self):
        """Should stop expanding once MAX_KEEP_TOKENS is hit."""
        huge = "x" * (MAX_KEEP_TOKENS * 4 + 1000)  # each msg > MAX_KEEP_TOKENS
        messages = [
            {"role": "user", "content": huge},
            {"role": "assistant", "content": huge},
            {"role": "user", "content": huge},
            {"role": "assistant", "content": "last"},
        ]
        idx = _calculate_keep_index(messages)
        # Should not keep all messages (would far exceed max)
        assert idx > 0

    def test_does_not_split_tool_pair(self):
        """If keep boundary falls on a tool_result, include the preceding assistant."""
        big = "x" * 60_000
        messages = [
            {"role": "user", "content": big},
            {"role": "assistant", "content": big},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
            {"role": "user", "content": "follow up"},
            {"role": "assistant", "content": "response"},
        ]
        idx = _calculate_keep_index(messages)
        # If tool_result is at boundary, it should pull in the assistant before it
        if idx > 0:
            kept = messages[idx:]
            if kept and kept[0].get("role") == "user":
                content = kept[0].get("content", "")
                if isinstance(content, list) and content:
                    # Should not start with a tool_result
                    assert content[0].get("type") != "tool_result"

    def test_needs_min_text_messages(self):
        """Expands until MIN_KEEP_TEXT_MESSAGES text messages are included."""
        messages = [
            {"role": "user", "content": "a" * 50_000},  # big enough for tokens
            {"role": "assistant", "content": "b" * 50_000},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ]
        idx = _calculate_keep_index(messages)
        kept = messages[idx:]
        text_count = sum(1 for m in kept if _has_text_content(m))
        assert text_count >= MIN_KEEP_TEXT_MESSAGES


class TestMicroCompact:
    def _make_tool_msg(self, data: str, tool_id: str = "t1") -> dict:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": data}],
        }

    def test_truncates_oversized_result(self):
        big = "x" * (MAX_TOOL_RESULT_CHARS + 1000)
        messages = [self._make_tool_msg(big)]
        result = micro_compact(messages)
        content = result[0]["content"][0]["content"]
        assert len(content) < MAX_TOOL_RESULT_CHARS
        assert "truncated" in content.lower()

    def test_small_result_unchanged(self):
        messages = [self._make_tool_msg("short result")]
        result = micro_compact(messages)
        assert result[0]["content"][0]["content"] == "short result"

    def test_stale_result_cleared(self):
        # Build enough messages so the tool result is > STALE_TURN_THRESHOLD from end
        messages = [self._make_tool_msg("x" * 1000)]
        for i in range(STALE_TURN_THRESHOLD + 1):
            messages.append({"role": "assistant" if i % 2 == 0 else "user", "content": f"msg {i}"})
        result = micro_compact(messages)
        assert result[0]["content"][0]["content"] == CLEARED_MESSAGE

    def test_recent_stale_not_cleared(self):
        # Tool result is recent (within threshold), should not be cleared even if large-ish
        messages = [
            {"role": "user", "content": "hello"},
            self._make_tool_msg("x" * 1000),
        ]
        result = micro_compact(messages)
        assert result[1]["content"][0]["content"] == "x" * 1000

    def test_non_tool_messages_untouched(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        result = micro_compact(messages)
        assert result == messages

    def test_preserves_non_string_content(self):
        msg = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": 12345}],
        }
        messages = [msg]
        result = micro_compact(messages)
        assert result[0]["content"][0]["content"] == 12345
