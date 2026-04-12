import re
import anthropic
from typing import Any

COMPACT_SYSTEM_PROMPT = """CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts. In your analysis:

1. Chronologically analyze each message. For each section identify:
   - The user's explicit requests and intents
   - Your approach to addressing them
   - Key decisions, technical concepts and code patterns
   - Specific details: file names, code snippets, function signatures, file edits
   - Errors and how you fixed them
   - Specific user feedback, especially corrections
2. Double-check for technical accuracy and completeness.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Include full code snippets where applicable.
4. Errors and fixes: List all errors encountered and how they were fixed.
5. Problem Solving: Document problems solved and ongoing troubleshooting.
6. All user messages: List ALL user messages that are not tool results.
7. Pending Tasks: Outline any pending tasks explicitly asked to work on.
8. Current Work: Describe precisely what was being worked on immediately before this summary request.
9. Optional Next Step: List the next step related to the most recent work.

Wrap your final summary in <summary> tags.

REMINDER: Do NOT call any tools. Respond with plain text only."""

COMPACT_USER_PREFIX = """This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

"""


def estimate_tokens(messages: list[dict], system: str = "") -> int:
    total = len(system) // 4
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    for v in block.values():
                        if isinstance(v, str):
                            total += len(v) // 4
    return total


def format_compact_summary(raw: str) -> str:
    result = re.sub(r"<analysis>[\s\S]*?</analysis>", "", raw)
    match = re.search(r"<summary>([\s\S]*?)</summary>", result)
    if match:
        content = match.group(1).strip()
        result = f"Summary:\n{content}"
    result = re.sub(r"\n\n+", "\n\n", result)
    return result.strip()


async def compact_conversation(
    client: anthropic.AsyncAnthropic,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> str:
    async with client.messages.stream(
        model=model,
        max_tokens=20000,
        system=f"{COMPACT_SYSTEM_PROMPT}\n\n---\nOriginal system prompt for context:\n{system_prompt}",
        messages=messages,
    ) as stream:
        final = await stream.get_final_message()

    raw = next((b.text for b in final.content if b.type == "text"), "")
    return format_compact_summary(raw)


MIN_KEEP_TOKENS = 10_000
MIN_KEEP_TEXT_MESSAGES = 3
MAX_KEEP_TOKENS = 40_000


async def auto_compact(
    client: anthropic.AsyncAnthropic,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str,
    context_window: int = 200_000,
    threshold_ratio: float = 0.85,
) -> list[dict[str, Any]] | None:
    messages = micro_compact(messages)

    token_est = estimate_tokens(messages, system_prompt)
    threshold = int(context_window * threshold_ratio)

    if token_est < threshold:
        return None

    keep_idx = _calculate_keep_index(messages)
    to_summarize = messages[:keep_idx]
    to_keep = messages[keep_idx:]

    if not to_summarize:
        return None

    summary = await compact_conversation(client, model, to_summarize, system_prompt)

    summary_message = COMPACT_USER_PREFIX + summary

    # Ensure first kept message is not a tool_result (API invariant)
    while to_keep and _is_tool_result(to_keep[0]):
        to_keep.pop(0)

    return [{"role": "user", "content": summary_message}] + to_keep


def _calculate_keep_index(messages: list[dict[str, Any]]) -> int:
    """Find the split point: messages[:idx] get summarized, messages[idx:] are kept.

    Expands backwards from end until BOTH minimums are met:
    - At least MIN_KEEP_TOKENS tokens
    - At least MIN_KEEP_TEXT_MESSAGES messages with text content
    Stops expanding if MAX_KEEP_TOKENS is reached.
    Also avoids splitting tool_use / tool_result pairs.
    """
    if not messages:
        return 0

    total_tokens = 0
    text_msg_count = 0
    start_idx = len(messages)

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        msg_tokens = estimate_tokens([msg])
        total_tokens += msg_tokens
        if _has_text_content(msg):
            text_msg_count += 1

        start_idx = i

        if total_tokens >= MAX_KEEP_TOKENS:
            break

        if (total_tokens >= MIN_KEEP_TOKENS
                and text_msg_count >= MIN_KEEP_TEXT_MESSAGES):
            break

    # Don't split a tool_use / tool_result pair: if start_idx points at a
    # tool_result message, include the preceding assistant (tool_use) too.
    if start_idx > 0 and _is_tool_result(messages[start_idx]):
        start_idx -= 1

    return start_idx


def _has_text_content(message: dict) -> bool:
    """Check if a message contains actual text (not just tool results)."""
    content = message.get("content", "")
    if isinstance(content, str) and content:
        return True
    if isinstance(content, list):
        return any(
            b.get("type") == "text" for b in content if isinstance(b, dict)
        )
    return False


def _is_tool_result(message: dict) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content", "")
    if isinstance(content, list) and content:
        return content[0].get("type") == "tool_result"
    return False


# ---------------------------------------------------------------------------
# Micro-compact: trim oversized / stale tool results before LLM compaction
# ---------------------------------------------------------------------------

MAX_TOOL_RESULT_CHARS = 50_000
STALE_TRIM_CHARS = 500
STALE_TURN_THRESHOLD = 10
CLEARED_MESSAGE = "[Old tool result content cleared]"


def micro_compact(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim oversized and stale tool results in-place-style (returns new list).

    Two passes:
    1. Any tool result > MAX_TOOL_RESULT_CHARS → keep head preview + truncation notice.
    2. Tool results older than STALE_TURN_THRESHOLD turns → shrink to STALE_TRIM_CHARS.
    """
    total = len(messages)
    result = []
    for idx, msg in enumerate(messages):
        if not _is_tool_result(msg):
            result.append(msg)
            continue

        turns_from_end = total - idx
        new_content = []
        changed = False
        for block in msg["content"]:
            if block.get("type") != "tool_result":
                new_content.append(block)
                continue

            data = block.get("content", "")
            if not isinstance(data, str):
                new_content.append(block)
                continue

            if turns_from_end > STALE_TURN_THRESHOLD and len(data) > STALE_TRIM_CHARS:
                new_content.append({
                    **block,
                    "content": CLEARED_MESSAGE,
                })
                changed = True
            elif len(data) > MAX_TOOL_RESULT_CHARS:
                preview = data[:2000]
                notice = f"\n[Result truncated, {len(data)} chars total. Use Read tool to see full content.]"
                new_content.append({
                    **block,
                    "content": preview + notice,
                })
                changed = True
            else:
                new_content.append(block)

        if changed:
            result.append({**msg, "content": new_content})
        else:
            result.append(msg)

    return result
