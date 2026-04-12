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


async def auto_compact(
    client: anthropic.AsyncAnthropic,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str,
    context_window: int = 200_000,
    threshold_ratio: float = 0.85,
) -> list[dict[str, Any]] | None:
    token_est = estimate_tokens(messages, system_prompt)
    threshold = int(context_window * threshold_ratio)

    if token_est < threshold:
        return None

    summary = await compact_conversation(client, model, messages, system_prompt)

    summary_message = COMPACT_USER_PREFIX + summary

    keep_recent = 4
    recent = messages[-keep_recent:] if len(messages) > keep_recent else []
    while recent and _is_tool_result(recent[0]):
        recent.pop(0)

    return [{"role": "user", "content": summary_message}] + recent


def _is_tool_result(message: dict) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content", "")
    if isinstance(content, list) and content:
        return content[0].get("type") == "tool_result"
    return False
