import asyncio
import os
import random
from typing import Any, AsyncGenerator
import anthropic
from dotenv import load_dotenv
from context import AgentContext
from permissions import PermissionManager
from streaming_executor import StreamingToolExecutor
from services.compact import auto_compact, compact_conversation, COMPACT_USER_PREFIX
from services.tool_result_storage import process_tool_result_content
from session import SessionManager
from agent_types import (
    ToolResult, ToolUseBlock, StreamEvent,
    TextDelta, ToolUseStart, ToolExecResult, TurnComplete,
    PermissionDeniedError,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Retry config
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5
_BASE_DELAY = 1.0
_MAX_DELAY = 60.0


class AgentLoop:
    """Core agent loop: call LLM → execute tools → repeat until done."""

    def __init__(
        self,
        context: AgentContext,
        permission_manager: PermissionManager,
        session: SessionManager | None = None,
    ):
        self.context = context
        self.pm = permission_manager
        self.session = session
        # Expose the permission manager on the context so sub-agents (dispatched
        # via AgentTool) can reuse it without threading it through every call site.
        context._pm = permission_manager
        self.model = os.environ.get("MODEL_ID", "claude-opus-4-5")
        self.client = anthropic.AsyncAnthropic()
        self._tool_map = {t.name: t for t in context.tools}

    def _append_message(self, message: dict) -> None:
        """Append to context and persist to session file."""
        self.context.messages.append(message)
        if self.session:
            self.session.record(message)

    async def run(self, user_message: str) -> str:
        """Non-streaming: collect full response text and return."""
        full_text = ""
        async for event in self.run_stream(user_message):
            match event:
                case TurnComplete(text=text):
                    full_text = text
        return full_text

    async def _call_api(self) -> Any:
        """Call the streaming API with exponential backoff retry on rate limits."""
        delay = _BASE_DELAY
        for attempt in range(_MAX_RETRIES):
            try:
                async with self.client.messages.stream(
                    model=self.model,
                    max_tokens=8096,
                    system=self.context.build_system_prompt(),
                    tools=[t.to_api_schema() for t in self.context.tools],
                    messages=self.context.messages,
                ) as stream:
                    full_text = ""
                    executor = StreamingToolExecutor(self._tool_map, self.pm, self.context)
                    pending_tool_blocks: dict[int, dict] = {}

                    async for event in stream:
                        match event.type:
                            case "text":
                                full_text += event.text
                                yield TextDelta(text=event.text)

                            case "content_block_start":
                                block = event.content_block
                                if block.type == "tool_use":
                                    pending_tool_blocks[event.index] = {
                                        "id": block.id,
                                        "name": block.name,
                                    }

                            case "content_block_stop":
                                if event.index in pending_tool_blocks:
                                    info = pending_tool_blocks.pop(event.index)
                                    snapshot = stream.current_message_snapshot
                                    for b in snapshot.content:
                                        if b.type == "tool_use" and b.id == info["id"]:
                                            block = ToolUseBlock(
                                                id=b.id, name=b.name, input=b.input,
                                            )
                                            executor.add_tool(block)
                                            # Yield only after input is fully parsed,
                                            # so listeners can render parameters.
                                            yield ToolUseStart(
                                                name=b.name, id=b.id, input=b.input,
                                            )
                                            break

                    final_message = await stream.get_final_message()
                    yield (full_text, executor, final_message)
                    return  # success

            except anthropic.RateLimitError as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                jitter = random.uniform(0, 1)
                wait = min(delay + jitter, _MAX_DELAY)
                yield TextDelta(text=f"\n[Rate limited, retrying in {wait:.1f}s...]\n")
                await asyncio.sleep(wait)
                delay = min(delay * 2, _MAX_DELAY)

    async def run_stream(self, user_message: str) -> AsyncGenerator[StreamEvent, None]:
        """Streaming agent loop. Yields events as they arrive."""
        self._append_message({"role": "user", "content": user_message})

        while True:
            # prompt_too_long: compact then retry once
            try:
                result = None
                async for event in self._call_api():
                    if isinstance(event, tuple):
                        result = event
                    else:
                        yield event
            except anthropic.BadRequestError as e:
                if "prompt is too long" in str(e).lower() or "prompt_too_long" in str(e).lower():
                    yield TextDelta(text="\n[Context too long, compacting...]\n")
                    summary = await compact_conversation(
                        self.client, self.model,
                        self.context.messages,
                        self.context.build_system_prompt(),
                    )
                    self.context.messages = [
                        {"role": "user", "content": COMPACT_USER_PREFIX + summary}
                    ]
                    # retry after compaction
                    result = None
                    async for event in self._call_api():
                        if isinstance(event, tuple):
                            result = event
                        else:
                            yield event
                else:
                    raise

            full_text, executor, final_message = result

            assistant_content = [b.model_dump() for b in final_message.content]
            self._append_message({"role": "assistant", "content": assistant_content})

            if final_message.stop_reason != "tool_use":
                yield TurnComplete(text=full_text)
                return

            try:
                async for result_event in executor.get_results():
                    yield result_event
            except PermissionDeniedError as e:
                # Append tool_result for every tool_use in the assistant message
                # so the message history stays valid for the API
                tool_use_blocks = [
                    b for b in final_message.content if b.type == "tool_use"
                ]
                self._append_message({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": b.id,
                            "content": (
                                "The user denied this tool call. "
                                "This is NOT a system error — the user chose to reject it. "
                                "Do not retry the same tool call. "
                                "Ask the user what they would like to do instead."
                            ),
                            "is_error": True,
                        }
                        for b in tool_use_blocks
                    ],
                })
                yield TurnComplete(text=f"[Permission denied: {e}]")
                return

            tool_results_dir = self.session.tool_results_dir if self.session else None
            self._append_message({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": process_tool_result_content(
                            result.data, block.name, block.id, tool_results_dir,
                        ),
                        "is_error": result.is_error,
                    }
                    for block, result in executor.get_tool_results()
                ],
            })

            compacted = await auto_compact(
                self.client, self.model,
                self.context.messages,
                self.context.build_system_prompt(),
            )
            if compacted is not None:
                self.context.messages = compacted
