import asyncio
import os
from typing import Any, AsyncGenerator
import anthropic
from dotenv import load_dotenv
from context import AgentContext
from permissions import PermissionManager
from streaming_executor import StreamingToolExecutor
from services.compact import auto_compact
from session import SessionManager
from agent_types import (
    ToolResult, ToolUseBlock, StreamEvent,
    TextDelta, ToolUseStart, ToolExecResult, TurnComplete,
    PermissionDeniedError,
)

load_dotenv()


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

    async def run_stream(self, user_message: str) -> AsyncGenerator[StreamEvent, None]:
        """Streaming agent loop. Yields events as they arrive."""
        self._append_message({"role": "user", "content": user_message})

        while True:
            full_text = ""
            executor = StreamingToolExecutor(self._tool_map, self.pm, self.context)

            pending_tool_blocks: dict[int, dict] = {}  # index → {id, name}

            async with self.client.messages.stream(
                model=self.model,
                max_tokens=8096,
                system=self.context.build_system_prompt(),
                tools=[t.to_api_schema() for t in self.context.tools],
                messages=self.context.messages,
            ) as stream:
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
                                yield ToolUseStart(name=block.name, id=block.id)

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
                                        break

                final_message = await stream.get_final_message()

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

            self._append_message({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result.data,
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
