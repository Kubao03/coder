import asyncio
import os
from typing import Any, AsyncGenerator
import anthropic
from dotenv import load_dotenv
from context import AgentContext
from permissions import PermissionManager
from streaming_executor import StreamingToolExecutor
from agent_types import (
    ToolResult, ToolUseBlock, StreamEvent,
    TextDelta, ToolUseStart, ToolExecResult, TurnComplete,
)

load_dotenv()


class AgentLoop:
    def __init__(self, context: AgentContext, permission_manager: PermissionManager):
        self.context = context
        self.pm = permission_manager
        self.model = os.environ.get("MODEL_ID", "claude-opus-4-5")
        self.client = anthropic.AsyncAnthropic()
        self._tool_map = {t.name: t for t in context.tools}

    async def run(self, user_message: str) -> str:
        """Non-streaming convenience: collect full text and return."""
        full_text = ""
        async for event in self.run_stream(user_message):
            match event:
                case TurnComplete(text=text):
                    full_text = text
        return full_text

    async def run_stream(self, user_message: str) -> AsyncGenerator[StreamEvent, None]:
        """Core agent loop with streaming output."""
        self.context.messages.append({"role": "user", "content": user_message})

        while True:
            full_text = ""
            executor = StreamingToolExecutor(self._tool_map, self.pm, self.context)

            # Track tool_use blocks being streamed so we can feed them
            # to the executor the moment they are complete.
            pending_tool_blocks: dict[int, dict] = {}  # index → partial info

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
                                # Get the fully accumulated block from the stream's snapshot
                                snapshot = stream.current_message_snapshot
                                for b in snapshot.content:
                                    if b.type == "tool_use" and b.id == info["id"]:
                                        block = ToolUseBlock(
                                            id=b.id, name=b.name, input=b.input,
                                        )
                                        executor.add_tool(block)
                                        break

                final_message = await stream.get_final_message()

            # Append assistant message
            assistant_content = [b.model_dump() for b in final_message.content]
            self.context.messages.append({"role": "assistant", "content": assistant_content})

            # No tool calls → done
            if final_message.stop_reason != "tool_use":
                yield TurnComplete(text=full_text)
                return

            # Yield tool execution results (some may already be done)
            async for result_event in executor.get_results():
                yield result_event

            # Append tool results to messages
            self.context.messages.append({
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

            self.context.compact_messages()
