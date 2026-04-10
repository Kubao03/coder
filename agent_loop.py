import asyncio
import json
import os
from typing import Any, AsyncGenerator
import anthropic
from dotenv import load_dotenv
from context import AgentContext
from permissions import PermissionManager
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
            # --- Stream LLM response ---
            assistant_content = []
            tool_use_blocks: list[ToolUseBlock] = []
            full_text = ""

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
                                yield ToolUseStart(name=block.name, id=block.id)

                        # input_json events are handled internally by SDK;
                        # we collect final blocks from the accumulated message.

                final_message = await stream.get_final_message()

            # Build assistant content from the final accumulated message
            assistant_content = [block.model_dump() for block in final_message.content]
            self.context.messages.append({"role": "assistant", "content": assistant_content})

            # Collect tool_use blocks
            tool_use_blocks = [
                ToolUseBlock(id=b.id, name=b.name, input=b.input)
                for b in final_message.content
                if b.type == "tool_use"
            ]

            # No tool calls → done
            if final_message.stop_reason != "tool_use":
                yield TurnComplete(text=full_text)
                return

            # --- Execute tools ---
            tool_results = await self._execute_tools(tool_use_blocks)

            # Yield execution results
            for block, result in zip(tool_use_blocks, tool_results):
                yield ToolExecResult(
                    name=block.name, id=block.id,
                    data=result.data, is_error=result.is_error,
                )

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
                    for block, result in zip(tool_use_blocks, tool_results)
                ],
            })

            self.context.compact_messages()

    async def _execute_tools(self, blocks: list[ToolUseBlock]) -> list[ToolResult]:
        results = []
        for batch in self._partition(blocks):
            if len(batch) == 1:
                results.append(await self._run_tool(batch[0]))
            else:
                results.extend(await asyncio.gather(*[self._run_tool(b) for b in batch]))
        return results

    def _partition(self, blocks: list[ToolUseBlock]) -> list[list[ToolUseBlock]]:
        """Group consecutive concurrent-safe tool calls; isolate unsafe ones."""
        batches: list[list[ToolUseBlock]] = []
        current: list[ToolUseBlock] = []
        for block in blocks:
            tool = self._tool_map.get(block.name)
            safe = tool.is_concurrent_safe(block.input) if tool else False
            if safe:
                current.append(block)
            else:
                if current:
                    batches.append(current)
                    current = []
                batches.append([block])
        if current:
            batches.append(current)
        return batches

    async def _run_tool(self, block: ToolUseBlock) -> ToolResult:
        tool = self._tool_map.get(block.name)
        if tool is None:
            return ToolResult(data=f"Unknown tool: {block.name}", is_error=True)
        if not self.pm.is_allowed(tool, block.input):
            return ToolResult(data=f"Permission denied for {block.name}", is_error=True)
        return await tool.call(block.input, self.context)
