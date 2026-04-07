import asyncio
import os
from typing import Any
import anthropic
from dotenv import load_dotenv
from context import AgentContext
from permissions import PermissionManager
from agent_types import ToolResult, ToolUseBlock

load_dotenv()


class AgentLoop:
    def __init__(self, context: AgentContext, permission_manager: PermissionManager):
        self.context = context
        self.pm = permission_manager
        self.model = os.environ.get("MODEL_ID", "claude-opus-4-5")
        self.client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL from env
        self._tool_map = {t.name: t for t in context.tools}

    async def run(self, user_message: str) -> str:
        self.context.messages.append({"role": "user", "content": user_message})

        while True:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=8096,
                system=self.context.build_system_prompt(),
                tools=[t.to_api_schema() for t in self.context.tools],
                messages=self.context.messages,
            )

            # Append assistant message
            assistant_content = [block.model_dump() for block in response.content]
            self.context.messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason != "tool_use":
                # Extract final text response
                for block in response.content:
                    if block.type == "text":
                        return block.text
                return ""

            # Execute tools
            tool_use_blocks = [
                ToolUseBlock(id=b.id, name=b.name, input=b.input)
                for b in response.content
                if b.type == "tool_use"
            ]
            tool_results = await self._execute_tools(tool_use_blocks)

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
