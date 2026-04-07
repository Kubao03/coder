from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from tools.base import Tool


@dataclass
class AgentContext:
    cwd: str
    tools: list[Tool]
    messages: list[dict[str, Any]] = field(default_factory=list)

    def build_system_prompt(self) -> str:
        parts = [
            "You are a coding agent. You help users with software engineering tasks.",
            f"\nCurrent working directory: {self.cwd}",
        ]

        coder_md = Path(self.cwd) / "CODER.md"
        if coder_md.exists():
            parts.append(f"\n<coder_md>\n{coder_md.read_text()}\n</coder_md>")

        tool_names = ", ".join(t.name for t in self.tools)
        parts.append(f"\nAvailable tools: {tool_names}")

        return "\n".join(parts)

    def compact_messages(self, max_messages: int = 40) -> None:
        """Snip oldest messages, always leaving history in a valid state.

        A valid history must start with a plain user message (not tool_results),
        because tool_results reference tool_use IDs that must exist in the
        preceding assistant message.
        """
        while len(self.messages) > max_messages:
            self.messages.pop(0)
            # Keep removing until the new head is a plain user message.
            # A tool_result user message references tool_use IDs from the
            # assistant message we just removed — keep it and the API errors.
            while self.messages and self._is_tool_result(self.messages[0]):
                self.messages.pop(0)

    @staticmethod
    def _is_tool_result(message: dict) -> bool:
        """Return True if this is a user message containing tool_results."""
        if message.get("role") != "user":
            return False
        content = message.get("content", "")
        if isinstance(content, list) and content:
            return content[0].get("type") == "tool_result"
        return False
