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
        """Snip oldest message pairs when history grows too long."""
        while len(self.messages) > max_messages:
            # Remove oldest user+assistant pair
            self.messages.pop(0)
            if self.messages and self.messages[0]["role"] == "assistant":
                self.messages.pop(0)
