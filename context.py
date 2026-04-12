from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from tools.base import Tool


@dataclass
class AgentContext:
    """Agent state: working directory, tools, and conversation history."""

    cwd: str
    tools: list[Tool]
    messages: list[dict[str, Any]] = field(default_factory=list)

    def build_system_prompt(self) -> str:
        """Assemble the system prompt with cwd, CODER.md, and tool list."""
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

