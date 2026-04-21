from abc import ABC, abstractmethod
from typing import Any
from ..agent_types import ToolResult


class Tool(ABC):
    """Base class for all agent tools."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def input_schema(self) -> dict: ...

    @abstractmethod
    async def call(self, args: dict[str, Any], context: Any) -> ToolResult: ...

    def is_read_only(self, args: dict[str, Any]) -> bool:
        """Whether this invocation only reads (no side effects)."""
        return False

    def is_concurrent_safe(self, args: dict[str, Any]) -> bool:
        """Whether this invocation can run concurrently with others."""
        return self.is_read_only(args)

    def to_api_schema(self) -> dict:
        """Convert to Anthropic API tool schema format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
