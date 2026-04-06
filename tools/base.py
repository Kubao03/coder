from abc import ABC, abstractmethod
from typing import Any
from agent_types import ToolResult


class Tool(ABC):

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
        return False

    def is_concurrent_safe(self, args: dict[str, Any]) -> bool:
        return self.is_read_only(args)

    def to_api_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
