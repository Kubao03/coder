"""AgentServices: immutable container for session-scoped dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from permissions.manager import PermissionManager
    from hooks.runner import HookRunner
    from settings import Settings


@dataclass
class AgentServices:
    """Session-scoped service dependencies shared across AgentLoop and sub-agents.

    Lifetime: created once per session, passed explicitly instead of
    monkey-patched onto AgentContext.
    """

    permissions: "PermissionManager"
    hooks: "HookRunner"
    settings: "Settings"
    subagent_listener: Callable | None = None
