"""Permission system: deny rules → read-only → allow rules → session → ask user."""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.base import Tool
from settings import Settings, add_permission_rule
from permission_rule import (
    PermissionRule, PermissionRuleValue,
    parse_rule, match_rule, load_rules_from_settings,
    build_rule_str,
)


# ---------------------------------------------------------------------------
# Hard-coded safety nets (always deny, regardless of settings)
# ---------------------------------------------------------------------------

HARD_DENY_COMMANDS = ("rm -rf /", "rm -rf ~", ":(){ :|:& };:")


# ---------------------------------------------------------------------------
# PermissionManager
# ---------------------------------------------------------------------------

class PermissionManager:
    """Rule-based permission check: hard-deny → settings deny → read-only → settings allow → session → ask."""

    def __init__(self, settings: Settings, cwd: str):
        self._cwd = cwd
        self._settings = settings
        self._session_rules: list[PermissionRule] = []

        # load persistent rules from merged settings
        perms = settings.permissions
        self._rules: list[PermissionRule] = []
        # user-level rules
        user_perms = settings._user_raw.get("permissions", {})
        self._rules.extend(load_rules_from_settings(
            user_perms.get("allow", []),
            user_perms.get("deny", []),
            source="user",
        ))
        # project-level rules
        proj_perms = settings._project_raw.get("permissions", {})
        self._rules.extend(load_rules_from_settings(
            proj_perms.get("allow", []),
            proj_perms.get("deny", []),
            source="project",
        ))

    def _find_matching_rule(
        self, tool: Tool, args: dict, behavior: str, sources: list[PermissionRule] | None = None,
    ) -> PermissionRule | None:
        """Find the first matching rule with the given behavior."""
        rules = sources if sources is not None else self._rules
        for rule in rules:
            if rule.behavior == behavior and match_rule(rule.value, tool.name, args):
                return rule
        return None

    def check(self, tool: Tool, args: dict) -> tuple[bool | None, str]:
        """Return (allowed, reason). None means 'needs user confirmation'."""
        # 1. hard-coded deny
        if tool.name == "Bash":
            cmd = args.get("command", "")
            if any(cmd.strip().startswith(d) for d in HARD_DENY_COMMANDS):
                return False, "denied: dangerous command"

        # 2. settings deny rules (deny always wins)
        deny_rule = self._find_matching_rule(tool, args, "deny")
        if deny_rule:
            return False, f"denied by {deny_rule.source} rule: {deny_rule.value}"

        # 3. read-only tools / invocations
        if tool.is_read_only(args):
            return True, "auto-allowed: read-only"

        # 4. settings allow rules
        allow_rule = self._find_matching_rule(tool, args, "allow")
        if allow_rule:
            return True, f"allowed by {allow_rule.source} rule: {allow_rule.value}"

        # 5. session rules
        session_allow = self._find_matching_rule(tool, args, "allow", self._session_rules)
        if session_allow:
            return True, "auto-allowed: session approval"

        # 6. needs user confirmation
        return None, "needs confirmation"

    def ask(self, tool: Tool, args: dict) -> bool:
        """Prompt the user for permission. Supports session and persistent approval."""
        rule_str = build_rule_str(tool.name, args)
        prompt = f"\nAllow {rule_str}? [y]es / [a]lways / [p]ersist / [n]o: "
        print(prompt, end="", flush=True)
        response = input().strip().lower()

        if response == "a":
            rule_val = parse_rule(rule_str)
            self._session_rules.append(
                PermissionRule(source="session", behavior="allow", value=rule_val)
            )
            return True

        if response == "p":
            add_permission_rule("project", self._cwd, "allow", rule_str)
            rule_val = parse_rule(rule_str)
            self._rules.append(
                PermissionRule(source="project", behavior="allow", value=rule_val)
            )
            return True

        return response == "y"

    def is_allowed(self, tool: Tool, args: dict) -> bool:
        """Full check: auto-decide or ask the user."""
        allowed, _ = self.check(tool, args)
        if allowed is True:
            return True
        if allowed is False:
            return False
        return self.ask(tool, args)
