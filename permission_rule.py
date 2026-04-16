"""Permission rule parsing and matching, compatible with CC's ToolName(pattern) format."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PermissionRuleValue:
    """Parsed rule: tool name + optional content pattern."""

    tool_name: str
    content: str | None = None

    def __str__(self) -> str:
        if self.content is None:
            return self.tool_name
        return f"{self.tool_name}({self.content})"


@dataclass
class PermissionRule:
    """A permission rule with source and behavior metadata."""

    source: str        # "user" | "project" | "cli" | "session"
    behavior: str      # "allow" | "deny"
    value: PermissionRuleValue


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Matches: ToolName  or  ToolName(content)
_RULE_RE = re.compile(r"^([A-Za-z]\w*)(?:\((.+)\))?$", re.DOTALL)


def parse_rule(rule_str: str) -> PermissionRuleValue:
    """Parse 'Bash(git *)' → PermissionRuleValue('Bash', 'git *')."""
    rule_str = rule_str.strip()
    m = _RULE_RE.match(rule_str)
    if not m:
        raise ValueError(f"Invalid permission rule: {rule_str!r}")
    tool_name = m.group(1)
    content = m.group(2)
    # unescape parentheses
    if content is not None:
        content = content.replace(r"\(", "(").replace(r"\)", ")")
    return PermissionRuleValue(tool_name=tool_name, content=content)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _extract_match_string(tool_name: str, tool_input: dict) -> str | None:
    """Extract the string to match against rule content, based on tool type."""
    if tool_name == "Bash":
        return tool_input.get("command", "")
    if tool_name in ("Read", "FileRead"):
        return tool_input.get("file_path", "")
    if tool_name in ("Edit", "FileEdit"):
        return tool_input.get("file_path", "")
    if tool_name in ("Write", "FileWrite"):
        return tool_input.get("file_path", "")
    if tool_name in ("Glob", "GlobTool"):
        return tool_input.get("pattern", "")
    if tool_name in ("Grep", "GrepTool"):
        return tool_input.get("pattern", "")
    return None


def match_rule(rule: PermissionRuleValue, tool_name: str, tool_input: dict) -> bool:
    """Check if a rule matches the given tool invocation."""
    if rule.tool_name != tool_name:
        return False
    # no content pattern → matches all invocations of this tool
    if rule.content is None:
        return True
    match_str = _extract_match_string(tool_name, tool_input)
    if match_str is None:
        return False
    # prefix match for commands, glob match for paths
    if tool_name == "Bash":
        return match_str.startswith(rule.content.rstrip("*").rstrip(" ")) if rule.content.endswith("*") else match_str.startswith(rule.content)
    # file path: glob match
    return fnmatch.fnmatch(match_str, rule.content)


# ---------------------------------------------------------------------------
# Rule generation
# ---------------------------------------------------------------------------

def _generalize_bash_command(cmd: str) -> str:
    """Extract a reusable prefix from a bash command.

    'git add main.py' → 'git add *'
    'npm install foo'  → 'npm install *'
    'python main.py'   → 'python *'
    'ls -la'           → 'ls *'
    """
    parts = cmd.strip().split()
    if not parts:
        return cmd
    # two-word commands that form a logical unit
    two_word_cmds = {
        "git", "npm", "pip", "cargo", "docker", "kubectl",
        "conda", "brew", "apt", "dnf", "yarn", "pnpm",
    }
    if len(parts) >= 2 and parts[0] in two_word_cmds:
        return f"{parts[0]} {parts[1]} *"
    return f"{parts[0]} *"


def build_rule_str(tool_name: str, tool_input: dict) -> str:
    """Build a rule string from a tool invocation with appropriate granularity.

    - Bash: generalize to prefix + wildcard, e.g. 'Bash(git add *)'
    - File tools: use directory glob, e.g. 'Write(src/*)'
    """
    content = _extract_match_string(tool_name, tool_input)
    if not content:
        return tool_name
    if tool_name == "Bash":
        return f"{tool_name}({_generalize_bash_command(content)})"
    # file tools: generalize to parent directory
    import os
    parent = os.path.dirname(content)
    if parent:
        return f"{tool_name}({parent}/*)"
    return f"{tool_name}({content})"


# ---------------------------------------------------------------------------
# Bulk helpers
# ---------------------------------------------------------------------------

def load_rules_from_settings(
    allow_list: list[str],
    deny_list: list[str],
    source: str,
) -> list[PermissionRule]:
    """Parse allow/deny string lists into PermissionRule objects."""
    rules: list[PermissionRule] = []
    for raw in deny_list:
        try:
            rules.append(PermissionRule(source=source, behavior="deny", value=parse_rule(raw)))
        except ValueError:
            pass
    for raw in allow_list:
        try:
            rules.append(PermissionRule(source=source, behavior="allow", value=parse_rule(raw)))
        except ValueError:
            pass
    return rules
