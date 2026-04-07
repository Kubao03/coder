from tools.base import Tool

ALWAYS_DENY = ("rm -rf /", "rm -rf ~", ":(){ :|:& };:")
ALWAYS_ALLOW = ("Read", "Glob", "Grep")


class PermissionManager:
    def __init__(self):
        self._session_allowed: set[str] = set()

    def check(self, tool: Tool, args: dict) -> tuple[bool, str]:
        """Return (allowed, reason). Does NOT prompt — purely decision logic."""
        # 1. Always deny dangerous commands
        if tool.name == "Bash":
            cmd = args.get("command", "")
            if any(cmd.strip().startswith(d) for d in ALWAYS_DENY):
                return False, "denied: dangerous command"

        # 2. Always allow safe tools
        if tool.name in ALWAYS_ALLOW:
            return True, "auto-allowed: read-only tool"

        # 3. Read-only operations
        if tool.is_read_only(args):
            return True, "auto-allowed: read-only"

        # 4. Session-level approval
        key = _session_key(tool, args)
        if key in self._session_allowed:
            return True, "auto-allowed: session approval"

        return None, "needs confirmation"

    def ask(self, tool: Tool, args: dict) -> bool:
        """Prompt the user and return whether to allow. Updates session state."""
        print(f"\nAllow {tool.name}({args})? [y]es / [a]lways / [n]o: ", end="", flush=True)
        response = input().strip().lower()
        if response == "a":
            self._session_allowed.add(_session_key(tool, args))
            return True
        return response == "y"

    def is_allowed(self, tool: Tool, args: dict) -> bool:
        """Full permission check including user prompt if needed."""
        allowed, _ = self.check(tool, args)
        if allowed is True:
            return True
        if allowed is False:
            return False
        return self.ask(tool, args)


def _session_key(tool: Tool, args: dict) -> str:
    return tool.name
