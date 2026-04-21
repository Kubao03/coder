"""Agent-level exceptions."""


class PermissionDeniedError(Exception):
    """Raised when the user denies a tool permission, ending the current turn."""
    pass
