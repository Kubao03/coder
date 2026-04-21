"""Entry point shim for `coder` CLI command (installed via pip/pipx)."""

import asyncio
from .cli.repl import repl


def main_entry() -> None:
    """Entry point for the `coder` CLI command."""
    asyncio.run(repl())


if __name__ == "__main__":
    main_entry()
