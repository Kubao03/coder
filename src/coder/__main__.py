"""Entry point for `python -m coder`."""
import asyncio
from .cli.repl import repl

if __name__ == "__main__":
    asyncio.run(repl())
