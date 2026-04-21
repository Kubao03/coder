"""Entry point for `python -m coder` (run from the project root directory)."""
import asyncio
from main import repl

if __name__ == "__main__":
    asyncio.run(repl())
