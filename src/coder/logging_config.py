"""Logging configuration for the coder agent.

Call setup_logging() once at CLI entry. All internal modules use:
    logger = logging.getLogger("coder.<module>")

Environment:
    CODER_LOG_LEVEL  Console verbosity (DEBUG/INFO/WARNING/ERROR). Default: WARNING.

File output:
    ~/.coder/logs/<session_id>.log  — DEBUG level, every session.
"""

import logging
import os
from pathlib import Path


def setup_logging(session_id: str = "") -> None:
    """Configure console + file handlers for the 'coder' logger hierarchy."""
    level_name = os.environ.get("CODER_LOG_LEVEL", "WARNING").upper()
    console_level = getattr(logging, level_name, logging.WARNING)

    root = logging.getLogger("coder")
    root.setLevel(logging.DEBUG)  # handlers decide what to surface
    root.propagate = False

    # Clear any handlers from a previous call (e.g. in tests).
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    root.addHandler(console)

    if session_id:
        log_dir = Path.home() / ".coder" / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / f"{session_id}.log", encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            root.addHandler(fh)
        except OSError:
            # Never let a logging failure surface to the user.
            pass
