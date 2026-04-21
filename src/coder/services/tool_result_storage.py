"""Offload large tool results: persist oversized content to disk, keep only a preview + path in the message."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"

DEFAULT_THRESHOLD_CHARS = 50_000
PREVIEW_SIZE_BYTES = 2_000


# ---------------------------------------------------------------------------
# Empty-result handling
# ---------------------------------------------------------------------------

def is_empty_content(content: str) -> bool:
    """Empty string or whitespace-only content counts as empty."""
    return not content or content.strip() == ""


def empty_result_marker(tool_name: str) -> str:
    """Placeholder for empty results to prevent some models from truncating after an empty tool_result."""
    return f"({tool_name} completed with no output)"


# ---------------------------------------------------------------------------
# Preview generation
# ---------------------------------------------------------------------------

def generate_preview(content: str, max_bytes: int) -> tuple[str, bool]:
    """Take the first max_bytes, preferring to cut at a newline. Returns (preview, has_more)."""
    if len(content) <= max_bytes:
        return content, False
    truncated = content[:max_bytes]
    last_newline = truncated.rfind("\n")
    # If the newline is too far back, cut at max_bytes to avoid a tiny preview.
    cut = last_newline if last_newline > max_bytes * 0.5 else max_bytes
    return content[:cut], True


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _format_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def persist_tool_result(
    content: str,
    tool_use_id: str,
    tool_results_dir: Path,
) -> Path | None:
    """Write content to <dir>/<id>.txt and return the path. Returns None on failure.

    - If a file for this id already exists, return its path (tool_use_id is unique
      and content is deterministic for a given id).
    - Creates the directory if missing.
    """
    try:
        tool_results_dir.mkdir(parents=True, exist_ok=True)
        path = tool_results_dir / f"{tool_use_id}.txt"
        if path.exists():
            return path
        # 'x' mode guards against concurrent overwrite; already-exists falls through.
        try:
            with path.open("x", encoding="utf-8") as f:
                f.write(content)
        except FileExistsError:
            pass
        return path
    except OSError:
        return None


def build_large_result_message(
    filepath: Path,
    original_size: int,
    preview: str,
    has_more: bool,
) -> str:
    """Build the placeholder message that replaces a large tool_result (preview + file path)."""
    lines = [
        PERSISTED_OUTPUT_TAG,
        f"Output too large ({_format_size(original_size)}). "
        f"Full output saved to: {filepath}",
        "",
        f"Preview (first {_format_size(PREVIEW_SIZE_BYTES)}):",
        preview,
        "..." if has_more else "",
        PERSISTED_OUTPUT_CLOSING_TAG,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_tool_result_content(
    content: str,
    tool_name: str,
    tool_use_id: str,
    tool_results_dir: Path | None,
    threshold_chars: int = DEFAULT_THRESHOLD_CHARS,
) -> str:
    """Process tool_result content: empty marker / offload / passthrough.

    - Empty content -> `(<tool> completed with no output)`
    - Over threshold with a tool_results_dir -> persist to disk, return preview message
    - Otherwise -> return content unchanged
    """
    if not isinstance(content, str):
        return content

    if is_empty_content(content):
        return empty_result_marker(tool_name)

    if tool_results_dir is None or len(content) <= threshold_chars:
        return content

    path = persist_tool_result(content, tool_use_id, tool_results_dir)
    if path is None:
        # On write failure, fall back to sending the full content to the model
        # (better to spend tokens than to drop data).
        return content

    preview, has_more = generate_preview(content, PREVIEW_SIZE_BYTES)
    return build_large_result_message(path, len(content), preview, has_more)
