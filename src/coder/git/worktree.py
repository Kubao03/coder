"""Git worktree helpers for sub-agent isolation.

A sub-agent with isolation="worktree" runs in a separate git worktree rooted at
a sibling directory, on a fresh branch. After the sub-agent finishes:

- If it made no changes, the worktree is removed (silent cleanup).
- If it made changes, the worktree is kept and the path + branch are returned
  so the caller can review/merge.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class WorktreeError(RuntimeError):
    """Raised when a worktree operation fails (not a git repo, git missing, etc)."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Worktree:
    """A created worktree, rooted at `path` and checked out to `branch`."""

    path: Path
    branch: str
    repo_root: Path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _run_git(args: list[str], cwd: Path | str) -> subprocess.CompletedProcess:
    """Run `git <args>` under cwd. Never raises — caller inspects returncode/stderr."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def find_git_root(cwd: str | Path) -> Path | None:
    """Walk up from cwd to find the enclosing git repo root. None if not in a repo."""
    if shutil.which("git") is None:
        return None
    result = _run_git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    return Path(root) if root else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_worktree(repo_root: Path, tag: str) -> Worktree:
    """Create a worktree for a sub-agent.

    - `tag`: short identifier (e.g. the sub-agent invocation id) used in the
      branch name and directory suffix to disambiguate parallel sub-agents.
    - Branch is based on the current HEAD so the sub-agent sees the same
      starting state as the parent.
    - Worktree directory is a sibling of `repo_root` (not inside it — placing
      it inside risks git confusing it with nested repo content).
    """
    head = _run_git(["rev-parse", "HEAD"], cwd=repo_root)
    if head.returncode != 0:
        raise WorktreeError(
            f"Cannot read HEAD in {repo_root} — worktree needs at least one commit."
        )

    branch = f"coder/subagent/{tag}"
    path = repo_root.parent / f"{repo_root.name}.subagent-{tag}"

    # If a stale path/branch exists from a crashed run, reuse the branch name
    # with -B so we don't fail on "already exists".
    result = _run_git(
        ["worktree", "add", "-B", branch, str(path), "HEAD"],
        cwd=repo_root,
    )
    if result.returncode != 0:
        raise WorktreeError(
            f"git worktree add failed: {result.stderr.strip() or result.stdout.strip()}"
        )

    return Worktree(path=path, branch=branch, repo_root=repo_root)


def has_changes(worktree: Worktree) -> bool:
    """True if the worktree has any staged, unstaged, or untracked changes."""
    result = _run_git(["status", "--porcelain"], cwd=worktree.path)
    if result.returncode != 0:
        # If we can't inspect it, err on the side of "keep" so no work is lost.
        return True
    return bool(result.stdout.strip())


def remove_worktree(worktree: Worktree) -> None:
    """Remove a worktree and delete its branch.

    Idempotent and best-effort — if git errors out (e.g. user already moved
    the path), swallow the error so cleanup doesn't mask a real sub-agent
    failure upstream.
    """
    _run_git(
        ["worktree", "remove", "--force", str(worktree.path)],
        cwd=worktree.repo_root,
    )
    _run_git(["branch", "-D", worktree.branch], cwd=worktree.repo_root)
