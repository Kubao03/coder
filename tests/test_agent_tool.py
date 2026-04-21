import pytest
from pathlib import Path
from unittest.mock import patch

from agent_types import ToolResult
from agent_services import AgentServices
from context import AgentContext
from hooks import HookRunner, register_builtin_hooks
from permissions import PermissionManager
from settings import Settings
from tools.agent_tool import AgentTool, _filter_tools, _SubagentContext
from tools.bash import BashTool
from tools.file_read import FileReadTool
from subagents.registry import AGENT_REGISTRY, AgentDefinition
from subagents.general_purpose import GENERAL_PURPOSE
from services import worktree as wt

from tests.test_agent_loop import make_text_stream, make_tool_stream


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def services(tmp_path):
    settings = Settings(
        permissions={"allow": [], "deny": []},
        _user_raw={},
        _project_raw={},
    )
    pm = PermissionManager(settings, cwd=str(tmp_path))
    hooks = HookRunner({}, cwd=str(tmp_path), session_id="test")
    register_builtin_hooks(hooks)
    return AgentServices(permissions=pm, hooks=hooks, settings=settings)


@pytest.fixture
def pm(tmp_path):
    settings = Settings(
        permissions={"allow": [], "deny": []},
        _user_raw={},
        _project_raw={},
    )
    return PermissionManager(settings, cwd=str(tmp_path))


@pytest.fixture
def ctx(tmp_path, services):
    tools = [BashTool(), FileReadTool(), AgentTool()]
    return AgentContext(cwd=str(tmp_path), tools=tools, services=services)


@pytest.fixture
def stub_worktree(monkeypatch, tmp_path):
    """Stub out services.worktree so E2E tests don't hit real git.

    - find_git_root: pretends cwd is a git root
    - create_worktree: returns a fake Worktree pointing at a tmp path
    - has_changes: returns False (triggers the remove-worktree path)
    - remove_worktree: no-op
    """
    def fake_find_git_root(cwd):
        return Path(cwd)

    def fake_create_worktree(repo_root, tag):
        path = tmp_path / f"wt-{tag}"
        path.mkdir(exist_ok=True)
        return wt.Worktree(path=path, branch=f"coder/subagent/{tag}", repo_root=repo_root)

    def fake_has_changes(w):
        return False

    def fake_remove_worktree(w):
        pass

    monkeypatch.setattr(wt, "find_git_root", fake_find_git_root)
    monkeypatch.setattr(wt, "create_worktree", fake_create_worktree)
    monkeypatch.setattr(wt, "has_changes", fake_has_changes)
    monkeypatch.setattr(wt, "remove_worktree", fake_remove_worktree)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_general_purpose_registered(self):
        assert "general-purpose" in AGENT_REGISTRY
        d = AGENT_REGISTRY["general-purpose"]
        assert isinstance(d, AgentDefinition)
        assert d.tools == ["*"]

    def test_general_purpose_matches_exported(self):
        assert AGENT_REGISTRY["general-purpose"] is GENERAL_PURPOSE

    def test_three_presets_registered(self):
        assert set(AGENT_REGISTRY.keys()) >= {"general-purpose", "Explore", "Plan"}

    def test_explore_is_read_only_tool_set(self):
        d = AGENT_REGISTRY["Explore"]
        # No write tools should appear in the allow list.
        assert "Edit" not in d.tools
        assert "Write" not in d.tools
        assert "Agent" not in d.tools
        assert set(d.tools) == {"Bash", "Read", "Glob", "Grep"}

    def test_plan_is_read_only_tool_set(self):
        d = AGENT_REGISTRY["Plan"]
        assert set(d.tools) == {"Bash", "Read", "Glob", "Grep"}

    def test_plan_prompt_requires_critical_files_section(self):
        # Plan's differentiator from Explore: required output format.
        assert "Critical Files for Implementation" in AGENT_REGISTRY["Plan"].system_prompt
        assert "Critical Files for Implementation" not in AGENT_REGISTRY["Explore"].system_prompt


# ---------------------------------------------------------------------------
# Tool filtering (recursion prevention + allow list)
# ---------------------------------------------------------------------------

class TestFilterTools:
    def test_wildcard_excludes_agent_tool(self):
        parent = [BashTool(), FileReadTool(), AgentTool()]
        filtered = _filter_tools(parent, ["*"])
        names = {t.name for t in filtered}
        assert "Agent" not in names
        assert names == {"Bash", "Read"}

    def test_allow_list_is_respected(self):
        parent = [BashTool(), FileReadTool(), AgentTool()]
        filtered = _filter_tools(parent, ["Read"])
        assert [t.name for t in filtered] == ["Read"]

    def test_allow_list_cannot_resurrect_agent_tool(self):
        # Even if an allow list includes "Agent", it must be filtered out.
        parent = [BashTool(), AgentTool()]
        filtered = _filter_tools(parent, ["Agent", "Bash"])
        names = {t.name for t in filtered}
        assert "Agent" not in names
        assert "Bash" in names

    def test_explore_allow_list_excludes_write_tools(self):
        # Simulate parent with the full tool set; apply Explore's whitelist.
        from tools.file_edit import FileEditTool
        from tools.file_write import FileWriteTool
        from tools.glob_tool import GlobTool
        from tools.grep_tool import GrepTool
        parent = [
            BashTool(), FileReadTool(), FileEditTool(), FileWriteTool(),
            GlobTool(), GrepTool(), AgentTool(),
        ]
        filtered = _filter_tools(parent, AGENT_REGISTRY["Explore"].tools)
        names = {t.name for t in filtered}
        assert names == {"Bash", "Read", "Glob", "Grep"}
        assert "Edit" not in names and "Write" not in names


# ---------------------------------------------------------------------------
# Schema + description
# ---------------------------------------------------------------------------

class TestSchema:
    def test_description_lists_registered_agents(self):
        desc = AgentTool().description
        assert "general-purpose" in desc

    def test_schema_has_required_fields(self):
        schema = AgentTool().input_schema
        assert set(schema["required"]) == {"description", "prompt"}
        assert "subagent_type" in schema["properties"]
        assert "general-purpose" in schema["properties"]["subagent_type"]["enum"]

    def test_not_read_only(self):
        tool = AgentTool()
        assert not tool.is_read_only({})

    def test_concurrent_safe(self):
        # Multiple AgentTool dispatches must be able to run in parallel —
        # otherwise fan-out (e.g. "explore 3 areas at once") is serial and slow.
        tool = AgentTool()
        assert tool.is_concurrent_safe({}) is True


# ---------------------------------------------------------------------------
# Subagent context
# ---------------------------------------------------------------------------

class TestSubagentContext:
    def test_override_prompt_returned(self, tmp_path):
        ctx = _SubagentContext(
            cwd=str(tmp_path),
            tools=[],
            messages=[],
            system_prompt_override="hello sub-agent",
        )
        assert ctx.build_system_prompt() == "hello sub-agent"


# ---------------------------------------------------------------------------
# AgentTool.call — error paths (sync; don't spin up a real sub-loop)
# ---------------------------------------------------------------------------

class TestCallErrors:
    @pytest.mark.asyncio
    async def test_empty_prompt(self, ctx):
        out = await AgentTool().call({"prompt": "  ", "subagent_type": "general-purpose"}, ctx)
        assert out.is_error
        assert "prompt" in out.data.lower()

    @pytest.mark.asyncio
    async def test_unknown_subagent_type(self, ctx):
        out = await AgentTool().call(
            {"prompt": "do the thing", "subagent_type": "nonexistent"}, ctx,
        )
        assert out.is_error
        assert "nonexistent" in out.data


# ---------------------------------------------------------------------------
# AgentTool.call — end-to-end with a stubbed AgentLoop
# ---------------------------------------------------------------------------

class _FakeAgentLoop:
    """Stand-in for AgentLoop that captures its init args and yields a scripted event."""

    instances: list["_FakeAgentLoop"] = []
    scripted_events: list = []  # optional extra events to yield before TurnComplete

    def __init__(self, context, services, session=None):
        self.context = context
        self.services = services
        self.session = session
        self.user_messages: list[str] = []
        _FakeAgentLoop.instances.append(self)

    async def run_stream(self, user_message):
        from agent_types import TurnComplete
        self.user_messages.append(user_message)
        for ev in _FakeAgentLoop.scripted_events:
            yield ev
        yield TurnComplete(text=f"report for: {user_message}")


class TestCallE2E:
    def setup_method(self):
        _FakeAgentLoop.instances.clear()
        _FakeAgentLoop.scripted_events = []

    @pytest.mark.asyncio
    async def test_simple_dispatch_returns_final_text(self, ctx, monkeypatch, stub_worktree):
        # Patch the AgentLoop symbol that _run_subagent imports lazily.
        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _FakeAgentLoop)

        out = await AgentTool().call(
            {"prompt": "please investigate", "subagent_type": "general-purpose"}, ctx,
        )

        assert isinstance(out, ToolResult)
        assert not out.is_error
        assert out.data == "report for: please investigate"

    @pytest.mark.asyncio
    async def test_child_loop_receives_scoped_context(self, ctx, services, monkeypatch, stub_worktree):
        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _FakeAgentLoop)

        await AgentTool().call(
            {"prompt": "go", "subagent_type": "general-purpose"}, ctx,
        )

        assert len(_FakeAgentLoop.instances) == 1
        child = _FakeAgentLoop.instances[0]
        # Sub-agent inherits permission manager from the parent services.
        assert child.services.permissions is services.permissions
        # With isolation="worktree", child cwd is the worktree path, not parent cwd.
        assert child.context.cwd != ctx.cwd
        # Sub-agent gets its own messages list — independent from parent.
        assert child.context.messages == []
        assert child.context.messages is not ctx.messages
        # AgentTool is filtered out of the sub-agent's tool list.
        names = {t.name for t in child.context.tools}
        assert "Agent" not in names
        # The sub-agent's system prompt is the preset, not the parent's.
        assert child.context.build_system_prompt() == GENERAL_PURPOSE.system_prompt
        # No session file for MVP sub-agents.
        assert child.session is None

    @pytest.mark.asyncio
    async def test_parent_messages_untouched(self, ctx, monkeypatch, stub_worktree):
        ctx.messages.append({"role": "user", "content": "parent turn"})
        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _FakeAgentLoop)

        await AgentTool().call(
            {"prompt": "go", "subagent_type": "general-purpose"}, ctx,
        )

        assert ctx.messages == [{"role": "user", "content": "parent turn"}]

    @pytest.mark.asyncio
    async def test_listener_receives_start_tool_end(self, ctx, services, monkeypatch):
        """The parent UI can watch sub-agent progress via services.subagent_listener."""
        from agent_types import ToolUseStart
        from dataclasses import replace

        events_seen: list[tuple] = []
        listener = lambda preset, inv_id, kind, payload: events_seen.append(
            (preset, kind, payload)
        )
        # Swap listener into services so AgentTool can read it.
        ctx.services = replace(services, subagent_listener=listener)

        # Script: sub-agent fires one ToolUseStart (Grep) before TurnComplete.
        _FakeAgentLoop.scripted_events = [
            ToolUseStart(name="Grep", id="t1", input={"pattern": "foo"}),
        ]
        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _FakeAgentLoop)

        await AgentTool().call(
            {
                "prompt": "search for foo",
                "subagent_type": "Explore",
                "description": "find foo callers",
            },
            ctx,
        )

        kinds = [e[1] for e in events_seen]
        assert kinds == ["start", "tool", "end"]
        assert events_seen[0][0] == "Explore"
        assert events_seen[0][2]["description"] == "find foo callers"
        assert events_seen[1][2]["name"] == "Grep"
        assert events_seen[1][2]["input"] == {"pattern": "foo"}
        assert events_seen[2][2]["error"] is False

    @pytest.mark.asyncio
    async def test_listener_end_marked_error_on_exception(self, ctx, services, monkeypatch, stub_worktree):
        from dataclasses import replace

        events_seen: list[tuple] = []
        ctx.services = replace(services, subagent_listener=lambda p, i, k, pl: events_seen.append((p, k, pl)))

        class _Boom:
            def __init__(self, *a, **kw):
                pass

            async def run_stream(self, _):
                raise RuntimeError("boom")
                yield  # pragma: no cover

        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _Boom)

        out = await AgentTool().call(
            {"prompt": "go", "subagent_type": "general-purpose"}, ctx,
        )

        assert out.is_error
        # Must still emit a terminal "end" event so UI can close the progress line.
        kinds = [e[1] for e in events_seen]
        assert kinds[-1] == "end"
        assert events_seen[-1][2]["error"] is True

    @pytest.mark.asyncio
    async def test_listener_errors_swallowed(self, ctx, services, monkeypatch, stub_worktree):
        """A broken listener must not crash sub-agent execution."""
        from dataclasses import replace
        ctx.services = replace(services, subagent_listener=lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad listener")))
        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _FakeAgentLoop)

        out = await AgentTool().call(
            {"prompt": "go", "subagent_type": "general-purpose"}, ctx,
        )
        assert not out.is_error

    @pytest.mark.asyncio
    async def test_exception_surfaces_as_error_result(self, ctx, monkeypatch, stub_worktree):
        class _Boom:
            def __init__(self, *a, **kw):
                pass

            async def run_stream(self, _):
                raise RuntimeError("boom")
                yield  # pragma: no cover — make this an async generator

        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _Boom)

        out = await AgentTool().call(
            {"prompt": "go", "subagent_type": "general-purpose"}, ctx,
        )

        assert out.is_error
        assert "boom" in out.data


# ---------------------------------------------------------------------------
# Worktree isolation — AgentTool-level integration (with stubbed git)
# ---------------------------------------------------------------------------

class TestWorktreeIsolation:
    def setup_method(self):
        _FakeAgentLoop.instances.clear()
        _FakeAgentLoop.scripted_events = []

    @pytest.mark.asyncio
    async def test_general_purpose_runs_in_worktree_cwd(
        self, ctx, pm, monkeypatch, stub_worktree,
    ):
        """Child AgentLoop's context.cwd must be the worktree path, not parent cwd."""
        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _FakeAgentLoop)

        await AgentTool().call(
            {"prompt": "go", "subagent_type": "general-purpose"}, ctx,
        )

        child = _FakeAgentLoop.instances[0]
        assert child.context.cwd != ctx.cwd
        assert "wt-" in child.context.cwd

    @pytest.mark.asyncio
    async def test_explore_does_not_create_worktree(self, ctx, monkeypatch):
        """Read-only presets skip worktree creation entirely — no git calls made."""

        def _should_not_be_called(*a, **kw):
            raise AssertionError("worktree API hit for non-isolated preset")

        monkeypatch.setattr(wt, "create_worktree", _should_not_be_called)
        monkeypatch.setattr(wt, "find_git_root", _should_not_be_called)

        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _FakeAgentLoop)

        await AgentTool().call(
            {"prompt": "search", "subagent_type": "Explore"}, ctx,
        )

        child = _FakeAgentLoop.instances[0]
        # Explore inherits parent cwd — no isolation.
        assert child.context.cwd == ctx.cwd

    @pytest.mark.asyncio
    async def test_non_git_cwd_hard_errors(self, ctx, monkeypatch):
        """general-purpose in a non-git dir returns an error result, doesn't silently fall back."""
        monkeypatch.setattr(wt, "find_git_root", lambda cwd: None)
        # If the code accidentally proceeds past the git-root check, fail loudly.
        monkeypatch.setattr(wt, "create_worktree", lambda *a, **kw: pytest.fail("should not create"))

        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _FakeAgentLoop)

        out = await AgentTool().call(
            {"prompt": "go", "subagent_type": "general-purpose"}, ctx,
        )

        assert out.is_error
        assert "not inside a git repository" in out.data

    @pytest.mark.asyncio
    async def test_worktree_event_fired(self, ctx, services, monkeypatch, stub_worktree):
        from dataclasses import replace
        events_seen: list[tuple] = []
        ctx.services = replace(services, subagent_listener=lambda preset, inv_id, kind, payload: events_seen.append(
            (kind, payload)
        ))
        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _FakeAgentLoop)

        await AgentTool().call(
            {"prompt": "go", "subagent_type": "general-purpose"}, ctx,
        )

        kinds = [e[0] for e in events_seen]
        assert "worktree" in kinds
        # The worktree event carries path + branch.
        wt_payload = next(p for k, p in events_seen if k == "worktree")
        assert "path" in wt_payload and "branch" in wt_payload
        assert wt_payload["branch"].startswith("coder/subagent/")

    @pytest.mark.asyncio
    async def test_clean_worktree_is_removed(self, ctx, monkeypatch, stub_worktree):
        """If sub-agent made no changes, worktree is removed and the report has no trailing note."""
        removes: list = []
        monkeypatch.setattr(wt, "remove_worktree", lambda w: removes.append(w))
        # stub_worktree already sets has_changes -> False

        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _FakeAgentLoop)

        out = await AgentTool().call(
            {"prompt": "go", "subagent_type": "general-purpose"}, ctx,
        )

        assert not out.is_error
        assert len(removes) == 1
        assert "worktree" not in out.data.lower()

    @pytest.mark.asyncio
    async def test_dirty_worktree_is_kept_and_reported(self, ctx, monkeypatch, stub_worktree):
        """If sub-agent made changes, worktree is preserved and path/branch appear in report."""
        monkeypatch.setattr(wt, "has_changes", lambda w: True)
        removes: list = []
        monkeypatch.setattr(wt, "remove_worktree", lambda w: removes.append(w))

        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _FakeAgentLoop)

        out = await AgentTool().call(
            {"prompt": "go", "subagent_type": "general-purpose"}, ctx,
        )

        assert not out.is_error
        assert removes == []  # not cleaned up
        assert "coder/subagent/" in out.data  # branch mentioned
        assert "preserved" in out.data.lower()

    @pytest.mark.asyncio
    async def test_worktree_removed_on_subagent_exception(self, ctx, monkeypatch, stub_worktree):
        """Even if sub-agent raises, the worktree cleanup still runs (finally block)."""
        removes: list = []
        monkeypatch.setattr(wt, "remove_worktree", lambda w: removes.append(w))

        class _Boom:
            def __init__(self, *a, **kw):
                pass

            async def run_stream(self, _):
                raise RuntimeError("boom")
                yield  # pragma: no cover

        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _Boom)

        out = await AgentTool().call(
            {"prompt": "go", "subagent_type": "general-purpose"}, ctx,
        )

        assert out.is_error
        assert len(removes) == 1, "worktree must be cleaned up on sub-agent failure"


# ---------------------------------------------------------------------------
# services.worktree — real git integration tests
# ---------------------------------------------------------------------------

import subprocess
import shutil


def _git_available() -> bool:
    return shutil.which("git") is not None


@pytest.fixture
def git_repo(tmp_path):
    """Create a real git repo with one commit; return its root path."""
    if not _git_available():
        pytest.skip("git not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    def run(*args):
        subprocess.run(args, cwd=repo, check=True, capture_output=True)
    run("git", "init", "-q", "-b", "main")
    run("git", "config", "user.email", "t@t.t")
    run("git", "config", "user.name", "t")
    (repo / "README").write_text("hi\n")
    run("git", "add", "README")
    run("git", "commit", "-q", "-m", "init")
    return repo


class TestWorktreeHelper:
    def test_find_git_root_inside_repo(self, git_repo):
        root = wt.find_git_root(git_repo)
        # Resolve both sides for macOS /private/var vs /var symlink.
        assert root is not None and root.resolve() == git_repo.resolve()

    def test_find_git_root_in_subdir(self, git_repo):
        sub = git_repo / "sub"
        sub.mkdir()
        root = wt.find_git_root(sub)
        assert root is not None and root.resolve() == git_repo.resolve()

    def test_find_git_root_outside_repo(self, tmp_path):
        if not _git_available():
            pytest.skip("git not installed")
        outside = tmp_path / "not-a-repo"
        outside.mkdir()
        assert wt.find_git_root(outside) is None

    def test_create_and_remove_worktree(self, git_repo):
        w = wt.create_worktree(git_repo, "abc123")
        try:
            assert w.path.exists()
            assert w.branch == "coder/subagent/abc123"
            # Worktree starts clean (HEAD checkout, no modifications).
            assert not wt.has_changes(w)
        finally:
            wt.remove_worktree(w)
        # After removal the directory is gone and the branch is deleted.
        assert not w.path.exists()
        result = subprocess.run(
            ["git", "branch", "--list", w.branch],
            cwd=git_repo, capture_output=True, text=True,
        )
        assert w.branch not in result.stdout

    def test_has_changes_detects_new_file(self, git_repo):
        w = wt.create_worktree(git_repo, "dirty")
        try:
            (w.path / "new.txt").write_text("hello\n")
            assert wt.has_changes(w) is True
        finally:
            wt.remove_worktree(w)

    def test_has_changes_detects_modified_file(self, git_repo):
        w = wt.create_worktree(git_repo, "mod")
        try:
            (w.path / "README").write_text("changed\n")
            assert wt.has_changes(w) is True
        finally:
            wt.remove_worktree(w)

    def test_create_worktree_reuses_stale_branch(self, git_repo):
        """-B lets us recreate when a stale branch exists from a crashed run."""
        w1 = wt.create_worktree(git_repo, "stale")
        wt.remove_worktree(w1)
        # Simulate a stale branch left behind (no worktree dir).
        subprocess.run(
            ["git", "branch", "coder/subagent/stale", "HEAD"],
            cwd=git_repo, check=True, capture_output=True,
        )
        w2 = wt.create_worktree(git_repo, "stale")
        try:
            assert w2.path.exists()
        finally:
            wt.remove_worktree(w2)
