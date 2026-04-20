import pytest
from unittest.mock import patch

from agent_types import ToolResult
from context import AgentContext
from permissions import PermissionManager
from settings import Settings
from tools.agent_tool import AgentTool, _filter_tools, _SubagentContext
from tools.bash import BashTool
from tools.file_read import FileReadTool
from subagents.registry import AGENT_REGISTRY, AgentDefinition
from subagents.general_purpose import GENERAL_PURPOSE

from tests.test_agent_loop import make_text_stream, make_tool_stream


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx(tmp_path):
    tools = [BashTool(), FileReadTool(), AgentTool()]
    return AgentContext(cwd=str(tmp_path), tools=tools)


@pytest.fixture
def pm(tmp_path):
    settings = Settings(
        permissions={"allow": [], "deny": []},
        _user_raw={},
        _project_raw={},
    )
    return PermissionManager(settings, cwd=str(tmp_path))


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
            settings=None,
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

    def __init__(self, context, pm, session=None):
        self.context = context
        self.pm = pm
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
    async def test_simple_dispatch_returns_final_text(self, ctx, pm, monkeypatch):
        ctx._pm = pm
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
    async def test_child_loop_receives_scoped_context(self, ctx, pm, monkeypatch):
        ctx._pm = pm
        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _FakeAgentLoop)

        await AgentTool().call(
            {"prompt": "go", "subagent_type": "general-purpose"}, ctx,
        )

        assert len(_FakeAgentLoop.instances) == 1
        child = _FakeAgentLoop.instances[0]
        # Sub-agent inherits cwd + permission manager.
        assert child.context.cwd == ctx.cwd
        assert child.pm is pm
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
    async def test_parent_messages_untouched(self, ctx, pm, monkeypatch):
        ctx._pm = pm
        ctx.messages.append({"role": "user", "content": "parent turn"})
        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _FakeAgentLoop)

        await AgentTool().call(
            {"prompt": "go", "subagent_type": "general-purpose"}, ctx,
        )

        assert ctx.messages == [{"role": "user", "content": "parent turn"}]

    @pytest.mark.asyncio
    async def test_listener_receives_start_tool_end(self, ctx, pm, monkeypatch):
        """The parent UI can watch sub-agent progress via context.subagent_listener."""
        from agent_types import ToolUseStart
        ctx._pm = pm

        events_seen: list[tuple] = []
        ctx.subagent_listener = lambda preset, inv_id, kind, payload: events_seen.append(
            (preset, kind, payload)
        )

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
    async def test_listener_end_marked_error_on_exception(self, ctx, pm, monkeypatch):
        ctx._pm = pm
        events_seen: list[tuple] = []
        ctx.subagent_listener = lambda preset, inv_id, kind, payload: events_seen.append(
            (preset, kind, payload)
        )

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
    async def test_listener_errors_swallowed(self, ctx, pm, monkeypatch):
        """A broken listener must not crash sub-agent execution."""
        ctx._pm = pm
        ctx.subagent_listener = lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad listener"))
        import agent_loop
        monkeypatch.setattr(agent_loop, "AgentLoop", _FakeAgentLoop)

        out = await AgentTool().call(
            {"prompt": "go", "subagent_type": "general-purpose"}, ctx,
        )
        assert not out.is_error

    @pytest.mark.asyncio
    async def test_exception_surfaces_as_error_result(self, ctx, pm, monkeypatch):
        ctx._pm = pm

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
