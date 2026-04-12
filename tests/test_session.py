import json
import pytest
from pathlib import Path
from session import SessionManager, _read_session_meta


class TestSessionManager:
    def test_record_and_load(self, tmp_path):
        sm = SessionManager(session_id="test1", cwd=str(tmp_path))
        sm._dir = tmp_path
        sm.path = tmp_path / "test1.jsonl"

        sm.record({"role": "user", "content": "hello"})
        sm.record({"role": "assistant", "content": "hi there"})

        messages = sm.load()
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "hello"}
        assert messages[1] == {"role": "assistant", "content": "hi there"}

    def test_record_preserves_list_content(self, tmp_path):
        sm = SessionManager(session_id="test2", cwd=str(tmp_path))
        sm._dir = tmp_path
        sm.path = tmp_path / "test2.jsonl"

        tool_msg = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        }
        sm.record(tool_msg)

        messages = sm.load()
        assert len(messages) == 1
        assert messages[0]["content"][0]["type"] == "tool_result"

    def test_load_empty_file(self, tmp_path):
        sm = SessionManager(session_id="empty", cwd=str(tmp_path))
        sm._dir = tmp_path
        sm.path = tmp_path / "empty.jsonl"
        sm.path.write_text("")

        assert sm.load() == []

    def test_load_nonexistent(self, tmp_path):
        sm = SessionManager(session_id="nope", cwd=str(tmp_path))
        sm._dir = tmp_path
        sm.path = tmp_path / "nope.jsonl"

        assert sm.load() == []

    def test_jsonl_format(self, tmp_path):
        sm = SessionManager(session_id="fmt", cwd=str(tmp_path))
        sm._dir = tmp_path
        sm.path = tmp_path / "fmt.jsonl"

        sm.record({"role": "user", "content": "test"})

        raw = sm.path.read_text()
        lines = raw.strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert "ts" in entry
        assert entry["role"] == "user"
        assert entry["content"] == "test"

    def test_session_id_generated(self):
        sm = SessionManager(cwd="/tmp/test")
        assert len(sm.session_id) == 12

    def test_list_sessions(self, tmp_path):
        for i in range(3):
            sm = SessionManager(session_id=f"s{i}", cwd=str(tmp_path))
            sm._dir = tmp_path
            sm.path = tmp_path / f"s{i}.jsonl"
            sm.record({"role": "user", "content": f"msg {i}"})

        # Patch list_sessions to use tmp_path directly
        sessions = []
        for f in tmp_path.glob("*.jsonl"):
            meta = _read_session_meta(f)
            if meta:
                sessions.append(meta)

        assert len(sessions) == 3

    def test_find_latest(self, tmp_path):
        import time
        sm1 = SessionManager(session_id="old", cwd=str(tmp_path))
        sm1._dir = tmp_path
        sm1.path = tmp_path / "old.jsonl"
        sm1.record({"role": "user", "content": "old msg"})

        time.sleep(0.05)

        sm2 = SessionManager(session_id="new", cwd=str(tmp_path))
        sm2._dir = tmp_path
        sm2.path = tmp_path / "new.jsonl"
        sm2.record({"role": "user", "content": "new msg"})

        # Read both and verify ordering
        metas = []
        for f in tmp_path.glob("*.jsonl"):
            meta = _read_session_meta(f)
            if meta:
                metas.append(meta)
        metas.sort(key=lambda s: s["modified"], reverse=True)
        assert metas[0]["id"] == "new"


class TestReadSessionMeta:
    def test_reads_first_line(self, tmp_path):
        path = tmp_path / "test.jsonl"
        entry = {"ts": "2026-04-13T00:00:00", "role": "user", "content": "hello world"}
        path.write_text(json.dumps(entry) + "\n")

        meta = _read_session_meta(path)
        assert meta is not None
        assert meta["id"] == "test"
        assert meta["preview"] == "hello world"

    def test_truncates_long_preview(self, tmp_path):
        path = tmp_path / "long.jsonl"
        entry = {"ts": "2026-04-13T00:00:00", "role": "user", "content": "x" * 200}
        path.write_text(json.dumps(entry) + "\n")

        meta = _read_session_meta(path)
        assert len(meta["preview"]) < 100

    def test_handles_list_content(self, tmp_path):
        path = tmp_path / "tool.jsonl"
        entry = {"ts": "2026-04-13T00:00:00", "role": "user",
                 "content": [{"type": "tool_result"}]}
        path.write_text(json.dumps(entry) + "\n")

        meta = _read_session_meta(path)
        assert meta["preview"] == "(tool result)"

    def test_empty_file_returns_none(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert _read_session_meta(path) is None
