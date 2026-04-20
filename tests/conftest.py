import sys
import os
import pytest

# Add project root to sys.path so `import types` finds our local types.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Isolate SessionManager from the real filesystem during tests.
#
# `SessionManager.__init__` eagerly mkdir's its project directory under
# SESSION_DIR (~/.coder/sessions/<sanitized-cwd>). Tests that construct a
# SessionManager with cwd=str(tmp_path) and later overwrite `_dir` still
# leak an empty directory into the real SESSION_DIR during the __init__
# step. Redirect SESSION_DIR per-test to a tmp path so no leak is possible.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch):
    import session
    monkeypatch.setattr(session, "SESSION_DIR", tmp_path / "_coder_sessions_isolated")
