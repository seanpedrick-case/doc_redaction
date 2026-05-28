"""Tests for per-session Pi workspace paths."""

import sys
from pathlib import Path
from types import ModuleType

import pytest

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

if "gradio" not in sys.modules:
    _gr = ModuleType("gradio")
    _gr.FileExplorer = lambda **kwargs: kwargs  # type: ignore[misc]
    sys.modules["gradio"] = _gr

from session_workspace import (  # noqa: E402
    ensure_session_workspace,
    resolve_session_hash,
    sanitize_session_id,
    session_workspace_dir,
    workspace_context_prefix,
)


@pytest.fixture
def workspace_base(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    base.mkdir()
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")
    monkeypatch.delenv("PI_DEPLOYMENT_PROFILE", raising=False)
    return base


def test_session_workspace_dir_uses_hash(workspace_base):
    path = session_workspace_dir("abc123session")
    assert path == workspace_base / "abc123session"


def test_ensure_session_workspace_creates_directory(workspace_base):
    created = ensure_session_workspace("user_session_1")
    assert created.is_dir()


def test_sanitize_session_id_strips_unsafe_chars():
    assert sanitize_session_id("foo@bar/baz") == "foo_bar_baz"


def test_workspace_context_prefix_includes_path(workspace_base):
    prefix = workspace_context_prefix("abc123")
    assert (workspace_base / "abc123").as_posix() in prefix
    assert "mandatory" in prefix.lower()


class _FakeRequest:
    session_hash = "gradio_session_xyz"
    username = None


def test_resolve_session_hash_from_gradio_request():
    assert resolve_session_hash(_FakeRequest()) == "gradio_session_xyz"
