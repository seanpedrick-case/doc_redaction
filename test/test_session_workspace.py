"""Tests for per-session Pi workspace paths."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

if "gradio" not in sys.modules:
    _gr = ModuleType("gradio")
    _gr.FileExplorer = lambda **kwargs: kwargs  # type: ignore[misc]
    sys.modules["gradio"] = _gr

from session_workspace import (  # noqa: E402
    effective_session_hash,
    ensure_session_workspace,
    prepare_session_workspace,
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


def test_session_workspace_enabled_by_default_local(tmp_path, monkeypatch):
    """Local-docker profile: session subfolders unless PI_SESSION_WORKSPACE=false."""
    base = tmp_path / "workspace"
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.delenv("PI_SESSION_WORKSPACE", raising=False)
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "local-docker")
    import session_workspace as sw

    monkeypatch.delenv("SESSION_OUTPUT_FOLDER", raising=False)
    assert sw.session_workspace_enabled() is True
    assert sw.session_workspace_dir("sess1") == base / "sess1"
    assert sw.workspace_context_prefix("sess1") != ""


def test_session_workspace_disabled_when_env_false(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "false")
    import session_workspace as sw

    monkeypatch.delenv("SESSION_OUTPUT_FOLDER", raising=False)
    assert sw.session_workspace_enabled() is False
    assert sw.session_workspace_dir("sess1") == base.resolve()
    assert sw.workspace_context_prefix("sess1") == ""


def test_session_workspace_dir_uses_hash(workspace_base):
    path = session_workspace_dir("abc123session")
    assert path == workspace_base / "abc123session"


def test_ensure_session_workspace_creates_directory(workspace_base):
    created = ensure_session_workspace("user_session_1")
    assert created.is_dir()


def test_sanitize_session_id_strips_unsafe_chars():
    assert sanitize_session_id("foo@bar/baz") == "foo@bar_baz"


def test_workspace_context_prefix_includes_path(workspace_base):
    prefix = workspace_context_prefix("abc123")
    assert (workspace_base / "abc123").as_posix() in prefix
    assert "mandatory" in prefix.lower()


class _FakeRequest:
    session_hash = "gradio_session_xyz"
    username = None


def test_effective_session_hash_uses_request_when_state_empty(
    workspace_base, monkeypatch
):
    request = _FakeRequest()
    assert effective_session_hash("", request) == "gradio_session_xyz"
    path = prepare_session_workspace("", request)[1]
    assert path == workspace_base / "gradio_session_xyz"
    assert path.is_dir()


def test_pi_session_workspace_env_true(monkeypatch, tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(ws))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")
    monkeypatch.delenv("SESSION_OUTPUT_FOLDER", raising=False)
    import importlib

    import session_workspace as sw

    importlib.reload(sw)
    assert sw.session_workspace_enabled() is True


def test_resolve_session_hash_from_gradio_request(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "tools.gradio_platform",
        SimpleNamespace(
            resolve_session_identity=lambda request: request.session_hash,
        ),
    )
    assert resolve_session_hash(_FakeRequest()) == "gradio_session_xyz"
