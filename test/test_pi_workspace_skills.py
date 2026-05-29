"""Tests for workspace-scoped Pi skills sync."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

if "gradio" not in sys.modules:
    import types

    _gr = types.ModuleType("gradio")
    _gr.FileExplorer = lambda **kwargs: kwargs  # type: ignore[misc]
    sys.modules["gradio"] = _gr


@pytest.fixture
def workspace_layout(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    skills = repo / "skills" / "doc-redaction-app"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(
        "---\nname: doc-redaction-app\ndescription: test\n---\n", encoding="utf-8"
    )
    (repo / "skills" / "Example prompt partnership.txt").write_text(
        "template", encoding="utf-8"
    )

    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(ws))
    monkeypatch.setenv("PI_WORKDIR", str(repo))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")
    monkeypatch.delenv("PI_SKILLS_RESYNC", raising=False)

    import importlib

    import pi_workspace_skills as pws
    import session_workspace as sw

    importlib.reload(sw)
    importlib.reload(pws)
    return repo, ws, pws


def test_sync_repo_skills_to_workspace(workspace_layout):
    repo, ws, pws = workspace_layout
    dest = pws.sync_repo_skills_to_workspace(force=True)
    assert dest == ws / ".pi" / "skills"
    assert (dest / "doc-redaction-app" / "SKILL.md").is_file()
    assert (dest / "Example prompt partnership.txt").read_text(
        encoding="utf-8"
    ) == "template"
    assert (ws / ".pi" / "settings.json").is_file()


def test_pi_rpc_cwd_uses_session_subfolder(workspace_layout, monkeypatch):
    _repo, ws, pws = workspace_layout
    import session_workspace as sw

    monkeypatch.setattr(sw, "SESSION_OUTPUT_FOLDER", False)
    session = ws / "abc123"
    session.mkdir()
    cwd = pws.pi_rpc_cwd("abc123")
    assert Path(cwd) == session.resolve()


def test_pi_rpc_args_disables_discovery(workspace_layout):
    _repo, _ws, pws = workspace_layout
    pws.sync_repo_skills_to_workspace(force=True)
    args = pws.pi_rpc_args()
    assert args[0] == "--no-skills"
    assert "--skill" in args
    assert Path(args[args.index("--skill") + 1]).name == "skills"
