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


def test_sync_workspace_helpers_copies_remote_redaction(workspace_layout):
    _repo, _ws, pws = workspace_layout
    helpers = pws.sync_workspace_helpers()
    helper_file = helpers / "remote_redaction.py"
    assert helper_file.is_file()
    assert "fetch_redaction_files" in helper_file.read_text(encoding="utf-8")


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

    monkeypatch.delenv("SESSION_OUTPUT_FOLDER", raising=False)
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


def test_resync_overwrites_readonly_skills(workspace_layout, monkeypatch):
    repo, ws, pws = workspace_layout
    config = repo / "skills" / "config"
    config.mkdir(parents=True)
    (config / "app_config.env").write_text("OLD=1\n", encoding="utf-8")

    dest = pws.sync_repo_skills_to_workspace(force=True)
    assert (dest / "config" / "app_config.env").read_text(encoding="utf-8") == "OLD=1\n"

    (config / "app_config.env").write_text("NEW=2\n", encoding="utf-8")
    monkeypatch.setenv("PI_SKILLS_RESYNC", "true")
    dest = pws.sync_repo_skills_to_workspace()
    assert (dest / "config" / "app_config.env").read_text(encoding="utf-8") == "NEW=2\n"


def test_sync_skips_archive_attempts_and_large_blobs(workspace_layout):
    repo, ws, pws = workspace_layout
    archive = repo / "skills" / "example_prompts" / "archive_attempts"
    archive.mkdir(parents=True)
    (archive / "blob.b64.txt").write_text("x" * 2000, encoding="utf-8")
    huge = repo / "skills" / "huge_skill.md"
    huge.write_bytes(b"x" * (600 * 1024))

    dest = pws.sync_repo_skills_to_workspace(force=True)
    assert not (dest / "example_prompts" / "archive_attempts").exists()
    assert not (dest / "huge_skill.md").exists()
    assert (dest / "doc-redaction-app" / "SKILL.md").is_file()


def test_hf_space_deployment_skill_written_before_readonly(
    workspace_layout, monkeypatch
):
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "hf-space")
    monkeypatch.setenv("DOC_REDACTION_GRADIO_URL", "https://example-redaction.hf.space")

    import importlib

    import pi_agent_config
    import pi_workspace_skills as pws
    import redaction_prompt

    importlib.reload(pi_agent_config)
    importlib.reload(redaction_prompt)
    importlib.reload(pws)

    dest = pws.ensure_workspace_skills(force=True)
    skill = dest / "hf-space-deployment" / "SKILL.md"
    assert skill.is_file()
    text = skill.read_text(encoding="utf-8")
    assert "https://example-redaction.hf.space" in text
    assert "host.docker.internal" in text
