"""Tests for Pi workspace path bootstrap."""

from __future__ import annotations

import os
from pathlib import Path


def _import_bootstrap(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("PI_WORKSPACE_DIR", raising=False)
    import importlib
    import sys

    pi_dir = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
    if str(pi_dir) not in sys.path:
        sys.path.insert(0, str(pi_dir))
    import bootstrap_pi_config

    return importlib.reload(bootstrap_pi_config)


def test_ensure_pi_workspace_dir_defaults_to_repo_workspace(monkeypatch, tmp_path):
    bootstrap = _import_bootstrap(monkeypatch, tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()

    resolved = bootstrap.ensure_pi_workspace_dir(repo)

    assert resolved == str((repo / "workspace").resolve())
    assert (repo / "workspace").is_dir()
    assert os.environ["PI_WORKSPACE_DIR"] == resolved


def test_ensure_pi_upload_root_defaults_to_repo_workspace_gradio(monkeypatch, tmp_path):
    bootstrap = _import_bootstrap(monkeypatch, tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()

    monkeypatch.delenv("PI_UPLOAD_ROOT", raising=False)
    monkeypatch.delenv("GRADIO_TEMP_DIR", raising=False)

    resolved = bootstrap.ensure_pi_upload_root(repo)

    expected = repo / "workspace" / ".gradio_uploads"
    assert resolved == str(expected.resolve())
    assert expected.is_dir()
    assert os.environ["PI_UPLOAD_ROOT"] == resolved
    assert os.environ["GRADIO_TEMP_DIR"] == resolved


def test_pi_default_provider_fallback_local_is_llama_not_gemini(monkeypatch):
    """Unset PI_DEFAULT_PROVIDER must default to llama-cpp outside HF Space."""
    from tools.config import resolve_pi_default_provider_fallback

    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "local-docker")
    assert resolve_pi_default_provider_fallback() == "llama-cpp"


def test_pi_default_provider_fallback_aws_ecs_is_bedrock(monkeypatch):
    from tools.config import resolve_pi_default_provider_fallback

    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "aws-ecs")
    assert resolve_pi_default_provider_fallback() == "amazon-bedrock"


def test_pi_default_model_fallback_aws_ecs_is_claude_sonnet(monkeypatch):
    from tools.config import resolve_pi_default_model_fallback

    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "aws-ecs")
    assert resolve_pi_default_model_fallback() == "anthropic.claude-sonnet-4-6"


def test_pi_default_provider_fallback_hf_space_is_gemini(monkeypatch):
    from tools.config import resolve_pi_default_provider_fallback

    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "hf-space")
    assert resolve_pi_default_provider_fallback() == "google-gemini"


def test_pi_default_model_fallback_local_is_empty(monkeypatch):
    from tools.config import resolve_pi_default_model_fallback

    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "local-docker")
    assert resolve_pi_default_model_fallback() == ""


def test_ensure_pi_config_env_loads_pi_agent_env_before_imports(monkeypatch, tmp_path):
    bootstrap = _import_bootstrap(monkeypatch, tmp_path)
    repo = tmp_path / "repo"
    config_dir = repo / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "pi_agent.env").write_text(
        "PI_DEFAULT_PROVIDER=google-gemini\nPI_DEFAULT_MODEL=gemini-flash-latest\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("PI_DEFAULT_PROVIDER", raising=False)
    monkeypatch.delenv("PI_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("APP_CONFIG_PATH", raising=False)

    bootstrap.ensure_pi_config_env(repo)

    assert os.environ.get("PI_DEFAULT_PROVIDER") == "google-gemini"
    assert os.environ.get("PI_DEFAULT_MODEL") == "gemini-flash-latest"


def test_ensure_pi_workdir_defaults_to_repo_when_unset(monkeypatch, tmp_path):
    bootstrap = _import_bootstrap(monkeypatch, tmp_path)
    repo = tmp_path / "repo"
    skills = repo / "skills"
    skills.mkdir(parents=True)
    (skills / "Example prompt partnership.txt").write_text("template", encoding="utf-8")
    monkeypatch.delenv("PI_WORKDIR", raising=False)

    resolved = bootstrap.ensure_pi_workdir(repo)

    assert resolved == str(repo.resolve())
    assert os.environ["PI_WORKDIR"] == resolved


def test_ensure_pi_workspace_dir_ignores_docker_path_outside_container(
    monkeypatch, tmp_path
):
    bootstrap = _import_bootstrap(monkeypatch, tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    fake_docker_ws = tmp_path / "home" / "user" / "app" / "workspace"
    fake_docker_ws.mkdir(parents=True)
    monkeypatch.setattr(bootstrap, "_DOCKER_WORKSPACE", fake_docker_ws)
    monkeypatch.setattr(bootstrap, "_pi_running_in_container", lambda: False)

    resolved = bootstrap.ensure_pi_workspace_dir(repo)

    assert resolved == str((repo / "workspace").resolve())


def test_ensure_pi_workspace_dir_uses_docker_mount_in_container(monkeypatch, tmp_path):
    bootstrap = _import_bootstrap(monkeypatch, tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    docker_ws = tmp_path / "container_workspace"
    docker_ws.mkdir()
    monkeypatch.setattr(bootstrap, "_DOCKER_WORKSPACE", docker_ws)
    monkeypatch.setattr(bootstrap, "_pi_running_in_container", lambda: True)

    resolved = bootstrap.ensure_pi_workspace_dir(repo)

    assert resolved == str(docker_ws.resolve())


def test_ensure_pi_writable_log_dirs_overrides_logs_on_aws_ecs(monkeypatch, tmp_path):
    bootstrap = _import_bootstrap(monkeypatch, tmp_path)
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "aws-ecs")
    monkeypatch.setenv("ACCESS_LOGS_FOLDER", "logs/")
    monkeypatch.setattr(bootstrap, "_pi_running_in_container", lambda: True)
    access = tmp_path / "pi-logs"
    monkeypatch.setattr(bootstrap, "_DOCKER_ACCESS_LOGS", access)
    monkeypatch.setattr(bootstrap, "_DOCKER_USAGE_LOGS", tmp_path / "pi-usage")
    monkeypatch.setattr(bootstrap, "_DOCKER_FEEDBACK_LOGS", tmp_path / "pi-feedback")

    bootstrap.ensure_pi_writable_log_dirs()

    assert os.environ["ACCESS_LOGS_FOLDER"] == access.as_posix() + "/"


def test_ensure_pi_writable_log_dirs_uses_tmp_in_container(monkeypatch, tmp_path):
    bootstrap = _import_bootstrap(monkeypatch, tmp_path)
    monkeypatch.delenv("ACCESS_LOGS_FOLDER", raising=False)
    monkeypatch.delenv("USAGE_LOGS_FOLDER", raising=False)
    monkeypatch.delenv("FEEDBACK_LOGS_FOLDER", raising=False)
    monkeypatch.setattr(bootstrap, "_pi_running_in_container", lambda: True)
    access = tmp_path / "pi-logs"
    usage = tmp_path / "pi-usage"
    feedback = tmp_path / "pi-feedback"
    monkeypatch.setattr(bootstrap, "_DOCKER_ACCESS_LOGS", access)
    monkeypatch.setattr(bootstrap, "_DOCKER_USAGE_LOGS", usage)
    monkeypatch.setattr(bootstrap, "_DOCKER_FEEDBACK_LOGS", feedback)

    bootstrap.ensure_pi_writable_log_dirs()

    assert access.is_dir()
    assert usage.is_dir()
    assert feedback.is_dir()
    assert os.environ["ACCESS_LOGS_FOLDER"] == access.as_posix() + "/"
    assert os.environ["USAGE_LOGS_FOLDER"] == usage.as_posix() + "/"
    assert os.environ["FEEDBACK_LOGS_FOLDER"] == feedback.as_posix() + "/"


def test_ensure_pi_workspace_dir_honours_explicit_env(monkeypatch, tmp_path):
    bootstrap = _import_bootstrap(monkeypatch, tmp_path)
    custom = tmp_path / "custom_ws"
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(custom))

    resolved = bootstrap.ensure_pi_workspace_dir(tmp_path / "repo")

    assert resolved == str(custom.resolve())
    assert custom.is_dir()
