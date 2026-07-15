"""Pi agent process bootstrap (env file + workspace) before ``tools.config`` import."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_DOCKER_WORKSPACE = Path("/home/user/app/workspace")
_DOCKER_UPLOAD_ROOT = Path("/tmp/gradio")
_DOCKER_PI_WORKDIR = Path("/workspace/doc_redaction")
# CSV log dirs must not live under read-only AGENT_WORKDIR (ECS/HF runtime images).
_DOCKER_ACCESS_LOGS = Path("/tmp/agent-logs")
_DOCKER_USAGE_LOGS = Path("/tmp/agent-usage")
_DOCKER_FEEDBACK_LOGS = Path("/tmp/agent-feedback")
_PARTNERSHIP_TEMPLATE = Path("skills") / "Example prompt partnership.txt"


def _pi_running_in_container() -> bool:
    """
    True when the Pi process is inside Docker / HF Space, not local Windows dev.

    Avoids treating ``C:\\home\\user\\app\\workspace`` (created by mistake on Windows)
    as the compose mount.
    """
    if Path("/.dockerenv").is_file():
        return True
    return _DOCKER_PI_WORKDIR.is_dir() and _partnership_template_exists(
        _DOCKER_PI_WORKDIR
    )


def ensure_pi_workspace_dir(repo_root: Path | None = None) -> str:
    """
    Resolve ``AGENT_WORKSPACE_DIR``, create it, and sync ``os.environ``.

    - Explicit ``AGENT_WORKSPACE_DIR`` wins.
    - Else use the Docker mount only when running in a container.
    - Else ``{repo_root}/workspace`` (local Windows/macOS/Linux dev).
    """
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    raw = (os.environ.get("AGENT_WORKSPACE_DIR") or "").strip()
    if raw:
        path = Path(raw)
    elif _pi_running_in_container() and _DOCKER_WORKSPACE.is_dir():
        path = _DOCKER_WORKSPACE
    else:
        path = root / "workspace"
    path.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    os.environ["AGENT_WORKSPACE_DIR"] = resolved
    return resolved


def _pi_runtime_needs_tmp_log_dirs() -> bool:
    """True when CSV logs must not live under read-only ``AGENT_WORKDIR`` (ECS/HF images)."""
    profile = os.environ.get("AGENT_DEPLOYMENT_PROFILE", "").strip().lower()
    if profile in ("aws-ecs", "hf-space"):
        return True
    return _pi_running_in_container()


def ensure_pi_writable_log_dirs() -> None:
    """
    Point access/usage/feedback CSV logs at ``/tmp`` when running in Docker/ECS.

    ``tools.config`` resolves relative ``logs/`` under ``AGENT_WORKDIR``, which is
    read-only in the Pi runtime image; ``/tmp`` is allowed by
    ``ensure_folder_within_app_directory`` for absolute paths.

    For ``aws-ecs`` / ``hf-space``, always override (S3/task env files often set
    ``logs/`` from the main app template).
    """
    if not _pi_running_in_container():
        return
    for path in (_DOCKER_ACCESS_LOGS, _DOCKER_USAGE_LOGS, _DOCKER_FEEDBACK_LOGS):
        path.mkdir(parents=True, exist_ok=True)
    access = _DOCKER_ACCESS_LOGS.as_posix() + "/"
    usage = _DOCKER_USAGE_LOGS.as_posix() + "/"
    feedback = _DOCKER_FEEDBACK_LOGS.as_posix() + "/"
    if _pi_runtime_needs_tmp_log_dirs():
        os.environ["ACCESS_LOGS_FOLDER"] = access
        os.environ["USAGE_LOGS_FOLDER"] = usage
        os.environ["FEEDBACK_LOGS_FOLDER"] = feedback
    else:
        os.environ.setdefault("ACCESS_LOGS_FOLDER", access)
        os.environ.setdefault("USAGE_LOGS_FOLDER", usage)
        os.environ.setdefault("FEEDBACK_LOGS_FOLDER", feedback)


def ensure_pi_upload_root(repo_root: Path | None = None) -> str:
    """
    Resolve where Gradio stores ``gr.File`` uploads and sync ``os.environ``.

    Must run before ``import gradio`` so ``GRADIO_TEMP_DIR`` matches validation
    in ``redaction_prompt._resolve_and_validate_upload_path``.

    - Explicit ``AGENT_UPLOAD_ROOT`` wins.
    - Else ``GRADIO_TEMP_DIR`` if already set.
    - Else Docker ``/tmp/gradio`` when that directory exists.
    - Else ``{repo}/workspace/.gradio_uploads`` (local dev; stays inside the app tree
      so ``tools.config.ensure_folder_within_app_directory`` accepts ``GRADIO_TEMP_DIR``).
    """
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    raw = (os.environ.get("AGENT_UPLOAD_ROOT") or "").strip()
    if raw:
        path = Path(raw)
    else:
        gradio_temp = (os.environ.get("GRADIO_TEMP_DIR") or "").strip()
        if gradio_temp:
            path = Path(gradio_temp)
        elif _pi_running_in_container() and _DOCKER_UPLOAD_ROOT.is_dir():
            path = _DOCKER_UPLOAD_ROOT
        else:
            path = root / "workspace" / ".gradio_uploads"
    path.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    os.environ["AGENT_UPLOAD_ROOT"] = resolved
    if not (os.environ.get("GRADIO_TEMP_DIR") or "").strip():
        os.environ["GRADIO_TEMP_DIR"] = resolved
    return resolved


def _partnership_template_exists(repo: Path) -> bool:
    return (repo / _PARTNERSHIP_TEMPLATE).is_file()


def ensure_pi_workdir(repo_root: Path | None = None) -> str:
    """
    Resolve ``AGENT_WORKDIR`` (monorepo root for skills/ and Pi RPC cwd).

    - Explicit ``AGENT_WORKDIR`` wins when the partnership prompt template exists there.
    - Else use the checkout root (``agent-redact/shared`` → parents[2]).
    - Docker images set ``AGENT_WORKDIR=/workspace/doc_redaction`` via env or ``start.sh``.
    """
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    raw = (os.environ.get("AGENT_WORKDIR") or "").strip()
    if raw:
        candidate = Path(raw)
        if _partnership_template_exists(candidate):
            resolved = str(candidate.resolve())
            os.environ["AGENT_WORKDIR"] = resolved
            return resolved
    if _pi_running_in_container() and _partnership_template_exists(_DOCKER_PI_WORKDIR):
        resolved = str(_DOCKER_PI_WORKDIR.resolve())
        os.environ["AGENT_WORKDIR"] = resolved
        return resolved
    resolved = str(root)
    os.environ["AGENT_WORKDIR"] = resolved
    return resolved


def pi_repo_root_path(repo_root: Path | None = None) -> Path:
    """Return ``AGENT_WORKDIR`` as a :class:`~pathlib.Path` (calls :func:`ensure_pi_workdir`)."""
    return Path(ensure_pi_workdir(repo_root))


def resolve_agent_env_file(config_dir: Path) -> Path:
    """
    Return the agent config file path, preferring ``agent.env`` over legacy ``pi_agent.env``.

    The config file was renamed from ``pi_agent.env`` to ``agent.env``. Prefer the
    new name; fall back to the legacy file only when the new one is absent but the
    old one exists. When neither exists, return the new-name path.
    """
    new_path = config_dir / "agent.env"
    legacy_path = config_dir / "pi_agent.env"
    if not new_path.is_file() and legacy_path.is_file():
        return legacy_path
    return new_path


def load_pi_agent_env_file(config_path: str | Path | None = None) -> bool:
    """
    Load ``config/agent.env`` into ``os.environ`` (does not override existing vars).

    Must run before ``import pi_agent_config`` so module-level defaults see the file.
    """
    path = Path(config_path or os.environ.get("APP_CONFIG_PATH", "")).expanduser()
    if not path.is_file():
        return False
    load_dotenv(path, override=False)
    return True


# Env vars owned by the external ``pi`` coding-agent CLI (not renamed).
_EXTERNAL_PI_ENV_VARS = frozenset({"PI_OFFLINE", "PI_SKIP_VERSION_CHECK"})


def migrate_legacy_pi_env_vars() -> None:
    """
    Backward-compat: mirror legacy ``PI_*`` env vars onto the new ``AGENT_*`` names.

    The app renamed its ``PI_*`` environment variables to ``AGENT_*``. Existing
    deployments / config files may still set the old names, so copy any legacy
    value onto the new key when the new key is unset. A legacy ``PI_AGENT_*`` key
    collapses to ``AGENT_*`` (e.g. legacy ``PI_AGENT_ENV_S3_KEY`` -> ``AGENT_ENV_S3_KEY``).
    Vars owned by the external ``pi`` CLI are left untouched. Safe to call repeatedly.
    """
    for key in list(os.environ.keys()):
        if not key.startswith("PI_") or key in _EXTERNAL_PI_ENV_VARS:
            continue
        rest = key[3:]
        new_key = rest if rest.startswith("AGENT") else "AGENT_" + rest
        if new_key not in os.environ:
            os.environ[new_key] = os.environ[key]


def ensure_pi_config_env(repo_root: Path | None = None) -> str:
    """
    Set process env so ``tools.config`` loads the Pi agent env file.

    Must run before any ``from pi_agent_config import ...`` or ``tools.config`` import
    that depends on Pi env vars. Safe to call multiple times; does not override
    existing environment variables.
    """
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    migrate_legacy_pi_env_vars()
    os.environ.setdefault("APP_TYPE", "agent")
    if not os.environ.get("APP_CONFIG_PATH", "").strip():
        os.environ["APP_CONFIG_PATH"] = str(resolve_agent_env_file(root / "config"))
    load_pi_agent_env_file()
    migrate_legacy_pi_env_vars()
    ensure_pi_workdir(root)
    ensure_pi_workspace_dir(root)
    ensure_pi_upload_root(root)
    ensure_pi_writable_log_dirs()
    from pi_workspace_skills import ensure_workspace_skills

    ensure_workspace_skills()
    return os.environ["APP_CONFIG_PATH"]
