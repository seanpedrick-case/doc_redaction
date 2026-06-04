"""Pi agent process bootstrap (env file + workspace) before ``tools.config`` import."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_DOCKER_WORKSPACE = Path("/home/user/app/workspace")
_DOCKER_UPLOAD_ROOT = Path("/tmp/gradio")
_DOCKER_PI_WORKDIR = Path("/workspace/doc_redaction")
# CSV log dirs must not live under read-only PI_WORKDIR (ECS/HF runtime images).
_DOCKER_ACCESS_LOGS = Path("/tmp/pi-logs")
_DOCKER_USAGE_LOGS = Path("/tmp/pi-usage")
_DOCKER_FEEDBACK_LOGS = Path("/tmp/pi-feedback")
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
    Resolve ``PI_WORKSPACE_DIR``, create it, and sync ``os.environ``.

    - Explicit ``PI_WORKSPACE_DIR`` wins.
    - Else use the Docker mount only when running in a container.
    - Else ``{repo_root}/workspace`` (local Windows/macOS/Linux dev).
    """
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    raw = (os.environ.get("PI_WORKSPACE_DIR") or "").strip()
    if raw:
        path = Path(raw)
    elif _pi_running_in_container() and _DOCKER_WORKSPACE.is_dir():
        path = _DOCKER_WORKSPACE
    else:
        path = root / "workspace"
    path.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    os.environ["PI_WORKSPACE_DIR"] = resolved
    return resolved


def ensure_pi_writable_log_dirs() -> None:
    """
    Point access/usage/feedback CSV logs at ``/tmp`` when running in Docker/ECS.

    ``tools.config`` resolves relative ``logs/`` under ``PI_WORKDIR``, which is
    read-only in the Pi runtime image; ``/tmp`` is allowed by
    ``ensure_folder_within_app_directory`` for absolute paths.
    """
    if not _pi_running_in_container():
        return
    for path in (_DOCKER_ACCESS_LOGS, _DOCKER_USAGE_LOGS, _DOCKER_FEEDBACK_LOGS):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ACCESS_LOGS_FOLDER", _DOCKER_ACCESS_LOGS.as_posix() + "/")
    os.environ.setdefault("USAGE_LOGS_FOLDER", _DOCKER_USAGE_LOGS.as_posix() + "/")
    os.environ.setdefault(
        "FEEDBACK_LOGS_FOLDER", _DOCKER_FEEDBACK_LOGS.as_posix() + "/"
    )


def ensure_pi_upload_root(repo_root: Path | None = None) -> str:
    """
    Resolve where Gradio stores ``gr.File`` uploads and sync ``os.environ``.

    Must run before ``import gradio`` so ``GRADIO_TEMP_DIR`` matches validation
    in ``redaction_prompt._resolve_and_validate_upload_path``.

    - Explicit ``PI_UPLOAD_ROOT`` wins.
    - Else ``GRADIO_TEMP_DIR`` if already set.
    - Else Docker ``/tmp/gradio`` when that directory exists.
    - Else ``{repo}/workspace/.gradio_uploads`` (local dev; stays inside the app tree
      so ``tools.config.ensure_folder_within_app_directory`` accepts ``GRADIO_TEMP_DIR``).
    """
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    raw = (os.environ.get("PI_UPLOAD_ROOT") or "").strip()
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
    os.environ["PI_UPLOAD_ROOT"] = resolved
    if not (os.environ.get("GRADIO_TEMP_DIR") or "").strip():
        os.environ["GRADIO_TEMP_DIR"] = resolved
    return resolved


def _partnership_template_exists(repo: Path) -> bool:
    return (repo / _PARTNERSHIP_TEMPLATE).is_file()


def ensure_pi_workdir(repo_root: Path | None = None) -> str:
    """
    Resolve ``PI_WORKDIR`` (monorepo root for skills/ and Pi RPC cwd).

    - Explicit ``PI_WORKDIR`` wins when the partnership prompt template exists there.
    - Else use the checkout root (``agent-redact/pi`` → parents[2]).
    - Docker images set ``PI_WORKDIR=/workspace/doc_redaction`` via env or ``start.sh``.
    """
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    raw = (os.environ.get("PI_WORKDIR") or "").strip()
    if raw:
        candidate = Path(raw)
        if _partnership_template_exists(candidate):
            resolved = str(candidate.resolve())
            os.environ["PI_WORKDIR"] = resolved
            return resolved
    if _pi_running_in_container() and _partnership_template_exists(_DOCKER_PI_WORKDIR):
        resolved = str(_DOCKER_PI_WORKDIR.resolve())
        os.environ["PI_WORKDIR"] = resolved
        return resolved
    resolved = str(root)
    os.environ["PI_WORKDIR"] = resolved
    return resolved


def pi_repo_root_path(repo_root: Path | None = None) -> Path:
    """Return ``PI_WORKDIR`` as a :class:`~pathlib.Path` (calls :func:`ensure_pi_workdir`)."""
    return Path(ensure_pi_workdir(repo_root))


def load_pi_agent_env_file(config_path: str | Path | None = None) -> bool:
    """
    Load ``config/pi_agent.env`` into ``os.environ`` (does not override existing vars).

    Must run before ``import pi_agent_config`` so module-level defaults see the file.
    """
    path = Path(config_path or os.environ.get("APP_CONFIG_PATH", "")).expanduser()
    if not path.is_file():
        return False
    load_dotenv(path, override=False)
    return True


def ensure_pi_config_env(repo_root: Path | None = None) -> str:
    """
    Set process env so ``tools.config`` loads the Pi agent env file.

    Must run before any ``from pi_agent_config import ...`` or ``tools.config`` import
    that depends on Pi env vars. Safe to call multiple times; does not override
    existing environment variables.
    """
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    os.environ.setdefault("APP_TYPE", "pi")
    if not os.environ.get("APP_CONFIG_PATH", "").strip():
        os.environ["APP_CONFIG_PATH"] = str(root / "config" / "pi_agent.env")
    load_pi_agent_env_file()
    ensure_pi_workdir(root)
    ensure_pi_workspace_dir(root)
    ensure_pi_upload_root(root)
    ensure_pi_writable_log_dirs()
    from pi_workspace_skills import ensure_workspace_skills

    ensure_workspace_skills()
    return os.environ["APP_CONFIG_PATH"]
