"""Sync doc_redaction skills into the Pi workspace and constrain Pi RPC to that tree."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from bootstrap_pi_config import pi_repo_root_path


def workspace_base_dir() -> Path:
    from session_workspace import workspace_base_dir as _base

    return _base()


def workspace_pi_dir() -> Path:
    return workspace_base_dir() / ".pi"


def workspace_skills_dir() -> Path:
    return workspace_pi_dir() / "skills"


def repo_skills_dir() -> Path:
    return pi_repo_root_path() / "skills"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _should_resync(dest: Path, src: Path) -> bool:
    if _env_flag("PI_SKILLS_RESYNC"):
        return True
    if not dest.is_dir():
        return True
    if not any(dest.iterdir()):
        return True
    try:
        return src.stat().st_mtime > dest.stat().st_mtime
    except OSError:
        return True


def _copy_tree_item(src: Path, dest: Path) -> None:
    if src.is_dir():
        if dest.exists():
            for child in src.iterdir():
                _copy_tree_item(child, dest / child.name)
        else:
            shutil.copytree(src, dest, copy_function=shutil.copy2)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _make_readonly(path: Path) -> None:
    if _env_flag("PI_SKILLS_WRITABLE"):
        return
    try:
        if path.is_dir():
            for root, dirs, files in os.walk(path):
                root_path = Path(root)
                for name in files:
                    file_path = root_path / name
                    mode = file_path.stat().st_mode
                    file_path.chmod(
                        mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH
                    )
                for name in dirs:
                    dir_path = root_path / name
                    mode = dir_path.stat().st_mode
                    dir_path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
            mode = path.stat().st_mode
            path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        else:
            mode = path.stat().st_mode
            path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    except OSError:
        pass


def write_workspace_pi_settings() -> Path:
    """
    Project Pi settings under ``{workspace}/.pi/settings.json``.

    Paths in that file resolve relative to ``{workspace}/.pi/`` per Pi docs.
    """
    pi_dir = workspace_pi_dir()
    pi_dir.mkdir(parents=True, exist_ok=True)
    settings_path = pi_dir / "settings.json"
    payload = {
        "skills": ["skills"],
        "extensions": [],
        "packages": [],
        "enableSkillCommands": True,
    }
    import json

    settings_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return settings_path


def sync_repo_skills_to_workspace(*, force: bool = False) -> Path:
    """
    Copy ``{repo}/skills/`` → ``{workspace}/.pi/skills/`` (read-only for the agent).

    Re-sync when the repo tree is newer or ``PI_SKILLS_RESYNC=true``.
    """
    src = repo_skills_dir()
    dest = workspace_skills_dir()
    workspace_pi_dir().mkdir(parents=True, exist_ok=True)

    if not src.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        write_workspace_pi_settings()
        return dest

    if force or _should_resync(dest, src):
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        for item in sorted(src.iterdir()):
            _copy_tree_item(item, dest / item.name)

    _make_readonly(dest)
    write_workspace_pi_settings()
    os.environ["PI_WORKSPACE_SKILLS_DIR"] = str(dest.resolve())
    return dest.resolve()


def ensure_workspace_skills(*, force: bool = False) -> Path:
    """Idempotent sync used at app startup and before Pi RPC starts."""
    return sync_repo_skills_to_workspace(force=force)


def partnership_template_in_workspace() -> Path | None:
    path = workspace_skills_dir() / "Example prompt partnership.txt"
    return path if path.is_file() else None


def pi_rpc_cwd(session_hash: str | None = None) -> str:
    """Subprocess cwd for ``pi --mode rpc`` (session subfolder when enabled)."""
    from session_workspace import session_workspace_dir, session_workspace_enabled

    base = workspace_base_dir()
    if session_hash and session_hash.strip() and session_workspace_enabled():
        return str(session_workspace_dir(session_hash))
    return str(base)


def pi_rpc_args() -> list[str]:
    """Load only workspace skills; do not discover repo ``skills/`` via ancestors."""
    skills_dir = ensure_workspace_skills()
    return ["--no-skills", "--skill", str(skills_dir)]


def workspace_boundary_prefix(session_hash: str | None = None) -> str:
    """Extra prompt text: workspace root, skills path, and path rules."""
    base = workspace_base_dir().as_posix().rstrip("/")
    skills = workspace_skills_dir().as_posix()
    from session_workspace import session_workspace_dir, session_workspace_enabled

    if session_hash and session_hash.strip() and session_workspace_enabled():
        root = session_workspace_dir(session_hash).as_posix().rstrip("/")
        scope = f"your session folder `{root}/`"
    else:
        root = base
        scope = f"the workspace `{base}/`"

    return (
        f"**Workspace boundary (mandatory):** work only under `{base}/`. "
        f"Your active directory is {scope}. "
        f"Do not read, write, or run shell commands targeting paths outside `{base}/` "
        f"(including the git checkout and `agent-redact/`). "
        f"**Skills (read-only):** doc_redaction skills are synced to `{skills}/`. "
        f"Use `/skill:doc-redaction-app`, `/skill:doc-redact-page-review`, etc. "
        f"Do not edit files under `{skills}/`.\n\n"
    )
