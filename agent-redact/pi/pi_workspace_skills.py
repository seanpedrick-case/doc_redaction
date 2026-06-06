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


_SKILLS_SKIP_DIR_NAMES = frozenset({"archive_attempts"})
_SKILLS_SKIP_SUFFIXES = (".b64.txt",)
_SKILLS_MAX_FILE_BYTES = int(
    os.environ.get("PI_SKILLS_MAX_FILE_BYTES", str(512 * 1024))
)


def _should_skip_skill_relpath(rel: Path, *, size_bytes: int | None = None) -> bool:
    """Skip archive blobs and other non-skill artifacts during workspace sync."""
    if any(part in _SKILLS_SKIP_DIR_NAMES for part in rel.parts):
        return True
    name_lower = rel.name.lower()
    if name_lower.endswith(_SKILLS_SKIP_SUFFIXES):
        return True
    if size_bytes is not None and size_bytes > _SKILLS_MAX_FILE_BYTES:
        return True
    return False


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
    _copy_tree_item_filtered(src, dest, src_root=src)


def _copy_tree_item_filtered(src: Path, dest: Path, *, src_root: Path) -> None:
    rel = src.relative_to(src_root)
    if _should_skip_skill_relpath(rel):
        return
    if src.is_file():
        try:
            size = src.stat().st_size
        except OSError:
            size = None
        if size is not None and size > _SKILLS_MAX_FILE_BYTES:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return
    if dest.exists():
        for child in sorted(src.iterdir()):
            _copy_tree_item_filtered(child, dest / child.name, src_root=src_root)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        for child in sorted(src.iterdir()):
            _copy_tree_item_filtered(child, dest / child.name, src_root=src_root)


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
            rel = item.relative_to(src)
            try:
                size = item.stat().st_size if item.is_file() else None
            except OSError:
                size = None
            if _should_skip_skill_relpath(rel, size_bytes=size):
                continue
            _copy_tree_item_filtered(item, dest / item.name, src_root=src)

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
