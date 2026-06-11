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


def workspace_helpers_dir() -> Path:
    return workspace_pi_dir() / "helpers"


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
        if dest.exists():
            _make_writable(dest)
        shutil.copy2(src, dest)
        return
    if dest.exists():
        for child in sorted(src.iterdir()):
            _copy_tree_item_filtered(child, dest / child.name, src_root=src_root)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        for child in sorted(src.iterdir()):
            _copy_tree_item_filtered(child, dest / child.name, src_root=src_root)


def _chmod_tree(path: Path, *, writable: bool) -> None:
    """Set or clear write bits on a file tree (needed for Windows resync)."""
    try:
        if path.is_dir():
            for root, dirs, files in os.walk(path):
                root_path = Path(root)
                for name in files:
                    file_path = root_path / name
                    mode = file_path.stat().st_mode
                    file_path.chmod(
                        (mode | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                        if writable
                        else (mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
                    )
                for name in dirs:
                    dir_path = root_path / name
                    mode = dir_path.stat().st_mode
                    dir_path.chmod(
                        (mode | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                        if writable
                        else (mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
                    )
            mode = path.stat().st_mode
            path.chmod(
                (mode | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                if writable
                else (mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
            )
        else:
            mode = path.stat().st_mode
            path.chmod(
                (mode | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
                if writable
                else (mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
            )
    except OSError:
        pass


def _make_writable(path: Path) -> None:
    _chmod_tree(path, writable=True)


def _make_readonly(path: Path) -> None:
    if _env_flag("PI_SKILLS_WRITABLE"):
        return
    _chmod_tree(path, writable=False)


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
            _make_writable(dest)
            shutil.rmtree(dest)
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


def sync_workspace_helpers() -> Path:
    """
    Copy Pi redaction helper scripts into ``{workspace}/.pi/helpers/``.

    Keeps ``remote_redaction.py`` inside the workspace boundary on AWS ECS so the
    agent does not search ``/workspace/doc_redaction/agent-redact/``.
    """
    helpers = workspace_helpers_dir()
    helpers.mkdir(parents=True, exist_ok=True)
    pi_dir = Path(__file__).resolve().parent
    for name in ("remote_redaction.py",):
        src = pi_dir / name
        dest = helpers / name
        if not src.is_file():
            continue
        if not dest.is_file() or src.stat().st_mtime > dest.stat().st_mtime:
            shutil.copy2(src, dest)
    return helpers.resolve()


def write_hf_space_deployment_skill(*, force: bool = False) -> Path | None:
    """
    Write a deployment-specific skill that overrides Docker URLs in generic skills.

    Only active when ``PI_DEPLOYMENT_PROFILE=hf-space``.
    """
    try:
        from pi_agent_config import is_hf_space_profile
        from redaction_prompt import doc_redaction_gradio_url
    except ImportError:
        return None
    if not is_hf_space_profile():
        return None

    dest_dir = workspace_skills_dir() / "hf-space-deployment"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "SKILL.md"
    url = doc_redaction_gradio_url()
    helpers = workspace_helpers_dir().as_posix()
    content = (
        "# HF Space deployment (read first)\n\n"
        "This Pi agent runs on **Hugging Face Spaces** with **Gemini** and calls a "
        "**remote** doc_redaction Space. Generic skills mention Docker URLs for "
        "local-docker or AWS ECS — **ignore those here**.\n\n"
        "## Authoritative settings\n\n"
        "| Setting | Value |\n"
        "|---------|--------|\n"
        f"| **doc_redaction URL** | `{url}` **only** |\n"
        "| **Auth** | `HF_TOKEN` (Space secret; already in Pi subprocess env) |\n"
        f"| **Helper module** | `{helpers}/remote_redaction.py` |\n\n"
        "## Rules\n\n"
        f"- Call `/doc_redact` via `make_redaction_client()` from `{helpers}/remote_redaction.py`.\n"
        "- **Do not** use `host.docker.internal`, `localhost`, `redaction:7861`, or probe "
        "alternate URLs.\n"
        "- **Do not** rewrite or duplicate `remote_redaction.py` — use the synced helper.\n"
        "- On rate limits, wait and retry the same URL (Pi UI handles backoff).\n"
        "- Write status updates as **normal assistant text**, not bash `#` comments.\n"
        "- After `/doc_redact`, download outputs with `fetch_redaction_files` into your "
        "session `output_redact/` folder.\n\n"
        "Then read `/skill:doc-redaction-app` and `/skill:doc-redaction-modifications` "
        "for workflow steps, substituting the URL above wherever examples show Docker hosts.\n"
    )
    if force or not dest.is_file() or dest.read_text(encoding="utf-8") != content:
        dest.write_text(content, encoding="utf-8")
    return dest


def ensure_workspace_skills(*, force: bool = False) -> Path:
    """Idempotent sync used at app startup and before Pi RPC starts."""
    dest = sync_repo_skills_to_workspace(force=force)
    sync_workspace_helpers()
    write_hf_space_deployment_skill(force=force)
    return dest


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

    hf_note = ""
    try:
        from pi_agent_config import is_hf_space_profile
        from redaction_prompt import doc_redaction_gradio_url

        if is_hf_space_profile():
            hf_note = (
                f"**HF Space redaction backend:** use `{doc_redaction_gradio_url()}` only "
                "(see `/skill:hf-space-deployment`). Do not use Docker host URLs from "
                "other skills. Write user-facing progress as normal chat text, not bash "
                "comments.\n\n"
            )
    except ImportError:
        pass

    return (
        f"**Workspace boundary (mandatory):** work only under `{base}/`. "
        f"Your active directory is {scope}. "
        f"Do not read, write, or run shell commands targeting paths outside `{base}/` "
        f"(including the git checkout and `agent-redact/`). "
        f"**Skills (read-only):** doc_redaction skills are synced to `{skills}/`. "
        f"Use `/skill:doc-redaction-app`, `/skill:doc-redact-page-review`, etc. "
        f"Do not edit files under `{skills}/`.\n\n"
        f"{hf_note}"
    )
