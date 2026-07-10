"""Resolve bundled Gradio example file locations (repo, package, Docker)."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_example_data_dirs() -> list[Path]:
    """Return all existing ``example_data`` roots (may be multiple layouts)."""
    roots: list[Path] = []

    def _add(candidate: Path) -> None:
        resolved = candidate.resolve()
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)

    try:
        import doc_redaction as _doc_redaction_pkg

        _add(Path(_doc_redaction_pkg.__file__).resolve().parent / "example_data")
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd()
    for base in (repo_root, cwd):
        for rel in ("doc_redaction/example_data", "example_data"):
            _add(base / rel)

    return roots


def is_trusted_bundled_example_path(local_path: str) -> bool:
    """
    True when ``local_path`` is a file under a resolved bundled ``example_data`` dir.

    Used to skip GuardDuty malware staging for Gradio ``Examples`` demo files shipped
    with the app (not end-user uploads).
    """
    if not local_path or not str(local_path).strip():
        return False
    try:
        abspath = Path(os.path.abspath(local_path)).resolve()
    except OSError:
        return False
    if not abspath.is_file():
        return False
    for root in resolve_example_data_dirs():
        try:
            abspath.relative_to(root)
            return True
        except ValueError:
            continue
    return False
