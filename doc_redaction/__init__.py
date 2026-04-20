"""
doc_redaction package.

This package layer is intentionally thin for now: it preserves existing
repo-root entrypoints (e.g. `app.py`, `cli_redact.py`) while providing stable
import paths for PyPI installs.
"""

from __future__ import annotations

__all__ = ["__version__", "choose_and_run_redactor", "run_redaction"]

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("doc_redaction")
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "0.0.0"
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

# Convenience re-exports (package-qualified import surface)
from doc_redaction.file_redaction import (
    choose_and_run_redactor,
    run_redaction,
)  # noqa: E402
