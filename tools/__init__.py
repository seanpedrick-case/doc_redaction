"""
Legacy `tools` package.

This repository historically treated `tools/` as a top-level import namespace
(e.g. `from tools.config import ...`). Making it a real package keeps those
imports working in PyPI installs and in repo-root executions.
"""

from __future__ import annotations

__all__: list[str] = []

