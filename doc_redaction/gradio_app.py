"""
Gradio/FastAPI app export for packaging.

We intentionally keep the authoritative implementation in the existing
repo-root `app.py` for backwards compatibility (Docker, HF Spaces, tests),
and re-export the module-level `app` and `blocks` objects here so a PyPI
install has a stable import path: `doc_redaction.gradio_app:app`.
"""

from __future__ import annotations

import importlib
from typing import Any

_root_app = importlib.import_module("app")

# `app` is the ASGI application (FastAPI or Gradio Blocks depending on config)
app: Any = getattr(_root_app, "app")

# `blocks` is the Gradio Blocks UI (present when RUN_FASTAPI is enabled)
blocks: Any = getattr(_root_app, "blocks", None)

__all__ = ["app", "blocks"]

