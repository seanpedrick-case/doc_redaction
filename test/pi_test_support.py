"""Helpers for importing agent-redact/pi modules in unit tests."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


def ensure_gradio_importable() -> None:
    """Prefer the installed gradio package; only inject a minimal stub when absent.

    Do not register a fake ``gradio`` in ``sys.modules`` when the real package is
    installed — pytest imports all test modules in one process, and an early stub
    breaks collection for tests that need ``from gradio import Progress``.
    """
    existing = sys.modules.get("gradio")
    if existing is not None and getattr(existing, "__file__", None):
        return
    try:
        importlib.import_module("gradio")
        return
    except (ImportError, ModuleNotFoundError, SystemError, OSError):
        pass
    if existing is not None:
        return
    stub = ModuleType("gradio")
    stub.FileExplorer = lambda **kwargs: kwargs  # type: ignore[misc]
    stub.Request = type("Request", (), {})  # type: ignore[misc, assignment]
    sys.modules["gradio"] = stub
