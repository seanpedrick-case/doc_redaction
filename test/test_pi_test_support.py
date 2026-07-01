"""Tests for pi_test_support gradio import helper."""

from __future__ import annotations

import sys
from types import ModuleType

from pi_test_support import ensure_gradio_importable


def test_ensure_gradio_importable_does_not_replace_real_gradio(monkeypatch):
    real = ModuleType("gradio")
    real.__file__ = "/site-packages/gradio/__init__.py"
    real.Progress = object()
    monkeypatch.setitem(sys.modules, "gradio", real)
    ensure_gradio_importable()
    assert sys.modules["gradio"] is real


def test_ensure_gradio_importable_installs_stub_when_gradio_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "gradio", raising=False)

    def _fail_import(name, package=None):
        raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", _fail_import)
    ensure_gradio_importable()
    stub = sys.modules["gradio"]
    assert getattr(stub, "__file__", None) is None
    assert callable(stub.FileExplorer)
