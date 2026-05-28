"""Tests for Pi workspace FileExplorer → gr.File download wiring."""

import sys
from types import ModuleType

# output_files imports gradio at module level; stub it for unit tests.
if "gradio" not in sys.modules:
    _gr = ModuleType("gradio")
    _gr.FileExplorer = lambda **kwargs: kwargs  # type: ignore[misc]
    sys.modules["gradio"] = _gr

if "pi_examples" not in sys.modules:
    _pi_examples = ModuleType("pi_examples")
    _pi_examples.gradio_example_allowed_paths = lambda: []  # type: ignore[attr-defined]
    sys.modules["pi_examples"] = _pi_examples

from docker.pi import output_files as of


def test_resolve_under_workspace_accepts_absolute_paths(tmp_path, monkeypatch):
    """Gradio FileExplorer preprocess joins root_dir and returns absolute paths."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    doc = workspace / "redact" / "demo"
    doc.mkdir(parents=True)
    pdf = doc / "output_redact.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(of, "WORKSPACE_DIR", workspace)

    resolved = of._resolve_under_workspace(str(pdf))
    assert resolved == pdf.resolve()


def test_resolve_under_workspace_accepts_relative_paths(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pdf = workspace / "report.csv"
    pdf.write_text("a,b\n1,2\n")

    monkeypatch.setattr(of, "WORKSPACE_DIR", workspace)

    resolved = of._resolve_under_workspace("report.csv")
    assert resolved == pdf.resolve()


def test_resolve_under_workspace_rejects_outside_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.pdf"
    outside.write_bytes(b"%PDF")

    monkeypatch.setattr(of, "WORKSPACE_DIR", workspace)

    assert of._resolve_under_workspace(str(outside)) is None


def test_workspace_files_download_fn_returns_absolute_paths(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pdf = workspace / "out.pdf"
    pdf.write_bytes(b"%PDF")

    monkeypatch.setattr(of, "WORKSPACE_DIR", workspace)

    result = of.workspace_files_download_fn([str(pdf)])
    assert result == [str(pdf.resolve())]
