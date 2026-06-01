"""Tests for Pi redaction upload filename normalization."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

if "gradio" not in sys.modules:
    _gr = ModuleType("gradio")
    _gr.FileExplorer = lambda **kwargs: kwargs  # type: ignore[misc]
    sys.modules["gradio"] = _gr

import redaction_prompt as rp


def _reload_redaction_prompt(monkeypatch):
    monkeypatch.setenv("PI_DEPLOYMENT_PROFILE", "local-docker")
    return importlib.reload(rp)


@pytest.mark.parametrize(
    ("original", "expected", "renamed"),
    [
        (
            "graduate-job-example-cover-letter.pdf",
            "graduate-job-example-cover-letter.pdf",
            False,
        ),
        ("Cover Letter.pdf", "Cover Letter.pdf", False),
        ("Report (draft).pdf", "Report (draft).pdf", False),
        ("FOI response, v2.pdf", "FOI response, v2.pdf", False),
        ("file:name.pdf", "file_name.pdf", True),
        (".pdf", "file.pdf", True),
        ("..", None, None),
    ],
)
def test_workspace_filename_from_upload(monkeypatch, original, expected, renamed):
    module = _reload_redaction_prompt(monkeypatch)
    if expected is None:
        with pytest.raises(ValueError, match="invalid name"):
            module._workspace_filename_from_upload(original)
        return
    got_original, got_name, got_renamed = module._workspace_filename_from_upload(
        original
    )
    assert got_original == Path(original).name.strip()
    assert got_name == expected
    assert got_renamed is renamed


def test_copy_upload_to_workspace_preserves_permissive_names(monkeypatch, tmp_path):
    module = _reload_redaction_prompt(monkeypatch)
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = upload_root / "Cover Letter.pdf"
    source.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(module, "upload_root", lambda: upload_root.resolve())

    dest, renamed_from = module.copy_upload_to_workspace(
        source,
        workspace_dir=workspace,
    )
    assert dest.name == "Cover Letter.pdf"
    assert renamed_from is None
    assert dest.read_bytes() == source.read_bytes()


def test_copy_upload_to_workspace_reports_minimal_rename(monkeypatch, tmp_path):
    module = _reload_redaction_prompt(monkeypatch)
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = upload_root / "bad:name.pdf"
    source.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(module, "upload_root", lambda: upload_root.resolve())

    dest, renamed_from = module.copy_upload_to_workspace(
        source,
        workspace_dir=workspace,
    )
    assert dest.name == "bad_name.pdf"
    assert renamed_from == "bad:name.pdf"
