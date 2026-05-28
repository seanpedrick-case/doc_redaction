"""Tests for Pi workspace FileExplorer → gr.File download wiring."""

import sys
from pathlib import Path
from types import ModuleType

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

# output_files imports gradio at module level; stub it for unit tests.
if "gradio" not in sys.modules:
    _gr = ModuleType("gradio")
    _gr.FileExplorer = lambda **kwargs: kwargs  # type: ignore[misc]
    sys.modules["gradio"] = _gr

if "pi_examples" not in sys.modules:
    _pi_examples = ModuleType("pi_examples")
    _pi_examples.gradio_example_allowed_paths = lambda: []  # type: ignore[attr-defined]
    sys.modules["pi_examples"] = _pi_examples

import output_files as of


def test_resolve_under_workspace_accepts_absolute_paths(tmp_path, monkeypatch):
    """Gradio FileExplorer preprocess joins root_dir and returns absolute paths."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    doc = workspace / "redact" / "demo"
    doc.mkdir(parents=True)
    pdf = doc / "output_redact.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(of, "WORKSPACE_DIR", workspace)

    resolved = of._resolve_under_workspace(str(pdf), workspace_root=workspace)
    assert resolved == pdf.resolve()


def test_resolve_under_workspace_accepts_relative_paths(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pdf = workspace / "report.csv"
    pdf.write_text("a,b\n1,2\n")

    monkeypatch.setattr(of, "WORKSPACE_DIR", workspace)

    resolved = of._resolve_under_workspace("report.csv", workspace_root=workspace)
    assert resolved == pdf.resolve()


def test_resolve_under_workspace_rejects_outside_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.pdf"
    outside.write_bytes(b"%PDF")

    monkeypatch.setattr(of, "WORKSPACE_DIR", workspace)

    assert of._resolve_under_workspace(str(outside), workspace_root=workspace) is None


def test_resolve_under_workspace_rejects_path_traversal(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.pdf"
    outside.write_bytes(b"%PDF")

    monkeypatch.setattr(of, "WORKSPACE_DIR", workspace)

    assert (
        of._resolve_under_workspace("../secret.pdf", workspace_root=workspace) is None
    )
    assert (
        of._resolve_under_workspace(f"../{outside.name}", workspace_root=workspace)
        is None
    )


def test_resolve_under_workspace_rejects_unsafe_relative_segments(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    nested = workspace / "redact" / "demo"
    nested.mkdir(parents=True)
    pdf = nested / "output.pdf"
    pdf.write_bytes(b"%PDF")

    assert of._is_safe_workspace_relative_path("redact/demo/output.pdf") is True
    assert of._is_safe_workspace_relative_path("../secret.pdf") is False
    assert of._is_safe_workspace_relative_path("/etc/passwd") is False

    resolved = of._resolve_under_workspace(
        "redact/demo/output.pdf",
        workspace_root=workspace,
    )
    assert resolved == pdf.resolve()


def test_workspace_files_download_fn_returns_absolute_paths(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    base.mkdir()
    session_dir = base / "user1"
    session_dir.mkdir()
    pdf = session_dir / "out.pdf"
    pdf.write_bytes(b"%PDF")

    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")

    result = of.workspace_files_download_fn([str(pdf)], "user1")
    assert result == [str(pdf.resolve())]


def test_collect_final_output_files_finds_review_final_folder(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    session_dir = base / "session"
    final_dir = session_dir / "redact" / "doc.pdf" / "review" / "output_review_final"
    final_dir.mkdir(parents=True)
    redacted = final_dir / "doc_redacted.pdf"
    redacted.write_bytes(b"%PDF")
    other = session_dir / "redact" / "doc.pdf" / "output_redact" / "draft.csv"
    other.parent.mkdir(parents=True)
    other.write_text("x")

    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")

    result = of.collect_final_output_files("session")
    assert result == [
        str((session_dir / "output_final_download" / "doc_redacted.pdf").resolve())
    ]
    assert (
        session_dir / "output_final_download" / "doc_redacted.pdf"
    ).read_bytes() == b"%PDF"


def test_collect_final_output_files_supports_output_final_alias(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    session_dir = base / "session"
    final_dir = session_dir / "redact" / "doc.pdf" / "review" / "output_final"
    final_dir.mkdir(parents=True)
    redacted = final_dir / "doc_redacted.pdf"
    redacted.write_bytes(b"%PDF")

    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")

    result = of.collect_final_output_files("session")
    assert result == [
        str((session_dir / "output_final_download" / "doc_redacted.pdf").resolve())
    ]


def test_strip_gradio_cache_prefix_removes_long_hash(tmp_path):
    assert of.strip_gradio_cache_prefix("abc12345678901234_report.pdf") == "report.pdf"
    assert of.strip_gradio_cache_prefix("report.pdf") == "report.pdf"
    assert (
        of.strip_gradio_cache_prefix("short_hash_report.pdf") == "short_hash_report.pdf"
    )


def test_collect_final_output_files_deduplicates_and_strips_prefix(
    tmp_path, monkeypatch
):
    import time

    base = tmp_path / "workspace"
    session_dir = base / "session"
    final_dir = session_dir / "redact" / "doc.pdf" / "review" / "output_review_final"
    final_dir.mkdir(parents=True)

    older = final_dir / "aaaaaaaaaaaaaaaa_report.pdf"
    older.write_bytes(b"old")
    time.sleep(0.02)
    newer = final_dir / "bbbbbbbbbbbbbbbb_report.pdf"
    newer.write_bytes(b"new")
    plain = final_dir / "notes.txt"
    plain.write_text("notes")

    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")

    result = of.collect_final_output_files("session")
    download_dir = session_dir / "output_final_download"
    assert result == [
        str((download_dir / "notes.txt").resolve()),
        str((download_dir / "report.pdf").resolve()),
    ]
    assert (download_dir / "report.pdf").read_bytes() == b"new"
    assert (download_dir / "notes.txt").read_text() == "notes"


def test_workspace_root_from_uses_session_hash_only(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    base.mkdir()
    session_dir = base / "abc123"
    session_dir.mkdir()

    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")

    assert of.workspace_root_from("abc123") == session_dir
    assert of.workspace_root_from(None) == base.resolve()
    evil = of.workspace_root_from("/etc/passwd")
    evil.relative_to(base.resolve())
