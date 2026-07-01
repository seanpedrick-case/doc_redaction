"""Tests for Pi workspace FileExplorer → gr.File download wiring."""

import sys
from pathlib import Path
from types import ModuleType

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

from pi_test_support import ensure_gradio_importable

ensure_gradio_importable()

if "pi_examples" not in sys.modules:
    _pi_examples = ModuleType("pi_examples")
    _pi_examples.gradio_example_allowed_paths = lambda: []  # type: ignore[attr-defined]
    sys.modules["pi_examples"] = _pi_examples

import output_files as of


def _minimal_pdf_bytes(body: bytes = b"content") -> bytes:
    """Build a PDF blob that passes ``_is_valid_pdf_file`` (size + %%EOF)."""
    min_bytes = 1024
    payload = b"%PDF-1.4\n" + body + b"\n%%EOF"
    if len(payload) < min_bytes:
        payload += b" " * (min_bytes - len(payload))
    return payload


def test_resolve_under_workspace_accepts_absolute_paths(tmp_path, monkeypatch):
    """Gradio FileExplorer preprocess joins root_dir and returns absolute paths."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    doc = workspace / "redact" / "demo"
    doc.mkdir(parents=True)
    pdf = doc / "output_redact.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    resolved = of._resolve_under_workspace(str(pdf), workspace_root=workspace)
    assert resolved == pdf.resolve()


def test_resolve_under_workspace_accepts_relative_paths(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pdf = workspace / "report.csv"
    pdf.write_text("a,b\n1,2\n")

    resolved = of._resolve_under_workspace("report.csv", workspace_root=workspace)
    assert resolved == pdf.resolve()


def test_resolve_under_workspace_rejects_outside_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.pdf"
    outside.write_bytes(b"%PDF")

    assert of._resolve_under_workspace(str(outside), workspace_root=workspace) is None


def test_resolve_under_workspace_rejects_path_traversal(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.pdf"
    outside.write_bytes(b"%PDF")

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


def test_final_download_dir_isolated_per_session_when_workspace_shared(
    tmp_path, monkeypatch
):
    base = tmp_path / "workspace"
    base.mkdir()
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "false")

    assert (
        of.final_download_dir("user_a")
        == (base / "user_a" / "output_final_download").resolve()
    )
    assert (
        of.final_download_dir("user_b")
        == (base / "user_b" / "output_final_download").resolve()
    )


def test_reset_download_dir_clears_without_removing_root(tmp_path):
    download_dir = tmp_path / "session" / "output_final_download"
    download_dir.mkdir(parents=True)
    stale = download_dir / "old.pdf"
    stale.write_bytes(b"%PDF")
    nested = download_dir / "nested"
    nested.mkdir()
    (nested / "x.txt").write_text("x")

    of._reset_download_dir(download_dir)

    assert download_dir.is_dir()
    assert list(download_dir.iterdir()) == []


def test_latest_redacted_pdf_path_returns_newest_match(tmp_path, monkeypatch):
    import time

    base = tmp_path / "workspace"
    session_dir = base / "session"
    draft_dir = session_dir / "redact" / "doc.pdf" / "output_redact"
    draft_dir.mkdir(parents=True)
    older = draft_dir / "doc_redacted.pdf"
    older.write_bytes(_minimal_pdf_bytes(b"draft"))
    time.sleep(0.02)
    final_dir = session_dir / "redact" / "doc.pdf" / "review" / "output_review_final"
    final_dir.mkdir(parents=True)
    newer = final_dir / "doc_redacted.pdf"
    newer.write_bytes(_minimal_pdf_bytes(b"final content"))
    unrelated = session_dir / "notes.txt"
    unrelated.write_text("x")

    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")

    path = of.latest_redacted_pdf_path("session")
    assert path is not None
    assert path.endswith("latest_redacted.pdf")
    staged = Path(path)
    assert staged.is_file()
    assert staged.read_bytes().startswith(b"%PDF-1.4")
    assert "final content" in staged.read_bytes().decode("latin-1")


def test_latest_redacted_pdf_path_prefers_final_output_over_newer_intermediate(
    tmp_path, monkeypatch
):
    import time

    base = tmp_path / "workspace"
    session_dir = base / "session"
    final_dir = session_dir / "redact" / "doc.pdf" / "review" / "output_review_final"
    final_dir.mkdir(parents=True)
    final_pdf = final_dir / "doc_redacted.pdf"
    final_pdf.write_bytes(_minimal_pdf_bytes(b"final deliverable"))
    time.sleep(0.02)
    draft_dir = session_dir / "redact" / "doc.pdf" / "output_redact"
    draft_dir.mkdir(parents=True)
    draft_pdf = draft_dir / "doc_redacted.pdf"
    draft_pdf.write_bytes(_minimal_pdf_bytes(b"intermediate draft"))

    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")

    path = of.latest_redacted_pdf_path("session")
    assert path is not None
    staged = Path(path)
    assert b"final deliverable" in staged.read_bytes()


def test_latest_redacted_pdf_path_uses_posix_separators(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    session_dir = base / "session"
    out_dir = session_dir / "redact" / "doc.pdf" / "output_redact"
    out_dir.mkdir(parents=True)
    (out_dir / "doc_redacted.pdf").write_bytes(_minimal_pdf_bytes())

    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")

    path = of.latest_redacted_pdf_path("session")
    assert path is not None
    assert "\\" not in path
    assert "/preview/latest_redacted.pdf" in path


def test_latest_redacted_pdf_path_skips_invalid_and_review_pdfs(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    session_dir = base / "session"
    out_dir = session_dir / "redact" / "doc.pdf" / "output_redact"
    out_dir.mkdir(parents=True)
    (out_dir / "doc_redactions_for_review.pdf").write_bytes(b"%PDF-1.4 review")
    (out_dir / "error_redacted.pdf").write_text("<html>429 Too Many Requests</html>")
    valid = out_dir / "doc_redacted.pdf"
    valid.write_bytes(_minimal_pdf_bytes(b"valid"))

    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")

    path = of.latest_redacted_pdf_path("session")
    assert path is not None
    assert Path(path).read_bytes().startswith(b"%PDF-1.4")


def test_latest_redacted_pdf_path_returns_none_when_missing(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    base.mkdir()
    session_dir = base / "session"
    session_dir.mkdir()

    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")

    assert of.latest_redacted_pdf_path("session") is None


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
