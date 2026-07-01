"""Tests for AgentCore workspace upload bridge."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

_PI_SRC = Path(__file__).resolve().parents[1] / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

from pi_test_support import ensure_gradio_importable

ensure_gradio_importable()

import agentcore_workspace_bridge as bridge


def test_discover_session_document_name_picks_newest_pdf(tmp_path, monkeypatch):
    import time

    base = tmp_path / "workspace"
    session = base / "sess"
    session.mkdir(parents=True)
    older = session / "old.pdf"
    older.write_bytes(b"%PDF-1.4 old")
    time.sleep(0.05)
    newer = session / "new.pdf"
    newer.write_bytes(b"%PDF-1.4 new")

    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")

    assert bridge.discover_session_document_name("sess") == "new.pdf"


def test_collect_session_files_for_agentcore_upload_includes_redact_tree(
    tmp_path, monkeypatch
):
    base = tmp_path / "workspace"
    session = base / "sess"
    doc = session / "report.pdf"
    review = (
        session / "redact" / "report.pdf" / "output_redact" / "report_review_file.csv"
    )
    review.parent.mkdir(parents=True)
    doc.write_bytes(b"%PDF-1.4 source")
    review.write_text("id,page\n", encoding="utf-8-sig")
    preview = session / "preview" / "latest_redacted.pdf"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"%PDF-1.4 preview")

    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")

    staged = bridge.collect_session_files_for_agentcore_upload("sess")
    paths = {item["relative_path"] for item in staged}
    assert "report.pdf" in paths
    assert "redact/report.pdf/output_redact/report_review_file.csv" in paths
    assert "preview/latest_redacted.pdf" not in paths


def test_build_agentcore_followup_context_mentions_review_csv(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    session = base / "sess"
    csv = session / "redact" / "doc.pdf" / "output_redact" / "doc_review_file.csv"
    csv.parent.mkdir(parents=True)
    (session / "doc.pdf").write_bytes(b"%PDF")
    csv.write_text("id\n", encoding="utf-8-sig")

    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")

    text = bridge.build_agentcore_followup_context(
        "sess",
        [{"role": "user", "content": "Redact Dr Hyde"}],
    )
    assert "follow-up" in text.lower()
    assert "doc_review_file.csv" in text
    assert "Redact Dr Hyde" in text


def test_collect_upload_skips_oversized_files(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    session = base / "sess"
    big = session / "redact" / "doc.pdf" / "output_redact" / "big.bin"
    big.parent.mkdir(parents=True)
    big.write_bytes(b"x" * 20)
    small = session / "redact" / "doc.pdf" / "output_redact" / "notes.txt"
    small.write_text("ok")

    monkeypatch.setenv("PI_WORKSPACE_DIR", str(base))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "true")
    monkeypatch.setenv("AGENTCORE_MAX_UPLOAD_BYTES", "10")

    staged = bridge.collect_session_files_for_agentcore_upload("sess")
    assert staged
    skipped = staged[-1]
    assert skipped["relative_path"] == ".agentcore_upload_skipped.txt"
    note = base64.b64decode(skipped["content_base64"]).decode("utf-8")
    assert "big.bin" in note
