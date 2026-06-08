"""Tests for doc_redaction.cli_api output path discovery."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_PI_SRC = REPO_ROOT / "agent-redact" / "pi"
if str(_PI_SRC) not in sys.path:
    sys.path.insert(0, str(_PI_SRC))

from remote_redaction import (  # noqa: E402
    discover_redaction_outputs,
    resolve_redaction_output_paths,
)

from doc_redaction.cli_api import (  # noqa: E402
    _run_cli,
    _snapshot_files_newer_than,
)


def test_snapshot_files_newer_than_includes_overwritten_files(tmp_path: Path) -> None:
    existing = tmp_path / "doc_review_file.csv"
    existing.write_text("old", encoding="utf-8")
    time.sleep(0.02)
    started = time.time()
    time.sleep(0.02)
    existing.write_text("new", encoding="utf-8")

    found = _snapshot_files_newer_than(str(tmp_path), started)
    assert str(existing.resolve()) in found


def test_run_cli_returns_touched_files_on_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    effective = tmp_path / "session_out"
    effective.mkdir(parents=True)
    stale = effective / "example_review_file.csv"
    stale.write_text("from prior run", encoding="utf-8")

    def _fake_effective_output_dir(merged: dict) -> str:
        return str(effective)

    def _fake_cli_main(direct_mode_args: dict | None = None) -> None:
        target = effective / "example_review_file.csv"
        target.write_text("updated this run", encoding="utf-8")
        (effective / "example_redacted.pdf").write_bytes(b"%PDF")

    monkeypatch.setattr(
        "doc_redaction.cli_api._effective_output_dir",
        _fake_effective_output_dir,
    )
    monkeypatch.setattr("cli_redact.main", _fake_cli_main)

    paths = _run_cli(
        gradio_api_name="doc_redact",
        overrides={"task": "redact", "input_file": ["example.pdf"]},
        output_dir=str(tmp_path / "base_out"),
    )
    assert str((effective / "example_review_file.csv").resolve()) in paths
    assert str((effective / "example_redacted.pdf").resolve()) in paths


def test_resolve_redaction_output_paths_falls_back_to_discover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_root = tmp_path / "output" / "user_session"
    out_root.mkdir(parents=True)
    review = out_root / "example_of_emails_sent_review_file.csv"
    review.write_text("a,b\n", encoding="utf-8")

    monkeypatch.setattr(
        "remote_redaction.doc_redaction_output_root",
        lambda: tmp_path / "output",
    )

    paths = resolve_redaction_output_paths(
        ([], "doc_redact completed"),
        document_stem="example_of_emails_sent",
    )
    assert str(review.resolve()) in paths


def test_discover_redaction_outputs_respects_since(tmp_path: Path) -> None:
    out_root = tmp_path / "output"
    out_root.mkdir()
    old = out_root / "example_of_emails_old.csv"
    old.write_text("old", encoding="utf-8")
    time.sleep(0.02)
    since = time.time()
    time.sleep(0.02)
    new = out_root / "example_of_emails_new.csv"
    new.write_text("new", encoding="utf-8")

    import remote_redaction as rr

    original = rr.doc_redaction_output_root
    rr.doc_redaction_output_root = lambda: out_root
    try:
        found = discover_redaction_outputs("example_of_emails", since=since)
    finally:
        rr.doc_redaction_output_root = original

    assert str(new.resolve()) in found
    assert str(old.resolve()) not in found
