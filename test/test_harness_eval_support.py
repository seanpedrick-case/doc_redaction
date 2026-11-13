"""Tests for harness post-run eval support."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.harness_eval_support import (
    HarnessEvalRecorder,
    harness_post_run_eval_enabled,
    normalize_harness_tool_name,
)


def test_normalize_harness_tool_name() -> None:
    assert normalize_harness_tool_name("verify_redaction_coverage") == "verify_coverage"
    assert normalize_harness_tool_name("doc_redact") == "doc_redact"


def test_harness_post_run_eval_enabled_reads_app_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tools.harness_eval_support.harness_eval_tracing_enabled",
        lambda: False,
    )
    monkeypatch.setattr("tools.config.APP_EVAL_ENABLED", True, raising=False)
    assert harness_post_run_eval_enabled() is True


def test_recorder_collects_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tools.harness_eval_support.harness_post_run_eval_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "tools.harness_eval_support.resolve_harness_gold_case",
        lambda *a, **k: (None, ""),
    )
    rec = HarnessEvalRecorder(session_hash="sess-1")
    ws = tmp_path / "agentcore" / "run1"
    ws.mkdir(parents=True)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    rec.begin(
        workspace=ws,
        pdf_path=pdf,
        document_name="doc.pdf",
        instructions="redact names",
        text_extract_method="Selectable text",
        pii_method="Local",
        ocr_method_short="",
    )
    rec.record_tool("verify_redaction_coverage", {"pdf_path": "x"})
    assert rec.cumulative_tool_calls == [
        {"tool": "verify_coverage", "args": {"pdf_path": "x"}},
    ]


def test_write_post_run_force_when_app_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_write(**kwargs):
        captured.update(kwargs)
        return tmp_path / "eval_summary.json"

    monkeypatch.setattr(
        "tools.harness_eval_support.harness_post_run_eval_enabled",
        lambda: True,
    )
    monkeypatch.setattr("tools.config.APP_EVAL_ENABLED", True, raising=False)

    rec = HarnessEvalRecorder()
    rec.started = True
    rec.cumulative_tool_calls = [{"tool": "doc_redact", "args": {}}]

    import tools.harness_eval_support as hes

    monkeypatch.setattr(hes, "_ensure_agent_eval_paths", lambda: None)

    def import_post_run_eval():
        class _M:
            write_session_post_run_eval = staticmethod(fake_write)

        return _M

    monkeypatch.setitem(
        __import__("sys").modules,
        "post_run_eval",
        import_post_run_eval(),  # type: ignore[arg-type]
    )

    ws = tmp_path / "run"
    ws.mkdir()
    out = rec.write_post_run(
        workspace=ws,
        document_name="d.pdf",
        instructions="task",
        assistant_output="done",
        workflow_incomplete=False,
        session_hash="h1",
    )
    assert out == tmp_path / "eval_summary.json"
    assert captured.get("force") is True
    assert captured.get("orchestrator") == "main-app-harness"
