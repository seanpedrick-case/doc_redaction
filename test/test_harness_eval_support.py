"""Tests for harness post-run eval support."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.harness_eval_support import (
    HarnessEvalRecorder,
    harness_agent_turn_span,
    harness_inline_tool_span,
    harness_post_run_eval_enabled,
    maybe_setup_harness_arize_tracing,
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
    assert any(e.get("kind") == "tool_start" for e in rec.trace_events)


def test_recorder_turn_and_tool_end_journal(
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
    rec = HarnessEvalRecorder(session_hash="sess-j")
    ws = tmp_path / "ws"
    ws.mkdir()
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    rec.begin(
        workspace=ws,
        pdf_path=pdf,
        document_name="a.pdf",
        instructions="task",
        text_extract_method="Local text",
        pii_method="Local",
        ocr_method_short="Local text",
    )
    rec.record_turn_start(round_no=1)
    rec.record_tool("doc_redact", {"document_file": "a.pdf"})
    rec.record_tool_end(
        "doc_redact",
        ok=True,
        duration_ms=12.5,
        output='{"paths":[]}',
        tool_use_id="tu-1",
    )
    rec.record_turn_end(
        round_no=1,
        stop_reason="tool_use",
        input_tokens=10,
        output_tokens=20,
        pending_tools=["doc_redact"],
    )
    rec.record_stream_status("max_rounds", rounds=60)
    kinds = [e["kind"] for e in rec.trace_events]
    assert kinds == [
        "turn_start",
        "tool_start",
        "tool_end",
        "turn_end",
        "stream_status",
    ]
    end = next(e for e in rec.trace_events if e["kind"] == "tool_end")
    assert end["ok"] is True
    assert end["duration_ms"] == 12.5
    assert end["tool"] == "doc_redact"

    run_dir = ws / "eval" / "run_test"
    run_dir.mkdir(parents=True)
    path = rec.write_trace_events_file(run_dir)
    assert path is not None and path.is_file()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    assert json.loads(lines[0])["kind"] == "turn_start"


def test_write_post_run_force_when_app_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_write(**kwargs):
        captured.update(kwargs)
        run_dir = tmp_path / "eval" / "run_fake"
        run_dir.mkdir(parents=True)
        summary = run_dir / "eval_summary.json"
        summary.write_text("{}", encoding="utf-8")
        return summary

    monkeypatch.setattr(
        "tools.harness_eval_support.harness_post_run_eval_enabled",
        lambda: True,
    )
    monkeypatch.setattr("tools.config.APP_EVAL_ENABLED", True, raising=False)

    rec = HarnessEvalRecorder(run_source="main_app_harness")
    rec.started = True
    rec.cumulative_tool_calls = [{"tool": "doc_redact", "args": {}}]
    rec.trace_events = [{"kind": "turn_start", "round_no": 1, "ts": 1.0}]

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
    assert out is not None
    assert captured.get("force") is True
    assert captured.get("orchestrator") == "main-app-harness"
    assert captured.get("run_source") == "main_app_harness"
    journal = out.parent / "trace_events.jsonl"
    assert journal.is_file()
    assert "turn_start" in journal.read_text(encoding="utf-8")


def test_harness_spans_noop_without_arize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARIZE_TRACING_ENABLED", raising=False)
    monkeypatch.setattr(
        "tools.harness_eval_support._ARIZE_SETUP_DONE",
        False,
    )
    assert maybe_setup_harness_arize_tracing() is False
    with harness_agent_turn_span(session_hash="s", message="hi") as meta:
        assert meta.get("span") is None
        with harness_inline_tool_span(
            tool_name="doc_redact",
            arguments={"x": 1},
            tool_use_id="id1",
        ) as tool_meta:
            tool_meta["output"] = "{}"
            tool_meta["ok"] = True
            assert tool_meta.get("span") is None
