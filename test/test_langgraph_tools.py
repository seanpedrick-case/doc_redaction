"""Tests for LangGraph redaction tools (coverage, workspace I/O)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_AGENT_REDACT = Path(__file__).resolve().parents[1] / "agent-redact"
if str(_AGENT_REDACT) not in sys.path:
    sys.path.insert(0, str(_AGENT_REDACT))

from pi_test_support import ensure_gradio_importable

ensure_gradio_importable()

from redaction_langgraph.tools import (  # noqa: E402
    _coerce_relative_path,
    _coerce_tool_text_content,
    _default_dest_for_pdf,
    _default_review_apply_dest_for_review_csv,
    _discover_ocr_words_csv,
    _ensure_workspace_output_dir,
    _parse_doc_redact_tool_input,
    _parse_write_workspace_text_input,
    _resolve_workspace_path,
    _resolve_workspace_pdf,
    read_workspace_text,
    run_doc_redact,
    run_review_apply,
    write_workspace_text,
)
from redaction_langgraph.verify_coverage_lib import (  # noqa: E402
    compile_patterns,
    normalize_regex_patterns,
)


def test_discover_ocr_words_csv(tmp_path):
    review = tmp_path / "doc_review_file.csv"
    review.write_text("id,page\n", encoding="utf-8-sig")
    words = tmp_path / "doc_word_level_ocr.csv"
    words.write_text("word_text,page\n", encoding="utf-8-sig")
    assert _discover_ocr_words_csv(review) == words


def test_read_workspace_text_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "1")
    out = read_workspace_text("missing.csv", session_hash="sess")
    payload = json.loads(out)
    assert "error" in payload


def test_workspace_text_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "1")
    out = write_workspace_text("a.txt", "hello", session_hash="sess")
    assert json.loads(out)["written"] == "a.txt"
    assert read_workspace_text("a.txt", session_hash="sess") == "hello"


def test_normalize_regex_patterns_pipe_string():
    raw = r"Hyde|Lauren\s+Lilley|Lauren|Lilley|University\s+of\s+Notre\s+Dame|Notre\s+Dame"
    assert normalize_regex_patterns(raw) == [
        "Hyde",
        r"Lauren\s+Lilley",
        "Lauren",
        "Lilley",
        r"University\s+of\s+Notre\s+Dame",
        r"Notre\s+Dame",
    ]


def test_compile_patterns_accepts_pipe_string_not_characters():
    patterns = compile_patterns(r"Kornbluth|Poss\b")
    assert len(patterns) == 2
    assert patterns[0].search("Kornbluth")
    assert patterns[1].search("Poss")


def test_compile_patterns_user_pipe_string():
    raw = r"Hyde|Lauren|Lilley|University of Notre Dame|David R\."
    patterns = compile_patterns(raw)
    assert len(patterns) == 5
    assert patterns[0].search("Hyde")
    assert patterns[-1].search("David R.")


def test_compile_patterns_fallback_literal_on_invalid_regex():
    patterns = compile_patterns(r"bad[")
    assert len(patterns) == 1
    assert patterns[0].search("bad[")


def test_coerce_relative_path_from_nested_dict():
    assert (
        _coerce_relative_path(
            {"dest_relative_dir": "redact/doc/output_redact"},
            label="dest_relative_dir",
        )
        == "redact/doc/output_redact"
    )


def test_resolve_workspace_path_accepts_dict_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "1")
    resolved = _resolve_workspace_path(
        "sess",
        {"pdf_relative_path": "uploads/doc.pdf"},
    )
    assert resolved == (tmp_path / "sess" / "uploads" / "doc.pdf").resolve()


def test_run_doc_redact_accepts_merged_tool_args_dict(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "1")
    session = tmp_path / "sess"
    pdf = session / "uploads" / "doc.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4")

    captured: dict[str, Any] = {}

    def fake_call_doc_redact(pdf_path, dest_dir, **kwargs):
        captured["pdf"] = Path(pdf_path)
        captured["dest"] = Path(dest_dir)
        captured["kwargs"] = kwargs
        return (
            ["C:/server/out/doc_review_file.csv"],
            [dest_dir / "doc_review_file.csv"],
        )

    monkeypatch.setattr(
        "redaction_langgraph.tools.call_doc_redact",
        fake_call_doc_redact,
    )

    payload = {
        "pdf_relative_path": "uploads/doc.pdf",
        "dest_relative_dir": "redact/doc/output_redact",
    }
    out = run_doc_redact(payload, None, session_hash="sess")
    data = json.loads(out)
    assert "error" not in data
    assert captured["pdf"] == pdf.resolve()
    assert captured["dest"] == (session / "redact/doc/output_redact").resolve()


def test_parse_doc_redact_tool_input_ignores_garbage_keys():
    messy = {
        "pdf_path": "example_of_emails_sent_to_a_professor_before_applying.pdf",
        "ocr_method": "Local model - selectable text",
        "pii_method": "Local",
        "}] }' http://host.docker.internal:7861/api/call/doc_redact": -1,
    }
    pdf_rel, dest_rel, ocr, pii = _parse_doc_redact_tool_input(
        "ignored.pdf",
        messy,
        ocr_method=None,
        pii_method=None,
    )
    assert pdf_rel == "example_of_emails_sent_to_a_professor_before_applying.pdf"
    assert dest_rel == (
        "redact/example_of_emails_sent_to_a_professor_before_applying/output_redact"
    )
    assert ocr == "Local model - selectable text"
    assert pii == "Local"


def test_default_dest_for_pdf():
    assert _default_dest_for_pdf("uploads/doc.pdf") == "redact/doc/output_redact"


def test_resolve_workspace_pdf_by_basename(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "1")
    session = tmp_path / "sess"
    pdf = session / "nested" / "doc.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4")
    resolved = _resolve_workspace_pdf("sess", "doc.pdf")
    assert resolved == pdf.resolve()


def test_read_workspace_text_nested_relative_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "1")
    session = tmp_path / "sess"
    csv_path = session / "redact" / "doc_review_file.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("id,page\n1,1\n", encoding="utf-8-sig")
    nested = {
        "relative_path": {
            "relative_path": "redact/doc_review_file.csv",
        }
    }
    assert read_workspace_text(nested, session_hash="sess") == "id,page\n1,1\n"


def test_read_workspace_text_csv_preview(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "1")
    monkeypatch.setenv("LANGGRAPH_READ_CSV_MAX_LINES", "2")
    session = tmp_path / "sess"
    csv_path = session / "big.csv"
    session.mkdir(parents=True)
    csv_path.write_text("a\nb\nc\nd\n", encoding="utf-8-sig")
    out = read_workspace_text("big.csv", session_hash="sess")
    assert "CSV preview" in out
    assert "lines 1-2 of 4" in out
    assert "a\nb" in out
    assert "\nc\n" not in out


def test_coerce_tool_text_content_extension_key_dict():
    script = "import csv\nprint('ok')\n"
    assert _coerce_tool_text_content({".py": script}) == script


def test_write_workspace_text_messy_local_model_args(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "1")
    script = "import csv\nprint('ok')\n"
    messy = {
        "relative_path": {
            "relative_path": "redact/example/fix_policy.py",
        },
        "content": {".py": script},
    }
    out = write_workspace_text(messy, None, session_hash="sess")
    data = json.loads(out)
    assert "error" not in data
    written = tmp_path / "sess" / "redact" / "example" / "fix_policy.py"
    assert written.read_text(encoding="utf-8-sig") == script


def test_parse_write_workspace_text_input():
    script = "import csv\n"
    rel, body = _parse_write_workspace_text_input(
        {
            "relative_path": {"relative_path": "redact/a/fix_policy.py"},
            "content": {".py": script},
        },
        None,
    )
    assert rel == "redact/a/fix_policy.py"
    assert body == script


def test_parse_write_workspace_text_input_script_content_dict():
    script = "import csv\nprint('ok')\n"
    rel, body = _parse_write_workspace_text_input(
        {"script": "fix_policy.py", "content": script},
        None,
    )
    assert rel == "fix_policy.py"
    assert body == script


def test_parse_write_workspace_text_input_doubly_nested():
    script = "import csv\n"
    rel, body = _parse_write_workspace_text_input(
        {
            "relative_path": {"relative_path": "fix_review.py"},
            "content": {"content": script},
        },
        None,
    )
    assert rel == "fix_review.py"
    assert body == script


def test_write_workspace_text_python_next_step(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "1")
    script = "import csv\nprint('ok')\n"
    out = write_workspace_text(
        {
            "relative_path": {"relative_path": "fix_review.py"},
            "content": {"content": script},
        },
        None,
        session_hash="sess",
    )
    data = json.loads(out)
    assert "error" not in data
    assert data["written"].endswith("fix_review.py")
    assert "next_step" in data
    assert "run_workspace_python_script" in data["next_step"]
    out2 = write_workspace_text(
        {
            "relative_path": {"relative_path": "fix_review.py"},
            "content": {"content": script},
        },
        None,
        session_hash="sess",
    )
    data2 = json.loads(out2)
    assert data2.get("unchanged") is True


def test_write_workspace_text_script_content_dict(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "1")
    script = "import csv\nprint('ok')\n"
    out = write_workspace_text(
        {"script": "fix_policy.py", "content": script},
        None,
        session_hash="sess",
    )
    data = json.loads(out)
    assert "error" not in data
    written = tmp_path / "sess" / "scripts" / "fix_policy.py"
    assert written.read_text(encoding="utf-8-sig") == script
    assert "next_step" in data


def test_default_review_apply_dest_for_review_csv():
    review_csv = (
        "redact/example_of_emails_sent_to_a_professor_before_applying/"
        "output_redact/abc_review_file.csv"
    )
    assert _default_review_apply_dest_for_review_csv(review_csv) == (
        "redact/example_of_emails_sent_to_a_professor_before_applying/"
        "review/output_review_final"
    )


def test_ensure_workspace_output_dir_repairs_pdf_dest(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "1")
    session = tmp_path / "sess"
    pdf_name = "example_of_emails_sent_to_a_professor_before_applying.pdf"
    pdf = session / pdf_name
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4")
    review_csv = (
        "redact/example_of_emails_sent_to_a_professor_before_applying/"
        "output_redact/abc_review_file.csv"
    )
    dest = _ensure_workspace_output_dir(
        "sess",
        pdf_name,
        pdf_relative_path=pdf_name,
        review_csv_relative_path=review_csv,
        default_for="review_apply",
    )
    assert (
        dest
        == (
            session
            / "redact"
            / "example_of_emails_sent_to_a_professor_before_applying"
            / "review"
            / "output_review_final"
        ).resolve()
    )
    assert dest.is_dir()


def test_run_review_apply_repairs_pdf_dest(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "1")
    session = tmp_path / "sess"
    pdf_name = "example_of_emails_sent_to_a_professor_before_applying.pdf"
    pdf = session / pdf_name
    review_csv = (
        session
        / "redact"
        / "example_of_emails_sent_to_a_professor_before_applying"
        / "output_redact"
        / "abc_review_file.csv"
    )
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4")
    review_csv.parent.mkdir(parents=True)
    review_csv.write_text("page,text,label\n1,foo,REDACT\n", encoding="utf-8-sig")

    captured: dict[str, Any] = {}

    class _FakeClient:
        def predict(self, **kwargs):
            captured["kwargs"] = kwargs
            return (["C:/server/out/final_redacted.pdf"], "ok")

    monkeypatch.setattr(
        "redaction_langgraph.tools.make_redaction_client",
        lambda: _FakeClient(),
    )
    monkeypatch.setattr(
        "redaction_langgraph.tools.fetch_redaction_files",
        lambda paths, dest: [dest / "final_redacted.pdf"],
    )

    out = run_review_apply(
        pdf_name,
        str(review_csv.relative_to(session)).replace("\\", "/"),
        pdf_name,
        session_hash="sess",
    )
    data = json.loads(out)
    assert "error" not in data
    assert captured["kwargs"]["pdf_file"] is not None
    assert captured["kwargs"]["review_csv_file"] is not None
    assert data["saved_paths"][0].endswith("final_redacted.pdf")
