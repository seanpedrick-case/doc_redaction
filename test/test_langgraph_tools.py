"""Tests for LangGraph redaction tools (coverage, workspace I/O)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pi_test_support import ensure_agent_redact_paths, ensure_gradio_importable

ensure_agent_redact_paths()
ensure_gradio_importable()

from redaction_langgraph.tools import (  # noqa: E402
    _coerce_relative_path,
    _coerce_tool_text_content,
    _default_dest_for_pdf,
    _default_review_apply_dest_for_review_csv,
    _discover_ocr_words_csv,
    _ensure_workspace_output_dir,
    _normalize_review_color_cell,
    _parse_doc_redact_tool_input,
    _parse_review_apply_tool_input,
    _parse_write_workspace_text_input,
    _remember_session_artifacts,
    _repair_review_csv_body,
    _resolve_optional_redacted_pdf,
    _resolve_workspace_path,
    _resolve_workspace_pdf,
    _validate_review_csv_body,
    normalize_tool_args,
    read_workspace_text,
    reset_langgraph_tool_session_state,
    run_doc_redact,
    run_review_apply,
    run_verify_coverage,
    write_workspace_text,
)
from redaction_langgraph.verify_coverage_lib import (  # noqa: E402
    compile_patterns,
    normalize_regex_patterns,
)


def test_normalize_review_color_cell():
    assert _normalize_review_color_cell("(12, 34, 56)") == "(12, 34, 56)"
    assert _normalize_review_color_cell("0,0,0") == "(0, 0, 0)"
    assert _normalize_review_color_cell("black") == "(0, 0, 0)"
    assert _normalize_review_color_cell("#ff0000") == "(255, 0, 0)"
    assert _normalize_review_color_cell("placeholder") == "(0, 0, 0)"
    assert _normalize_review_color_cell("") == "(0, 0, 0)"
    assert _normalize_review_color_cell((1, 2, 3)) == "(1, 2, 3)"


def test_repair_review_csv_body_colors():
    body = (
        "page,xmin,xmax,ymin,ymax,color,text\n"
        "1,0.1,0.2,0.3,0.4,black,Name\n"
        "1,0.1,0.2,0.3,0.4,0,0,0,Other\n"
    )
    # The second row has broken CSV because color "0,0,0" splits columns —
    # use a quoted form instead.
    body = (
        "page,xmin,xmax,ymin,ymax,color,text\n"
        "1,0.1,0.2,0.3,0.4,black,Name\n"
        '1,0.2,0.3,0.4,0.5,"0, 0, 0",Other\n'
        "1,0.3,0.4,0.5,0.6,placeholder,Third\n"
    )
    repaired, changed = _repair_review_csv_body(body)
    assert changed == 3
    assert "(0, 0, 0)" in repaired
    assert "black" not in repaired
    assert "placeholder" not in repaired


def test_write_workspace_text_repairs_review_colors(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    reset_langgraph_tool_session_state("sess")
    body = "page,xmin,xmax,ymin,ymax,color,text\n" "1,0.1,0.2,0.3,0.4,black,Name\n"
    out = write_workspace_text(
        "redact/doc/output_redact/doc_review_file.csv",
        body,
        session_hash="sess",
    )
    data = json.loads(out)
    assert "error" not in data
    assert data.get("color_cells_repaired") == 1
    written = (
        tmp_path / "sess" / "redact" / "doc" / "output_redact" / "doc_review_file.csv"
    )
    text = written.read_text(encoding="utf-8-sig")
    assert "(0, 0, 0)" in text
    assert "black" not in text


def test_discover_ocr_words_csv(tmp_path):
    review = tmp_path / "doc_review_file.csv"
    review.write_text("id,page\n", encoding="utf-8-sig")
    words = tmp_path / "doc_word_level_ocr.csv"
    words.write_text("word_text,page\n", encoding="utf-8-sig")
    assert _discover_ocr_words_csv(review) == words


def test_discover_ocr_results_with_words_preferred(tmp_path):
    review = tmp_path / "doc_review_file.csv"
    review.write_text("id,page\n", encoding="utf-8-sig")
    other = tmp_path / "doc_ocr_summary.csv"
    other.write_text("a,b\n", encoding="utf-8-sig")
    words = tmp_path / "doc_0_0_ocr_results_with_words_local_ocr.csv"
    words.write_text("word_text,page\n", encoding="utf-8-sig")
    assert _discover_ocr_words_csv(review) == words


def test_normalize_tool_args_flattens_nested_path():
    out = normalize_tool_args(
        "doc_redact",
        {"pdf_relative_path": {"pdf_relative_path": "file.pdf"}},
    )
    assert out["pdf_relative_path"] == "file.pdf"


def test_normalize_tool_args_wrong_inner_key():
    out = normalize_tool_args(
        "doc_redact",
        {"pdf_relative_path": {"relative_path": "file.pdf"}},
    )
    assert out["pdf_relative_path"] == "file.pdf"


def test_validate_review_csv_rejects_placeholder():
    body = "page,xmin,xmax,ymin,ymax,text\n1,placeholder,0.2,0.1,0.2,Name\n"
    err = _validate_review_csv_body(body)
    assert err is not None
    assert "placeholder" in err.lower()


def test_validate_review_csv_accepts_numeric():
    body = "page,xmin,xmax,ymin,ymax,text\n1,0.1,0.2,0.3,0.4,Name\n"
    assert _validate_review_csv_body(body) is None


def test_write_workspace_text_rejects_placeholder_review_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    reset_langgraph_tool_session_state("sess")
    body = "page,xmin,xmax,ymin,ymax,text\n1,placeholder,0.2,0.1,0.2,Name\n"
    out = write_workspace_text(
        "redact/doc/output_redact/doc_review_file.csv",
        body,
        session_hash="sess",
    )
    data = json.loads(out)
    assert "error" in data
    assert "placeholder" in data["error"].lower()


def test_write_storm_blocks_third_python_rewrite(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    reset_langgraph_tool_session_state("sess")
    script_a = "import csv\nprint(1)\n"
    script_b = "import csv\nprint(2)\n"
    script_c = "import csv\nprint(3)\n"
    out1 = write_workspace_text("fix_review.py", script_a, session_hash="sess")
    assert "error" not in json.loads(out1)
    out2 = write_workspace_text("fix_review.py", script_b, session_hash="sess")
    assert "error" not in json.loads(out2)
    out3 = write_workspace_text("fix_review.py", script_c, session_hash="sess")
    data3 = json.loads(out3)
    assert "error" in data3
    assert data3.get("blocked_write_storm") is True
    assert "run_workspace_python_script" in data3["error"]


def test_read_workspace_text_autofills_empty_dict(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    reset_langgraph_tool_session_state("sess")
    session = tmp_path / "sess"
    review = session / "redact" / "doc" / "output_redact" / "doc_review_file.csv"
    ocr = (
        session
        / "redact"
        / "doc"
        / "output_redact"
        / "doc_ocr_results_with_words_local_ocr.csv"
    )
    review.parent.mkdir(parents=True)
    review.write_text("page,text\n1,hello\n", encoding="utf-8-sig")
    ocr.write_text("word_text,page\nhello,1\n", encoding="utf-8-sig")
    _remember_session_artifacts(
        "sess",
        review_csv_relative_path=str(review.relative_to(session)).replace("\\", "/"),
        ocr_words_csv_relative_path=str(ocr.relative_to(session)).replace("\\", "/"),
    )
    out1 = read_workspace_text({}, session_hash="sess")
    assert "auto-filled relative_path=" in out1
    assert "doc_review_file.csv" in out1
    assert "hello" in out1
    # Second empty read rotates to OCR words CSV.
    out2 = read_workspace_text({"relative_path": {}}, session_hash="sess")
    assert "ocr_results_with_words" in out2
    assert "hello" in out2


def test_read_workspace_text_empty_dict_discovers_review_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    reset_langgraph_tool_session_state("sess")
    session = tmp_path / "sess"
    review = session / "output_redact" / "example_review_file.csv"
    review.parent.mkdir(parents=True)
    review.write_text("page,text\n1,world\n", encoding="utf-8-sig")
    out = read_workspace_text({}, session_hash="sess")
    assert "error" not in out[:20]
    assert "world" in out
    assert "example_review_file.csv" in out


def test_write_content_soft_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    monkeypatch.setenv("LANGGRAPH_MAX_WRITE_CONTENT_BYTES", "100")
    # Re-import bound constant is read at module load; patch the module attribute.
    import redaction_langgraph.tools as tools_mod

    monkeypatch.setattr(tools_mod, "_MAX_WRITE_CONTENT_BYTES", 100)
    reset_langgraph_tool_session_state("sess")
    body = "x" * 150
    out = write_workspace_text("notes.txt", body, session_hash="sess")
    data = json.loads(out)
    assert "error" in data
    assert "too large" in data["error"].lower()


def test_read_workspace_text_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    out = read_workspace_text("missing.csv", session_hash="sess")
    payload = json.loads(out)
    assert "error" in payload


def test_workspace_text_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
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
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    resolved = _resolve_workspace_path(
        "sess",
        {"pdf_relative_path": "uploads/doc.pdf"},
    )
    assert resolved == (tmp_path / "sess" / "uploads" / "doc.pdf").resolve()


def test_run_doc_redact_accepts_merged_tool_args_dict(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
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


def test_parse_doc_redact_nested_absolute_path_key(tmp_path, monkeypatch):
    """Local Qwen models nest args under an absolute path dict key."""
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    abs_path = (
        tmp_path / "sess" / "example_of_emails_sent_to_a_professor_before_applying.pdf"
    ).as_posix()
    messy = {
        "pdf_relative_path": {
            abs_path: {
                "pdf_relative_path": (
                    "example_of_emails_sent_to_a_professor_before_applying.pdf"
                ),
                "dest_relative_dir": (
                    "redact/example_of_emails_sent_to_a_professor_before_applying/"
                    "output_redact"
                ),
            }
        }
    }
    pdf_rel, dest_rel, ocr, pii = _parse_doc_redact_tool_input(
        messy,
        None,
        ocr_method=None,
        pii_method=None,
        session_hash="sess",
    )
    assert pdf_rel == "example_of_emails_sent_to_a_professor_before_applying.pdf"
    assert dest_rel == (
        "redact/example_of_emails_sent_to_a_professor_before_applying/output_redact"
    )
    assert ocr is None
    assert pii is None


def test_parse_doc_redact_output_redact_as_pdf_autodiscovers(tmp_path, monkeypatch):
    """Models often pass output_redact in the PDF slot after compaction."""
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    reset_langgraph_tool_session_state("sess")
    session = tmp_path / "sess"
    session.mkdir()
    (session / "letter.pdf").write_bytes(b"%PDF-1.4")
    out = session / "redact" / "letter" / "output_redact"
    out.mkdir(parents=True)

    pdf_rel, dest_rel, ocr, pii = _parse_doc_redact_tool_input(
        "output_redact",
        None,
        ocr_method=None,
        pii_method=None,
        session_hash="sess",
    )
    assert pdf_rel == "letter.pdf"
    assert "output_redact" in dest_rel.replace("\\", "/")
    assert ocr is None
    assert pii is None


def test_parse_doc_redact_empty_object_autodiscovers_single_pdf(tmp_path, monkeypatch):
    """Local models often emit pdf_relative_path={} after losing the filename."""
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    session = tmp_path / "sess"
    session.mkdir()
    (session / "example.pdf").write_bytes(b"%PDF-1.4")
    # Redacted outputs must not be chosen as the source.
    out = session / "redact" / "example" / "output_redact"
    out.mkdir(parents=True)
    (out / "example_redacted.pdf").write_bytes(b"%PDF-1.4")

    pdf_rel, dest_rel, ocr, pii = _parse_doc_redact_tool_input(
        {"pdf_relative_path": {}},
        None,
        ocr_method=None,
        pii_method=None,
        session_hash="sess",
    )
    assert pdf_rel == "example.pdf"
    assert dest_rel == "redact/example/output_redact"
    assert ocr is None
    assert pii is None


def test_parse_doc_redact_empty_object_lists_choices_when_ambiguous(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    session = tmp_path / "sess"
    session.mkdir()
    (session / "a.pdf").write_bytes(b"%PDF-1.4")
    (session / "b.pdf").write_bytes(b"%PDF-1.4")

    try:
        _parse_doc_redact_tool_input(
            {},
            None,
            ocr_method=None,
            pii_method=None,
            session_hash="sess",
        )
        raise AssertionError("expected ValueError for ambiguous PDFs")
    except ValueError as exc:
        msg = str(exc)
        assert "plain string" in msg
        assert "a.pdf" in msg
        assert "b.pdf" in msg


def test_default_dest_for_pdf():
    assert _default_dest_for_pdf("uploads/doc.pdf") == "redact/doc/output_redact"


def test_resolve_workspace_pdf_by_basename(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    session = tmp_path / "sess"
    pdf = session / "nested" / "doc.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4")
    resolved = _resolve_workspace_pdf("sess", "doc.pdf")
    assert resolved == pdf.resolve()


def test_read_workspace_text_nested_relative_path(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
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
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
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
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
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
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    reset_langgraph_tool_session_state("sess")
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
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
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
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
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


def test_parse_review_apply_repairs_dest_dir_as_pdf(tmp_path, monkeypatch):
    """Model often puts output_review_final in the PDF slot near the end of a run."""
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    reset_langgraph_tool_session_state("sess")
    session = tmp_path / "sess"
    session.mkdir()
    (session / "source.pdf").write_bytes(b"%PDF-1.4")
    review = session / "redact" / "source" / "output_redact" / "source_review_file.csv"
    review.parent.mkdir(parents=True)
    review.write_text("page,text\n1,a\n", encoding="utf-8-sig")
    _remember_session_artifacts(
        "sess",
        pdf_relative_path="source.pdf",
        review_csv_relative_path=str(review.relative_to(session)).replace("\\", "/"),
    )
    pdf_rel, review_rel, dest_rel = _parse_review_apply_tool_input(
        "output_review_final",
        str(review.relative_to(session)).replace("\\", "/"),
        "",
        session_hash="sess",
    )
    assert pdf_rel == "source.pdf"
    assert review_rel.endswith("source_review_file.csv")
    assert "output_review_final" in dest_rel.replace("\\", "/")


def test_run_review_apply_dest_as_pdf_returns_json_error(tmp_path, monkeypatch):
    """Uncaught FileNotFoundError used to crash the LangGraph tools node."""
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    reset_langgraph_tool_session_state("sess")
    session = tmp_path / "sess"
    session.mkdir()
    # No source PDF — repair cannot autodiscover; must return JSON error, not raise.
    review = session / "doc_review_file.csv"
    review.write_text("page,text\n1,a\n", encoding="utf-8-sig")
    out = run_review_apply(
        "output_review_final",
        "doc_review_file.csv",
        "output_review_final",
        session_hash="sess",
    )
    data = json.loads(out)
    assert "error" in data
    assert "fix_example" in data


def test_run_review_apply_repairs_pdf_dest(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
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


def test_resolve_optional_redacted_pdf_rejects_review_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    session = tmp_path / "sess"
    review = session / "doc_review_file.csv"
    review.parent.mkdir(parents=True)
    review.write_text("id,page\n", encoding="utf-8-sig")
    try:
        _resolve_optional_redacted_pdf(
            "sess",
            "doc_review_file.csv",
            review_csv=review,
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "must be a PDF" in str(exc)


def test_run_verify_coverage_rejects_csv_as_redacted_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_SESSION_WORKSPACE", "1")
    session = tmp_path / "sess"
    out = session / "output_redact"
    out.mkdir(parents=True)
    review = out / "doc_review_file.csv"
    review.write_text("id,page,text\n1,1,hello\n", encoding="utf-8-sig")
    words = out / "doc_ocr_results_with_words_local_ocr.csv"
    words.write_text("word_text,page\nhello,1\n", encoding="utf-8-sig")
    result = run_verify_coverage(
        "output_redact/doc_review_file.csv",
        session_hash="sess",
        redacted_pdf_relative_path="output_redact/doc_review_file.csv",
    )
    data = json.loads(result)
    assert "error" in data
    assert "PDF" in data["error"]
    assert "hint" in data
