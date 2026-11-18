"""Tests for gold case auto-match (strict / soft tiers)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
_SHARED = _REPO / "agent-redact" / "shared"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from gold_match import (  # noqa: E402
    app_eval_entity_instructions_equivalent,
    normalize_instructions_text,
    resolve_gold_case_for_task,
    sha256_file,
)
from post_run_eval import resolve_eval_gold_dir  # noqa: E402


def test_normalize_instructions_strips_blank_edges():
    raw = "\r\n  line one  \n\nline two\n\n"
    assert normalize_instructions_text(raw) == "  line one\n\nline two"


def test_strict_match_single_case(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 gold-bytes")
    instructions = "Redact Alice.\nKeep Council."
    case = tmp_path / "case_a"
    case.mkdir()
    (case / "input.pdf").write_bytes(pdf.read_bytes())
    (case / "instructions.txt").write_text(instructions, encoding="utf-8")
    (case / "manifest.json").write_text(
        json.dumps({"case_id": "case_a", "input_pdf": "input.pdf"}),
        encoding="utf-8",
    )
    root = tmp_path / "gold_root"
    root.mkdir()
    shutil.copytree(case, root / "case_a")

    upload = tmp_path / "upload.pdf"
    upload.write_bytes(pdf.read_bytes())
    result = resolve_gold_case_for_task(
        pdf_path=upload,
        document_name="upload.pdf",
        task_instructions=instructions,
        gold_root=root,
        match_mode="strict_then_soft",
    )
    assert result.match_level == "strict"
    assert result.case_id == "case_a"
    assert result.gold_dir is not None


def test_ambiguous_strict_returns_none(tmp_path):
    instructions = "Same brief."
    pdf_bytes = b"%PDF twin"
    root = tmp_path / "gold_root"
    root.mkdir()
    for name in ("case_one", "case_two"):
        case = root / name
        case.mkdir()
        (case / "input.pdf").write_bytes(pdf_bytes)
        (case / "instructions.txt").write_text(instructions, encoding="utf-8")
        (case / "manifest.json").write_text(
            json.dumps({"case_id": name, "input_pdf": "input.pdf"}),
            encoding="utf-8",
        )
    upload = tmp_path / "upload.pdf"
    upload.write_bytes(pdf_bytes)
    result = resolve_gold_case_for_task(
        pdf_path=upload,
        document_name="upload.pdf",
        task_instructions=instructions,
        gold_root=root,
        match_mode="strict_only",
    )
    assert result.gold_dir is None
    assert "ambiguous" in result.reason


def test_soft_basename_and_line_subset(tmp_path):
    gold_pdf = tmp_path / "canonical.pdf"
    gold_pdf.write_bytes(b"%PDF soft match")
    instructions_gold = "Line A\nLine B"
    case = tmp_path / "case_soft"
    case.mkdir()
    (case / "canonical.pdf").write_bytes(gold_pdf.read_bytes())
    (case / "instructions.txt").write_text(instructions_gold, encoding="utf-8")
    (case / "manifest.json").write_text(
        json.dumps({"case_id": "case_soft", "input_pdf": "canonical.pdf"}),
        encoding="utf-8",
    )
    root = tmp_path / "gold_root"
    root.mkdir()
    shutil.copytree(case, root / "case_soft")

    upload = tmp_path / "renamed_upload.pdf"
    upload.write_bytes(b"%PDF different bytes")
    user_text = "Extra preamble\nLine A\nLine B\nFooter"
    result = resolve_gold_case_for_task(
        pdf_path=upload,
        document_name="canonical.pdf",
        task_instructions=user_text,
        gold_root=root,
        match_mode="soft_only",
    )
    assert result.match_level == "soft"
    assert result.case_id == "case_soft"


def test_manifest_hashes_strict(tmp_path):
    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(b"hash-me")
    instructions = "Brief"
    case = tmp_path / "hashed_case"
    case.mkdir()
    (case / "input.pdf").write_bytes(pdf.read_bytes())
    (case / "instructions.txt").write_text(instructions, encoding="utf-8")
    inst_hash = (
        __import__("hashlib")
        .sha256(normalize_instructions_text(instructions).encode("utf-8"))
        .hexdigest()
    )
    (case / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": "hashed_case",
                "input_pdf": "input.pdf",
                "input_pdf_sha256": sha256_file(pdf),
                "instructions_sha256": inst_hash,
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "gold_root"
    root.mkdir()
    shutil.copytree(case, root / "hashed_case")

    upload = tmp_path / "upload.pdf"
    upload.write_bytes(pdf.read_bytes())
    result = resolve_gold_case_for_task(
        pdf_path=upload,
        document_name="upload.pdf",
        task_instructions=instructions,
        gold_root=root,
        match_mode="strict_only",
    )
    assert result.match_level == "strict"


def test_app_eval_entity_synonyms_equivalent():
    gold = "- Redact any instance of PERSON, NAME, PHONE, PHONE_NUMBER"
    user = "- Redact any instance of PERSON, PHONE_NUMBER"
    assert app_eval_entity_instructions_equivalent(gold, user)


def test_app_eval_entity_synonyms_strict_match_with_pinned_hash(tmp_path):
    pdf = tmp_path / "input.pdf"
    pdf.write_bytes(b"entity-alias-pdf")
    gold_inst = "- Redact any instance of PERSON, NAME, PHONE, PHONE_NUMBER"
    user_inst = "- Redact any instance of PERSON, PHONE_NUMBER"
    case = tmp_path / "alias_case"
    case.mkdir()
    (case / "input.pdf").write_bytes(pdf.read_bytes())
    (case / "instructions.txt").write_text(gold_inst, encoding="utf-8")
    wrong_hash = "0" * 64
    (case / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": "alias_case",
                "input_pdf": "input.pdf",
                "input_pdf_sha256": sha256_file(pdf),
                "instructions_sha256": wrong_hash,
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "gold_root"
    root.mkdir()
    shutil.copytree(case, root / "alias_case")
    upload = tmp_path / "upload.pdf"
    upload.write_bytes(pdf.read_bytes())
    result = resolve_gold_case_for_task(
        pdf_path=upload,
        document_name="upload.pdf",
        task_instructions=user_inst,
        gold_root=root,
        match_mode="strict_only",
    )
    assert result.match_level == "strict"
    assert result.case_id == "alias_case"


def test_resolve_eval_gold_dir_auto_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_EVAL_AUTO_GOLD", "1")
    monkeypatch.delenv("AGENT_EVAL_GOLD", raising=False)
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    assert resolve_eval_gold_dir("no-task-session") is None


def test_resolve_eval_gold_dir_auto_uses_session_task(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_EVAL_AUTO_GOLD", "1")
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", str(tmp_path))
    gold_dir = _EVAL / "gold" / "example_smoke"
    session = "abc123"
    ws = tmp_path / session
    ws.mkdir()
    from session_workspace import write_active_redaction_task_file

    write_active_redaction_task_file(
        session,
        {
            "document_name": "doc.pdf",
            "task_instructions": "smoke",
            "follow_up_steer_count": 0,
            "cumulative_tool_calls": [],
            "eval_gold_path": str(gold_dir.resolve()),
            "eval_gold_match_level": "strict",
            "eval_gold_match_reason": "test",
        },
    )
    resolved = resolve_eval_gold_dir(session)
    assert resolved == gold_dir.resolve()
