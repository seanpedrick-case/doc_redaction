"""Tests for post_run_app_eval staging (gates-only smoke)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from post_run_app_eval import (  # noqa: E402
    resolve_app_eval_gold_dir,
    run_app_deterministic_gates,
)


def test_resolve_gold_none_when_auto_off(monkeypatch, tmp_path):
    monkeypatch.setattr("post_run_app_eval.app_eval_auto_gold_enabled", lambda: False)
    monkeypatch.setattr(
        "post_run_app_eval.resolve_fixed_app_eval_gold", lambda _r: None
    )
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    gold, msg = resolve_app_eval_gold_dir(
        pdf_path=pdf,
        document_name="doc.pdf",
        derived_instructions="- Redact any instance of PERSON",
    )
    assert gold is None
    assert "APP_EVAL_AUTO_GOLD" in msg


def test_run_app_gates_ocr_only_on_example_smoke(monkeypatch, tmp_path):
    smoke = _REPO / "agent-redact" / "eval" / "gold" / "example_smoke"
    if not smoke.is_dir():
        return
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ocr_src = smoke / "ocr_results_with_words.csv"
    if ocr_src.is_file():
        import shutil

        shutil.copy2(ocr_src, run_dir / "ocr_results_with_words.csv")
    summary = run_app_deterministic_gates(
        run_dir,
        smoke,
        text_extraction_only=True,
    )
    assert summary.ocr_gold is not None
    assert (run_dir / "ocr_gold.json").is_file()
