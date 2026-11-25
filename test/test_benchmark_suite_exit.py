"""Benchmark suite exit codes: technical vs QA."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from run_benchmark_suite import (  # noqa: E402
    _accumulate_surface_outcome,
    _apply_qa_outcome,
    _env_default,
)


def test_apply_qa_outcome_poor_score_technical_zero():
    rec: dict = {"exit_code": 1}
    _apply_qa_outcome(
        rec, technical_exit=0, qa_pass=False, message="overall_pass=False"
    )
    assert rec["exit_code"] == 0
    assert rec["qa_pass"] is False
    assert rec["qa_completed"] is True


def test_accumulate_surface_qa_fail_does_not_raise_technical():
    rec = {"skipped": False, "exit_code": 0, "qa_completed": True, "qa_pass": False}
    max_t, any_qa = _accumulate_surface_outcome(rec, 0, False)
    assert max_t == 0
    assert any_qa is True


def test_env_default_prefers_pristine_env_then_env_file(monkeypatch) -> None:
    import run_benchmark_suite as rbs

    monkeypatch.setitem(rbs._PRISTINE_ENV, "BENCHMARK_OCR_METHOD", "from-env")
    monkeypatch.setattr(
        rbs,
        "_env_file_values",
        lambda: {"BENCHMARK_OCR_METHOD": "from-file", "OTHER_VAR": "file-only"},
    )
    assert (
        _env_default("BENCHMARK_OCR_METHOD", "DEFAULT_TEXT_EXTRACTION_MODEL")
        == "from-env"
    )
    assert _env_default("OTHER_VAR") == "file-only"
    monkeypatch.delitem(rbs._PRISTINE_ENV, "BENCHMARK_OCR_METHOD")
    assert _env_default("BENCHMARK_OCR_METHOD") == "from-file"
    assert _env_default("NO_SUCH_VAR_ANYWHERE") is None


def test_parser_stack_flags_fall_back_to_env(monkeypatch) -> None:
    import run_benchmark_suite as rbs

    monkeypatch.setitem(rbs._PRISTINE_ENV, "BENCHMARK_LOCAL_OCR_MODEL", "paddle")
    monkeypatch.setattr(
        rbs,
        "_env_file_values",
        lambda: {"DEFAULT_TEXT_EXTRACTION_MODEL": "Local model - selectable text"},
    )
    args = rbs._build_parser().parse_args([])
    assert args.chosen_local_ocr_model == "paddle"  # BENCHMARK_* env var
    assert args.ocr_method == "Local model - selectable text"  # app config env file
    # explicit flag beats env
    args2 = rbs._build_parser().parse_args(["--chosen-local-ocr-model", "tesseract"])
    assert args2.chosen_local_ocr_model == "tesseract"
