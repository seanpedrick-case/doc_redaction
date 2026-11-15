"""Tests for benchmark stack overrides."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from benchmark_stack import (  # noqa: E402
    BenchmarkStackOverrides,
    apply_benchmark_overrides,
    overrides_from_argv,
)
from derive_app_eval_instructions import AppEvalRedactionSettings  # noqa: E402


def test_apply_benchmark_overrides() -> None:
    base = AppEvalRedactionSettings(
        text_extraction_method="Local text",
        pii_identification_method="Local",
    )
    out = apply_benchmark_overrides(
        base,
        BenchmarkStackOverrides(
            text_extraction_method="Local OCR model - PDFs without selectable text",
            chosen_local_ocr_model="paddle",
            pii_identification_method="AWS Comprehend",
        ),
    )
    assert "Local OCR" in out.text_extraction_method
    assert out.chosen_local_ocr_model == "paddle"
    assert out.pii_identification_method == "AWS Comprehend"


def test_overrides_from_argv_empty() -> None:
    assert (
        overrides_from_argv(
            ocr_method=None,
            pii_method=None,
            chosen_local_ocr_model=None,
            app_config_path=None,
        )
        is None
    )
