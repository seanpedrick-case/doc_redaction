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
            chosen_local_ocr_model="hybrid-paddle-inference-server",
            chosen_ocr_model="qwen_3_5_27b",
            pii_identification_method="LLM (AWS Bedrock)",
            pii_llm_model="anthropic.claude-3-7-sonnet-20250219-v1:0",
        ),
    )
    assert "Local OCR" in out.text_extraction_method
    assert out.chosen_local_ocr_model == "hybrid-paddle-inference-server"
    assert out.chosen_ocr_model == "qwen_3_5_27b"
    assert out.pii_identification_method == "LLM (AWS Bedrock)"
    assert out.pii_llm_model == "anthropic.claude-3-7-sonnet-20250219-v1:0"


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


def test_overrides_from_argv_chosen_ocr_model() -> None:
    out = overrides_from_argv(
        ocr_method=None,
        pii_method=None,
        chosen_local_ocr_model="hybrid-paddle-vlm",
        app_config_path=None,
        chosen_ocr_model="gemma-3-27b",
    )
    assert out is not None
    assert out.chosen_ocr_model == "gemma-3-27b"
    assert out.chosen_local_ocr_model == "hybrid-paddle-vlm"


def test_apply_benchmark_overrides_local_model_switches_extraction_method() -> None:
    """Overriding only the local OCR engine must also select the local-OCR method."""
    base = AppEvalRedactionSettings(
        text_extraction_method="Local model - selectable text",
        pii_identification_method="Local",
    )
    out = apply_benchmark_overrides(
        base, BenchmarkStackOverrides(chosen_local_ocr_model="paddle")
    )
    assert out.chosen_local_ocr_model == "paddle"
    assert "local ocr" in out.text_extraction_method.casefold()


def test_apply_benchmark_overrides_explicit_ocr_method_wins() -> None:
    base = AppEvalRedactionSettings(
        text_extraction_method="Local model - selectable text",
        pii_identification_method="Local",
    )
    out = apply_benchmark_overrides(
        base,
        BenchmarkStackOverrides(
            text_extraction_method="AWS Textract",
            chosen_local_ocr_model="paddle",
        ),
    )
    assert out.text_extraction_method == "AWS Textract"
    assert out.chosen_local_ocr_model == "paddle"
