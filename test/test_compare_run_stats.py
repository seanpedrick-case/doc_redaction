"""Tests for compare_run_stats config grouping defaults."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from compare_run_stats import (  # noqa: E402
    DEFAULT_CONFIG_COLS,
    _config_label,
    build_task_config_table,
)


def test_default_config_cols_cover_stack_dimensions():
    assert "model_name" in DEFAULT_CONFIG_COLS
    assert "agent_orchestrator" in DEFAULT_CONFIG_COLS
    assert "chosen_ocr_model" in DEFAULT_CONFIG_COLS
    assert "pii_llm_model" in DEFAULT_CONFIG_COLS
    assert "pii_method" in DEFAULT_CONFIG_COLS
    assert "ocr_method" in DEFAULT_CONFIG_COLS
    assert "chosen_local_ocr_model" in DEFAULT_CONFIG_COLS


def test_config_label_missing_columns_read_as_dash():
    row = {
        "model_name": "amazon.nova-pro-v1:0",
        "ocr_method": "AWS Textract",
        # older CSV without the newer columns
    }
    label = _config_label(row, DEFAULT_CONFIG_COLS)
    assert label.startswith("amazon.nova-pro-v1:0 | ")
    assert "AWS Textract" in label
    assert "-" in label  # missing columns render as "-"


def test_build_task_config_table_groups_by_new_columns():
    rows = [
        {
            "document_name": "doc.pdf",
            "model_name": "m1",
            "agent_orchestrator": "pi",
            "ocr_method": "Local OCR",
            "chosen_local_ocr_model": "hybrid-paddle-vlm",
            "chosen_ocr_model": "vlm-model-a",
            "pii_method": "Local",
            "pii_llm_model": "",
            "gold_geometry_f1": "0.9",
        },
        {
            "document_name": "doc.pdf",
            "model_name": "m1",
            "agent_orchestrator": "pi",
            "ocr_method": "Local OCR",
            "chosen_local_ocr_model": "hybrid-paddle-inference-server",
            "chosen_ocr_model": "vlm-model-b",
            "pii_method": "Local",
            "pii_llm_model": "",
            "gold_geometry_f1": "0.8",
        },
        {
            # same OCR stack but different PII method -> separate config
            "document_name": "doc.pdf",
            "model_name": "m1",
            "agent_orchestrator": "pi",
            "ocr_method": "Local OCR",
            "chosen_local_ocr_model": "hybrid-paddle-vlm",
            "chosen_ocr_model": "vlm-model-a",
            "pii_method": "AWS Comprehend",
            "pii_llm_model": "",
            "gold_geometry_f1": "0.85",
        },
        {
            # same UI ocr_method label, different local engine -> separate config
            "document_name": "doc.pdf",
            "model_name": "m1",
            "agent_orchestrator": "pi",
            "ocr_method": "Local OCR",
            "chosen_local_ocr_model": "tesseract",
            "chosen_ocr_model": "",
            "pii_method": "Local",
            "pii_llm_model": "",
            "gold_geometry_f1": "0.7",
        },
    ]
    table = build_task_config_table(
        rows,
        task_col="document_name",
        config_cols=DEFAULT_CONFIG_COLS,
        metrics=("gold_geometry_f1",),
        binary_cols=(),
    )
    configs = table["doc.pdf"]
    assert (
        len(configs) == 4
    )  # chosen_ocr_model, pii_method, chosen_local_ocr_model each split
