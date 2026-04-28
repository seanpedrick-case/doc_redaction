"""
Stable programmatic API surface matching Gradio `api_name` values.

This module provides names that exactly match the Gradio endpoint `api_name`
strings from `app.py`.

By default these names point to the **CLI-first** Python API (`doc_redaction.cli_api`),
which is the most stable and runnable interface outside Gradio session state.
"""

from __future__ import annotations

from doc_redaction.cli_api import (
    apply_review_redactions,
    combine_review_csvs,
    combine_review_pdfs,
    export_review_page_ocr_visualisation,
    export_review_redaction_overlay,
    find_duplicate_pages,
    find_duplicate_tabular,
    load_and_prepare_documents_or_data,
    redact_data,
    redact_document,
    summarise_document,
    word_level_ocr_text_search,
)

__all__ = [
    "redact_document",
    "load_and_prepare_documents_or_data",
    "apply_review_redactions",
    "export_review_page_ocr_visualisation",
    "export_review_redaction_overlay",
    "word_level_ocr_text_search",
    "redact_data",
    "find_duplicate_pages",
    "find_duplicate_tabular",
    "summarise_document",
    "combine_review_csvs",
    "combine_review_pdfs",
]
