"""
Typed containers for redactor settings vs session artifacts.

`RedactionOptions` holds user-facing / configuration inputs.
`RedactionContext` holds mutable session state and intermediate file artifacts.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from tools.config import (
    DEFAULT_LANGUAGE,
    DEFAULT_LOCAL_OCR_MODEL,
    EFFICIENT_OCR,
    EFFICIENT_OCR_MIN_WORDS,
    HYBRID_TEXTRACT_BEDROCK_VLM,
    INPUT_FOLDER,
    OUTPUT_FOLDER,
    OVERWRITE_EXISTING_OCR_RESULTS,
    RETURN_PDF_FOR_REVIEW,
    RETURN_REDACTED_PDF,
    SAVE_PAGE_OCR_VISUALISATIONS,
)


def _merge_dataclass(cls, overrides: Dict[str, Any]):
    """Build an instance of `cls`, applying `overrides` on top of default field values."""
    base_kw: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name in overrides:
            base_kw[f.name] = overrides[f.name]
            continue
        if f.default_factory is not MISSING:
            base_kw[f.name] = f.default_factory()  # type: ignore[misc]
        elif f.default is not MISSING:
            base_kw[f.name] = f.default
    return cls(**base_kw)  # type: ignore[arg-type]


@dataclass(frozen=True)
class RedactionOptions:
    """Configuration for a redaction run (no file-path session lists beyond policy)."""

    chosen_redact_entities: Optional[List[str]] = None
    chosen_redact_comprehend_entities: Optional[List[str]] = None
    chosen_llm_entities: Optional[List[str]] = None
    text_extraction_method: Optional[str] = None
    in_allow_list: List[str] = field(default_factory=list)
    in_deny_list: List[str] = field(default_factory=list)
    redact_whole_page_list: List[str] = field(default_factory=list)
    page_min: int = 0
    page_max: int = 0
    handwrite_signature_checkbox: List[str] = field(
        default_factory=lambda: ["Extract handwriting"]
    )
    pii_identification_method: str = "Local"
    max_fuzzy_spelling_mistakes_num: int = 1
    match_fuzzy_whole_phrase_bool: bool = True
    aws_access_key_textbox: str = ""
    aws_secret_key_textbox: str = ""
    annotate_max_pages: int = 1
    output_folder: str = OUTPUT_FOLDER
    input_folder: str = INPUT_FOLDER
    textract_output_found: bool = False
    text_extraction_only: bool = False
    chosen_local_ocr_model: str = DEFAULT_LOCAL_OCR_MODEL
    language: str = DEFAULT_LANGUAGE
    custom_llm_instructions: str = ""
    inference_server_vlm_model: str = ""
    efficient_ocr: bool = EFFICIENT_OCR
    efficient_ocr_min_words: Union[int, float, None] = EFFICIENT_OCR_MIN_WORDS
    efficient_ocr_min_image_coverage_fraction: Optional[float] = None
    efficient_ocr_min_embedded_image_px: Optional[int] = None
    hybrid_textract_bedrock_vlm: bool = HYBRID_TEXTRACT_BEDROCK_VLM
    overwrite_existing_ocr_results: bool = OVERWRITE_EXISTING_OCR_RESULTS
    save_page_ocr_visualisations: bool = SAVE_PAGE_OCR_VISUALISATIONS
    ocr_first_pass_max_workers: Optional[int] = None
    prepare_images: bool = True
    RETURN_REDACTED_PDF: bool = RETURN_REDACTED_PDF
    RETURN_PDF_FOR_REVIEW: bool = RETURN_PDF_FOR_REVIEW


@dataclass
class RedactionContext:
    """Session / intermediate state for a redaction run."""

    prepared_pdf_file_paths: Optional[List[str]] = None
    pdf_image_file_paths: Optional[List[str]] = None
    latest_file_completed: int = 0
    combined_out_message: List = field(default_factory=list)
    out_file_paths: List = field(default_factory=list)
    log_files_output_paths: List = field(default_factory=list)
    estimated_time_taken_state: float = 0.0
    all_request_metadata_str: str = ""
    annotations_all_pages: List[dict] = field(default_factory=list)
    all_page_line_level_ocr_results_df: Optional[pd.DataFrame] = None
    all_pages_decision_process_table: Optional[pd.DataFrame] = None
    pymupdf_doc: Any = field(default_factory=list)
    review_file_state: Any = field(default_factory=list)
    document_cropboxes: List = field(default_factory=list)
    page_sizes: List[dict] = field(default_factory=list)
    duplication_file_path_outputs: list = field(default_factory=list)
    review_file_path: str = ""
    ocr_file_path: str = ""
    all_page_line_level_ocr_results: list[dict] = field(default_factory=list)
    all_page_line_level_ocr_results_with_words: list[dict] = field(default_factory=list)
    all_page_line_level_ocr_results_with_words_df: Optional[pd.DataFrame] = None
    ocr_review_files: list = field(default_factory=list)


def to_legacy_kwargs(
    options: RedactionOptions, context: RedactionContext
) -> Dict[str, Any]:
    """Merge options + context into the flat kwargs dict expected by `_choose_and_run_redactor_impl`."""
    return {**asdict(options), **asdict(context)}


def from_legacy_dict(flat: Dict[str, Any]) -> tuple[RedactionOptions, RedactionContext]:
    """
    Split a flat parameter dict (same keys as the legacy redactor) into Options + Context.
    Ignores ``file_paths`` and ``progress`` if present.
    """
    d = {k: v for k, v in flat.items() if k not in ("file_paths", "progress")}
    opt_keys = {f.name for f in fields(RedactionOptions)}
    ctx_keys = {f.name for f in fields(RedactionContext)}
    opts_d = {k: d[k] for k in d if k in opt_keys}
    ctx_d = {k: d[k] for k in d if k in ctx_keys}
    return _merge_dataclass(RedactionOptions, opts_d), _merge_dataclass(
        RedactionContext, ctx_d
    )
