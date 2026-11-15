"""Wire AgentCore harness deliverables into Review / Duplicate / Summarise tabs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tools.file_conversion import get_document_file_names, read_file


@dataclass(frozen=True)
class AgentOutputArtifacts:
    review_csv: str | None
    ocr_lines_csv: str | None
    ocr_words_csv: str | None
    review_overlay_pdf: str | None
    source_pdf: str | None


def _existing(paths: list[str]) -> list[str]:
    return [p for p in paths if isinstance(p, str) and p and os.path.isfile(p)]


def _first_uploaded_path(files: Any) -> str | None:
    if not files:
        return None
    if isinstance(files, str):
        return files if os.path.isfile(files) else None
    items = files if isinstance(files, list) else [files]
    for item in items:
        path = getattr(item, "name", None) or str(item)
        if path and os.path.isfile(path):
            return path
    return None


def _newest(paths: list[str], *, contains: str, suffix: str = "") -> str | None:
    matches: list[tuple[float, str]] = []
    needle = contains.lower()
    for path in _existing(paths):
        name = Path(path).name.lower()
        if needle not in name:
            continue
        if suffix and not name.endswith(suffix.lower()):
            continue
        matches.append((Path(path).stat().st_mtime, path))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def classify_agent_output_paths(
    output_paths: list[str] | None,
    *,
    source_hint: str | None = None,
) -> AgentOutputArtifacts:
    paths = _existing(list(output_paths or []))
    review_csv = _newest(paths, contains="_review_file", suffix=".csv")
    ocr_lines = _newest(paths, contains="_ocr_output_", suffix=".csv")
    if ocr_lines and "ocr_results_with_words" in Path(ocr_lines).name.lower():
        ocr_lines = None
    ocr_words = _newest(paths, contains="ocr_results_with_words", suffix=".csv")
    review_overlay = _newest(paths, contains="redactions_for_review", suffix=".pdf")
    source = source_hint if source_hint and os.path.isfile(source_hint) else None
    if source is None:
        for path in paths:
            low = path.lower()
            if low.endswith((".pdf", ".png", ".jpg", ".jpeg")) and (
                "redactions_for_review" not in low
            ):
                if low.endswith("_redacted.pdf"):
                    continue
                source = path
                break
    return AgentOutputArtifacts(
        review_csv=review_csv,
        ocr_lines_csv=ocr_lines,
        ocr_words_csv=ocr_words,
        review_overlay_pdf=review_overlay,
        source_pdf=source,
    )


def _noop_tab_outputs(
    summary: str,
    review_file_df: Any,
    pdf_doc_state: Any,
    all_image_annotations_state: Any,
    page_sizes: Any,
    annotate_max_pages: int,
    all_page_line_level_ocr_results_df_base: Any,
    all_page_line_level_ocr_results_with_words_df_base: Any,
):
    """Return outputs unchanged when harness files cannot be wired to Review."""
    msg = summary or ""
    note = (
        "\n\n_(Review tab: no *_review_file.csv in agent outputs — "
        "upload the review CSV on the Review tab manually if needed.)_"
    )
    if note.strip() not in msg:
        msg = (msg + note).strip()
    dup_paths: list[str] = []
    return (
        msg,
        [],
        [],
        annotate_max_pages,
        annotate_max_pages,
        pdf_doc_state,
        all_image_annotations_state,
        review_file_df,
        [],
        page_sizes,
        [],
        None,
        None,
        [],
        dup_paths,
        dup_paths,
        dup_paths,
        [],
        [],
        all_page_line_level_ocr_results_with_words_df_base,
        all_page_line_level_ocr_results_df_base,
        annotate_max_pages,
    )


def load_agentic_harness_outputs_for_gradio_tabs(
    output_file_list_state,
    in_doc_files,
    text_extract_method_radio,
    all_page_line_level_ocr_results_df_base,
    all_page_line_level_ocr_results_with_words_df_base,
    latest_file_completed_num,
    redaction_output_summary_textbox,
    annotate_max_pages,
    all_image_annotations_state,
    prepare_for_review_bool,
    in_fully_redacted_list_state,
    output_folder_textbox,
    input_folder_textbox,
    efficient_ocr_checkbox,
    prepare_images_bool_false,
    page_sizes,
    pdf_doc_state,
    page_min,
    page_max,
    review_file_df,
):
    """
    After an agent run, load review CSV + source PDF into session state (Review tab)
    and pass OCR CSV paths to Duplicate / Summarise inputs.
    """
    paths = _existing(
        output_file_list_state
        if isinstance(output_file_list_state, list)
        else ([output_file_list_state] if output_file_list_state else [])
    )
    source_hint = _first_uploaded_path(in_doc_files)
    artifacts = classify_agent_output_paths(paths, source_hint=source_hint)

    if not artifacts.source_pdf or not artifacts.review_csv:
        return _noop_tab_outputs(
            redaction_output_summary_textbox,
            review_file_df,
            pdf_doc_state,
            all_image_annotations_state,
            page_sizes,
            int(annotate_max_pages or 1),
            all_page_line_level_ocr_results_df_base,
            all_page_line_level_ocr_results_with_words_df_base,
        )

    from tools.file_conversion import prepare_image_or_pdf_with_efficient_ocr
    from tools.redaction_review import (
        apply_redactions_to_review_df_and_files_for_review_ui,
    )

    prep_paths = [artifacts.source_pdf, artifacts.review_csv]
    prep = prepare_image_or_pdf_with_efficient_ocr(
        prep_paths,
        text_extract_method_radio,
        all_page_line_level_ocr_results_df_base,
        all_page_line_level_ocr_results_with_words_df_base,
        latest_file_completed_num,
        redaction_output_summary_textbox,
        True,
        annotate_max_pages,
        all_image_annotations_state,
        bool(prepare_for_review_bool),
        in_fully_redacted_list_state,
        output_folder_textbox,
        input_folder_textbox,
        efficient_ocr_checkbox,
        prepare_images_bool_false,
        page_sizes,
        pdf_doc_state,
        page_min,
        page_max,
    )
    (
        prep_msg,
        prepared_pdf_state,
        images_pdf_state,
        annotate_max_pages_out,
        annotate_max_pages_bottom,
        pdf_doc_state_out,
        all_image_annotations_state_out,
        review_file_df_out,
        document_cropboxes,
        page_sizes_out,
        _textract_found,
        _all_img_details,
        line_df_from_prep,
        _words_found_flag,
        words_df_from_prep,
    ) = prep

    names = get_document_file_names(prep_paths)
    doc_full_file_name = names[2]
    total_pages = names[4]

    (
        pdf_doc_state_out,
        all_image_annotations_state_out,
        input_pdf_for_review,
        _log_files,
        review_file_df_out,
    ) = apply_redactions_to_review_df_and_files_for_review_ui(
        doc_full_file_name,
        pdf_doc_state_out,
        all_image_annotations_state_out,
        1,
        review_file_df_out,
        output_folder_textbox,
        True,
        page_sizes_out,
        input_folder_textbox,
    )

    line_df = line_df_from_prep
    words_df = words_df_from_prep
    if artifacts.ocr_lines_csv and os.path.isfile(artifacts.ocr_lines_csv):
        try:
            line_df = read_file(artifacts.ocr_lines_csv)
        except Exception:
            pass
    if artifacts.ocr_words_csv and os.path.isfile(artifacts.ocr_words_csv):
        try:
            words_df = read_file(artifacts.ocr_words_csv)
        except Exception:
            pass

    ocr_review_files = _existing(
        [artifacts.ocr_lines_csv, artifacts.ocr_words_csv, artifacts.review_csv]
    )
    dup_paths = _existing([artifacts.ocr_lines_csv]) or ocr_review_files[:1]
    review_tab_files = _existing([artifacts.source_pdf, artifacts.review_csv])

    agent_summary = (redaction_output_summary_textbox or "").strip()
    review_note = (
        "Review tab:\n"
        "Agent outputs were loaded into the Review tab "
        "(OCR paths were sent to Duplicate / Summarise where available)."
    )
    if (prep_msg or "").strip() and not agent_summary:
        summary = f"{prep_msg.strip()}\n\n---\n\n{review_note}"
    elif agent_summary:
        summary = f"{agent_summary}\n\n---\n\n{review_note}"
    else:
        summary = review_note

    ocr_lines_path = artifacts.ocr_lines_csv or ""
    all_page_line_level_ocr_results = [ocr_lines_path] if ocr_lines_path else []
    all_page_line_level_ocr_results_with_words = (
        [artifacts.ocr_words_csv] if artifacts.ocr_words_csv else []
    )

    return (
        summary,
        prepared_pdf_state,
        images_pdf_state,
        annotate_max_pages_out,
        annotate_max_pages_bottom,
        pdf_doc_state_out,
        all_image_annotations_state_out,
        review_file_df_out,
        document_cropboxes,
        page_sizes_out,
        review_tab_files,
        artifacts.review_csv,
        ocr_lines_path or None,
        ocr_review_files,
        dup_paths,
        dup_paths,
        dup_paths,
        all_page_line_level_ocr_results,
        all_page_line_level_ocr_results_with_words,
        words_df if isinstance(words_df, pd.DataFrame) else words_df_from_prep,
        line_df if isinstance(line_df, pd.DataFrame) else line_df_from_prep,
        total_pages or annotate_max_pages_out,
    )
