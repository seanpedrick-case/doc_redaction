"""
CLI-first programmatic API surface.

These functions provide a minimal, runnable Python interface that mirrors the
Gradio `api_name` routes, but executes the underlying workflows via the CLI
engine (`cli_redact.main(direct_mode_args=...)`).

Return values are lists of output file paths created in `output_dir`.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def _ensure_list(v: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def _snapshot_files(folder: str) -> set[str]:
    root = Path(folder)
    if not root.exists():
        return set()
    out: set[str] = set()
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            out.add(str(Path(dirpath) / name))
    return out


def _default_output_dir(prefix: str) -> str:
    return tempfile.mkdtemp(prefix=f"doc_redaction_{prefix}_")


def _run_cli(
    *,
    gradio_api_name: str,
    overrides: dict[str, Any],
    output_dir: str | None,
) -> list[str]:
    """
    Run cli_redact.main with merged defaults and return newly created files.
    """
    from cli_redact import get_cli_default_args_dict
    from cli_redact import main as cli_main

    merged = get_cli_default_args_dict()
    merged.update(overrides)

    if output_dir is None:
        output_dir = _default_output_dir(gradio_api_name)
    merged["output_dir"] = str(output_dir)

    before = _snapshot_files(str(output_dir))
    cli_main(direct_mode_args=merged)
    after = _snapshot_files(str(output_dir))

    created = sorted(after - before)
    return created


# ---------------------------------------------------------------------------
# Implemented via CLI engine (matches agent_routes.py)
# ---------------------------------------------------------------------------


def redact_document(
    input_files: str | list[str],
    *,
    output_dir: str | None = None,
    ocr_method: str | None = None,
    pii_detector: str | None = None,
    instruction: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    """
    Parity with Gradio `api_name='redact_document'`.
    Runs CLI task `redact` (PDF/PNG/JPG) or relevant workflow based on file type.
    """
    direct: dict[str, Any] = {
        "task": "redact",
        "input_file": _ensure_list(input_files),
    }
    if ocr_method is not None:
        direct["ocr_method"] = ocr_method
    if pii_detector is not None:
        direct["pii_detector"] = pii_detector
    if instruction is not None:
        direct["custom_llm_instructions"] = instruction
    if overrides:
        direct.update(overrides)
    return _run_cli(
        gradio_api_name="redact_document", overrides=direct, output_dir=output_dir
    )


def redact_data(
    input_files: str | list[str],
    *,
    output_dir: str | None = None,
    instruction: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    """Parity with Gradio `api_name='redact_data'` (same CLI task: `redact`)."""
    direct: dict[str, Any] = {"task": "redact", "input_file": _ensure_list(input_files)}
    if instruction is not None:
        direct["custom_llm_instructions"] = instruction
    if overrides:
        direct.update(overrides)
    return _run_cli(
        gradio_api_name="redact_data", overrides=direct, output_dir=output_dir
    )


def find_duplicate_pages(
    input_files: str | list[str],
    *,
    output_dir: str | None = None,
    similarity_threshold: float | None = None,
    min_word_count: int | None = None,
    min_consecutive_pages: int | None = None,
    greedy_match: bool | None = None,
    combine_pages: bool | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    """Parity with Gradio `api_name='find_duplicate_pages'`."""
    direct: dict[str, Any] = {
        "task": "deduplicate",
        "duplicate_type": "pages",
        "input_file": _ensure_list(input_files),
    }
    if similarity_threshold is not None:
        direct["similarity_threshold"] = similarity_threshold
    if min_word_count is not None:
        direct["min_word_count"] = min_word_count
    if min_consecutive_pages is not None:
        direct["min_consecutive_pages"] = min_consecutive_pages
    if greedy_match is not None:
        direct["greedy_match"] = "True" if greedy_match else "False"
    if combine_pages is not None:
        direct["combine_pages"] = "True" if combine_pages else "False"
    if overrides:
        direct.update(overrides)
    return _run_cli(
        gradio_api_name="find_duplicate_pages", overrides=direct, output_dir=output_dir
    )


def find_duplicate_tabular(
    input_files: str | list[str],
    *,
    output_dir: str | None = None,
    text_columns: list[str] | None = None,
    similarity_threshold: float | None = None,
    min_word_count: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    """Parity with Gradio `api_name='find_duplicate_tabular'`."""
    direct: dict[str, Any] = {
        "task": "deduplicate",
        "duplicate_type": "tabular",
        "input_file": _ensure_list(input_files),
    }
    if text_columns is not None:
        direct["text_columns"] = list(text_columns)
    if similarity_threshold is not None:
        direct["similarity_threshold"] = similarity_threshold
    if min_word_count is not None:
        direct["min_word_count"] = min_word_count
    if overrides:
        direct.update(overrides)
    return _run_cli(
        gradio_api_name="find_duplicate_tabular",
        overrides=direct,
        output_dir=output_dir,
    )


def summarise_document(
    input_files: str | list[str],
    *,
    output_dir: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    """Parity with Gradio `api_name='summarise_document'` (CLI task: `summarise`)."""
    direct: dict[str, Any] = {
        "task": "summarise",
        "input_file": _ensure_list(input_files),
    }
    if overrides:
        direct.update(overrides)
    return _run_cli(
        gradio_api_name="summarise_document", overrides=direct, output_dir=output_dir
    )


def combine_review_pdfs(
    input_files: str | list[str],
    *,
    output_dir: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    """Parity with Gradio `api_name='combine_review_pdfs'` (CLI task: `combine_review_pdfs`)."""
    direct: dict[str, Any] = {
        "task": "combine_review_pdfs",
        "input_file": _ensure_list(input_files),
    }
    if overrides:
        direct.update(overrides)
    return _run_cli(
        gradio_api_name="combine_review_pdfs", overrides=direct, output_dir=output_dir
    )


# ---------------------------------------------------------------------------
# Implemented without CLI (as per agent_routes.py)
# ---------------------------------------------------------------------------


def combine_review_csvs(
    input_files: Iterable[str],
    *,
    output_dir: str | None = None,
) -> list[str]:
    """Parity with Gradio `api_name='combine_review_csvs'`."""
    from tools.config import OUTPUT_FOLDER
    from tools.helper_functions import merge_csv_files

    out_dir = str(output_dir or OUTPUT_FOLDER)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sep = "/" if not out_dir.endswith(("/", "\\")) else ""

    return merge_csv_files([str(p) for p in input_files], output_folder=out_dir + sep)


def export_review_redaction_overlay(
    *,
    page_image_path: str,
    boxes: list[dict[str, Any]],
    page_number: int = 1,
    doc_base_name: str = "review",
    review_df_records: list[dict[str, Any]] | None = None,
    label_abbrev_chars: int | None = None,
) -> list[str]:
    """Same behaviour as Gradio ``api_name='page_redaction_review_image'``; Agent API route ``export_review_redaction_overlay``."""
    import pandas as pd

    from tools.config import OUTPUT_FOLDER
    from tools.redaction_review import visualise_review_redaction_boxes

    annotator: dict[str, Any] = {"image": page_image_path, "boxes": boxes}
    review_df = pd.DataFrame(review_df_records) if review_df_records else pd.DataFrame()

    out_dir = str(Path(OUTPUT_FOLDER).expanduser().resolve())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = visualise_review_redaction_boxes(
        annotator,
        review_df=review_df,
        output_folder=out_dir,
        page_number=page_number,
        doc_base_name=doc_base_name,
        label_abbrev_chars=label_abbrev_chars,
    )
    return [out_path] if out_path else []


def export_review_page_ocr_visualisation(
    *,
    page_image_path: str,
    ocr_results: dict[str, Any],
    page_number: int = 1,
    doc_base_name: str = "review",
) -> list[str]:
    """Same behaviour as Gradio ``api_name='page_ocr_review_image'``; Agent API route ``export_review_page_ocr_visualisation``."""
    from PIL import Image

    from tools.config import OUTPUT_FOLDER
    from tools.file_redaction import visualise_ocr_words_bounding_boxes

    out_dir = str(Path(OUTPUT_FOLDER).expanduser().resolve())
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    image_name = f"{str(doc_base_name or 'review')}_page{int(page_number)}.png"
    log_paths: list[str] = []
    log_paths = visualise_ocr_words_bounding_boxes(
        Image.open(page_image_path).convert("RGB"),
        ocr_results,
        image_name=image_name,
        output_folder=out_dir,
        visualisation_folder="review_ocr_visualisations",
        add_legend=True,
        log_files_output_paths=log_paths,
    )
    return list(log_paths)


# ---------------------------------------------------------------------------
# Gradio-session-only (no single CLI task)
# ---------------------------------------------------------------------------


def load_and_prepare_documents_or_data(*args: Any, **kwargs: Any) -> list[str]:
    raise NotImplementedError(
        "load_and_prepare_documents_or_data is Gradio-session-state driven and is not exposed as a single CLI task."
    )


def apply_review_redactions(
    pdf_path: str,
    review_csv_path: str,
    *,
    output_dir: str | None = None,
    input_dir: str | None = None,
    text_extract_method: str | None = None,
    efficient_ocr: bool | None = None,
) -> list[str]:
    """
    Headless parity with Gradio ``api_name='apply_review_redactions'``.

    Returns output file paths (redacted PDF, review CSV, logs, etc.).
    """
    from tools.simplified_api import run_apply_review_redactions

    r = run_apply_review_redactions(
        pdf_path=pdf_path,
        review_csv_path=review_csv_path,
        output_dir=output_dir,
        input_dir=input_dir,
        text_extract_method=text_extract_method,
        efficient_ocr=efficient_ocr,
    )
    return list(r.get("output_paths") or [])


def word_level_ocr_text_search(*args: Any, **kwargs: Any) -> list[str]:
    raise NotImplementedError(
        "word_level_ocr_text_search is Gradio-session-state driven; no CLI-first equivalent is currently provided."
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
