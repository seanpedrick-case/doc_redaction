"""
Headless and short ``gr.api`` entrypoints for agents and Gradio clients.

Consolidates:
- Review apply (``run_apply_review_redactions``, short `review_apply`)
- PDF summarisation (short `pdf_summarise`)
- Tabular redaction (short `tabular_redact`)
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, Mapping

import pandas as pd

from tools.config import (
    AZURE_OPENAI_INFERENCE_ENDPOINT,
    DEFAULT_FUZZY_SPELLING_MISTAKES_NUM,
    DEFAULT_INFERENCE_SERVER_VLM_MODEL,
    EFFICIENT_OCR,
    EFFICIENT_OCR_MIN_EMBEDDED_IMAGE_PX,
    EFFICIENT_OCR_MIN_IMAGE_COVERAGE_FRACTION,
    EFFICIENT_OCR_MIN_WORDS,
    HYBRID_TEXTRACT_BEDROCK_VLM,
    INPUT_FOLDER,
    OCR_FIRST_PASS_MAX_WORKERS,
    OUTPUT_FOLDER,
    OVERWRITE_EXISTING_OCR_RESULTS,
    SAVE_PAGE_OCR_VISUALISATIONS,
)
from tools.data_anonymise import anonymise_files_with_open_text
from tools.file_conversion import (
    is_pdf,
    prepare_image_or_pdf,
    prepare_image_or_pdf_with_efficient_ocr,
)
from tools.file_redaction import run_redaction
from tools.helper_functions import get_file_name_without_type
from tools.redaction_review import apply_redactions_to_review_df_and_files
from tools.redaction_types import RedactionContext, RedactionOptions
from tools.summaries import (
    concise_summary_format_prompt,
    detailed_summary_format_prompt,
    summarise_document_wrapper,
)

# prepare_image_or_pdf return indices — see tools/file_conversion.py ~1967
_IX_MSG = 0
_IX_PYMUPDF_DOC = 5
_IX_ANNOTATIONS = 6
_IX_REVIEW_DF = 7
_IX_PAGE_SIZES = 9


class HeadlessGradioProgress:
    """Minimal Gradio Progress stand-in (callable + tqdm) for headless runs."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None

    def tqdm(self, iterable, desc: str | None = None, unit: str | None = None):
        return iterable


def _folder_with_trailing_sep(folder: str) -> str:
    folder = os.path.normpath(folder)
    sep = os.sep
    if not folder.endswith(("/", "\\")):
        return folder + sep
    return folder


def _resolve_dir_within_base(candidate_dir: str | None, base_dir: str) -> str:
    """
    Resolve candidate_dir (or base_dir when None) and enforce containment in base_dir.

    This is a defense-in-depth guard for agent-facing wrappers: it prevents a caller from
    writing outputs outside the configured base folders.
    """
    base_abs = os.path.normpath(os.path.abspath(os.path.expanduser(base_dir)))
    raw = candidate_dir if candidate_dir is not None else base_dir
    resolved = os.path.normpath(os.path.abspath(os.path.expanduser(str(raw))))
    try:
        common = os.path.commonpath([resolved, base_abs])
    except ValueError as exc:
        raise ValueError(f"Invalid directory path: {raw}") from exc
    if common != base_abs:
        raise ValueError(f"Directory must be within configured base folder: {base_abs}")
    return resolved


def _filter_files_within_root(paths: Iterable[Any], root_dir: str) -> list[str]:
    """
    Keep only existing files contained within root_dir, returning real paths.
    """
    safe_root = os.path.realpath(str(root_dir))
    seen: set[str] = set()
    kept: list[str] = []
    for p in paths:
        if not p:
            continue
        resolved = os.path.realpath(str(p))
        try:
            within = os.path.commonpath([safe_root, resolved]) == safe_root
        except ValueError:
            within = False
        if not within:
            continue
        if not os.path.isfile(resolved):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        kept.append(resolved)
    return kept


def _validate_review_csv_path(path: str) -> None:
    base = (get_file_name_without_type(path) or "").lower()
    if "_review_file" not in base:
        raise ValueError(
            "review_csv_path basename must contain '_review_file' (required by "
            "prepare_image_or_pdf CSV branch), e.g. 'mydoc_review_file.csv'."
        )


def run_apply_review_redactions(
    *,
    pdf_path: str,
    review_csv_path: str,
    output_dir: str | None = None,
    input_dir: str | None = None,
    text_extract_method: str | None = None,
    efficient_ocr: bool | None = None,
    merged_cli_defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run prepare (PDF then review CSV) and apply redactions; return output paths.

    Args:
        pdf_path: Absolute path to source PDF (under allowed roots).
        review_csv_path: Absolute path to *_review_file.csv.
        output_dir: Folder for outputs; trailing slash normalized. Defaults to OUTPUT_FOLDER.
        input_dir: Folder for page images / intermediates; defaults to INPUT_FOLDER.
        text_extract_method: Passed to prepare (e.g. CLI ocr_method). Defaults from merged_cli_defaults or fresh CLI dict.
        efficient_ocr: If None, uses tools.config.EFFICIENT_OCR.
        merged_cli_defaults: Optional pre-built dict from get_cli_default_args_dict() (avoids re-parsing CLI).

    Returns:
        dict with keys: output_paths, output_dir, input_dir, message, gradio_api_name
    """
    _validate_review_csv_path(review_csv_path)

    if merged_cli_defaults is None:
        from cli_redact import get_cli_default_args_dict

        cli = get_cli_default_args_dict()
    else:
        cli = dict(merged_cli_defaults)

    out_folder = _folder_with_trailing_sep(
        _resolve_dir_within_base(output_dir, OUTPUT_FOLDER)
    )
    in_folder = _folder_with_trailing_sep(
        _resolve_dir_within_base(input_dir, INPUT_FOLDER)
    )

    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(in_folder, exist_ok=True)

    textract_method = (
        text_extract_method
        if text_extract_method is not None
        else str(cli.get("ocr_method") or "Local text")
    )
    use_efficient = EFFICIENT_OCR if efficient_ocr is None else bool(efficient_ocr)

    prep_progress = HeadlessGradioProgress()
    file_paths = [pdf_path, review_csv_path]

    prep_tuple = prepare_image_or_pdf_with_efficient_ocr(
        file_paths,
        textract_method,
        pd.DataFrame(),
        pd.DataFrame(),
        0,
        [],
        True,
        0,
        [],
        True,
        [],
        out_folder,
        in_folder,
        use_efficient,
        False,
        [],
        [],
        0,
        0,
        prep_progress,
    )

    pymupdf_doc = prep_tuple[_IX_PYMUPDF_DOC]
    all_annotations = prep_tuple[_IX_ANNOTATIONS]
    review_df = prep_tuple[_IX_REVIEW_DF]
    page_sizes = prep_tuple[_IX_PAGE_SIZES]
    prep_msg = prep_tuple[_IX_MSG]

    if not isinstance(review_df, pd.DataFrame):
        review_df = pd.DataFrame()
    if not page_sizes:
        raise ValueError(
            "prepare_image_or_pdf produced empty page_sizes; check pdf_path and logs."
        )
    if not all_annotations:
        raise ValueError(
            "prepare_image_or_pdf produced no annotation objects; check pdf_path and prepare_for_review path."
        )

    current_page = 1
    if current_page < 1 or current_page > len(all_annotations):
        raise ValueError(
            f"Invalid annotation page list length {len(all_annotations)} for current_page={current_page}."
        )
    page_annotator = all_annotations[current_page - 1]

    apply_progress = HeadlessGradioProgress()
    try:
        _doc_out, _ann_out, output_files, output_log_files, _review_out = (
            apply_redactions_to_review_df_and_files(
                page_annotator,
                [pdf_path],
                pymupdf_doc,
                all_annotations,
                current_page,
                review_df,
                out_folder,
                True,
                page_sizes,
                in_folder,
                progress=apply_progress,
            )
        )
    finally:
        if pymupdf_doc is not None and hasattr(pymupdf_doc, "is_closed"):
            try:
                if not pymupdf_doc.is_closed:
                    pymupdf_doc.close()
            except Exception:
                pass

    out_paths: list[str] = []
    for item in (output_files, output_log_files):
        if not item:
            continue
        if isinstance(item, str):
            out_paths.append(item)
        else:
            out_paths.extend(str(p) for p in item if p)

    safe_output_root = os.path.realpath(out_folder)
    seen: set[str] = set()
    unique_paths: list[str] = []
    for p in out_paths:
        if not p:
            continue
        resolved = os.path.realpath(str(p))
        try:
            within_output_root = (
                os.path.commonpath([safe_output_root, resolved]) == safe_output_root
            )
        except ValueError:
            within_output_root = False
        if not within_output_root:
            continue
        if not os.path.isfile(resolved):
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)

    return {
        "output_paths": unique_paths,
        "output_dir": out_folder.rstrip(os.sep),
        "input_dir": in_folder.rstrip(os.sep),
        "message": (str(prep_msg) if prep_msg else "apply_review_redactions completed"),
        "gradio_api_name": "apply_review_redactions",
    }


def normalize_gradio_file_to_path(value: Any) -> str:
    """
    Turn Gradio file payloads from the HTTP/client API into a local path string.

    Accepts a bare path string, a FileData-like dict (path / name), or an object
    with ``path`` or ``name``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("path", "name", "file_path"):
            v = value.get(key)
            if v:
                return str(v).strip()
        return ""
    path_attr = getattr(value, "path", None) or getattr(value, "name", None)
    return str(path_attr).strip() if path_attr else ""


def apply_review_redactions_from_uploads_for_gradio_api(
    pdf_file: Any,
    review_csv_file: Any,
    output_dir: str | None = None,
) -> tuple[list[str], str]:
    """
    Args:
        pdf_file (Any): The original PDF file. May be a path string or a Gradio upload payload (dict/object with "path" or "name").
        review_csv_file (Any): The review CSV file (a *_review_file.csv plan). May be a path string or a Gradio upload payload (dict/object with "path" or "name").
        output_dir (str, optional): Directory to write redacted PDFs, CSV, and logs. If omitted or blank, defaults to configuration OUTPUT_FOLDER.

    Returns:
        tuple[list[str], str]:
            A tuple containing:
                - output_paths (list[str]): Paths to generated artifacts.
                - message (str): Short status string.

    Gradio ``gr.api`` handler for a short programmatic apply. Prefer calling it via the
    short route `api_name='/review_apply'`.
    This path does not update the interactive Review tab session.
    """
    pdf_path = normalize_gradio_file_to_path(pdf_file)
    csv_path = normalize_gradio_file_to_path(review_csv_file)
    if not pdf_path:
        raise ValueError(
            "pdf_file is missing or could not be resolved to a path (upload the PDF first)."
        )
    if not csv_path:
        raise ValueError(
            "review_csv_file is missing or could not be resolved to a path (upload the CSV first)."
        )
    out_dir: str | None = output_dir
    if isinstance(out_dir, str) and not out_dir.strip():
        out_dir = None
    result = run_apply_review_redactions(
        pdf_path=pdf_path,
        review_csv_path=csv_path,
        output_dir=out_dir,
    )
    paths = list(result.get("output_paths") or [])
    msg = str(result.get("message") or "ok")
    return paths, msg


def redact_data_from_upload_for_gradio_api(
    data_file: Any,
    redact_entities: list[str] | None = None,
    output_dir: str | None = None,
    pii_method: str | None = "Local",
    columns: list[str] | None = None,
    anon_strategy: str | None = "redact",
    allow_list: list[str] | None = None,
    deny_list: list[str] | None = None,
    language: str | None = "en",
    max_fuzzy_spelling_mistakes_num: int | None = 0,
    do_initial_clean: bool | None = True,
    llm_instruction: str | None = "",
    llm_entities: list[str] | None = None,
    comprehend_entities: list[str] | None = None,
    aws_access_key: str | None = "",
    aws_secret_key: str | None = "",
) -> tuple[list[str], str]:
    """
    Short, stateless ``gr.api`` wrapper for the tabular redaction workflow.

    Args:
        data_file: CSV/XLSX/Parquet/DOCX file. Accepts a path string, a Gradio upload
            payload (dict/object with ``path``/``name``), or other FileData-like values.
        redact_entities: Presidio-style entity labels (e.g. PERSON, PHONE_NUMBER).
        output_dir: Directory to write redacted files and logs. Defaults to OUTPUT_FOLDER.
        pii_method: One of the tabular PII methods (commonly ``Local`` or ``AWS Comprehend``;
            LLM-backed methods depend on deployment config).
        columns: Column names to process (empty/None typically means “auto / all text-like columns”).
        anon_strategy: Tabular anonymisation strategy (defaults to ``redact``).
        allow_list / deny_list: Whitelist/blacklist terms.
        language: Language code (default ``en``).
        max_fuzzy_spelling_mistakes_num: 0–9; defaults to 0.
        do_initial_clean: Whether to clean text before detection.
        llm_instruction / llm_entities: Used only when an LLM PII method is selected.
        comprehend_entities: Used only when AWS Comprehend is selected.
        aws_access_key / aws_secret_key: Only needed for AWS Comprehend deployments that do not
            use IAM role/SSO.

    Returns:
        (output_paths, message)

    This wrapper deliberately avoids the long Gradio session-driven ``api_name='redact_data'``
    signature. Prefer calling it via the short route `api_name='/tabular_redact'`.
    """
    data_path = normalize_gradio_file_to_path(data_file)
    if not data_path:
        raise ValueError(
            "data_file is missing or could not be resolved to a path (upload the file first)."
        )

    out_dir = output_dir
    if isinstance(out_dir, str) and not out_dir.strip():
        out_dir = None
    safe_out_dir = _resolve_dir_within_base(out_dir, OUTPUT_FOLDER)
    os.makedirs(safe_out_dir, exist_ok=True)

    entities = list(redact_entities or [])
    chosen_cols = list(columns or [])

    (
        out_file_paths,
        out_message,
        _key_string,
        log_files_output_paths,
        _cq,
        _lt_in,
        _lt_out,
        _lm,
    ) = anonymise_files_with_open_text(
        file_paths=[data_path],
        in_text="",
        anon_strategy=str(anon_strategy or "redact"),
        chosen_cols=chosen_cols,
        chosen_redact_entities=entities,
        in_allow_list=list(allow_list or []),
        output_folder=str(safe_out_dir),
        in_deny_list=list(deny_list or []),
        max_fuzzy_spelling_mistakes_num=(
            int(max_fuzzy_spelling_mistakes_num)
            if max_fuzzy_spelling_mistakes_num is not None
            else 0
        ),
        pii_identification_method=str(pii_method or "Local"),
        chosen_redact_comprehend_entities=list(comprehend_entities or []),
        aws_access_key_textbox=str(aws_access_key or ""),
        aws_secret_key_textbox=str(aws_secret_key or ""),
        do_initial_clean=(
            bool(do_initial_clean) if do_initial_clean is not None else True
        ),
        language=str(language or "en"),
        custom_llm_instructions=str(llm_instruction or ""),
        chosen_llm_entities=(
            list(llm_entities or []) if llm_entities is not None else None
        ),
    )

    flat_paths: list[str] = []
    for item in (out_file_paths, log_files_output_paths):
        if not item:
            continue
        if isinstance(item, str):
            flat_paths.append(item)
        else:
            flat_paths.extend(str(p) for p in item if p)
    paths = _filter_files_within_root(flat_paths, safe_out_dir)

    # out_message is usually a list of strings in this workflow
    if isinstance(out_message, list):
        msg = "\n".join(str(x) for x in out_message if x)
    else:
        msg = str(out_message or "")
    msg = msg.strip() or "redact_data completed"
    return paths, msg


def redact_document_from_upload_for_gradio_api(
    document_file: Any,
    redact_entities: list[str] | None = None,
    output_dir: str | None = None,
    ocr_method: str | None = None,
    pii_method: str | None = "Local",
    allow_list: list[str] | None = None,
    deny_list: list[str] | None = None,
    page_min: int | None = None,
    page_max: int | None = None,
    llm_instruction: str | None = "",
) -> tuple[list[str], str]:
    """
    Short, stateless ``gr.api`` wrapper for PDF/image document redaction.

    Args:
        document_file: PDF/image path or Gradio upload payload (dict/object with path/name).
        redact_entities: Entity labels to detect/redact (e.g. PERSON, EMAIL_ADDRESS).
        output_dir: Directory to write outputs; constrained to OUTPUT_FOLDER.
        ocr_method: Optional OCR extraction mode override.
        pii_method: PII detector method (e.g. Local, AWS Comprehend).
        allow_list / deny_list: Optional explicit token lists for matching behaviour.
        page_min / page_max: Optional page bounds (0 means all, CLI semantics).
        llm_instruction: Optional custom instruction for LLM-backed detection.

    Returns:
        (output_paths, message)

    Prefer calling through the short route `api_name='/doc_redact'`.
    """
    from doc_redaction.cli_api import redact_document as cli_redact_document

    document_path = normalize_gradio_file_to_path(document_file)
    if not document_path:
        raise ValueError(
            "document_file is missing or could not be resolved to a path (upload the file first)."
        )
    if not os.path.isfile(document_path):
        raise ValueError(f"document_file not found or not a file: {document_path}")

    out_dir = output_dir
    if isinstance(out_dir, str) and not out_dir.strip():
        out_dir = None
    safe_out_dir = _resolve_dir_within_base(out_dir, OUTPUT_FOLDER)
    os.makedirs(safe_out_dir, exist_ok=True)

    overrides: dict[str, Any] = {}
    if redact_entities is not None:
        overrides["local_redact_entities"] = list(redact_entities)
    if allow_list is not None:
        overrides["allow_list"] = list(allow_list)
    if deny_list is not None:
        overrides["deny_list"] = list(deny_list)
    if page_min is not None:
        overrides["page_min"] = int(page_min)
    if page_max is not None:
        overrides["page_max"] = int(page_max)

    paths = cli_redact_document(
        input_files=[document_path],
        output_dir=safe_out_dir,
        ocr_method=ocr_method,
        pii_detector=pii_method,
        instruction=llm_instruction,
        overrides=overrides or None,
    )

    safe_paths = _filter_files_within_root(paths, safe_out_dir)
    return safe_paths, "doc_redact completed"


def summarise_document_from_upload_for_gradio_api(
    pdf_file: Any,
    ocr_method: str | None = None,
    summarisation_inference_method: str | None = None,
    summarisation_format: str | None = None,
    summarisation_context: str | None = None,
    summarisation_additional_instructions: str | None = None,
    summarisation_temperature: float | None = None,
    summarisation_max_pages_per_group: int | None = None,
    summarisation_api_key: str | None = None,
    output_dir: str | None = None,
    input_dir: str | None = None,
    page_min: int | None = None,
    page_max: int | None = None,
) -> tuple[list[str], str, str]:
    """
    ``gr.api`` handler: ``pdf_file`` (original PDF path or upload payload) plus optional
    overrides matching the main CLI summarise knobs (``ocr_method``,
    ``summarisation_*``, ``output_dir``, ``input_dir``, ``page_min`` / ``page_max``).
    Unset optional parameters use ``get_cli_default_args_dict()`` like ``cli_redact``.

    Returns ``(output_file_paths, status_message, summary_text)``.
    """
    from cli_redact import get_cli_default_args_dict

    pdf_path = normalize_gradio_file_to_path(pdf_file)
    if not pdf_path:
        raise ValueError(
            "pdf_file is missing or could not be resolved to a path (upload the PDF first)."
        )
    if not is_pdf(pdf_path):
        raise ValueError(
            "This route expects a PDF input. For OCR CSV-only summarisation, use the "
            "full Gradio api_name='summarise_document' chain or the CLI summarise task."
        )
    if not os.path.isfile(pdf_path):
        raise ValueError(f"PDF not found or not a file: {pdf_path}")

    a = get_cli_default_args_dict()

    def _pick(key: str, override: Any) -> Any:
        if override is not None and override != "":
            return override
        return a[key]

    ocr_m = str(_pick("ocr_method", ocr_method))
    out_folder = _resolve_dir_within_base(
        str(_pick("output_dir", output_dir)).strip() or str(a["output_dir"]),
        OUTPUT_FOLDER,
    )
    in_folder = _resolve_dir_within_base(
        str(_pick("input_dir", input_dir)).strip() or str(a["input_dir"]),
        INPUT_FOLDER,
    )
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(in_folder, exist_ok=True)
    p_min = int(_pick("page_min", page_min))
    p_max = int(_pick("page_max", page_max))

    summ_method = str(
        _pick("summarisation_inference_method", summarisation_inference_method)
    )
    summ_temp = float(_pick("summarisation_temperature", summarisation_temperature))
    summ_max_pages = int(
        _pick("summarisation_max_pages_per_group", summarisation_max_pages_per_group)
    )
    summ_api_key = str(_pick("summarisation_api_key", summarisation_api_key) or "")
    summ_ctx = str(_pick("summarisation_context", summarisation_context) or "")
    summ_extra = str(
        _pick(
            "summarisation_additional_instructions",
            summarisation_additional_instructions,
        )
        or ""
    )
    fmt_key = str(_pick("summarisation_format", summarisation_format) or "detailed")
    format_map = {
        "concise": concise_summary_format_prompt,
        "detailed": detailed_summary_format_prompt,
    }
    summarise_format_radio = format_map.get(fmt_key, detailed_summary_format_prompt)

    prepare_images = ocr_m in ["Local OCR", "AWS Textract"]

    prep = prepare_image_or_pdf(
        file_paths=[pdf_path],
        text_extract_method=ocr_m,
        all_line_level_ocr_results_df=pd.DataFrame(),
        all_page_line_level_ocr_results_with_words_df=pd.DataFrame(),
        first_loop_state=True,
        prepare_for_review=False,
        output_folder=out_folder,
        input_folder=in_folder,
        prepare_images=prepare_images,
        page_min=p_min,
        page_max=p_max,
    )
    _prep_summary = prep[0]
    prepared_pdf_paths = prep[1]
    image_file_paths = prep[2]
    pdf_doc = prep[5]
    image_annotations = prep[6]
    original_cropboxes = prep[8]
    page_sizes = prep[9]
    print(_prep_summary)

    try:
        red_tuple = run_redaction(
            [pdf_path],
            RedactionOptions(
                chosen_redact_entities=a.get("local_redact_entities") or [],
                chosen_redact_comprehend_entities=a.get("aws_redact_entities") or [],
                chosen_llm_entities=a.get("llm_redact_entities") or [],
                text_extraction_method=ocr_m,
                in_allow_list=a.get("allow_list_file"),
                in_deny_list=a.get("deny_list_file"),
                redact_whole_page_list=a.get("redact_whole_page_file"),
                page_min=p_min,
                page_max=p_max,
                handwrite_signature_checkbox=a.get("handwrite_signature_extraction")
                or [],
                max_fuzzy_spelling_mistakes_num=int(
                    a.get("fuzzy_mistakes", DEFAULT_FUZZY_SPELLING_MISTAKES_NUM)
                ),
                match_fuzzy_whole_phrase_bool=bool(
                    a.get("match_fuzzy_whole_phrase_bool", True)
                ),
                pii_identification_method=str(a.get("pii_detector") or "Local"),
                aws_access_key_textbox=str(a.get("aws_access_key") or ""),
                aws_secret_key_textbox=str(a.get("aws_secret_key") or ""),
                language=a.get("language"),
                output_folder=out_folder,
                input_folder=in_folder,
                custom_llm_instructions=str(a.get("custom_llm_instructions") or ""),
                inference_server_vlm_model=str(
                    a.get("inference_server_vlm_model")
                    or DEFAULT_INFERENCE_SERVER_VLM_MODEL
                ),
                efficient_ocr=bool(a.get("efficient_ocr", EFFICIENT_OCR)),
                efficient_ocr_min_words=int(
                    a.get("efficient_ocr_min_words") or EFFICIENT_OCR_MIN_WORDS
                ),
                efficient_ocr_min_image_coverage_fraction=float(
                    a.get("efficient_ocr_min_image_coverage_fraction")
                    if a.get("efficient_ocr_min_image_coverage_fraction") is not None
                    else EFFICIENT_OCR_MIN_IMAGE_COVERAGE_FRACTION
                ),
                efficient_ocr_min_embedded_image_px=int(
                    a.get("efficient_ocr_min_embedded_image_px")
                    if a.get("efficient_ocr_min_embedded_image_px") is not None
                    else EFFICIENT_OCR_MIN_EMBEDDED_IMAGE_PX
                ),
                ocr_first_pass_max_workers=int(
                    a.get("ocr_first_pass_max_workers") or OCR_FIRST_PASS_MAX_WORKERS
                ),
                hybrid_textract_bedrock_vlm=bool(
                    a.get("hybrid_textract_bedrock_vlm", HYBRID_TEXTRACT_BEDROCK_VLM)
                ),
                overwrite_existing_ocr_results=bool(
                    a.get(
                        "overwrite_existing_ocr_results",
                        OVERWRITE_EXISTING_OCR_RESULTS,
                    )
                ),
                save_page_ocr_visualisations=(
                    a.get("save_page_ocr_visualisations")
                    if a.get("save_page_ocr_visualisations") is not None
                    else SAVE_PAGE_OCR_VISUALISATIONS
                ),
                text_extraction_only=True,
            ),
            RedactionContext(
                prepared_pdf_file_paths=prepared_pdf_paths,
                pdf_image_file_paths=image_file_paths,
                pymupdf_doc=pdf_doc,
                annotations_all_pages=image_annotations,
                page_sizes=page_sizes,
                document_cropboxes=original_cropboxes,
            ),
        )
    finally:
        if pdf_doc is not None and hasattr(pdf_doc, "is_closed"):
            try:
                if not pdf_doc.is_closed:
                    pdf_doc.close()
            except Exception:
                pass

    ocr_df = red_tuple[12]
    if ocr_df is None or (isinstance(ocr_df, pd.DataFrame) and ocr_df.empty):
        return (
            [],
            "No OCR text extracted from PDF. Cannot summarise.",
            "",
        )

    basename = os.path.basename(pdf_path)
    file_name = os.path.splitext(basename)[0][:20]
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        file_name = file_name.replace(char, "_")
    file_name = file_name if file_name else "document"

    (
        output_files,
        status_message,
        _llm_model_name,
        _llm_in,
        _llm_out,
        summary_display_text,
        _elapsed,
    ) = summarise_document_wrapper(
        ocr_df,
        out_folder,
        summ_method,
        summ_api_key,
        summ_temp,
        file_name,
        summ_ctx,
        str(a.get("aws_access_key") or ""),
        str(a.get("aws_secret_key") or ""),
        "",
        AZURE_OPENAI_INFERENCE_ENDPOINT or "",
        summarise_format_radio,
        summ_extra,
        summ_max_pages,
        None,
    )

    safe_paths = _filter_files_within_root(list(output_files or []), out_folder)
    return safe_paths, str(status_message or ""), str(summary_display_text or "")


def review_apply_api(
    pdf_file: Any,
    review_csv_file: Any,
    output_dir: str | None = None,
) -> tuple[list[str], str]:
    """Short-name wrapper; prefer calling this via `api_name='/review_apply'`."""
    return apply_review_redactions_from_uploads_for_gradio_api(
        pdf_file=pdf_file, review_csv_file=review_csv_file, output_dir=output_dir
    )


def pdf_summarise_api(
    pdf_file: Any,
    ocr_method: str | None = None,
    summarisation_inference_method: str | None = None,
    summarisation_format: str | None = None,
    summarisation_context: str | None = None,
    summarisation_additional_instructions: str | None = None,
    summarisation_temperature: float | None = None,
    summarisation_max_pages_per_group: int | None = None,
    summarisation_api_key: str | None = None,
    output_dir: str | None = None,
    input_dir: str | None = None,
    page_min: int | None = None,
    page_max: int | None = None,
) -> tuple[list[str], str, str]:
    """Short-name wrapper; prefer calling this via `api_name='/pdf_summarise'`."""
    return summarise_document_from_upload_for_gradio_api(
        pdf_file=pdf_file,
        ocr_method=ocr_method,
        summarisation_inference_method=summarisation_inference_method,
        summarisation_format=summarisation_format,
        summarisation_context=summarisation_context,
        summarisation_additional_instructions=summarisation_additional_instructions,
        summarisation_temperature=summarisation_temperature,
        summarisation_max_pages_per_group=summarisation_max_pages_per_group,
        summarisation_api_key=summarisation_api_key,
        output_dir=output_dir,
        input_dir=input_dir,
        page_min=page_min,
        page_max=page_max,
    )


def tabular_redact_api(
    data_file: Any,
    redact_entities: list[str] | None = None,
    output_dir: str | None = None,
    pii_method: str | None = "Local",
    columns: list[str] | None = None,
    anon_strategy: str | None = "redact",
    allow_list: list[str] | None = None,
    deny_list: list[str] | None = None,
    language: str | None = "en",
    max_fuzzy_spelling_mistakes_num: int | None = 0,
    do_initial_clean: bool | None = True,
    llm_instruction: str | None = "",
    llm_entities: list[str] | None = None,
    comprehend_entities: list[str] | None = None,
    aws_access_key: str | None = "",
    aws_secret_key: str | None = "",
) -> tuple[list[str], str]:
    """Short-name wrapper; prefer calling this via `api_name='/tabular_redact'`."""
    return redact_data_from_upload_for_gradio_api(
        data_file=data_file,
        redact_entities=redact_entities,
        output_dir=output_dir,
        pii_method=pii_method,
        columns=columns,
        anon_strategy=anon_strategy,
        allow_list=allow_list,
        deny_list=deny_list,
        language=language,
        max_fuzzy_spelling_mistakes_num=max_fuzzy_spelling_mistakes_num,
        do_initial_clean=do_initial_clean,
        llm_instruction=llm_instruction,
        llm_entities=llm_entities,
        comprehend_entities=comprehend_entities,
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
    )


def doc_redact_api(
    document_file: Any,
    redact_entities: list[str] | None = None,
    output_dir: str | None = None,
    ocr_method: str | None = None,
    pii_method: str | None = "Local",
    allow_list: list[str] | None = None,
    deny_list: list[str] | None = None,
    page_min: int | None = None,
    page_max: int | None = None,
    llm_instruction: str | None = "",
) -> tuple[list[str], str]:
    """Short-name wrapper; prefer calling this via `api_name='/doc_redact'`."""
    return redact_document_from_upload_for_gradio_api(
        document_file=document_file,
        redact_entities=redact_entities,
        output_dir=output_dir,
        ocr_method=ocr_method,
        pii_method=pii_method,
        allow_list=allow_list,
        deny_list=deny_list,
        page_min=page_min,
        page_max=page_max,
        llm_instruction=llm_instruction,
    )


__all__ = [
    "HeadlessGradioProgress",
    "apply_review_redactions_from_uploads_for_gradio_api",
    "review_apply_api",
    "normalize_gradio_file_to_path",
    "redact_data_from_upload_for_gradio_api",
    "redact_document_from_upload_for_gradio_api",
    "tabular_redact_api",
    "doc_redact_api",
    "run_apply_review_redactions",
    "summarise_document_from_upload_for_gradio_api",
    "pdf_summarise_api",
]
