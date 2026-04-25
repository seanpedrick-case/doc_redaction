"""
Headless and short ``gr.api`` entrypoints for agents and Gradio clients.

Consolidates:
- Review apply (``run_apply_review_redactions``, short `review_apply`)
- PDF summarisation (short `pdf_summarise`)
- Tabular redaction (short `tabular_redact`)
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from tools.config import (
    AWS_LLM_PII_OPTION,
    AWS_PII_OPTION,
    AZURE_OPENAI_INFERENCE_ENDPOINT,
    DEFAULT_FUZZY_SPELLING_MISTAKES_NUM,
    DEFAULT_INFERENCE_SERVER_VLM_MODEL,
    EFFICIENT_OCR,
    EFFICIENT_OCR_MIN_EMBEDDED_IMAGE_PX,
    EFFICIENT_OCR_MIN_IMAGE_COVERAGE_FRACTION,
    EFFICIENT_OCR_MIN_WORDS,
    HYBRID_TEXTRACT_BEDROCK_VLM,
    INFERENCE_SERVER_PII_OPTION,
    INPUT_FOLDER,
    LOCAL_OCR_MODEL_OPTIONS,
    LOCAL_PII_OPTION,
    LOCAL_TRANSFORMERS_LLM_PII_OPTION,
    NO_REDACTION_PII_OPTION,
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
from tools.secure_path_utils import validate_path_safety
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
    base_real = os.path.realpath(base_abs)
    raw = candidate_dir if candidate_dir is not None else base_dir
    resolved = os.path.normpath(os.path.abspath(os.path.expanduser(str(raw))))
    resolved_real = os.path.realpath(resolved)
    try:
        common = os.path.commonpath([resolved_real, base_real])
    except ValueError as exc:
        raise ValueError(f"Invalid directory path: {raw}") from exc
    if common != base_real:
        raise ValueError(
            f"Directory must be within configured base folder: {base_real}"
        )
    if not validate_path_safety(resolved_real, base_path=base_real):
        raise ValueError(f"Unsafe directory path rejected: {raw}")
    return _folder_with_trailing_sep(resolved_real)


def _mkdir_within_base(dir_path: str, base_dir: str) -> str:
    """
    Create dir_path (and parents) after enforcing it is within base_dir.

    Uses pathlib containment checks on canonicalized paths. This is largely to satisfy
    CodeQL path-injection dataflow expectations while preserving existing behaviour
    (allowing caller overrides within the configured base).
    """
    try:
        base = Path(base_dir).expanduser().resolve(strict=False)
        candidate = Path(dir_path).expanduser().resolve(strict=False)
        candidate.relative_to(base)
    except Exception as exc:
        raise ValueError(
            f"Directory must be within configured base folder: {base_dir}"
        ) from exc

    if not validate_path_safety(str(candidate), base_path=str(base)):
        raise ValueError(f"Unsafe directory path rejected: {candidate}")

    candidate.mkdir(parents=True, exist_ok=True)
    return _folder_with_trailing_sep(str(candidate))


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
        if not validate_path_safety(resolved, base_path=safe_root):
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


def _resolve_cli_ocr_inputs(
    ocr_method: str | None,
) -> tuple[str | None, dict[str, Any]]:
    """
    Normalize user-provided OCR input into CLI-compatible ocr_method/overrides.

    The CLI separates high-level extraction mode (`ocr_method`) from local engine
    choice (`chosen_local_ocr_model`). This helper accepts convenient inputs like
    "paddle" and maps them to:
      - ocr_method="Local OCR"
      - overrides={"chosen_local_ocr_model": "paddle"}
    """
    if ocr_method is None:
        return None, {}

    raw = str(ocr_method).strip()
    if not raw:
        return None, {}

    lower = raw.lower()
    mode_aliases = {
        "aws textract": "AWS Textract",
        "textract": "AWS Textract",
        "local ocr": "Local OCR",
        "local": "Local OCR",
        "local text": "Local text",
        "text": "Local text",
        "simple text": "Local text",
    }
    if lower in mode_aliases:
        return mode_aliases[lower], {}

    model_aliases = {
        "hybrid paddle": "hybrid-paddle",
        "hybrid vlm": "hybrid-vlm",
        "hybrid paddle vlm": "hybrid-paddle-vlm",
        "hybrid paddle inference server": "hybrid-paddle-inference-server",
        "inference server": "inference-server",
        "bedrock": "bedrock-vlm",
        "gemini": "gemini-vlm",
        "azure": "azure-openai-vlm",
    }
    canonical_local_models = (
        "tesseract",
        "paddle",
        "hybrid-paddle",
        "hybrid-vlm",
        "hybrid-paddle-vlm",
        "hybrid-paddle-inference-server",
        "vlm",
        "inference-server",
        "bedrock-vlm",
        "gemini-vlm",
        "azure-openai-vlm",
    )
    available_models = {
        str(m).lower(): str(m)
        for m in (*canonical_local_models, *LOCAL_OCR_MODEL_OPTIONS)
    }
    for alias, model in model_aliases.items():
        available_models[alias] = model

    compact = re.sub(r"[\s_]+", "-", lower)
    if compact in available_models:
        chosen_model = available_models[compact]
        return "Local OCR", {"chosen_local_ocr_model": chosen_model}
    if lower in available_models:
        chosen_model = available_models[lower]
        return "Local OCR", {"chosen_local_ocr_model": chosen_model}

    return raw, {}


def _resolve_cli_pii_method(pii_method: str | None) -> str | None:
    """
    Normalize PII detector strings to configured display labels.

    Supports common aliases while preserving deployment-specific configured names.
    """
    if pii_method is None:
        return None

    raw = str(pii_method).strip()
    if not raw:
        return None

    normalized = re.sub(r"[\s_]+", " ", raw.strip().lower())
    aliases = {
        "local": LOCAL_PII_OPTION,
        "aws": AWS_PII_OPTION,
        "aws comprehend": AWS_PII_OPTION,
        "comprehend": AWS_PII_OPTION,
        "llm (aws bedrock)": AWS_LLM_PII_OPTION,
        "aws bedrock llm": AWS_LLM_PII_OPTION,
        "bedrock llm": AWS_LLM_PII_OPTION,
        "local inference server": INFERENCE_SERVER_PII_OPTION,
        "inference server": INFERENCE_SERVER_PII_OPTION,
        "local transformers llm": LOCAL_TRANSFORMERS_LLM_PII_OPTION,
        "transformers llm": LOCAL_TRANSFORMERS_LLM_PII_OPTION,
        "none": "None",
        "no redaction": "None",
        "only extract text (no redaction)": NO_REDACTION_PII_OPTION,
    }
    if normalized in aliases:
        return aliases[normalized]

    return raw


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

    out_folder = _resolve_dir_within_base(output_dir, OUTPUT_FOLDER)
    in_folder = _resolve_dir_within_base(input_dir, INPUT_FOLDER)

    out_folder = _mkdir_within_base(out_folder, OUTPUT_FOLDER)
    in_folder = _mkdir_within_base(in_folder, INPUT_FOLDER)

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

    def _resolve_safe_output_file(candidate_path: Any, output_root: str) -> str | None:
        if candidate_path is None:
            return None
        candidate_text = str(candidate_path).strip()
        if not candidate_text:
            return None
        resolved_candidate = os.path.realpath(candidate_text)
        try:
            within_output_root = (
                os.path.commonpath([output_root, resolved_candidate]) == output_root
            )
        except ValueError:
            return None
        if not within_output_root:
            return None
        if not validate_path_safety(resolved_candidate, base_path=output_root):
            return None
        try:
            if not Path(resolved_candidate).is_file():
                return None
        except OSError:
            return None
        return resolved_candidate

    seen: set[str] = set()
    unique_paths: list[str] = []
    for p in out_paths:
        resolved = _resolve_safe_output_file(p, safe_output_root)
        if not resolved:
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


def _is_gradio_ephemeral_upload_path(path: str) -> bool:
    """True for Gradio ``/gradio_api/upload`` temp files that may be deleted mid-request."""
    norm = os.path.normpath(path or "").replace("\\", "/").lower()
    return "gradio_tmp" in norm or "/tmp/gradio/" in norm


def _api_upload_staging_dir() -> str:
    base = _resolve_dir_within_base(None, INPUT_FOLDER).rstrip(os.sep)
    return os.path.join(base, "api_upload_staging")


def stage_gradio_upload_if_ephemeral(src: str) -> str:
    """
    Copy HTTP-uploaded files from Gradio's temp tree into ``INPUT_FOLDER`` staging.

    Long-running pipelines (OCR, redaction) otherwise race Gradio/tmp reapers or
    concurrent uploads, producing "Failed to open file '/tmp/gradio_tmp/...'".
    """
    if not src or not os.path.isfile(src):
        return src
    if not _is_gradio_ephemeral_upload_path(src):
        return src
    staging = _api_upload_staging_dir()
    os.makedirs(staging, exist_ok=True)
    base = os.path.basename(src) or "upload.bin"
    dest = os.path.join(staging, f"{uuid.uuid4().hex}_{base}")
    shutil.copy2(src, dest)
    return dest


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
    if not os.path.isfile(pdf_path):
        raise ValueError(f"pdf_file not found or not a file: {pdf_path}")
    if not os.path.isfile(csv_path):
        raise ValueError(f"review_csv_file not found or not a file: {csv_path}")
    pdf_path = stage_gradio_upload_if_ephemeral(pdf_path)
    csv_path = stage_gradio_upload_if_ephemeral(csv_path)
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
    if not os.path.isfile(data_path):
        raise ValueError(f"data_file not found or not a file: {data_path}")
    data_path = stage_gradio_upload_if_ephemeral(data_path)

    out_dir = output_dir
    if isinstance(out_dir, str) and not out_dir.strip():
        out_dir = None
    safe_out_dir = _resolve_dir_within_base(out_dir, OUTPUT_FOLDER)
    os.makedirs(safe_out_dir, exist_ok=True)

    entities = list(redact_entities or [])
    chosen_cols = list(columns or [])

    (
        out_message_out,
        out_file_paths,
        _out_paths_dup,
        _latest_completed,
        log_files_output_paths,
        _log_paths_dup,
        _actual_time,
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

    # anonymise_files_with_open_text returns a single final message string at [0]
    if isinstance(out_message_out, list):
        msg = "\n".join(str(x) for x in out_message_out if x)
    else:
        msg = str(out_message_out or "")
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
        ocr_method: OCR extraction mode override. Accepts high-level methods
            (`Local OCR`, `AWS Textract`, `Local text`) and also local engine
            shortcuts such as `paddle`/`tesseract`, which are auto-mapped to
            `Local OCR` plus the matching `chosen_local_ocr_model`.
        pii_method: PII detector method. Accepts configured labels
            (`Local`, `AWS Comprehend`, `LLM (AWS Bedrock)`, `Local inference server`,
            `Local transformers LLM`, `None`) plus common aliases.
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
    document_path = stage_gradio_upload_if_ephemeral(document_path)

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

    cli_ocr_method, ocr_overrides = _resolve_cli_ocr_inputs(ocr_method)
    cli_pii_method = _resolve_cli_pii_method(pii_method)
    merged_overrides = dict(overrides)
    merged_overrides.update(ocr_overrides)

    paths = cli_redact_document(
        input_files=[document_path],
        output_dir=safe_out_dir,
        ocr_method=cli_ocr_method,
        pii_detector=cli_pii_method,
        instruction=llm_instruction,
        overrides=merged_overrides or None,
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
    out_folder = _mkdir_within_base(out_folder, OUTPUT_FOLDER)
    in_folder = _mkdir_within_base(in_folder, INPUT_FOLDER)
    pdf_path = stage_gradio_upload_if_ephemeral(pdf_path)
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


def preview_boxes_api(
    pdf_file: Any,
    review_csv_file: Any,
    dpi: int | None = 150,
    max_width: int | None = 1280,
    draw_grid: bool | None = True,
    pages: str | None = None,
) -> tuple[str, str]:
    """
    Render proposed redaction boxes from *review_csv_file* onto the
    original *pdf_file* and return a ZIP archive of preview PNGs.

    Use this endpoint when you do **not** have a local copy of the
    original PDF and want to verify box positions without calling
    ``/review_apply``.  For agents that already hold local files,
    calling ``tools.preview_redaction_boxes.preview_redaction_boxes``
    directly is faster (no upload/download round-trip).

    Parameters
    ----------
    pdf_file:
        The original (un-redacted) PDF uploaded by the caller.
    review_csv_file:
        The ``*_review_file.csv`` (original or edited) uploaded by the
        caller.
    dpi:
        Render resolution (default 150).
    max_width:
        Maximum output image width in pixels (default 1280).
    draw_grid:
        If True (default), overlay percentage-grid lines so normalized
        y-coordinates can be read by eye.
    pages:
        Optional comma-separated 1-indexed page numbers, e.g. ``"1,3,5"``.
        If omitted, all pages are rendered.

    Returns
    -------
    tuple[str, str]
        ``(zip_path, message)`` where *zip_path* is a server-side path to
        a ZIP file of preview PNGs retrievable via
        ``GET /gradio_api/file=<zip_path>``.
    """
    import tempfile

    from tools.preview_redaction_boxes import preview_redaction_boxes

    pdf_path = normalize_gradio_file_to_path(pdf_file)
    csv_path = normalize_gradio_file_to_path(review_csv_file)

    if not pdf_path or not csv_path:
        return "", "Error: both pdf_file and review_csv_file are required."

    pdf_path = stage_gradio_upload_if_ephemeral(pdf_path, INPUT_FOLDER)
    csv_path = stage_gradio_upload_if_ephemeral(csv_path, INPUT_FOLDER)

    page_list: list[int] | None = None
    if pages:
        try:
            page_list = [int(p.strip()) for p in pages.split(",") if p.strip()]
        except ValueError:
            return (
                "",
                f"Error: 'pages' must be comma-separated integers, got: {pages!r}",
            )

    with tempfile.TemporaryDirectory() as tmp:
        out_paths = preview_redaction_boxes(
            pdf_path,
            csv_path,
            out_dir=tmp,
            dpi=int(dpi or 150),
            max_width=int(max_width or 1280),
            draw_grid=bool(draw_grid),
            pages=page_list,
        )

        if not out_paths:
            return (
                "",
                "No pages rendered — check that the CSV contains rows with valid page numbers.",
            )

        out_base = Path(OUTPUT_FOLDER) / f"preview_{Path(pdf_path).stem}"
        out_base.mkdir(parents=True, exist_ok=True)
        zip_path = str(out_base / "preview_boxes.zip")

        import zipfile

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in out_paths:
                zf.write(p, arcname=Path(p).name)

    n = len(out_paths)
    msg = f"Preview complete: {n} page(s) rendered. Download the ZIP to inspect box positions."
    return zip_path, msg


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
    "stage_gradio_upload_if_ephemeral",
    "redact_data_from_upload_for_gradio_api",
    "redact_document_from_upload_for_gradio_api",
    "tabular_redact_api",
    "doc_redact_api",
    "run_apply_review_redactions",
    "summarise_document_from_upload_for_gradio_api",
    "pdf_summarise_api",
    "preview_boxes_api",
]
