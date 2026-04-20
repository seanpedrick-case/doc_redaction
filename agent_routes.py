"""
FastAPI routes for programmatic / agent callers.

HTTP paths align with Gradio ``api_name`` values in app.py. See GET /agent/operations
for the full map. Uses cli_redact.main(direct_mode_args=...) where a CLI task exists.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from tools.config import INPUT_FOLDER, OUTPUT_FOLDER
from tools.secure_path_utils import validate_path_safety

router = APIRouter(tags=["Agent"])

REPO_ROOT = Path(__file__).resolve().parent
_MAX_INSTRUCTION_LEN = 16_000

# NOTE: Paths from request bodies are untrusted. Avoid Path.resolve() on untrusted
# input (CodeQL py/path-injection); instead normalize via os.path and enforce
# containment under trusted roots.

# Mirrors app.py api_name values (Gradio).
GRADIO_API_NAMES: tuple[str, ...] = (
    "redact_document",
    "load_and_prepare_documents_or_data",
    "apply_review_redactions",
    "word_level_ocr_text_search",
    "redact_data",
    "find_duplicate_pages",
    "find_duplicate_tabular",
    "summarise_document",
    "combine_review_csvs",
    "combine_review_pdfs",
    "export_review_redaction_overlay",
    "export_review_page_ocr_visualisation",
)


def _allowed_path_roots() -> list[Path]:
    roots = [REPO_ROOT.resolve()]
    for folder in (INPUT_FOLDER, OUTPUT_FOLDER):
        try:
            roots.append(Path(folder).resolve())
        except (OSError, TypeError, ValueError):
            continue
    return roots


def _sanitize_untrusted_path_input(path_str: str) -> str:
    """Basic raw-input validation before any path normalization."""
    if not isinstance(path_str, str):
        raise HTTPException(status_code=400, detail="Path must be a string.")
    cleaned = path_str.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Path must not be empty.")
    if "\x00" in cleaned:
        raise HTTPException(status_code=400, detail="Path contains invalid null byte.")
    return cleaned


def _normalize_untrusted_path_to_abs(path_str: str) -> str:
    """
    Expand ~, then normalize to an absolute path.

    Relative paths are interpreted relative to REPO_ROOT (matching prior behaviour).
    """
    safe_input = _sanitize_untrusted_path_input(path_str)
    expanded = os.path.expanduser(safe_input)
    if os.path.isabs(expanded):
        return os.path.normpath(os.path.abspath(expanded))
    return os.path.normpath(os.path.abspath(os.path.join(str(REPO_ROOT), expanded)))


def _must_be_under_allowed_roots(candidate_abs: str, original: str) -> None:
    """Enforce candidate is contained under repo, INPUT_FOLDER, or OUTPUT_FOLDER."""
    allowed_roots = [
        os.path.normpath(os.path.abspath(str(p))) for p in _allowed_path_roots()
    ]
    for root in allowed_roots:
        try:
            common = os.path.commonpath([candidate_abs, root])
        except ValueError:
            # Different drive on Windows or invalid path mix
            continue
        if common == root:
            return
    raise HTTPException(
        status_code=403,
        detail="Path must be under the app repo, INPUT_FOLDER, or OUTPUT_FOLDER",
    )


def _path_must_be_allowed_file(path_str: str) -> str:
    """Resolve path, ensure it is under an allowed root and exists as a file."""
    candidate_abs = _normalize_untrusted_path_to_abs(path_str)
    # Validate both "safe path" patterns and containment under trusted roots.
    ok = False
    for root in _allowed_path_roots():
        if validate_path_safety(candidate_abs, base_path=str(root)):
            ok = True
            break
    if not ok:
        raise HTTPException(status_code=400, detail=f"Unsafe path rejected: {path_str}")
    if not os.path.isfile(candidate_abs):
        raise HTTPException(
            status_code=400, detail=f"Not a file or missing: {candidate_abs}"
        )
    return candidate_abs


def _path_must_be_allowed_directory(path_str: str, *, must_exist: bool = True) -> str:
    """
    Normalize and validate a directory path under allowed roots.

    By default the directory must already exist; callers can opt out (e.g. output_dir
    that will be created later by the CLI).
    """
    candidate_abs = _normalize_untrusted_path_to_abs(path_str)
    ok = False
    for root in _allowed_path_roots():
        if validate_path_safety(candidate_abs, base_path=str(root)):
            ok = True
            break
    if not ok:
        raise HTTPException(status_code=400, detail=f"Unsafe path rejected: {path_str}")
    if must_exist and not os.path.isdir(candidate_abs):
        raise HTTPException(status_code=400, detail=f"Not a directory: {candidate_abs}")
    return candidate_abs


def _optional_agent_api_key(x_agent_api_key: Optional[str] = Header(None)) -> None:
    expected = os.environ.get("AGENT_API_KEY", "").strip()
    if not expected:
        return
    if not x_agent_api_key or x_agent_api_key.strip() != expected:
        raise HTTPException(
            status_code=401,
            detail="Set header X-Agent-API-Key to match AGENT_API_KEY environment variable",
        )


class AgentRedactDocumentRequest(BaseModel):
    """Parity with Gradio api_name ``redact_document``."""

    input_files: list[str] = Field(
        ...,
        min_length=1,
        description="Paths to input files (PDF, images, or tabular/Word for anonymisation)",
    )
    instruction: Optional[str] = Field(
        None,
        description="Optional instructions for LLM-based PII detection (custom_llm_instructions)",
    )
    output_dir: Optional[str] = None
    input_dir: Optional[str] = None
    ocr_method: Optional[str] = None
    pii_detector: Optional[str] = None
    overrides: Optional[dict[str, Any]] = Field(
        None,
        description="Optional CLI flag overrides; keys must match argparse destination names",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input_files": [
                        "example_data/example_of_emails_sent_to_a_professor_before_applying.pdf"
                    ],
                    "instruction": "Do not redact the university name.",
                }
            ]
        }
    }

    @field_validator("instruction")
    @classmethod
    def _cap_instruction(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if len(v) > _MAX_INSTRUCTION_LEN:
            raise ValueError(f"instruction exceeds {_MAX_INSTRUCTION_LEN} characters")
        return v


class AgentRedactDataRequest(AgentRedactDocumentRequest):
    """Parity with Gradio api_name ``redact_data``; same CLI task as redact_document."""


class AgentTaskResponse(BaseModel):
    status: str
    gradio_api_name: str
    task: str
    output_dir: str
    input_dir: str
    message: str
    log_excerpt: Optional[str] = None
    output_paths: Optional[list[str]] = None


def _merge_redact_direct_mode(body: AgentRedactDocumentRequest) -> dict[str, Any]:
    from cli_redact import get_cli_default_args_dict

    merged: dict[str, Any] = get_cli_default_args_dict()
    merged["task"] = "redact"
    merged["input_file"] = [_path_must_be_allowed_file(p) for p in body.input_files]

    if body.instruction is not None:
        merged["custom_llm_instructions"] = body.instruction
    if body.output_dir is not None:
        # Output folders may not exist yet (CLI will create). Still constrain to allowed roots.
        merged["output_dir"] = _path_must_be_allowed_directory(
            body.output_dir, must_exist=False
        )
    if body.input_dir is not None:
        # Input dir should exist if provided.
        merged["input_dir"] = _path_must_be_allowed_directory(
            body.input_dir, must_exist=True
        )
    if body.ocr_method is not None:
        merged["ocr_method"] = body.ocr_method
    if body.pii_detector is not None:
        merged["pii_detector"] = body.pii_detector

    if body.overrides:
        allowed = set(merged.keys())
        for key, value in body.overrides.items():
            if key not in allowed:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown override key '{key}'. Must be a known CLI argument name.",
                )
            merged[key] = value

    return merged


def _run_cli_main(direct: dict[str, Any], gradio_api_name: str) -> AgentTaskResponse:
    from cli_redact import main as cli_main

    buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        cli_main(direct_mode_args=direct)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        sys.stdout = old_stdout

    log_excerpt = buf.getvalue()
    if len(log_excerpt) > 8000:
        log_excerpt = log_excerpt[-8000:]

    return AgentTaskResponse(
        status="completed",
        gradio_api_name=gradio_api_name,
        task=str(direct.get("task", "")),
        output_dir=str(direct.get("output_dir", "")),
        input_dir=str(direct.get("input_dir", "")),
        message="cli_redact.main finished; see log_excerpt for console output",
        log_excerpt=log_excerpt or None,
    )


@router.post(
    "/redact_document",
    response_model=AgentTaskResponse,
    summary="redact_document (Gradio api_name)",
    description=(
        "Matches Gradio ``api_name='redact_document'``. "
        "``python cli_redact.py --task redact --input_file ...``. "
        "Optional ``instruction`` maps to ``custom_llm_instructions``."
    ),
)
def post_redact_document(
    body: AgentRedactDocumentRequest,
    _: None = Depends(_optional_agent_api_key),
) -> AgentTaskResponse:
    direct = _merge_redact_direct_mode(body)
    return _run_cli_main(direct, "redact_document")


@router.post(
    "/redact_data",
    response_model=AgentTaskResponse,
    summary="redact_data (Gradio api_name)",
    description=(
        "Matches Gradio ``api_name='redact_data'``. Same CLI ``redact`` task as "
        "/redact_document; use CSV/XLSX/DOCX paths for tabular/Word flows."
    ),
)
def post_redact_data(
    body: AgentRedactDataRequest,
    _: None = Depends(_optional_agent_api_key),
) -> AgentTaskResponse:
    direct = _merge_redact_direct_mode(body)
    return _run_cli_main(direct, "redact_data")


@router.post(
    "/tasks/redact",
    response_model=AgentTaskResponse,
    summary="Legacy: same as /redact_document",
    description="Deprecated alias; prefer POST /agent/redact_document.",
    deprecated=True,
    include_in_schema=True,
)
def post_tasks_redact_legacy(
    body: AgentRedactDocumentRequest,
    _: None = Depends(_optional_agent_api_key),
) -> AgentTaskResponse:
    direct = _merge_redact_direct_mode(body)
    return _run_cli_main(direct, "redact_document")


class AgentFindDuplicatePagesRequest(BaseModel):
    input_files: list[str] = Field(..., min_length=1)
    similarity_threshold: Optional[float] = None
    min_word_count: Optional[int] = None
    min_consecutive_pages: Optional[int] = None
    greedy_match: Optional[bool] = None
    combine_pages: Optional[bool] = None
    overrides: Optional[dict[str, Any]] = None


@router.post(
    "/find_duplicate_pages",
    response_model=AgentTaskResponse,
    summary="find_duplicate_pages (Gradio api_name)",
    description="``cli_redact --task deduplicate --duplicate_type pages``.",
)
def post_find_duplicate_pages(
    body: AgentFindDuplicatePagesRequest,
    _: None = Depends(_optional_agent_api_key),
) -> AgentTaskResponse:
    from cli_redact import get_cli_default_args_dict

    merged = get_cli_default_args_dict()
    merged["task"] = "deduplicate"
    merged["duplicate_type"] = "pages"
    merged["input_file"] = [_path_must_be_allowed_file(p) for p in body.input_files]
    if body.similarity_threshold is not None:
        merged["similarity_threshold"] = body.similarity_threshold
    if body.min_word_count is not None:
        merged["min_word_count"] = body.min_word_count
    if body.min_consecutive_pages is not None:
        merged["min_consecutive_pages"] = body.min_consecutive_pages
    if body.greedy_match is not None:
        merged["greedy_match"] = "True" if body.greedy_match else "False"
    if body.combine_pages is not None:
        merged["combine_pages"] = "True" if body.combine_pages else "False"
    if body.overrides:
        allowed = set(merged.keys())
        for k, v in body.overrides.items():
            if k not in allowed:
                raise HTTPException(400, f"Unknown override key: {k}")
            merged[k] = v
    return _run_cli_main(merged, "find_duplicate_pages")


class AgentFindDuplicateTabularRequest(BaseModel):
    input_files: list[str] = Field(..., min_length=1)
    text_columns: Optional[list[str]] = None
    similarity_threshold: Optional[float] = None
    min_word_count: Optional[int] = None
    overrides: Optional[dict[str, Any]] = None


@router.post(
    "/find_duplicate_tabular",
    response_model=AgentTaskResponse,
    summary="find_duplicate_tabular (Gradio api_name)",
)
def post_find_duplicate_tabular(
    body: AgentFindDuplicateTabularRequest,
    _: None = Depends(_optional_agent_api_key),
) -> AgentTaskResponse:
    from cli_redact import get_cli_default_args_dict

    merged = get_cli_default_args_dict()
    merged["task"] = "deduplicate"
    merged["duplicate_type"] = "tabular"
    merged["input_file"] = [_path_must_be_allowed_file(p) for p in body.input_files]
    if body.text_columns is not None:
        merged["text_columns"] = body.text_columns
    if body.similarity_threshold is not None:
        merged["similarity_threshold"] = body.similarity_threshold
    if body.min_word_count is not None:
        merged["min_word_count"] = body.min_word_count
    if body.overrides:
        allowed = set(merged.keys())
        for k, v in body.overrides.items():
            if k not in allowed:
                raise HTTPException(400, f"Unknown override key: {k}")
            merged[k] = v
    return _run_cli_main(merged, "find_duplicate_tabular")


class AgentSummariseDocumentRequest(BaseModel):
    input_files: list[str] = Field(..., min_length=1)
    summarisation_inference_method: Optional[str] = None
    summarisation_format: Optional[str] = None
    summarisation_context: Optional[str] = None
    summarisation_additional_instructions: Optional[str] = None
    overrides: Optional[dict[str, Any]] = None


@router.post(
    "/summarise_document",
    response_model=AgentTaskResponse,
    summary="summarise_document (Gradio api_name)",
)
def post_summarise_document(
    body: AgentSummariseDocumentRequest,
    _: None = Depends(_optional_agent_api_key),
) -> AgentTaskResponse:
    from cli_redact import get_cli_default_args_dict

    merged = get_cli_default_args_dict()
    merged["task"] = "summarise"
    merged["input_file"] = [_path_must_be_allowed_file(p) for p in body.input_files]
    if body.summarisation_inference_method is not None:
        merged["summarisation_inference_method"] = body.summarisation_inference_method
    if body.summarisation_format is not None:
        merged["summarisation_format"] = body.summarisation_format
    if body.summarisation_context is not None:
        merged["summarisation_context"] = body.summarisation_context
    if body.summarisation_additional_instructions is not None:
        merged["summarisation_additional_instructions"] = (
            body.summarisation_additional_instructions
        )
    if body.overrides:
        allowed = set(merged.keys())
        for k, v in body.overrides.items():
            if k not in allowed:
                raise HTTPException(400, f"Unknown override key: {k}")
            merged[k] = v
    return _run_cli_main(merged, "summarise_document")


class AgentCombineReviewPdfsRequest(BaseModel):
    input_files: list[str] = Field(..., min_length=2)
    output_dir: Optional[str] = None


@router.post(
    "/combine_review_pdfs",
    response_model=AgentTaskResponse,
    summary="combine_review_pdfs (Gradio api_name)",
)
def post_combine_review_pdfs(
    body: AgentCombineReviewPdfsRequest,
    _: None = Depends(_optional_agent_api_key),
) -> AgentTaskResponse:
    from cli_redact import get_cli_default_args_dict

    merged = get_cli_default_args_dict()
    merged["task"] = "combine_review_pdfs"
    merged["input_file"] = [_path_must_be_allowed_file(p) for p in body.input_files]
    if body.output_dir is not None:
        merged["output_dir"] = _path_must_be_allowed_directory(body.output_dir)
    return _run_cli_main(merged, "combine_review_pdfs")


class _NamedPath:
    """merge_csv_files expects objects with a .name attribute (Gradio file-like)."""

    __slots__ = ("name",)

    def __init__(self, path: str) -> None:
        self.name = path


class AgentCombineReviewCsvsRequest(BaseModel):
    input_files: list[str] = Field(..., min_length=1)
    output_dir: Optional[str] = Field(
        None, description="Defaults to config OUTPUT_FOLDER"
    )


@router.post(
    "/combine_review_csvs",
    response_model=AgentTaskResponse,
    summary="combine_review_csvs (Gradio api_name)",
    description="Uses tools.helper_functions.merge_csv_files (not cli_redact).",
)
def post_combine_review_csvs(
    body: AgentCombineReviewCsvsRequest,
    _: None = Depends(_optional_agent_api_key),
) -> AgentTaskResponse:
    from tools.helper_functions import merge_csv_files

    paths = [_NamedPath(_path_must_be_allowed_file(p)) for p in body.input_files]
    out_dir = body.output_dir or OUTPUT_FOLDER
    out_dir_resolved = _path_must_be_allowed_directory(str(out_dir), must_exist=True)
    sep = "/" if not out_dir_resolved.endswith(("/", "\\")) else ""
    out_files = merge_csv_files(paths, output_folder=out_dir_resolved + sep)
    return AgentTaskResponse(
        status="completed",
        gradio_api_name="combine_review_csvs",
        task="combine_review_csvs",
        output_dir=out_dir_resolved,
        input_dir="",
        message="merge_csv_files completed",
        output_paths=out_files,
    )


class AgentExportReviewRedactionOverlayRequest(BaseModel):
    """Parity with Gradio ``api_name='export_review_redaction_overlay'``."""

    page_image_path: str = Field(
        ...,
        description="Path to page raster (PNG/JPEG) used as underlay; must be under allowed roots.",
    )
    boxes: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="Annotator-style boxes: label, color, xmin, ymin, xmax, ymax (normalized 0–1).",
    )
    page_number: int = Field(
        1, ge=1, description="1-based page index for the output filename."
    )
    doc_base_name: str = Field(
        "review",
        description="Basename for output file (e.g. document name without extension).",
    )
    review_df_records: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Optional rows (include at least 'label') for stable label→line-pattern mapping.",
    )
    label_abbrev_chars: Optional[int] = Field(
        None,
        ge=0,
        le=24,
        description="Draw this many leading characters of each label on the image; omit to use REVIEW_OVERLAY_LABEL_ABBREV_CHARS from config (0 = off).",
    )


class AgentExportReviewPageOcrVisualisationRequest(BaseModel):
    """Parity with Gradio ``api_name='export_review_page_ocr_visualisation'``."""

    page_image_path: str = Field(
        ...,
        description="Path to page raster (PNG/JPEG) used as underlay; must be under allowed roots.",
    )
    ocr_results: Dict[str, Any] = Field(
        ...,
        description="Word-level OCR results dict (line_key -> {words:[{text, bounding_box, conf, ...}]}).",
    )
    page_number: int = Field(
        1, ge=1, description="1-based page index (used for naming)."
    )
    doc_base_name: str = Field(
        "review",
        description="Basename for output file (e.g. document name without extension).",
    )


@router.post(
    "/export_review_redaction_overlay",
    response_model=AgentTaskResponse,
    summary="export_review_redaction_overlay (Gradio api_name)",
    description=(
        "Renders hollow redaction outlines and a top-right legend on the page image; "
        "writes ``redaction_overlay/{doc_base_name}_page{n}_redaction_overlay.jpg`` under OUTPUT_FOLDER "
        "(scaled per REVIEW_OVERLAY_MAX_PIXELS, JPEG capped by REVIEW_OVERLAY_MAX_FILE_BYTES). "
        "Uses ``tools.redaction_review.visualise_review_redaction_boxes``."
    ),
)
def post_export_review_redaction_overlay(
    body: AgentExportReviewRedactionOverlayRequest,
    _: None = Depends(_optional_agent_api_key),
) -> AgentTaskResponse:
    import pandas as pd

    from tools.redaction_review import visualise_review_redaction_boxes

    img_path = _path_must_be_allowed_file(body.page_image_path)
    annotator: dict[str, Any] = {"image": img_path, "boxes": body.boxes}
    review_df = (
        pd.DataFrame(body.review_df_records)
        if body.review_df_records
        else pd.DataFrame()
    )
    out_folder_path = Path(OUTPUT_FOLDER).expanduser().resolve()
    if not validate_path_safety(str(out_folder_path)):
        raise HTTPException(status_code=400, detail="Unsafe OUTPUT_FOLDER path")
    allowed_roots = _allowed_path_roots()
    under_root = False
    for root in allowed_roots:
        try:
            out_folder_path.relative_to(root)
            under_root = True
            break
        except ValueError:
            continue
    if not under_root:
        raise HTTPException(
            status_code=403,
            detail="OUTPUT_FOLDER must resolve under repo, INPUT_FOLDER, or OUTPUT_FOLDER",
        )
    out_folder_path.mkdir(parents=True, exist_ok=True)
    out_folder = str(out_folder_path)

    path = visualise_review_redaction_boxes(
        annotator,
        review_df=review_df,
        output_folder=out_folder,
        page_number=body.page_number,
        doc_base_name=body.doc_base_name,
        label_abbrev_chars=body.label_abbrev_chars,
    )
    if not path:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not produce overlay PNG (invalid image/boxes or write failed). "
                "Ensure boxes are valid and the image loads."
            ),
        )
    return AgentTaskResponse(
        status="completed",
        gradio_api_name="export_review_redaction_overlay",
        task="export_review_redaction_overlay",
        output_dir=out_folder,
        input_dir="",
        message="Redaction overlay PNG written",
        output_paths=[path],
    )


@router.post(
    "/export_review_page_ocr_visualisation",
    response_model=AgentTaskResponse,
    summary="export_review_page_ocr_visualisation (Gradio api_name)",
    description=(
        "Renders a per-page OCR visualisation using tools.file_redaction.visualise_ocr_words_bounding_boxes; "
        "writes under OUTPUT_FOLDER/review_ocr_visualisations/."
    ),
)
def post_export_review_page_ocr_visualisation(
    body: AgentExportReviewPageOcrVisualisationRequest,
    _: None = Depends(_optional_agent_api_key),
) -> AgentTaskResponse:
    from PIL import Image

    from tools.file_redaction import visualise_ocr_words_bounding_boxes

    img_path = _path_must_be_allowed_file(body.page_image_path)

    out_folder_path = Path(OUTPUT_FOLDER).expanduser().resolve()
    if not validate_path_safety(str(out_folder_path)):
        raise HTTPException(status_code=400, detail="Unsafe OUTPUT_FOLDER path")
    allowed_roots = _allowed_path_roots()
    under_root = False
    for root in allowed_roots:
        try:
            out_folder_path.relative_to(root)
            under_root = True
            break
        except ValueError:
            continue
    if not under_root:
        raise HTTPException(
            status_code=403,
            detail="OUTPUT_FOLDER must resolve under repo, INPUT_FOLDER, or OUTPUT_FOLDER",
        )
    out_folder_path.mkdir(parents=True, exist_ok=True)
    out_folder = str(out_folder_path)

    safe_base = str(body.doc_base_name or "review")
    image_name = f"{safe_base}_page{int(body.page_number)}.png"
    log_paths: list[str] = []
    try:
        log_paths = visualise_ocr_words_bounding_boxes(
            Image.open(img_path).convert("RGB"),
            body.ocr_results,
            image_name=image_name,
            output_folder=out_folder,
            visualisation_folder="review_ocr_visualisations",
            add_legend=True,
            log_files_output_paths=log_paths,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if not log_paths:
        raise HTTPException(
            status_code=500,
            detail="Could not produce OCR visualisation (invalid image/ocr_results or write failed).",
        )
    out_path = log_paths[-1]
    return AgentTaskResponse(
        status="completed",
        gradio_api_name="export_review_page_ocr_visualisation",
        task="export_review_page_ocr_visualisation",
        output_dir=out_folder,
        input_dir="",
        message="OCR visualisation written",
        output_paths=[out_path],
    )


def _gradio_only(api_name: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "gradio_api_name": api_name,
            "detail": detail,
            "hint": "Use the Gradio UI or gradio_client with this api_name.",
        },
    )


@router.post("/load_and_prepare_documents_or_data")
def post_load_and_prepare_documents_or_data() -> JSONResponse:
    return _gradio_only(
        "load_and_prepare_documents_or_data",
        "Preparation uses Gradio session state and prepare_image_or_pdf_with_efficient_ocr; no single CLI task.",
    )


@router.post("/apply_review_redactions")
def post_apply_review_redactions() -> JSONResponse:
    return _gradio_only(
        "apply_review_redactions",
        "Review PDF/annotation state is managed in the Gradio UI.",
    )


@router.post("/word_level_ocr_text_search")
def post_word_level_ocr_text_search() -> JSONResponse:
    return _gradio_only(
        "word_level_ocr_text_search",
        "Search uses in-memory OCR dataframes in the UI session.",
    )


@router.get("/operations")
def list_operations() -> dict[str, Any]:
    return {
        "gradio_api_names": list(GRADIO_API_NAMES),
        "routes": [
            {
                "gradio_api_name": "redact_document",
                "method": "POST",
                "path": "/agent/redact_document",
                "implementation": "cli_redact task redact",
            },
            {
                "gradio_api_name": "redact_data",
                "method": "POST",
                "path": "/agent/redact_data",
                "implementation": "cli_redact task redact",
            },
            {
                "gradio_api_name": "find_duplicate_pages",
                "method": "POST",
                "path": "/agent/find_duplicate_pages",
                "implementation": "cli_redact deduplicate pages",
            },
            {
                "gradio_api_name": "find_duplicate_tabular",
                "method": "POST",
                "path": "/agent/find_duplicate_tabular",
                "implementation": "cli_redact deduplicate tabular",
            },
            {
                "gradio_api_name": "summarise_document",
                "method": "POST",
                "path": "/agent/summarise_document",
                "implementation": "cli_redact task summarise",
            },
            {
                "gradio_api_name": "combine_review_pdfs",
                "method": "POST",
                "path": "/agent/combine_review_pdfs",
                "implementation": "cli_redact combine_review_pdfs",
            },
            {
                "gradio_api_name": "export_review_redaction_overlay",
                "method": "POST",
                "path": "/agent/export_review_redaction_overlay",
                "implementation": "visualise_review_redaction_boxes",
            },
            {
                "gradio_api_name": "export_review_page_ocr_visualisation",
                "method": "POST",
                "path": "/agent/export_review_page_ocr_visualisation",
                "implementation": "visualise_ocr_words_bounding_boxes",
            },
            {
                "gradio_api_name": "combine_review_csvs",
                "method": "POST",
                "path": "/agent/combine_review_csvs",
                "implementation": "helper merge_csv_files",
            },
            {
                "gradio_api_name": "load_and_prepare_documents_or_data",
                "method": "POST",
                "path": "/agent/load_and_prepare_documents_or_data",
                "implementation": "not_implemented_http",
            },
            {
                "gradio_api_name": "apply_review_redactions",
                "method": "POST",
                "path": "/agent/apply_review_redactions",
                "implementation": "not_implemented_http",
            },
            {
                "gradio_api_name": "word_level_ocr_text_search",
                "method": "POST",
                "path": "/agent/word_level_ocr_text_search",
                "implementation": "not_implemented_http",
            },
        ],
    }


@router.get("/health")
def agent_health() -> dict[str, str]:
    return {"status": "ok", "service": "agent"}
