"""Build Pi redaction task prompts from the partnership example template."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from pi_agent_config import is_hf_space_profile
from session_workspace import WORKSPACE_BASE_DIR

UPLOAD_ROOT = Path(os.environ.get("PI_UPLOAD_ROOT", "/tmp/gradio")).resolve()
_SAFE_UPLOAD_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")

REPO_ROOT = Path(os.environ.get("PI_WORKDIR", "/workspace/doc_redaction"))
TEMPLATE_PATH = REPO_ROOT / "skills" / "Example prompt partnership.txt"
WORKSPACE_DIR = WORKSPACE_BASE_DIR

HF_DEFAULT_OCR = "Local model - selectable text"
HF_DEFAULT_PII = "Local"
HF_DEFAULT_GRADIO_URL = "https://seanpedrickcase-document-redaction.hf.space"

# Used only when PI_DEFAULT_OCR_METHOD / PI_DEFAULT_PII_METHOD are unset (local-docker profile).
_FALLBACK_LOCAL_OCR = "hybrid-paddle-inference-server"
_FALLBACK_LOCAL_PII = "Local"


def _env_default(key: str, *, hf_default: str, local_fallback: str) -> str:
    """Resolve Pi redaction defaults from env (e.g. config/pi_agent.env) with profile fallbacks."""
    explicit = (os.environ.get(key) or "").strip()
    if explicit:
        return explicit
    if is_hf_space_profile():
        return hf_default
    return local_fallback


DEFAULT_OCR_METHOD = _env_default(
    "PI_DEFAULT_OCR_METHOD",
    hf_default=HF_DEFAULT_OCR,
    local_fallback=_FALLBACK_LOCAL_OCR,
)
DEFAULT_PII_METHOD = _env_default(
    "PI_DEFAULT_PII_METHOD",
    hf_default=HF_DEFAULT_PII,
    local_fallback=_FALLBACK_LOCAL_PII,
)

OCR_METHOD_CHOICES: tuple[str, ...] = (
    "hybrid-paddle-inference-server",
    "hybrid-paddle-vlm",
    "Local model - selectable text",
    "Local OCR",
    "AWS Textract service - all PDF types",
    "tesseract",
    "paddle",
    "hybrid-paddle",
    "vlm",
    "inference-server",
)

PII_METHOD_CHOICES: tuple[str, ...] = (
    "Local",
    "AWS Comprehend",
    "LLM (AWS Bedrock)",
    "Local inference server",
    "Local transformers LLM",
    "Only extract text (no redaction)",
)

_DEFAULT_MAX_PAGES = 3000


def max_pages_limit() -> int:
    """
    Maximum PDF pages allowed for a Pi redaction task.

    Resolution order: ``PI_MAX_PAGES`` → ``MAX_PAGES`` → ``MAX_DOC_PAGES`` → 3000.
    """
    for key in ("PI_MAX_PAGES", "MAX_PAGES", "MAX_DOC_PAGES"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            value = int(raw)
            if value < 1:
                raise ValueError(f"{key} must be a positive integer.")
            return value
    return _DEFAULT_MAX_PAGES


def pages_to_process_count(page_range: str, total_pages: int) -> int:
    """Return how many pages ``page_range`` selects from a ``total_pages`` PDF."""
    if total_pages < 1:
        raise ValueError("PDF has no pages.")

    text = (page_range or "all").strip().lower()
    if not text or text == "all":
        return total_pages

    if "-" in text:
        start_text, end_text = text.split("-", 1)
        try:
            start = int(start_text.strip())
            end = int(end_text.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid page range: {page_range!r}") from exc
        if start < 1 or end < start:
            raise ValueError(f"Invalid page range: {page_range!r}")
        if end > total_pages:
            raise ValueError(
                f"Page range {page_range!r} exceeds document length "
                f"({total_pages} pages)."
            )
        return end - start + 1

    try:
        page = int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid page range: {page_range!r}") from exc
    if page < 1 or page > total_pages:
        raise ValueError(
            f"Page {page} is out of range (document has {total_pages} pages)."
        )
    return 1


def pdf_page_count(file_path: str | Path) -> int:
    import pymupdf

    path = Path(file_path)
    with pymupdf.open(path) as doc:
        return int(doc.page_count)


def validate_pdf_page_limit(
    file_path: str | Path,
    *,
    page_range: str = "all",
    max_pages: int | None = None,
) -> None:
    """Reject PDFs whose selected page count exceeds ``max_pages_limit()``."""
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return

    limit = max_pages if max_pages is not None else max_pages_limit()
    try:
        total = pdf_page_count(path)
    except Exception as exc:
        raise ValueError(f"Could not read PDF page count for {path.name}.") from exc

    count = pages_to_process_count(page_range, total)
    if count > limit:
        scope = page_range.strip() or "all"
        raise ValueError(
            f"Number of pages to process ({count}) exceeds the maximum allowed "
            f"({limit}). Submit a smaller document or narrow the page range "
            f"({scope!r})."
        )


@dataclass(frozen=True)
class RedactionTaskSettings:
    ocr_method: str = DEFAULT_OCR_METHOD
    pii_method: str = DEFAULT_PII_METHOD
    encourage_vlm_faces: bool = False if is_hf_space_profile() else True
    encourage_vlm_signatures: bool = False if is_hf_space_profile() else True

    @classmethod
    def hf_space_defaults(cls) -> RedactionTaskSettings:
        return cls(
            ocr_method=HF_DEFAULT_OCR,
            pii_method=HF_DEFAULT_PII,
            encourage_vlm_faces=False,
            encourage_vlm_signatures=False,
        )

    @classmethod
    def from_ui(
        cls,
        ocr_method: str,
        pii_method: str,
        encourage_vlm_faces: bool,
        encourage_vlm_signatures: bool,
    ) -> RedactionTaskSettings:
        ocr = (ocr_method or DEFAULT_OCR_METHOD).strip()
        pii = (pii_method or DEFAULT_PII_METHOD).strip()
        if ocr not in OCR_METHOD_CHOICES:
            ocr = DEFAULT_OCR_METHOD
        if pii not in PII_METHOD_CHOICES:
            pii = DEFAULT_PII_METHOD
        return cls(
            ocr_method=ocr,
            pii_method=pii,
            encourage_vlm_faces=bool(encourage_vlm_faces),
            encourage_vlm_signatures=bool(encourage_vlm_signatures),
        )


def _default_gradio_url() -> str:
    if is_hf_space_profile():
        return os.environ.get("DOC_REDACTION_GRADIO_URL", HF_DEFAULT_GRADIO_URL)
    return os.environ.get("DOC_REDACTION_GRADIO_URL", "http://redaction-app-llama:7860")


def _default_vlm_base_url() -> str:
    return os.environ.get("PI_VLM_BASE_URL", "http://llama-inference:8080")


def _default_vlm_model() -> str:
    return os.environ.get("PI_VLM_MODEL", "unsloth/Qwen3.6-27B-MTP-GGUF")


def load_template(path: Path | None = None) -> str:
    template_file = path or TEMPLATE_PATH
    if not template_file.is_file():
        raise FileNotFoundError(f"Prompt template not found: {template_file}")
    return template_file.read_text(encoding="utf-8")


def format_user_requirements(instructions: str) -> str:
    lines: list[str] = []
    for raw in instructions.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        if not line.startswith("-"):
            line = f"- {line}"
        lines.append(line)
    return "\n".join(lines)


def replace_user_requirements_section(template: str, instructions: str) -> str:
    marker = "## User redaction requirements"
    idx = template.find(marker)
    formatted = format_user_requirements(instructions)
    if idx == -1:
        return f"{template.rstrip()}\n\n{marker} (authoritative for this task)\n\n{formatted}\n"
    head = template[:idx]
    return f"{head}{marker} (authoritative for this task)\n\n{formatted}\n"


def _is_textract_ocr_method(ocr_method: str) -> bool:
    lowered = ocr_method.casefold()
    return "textract" in lowered or lowered in {"textract", "aws textract"}


def build_vlm_faces_guidance(encourage: bool) -> str:
    if is_hf_space_profile():
        return (
            "Pass 2 VLM and CUSTOM_VLM_FACES are not available on this deployment. "
            "Do not pass CUSTOM_VLM_FACES or request face detection."
        )
    if encourage:
        return (
            "If the user asks to redact faces, then pass the entity CUSTOM_VLM_FACES "
            "in the initial redaction entity selection"
        )
    return (
        "Do not pass CUSTOM_VLM_FACES in the initial redaction entity list unless "
        "the user explicitly asks to redact faces"
    )


def build_vlm_signature_guidance(encourage: bool, ocr_method: str) -> str:
    if is_hf_space_profile():
        return (
            "Pass 2 VLM and CUSTOM_VLM_SIGNATURE are not available on this deployment. "
            "Do not pass CUSTOM_VLM_SIGNATURE or request signature detection."
        )
    if encourage:
        if _is_textract_ocr_method(ocr_method):
            return (
                "If the user asked to redact signatures, then pass the CUSTOM_VLM_SIGNATURE "
                "entity in the initial redaction entity selection, unless the text extraction "
                "option is AWS Textract, in which case the handwrite_signature_textbox parameter "
                "for the doc_redact endpoint should include 'Extract signatures'"
            )
        return (
            "If the user asked to redact signatures, then pass the CUSTOM_VLM_SIGNATURE "
            "entity in the initial redaction entity selection"
        )
    return (
        "Do not pass CUSTOM_VLM_SIGNATURE in the initial redaction entity list unless "
        "the user explicitly asks to redact signatures"
    )


def build_remote_backend_guidance(
    *,
    gradio_url: str,
    output_base: str,
    workspace_root: str,
) -> str:
    if not is_hf_space_profile():
        return ""
    return (
        f"- **Remote redaction backend:** the doc_redaction app runs at `{gradio_url}` "
        "(private Hugging Face Space). Use **`gradio_client` only** — upload local files "
        f"with `handle_file()` from `{workspace_root.rstrip('/')}/`. "
        "**Do not** call `/agent/*` routes or use server-side paths from the redaction container.\n"
        f"- Download all `/doc_redact` and `/review_apply` outputs via "
        f"`{gradio_url.rstrip('/')}/gradio_api/file=…` with "
        "`Authorization: Bearer $HF_TOKEN` into `{output_base}` (create subdirs as needed).\n"
        "- Run **`verify_redaction_coverage`** locally on downloaded CSV/PDF paths in this "
        "workspace (pandas/PyMuPDF), not via Agent API.\n"
        "- **Pass 2 VLM is not available** — do not call a VLM endpoint or use "
        "`CUSTOM_VLM_FACES` / `CUSTOM_VLM_SIGNATURE` entities.\n"
        "- Helper module: `agent-redact/pi/remote_redaction.py` (`make_redaction_client`, "
        "`download_gradio_files`)."
    ).format(output_base=output_base.rstrip("/") + "/")


def _sanitize_upload_filename(name: str) -> str:
    safe = Path(name).name.strip()
    if not safe or safe in {".", ".."}:
        raise ValueError("Uploaded file has an invalid name.")
    if not _SAFE_UPLOAD_FILENAME_RE.fullmatch(safe):
        raise ValueError("Uploaded file has an invalid name.")
    return safe


def _resolve_and_validate_upload_path(upload_path: str | Path) -> Path:
    if not isinstance(upload_path, (str, Path)):
        raise ValueError("Uploaded file path has an invalid type.")
    if not str(upload_path).strip():
        raise ValueError("Uploaded file path is empty.")

    root = UPLOAD_ROOT.resolve(strict=True)
    raw_path = Path(upload_path).expanduser()
    try:
        source = raw_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Uploaded file not found: {raw_path}") from exc

    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Uploaded file path resolves outside allowed upload root: {source}"
        ) from exc
    if not source.is_file():
        raise FileNotFoundError(f"Uploaded file not found: {source}")
    if source.is_symlink():
        raise ValueError(f"Symlink uploads are not allowed: {source}")
    return source


def _resolve_and_validate_workspace_dir(workspace_dir: Path | None) -> Path:
    if workspace_dir is not None and not isinstance(workspace_dir, Path):
        raise ValueError("Workspace path has an invalid type.")
    base_root = WORKSPACE_DIR.resolve()
    candidate = (
        workspace_dir if workspace_dir is not None else WORKSPACE_DIR
    ).resolve()
    try:
        candidate.relative_to(base_root)
    except ValueError as exc:
        raise ValueError(
            f"Workspace path resolves outside allowed workspace root: {candidate}"
        ) from exc
    return candidate


def copy_upload_to_workspace(
    upload_path: str | Path,
    *,
    workspace_dir: Path | None = None,
) -> Path:
    source = _resolve_and_validate_upload_path(upload_path)
    if not source.is_file():
        raise FileNotFoundError(f"Uploaded file not found: {source}")
    workspace_root = _resolve_and_validate_workspace_dir(workspace_dir)
    workspace_root.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_upload_filename(source.name)
    dest = (workspace_root / safe_name).resolve()
    try:
        dest.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(f"Destination path is outside workspace: {dest}") from exc
    if source == dest:
        return dest
    # copyfile only: copy2/copystat raises EPERM when overwriting on Docker Desktop bind mounts.
    shutil.copyfile(source, dest)
    return dest


def build_redaction_prompt(
    file_name: str,
    user_instructions: str,
    *,
    page_range: str = "all",
    template: str | None = None,
    settings: RedactionTaskSettings | None = None,
    workspace_dir: Path | None = None,
) -> str:
    if not file_name.strip():
        raise ValueError("A document file name is required.")
    if not user_instructions.strip():
        raise ValueError("Redaction requirements are required (use bullet points).")

    task_settings = settings or RedactionTaskSettings()
    workspace_root = (workspace_dir or WORKSPACE_DIR).resolve()
    file_name = Path(file_name).name
    input_path = f"{workspace_root.as_posix().rstrip('/')}/{file_name}"
    output_base = f"{workspace_root.as_posix().rstrip('/')}/redact/{file_name}/"

    text = template if template is not None else load_template()
    remote_guidance = build_remote_backend_guidance(
        gradio_url=_default_gradio_url(),
        output_base=output_base,
        workspace_root=workspace_root.as_posix(),
    )
    replacements = {
        "{FILE_NAME}": file_name,
        "{INPUT_PATH}": input_path,
        "{OUTPUT_BASE}": output_base,
        "{GRADIO_URL}": _default_gradio_url(),
        "{PAGE_RANGE}": page_range.strip() or "all",
        "{VLM_BASE_URL}": _default_vlm_base_url(),
        "{VLM_MODEL}": _default_vlm_model(),
        "{DEFAULT_OCR_METHOD}": task_settings.ocr_method,
        "{DEFAULT_PII_METHOD}": task_settings.pii_method,
        "{VLM_FACES_GUIDANCE}": build_vlm_faces_guidance(
            task_settings.encourage_vlm_faces
        ),
        "{VLM_SIGNATURE_GUIDANCE}": build_vlm_signature_guidance(
            task_settings.encourage_vlm_signatures,
            task_settings.ocr_method,
        ),
    }
    if remote_guidance:
        replacements["{REMOTE_BACKEND_GUIDANCE}"] = remote_guidance
    else:
        text = text.replace("- {REMOTE_BACKEND_GUIDANCE}\n", "")
    for key, value in replacements.items():
        text = text.replace(key, value)

    return replace_user_requirements_section(text, user_instructions)


def prepare_redaction_task(
    upload_path: str | Path | None,
    user_instructions: str,
    *,
    page_range: str = "all",
    settings: RedactionTaskSettings | None = None,
    workspace_dir: Path | None = None,
) -> tuple[str, str]:
    """
    Copy upload into workspace and return (file_name, full_prompt).
    """
    if upload_path is None:
        raise ValueError("Please upload a document.")
    root = _resolve_and_validate_workspace_dir(workspace_dir)
    validate_pdf_page_limit(upload_path, page_range=page_range)
    dest = copy_upload_to_workspace(upload_path, workspace_dir=root)
    prompt = build_redaction_prompt(
        dest.name,
        user_instructions,
        page_range=page_range,
        settings=settings,
        workspace_dir=root,
    )
    return dest.name, prompt
