"""Build Pi redaction task prompts from the partnership example template."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from pi_agent_config import is_aws_ecs_profile, is_hf_space_profile
from session_workspace import workspace_base_dir


def upload_root() -> Path:
    """Gradio upload directory (created by ``bootstrap_pi_config.ensure_pi_upload_root``)."""
    raw = (os.environ.get("PI_UPLOAD_ROOT") or "").strip()
    if not raw:
        from bootstrap_pi_config import ensure_pi_upload_root

        raw = ensure_pi_upload_root(pi_repo_root())
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


_SAFE_UPLOAD_FILENAME_MAX_BYTES = 255
# Path separators, nulls, and characters unsafe on common filesystems — not general punctuation.
_UNSAFE_UPLOAD_FILENAME_CHARS_RE = re.compile(r'[\x00-\x1f<>:"|?*\\/]')


def _truncate_upload_filename(
    name: str, *, max_bytes: int = _SAFE_UPLOAD_FILENAME_MAX_BYTES
) -> str:
    encoded = name.encode("utf-8")
    if len(encoded) <= max_bytes:
        return name
    stem, suffix = os.path.splitext(name)
    suffix_bytes = suffix.encode("utf-8")
    max_stem_bytes = max(1, max_bytes - len(suffix_bytes))
    while stem and len(stem.encode("utf-8")) > max_stem_bytes:
        stem = stem[:-1]
    if not stem:
        stem = "file"
    return stem + suffix


def _split_upload_basename(name: str) -> tuple[str, str]:
    """Split an upload basename into stem and extension (handles ``.pdf`` on Windows)."""
    if re.fullmatch(r"\.[^./\\]+", name):
        return "", name
    path = Path(name)
    return path.stem, path.suffix


def _workspace_filename_from_upload(name: str) -> tuple[str, str, bool]:
    """
    Derive a workspace-safe basename, changing the name only when required for security.

    Returns ``(original_basename, workspace_basename, renamed)``.
    """
    original = Path(name).name.strip()
    if not original or original in {".", ".."}:
        raise ValueError("Uploaded file has an invalid name.")
    if "\x00" in original or "/" in original or "\\" in original:
        raise ValueError("Uploaded file has an invalid name.")

    stem, suffix = _split_upload_basename(original)
    safe_stem = _UNSAFE_UPLOAD_FILENAME_CHARS_RE.sub("_", stem)
    safe_suffix = _UNSAFE_UPLOAD_FILENAME_CHARS_RE.sub("_", suffix)
    safe_stem = safe_stem.strip(". ")
    if not safe_stem:
        safe_stem = "file"
    safe_name = _truncate_upload_filename(safe_stem + safe_suffix)
    return original, safe_name, safe_name != original


_PARTNERSHIP_TEMPLATE = Path("skills") / "Example prompt partnership.txt"


def _workspace_root() -> Path:
    return workspace_base_dir()


def pi_repo_root() -> Path:
    """Monorepo checkout root (skills/, config/). Set via :func:`bootstrap_pi_config.ensure_pi_workdir`."""
    from bootstrap_pi_config import pi_repo_root_path

    return pi_repo_root_path()


def partnership_template_path() -> Path:
    from pi_workspace_skills import partnership_template_in_workspace

    synced = partnership_template_in_workspace()
    if synced is not None:
        return synced
    return pi_repo_root() / _PARTNERSHIP_TEMPLATE


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


def doc_redaction_gradio_url() -> str:
    """
    Base URL of the doc_redaction Gradio app used for ``/doc_redact`` and review APIs.

    Set ``DOC_REDACTION_GRADIO_URL`` in ``config/pi_agent.env`` (or the process environment).
    Reads the environment on each call so runtime overrides apply before ``tools.config``
    is imported (e.g. HF Space Docker ``ENV``, tests, and late ``load_dotenv``).
    """
    raw = (os.environ.get("DOC_REDACTION_GRADIO_URL") or "").strip().rstrip("/")
    if raw:
        return raw
    try:
        from tools.config import DOC_REDACTION_GRADIO_URL

        return str(DOC_REDACTION_GRADIO_URL).strip().rstrip("/")
    except ImportError:
        return (
            HF_DEFAULT_GRADIO_URL if is_hf_space_profile() else "http://127.0.0.1:7860"
        )


def _default_gradio_url() -> str:
    """Back-compat alias for prompt template substitution."""
    return doc_redaction_gradio_url()


def _default_vlm_base_url() -> str:
    return os.environ.get("PI_VLM_BASE_URL", "http://llama-inference:8080")


def _default_vlm_model() -> str:
    return os.environ.get("PI_VLM_MODEL", "unsloth/Qwen3.6-27B-MTP-GGUF")


def load_template(path: Path | None = None) -> str:
    template_file = path or partnership_template_path()
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


def build_local_redaction_client_guidance(
    *,
    gradio_url: str,
    output_base: str,
    workspace_root: str = "",
) -> str:
    """Pi agent and doc_redaction on the same host (local dev / shared Docker volumes)."""
    output_redact = f"{output_base.rstrip('/')}/output_redact/"
    helpers = (
        f"{workspace_root.rstrip('/')}/.pi/helpers/remote_redaction.py"
        if workspace_root.strip()
        else "`.pi/helpers/remote_redaction.py` (under `PI_WORKSPACE_DIR`)"
    )
    doc_output_hint = ""
    try:
        from tools.config import OUTPUT_FOLDER, SESSION_OUTPUT_FOLDER

        doc_output_hint = (
            f"- **doc_redaction writes to** `{OUTPUT_FOLDER}`"
            + (
                " (per-user subfolders when `SESSION_OUTPUT_FOLDER=True`). "
                if SESSION_OUTPUT_FOLDER
                else ". "
            )
            + "Do **not** pass a Pi workspace path as `output_dir` — the server only "
            "accepts directories under that folder.\n"
        )
    except ImportError:
        doc_output_hint = (
            "- Do **not** pass a Pi workspace path as `/doc_redact` `output_dir` — "
            "the server restricts `output_dir` to its own `OUTPUT_FOLDER`.\n"
        )
    return (
        f"- **Local doc_redaction backend:** `{gradio_url}` (same machine as this workspace).\n"
        f"{doc_output_hint}"
        "- Do not pass `CUSTOM_FUZZY` in `redact_entities` on `/doc_redact` unless the user explicitly requests fuzzy matching; it can be very CPU/RAM intensive and may return an empty path list even when the job completes. Use `CUSTOM` with an explicit `deny_list` on `/doc_redact`, or use `/redact_document` with `max_fuzzy_spelling_mistakes_num > 0` for fuzzy matching.\n"
        f"- Call **`/doc_redact`** (omit `output_dir` or leave it empty), then copy artifacts "
        f"into `{output_redact}` with `remote_redaction.resolve_redaction_output_paths` "
        f"and `fetch_redaction_files`.\n"
        "- When the API returns **Windows paths** (`C:\\\\...`) or paths under "
        "`workspace/.gradio_uploads/`, **copy from disk** with `shutil.copy2` — do not "
        "assume `gradio_api/file=` works (403 until allowed_paths includes that folder).\n"
        "- Path walkers must accept Windows drive paths, not only strings starting with `/`.\n"
        f"- Use `{helpers}`: `extract_server_paths(result)` "
        "then `fetch_redaction_files(paths, dest_dir)` (local copy, then HTTP fallback).\n"
    )


def build_hf_space_backend_guidance(
    *,
    gradio_url: str,
    output_base: str,
    workspace_root: str,
) -> str:
    helpers = f"{workspace_root.rstrip('/')}/.pi/helpers/remote_redaction.py"
    return (
        f"- **Remote redaction backend (authoritative URL):** `{gradio_url}` **only**. "
        "This Pi Space orchestrates a separate private doc_redaction Hugging Face Space "
        "over HTTPS.\n"
        "- **Read `/skill:hf-space-deployment` first** — it overrides Docker/local URLs "
        "(`host.docker.internal`, `localhost`, `redaction:7861`, internal service names) "
        "that appear in generic skills for local-docker or AWS ECS.\n"
        "- **Do not** probe alternate hosts, rewrite `{helpers}`, or hand-roll a new "
        "Gradio client script. Import `make_redaction_client`, `fetch_redaction_files`, "
        "and `resolve_redaction_output_paths` from that helper (`HF_TOKEN` is already in "
        "the Pi subprocess environment).\n"
        "- Use **`gradio_client` only** — upload local files with `handle_file()` from "
        f"`{workspace_root.rstrip('/')}/`. **Do not** call `/agent/*` routes or use "
        "server-side paths from the redaction container.\n"
        f"- Download all `/doc_redact` and `/review_apply` outputs via "
        f"`{gradio_url.rstrip('/')}/gradio_api/file=…` with "
        "`Authorization: Bearer $HF_TOKEN` into `{output_base}` (create subdirs as needed).\n"
        "- On Hugging Face rate limits (`TooManyRequestsError`), wait and retry the **same** "
        "URL via the helper — do not switch to another host.\n"
        "- Do not pass `CUSTOM_FUZZY` in `redact_entities` on `/doc_redact` unless the user explicitly requests fuzzy matching; it can be very CPU/RAM intensive and may return an empty path list even when the job completes. Use `CUSTOM` with an explicit `deny_list` on `/doc_redact`, or use `/redact_document` with `max_fuzzy_spelling_mistakes_num > 0` for fuzzy matching.\n"
        "- Run **`verify_redaction_coverage`** locally on downloaded CSV/PDF paths in this "
        "workspace (pandas/PyMuPDF), not via Agent API.\n"
        "- **Pass 2 VLM is not available** — do not call a VLM endpoint or use "
        "`CUSTOM_VLM_FACES` / `CUSTOM_VLM_SIGNATURE` entities.\n"
        "- **User-facing updates:** write progress and reasoning as normal assistant text. "
        "Do not put commentary in bash `#` comments — the UI shows those as tool lines.\n"
        f"- Helper module: `{helpers}`."
    ).format(output_base=output_base.rstrip("/") + "/", helpers=helpers)


def build_split_container_redaction_guidance(
    *,
    gradio_url: str,
    output_base: str,
    workspace_root: str,
) -> str:
    """AWS ECS (and similar): Pi agent and doc_redaction are separate containers."""
    output_redact = f"{output_base.rstrip('/')}/output_redact/"
    helpers = f"{workspace_root.rstrip('/')}/.pi/helpers/remote_redaction.py"
    return (
        f"- **Split-container redaction backend:** doc_redaction runs at `{gradio_url}` "
        "(separate service from this Pi agent). Use **`gradio_client` only**.\n"
        f"- **Deliverables belong in your session workspace:** `{output_redact}` "
        f"(and `{output_base.rstrip('/')}/review/output_review_final/` after apply). "
        "That is the **only** output tree you should populate for this task.\n"
        "- **Do not** search this container for redaction outputs: no `find /workspace`, "
        "no `ls /home/user/app/output`, no `import tools.config OUTPUT_FOLDER` on the Pi "
        "agent — those paths are on the **redaction service**, not here (or are a read-only "
        "git checkout without live run artifacts).\n"
        "- Do not pass `CUSTOM_FUZZY` in `redact_entities` on `/doc_redact` unless the user explicitly requests fuzzy matching; it can be very CPU/RAM intensive and may return an empty path list even when the job completes. Use `CUSTOM` with an explicit `deny_list` on `/doc_redact`, or use `/redact_document` with `max_fuzzy_spelling_mistakes_num > 0` for fuzzy matching.\n"
        f'- **Initial redaction:** `Client("{gradio_url}")` → `/doc_redact` with '
        f"`document_file=handle_file(\"<file under {workspace_root.rstrip('/')}/>\")`. "
        "Omit `output_dir` (server picks its own `OUTPUT_FOLDER`).\n"
        f"- **Collect paths:** `extract_server_paths(result)` from the predict tuple. "
        "When the path list is `[]`, parse the status `message` for embedded paths, or retry "
        "once — **do not** spend turns grepping the filesystem.\n"
        f'- **Download:** `fetch_redaction_files(paths, "{output_redact}")` from '
        f"`{helpers}` (HTTP `GET /gradio_api/file=` — no shared disk copy).\n"
        "- **`POST /agent/*`** only accepts paths on the **redaction server**. After "
        "download, run `verify_redaction_coverage` on CSV/PDF under your workspace, not with "
        "bare Agent API paths from this container.\n"
        f"- Helper module (inside workspace boundary): `{helpers}`."
    )


def build_remote_backend_guidance(
    *,
    gradio_url: str,
    output_base: str,
    workspace_root: str,
) -> str:
    if is_hf_space_profile():
        return build_hf_space_backend_guidance(
            gradio_url=gradio_url,
            output_base=output_base,
            workspace_root=workspace_root,
        )
    if is_aws_ecs_profile():
        return build_split_container_redaction_guidance(
            gradio_url=gradio_url,
            output_base=output_base,
            workspace_root=workspace_root,
        )
    return build_local_redaction_client_guidance(
        gradio_url=gradio_url,
        output_base=output_base,
        workspace_root=workspace_root,
    )


def _resolve_and_validate_upload_path(upload_path: str | Path) -> Path:
    if not isinstance(upload_path, (str, Path)):
        raise ValueError("Uploaded file path has an invalid type.")
    if not str(upload_path).strip():
        raise ValueError("Uploaded file path is empty.")

    root = upload_root()
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
    base_root = _workspace_root().resolve()
    candidate = (
        workspace_dir if workspace_dir is not None else _workspace_root()
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
) -> tuple[Path, str | None]:
    """
    Copy upload into the session workspace.

    Returns ``(destination_path, original_basename)`` where ``original_basename`` is
    set only when the file was renamed for path safety.
    """
    source = _resolve_and_validate_upload_path(upload_path)
    if not source.is_file():
        raise FileNotFoundError(f"Uploaded file not found: {source}")
    workspace_root = _resolve_and_validate_workspace_dir(workspace_dir)
    workspace_root.mkdir(parents=True, exist_ok=True)
    _original_name, safe_name, renamed = _workspace_filename_from_upload(source.name)
    dest = (workspace_root / safe_name).resolve()
    try:
        dest.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(f"Destination path is outside workspace: {dest}") from exc
    if source != dest:
        # copyfile only: copy2/copystat raises EPERM when overwriting on Docker Desktop bind mounts.
        shutil.copyfile(source, dest)
    return dest, (_original_name if renamed else None)


def _strip_long_document_section(template: str) -> str:
    """Remove the 100+ page operator block (keeps user requirements)."""
    marker = "## Specific rules for long documents"
    start = template.find(marker)
    if start == -1:
        return template
    end = template.find("## User redaction requirements", start)
    if end == -1:
        return template[:start].rstrip() + "\n\n"
    return template[:start].rstrip() + "\n\n" + template[end:]


def _include_long_document_rules(page_range: str, total_pages: int) -> bool:
    if total_pages <= 0:
        return False
    if total_pages >= 100:
        return True
    return pages_to_process_count(page_range or "all", total_pages) >= 100


def build_redaction_prompt(
    file_name: str,
    user_instructions: str,
    *,
    page_range: str = "all",
    template: str | None = None,
    settings: RedactionTaskSettings | None = None,
    workspace_dir: Path | None = None,
    total_pages: int = 0,
) -> str:
    if not file_name.strip():
        raise ValueError("A document file name is required.")
    if not user_instructions.strip():
        raise ValueError("Redaction requirements are required (use bullet points).")

    task_settings = settings or RedactionTaskSettings()
    workspace_root = (workspace_dir or _workspace_root()).resolve()
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

    if not _include_long_document_rules(page_range, total_pages):
        text = _strip_long_document_section(text)

    return replace_user_requirements_section(text, user_instructions)


def prepare_redaction_task(
    upload_path: str | Path | None,
    user_instructions: str,
    *,
    page_range: str = "all",
    settings: RedactionTaskSettings | None = None,
    workspace_dir: Path | None = None,
) -> tuple[str, str, str | None]:
    """
    Copy upload into workspace and return ``(file_name, full_prompt, renamed_from)``.

    ``renamed_from`` is the original upload basename when it was adjusted for path
    safety; otherwise ``None``.
    """
    if upload_path is None:
        raise ValueError("Please upload a document.")
    root = _resolve_and_validate_workspace_dir(workspace_dir)
    validate_pdf_page_limit(upload_path, page_range=page_range)
    dest, renamed_from = copy_upload_to_workspace(upload_path, workspace_dir=root)
    total_pages = 0
    if str(dest).lower().endswith(".pdf"):
        try:
            total_pages = pdf_page_count(dest)
        except (ValueError, OSError):
            total_pages = 0
    prompt = build_redaction_prompt(
        dest.name,
        user_instructions,
        page_range=page_range,
        settings=settings,
        workspace_dir=root,
        total_pages=total_pages,
    )
    return dest.name, prompt, renamed_from
