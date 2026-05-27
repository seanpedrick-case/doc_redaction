"""Build Pi redaction task prompts from the partnership example template."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(os.environ.get("PI_WORKDIR", "/workspace/doc_redaction"))
TEMPLATE_PATH = REPO_ROOT / "skills" / "Example prompt partnership.txt"
WORKSPACE_DIR = Path(os.environ.get("PI_WORKSPACE_DIR", "/home/user/app/workspace"))
UPLOAD_ROOT = Path(os.environ.get("PI_UPLOAD_ROOT", "/tmp/gradio")).resolve()


def _default_gradio_url() -> str:
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


def _resolve_and_validate_upload_path(upload_path: str | Path) -> Path:
    source = Path(upload_path).resolve()
    try:
        source.relative_to(UPLOAD_ROOT)
    except ValueError as exc:
        raise ValueError(f"Uploaded file path is outside allowed upload root: {source}") from exc
    return source


def copy_upload_to_workspace(upload_path: str | Path) -> Path:
    source = _resolve_and_validate_upload_path(upload_path)
    if not source.is_file():
        raise FileNotFoundError(f"Uploaded file not found: {source}")
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    dest = (WORKSPACE_DIR / source.name).resolve()
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
) -> str:
    if not file_name.strip():
        raise ValueError("A document file name is required.")
    if not user_instructions.strip():
        raise ValueError("Redaction requirements are required (use bullet points).")

    file_name = Path(file_name).name
    input_path = f"{WORKSPACE_DIR.as_posix().rstrip('/')}/{file_name}"
    output_base = f"{WORKSPACE_DIR.as_posix().rstrip('/')}/redact/{file_name}/"

    text = template if template is not None else load_template()
    replacements = {
        "{FILE_NAME}": file_name,
        "{INPUT_PATH}": input_path,
        "{OUTPUT_BASE}": output_base,
        "{GRADIO_URL}": _default_gradio_url(),
        "{PAGE_RANGE}": page_range.strip() or "all",
        "{VLM_BASE_URL}": _default_vlm_base_url(),
        "{VLM_MODEL}": _default_vlm_model(),
    }
    for key, value in replacements.items():
        text = text.replace(key, value)

    return replace_user_requirements_section(text, user_instructions)


def prepare_redaction_task(
    upload_path: str | Path | None,
    user_instructions: str,
    *,
    page_range: str = "all",
) -> tuple[str, str]:
    """
    Copy upload into workspace and return (file_name, full_prompt).
    """
    if upload_path is None:
        raise ValueError("Please upload a document.")
    dest = copy_upload_to_workspace(upload_path)
    prompt = build_redaction_prompt(
        dest.name,
        user_instructions,
        page_range=page_range,
    )
    return dest.name, prompt
