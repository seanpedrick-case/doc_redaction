"""Pi agent Gradio examples aligned with the main app SHOW_EXAMPLES redaction demos."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pi_agent_config import is_hf_space_profile
from redaction_prompt import HF_DEFAULT_OCR


def _show_examples_from_env() -> bool:
    """True unless PI_GRADIO_SHOW_EXAMPLES or SHOW_PI_EXAMPLES is explicitly false."""
    for key in ("PI_GRADIO_SHOW_EXAMPLES", "SHOW_PI_EXAMPLES"):
        raw = os.environ.get(key)
        if raw is None:
            continue
        lowered = raw.strip().lower()
        if lowered in {"0", "false", "no"}:
            return False
        if lowered in {"1", "true", "yes"}:
            return True
    return True


SHOW_PI_EXAMPLES = _show_examples_from_env()


@dataclass(frozen=True)
class PiRedactionExample:
    label: str
    file_name: str
    instructions: str
    ocr_method: str
    pii_method: str = "Local"
    encourage_vlm_faces: bool = False
    encourage_vlm_signatures: bool = False
    page_range: str = "all"


def resolve_example_data_dir() -> Path | None:
    """Locate bundled example PDFs (repo checkout, PyPI package, or Docker layout)."""
    from bootstrap_pi_config import pi_repo_root_path

    workdir = pi_repo_root_path()
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        workdir / "doc_redaction" / "example_data",
        workdir / "example_data",
        repo_root / "doc_redaction" / "example_data",
        repo_root / "example_data",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def example_file_path(file_name: str) -> Path | None:
    root = resolve_example_data_dir()
    if root is None:
        return None
    path = (root / file_name).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    if not path.is_file():
        return None
    if _is_lfs_pointer(path):
        return None
    return path


def _is_lfs_pointer(path: Path) -> bool:
    try:
        first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except (OSError, IndexError):
        return False
    return first_line.startswith("version https://git-lfs.github.com/spec/v1")


def _catalog() -> tuple[PiRedactionExample, ...]:
    selectable_text_ocr = (
        HF_DEFAULT_OCR if is_hf_space_profile() else "Local model - selectable text"
    )
    # local_ocr = (
    #     HF_DEFAULT_OCR
    #     if is_hf_space_profile()
    #     else "Local OCR model - PDFs without selectable text"
    # )
    return (
        PiRedactionExample(
            label="Emails to a professor",
            file_name="example_of_emails_sent_to_a_professor_before_applying.pdf",
            ocr_method=selectable_text_ocr,
            pii_method="Local",
            instructions=(
                "- Any redaction box related to Dr Kornbluth should be removed\n"
                "- References to Dr Hyde, or Dr Hyde's lab should be redacted. Also any references to Lauren, or Lauren Lilley\n"
                "- All mentions of Universities and their names should be redacted\n"
            ),
        ),
        PiRedactionExample(
            label="Graduate cover letter",
            file_name="graduate-job-example-cover-letter.pdf",
            ocr_method=selectable_text_ocr,
            pii_method="Local",
            instructions=(
                "- Redact any names and titles, apart from Mr Wilson\n"
                "- Redact any organisation names\n"
                "- Redact any place names\n"
            ),
        ),
    )


def available_pi_examples() -> list[PiRedactionExample]:
    if not SHOW_PI_EXAMPLES:
        return []
    available: list[PiRedactionExample] = []
    for example in _catalog():
        if example_file_path(example.file_name) is not None:
            available.append(example)
    return available


def example_rows() -> tuple[list[list], list[str]]:
    """Return (gr.Examples rows, labels) for available demos."""
    rows: list[list] = []
    labels: list[str] = []
    for example in available_pi_examples():
        path = example_file_path(example.file_name)
        if path is None:
            continue
        rows.append(
            [
                str(path),
                example.instructions,
                example.page_range,
                example.ocr_method,
                example.pii_method,
                example.encourage_vlm_faces,
                example.encourage_vlm_signatures,
            ]
        )
        labels.append(example.label)
    return rows, labels


def gradio_example_allowed_paths() -> list[str]:
    root = resolve_example_data_dir()
    if root is None:
        return []
    return [str(root)]


def examples_status_markdown() -> str:
    """Human-readable status for the UI when examples are missing or disabled."""
    if not SHOW_PI_EXAMPLES:
        return (
            "_Examples are disabled. Set Space variable "
            "`PI_GRADIO_SHOW_EXAMPLES=true` (or `SHOW_PI_EXAMPLES=true`) and restart._"
        )
    root = resolve_example_data_dir()
    if root is None:
        return (
            "_Example PDFs not found — expected under "
            "`doc_redaction/example_data/` in the Space image._"
        )
    available = available_pi_examples()
    if not available:
        return (
            f"_Example PDFs not found under `{root}`. "
            "Rebuild the Space after syncing example files from the monorepo._"
        )
    names = ", ".join(f"`{ex.file_name}`" for ex in available)
    return f"_Examples loaded from `{root}`: {names}_"
