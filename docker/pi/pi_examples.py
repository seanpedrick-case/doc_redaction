"""Pi agent Gradio examples aligned with the main app SHOW_EXAMPLES redaction demos."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pi_agent_config import is_hf_space_profile
from redaction_prompt import HF_DEFAULT_OCR

SHOW_PI_EXAMPLES = os.environ.get("PI_GRADIO_SHOW_EXAMPLES", "true").lower() in {
    "1",
    "true",
    "yes",
}


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
    workdir = Path(os.environ.get("PI_WORKDIR", "/workspace/doc_redaction"))
    candidates = [
        workdir / "doc_redaction" / "example_data",
        workdir / "example_data",
        Path(__file__).resolve().parents[2] / "doc_redaction" / "example_data",
        Path(__file__).resolve().parents[2] / "example_data",
    ]
    try:
        import doc_redaction as pkg

        candidates.append(Path(pkg.__file__).resolve().parent / "example_data")
    except ImportError:
        pass

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
    return path if path.is_file() else None


def _catalog() -> tuple[PiRedactionExample, ...]:
    selectable_text_ocr = (
        HF_DEFAULT_OCR if is_hf_space_profile() else "Local model - selectable text"
    )
    local_ocr = (
        HF_DEFAULT_OCR
        if is_hf_space_profile()
        else "Local OCR model - PDFs without selectable text"
    )
    return (
        PiRedactionExample(
            label="PDF with selectable text redaction",
            file_name="example_of_emails_sent_to_a_professor_before_applying.pdf",
            ocr_method=selectable_text_ocr,
            instructions=(
                "- Redact all personal names, email addresses, and phone numbers\n"
                "- Keep university or department names visible unless they identify "
                "a specific individual\n"
                "- Use Local model - selectable text for text extraction\n"
                "- Use Local PII detection with the default entity set"
            ),
        ),
        PiRedactionExample(
            label="PDF redaction with custom entities (Titles, Person, Dates)",
            file_name="graduate-job-example-cover-letter.pdf",
            ocr_method=local_ocr,
            instructions=(
                "- Redact honorifics and titles (Mr, Mrs, Ms, Dr, Professor), "
                "person names, and dates\n"
                "- For the initial `/doc_redact` call, pass `redact_entities`: "
                "TITLES, PERSON, DATE_TIME\n"
                "- Do not redact organisation names, job descriptions, or addresses "
                "unless they contain a person's name\n"
                "- Use Local PII detection"
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
