"""Gradio agent examples for the main app (aligned with agent-redact ``pi_examples``)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from tools.config import LOCAL_PII_OPTION, SELECTABLE_TEXT_EXTRACT_OPTION
from tools.example_data_paths import is_lfs_pointer


def _show_harness_examples() -> bool:
    for key in (
        "MAIN_APP_AGENT_EXAMPLES",
        "SHOW_AGENT_HARNESS_EXAMPLES",
        "SHOW_EXAMPLES",
    ):
        raw = os.environ.get(key)
        if raw is None:
            continue
        lowered = raw.strip().lower()
        if lowered in {"0", "false", "no"}:
            return False
        if lowered in {"1", "true", "yes"}:
            return True
    return True


@dataclass(frozen=True)
class HarnessRedactionExample:
    label: str
    file_name: str
    instructions: str
    text_extract_method: str = SELECTABLE_TEXT_EXTRACT_OPTION
    pii_method: str = LOCAL_PII_OPTION
    page_min: int = 0
    page_max: int = 0


def _catalog() -> tuple[HarnessRedactionExample, ...]:
    return (
        HarnessRedactionExample(
            label="Emails to a professor",
            file_name="example_of_emails_sent_to_a_professor_before_applying.pdf",
            instructions=(
                "- Any redaction box related to Dr Kornbluth should be removed\n"
                "- References to Dr Hyde, or Dr Hyde's lab should be redacted. Also any "
                "references to Lauren, or Lauren Lilley\n"
                "- All mentions of Universities and their names should be redacted\n"
            ),
        ),
        HarnessRedactionExample(
            label="Graduate cover letter",
            file_name="graduate-job-example-cover-letter.pdf",
            instructions=(
                "- Redact any names and titles, apart from Mr Wilson\n"
                "- Redact any organisation names\n"
                "- Redact any place names\n"
            ),
        ),
    )


def _example_path(file_name: str) -> str | None:
    from tools.example_data_paths import bundled_example_candidate_path

    path = bundled_example_candidate_path(file_name)
    if path is None or not path.is_file() or is_lfs_pointer(path):
        return None
    return str(path)


def available_harness_examples() -> list[HarnessRedactionExample]:
    if not _show_harness_examples():
        return []
    out: list[HarnessRedactionExample] = []
    for example in _catalog():
        if _example_path(example.file_name):
            out.append(example)
    return out


def harness_example_rows(
    *,
    local_entities: list[str],
    comprehend_entities: list[str],
    llm_entities: list[str],
) -> tuple[list[list], list[str]]:
    """Build ``gr.Examples`` rows for the Redact tab agent UI."""
    rows: list[list] = []
    labels: list[str] = []
    for example in available_harness_examples():
        path = _example_path(example.file_name)
        if path is None:
            continue
        rows.append(
            [
                path,
                example.instructions,
                example.text_extract_method,
                "",  # local_ocr_model_radio (unused for selectable-text demos)
                example.pii_method,
                list(local_entities),
                list(comprehend_entities),
                list(llm_entities),
                [],
                [],
                [],
                example.page_min,
                example.page_max,
            ]
        )
        labels.append(example.label)
    return rows, labels


def harness_examples_status_md() -> str:
    if not _show_harness_examples():
        return "_Agent examples are disabled (`MAIN_APP_AGENT_EXAMPLES=false`)._"
    available = available_harness_examples()
    if not available:
        return (
            "_Example PDFs not found — add files under `example_data/` "
            "(see `tools/example_data_paths.py`)._"
        )
    names = ", ".join(f"`{ex.file_name}`" for ex in available)
    return f"_Agent examples: {names}_"
