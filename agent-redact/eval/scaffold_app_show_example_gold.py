#!/usr/bin/env python3
"""Scaffold eval/gold cases from app.py SHOW_EXAMPLES and SHOW_DIFFICULT_OCR_EXAMPLES."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
_REPO = _EVAL.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from derive_app_eval_instructions import (  # noqa: E402
    DERIVED_INSTRUCTIONS_VERSION,
    AppEvalRedactionSettings,
    derive_app_eval_instructions,
)
from gold_policy_scaffold import write_partial_policy_files  # noqa: E402

GOLD_ROOT = _EVAL / "gold"
# Resolved from repo root (see run_agent_regression.resolve_gold_case_pdf).
INPUT_PDF_REL_PREFIX = "doc_redaction/example_data"


def _parse_entity_list(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = (raw or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    return [str(parsed)]


def _default_local_entities() -> list[str]:
    from tools.config import CHOSEN_REDACT_ENTITIES

    return _parse_entity_list(CHOSEN_REDACT_ENTITIES)


def _default_comprehend_entities() -> list[str]:
    from tools.config import CHOSEN_COMPREHEND_ENTITIES

    return _parse_entity_list(CHOSEN_COMPREHEND_ENTITIES)


def _default_llm_entities() -> list[str]:
    from tools.config import CHOSEN_LLM_ENTITIES

    return _parse_entity_list(CHOSEN_LLM_ENTITIES)


def _write_case(
    *,
    case_id: str,
    description: str,
    input_pdf_basename: str,
    settings: AppEvalRedactionSettings,
    extra_instruction_lines: list[str] | None = None,
    manifest_extra: dict | None = None,
    notes: str = "",
) -> Path:
    case_dir = GOLD_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    instructions = derive_app_eval_instructions(settings)
    if extra_instruction_lines:
        extra = "\n".join(
            ln if ln.strip().startswith("-") else f"- {ln.strip()}"
            for ln in extra_instruction_lines
            if ln.strip()
        )
        instructions = f"{instructions}\n{extra}".strip()

    (case_dir / "instructions.txt").write_text(instructions + "\n", encoding="utf-8")

    manifest = {
        "case_id": case_id,
        "description": description,
        "input_pdf": f"{INPUT_PDF_REL_PREFIX}/{input_pdf_basename}",
        "derived_instructions_version": DERIVED_INSTRUCTIONS_VERSION,
        "eval_consumer": "app",
        "source": manifest_extra.get("source") if manifest_extra else "app.py",
        "app_example_label": (
            manifest_extra.get("app_example_label") if manifest_extra else ""
        ),
        "text_extract_method_ui": settings.text_extraction_method or None,
        "pii_method": settings.pii_identification_method or None,
        "local_ocr_method": (
            manifest_extra.get("local_ocr_method") if manifest_extra else None
        ),
        "page_min": settings.page_min or None,
        "page_max": settings.page_max or None,
        "text_extraction_only": settings.text_extraction_only,
        "notes": notes
        or "Scaffolded from app.py examples. Add ideal_review_file.csv, "
        "ocr_results_with_words.csv, and tune must_redact.txt after a blessed run.",
    }
    if manifest_extra:
        for key, val in manifest_extra.items():
            if key not in manifest and val is not None:
                manifest[key] = val
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    (case_dir / "app_eval_settings.json").write_text(
        json.dumps(settings.to_settings_dict(), indent=2), encoding="utf-8"
    )

    write_partial_policy_files(
        case_dir,
        settings,
        repo_root=_REPO,
        input_pdf_basename=input_pdf_basename,
    )

    readme = case_dir / "README.md"
    if not readme.is_file():
        readme.write_text(
            f"# {case_id}\n\n"
            f"{description}\n\n"
            "**You add:** `ideal_review_file.csv`, `ocr_results_with_words.csv`, "
            "and extend `must_redact.txt` (partial lists are pre-filled). "
            "Optional: `redacted.pdf`.\n\n"
            f"**Input PDF (from repo root):** `{manifest['input_pdf']}`\n",
            encoding="utf-8",
        )
    return case_dir


def main() -> int:
    local_ent = _default_local_entities()
    comp_ent = _default_comprehend_entities()
    llm_ent = _default_llm_entities()

    cases: list[Path] = []

    # --- SHOW_EXAMPLES (app.py ~1812) ---
    cases.append(
        _write_case(
            case_id="show_ex_professor_emails_selectable_text",
            description="Gradio SHOW_EXAMPLES: PDF with selectable text redaction.",
            input_pdf_basename="example_of_emails_sent_to_a_professor_before_applying.pdf",
            settings=AppEvalRedactionSettings(
                chosen_redact_entities=list(local_ent),
                chosen_redact_comprehend_entities=list(comp_ent),
                text_extraction_method="Local model - selectable text",
                pii_identification_method="Local",
            ),
            manifest_extra={
                "source": "SHOW_EXAMPLES",
                "app_example_label": "PDF with selectable text redaction",
            },
        )
    )

    cases.append(
        _write_case(
            case_id="show_ex_complaint_image_local_ocr",
            description="Gradio SHOW_EXAMPLES: Image redaction with local OCR.",
            input_pdf_basename="example_complaint_letter.jpg",
            settings=AppEvalRedactionSettings(
                chosen_redact_entities=list(local_ent),
                chosen_redact_comprehend_entities=list(comp_ent),
                text_extraction_method="Local OCR model - PDFs without selectable text",
                pii_identification_method="Local",
            ),
            manifest_extra={
                "source": "SHOW_EXAMPLES",
                "app_example_label": "Image redaction with local OCR",
            },
        )
    )

    cases.append(
        _write_case(
            case_id="show_ex_cover_letter_custom_entities",
            description="Gradio SHOW_EXAMPLES: Titles, Person, Dates entity sets.",
            input_pdf_basename="graduate-job-example-cover-letter.pdf",
            settings=AppEvalRedactionSettings(
                chosen_redact_entities=["TITLES", "PERSON", "DATE_TIME"],
                chosen_redact_comprehend_entities=["TITLES", "NAME", "DATE_TIME"],
                text_extraction_method="Local OCR model - PDFs without selectable text",
                pii_identification_method="Local",
            ),
            manifest_extra={
                "source": "SHOW_EXAMPLES",
                "app_example_label": "PDF redaction with custom entities (Titles, Person, Dates)",
            },
        )
    )

    cases.append(
        _write_case(
            case_id="show_ex_partnership_aws_textract_signatures",
            description="Gradio SHOW_EXAMPLES: AWS Textract + Comprehend with signature/handwriting (when SHOW_AWS_EXAMPLES).",
            input_pdf_basename="Partnership-Agreement-Toolkit_0_0.pdf",
            settings=AppEvalRedactionSettings(
                chosen_redact_entities=list(local_ent),
                chosen_redact_comprehend_entities=list(comp_ent),
                text_extraction_method="AWS Textract service - all PDF types",
                pii_identification_method="AWS Comprehend",
                handwrite_signature_checkbox=[
                    "Extract handwriting",
                    "Extract signatures",
                ],
            ),
            manifest_extra={
                "source": "SHOW_EXAMPLES",
                "app_example_label": "PDF redaction with AWS services and signature detection",
                "requires_show_aws_examples": True,
            },
        )
    )

    deny_terms = [
        "Sister",
        "Sister City",
        "Sister Cities",
        "Friendship City",
    ]
    cases.append(
        _write_case(
            case_id="show_ex_partnership_deny_list_whole_pages",
            description="Gradio SHOW_EXAMPLES: CUSTOM entity, deny list CSV, whole-page redaction.",
            input_pdf_basename="Partnership-Agreement-Toolkit_0_0.pdf",
            settings=AppEvalRedactionSettings(
                chosen_redact_entities=["CUSTOM"],
                chosen_redact_comprehend_entities=["CUSTOM"],
                deny_list=deny_terms,
                text_extraction_method="Local OCR model - PDFs without selectable text",
                pii_identification_method="Local",
                handwrite_signature_checkbox=[
                    "Extract handwriting",
                    "Extract signatures",
                ],
            ),
            extra_instruction_lines=[
                "Redact entire pages: 2, 5",
            ],
            manifest_extra={
                "source": "SHOW_EXAMPLES",
                "app_example_label": "PDF redaction with deny list and page redaction",
                "deny_list_csv": "partnership_toolkit_redact_custom_deny_list.csv",
                "whole_page_list_csv": "partnership_toolkit_redact_some_pages.csv",
            },
        )
    )

    # --- SHOW_DIFFICULT_OCR_EXAMPLES (app.py ~2071) ---
    partnership = "Partnership-Agreement-Toolkit_0_0.pdf"

    cases.append(
        _write_case(
            case_id="difficult_ocr_baseline_easy_page",
            description="Gradio SHOW_DIFFICULT_OCR_EXAMPLES: Baseline easy page, extract text only.",
            input_pdf_basename=partnership,
            settings=AppEvalRedactionSettings(
                chosen_redact_entities=list(local_ent),
                chosen_llm_entities=list(llm_ent),
                text_extraction_method="Local OCR model - PDFs without selectable text",
                pii_identification_method="Only extract text (no redaction)",
                text_extraction_only=True,
                page_min=1,
                page_max=1,
            ),
            manifest_extra={
                "source": "SHOW_DIFFICULT_OCR_EXAMPLES",
                "app_example_label": "Baseline 'easy' document page",
                "local_ocr_method": "tesseract",
            },
        )
    )

    cases.append(
        _write_case(
            case_id="difficult_ocr_partnership_signatures_page",
            description="Gradio SHOW_DIFFICULT_OCR_EXAMPLES: Scanned page with signatures (hybrid OCR + CUSTOM_VLM_SIGNATURE).",
            input_pdf_basename=partnership,
            settings=AppEvalRedactionSettings(
                chosen_redact_entities=list(local_ent) + ["CUSTOM_VLM_SIGNATURE"],
                chosen_llm_entities=list(llm_ent),
                text_extraction_method="Local OCR model - PDFs without selectable text",
                pii_identification_method="Local",
                handwrite_signature_checkbox=[
                    "Extract handwriting",
                    "Extract signatures",
                ],
                page_min=6,
                page_max=6,
            ),
            manifest_extra={
                "source": "SHOW_DIFFICULT_OCR_EXAMPLES",
                "app_example_label": "Scanned document page with signatures",
                "local_ocr_method": "hybrid-paddle-vlm",
            },
        )
    )

    cases.append(
        _write_case(
            case_id="difficult_ocr_handwritten_note_extract",
            description="Gradio SHOW_DIFFICULT_OCR_EXAMPLES: Handwritten note, extract text only, VLM OCR.",
            input_pdf_basename="Difficult handwritten note.jpg",
            settings=AppEvalRedactionSettings(
                chosen_redact_entities=list(local_ent),
                chosen_llm_entities=list(llm_ent),
                text_extraction_method="Local OCR model - PDFs without selectable text",
                pii_identification_method="Only extract text (no redaction)",
                text_extraction_only=True,
                handwrite_signature_checkbox=["Extract handwriting"],
                page_min=0,
                page_max=0,
            ),
            manifest_extra={
                "source": "SHOW_DIFFICULT_OCR_EXAMPLES",
                "app_example_label": "Unclear text on handwritten note",
                "local_ocr_method": "vlm",
            },
        )
    )

    cases.append(
        _write_case(
            case_id="difficult_ocr_cv_photo_faces",
            description="Gradio SHOW_DIFFICULT_OCR_EXAMPLES: CV with photo, CUSTOM_VLM_FACES.",
            input_pdf_basename="Example-cv-university-graduaty-hr-role-with-photo-2.pdf",
            settings=AppEvalRedactionSettings(
                chosen_redact_entities=list(local_ent) + ["CUSTOM_VLM_FACES"],
                chosen_llm_entities=list(llm_ent),
                text_extraction_method="Local OCR model - PDFs without selectable text",
                pii_identification_method="Local",
                handwrite_signature_checkbox=["Extract handwriting"],
                page_min=0,
                page_max=0,
            ),
            manifest_extra={
                "source": "SHOW_DIFFICULT_OCR_EXAMPLES",
                "app_example_label": "CV with photo - face identification",
                "local_ocr_method": "hybrid-paddle-vlm",
            },
        )
    )

    from tools.config import LOCAL_TRANSFORMERS_LLM_PII_OPTION

    llm_custom = (
        "Redact Lauren's name (always cover the full name if available), email addresses, "
        "and phone numbers with the label LAUREN. Redact university names with the label "
        "UNIVERSITY. Always include the full university name if available."
    )
    cases.append(
        _write_case(
            case_id="difficult_ocr_professor_email_llm_pii",
            description="Gradio SHOW_DIFFICULT_OCR_EXAMPLES: Professor emails PDF with LLM PII + custom instructions.",
            input_pdf_basename="example_of_emails_sent_to_a_professor_before_applying.pdf",
            settings=AppEvalRedactionSettings(
                chosen_redact_entities=["CUSTOM"],
                chosen_llm_entities=["CUSTOM"],
                text_extraction_method="Local model - selectable text",
                pii_identification_method=str(LOCAL_TRANSFORMERS_LLM_PII_OPTION),
                custom_llm_instructions=llm_custom,
                page_min=0,
                page_max=0,
            ),
            manifest_extra={
                "source": "SHOW_DIFFICULT_OCR_EXAMPLES",
                "app_example_label": "Example email LLM PII detection",
                "local_ocr_method": "paddle",
            },
        )
    )

    index_lines = [
        "# App.py example gold cases (scaffold)",
        "",
        "Generated by `scaffold_app_show_example_gold.py`. Add review/OCR gold files locally.",
        "",
        "| case_id | app.py source |",
        "|---------|---------------|",
    ]
    for path in cases:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        index_lines.append(
            f"| `{path.name}` | {manifest.get('app_example_label', '')} |"
        )
    (GOLD_ROOT / "APP_SHOW_EXAMPLES_INDEX.md").write_text(
        "\n".join(index_lines) + "\n", encoding="utf-8"
    )

    from update_gold_manifest_hashes import update_manifest_hashes

    print(f"Created {len(cases)} gold scaffolds under {GOLD_ROOT}")
    for path in cases:
        print(f"  - {path.name}")
        try:
            hashes = update_manifest_hashes(path)
            print(f"      pinned sha256 pdf={hashes['input_pdf_sha256'][:12]}…")
        except (OSError, FileNotFoundError) as exc:
            print(f"      (skip hash pin: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
