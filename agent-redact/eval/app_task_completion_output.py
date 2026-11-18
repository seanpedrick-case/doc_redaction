"""Synthetic DeepEval Task Completion actual_output for main-app (Gradio) eval."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from derive_app_eval_instructions import AppEvalRedactionSettings

_MAX_OUTPUT_CHARS = 12000
_MAX_TERM_SAMPLES = 5


def _env_truthy(name: str) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _review_csv_stats(run_dir: Path) -> tuple[int, set[int]]:
    path = run_dir / "agent_review_file.csv"
    if not path.is_file():
        return 0, set()
    rows = 0
    pages: set[int] = set()
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows += 1
                raw_page = (row.get("page") or "").strip()
                if raw_page.isdigit():
                    pages.add(int(raw_page))
    except (OSError, csv.Error):
        return 0, set()
    return rows, pages


def _uncovered_term_stats(
    policy_gate: dict[str, Any] | None,
) -> tuple[int, int, list[str]]:
    """Return (pages_with_uncovered, total_tokens, sample_placeholders)."""
    if not policy_gate:
        return 0, 0, []
    report = policy_gate.get("report")
    if not isinstance(report, dict):
        return 0, 0, []
    pages = report.get("pages")
    if not isinstance(pages, dict):
        return 0, 0, []

    pages_with = 0
    total = 0
    samples: list[str] = []
    include_samples = _env_truthy("APP_EVAL_TASK_COMPLETION_INCLUDE_TERM_SAMPLES")

    for _key, page_data in pages.items():
        if not isinstance(page_data, dict):
            continue
        uncovered = page_data.get("uncovered_terms")
        if not isinstance(uncovered, list) or not uncovered:
            continue
        pages_with += 1
        total += len(uncovered)
        if include_samples:
            for item in uncovered:
                if len(samples) >= _MAX_TERM_SAMPLES:
                    break
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                samples.append(f"[REDACTED len={len(text)}]")
            if len(samples) >= _MAX_TERM_SAMPLES:
                break

    return pages_with, total, samples


def _fmt_f1(block: dict[str, Any] | None) -> str:
    if not block:
        return "n/a"
    if block.get("skipped"):
        return f"skipped ({block.get('skip_reason') or '—'})"
    f1 = block.get("f1")
    if f1 is None:
        return "n/a"
    return str(f1)


def _policy_extraction_line(block: dict[str, Any] | None) -> str:
    if not block:
        return "n/a"
    guard = block.get("guardrail")
    if isinstance(guard, dict) and guard.get("pass") is False:
        return "fail (guardrail)"
    must_f1 = block.get("must_redact_f1")
    if must_f1 is not None:
        return f"must_redact phrase F1={must_f1}"
    return "recorded"


def build_app_task_completion_actual_output(
    *,
    run_dir: Path,
    settings: AppEvalRedactionSettings,
    document_name: str,
    gate_summary: Any | None,
    gold_case_id: str | None,
) -> str:
    """
    Privacy-safe narrative for DeepEval TaskCompletionMetric (main app).

    Does not include uncovered term text unless
    ``APP_EVAL_TASK_COMPLETION_INCLUDE_TERM_SAMPLES=1`` (then length-only placeholders).
    """
    run_dir = run_dir.resolve()
    lines: list[str] = [
        "Main-app redaction run completed (eval_profile=app_gradio).",
        "",
    ]

    doc = (document_name or "").strip() or "unknown document"
    ocr = (settings.text_extraction_method or "").strip() or "unspecified"
    pii = (settings.pii_identification_method or "").strip() or "unspecified"
    workflow = (
        "extract text only" if settings.text_extraction_only else "full redaction"
    )
    lines.append(f"Document: {doc}")
    lines.append(
        f"Configuration: text extraction = {ocr}; PII = {pii}; workflow = {workflow}"
    )
    lines.append("")

    review_rows, review_pages = _review_csv_stats(run_dir)
    has_review = (run_dir / "agent_review_file.csv").is_file()
    has_redacted = (run_dir / "redacted.pdf").is_file()
    has_ocr = (run_dir / "ocr_results_with_words.csv").is_file()

    lines.append("Deliverables:")
    if settings.text_extraction_only:
        lines.append(f"- OCR words CSV: {'yes' if has_ocr else 'no'}")
    else:
        page_note = ""
        if review_pages:
            page_note = f" ({len(review_pages)} page(s))"
        lines.append(
            f"- Review CSV: {review_rows} rows{page_note if has_review else ' (missing)'}"
        )
        lines.append(f"- Redacted PDF: {'yes' if has_redacted else 'no'}")
        lines.append(f"- OCR words CSV: {'yes' if has_ocr else 'no'}")
    lines.append("")

    policy_gate = None
    gold_geometry = None
    ocr_gold = None
    policy_extraction = None
    if gate_summary is not None:
        policy_gate = getattr(gate_summary, "policy_gate", None)
        gold_geometry = getattr(gate_summary, "gold_geometry", None)
        ocr_gold = getattr(gate_summary, "ocr_gold", None)
        policy_extraction = getattr(gate_summary, "policy_extraction", None)
    if policy_gate is None:
        policy_gate = _load_json(run_dir / "policy_gate.json")
    if gold_geometry is None:
        gold_geometry = _load_json(run_dir / "gold_geometry.json")
    if ocr_gold is None:
        ocr_gold = _load_json(run_dir / "ocr_gold.json")
    if policy_extraction is None:
        policy_extraction = _load_json(run_dir / "policy_extraction.json")

    case_label = (gold_case_id or "").strip() or "unknown"
    lines.append(f"Gold QA (case {case_label}):")

    if policy_gate is not None:
        pass_strict = policy_gate.get("pass_strict")
        must_count = policy_gate.get("must_redact_count")
        report = (
            policy_gate.get("report")
            if isinstance(policy_gate.get("report"), dict)
            else {}
        )
        summary = (
            report.get("summary") if isinstance(report.get("summary"), dict) else {}
        )
        pages_total = summary.get("pages_total")
        pages_issues = summary.get("pages_with_issues")
        pages_with_uncovered, uncovered_total, samples = _uncovered_term_stats(
            policy_gate
        )
        detail = []
        if must_count is not None:
            detail.append(f"{must_count} must-redact patterns")
        if pages_total is not None and pages_issues is not None:
            detail.append(f"{pages_issues}/{pages_total} pages with issues")
        if uncovered_total:
            detail.append(
                f"uncovered terms on {pages_with_uncovered} page(s), "
                f"{uncovered_total} token(s) total — text omitted"
            )
        if samples:
            detail.append(f"sample placeholders: {', '.join(samples)}")
        suffix = f" ({'; '.join(detail)})" if detail else ""
        lines.append(f"- Policy gate pass_strict: {pass_strict}{suffix}")
    else:
        lines.append("- Policy gate: not run")

    if gold_geometry is not None and not settings.text_extraction_only:
        f1 = _fmt_f1(gold_geometry)
        threshold = gold_geometry.get("iou_threshold")
        th_note = f", threshold {threshold}" if threshold is not None else ""
        lines.append(f"- Gold geometry F1: {f1} (core coverage{th_note})")
    elif not settings.text_extraction_only:
        lines.append("- Gold geometry: not run")

    if ocr_gold is not None:
        lines.append(f"- OCR gold F1: {_fmt_f1(ocr_gold)}")
    else:
        lines.append("- OCR gold: not run")

    if policy_extraction is not None and not settings.text_extraction_only:
        lines.append(
            f"- Policy extraction: {_policy_extraction_line(policy_extraction)}"
        )
    elif not settings.text_extraction_only:
        lines.append("- Policy extraction: not run")

    lines.append("")
    if not has_review and not has_ocr and not settings.text_extraction_only:
        lines.append(
            "Interpretation: Redaction workflow finished; no review artifacts found for eval."
        )
    elif settings.text_extraction_only:
        lines.append(
            "Interpretation: Text extraction completed; compare OCR gold metrics for quality."
        )
    else:
        lines.append(
            "Interpretation: Automated redaction and review artifacts were produced. "
            "Deterministic gates summarize policy and geometry QA; human review_apply "
            "may still be required when pass_strict or geometry F1 are below thresholds."
        )

    text = "\n".join(lines).strip()
    if not text:
        text = "Redaction workflow finished; no review artifacts found for eval."
    return text[:_MAX_OUTPUT_CHARS]
