"""
post_redaction_pass1_qa.py
==========================
Optional Pass 1 sanity QA at the end of initial redaction (pre-review-apply).

Writes a coverage JSON report and optionally a sibling pruned review CSV.
Does not run VLM or call /review_apply.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from tools.config import (
    POST_REDACT_PASS1_AUTO_PRUNE,
    POST_REDACT_PASS1_INCLUDE_IN_OUTPUTS,
    POST_REDACT_PASS1_MIN_WORD_LENGTH,
    POST_REDACT_PASS1_MUST_NOT_REDACT_PATH,
    POST_REDACT_PASS1_MUST_REDACT_PATH,
    POST_REDACT_PASS1_QA,
    POST_REDACT_PASS1_USE_DENY_ALLOW_LISTS,
)
from tools.verify_redaction_coverage import (
    prune_suspicious_review_csv,
    verify_redaction_coverage,
)


def load_regex_patterns_from_csv(path: str | Path) -> list[str]:
    """Load regex patterns from column 0 of a CSV (same shape as deny/allow list files)."""
    p = Path(path)
    if not p.is_file():
        return []
    df = pd.read_csv(p, header=None, low_memory=False)
    if df.empty:
        return []
    return [str(x).strip() for x in df.iloc[:, 0].dropna().tolist() if str(x).strip()]


def merge_policy_patterns(
    deny_list: list[str] | None,
    allow_list: list[str] | None,
    *,
    must_redact_path: str = "",
    must_not_redact_path: str = "",
    use_deny_allow_lists: bool = True,
) -> tuple[list[str], list[str]]:
    """Build must_redact / must_not_redact regex lists from run lists and env CSV paths."""
    must_redact: list[str] = []
    must_not: list[str] = []

    if use_deny_allow_lists:
        if deny_list:
            must_redact.extend(str(x).strip() for x in deny_list if str(x).strip())
        if allow_list:
            must_not.extend(str(x).strip() for x in allow_list if str(x).strip())

    env_must = must_redact_path or POST_REDACT_PASS1_MUST_REDACT_PATH
    env_must_not = must_not_redact_path or POST_REDACT_PASS1_MUST_NOT_REDACT_PATH
    if env_must:
        must_redact.extend(load_regex_patterns_from_csv(env_must))
    if env_must_not:
        must_not.extend(load_regex_patterns_from_csv(env_must_not))

    return must_redact, must_not


def _pruned_review_csv_path(review_csv_path: str | Path) -> Path:
    p = Path(review_csv_path)
    return p.with_name(f"{p.stem}_pruned.csv")


def _coverage_report_path(review_csv_path: str | Path) -> Path:
    p = Path(review_csv_path)
    return p.with_name(f"{p.stem}_coverage_report.json")


def build_qa_summary(report: dict[str, Any]) -> str:
    """Human-readable summary for combined_out_message."""
    summary = report.get("summary") or {}
    n_vlm = len(summary.get("pages_flagged_for_vlm") or [])
    n_cleanup = len(summary.get("pages_needing_csv_cleanup") or [])
    return (
        "Pass 1 QA: "
        f"pass_strict={report.get('pass_strict', report.get('pass'))}, "
        f"pass_with_cleanup={report.get('pass_with_cleanup')}, "
        f"pages_flagged_for_vlm={n_vlm}, "
        f"pages_needing_csv_cleanup={n_cleanup}."
    )


def run_post_redaction_pass1_qa(
    *,
    review_csv_path: str | Path,
    ocr_words_csv_path: str | Path,
    output_folder: str | None = None,
    total_pages: int | None = None,
    must_redact: list[str] | None = None,
    must_not_redact: list[str] | None = None,
    deny_list: list[str] | None = None,
    allow_list: list[str] | None = None,
    auto_prune: bool | None = None,
    min_word_length: int | None = None,
    enabled: bool | None = None,
    use_deny_allow_lists: bool | None = None,
    include_in_outputs: bool | None = None,
) -> dict[str, Any]:
    """
    Run post-redaction Pass 1 QA on initial review CSV + word OCR.

    Returns dict with keys: enabled, paths_created, report, summary, prune_log.
    """
    use_enabled = POST_REDACT_PASS1_QA if enabled is None else bool(enabled)
    if not use_enabled:
        return {
            "enabled": False,
            "paths_created": [],
            "report": None,
            "summary": "",
            "prune_log": None,
        }

    review_path = Path(review_csv_path)
    ocr_path = Path(ocr_words_csv_path)
    if not review_path.is_file():
        print("Post-redaction Pass 1 QA skipped: review CSV not found.")
        return {
            "enabled": True,
            "paths_created": [],
            "report": None,
            "summary": "",
            "prune_log": None,
            "error": "review_csv_missing",
        }
    if not ocr_path.is_file():
        print("Post-redaction Pass 1 QA skipped: OCR words CSV not found.")
        return {
            "enabled": True,
            "paths_created": [],
            "report": None,
            "summary": "",
            "prune_log": None,
            "error": "ocr_words_csv_missing",
        }

    if must_redact is None or must_not_redact is None:
        merged_must, merged_must_not = merge_policy_patterns(
            deny_list,
            allow_list,
            use_deny_allow_lists=(
                POST_REDACT_PASS1_USE_DENY_ALLOW_LISTS
                if use_deny_allow_lists is None
                else use_deny_allow_lists
            ),
        )
        if must_redact is None:
            must_redact = merged_must
        if must_not_redact is None:
            must_not_redact = merged_must_not

    min_wl = (
        POST_REDACT_PASS1_MIN_WORD_LENGTH
        if min_word_length is None
        else min_word_length
    )
    do_prune = POST_REDACT_PASS1_AUTO_PRUNE if auto_prune is None else bool(auto_prune)

    paths_created: list[str] = []
    prune_log: dict[str, Any] | None = None
    csv_for_report = review_path

    if do_prune:
        pruned_path = _pruned_review_csv_path(review_path)
        prune_log = prune_suspicious_review_csv(
            review_path,
            pruned_path,
            must_redact=must_redact,
            min_word_length=min_wl,
        )
        csv_for_report = pruned_path
        paths_created.append(str(pruned_path))

    report_obj = verify_redaction_coverage(
        csv_for_report,
        ocr_path,
        must_redact=must_redact,
        must_not_redact=must_not_redact,
        total_pages=total_pages,
        min_word_length=min_wl,
    )
    report = report_obj.to_dict()

    report_path = _coverage_report_path(review_path)
    if output_folder:
        report_path = Path(output_folder) / report_path.name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    paths_created.append(str(report_path))

    include = (
        POST_REDACT_PASS1_INCLUDE_IN_OUTPUTS
        if include_in_outputs is None
        else include_in_outputs
    )
    if not include:
        paths_created = []

    summary = build_qa_summary(report)
    print(summary)

    return {
        "enabled": True,
        "paths_created": paths_created,
        "report": report,
        "summary": summary,
        "prune_log": prune_log,
        "review_csv_for_report": str(csv_for_report),
        "coverage_report_path": str(report_path),
    }
