"""Partial gold policy list files from app example settings (not full document coverage)."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from derive_app_eval_instructions import AppEvalRedactionSettings

_HEADER = (
    "# Partial scaffold from app.py example settings — tune after a blessed run.\n"
    "# Lines are regex patterns for verify_redaction_coverage (OCR word match).\n"
)


def _literal_pattern(term: str) -> str:
    text = (term or "").strip()
    if not text:
        return ""
    return re.escape(text)


def _section(title: str, lines: list[str]) -> list[str]:
    if not lines:
        return []
    out = [f"# --- {title} ---"]
    out.extend(lines)
    return out


_LLM_STOPWORDS = frozenset(
    {
        "redact",
        "always",
        "include",
        "available",
        "cover",
        "full",
        "name",
        "names",
        "label",
        "with",
        "the",
        "and",
        "when",
    }
)


def _llm_instruction_literals(custom: str) -> list[str]:
    """Obvious name-like tokens from custom LLM redaction instructions."""
    text = (custom or "").strip()
    if not text:
        return []
    patterns: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        token = raw.strip().strip("'\"")
        if len(token) < 2:
            return
        key = token.casefold()
        if key in seen or key in _LLM_STOPWORDS:
            return
        seen.add(key)
        patterns.append(_literal_pattern(token))

    for match in re.finditer(r"\b([A-Z][a-z]+(?:'s)?)\b", text):
        _add(match.group(1))
    for match in re.finditer(r"\b([A-Z]{2,})\b", text):
        if match.group(1) in {"LAUREN", "UNIVERSITY"}:
            _add(match.group(1))

    if "email" in text.casefold():
        patterns.append("@")
    if "phone" in text.casefold():
        patterns.append(r"\+?\d[\d\s\-]{6,}\d")
    if "university" in text.casefold():
        patterns.append(r"(?i)university")

    deduped: list[str] = []
    seen_pat: set[str] = set()
    for p in patterns:
        if p and p not in seen_pat:
            seen_pat.add(p)
            deduped.append(p)
    return deduped


def _entity_heuristic_patterns(settings: AppEvalRedactionSettings) -> list[str]:
    if settings.text_extraction_only:
        return []
    entities = {e.casefold() for e in settings.merged_entities()}
    lines: list[str] = []
    if any(x in entities for x in ("email", "email_address")):
        lines.append("@")
    if any(
        x in entities
        for x in (
            "phone",
            "phone_number",
            "phone_number",
        )
    ):
        lines.append(r"\+?\d")
        lines.append(r"\d{5}")
    if "ukpostcode" in entities:
        lines.append(r"(?i)[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}")
    return lines


def _find_example_output_review_csv(repo_root: Path, pdf_basename: str) -> Path | None:
    stem = Path(pdf_basename).stem
    out_dir = repo_root / "doc_redaction" / "example_data" / "example_outputs"
    if not out_dir.is_dir():
        return None
    candidates = [
        out_dir / f"{stem}_review_file.csv",
        out_dir / f"{stem}.pdf_review_file.csv",
        out_dir / f"{Path(pdf_basename).name}_review_file.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    for path in sorted(out_dir.glob("*review_file*.csv")):
        if stem.casefold() in path.name.casefold():
            return path
    return None


def _mine_review_text_patterns(review_csv: Path, *, max_lines: int = 40) -> list[str]:
    """Distinct review row text snippets (PII-like labels only)."""
    if not review_csv.is_file():
        return []
    pii_labels = (
        "NAME",
        "PERSON",
        "EMAIL",
        "PHONE",
        "ADDRESS",
        "TITLE",
        "CUSTOM",
        "LAUREN",
        "UNIVERSITY",
    )
    snippets: list[str] = []
    seen: set[str] = set()
    try:
        with review_csv.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = str(row.get("label") or "").upper()
                if not any(tag in label for tag in pii_labels):
                    continue
                text = str(row.get("text") or "").strip()
                if len(text) < 2 or len(text) > 80:
                    continue
                key = text.casefold()
                if key in seen:
                    continue
                seen.add(key)
                snippets.append(_literal_pattern(text))
                if len(snippets) >= max_lines:
                    break
    except (OSError, csv.Error):
        return []
    return snippets


def build_partial_must_redact_lines(
    settings: AppEvalRedactionSettings,
    *,
    repo_root: Path,
    input_pdf_basename: str,
) -> list[str]:
    lines: list[str] = []

    deny_patterns = [_literal_pattern(t) for t in settings.deny_list if str(t).strip()]
    lines.extend(_section("deny list (literal)", deny_patterns))

    llm_lines = _llm_instruction_literals(settings.custom_llm_instructions)
    lines.extend(_section("custom LLM instruction literals", llm_lines))

    review_path = _find_example_output_review_csv(repo_root, input_pdf_basename)
    if review_path is not None:
        mined = _mine_review_text_patterns(review_path)
        rel = review_path.relative_to(repo_root).as_posix()
        lines.extend(_section(f"mined from shipped {rel} (bootstrap)", mined))

    heuristics = _entity_heuristic_patterns(settings)
    lines.extend(_section("entity heuristics (incomplete)", heuristics))

    if not any(line for line in lines if not line.startswith("#")):
        lines.append("# No partial patterns yet — add regex lines after blessing.")
    return lines


def write_partial_policy_files(
    case_dir: Path,
    settings: AppEvalRedactionSettings,
    *,
    repo_root: Path,
    input_pdf_basename: str,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)

    must_lines = build_partial_must_redact_lines(
        settings,
        repo_root=repo_root,
        input_pdf_basename=input_pdf_basename,
    )
    (case_dir / "must_redact.txt").write_text(
        _HEADER + "\n".join(must_lines) + "\n",
        encoding="utf-8",
    )

    allow = [t.strip() for t in settings.allow_list if str(t).strip()]
    if allow:
        must_not_body = _HEADER + "\n".join(
            _section("allow list (literal)", [_literal_pattern(t) for t in allow])
        )
        (case_dir / "must_not_redact.txt").write_text(
            must_not_body + "\n",
            encoding="utf-8",
        )

    deny = [t.strip() for t in settings.deny_list if str(t).strip()]
    if deny:
        (case_dir / "expected_deny_list.txt").write_text(
            "\n".join(deny) + "\n",
            encoding="utf-8",
        )
