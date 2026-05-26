"""
verify_redaction_coverage.py
============================
Pass 1 programmatic redaction verification without per-page VLM.

Checks review CSV against word-level OCR and an optional policy spec:
- uncovered must-redact terms (OCR words with no intersecting review box)
- over-redacted must-not-redact terms (review rows matching deny patterns)
- suspicious rows (single-char boxes, empty TITLES-only rows)
- pages with zero review rows
- optional text-layer leaks on an applied *_redacted.pdf
- optional pixel sampling at box centres on redacted pages

CLI:
    python tools/verify_redaction_coverage.py review_file.csv ocr_words.csv \\
        --must-redact "cora|fuller|fyller" --must-not-redact "dr\\.|macrae|gibson" \\
        --redacted-pdf redacted.pdf --output-json report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore


@dataclass
class TermHit:
    page: int
    text: str
    word_x0: float | None = None
    word_y0: float | None = None
    word_x1: float | None = None
    word_y1: float | None = None
    line: str | None = None
    covered: bool = False


@dataclass
class ReviewRowHit:
    page: int
    row_id: str
    text: str
    label: str
    reason: str


@dataclass
class PageReport:
    page: int
    pass_strict: bool = True
    pass_with_cleanup: bool = True
    review_row_count: int = 0
    uncovered_terms: list[TermHit] = field(default_factory=list)
    over_redacted: list[ReviewRowHit] = field(default_factory=list)
    suspicious_rows: list[ReviewRowHit] = field(default_factory=list)
    text_layer_leaks: list[str] = field(default_factory=list)
    pixel_failures: list[str] = field(default_factory=list)
    leak_likely_causes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # ``pass`` mirrors ``pass_strict`` (policy/visual gate for Pass 1 completion).
        d["pass"] = d["pass_strict"]
        return d


@dataclass
class CoverageReport:
    pages_total: int = 0
    pages_with_policy_issues: int = 0
    pages_with_cleanup_issues: int = 0
    pages_flagged_for_vlm: list[int] = field(default_factory=list)
    pages_needing_csv_cleanup: list[int] = field(default_factory=list)
    pass_strict: bool = True
    pass_with_cleanup: bool = True
    pages: dict[str, PageReport] = field(default_factory=dict)

    @property
    def pass_(self) -> bool:
        """Backward-compatible alias: policy-only pass (not blocked by suspicious rows)."""
        return self.pass_strict

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": self.pass_strict,
            "pass_strict": self.pass_strict,
            "pass_with_cleanup": self.pass_with_cleanup,
            "summary": {
                "pages_total": self.pages_total,
                "pages_with_policy_issues": self.pages_with_policy_issues,
                "pages_with_cleanup_issues": self.pages_with_cleanup_issues,
                "pages_flagged_for_vlm": self.pages_flagged_for_vlm,
                "pages_needing_csv_cleanup": self.pages_needing_csv_cleanup,
                # Deprecated alias kept for older clients.
                "pages_with_issues": self.pages_with_policy_issues,
            },
            "pages": {k: v.to_dict() for k, v in self.pages.items()},
        }


def page_int(row: dict) -> int:
    return int(float(row.get("page", 0) or 0))


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    out: list[re.Pattern] = []
    for p in patterns:
        p = (p or "").strip()
        if not p:
            continue
        out.append(re.compile(p, re.I))
    return out


def boxes_intersect(
    ax0: float,
    ay0: float,
    ax1: float,
    ay1: float,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
    tol: float = 0.002,
) -> bool:
    return not (
        ax1 < bx0 - tol or bx1 < ax0 - tol or ay1 < by0 - tol or by1 < ay0 - tol
    )


def word_box(row: dict) -> tuple[float, float, float, float] | None:
    try:
        return (
            float(row.get("word_x0", row.get("xmin", ""))),
            float(row.get("word_y0", row.get("ymin", ""))),
            float(row.get("word_x1", row.get("xmax", ""))),
            float(row.get("word_y1", row.get("ymax", ""))),
        )
    except (TypeError, ValueError):
        return None


def review_box(row: dict) -> tuple[float, float, float, float] | None:
    try:
        return (
            float(row["xmin"]),
            float(row["ymin"]),
            float(row["xmax"]),
            float(row["ymax"]),
        )
    except (TypeError, ValueError, KeyError):
        return None


def is_covered_by_review(
    wx0: float, wy0: float, wx1: float, wy1: float, review_rows: list[dict]
) -> bool:
    for r in review_rows:
        rb = review_box(r)
        if rb is None:
            continue
        if boxes_intersect(wx0, wy0, wx1, wy1, *rb):
            return True
    return False


def _row_has_non_normalized_bbox(row: dict) -> bool:
    rb = review_box(row)
    if rb is None:
        return False
    return any(v > 1.0 or v < 0.0 for v in rb)


def infer_leak_likely_causes(
    page_report: PageReport, page_review: list[dict]
) -> list[str]:
    """
    Suggest why ``text_layer_leaks`` appear despite review boxes.

    Helps agents avoid misdiagnosing leaks as a broken ``/review_apply`` endpoint.
    """
    if not page_report.text_layer_leaks:
        return []
    causes: list[str] = []
    if page_report.review_row_count == 0:
        causes.append("missing_page_boxes")
    if page_report.uncovered_terms:
        causes.append("missing_review_boxes")
    if page_review and any(_row_has_non_normalized_bbox(r) for r in page_review):
        causes.append("coord_not_normalized")
    if (
        not page_report.uncovered_terms
        and page_report.review_row_count > 0
        and "coord_not_normalized" not in causes
    ):
        causes.append("coord_mismatch_or_image_text")
    return causes


def matches_any(text: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(text or "") for p in patterns)


def is_suspicious_row(row: dict, min_word_length: int = 3) -> str | None:
    text = (row.get("text") or "").strip()
    label = (row.get("label") or "").upper()
    if not text and label == "TITLES":
        return "empty_titles_row"
    if text and len(text) < min_word_length and not re.search(r"\d", text):
        return "single_char_or_short_box"
    return None


def is_prunable_suspicious_row(
    row: dict,
    must_redact: list[str] | list[re.Pattern] | None = None,
    *,
    min_word_length: int = 3,
) -> bool:
    """
    Return True when a suspicious short/OCR-fragment row can be removed safely.

    Rows matching ``must_redact`` are kept even when short (e.g. initials in policy).
    """
    reason = is_suspicious_row(row, min_word_length=min_word_length)
    if not reason:
        return False
    text = (row.get("text") or "").strip()
    if not text:
        return reason == "empty_titles_row"
    patterns: list[re.Pattern]
    if not must_redact:
        patterns = []
    elif isinstance(must_redact[0], re.Pattern):
        patterns = list(must_redact)  # type: ignore[arg-type]
    else:
        patterns = compile_patterns(list(must_redact))
    if patterns and matches_any(text, patterns):
        return False
    return True


def prune_suspicious_review_rows(
    review_rows: list[dict],
    *,
    must_redact: list[str] | None = None,
    min_word_length: int = 3,
) -> tuple[list[dict], dict[str, Any]]:
    """Drop prunable suspicious rows; return kept rows and a removal log."""
    must_redact_re = compile_patterns(must_redact or [])
    kept: list[dict] = []
    removed: list[dict[str, Any]] = []
    for row in review_rows:
        if is_prunable_suspicious_row(
            row, must_redact_re, min_word_length=min_word_length
        ):
            removed.append(
                {
                    "page": page_int(row),
                    "row_id": str(row.get("id", "")),
                    "text": (row.get("text") or "").strip(),
                    "label": row.get("label") or "",
                }
            )
        else:
            kept.append(row)
    log = {
        "removed_count": len(removed),
        "removed_rows": removed,
        "kept_count": len(kept),
    }
    return kept, log


def prune_suspicious_review_csv(
    review_csv_path: str | Path,
    output_csv_path: str | Path,
    *,
    must_redact: list[str] | None = None,
    min_word_length: int = 3,
) -> dict[str, Any]:
    """Load review CSV, prune suspicious rows, write ``output_csv_path``."""
    path = Path(review_csv_path)
    out = Path(output_csv_path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    pruned, log = prune_suspicious_review_rows(
        rows, must_redact=must_redact, min_word_length=min_word_length
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pruned)
    log["input_csv"] = str(path)
    log["output_csv"] = str(out)
    return log


def sample_pixels_dark(
    pdf_path: Path,
    page_num: int,
    boxes: list[tuple[float, float, float, float]],
    dpi: int = 72,
    dark_threshold: int = 40,
) -> list[str]:
    """Return box indices (as strings) whose centre pixel is not dark on redacted page."""
    if fitz is None or not boxes:
        return []
    failures: list[str] = []
    doc = fitz.open(pdf_path)
    try:
        if page_num < 1 or page_num > doc.page_count:
            return failures
        page = doc[page_num - 1]
        pix = page.get_pixmap(
            matrix=fitz.Matrix(dpi / 72, dpi / 72), colorspace=fitz.csGRAY
        )
        w, h = pix.width, pix.height
        samples = pix.samples
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            cx = int((x0 + x1) / 2 * w)
            cy = int((y0 + y1) / 2 * h)
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            val = samples[cy * w + cx]
            if val > dark_threshold:
                failures.append(f"box_{i}_center_not_dark")
    finally:
        doc.close()
    return failures


def verify_redaction_coverage(
    review_csv_path: str | Path,
    ocr_words_csv_path: str | Path,
    *,
    must_redact: list[str] | None = None,
    must_not_redact: list[str] | None = None,
    redacted_pdf_path: str | Path | None = None,
    total_pages: int | None = None,
    min_word_length: int = 3,
    sample_pixels: bool = False,
    pixel_sample_max_boxes_per_page: int = 20,
) -> CoverageReport:
    """
    Build a per-page coverage report for Pass 1 verification.

    Policy patterns are regex strings. ``must_redact`` terms should appear in OCR
    and be covered by a review box. ``must_not_redact`` terms should not appear
    in review rows (unless also matched by must_redact on the same row text).
    """
    review_rows = load_csv_rows(Path(review_csv_path))
    ocr_rows = load_csv_rows(Path(ocr_words_csv_path))

    must_redact_re = compile_patterns(must_redact or [])
    must_not_re = compile_patterns(must_not_redact or [])

    pages_in_review = {page_int(r) for r in review_rows}
    pages_in_ocr = {page_int(r) for r in ocr_rows}
    if total_pages is None:
        total_pages = max(pages_in_review | pages_in_ocr | {0})
    if total_pages <= 0 and redacted_pdf_path and fitz is not None:
        total_pages = fitz.open(redacted_pdf_path).page_count

    report = CoverageReport(pages_total=total_pages)

    redacted_text_by_page: dict[int, str] = {}
    if redacted_pdf_path and fitz is not None and Path(redacted_pdf_path).is_file():
        doc = fitz.open(redacted_pdf_path)
        try:
            for i in range(doc.page_count):
                redacted_text_by_page[i + 1] = doc[i].get_text()
        finally:
            doc.close()

    for page in range(1, total_pages + 1):
        page_report = PageReport(page=page)
        page_review = [r for r in review_rows if page_int(r) == page]
        page_ocr = [r for r in ocr_rows if page_int(r) == page]
        page_report.review_row_count = len(page_review)

        if (
            page_report.review_row_count == 0
            and page in pages_in_ocr
            and must_redact_re
        ):
            has_must_redact_on_page = any(
                matches_any(
                    (wr.get("word_text") or wr.get("text") or ""), must_redact_re
                )
                for wr in page_ocr
            )
            if has_must_redact_on_page:
                page_report.pass_strict = False

        for r in page_review:
            text = (r.get("text") or "").strip()
            label = r.get("label") or ""
            if must_not_re and text and matches_any(text, must_not_re):
                if not (must_redact_re and matches_any(text, must_redact_re)):
                    page_report.over_redacted.append(
                        ReviewRowHit(
                            page=page,
                            row_id=str(r.get("id", "")),
                            text=text,
                            label=label,
                            reason="must_not_redact",
                        )
                    )
                    page_report.pass_strict = False
            reason = is_suspicious_row(r, min_word_length=min_word_length)
            if reason:
                page_report.suspicious_rows.append(
                    ReviewRowHit(
                        page=page,
                        row_id=str(r.get("id", "")),
                        text=text,
                        label=label,
                        reason=reason,
                    )
                )
                page_report.pass_with_cleanup = False

        seen_terms: set[tuple[int, str]] = set()
        if must_redact_re:
            for wr in page_ocr:
                wt = (wr.get("word_text") or wr.get("text") or "").strip()
                if not wt or len(wt) < min_word_length:
                    continue
                if not matches_any(wt, must_redact_re):
                    continue
                wb = word_box(wr)
                if wb is None:
                    continue
                key = (page, wt.lower())
                if key in seen_terms:
                    continue
                seen_terms.add(key)
                covered = is_covered_by_review(*wb, page_review)
                if not covered:
                    page_report.uncovered_terms.append(
                        TermHit(
                            page=page,
                            text=wt,
                            word_x0=wb[0],
                            word_y0=wb[1],
                            word_x1=wb[2],
                            word_y1=wb[3],
                            line=str(wr.get("line", "")),
                            covered=False,
                        )
                    )
                    page_report.pass_strict = False

        if redacted_text_by_page and must_redact_re:
            page_text = redacted_text_by_page.get(page, "")
            for pat in must_redact_re:
                for m in pat.finditer(page_text):
                    leak = m.group()
                    if leak and leak not in page_report.text_layer_leaks:
                        page_report.text_layer_leaks.append(leak)
                        page_report.pass_strict = False

        if page_report.text_layer_leaks:
            page_report.leak_likely_causes = infer_leak_likely_causes(
                page_report, page_review
            )

        if sample_pixels and redacted_pdf_path and page_review:
            boxes: list[tuple[float, float, float, float]] = []
            for r in page_review[:pixel_sample_max_boxes_per_page]:
                rb = review_box(r)
                if rb:
                    boxes.append(rb)
            page_report.pixel_failures = sample_pixels_dark(
                Path(redacted_pdf_path), page, boxes
            )
            if page_report.pixel_failures:
                page_report.pass_strict = False

        report.pages[str(page)] = page_report
        if not page_report.pass_strict:
            report.pages_with_policy_issues += 1
            report.pages_flagged_for_vlm.append(page)
            report.pass_strict = False
        if page_report.suspicious_rows:
            report.pages_with_cleanup_issues += 1
            report.pages_needing_csv_cleanup.append(page)
            report.pass_with_cleanup = False

    return report


def search_words_in_ocr_csv(
    ocr_words_csv_path: str | Path,
    search_text: str,
    *,
    use_regex: bool = False,
    case_insensitive: bool = True,
) -> list[dict]:
    """Literal or regex search in word OCR CSV without Gradio session state."""
    flags = re.I if case_insensitive else 0
    if use_regex:
        pat = re.compile(search_text.strip(), flags)
    else:
        pat = re.compile(re.escape(search_text.strip()), flags)

    hits: list[dict] = []
    for row in load_csv_rows(Path(ocr_words_csv_path)):
        wt = (row.get("word_text") or row.get("text") or "").strip()
        if wt and pat.search(wt):
            hits.append(row)
    return hits


def run_word_level_ocr_text_search(
    ocr_words_csv_path: str | Path,
    search_text: str,
    *,
    similarity_threshold: float = 1.0,
    use_regex: bool = False,
    review_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Headless word-level OCR search (Gradio ``word_level_ocr_text_search`` equivalent).

    Uses CSV word boxes directly. ``similarity_threshold`` is accepted for API
    parity but literal/regex matching is used (threshold 1.0 = exact word match).
    """
    if not search_text or len(search_text.strip()) < 3:
        raise ValueError("search_text must be at least 3 characters")

    if similarity_threshold < 1.0:
        raise ValueError(
            "similarity_threshold < 1.0 is not supported in headless CSV search; "
            "use use_regex=true for pattern matching."
        )

    review_rows: list[dict] = []
    if review_csv_path:
        review_rows = load_csv_rows(Path(review_csv_path))

    raw_hits = search_words_in_ocr_csv(
        ocr_words_csv_path, search_text, use_regex=use_regex
    )

    hits: list[dict[str, Any]] = []
    for ocr_hit in raw_hits:
        page = page_int(ocr_hit)
        wt = str(ocr_hit.get("word_text") or "")
        wb = word_box(ocr_hit)
        page_review = [r for r in review_rows if page_int(r) == page]
        covered = is_covered_by_review(*wb, page_review) if wb and page_review else None
        hits.append(
            {
                "page": page,
                "line": ocr_hit.get("line"),
                "word_text": wt,
                "word_x0": ocr_hit.get("word_x0"),
                "word_y0": ocr_hit.get("word_y0"),
                "word_x1": ocr_hit.get("word_x1"),
                "word_y1": ocr_hit.get("word_y1"),
                "covered_by_review_box": covered,
            }
        )

    return {
        "search_text": search_text,
        "use_regex": use_regex,
        "match_count": len(hits),
        "matches": hits,
        "duplicate_files": [],
        "full_data_keys": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify redaction coverage (Pass 1)")
    parser.add_argument("review_csv", type=Path)
    parser.add_argument("ocr_words_csv", type=Path)
    parser.add_argument(
        "--must-redact", action="append", default=[], help="Regex (repeatable)"
    )
    parser.add_argument(
        "--must-not-redact", action="append", default=[], help="Regex (repeatable)"
    )
    parser.add_argument("--redacted-pdf", type=Path, default=None)
    parser.add_argument(
        "--pages", type=int, default=None, help="Total page count override"
    )
    parser.add_argument("--min-word-length", type=int, default=3)
    parser.add_argument("--sample-pixels", action="store_true")
    parser.add_argument(
        "--prune-suspicious",
        action="store_true",
        help="Write a pruned review CSV with suspicious short rows removed.",
    )
    parser.add_argument(
        "--pruned-output",
        type=Path,
        default=None,
        help="Output path for --prune-suspicious (default: <review_csv>_pruned.csv).",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.prune_suspicious:
        out = args.pruned_output or args.review_csv.with_name(
            f"{args.review_csv.stem}_pruned.csv"
        )
        prune_log = prune_suspicious_review_csv(
            args.review_csv,
            out,
            must_redact=args.must_redact,
            min_word_length=args.min_word_length,
        )
        print(json.dumps(prune_log, indent=2))
        args.review_csv = out

    report = verify_redaction_coverage(
        args.review_csv,
        args.ocr_words_csv,
        must_redact=args.must_redact,
        must_not_redact=args.must_not_redact,
        redacted_pdf_path=args.redacted_pdf,
        total_pages=args.pages,
        min_word_length=args.min_word_length,
        sample_pixels=args.sample_pixels,
    )
    payload = report.to_dict()
    text = json.dumps(payload, indent=2)
    if args.output_json:
        args.output_json.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output_json}")
    else:
        print(text)
    raise SystemExit(0 if report.pass_strict else 1)


if __name__ == "__main__":
    main()
