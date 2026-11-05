#!/usr/bin/env python3
"""Compare gold vs run word-level OCR CSVs (token precision / recall / F1).

Example::

    python agent-redact/eval/compare_ocr_words_csvs.py \\
        gold/example/ocr_results_with_words.csv \\
        runs/run_xxx/ocr_results_with_words.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_SAMPLE_CAP = 20


def normalize_word_text(raw: str) -> str:
    text = (raw or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


@dataclass(frozen=True)
class WordRow:
    page: int
    text_norm: str
    text_raw: str
    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None
    source_index: int = -1

    @property
    def has_bbox(self) -> bool:
        return None not in (self.x0, self.y0, self.x1, self.y1)


def _page_int(raw: Any) -> int:
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return 0


def _float_cell(row: dict[str, Any], key: str) -> float | None:
    raw = row.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def load_word_rows(path: str | Path) -> list[WordRow]:
    """Load word rows from a ``*ocr_results_with_words*`` CSV."""
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing OCR CSV: {csv_path}")

    rows: list[WordRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return rows
        lower_map = {name.lower(): name for name in reader.fieldnames if name}
        text_key = lower_map.get("word_text") or lower_map.get("text")
        if not text_key:
            raise ValueError(
                f"{csv_path.name} missing word_text column; found {list(reader.fieldnames)}"
            )
        page_key = lower_map.get("page")
        x0_key = lower_map.get("word_x0") or lower_map.get("x0")
        y0_key = lower_map.get("word_y0") or lower_map.get("y0")
        x1_key = lower_map.get("word_x1") or lower_map.get("x1")
        y1_key = lower_map.get("word_y1") or lower_map.get("y1")

        for idx, row in enumerate(reader):
            raw_text = str(row.get(text_key, "") or "").strip()
            if not raw_text:
                continue
            norm = normalize_word_text(raw_text)
            if not norm:
                continue
            page = _page_int(row.get(page_key, 0)) if page_key else 0
            rows.append(
                WordRow(
                    page=page,
                    text_norm=norm,
                    text_raw=raw_text,
                    x0=_float_cell(row, x0_key) if x0_key else None,
                    y0=_float_cell(row, y0_key) if y0_key else None,
                    x1=_float_cell(row, x1_key) if x1_key else None,
                    y1=_float_cell(row, y1_key) if y1_key else None,
                    source_index=idx,
                )
            )
    return rows


def word_bbox_iou(a: WordRow, b: WordRow) -> float:
    if a.page != b.page or not a.has_bbox or not b.has_bbox:
        return 0.0
    assert (
        a.x0 is not None and a.y0 is not None and a.x1 is not None and a.y1 is not None
    )
    assert (
        b.x0 is not None and b.y0 is not None and b.x1 is not None and b.y1 is not None
    )
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x1 - a.x0) * max(0.0, a.y1 - a.y0)
    area_b = max(0.0, b.x1 - b.x0) * max(0.0, b.y1 - b.y0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


@dataclass
class OcrCompareReport:
    gold_word_count: int = 0
    observed_word_count: int = 0
    matched: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    skipped: bool = False
    skip_reason: str = ""
    bbox_iou_threshold: float | None = None
    unmatched_gold_samples: list[str] = field(default_factory=list)
    unmatched_observed_samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_ocr_words_csvs(
    gold_csv: str | Path,
    observed_csv: str | Path,
    *,
    bbox_iou_threshold: float | None = None,
) -> OcrCompareReport:
    """
    Match words by ``(page, normalized word_text)`` in file order (1:1 greedy).

    When *bbox_iou_threshold* is set, a candidate pair must also meet word bbox IoU.
    """
    gold_rows = load_word_rows(gold_csv)
    obs_rows = load_word_rows(observed_csv)

    used_obs: set[int] = set()
    matched = 0
    unmatched_gold: list[str] = []

    for g in gold_rows:
        found = False
        for oi, o in enumerate(obs_rows):
            if oi in used_obs:
                continue
            if g.page != o.page or g.text_norm != o.text_norm:
                continue
            if bbox_iou_threshold is not None:
                if word_bbox_iou(g, o) < bbox_iou_threshold:
                    continue
            used_obs.add(oi)
            matched += 1
            found = True
            break
        if not found:
            unmatched_gold.append(f"p{g.page}:{g.text_raw}")

    unmatched_obs = [
        f"p{o.page}:{o.text_raw}" for oi, o in enumerate(obs_rows) if oi not in used_obs
    ]

    g_n = len(gold_rows)
    o_n = len(obs_rows)
    recall = (matched / g_n) if g_n else 1.0
    precision = (matched / o_n) if o_n else 1.0
    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return OcrCompareReport(
        gold_word_count=g_n,
        observed_word_count=o_n,
        matched=matched,
        precision=precision,
        recall=recall,
        f1=f1,
        bbox_iou_threshold=bbox_iou_threshold,
        unmatched_gold_samples=unmatched_gold[:_SAMPLE_CAP],
        unmatched_observed_samples=unmatched_obs[:_SAMPLE_CAP],
    )


def _read_manifest_ocr_method(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("ocr_method") or "").strip().casefold()


def ocr_method_mismatch(
    *,
    gold_case_dir: Path,
    run_dir: Path,
) -> tuple[bool, str]:
    """
    Return (should_skip, reason) when fixture manifest ocr_method differs from run manifest.
    """
    gold_method = _read_manifest_ocr_method(gold_case_dir / "manifest.json")
    run_method = _read_manifest_ocr_method(run_dir / "manifest.json")
    if not gold_method or not run_method:
        return False, ""
    if gold_method == run_method:
        return False, ""
    return True, f"ocr_method mismatch: gold={gold_method!r} run={run_method!r}"


def env_bbox_iou_threshold() -> float | None:
    raw = (os.environ.get("AGENT_EVAL_OCR_IOU_THRESHOLD") or "").strip()
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if val > 0 else None


def compare_ocr_gold_for_run(
    *,
    gold_case_dir: Path,
    run_dir: Path,
    gold_csv: Path | None = None,
    observed_csv: Path | None = None,
    fixture_dir: Path | None = None,
) -> OcrCompareReport:
    """Compare gold case OCR to run OCR with manifest guardrail."""
    if fixture_dir is not None:
        gold_case_dir = fixture_dir
    gold_case_dir = gold_case_dir.resolve()
    run_dir = run_dir.resolve()
    gold_path = gold_csv or (gold_case_dir / "ocr_results_with_words.csv")
    obs_path = observed_csv or (run_dir / "ocr_results_with_words.csv")

    if not gold_path.is_file():
        return OcrCompareReport(
            skipped=True,
            skip_reason="missing gold case ocr_results_with_words.csv",
        )
    if not obs_path.is_file():
        return OcrCompareReport(
            skipped=True,
            skip_reason="missing run ocr_results_with_words.csv",
        )

    skip, reason = ocr_method_mismatch(gold_case_dir=gold_case_dir, run_dir=run_dir)
    if skip:
        return OcrCompareReport(skipped=True, skip_reason=reason)

    return compare_ocr_words_csvs(
        gold_path,
        obs_path,
        bbox_iou_threshold=env_bbox_iou_threshold(),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare gold vs observed word-level OCR CSVs (F1)."
    )
    p.add_argument("gold_csv", help="Fixture ocr_results_with_words.csv")
    p.add_argument("observed_csv", help="Run ocr_results_with_words.csv")
    p.add_argument(
        "--bbox-iou-threshold",
        type=float,
        default=None,
        help="Require word bbox IoU >= threshold (overrides AGENT_EVAL_OCR_IOU_THRESHOLD).",
    )
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument(
        "--min-f1",
        type=float,
        default=None,
        help="Exit 1 if F1 is below this threshold.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    threshold = args.bbox_iou_threshold
    if threshold is None:
        threshold = env_bbox_iou_threshold()
    report = compare_ocr_words_csvs(
        args.gold_csv,
        args.observed_csv,
        bbox_iou_threshold=threshold,
    )
    payload = report.to_dict()
    text = json.dumps(payload, indent=2)
    if args.json_out:
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.min_f1 is not None and not report.skipped and report.f1 < args.min_f1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
