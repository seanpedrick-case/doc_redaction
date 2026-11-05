#!/usr/bin/env python3
"""Compare gold vs agent ``*_review_file.csv`` boxes (IoU / containment F1).

Offline regression metric for agentic Pass 1 quality. See ``README.md`` in this
folder for the gold fixture protocol and recommended thresholds.

Example::

    python agent-redact/eval/compare_review_csvs.py \\
        gold.csv agent.csv --iou-threshold 0.5 --min-f1 0.9
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

MatchMode = Literal["iou", "gold_contained", "agent_contained"]
ScoringMode = Literal["core_coverage", "legacy_box_match"]
Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class Box:
    """One review row used for geometric matching."""

    page: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str = ""
    text: str = ""
    row_id: str = ""
    source_index: int = -1

    @property
    def area(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)


@dataclass
class MatchPair:
    gold_index: int
    agent_index: int
    score: float
    mode: str


@dataclass
class CompareReport:
    """Precision / recall / F1 for gold vs agent review boxes."""

    gold_count: int = 0
    agent_count: int = 0
    matched_gold: int = 0
    matched_agent: int = 0
    unmatched_gold_indices: list[int] = field(default_factory=list)
    unmatched_agent_indices: list[int] = field(default_factory=list)
    matches: list[MatchPair] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    strict_pass: bool = False
    iou_threshold: float = 0.5
    match_mode: str = "iou"
    require_label_match: bool = False
    label_agreement: float | None = None
    pages: list[int] = field(default_factory=list)
    scoring: str = "legacy_box_match"
    core_gold_count: int = 0
    core_agent_count: int = 0
    removed_gold_indices: list[int] = field(default_factory=list)
    removed_agent_indices: list[int] = field(default_factory=list)
    per_core_gold: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["matches"] = [asdict(m) for m in self.matches]
        return data


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


def load_review_boxes(path: str | Path) -> list[Box]:
    """Load normalized review boxes from a ``*_review_file.csv``."""
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing review CSV: {csv_path}")

    boxes: list[Box] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return boxes
        required = {"xmin", "ymin", "xmax", "ymax"}
        lower_map = {name.lower(): name for name in reader.fieldnames if name}
        missing = [c for c in required if c not in lower_map]
        if missing:
            raise ValueError(
                f"{csv_path.name} missing required columns {missing}; "
                f"found {list(reader.fieldnames)}"
            )

        def col(name: str) -> str:
            return lower_map[name.lower()]

        for idx, row in enumerate(reader):
            coords = {
                "xmin": _float_cell(row, col("xmin")),
                "ymin": _float_cell(row, col("ymin")),
                "xmax": _float_cell(row, col("xmax")),
                "ymax": _float_cell(row, col("ymax")),
            }
            if any(v is None for v in coords.values()):
                continue
            xmin, ymin, xmax, ymax = (
                coords["xmin"],
                coords["ymin"],
                coords["xmax"],
                coords["ymax"],
            )
            assert xmin is not None and ymin is not None
            assert xmax is not None and ymax is not None
            if xmax <= xmin or ymax <= ymin:
                continue
            page_key = lower_map.get("page")
            label_key = lower_map.get("label")
            text_key = lower_map.get("text")
            id_key = lower_map.get("id")
            boxes.append(
                Box(
                    page=_page_int(row.get(page_key, 0)) if page_key else 0,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    label=(
                        str(row.get(label_key, "") or "").strip() if label_key else ""
                    ),
                    text=str(row.get(text_key, "") or "").strip() if text_key else "",
                    row_id=str(row.get(id_key, "") or "").strip() if id_key else "",
                    source_index=idx,
                )
            )
    return boxes


def box_intersection_area(a: Box, b: Box) -> float:
    if a.page != b.page:
        return 0.0
    x0 = max(a.xmin, b.xmin)
    y0 = max(a.ymin, b.ymin)
    x1 = min(a.xmax, b.xmax)
    y1 = min(a.ymax, b.ymax)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def box_iou(a: Box, b: Box) -> float:
    inter = box_intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def gold_containment(gold: Box, agent: Box) -> float:
    """Fraction of gold area covered by intersection with one agent box."""
    if gold.area <= 0:
        return 0.0
    return box_intersection_area(gold, agent) / gold.area


def agent_containment(gold: Box, agent: Box) -> float:
    """Fraction of agent area covered by intersection with one gold box."""
    if agent.area <= 0:
        return 0.0
    return box_intersection_area(gold, agent) / agent.area


def pair_score(gold: Box, agent: Box, mode: MatchMode) -> float:
    if mode == "iou":
        return box_iou(gold, agent)
    if mode == "gold_contained":
        return gold_containment(gold, agent)
    if mode == "agent_contained":
        return agent_containment(gold, agent)
    raise ValueError(f"Unknown match mode: {mode}")


def _collapse_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def _text_tokens(text: str) -> list[str]:
    return [t for t in _collapse_text(text).split(" ") if t]


def _text_subsumed_by(inner: Box, outer: Box) -> bool:
    """True when inner text is considered a sub-span of outer (for core extraction)."""
    a = _collapse_text(inner.text)
    b = _collapse_text(outer.text)
    if not a:
        return True
    if not b:
        return False
    if a in b:
        return True
    tok_a = _text_tokens(inner.text)
    tok_b = _text_tokens(outer.text)
    if tok_a and len(tok_a) <= len(tok_b) and tok_a == tok_b[: len(tok_a)]:
        return True
    return False


def _box_subsumed(
    inner: Box,
    outer: Box,
    *,
    subume_threshold: float,
) -> bool:
    if inner.page != outer.page:
        return False
    if outer.area <= 0 or inner.area <= 0:
        return False
    if gold_containment(inner, outer) < subume_threshold:
        return False
    return _text_subsumed_by(inner, outer)


def extract_core_box_indices(
    boxes: list[Box],
    *,
    subume_threshold: float = 0.9,
) -> tuple[list[int], list[int]]:
    """
    Return (core_indices, removed_indices) into *boxes*.

    Larger boxes are kept; smaller boxes on the same page drop when subsumed.
    """
    by_page: dict[int, list[int]] = {}
    for i, b in enumerate(boxes):
        by_page.setdefault(b.page, []).append(i)
    core: list[int] = []
    removed: list[int] = []
    for page in sorted(by_page):
        indices = by_page[page]
        indices.sort(key=lambda i: boxes[i].area, reverse=True)
        page_core: list[int] = []
        for i in indices:
            b = boxes[i]
            if any(
                _box_subsumed(b, boxes[j], subume_threshold=subume_threshold)
                for j in page_core
            ):
                removed.append(i)
            else:
                page_core.append(i)
        core.extend(page_core)
    core.sort()
    removed.sort()
    return core, removed


def _rect_area(rect: Rect) -> float:
    x0, y0, x1, y1 = rect
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _rects_overlap(a: Rect, b: Rect) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _merge_rect(a: Rect, b: Rect) -> Rect:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (
        min(ax0, bx0),
        min(ay0, by0),
        max(ax1, bx1),
        max(ay1, by1),
    )


def _union_rect_area(rects: list[Rect]) -> float:
    if not rects:
        return 0.0
    merged: list[Rect] = list(rects)
    while True:
        merged_any = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if _rects_overlap(merged[i], merged[j]):
                    merged[i] = _merge_rect(merged[i], merged[j])
                    merged.pop(j)
                    merged_any = True
                    break
            if merged_any:
                break
        if not merged_any:
            break
    return sum(_rect_area(r) for r in merged)


def _clip_agent_rect_to_gold(agent: Box, gold: Box) -> Rect | None:
    if agent.page != gold.page:
        return None
    x0 = max(gold.xmin, agent.xmin)
    y0 = max(gold.ymin, agent.ymin)
    x1 = min(gold.xmax, agent.xmax)
    y1 = min(gold.ymax, agent.ymax)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def union_gold_coverage(
    gold: Box,
    agent_boxes: list[Box],
) -> float:
    """Fraction of *gold* area covered by the union of (gold ∩ agent) clips."""
    if gold.area <= 0:
        return 0.0
    clips: list[Rect] = []
    for agent in agent_boxes:
        if agent.page != gold.page:
            continue
        clip = _clip_agent_rect_to_gold(agent, gold)
        if clip is not None:
            clips.append(clip)
    return _union_rect_area(clips) / gold.area


def env_core_subume_threshold() -> float:
    raw = (os.environ.get("AGENT_EVAL_GOLD_CORE_SUBSUME_THRESHOLD") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return 0.9


def env_agent_overlap_epsilon() -> float:
    raw = (os.environ.get("AGENT_EVAL_GOLD_CORE_AGENT_OVERLAP") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return 0.05


def match_boxes_core_coverage(
    gold_boxes: list[Box],
    agent_boxes: list[Box],
    *,
    threshold: float = 0.85,
    subume_threshold: float | None = None,
    agent_overlap_epsilon: float | None = None,
) -> CompareReport:
    """Core gold spans + union agent coverage (recall); core agent precision."""
    subume = (
        env_core_subume_threshold() if subume_threshold is None else subume_threshold
    )
    overlap_eps = (
        env_agent_overlap_epsilon()
        if agent_overlap_epsilon is None
        else agent_overlap_epsilon
    )

    core_gold_idx, removed_gold = extract_core_box_indices(
        gold_boxes, subume_threshold=subume
    )
    core_agent_idx, removed_agent = extract_core_box_indices(
        agent_boxes, subume_threshold=subume
    )

    per_core: list[dict[str, Any]] = []
    matched_core_gold = 0
    unmatched_core_gold: list[int] = []

    for ci, gi in enumerate(core_gold_idx):
        g = gold_boxes[gi]
        cov = union_gold_coverage(g, agent_boxes)
        matched = cov >= threshold
        if matched:
            matched_core_gold += 1
        else:
            unmatched_core_gold.append(ci)
        per_core.append(
            {
                "core_index": ci,
                "source_index": gi,
                "page": g.page,
                "text": g.text,
                "label": g.label,
                "union_coverage": cov,
                "matched": matched,
            }
        )

    core_gold_n = len(core_gold_idx)
    recall = (matched_core_gold / core_gold_n) if core_gold_n else 1.0

    matched_core_agent = 0
    unmatched_core_agent: list[int] = []
    for ci, ai in enumerate(core_agent_idx):
        a = agent_boxes[ai]
        tp = False
        for gi in core_gold_idx:
            g = gold_boxes[gi]
            if g.page != a.page:
                continue
            inter = box_intersection_area(g, a)
            if a.area > 0 and (inter / a.area) >= overlap_eps:
                tp = True
                break
        if tp:
            matched_core_agent += 1
        else:
            unmatched_core_agent.append(ci)

    core_agent_n = len(core_agent_idx)
    precision = (matched_core_agent / core_agent_n) if core_agent_n else 1.0

    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    pages = sorted({b.page for b in gold_boxes} | {b.page for b in agent_boxes})
    strict_pass = (
        core_gold_n == matched_core_gold
        and core_agent_n == matched_core_agent
        and not unmatched_core_gold
        and not unmatched_core_agent
    )

    return CompareReport(
        gold_count=len(gold_boxes),
        agent_count=len(agent_boxes),
        matched_gold=matched_core_gold,
        matched_agent=matched_core_agent,
        unmatched_gold_indices=unmatched_core_gold,
        unmatched_agent_indices=unmatched_core_agent,
        matches=[],
        precision=precision,
        recall=recall,
        f1=f1,
        strict_pass=strict_pass,
        iou_threshold=threshold,
        match_mode="core_coverage_union",
        require_label_match=False,
        label_agreement=None,
        pages=pages,
        scoring="core_coverage",
        core_gold_count=core_gold_n,
        core_agent_count=core_agent_n,
        removed_gold_indices=removed_gold,
        removed_agent_indices=removed_agent,
        per_core_gold=per_core,
    )


def _labels_compatible(gold: Box, agent: Box, require_label_match: bool) -> bool:
    if not require_label_match:
        return True
    g = gold.label.strip().casefold()
    a = agent.label.strip().casefold()
    if not g or not a:
        return True
    return g == a


def match_boxes(
    gold_boxes: list[Box],
    agent_boxes: list[Box],
    *,
    threshold: float = 0.5,
    mode: MatchMode = "iou",
    require_label_match: bool = False,
    many_to_one: bool = False,
) -> CompareReport:
    """
    Greedy match gold↔agent boxes by descending score.

    When ``many_to_one`` is True, multiple agent boxes may match the same gold
    box (agent indices unique; gold may repeat). Precision still uses unique
    agent matches; recall uses unique gold matches.
    """
    pages = sorted({b.page for b in gold_boxes} | {b.page for b in agent_boxes})
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gold_boxes):
        for ai, a in enumerate(agent_boxes):
            if g.page != a.page:
                continue
            if not _labels_compatible(g, a, require_label_match):
                continue
            score = pair_score(g, a, mode)
            if score >= threshold:
                candidates.append((score, gi, ai))
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_gold: set[int] = set()
    used_agent: set[int] = set()
    matches: list[MatchPair] = []
    for score, gi, ai in candidates:
        if ai in used_agent:
            continue
        if not many_to_one and gi in used_gold:
            continue
        used_gold.add(gi)
        used_agent.add(ai)
        matches.append(MatchPair(gold_index=gi, agent_index=ai, score=score, mode=mode))

    matched_gold_set = {m.gold_index for m in matches}
    matched_agent_set = {m.agent_index for m in matches}
    unmatched_gold = [i for i in range(len(gold_boxes)) if i not in matched_gold_set]
    unmatched_agent = [i for i in range(len(agent_boxes)) if i not in matched_agent_set]

    gold_n = len(gold_boxes)
    agent_n = len(agent_boxes)
    matched_g = len(matched_gold_set)
    matched_a = len(matched_agent_set)
    recall = (matched_g / gold_n) if gold_n else 1.0
    precision = (matched_a / agent_n) if agent_n else 1.0
    if precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    label_agreement: float | None = None
    if matches:
        agree = 0
        considered = 0
        for m in matches:
            g = gold_boxes[m.gold_index]
            a = agent_boxes[m.agent_index]
            if not g.label.strip() or not a.label.strip():
                continue
            considered += 1
            if g.label.strip().casefold() == a.label.strip().casefold():
                agree += 1
        if considered:
            label_agreement = agree / considered

    strict_pass = gold_n == matched_g and agent_n == matched_a and unmatched_gold == []

    return CompareReport(
        gold_count=gold_n,
        agent_count=agent_n,
        matched_gold=matched_g,
        matched_agent=matched_a,
        unmatched_gold_indices=unmatched_gold,
        unmatched_agent_indices=unmatched_agent,
        matches=matches,
        precision=precision,
        recall=recall,
        f1=f1,
        strict_pass=strict_pass,
        iou_threshold=threshold,
        match_mode=mode,
        require_label_match=require_label_match,
        label_agreement=label_agreement,
        pages=pages,
        scoring="legacy_box_match",
        core_gold_count=gold_n,
        core_agent_count=agent_n,
    )


def compare_review_csvs(
    gold_csv: str | Path,
    agent_csv: str | Path,
    *,
    threshold: float = 0.5,
    mode: MatchMode = "iou",
    require_label_match: bool = False,
    many_to_one: bool = False,
    scoring: ScoringMode = "legacy_box_match",
) -> CompareReport:
    """Load two review CSVs and return a geometric comparison report."""
    gold_boxes = load_review_boxes(gold_csv)
    agent_boxes = load_review_boxes(agent_csv)
    if scoring == "core_coverage":
        return match_boxes_core_coverage(
            gold_boxes,
            agent_boxes,
            threshold=threshold,
        )
    return match_boxes(
        gold_boxes,
        agent_boxes,
        threshold=threshold,
        mode=mode,
        require_label_match=require_label_match,
        many_to_one=many_to_one,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare gold vs agent review_file.csv boxes (IoU / containment)."
    )
    parser.add_argument("gold_csv", help="Ideal / gold *_review_file.csv")
    parser.add_argument("agent_csv", help="Agent-produced *_review_file.csv")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum score to count as a match (default 0.5).",
    )
    parser.add_argument(
        "--mode",
        choices=("iou", "gold_contained", "agent_contained"),
        default="iou",
        help="Matching score: IoU, gold containment, or agent containment.",
    )
    parser.add_argument(
        "--require-label-match",
        action="store_true",
        help="Only match pairs with the same label (case-insensitive).",
    )
    parser.add_argument(
        "--many-to-one",
        action="store_true",
        help="Allow multiple agent boxes to cover one gold box.",
    )
    parser.add_argument(
        "--scoring",
        choices=("core_coverage", "legacy_box_match"),
        default="legacy_box_match",
        help="core_coverage: core spans + union recall; legacy_box_match: greedy pairwise.",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=None,
        help="Exit 1 if F1 is below this threshold.",
    )
    parser.add_argument(
        "--require-strict",
        action="store_true",
        help="Exit 1 unless every gold and agent box is matched (no extras).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write full report JSON to this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = compare_review_csvs(
        args.gold_csv,
        args.agent_csv,
        threshold=args.iou_threshold,
        mode=args.mode,
        require_label_match=args.require_label_match,
        many_to_one=args.many_to_one,
        scoring=args.scoring,
    )
    summary = {
        "precision": round(report.precision, 4),
        "recall": round(report.recall, 4),
        "f1": round(report.f1, 4),
        "strict_pass": report.strict_pass,
        "gold_count": report.gold_count,
        "agent_count": report.agent_count,
        "matched_gold": report.matched_gold,
        "matched_agent": report.matched_agent,
        "label_agreement": report.label_agreement,
        "mode": report.match_mode,
        "threshold": report.iou_threshold,
        "scoring": report.scoring,
        "core_gold_count": report.core_gold_count,
        "core_agent_count": report.core_agent_count,
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )

    failed = False
    if args.min_f1 is not None and report.f1 < args.min_f1:
        print(
            f"F1 {report.f1:.4f} below --min-f1 {args.min_f1}",
            file=sys.stderr,
        )
        failed = True
    if args.require_strict and not report.strict_pass:
        print(
            "strict_pass=false (unmatched gold or extra agent boxes)", file=sys.stderr
        )
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
