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
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

MatchMode = Literal["iou", "gold_contained", "agent_contained"]


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
    )


def compare_review_csvs(
    gold_csv: str | Path,
    agent_csv: str | Path,
    *,
    threshold: float = 0.5,
    mode: MatchMode = "iou",
    require_label_match: bool = False,
    many_to_one: bool = False,
) -> CompareReport:
    """Load two review CSVs and return a geometric comparison report."""
    return match_boxes(
        load_review_boxes(gold_csv),
        load_review_boxes(agent_csv),
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
