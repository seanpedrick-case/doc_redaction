"""
Column-aware reading order for local OCR boxes (words or line-level detections).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

from tools.config import (
    CONVERT_LINE_TO_WORD_LEVEL,
    LOCAL_OCR_READING_ORDER,
    OCR_COLUMN_FOOTER_ZONE_FRACTION,
    OCR_COLUMN_GAP_MIN_FRACTION,
    OCR_COLUMN_GUTTER_MIN_FRACTION,
    OCR_COLUMN_MAX_BOX_HEIGHT_RATIO,
    OCR_COLUMN_MAX_CONSECUTIVE_GUTTER_GAP,
    OCR_COLUMN_MIN_GUTTER_ROWS,
    OCR_COLUMN_SUBGUTTER_MIN_FRACTION,
    OCR_FULL_SPAN_WIDTH_RATIO,
    OCR_LINE_SPLIT_GAP_FRACTION,
    OCR_LINE_Y_THRESHOLD_FRACTION,
    OCR_LINE_Y_THRESHOLD_MIN_PX,
    PADDLE_PRESERVE_LINE_BOXES,
)

_PADDLE_LINE_OCR_MODELS = (
    "paddle",
    "hybrid-paddle-vlm",
    "hybrid-paddle-inference-server",
)


def should_preserve_paddle_line_boxes(chosen_local_ocr_model: str) -> bool:
    """True when Paddle line boxes should not be split/regrouped in combine_ocr_results."""
    if PADDLE_PRESERVE_LINE_BOXES:
        return chosen_local_ocr_model in _PADDLE_LINE_OCR_MODELS
    return (
        chosen_local_ocr_model in _PADDLE_LINE_OCR_MODELS
        and not CONVERT_LINE_TO_WORD_LEVEL
    )


@dataclass
class _LayoutBox:
    result: Any
    zone: str  # "full_span" | "column"
    column_index: int  # -1 for full_span


def infer_page_dimensions(boxes: Sequence[Any]) -> Tuple[float, float]:
    """Infer page width/height from box extents (same coordinate space as inputs)."""
    if not boxes:
        return 1000.0, 1000.0
    page_width = max(float(b.left) + float(b.width) for b in boxes)
    page_height = max(float(b.top) + float(b.height) for b in boxes)
    return max(page_width, 1.0), max(page_height, 1.0)


def _median_box_width(boxes: Sequence[Any]) -> float:
    widths = [float(b.width) for b in boxes if float(b.width) > 0]
    if not widths:
        return 1.0
    return float(statistics.median(widths))


def _median_box_height(boxes: Sequence[Any]) -> float:
    heights = [float(b.height) for b in boxes if float(b.height) > 0]
    if not heights:
        return 1.0
    return float(statistics.median(heights))


def _median_line_height(boxes: Sequence[Any]) -> float:
    heights = [float(b.height) for b in boxes if float(b.height) > 0]
    if not heights:
        return 12.0
    return float(statistics.median(heights))


def is_full_span_box(box: Any, page_width: float, full_span_width_ratio: float) -> bool:
    if page_width <= 0:
        return False
    return (float(box.width) / page_width) >= full_span_width_ratio


def _coordinates_likely_normalized(page_width: float, page_height: float) -> bool:
    """True when extents look like 0-1 relative coords rather than pixels."""
    return page_width <= 2.0 and page_height <= 2.0


def compute_y_threshold(
    page_height: float,
    boxes: Sequence[Any],
    y_threshold_min_px: float = OCR_LINE_Y_THRESHOLD_MIN_PX,
    y_threshold_fraction: float = OCR_LINE_Y_THRESHOLD_FRACTION,
    page_width: float | None = None,
) -> float:
    median_h = _median_line_height(boxes)
    from_fraction = y_threshold_fraction * page_height
    from_median = 0.5 * median_h
    if page_width is not None and _coordinates_likely_normalized(
        page_width, page_height
    ):
        return max(from_fraction, from_median, 0.001)
    return max(y_threshold_min_px, from_fraction, from_median)


def compute_column_gap_threshold(
    page_width: float,
    boxes: Sequence[Any],
    column_gap_min_fraction: float = OCR_COLUMN_GAP_MIN_FRACTION,
) -> float:
    median_w = _median_box_width(boxes)
    # Use 1.2× median width so typical 3-column gutters still split; avoids tiny
    # x-center gaps from indentation on single-column pages when gutter check passes.
    return max(column_gap_min_fraction * page_width, 1.2 * median_w)


def _group_boxes_into_rows(boxes: Sequence[Any], y_threshold: float) -> List[List[Any]]:
    """Group boxes into text rows by similar top (sorted top-to-left)."""
    sorted_boxes = sorted(boxes, key=lambda b: (float(b.top), float(b.left)))
    rows: List[List[Any]] = []
    i = 0
    while i < len(sorted_boxes):
        row = [sorted_boxes[i]]
        row_top = float(sorted_boxes[i].top)
        i += 1
        while (
            i < len(sorted_boxes)
            and float(sorted_boxes[i].top) - row_top <= y_threshold
        ):
            row.append(sorted_boxes[i])
            i += 1
        rows.append(row)
    return rows


def has_side_by_side_columns(
    boxes: Sequence[Any],
    page_width: float,
    page_height: float,
    full_span_width_ratio: float = OCR_FULL_SPAN_WIDTH_RATIO,
    gutter_min_fraction: float = OCR_COLUMN_GUTTER_MIN_FRACTION,
    min_gutter_rows: int = OCR_COLUMN_MIN_GUTTER_ROWS,
    max_box_height_ratio: float = OCR_COLUMN_MAX_BOX_HEIGHT_RATIO,
    max_consecutive_gutter_gap: float = OCR_COLUMN_MAX_CONSECUTIVE_GUTTER_GAP,
    footer_zone_fraction: float = OCR_COLUMN_FOOTER_ZONE_FRACTION,
) -> bool:
    """True when the page body contains a coherent multi-column band.

    Gutter rows (text rows with a clear horizontal gap between adjacent boxes) are
    collected and then clustered by their y-positions.  Column mode is triggered only
    when a **consecutive cluster** of gutter rows satisfies ALL of the following:

    * cluster size ≥ ``min_gutter_rows``
    * consecutive y-gap inside the cluster ≤ ``max_consecutive_gutter_gap * page_height``
    * the cluster's topmost row is above ``footer_zone_fraction * page_height``

    The third condition prevents a pair of side-by-side signature blocks at the page
    bottom (y ≥ 0.75) from triggering column-major reading order on the single-column
    body text above them.  The first two conditions prevent an isolated header gutter
    (logo | title) from being counted together with distant footer rows.

    Additional guards:

    * ``max_box_height_ratio`` — boxes whose height exceeds this multiple of the page
      median are excluded.  Image regions misdetected as text (e.g. a city-seal stamp)
      would otherwise create spurious gutter rows.
    """
    column_boxes = [
        b for b in boxes if not is_full_span_box(b, page_width, full_span_width_ratio)
    ]
    if len(column_boxes) < 2:
        return False

    # Exclude abnormally tall boxes (images / multi-line artefacts).
    median_h = _median_box_height(column_boxes)
    max_h = max_box_height_ratio * median_h
    column_boxes = [b for b in column_boxes if float(b.height) <= max_h]
    if len(column_boxes) < 2:
        return False

    y_thresh = compute_y_threshold(page_height, column_boxes, page_width=page_width)
    gutter_min = max(
        gutter_min_fraction * page_width,
        0.2 * _median_box_width(column_boxes),
    )

    # Collect the top y-position of each row that shows a clear gutter.
    gutter_tops: List[float] = []
    for row in _group_boxes_into_rows(column_boxes, y_thresh):
        if len(row) < 2:
            continue
        row_top = min(float(b.top) for b in row)
        row_sorted = sorted(row, key=lambda b: float(b.left))
        for j in range(len(row_sorted) - 1):
            right = float(row_sorted[j].left) + float(row_sorted[j].width)
            next_left = float(row_sorted[j + 1].left)
            gap = next_left - right
            if next_left >= right and gap + 1e-6 >= gutter_min:
                gutter_tops.append(row_top)
                break  # one gutter per row

    # Fast exit: not enough gutter rows in total.
    if len(gutter_tops) < min_gutter_rows:
        return False

    # Cluster consecutive gutter rows (y-gap ≤ max_gap within a cluster).
    # A cluster qualifies only when it has >= min_gutter_rows rows AND its
    # topmost row is above the footer zone.
    max_gap = max_consecutive_gutter_gap * page_height
    footer_start = footer_zone_fraction * page_height

    gutter_tops.sort()
    i = 0
    while i < len(gutter_tops):
        cluster_top = gutter_tops[i]
        cluster_size = 1
        j = i
        while (
            j + 1 < len(gutter_tops)
            and (gutter_tops[j + 1] - gutter_tops[j]) <= max_gap
        ):
            j += 1
            cluster_size += 1
        if cluster_size >= min_gutter_rows and cluster_top < footer_start:
            return True
        i = j + 1

    return False


def should_use_column_reading_order(
    boxes: Sequence[Any],
    page_width: float,
    page_height: float,
    reading_order_mode: str | None = None,
    full_span_width_ratio: float = OCR_FULL_SPAN_WIDTH_RATIO,
    gutter_min_fraction: float = OCR_COLUMN_GUTTER_MIN_FRACTION,
    min_gutter_rows: int = OCR_COLUMN_MIN_GUTTER_ROWS,
    max_box_height_ratio: float = OCR_COLUMN_MAX_BOX_HEIGHT_RATIO,
    max_consecutive_gutter_gap: float = OCR_COLUMN_MAX_CONSECUTIVE_GUTTER_GAP,
    footer_zone_fraction: float = OCR_COLUMN_FOOTER_ZONE_FRACTION,
) -> bool:
    """Whether to apply column-major reading order (vs legacy top-left)."""
    mode = (reading_order_mode or LOCAL_OCR_READING_ORDER).strip().lower()
    if mode == "legacy":
        return False
    if mode != "column":
        return False
    return has_side_by_side_columns(
        boxes,
        page_width,
        page_height,
        full_span_width_ratio=full_span_width_ratio,
        gutter_min_fraction=gutter_min_fraction,
        min_gutter_rows=min_gutter_rows,
        max_box_height_ratio=max_box_height_ratio,
        max_consecutive_gutter_gap=max_consecutive_gutter_gap,
        footer_zone_fraction=footer_zone_fraction,
    )


def _sort_single_column_reading_order(
    boxes: Sequence[Any],
    page_width: float,
    full_span_width_ratio: float = OCR_FULL_SPAN_WIDTH_RATIO,
) -> List[Any]:
    """Full-span lines first, then remaining lines by (top, left)."""
    full_span = sorted(
        [b for b in boxes if is_full_span_box(b, page_width, full_span_width_ratio)],
        key=lambda b: (float(b.top), float(b.left)),
    )
    rest = sorted(
        [
            b
            for b in boxes
            if not is_full_span_box(b, page_width, full_span_width_ratio)
        ],
        key=lambda b: (float(b.top), float(b.left)),
    )
    return full_span + rest


def detect_column_split_xpoints(
    boxes: Sequence[Any],
    page_width: float,
    page_height: float,
    full_span_width_ratio: float = OCR_FULL_SPAN_WIDTH_RATIO,
    subgutter_min_fraction: float = OCR_COLUMN_SUBGUTTER_MIN_FRACTION,
    min_gutter_rows: int = OCR_COLUMN_MIN_GUTTER_ROWS,
    max_box_height_ratio: float = OCR_COLUMN_MAX_BOX_HEIGHT_RATIO,
    max_consecutive_gutter_gap: float = OCR_COLUMN_MAX_CONSECUTIVE_GUTTER_GAP,
    footer_zone_fraction: float = OCR_COLUMN_FOOTER_ZONE_FRACTION,
) -> List[float]:
    """Return sorted ``(split_x, y_min)`` pairs describing column split points.

    Called only on pages already confirmed as multi-column (by
    ``has_side_by_side_columns``).  Uses ``subgutter_min_fraction`` — a lower threshold
    than the standard ``OCR_COLUMN_GUTTER_MIN_FRACTION`` — so narrow sub-column gutters
    (e.g. a 1.9 % gap on a two-page spread) are detected in addition to the wide
    macro-column gutter that triggered column mode.

    Each returned pair is ``(split_x, y_min)`` where *split_x* is placed in the
    narrowest actual gap between the two columns (see below) and *y_min* is the topmost
    row in the qualifying cluster.  Callers **must** apply each split only to boxes
    whose ``top >= y_min``; this prevents a narrow sub-column gutter from mis-splitting
    full-width content that sits above the two-column section.

    Algorithm
    ---------
    1. Scan every text row for horizontal gaps ≥ subgutter_min.  Record all
       ``(row_top, right_edge, next_left_edge)`` triples (one per gap per row).
    2. Cluster by per-row midpoint ``(right + next_left) / 2`` (tolerance 6 % of page
       width) so that entries from the same physical column boundary are grouped together.
    3. Within each x-cluster, find the longest run of consecutive row-top values
       (gap ≤ ``max_consecutive_gutter_gap * page_height``).  If the run is long enough
       (≥ ``min_gutter_rows``) and starts above the footer zone, compute
       ``split_x = (max_right_in_run + min_next_left_in_run) / 2``.  This "stable
       midpoint" sits in the narrowest guaranteed gap between the two columns, so it is
       correct for both word-level boxes (where per-row midpoints are consistent) and
       line-level boxes (where a short left-column line gives a very different midpoint
       than a long one, but the right-column left edge stays constant).
    4. Return all qualifying pairs sorted by split_x (left-to-right).
    """
    column_boxes = [
        b for b in boxes if not is_full_span_box(b, page_width, full_span_width_ratio)
    ]
    if len(column_boxes) < 2:
        return []

    median_h = _median_box_height(column_boxes)
    max_h = max_box_height_ratio * median_h
    column_boxes = [b for b in column_boxes if float(b.height) <= max_h]
    if len(column_boxes) < 2:
        return []

    y_thresh = compute_y_threshold(page_height, column_boxes, page_width=page_width)
    # Use a small coefficient (0.05) rather than 0.2 so that line-level OCR boxes
    # (median width ~0.18) do not inflate gutter_min to the point where narrow
    # column gutters (e.g. 1.9–3.2 % of page width) are excluded from detection,
    # which would shift the computed split point leftward into col-A text.
    gutter_min = max(
        subgutter_min_fraction * page_width,
        0.05 * _median_box_width(column_boxes),
    )

    # Step 1: collect ALL gutter observations (not just the first per row).
    # Each observation is (row_top, right_edge_of_left_group, left_edge_of_right_group).
    # Storing both edges lets us compute a "stable midpoint" later (step 3) that is
    # invariant to left-column line length.
    GutterObs = Tuple[float, float, float]  # (row_top, right, next_left)
    gutter_obs: List[GutterObs] = []
    for row in _group_boxes_into_rows(column_boxes, y_thresh):
        if len(row) < 2:
            continue
        row_top = min(float(b.top) for b in row)
        row_sorted = sorted(row, key=lambda b: float(b.left))
        for j in range(len(row_sorted) - 1):
            right = float(row_sorted[j].left) + float(row_sorted[j].width)
            next_left = float(row_sorted[j + 1].left)
            gap = next_left - right
            if next_left >= right and gap + 1e-6 >= gutter_min:
                gutter_obs.append((row_top, right, next_left))

    if len(gutter_obs) < min_gutter_rows:
        return []

    # Step 2: cluster by per-row midpoint (right + next_left) / 2.
    x_tolerance = 0.06 * page_width
    by_x = sorted(gutter_obs, key=lambda obs: (obs[1] + obs[2]) / 2.0)
    x_clusters: List[List[GutterObs]] = [[by_x[0]]]
    for obs in by_x[1:]:
        last = x_clusters[-1][-1]
        last_mid = (last[1] + last[2]) / 2.0
        this_mid = (obs[1] + obs[2]) / 2.0
        if this_mid - last_mid <= x_tolerance:
            x_clusters[-1].append(obs)
        else:
            x_clusters.append([obs])

    # Steps 3–4: find qualifying consecutive runs within each x-cluster.
    # split_x = (max_right_in_run + min_next_left_in_run) / 2  — the midpoint of the
    # narrowest guaranteed gap between the two columns.  This is stable regardless of
    # whether individual rows have short or long left-column lines.
    max_gap = max_consecutive_gutter_gap * page_height
    footer_start = footer_zone_fraction * page_height

    split_xpoints: List[Tuple[float, float]] = []
    for x_cluster in x_clusters:
        tops = sorted(obs[0] for obs in x_cluster)
        i = 0
        while i < len(tops):
            cluster_top = tops[i]
            j = i
            while j + 1 < len(tops) and (tops[j + 1] - tops[j]) <= max_gap:
                j += 1
            cluster_size = j - i + 1
            if cluster_size >= min_gutter_rows and cluster_top < footer_start:
                run_obs = [obs for obs in x_cluster if tops[i] <= obs[0] <= tops[j]]
                max_right = max(obs[1] for obs in run_obs)
                min_next_left = min(obs[2] for obs in run_obs)
                stable_split = (max_right + min_next_left) / 2.0
                split_xpoints.append((stable_split, cluster_top))
                break
            i = j + 1

    split_xpoints.sort(
        key=lambda t: t[0]
    )  # sort by x so column indices are assigned left-to-right
    return split_xpoints


def assign_layout_boxes(
    boxes: Sequence[Any],
    page_width: float,
    page_height: float,
    full_span_width_ratio: float = OCR_FULL_SPAN_WIDTH_RATIO,
    column_gap_min_fraction: float = OCR_COLUMN_GAP_MIN_FRACTION,
) -> List[_LayoutBox]:
    """Classify boxes as full-span or column and assign column indices."""
    if not boxes:
        return []

    column_candidates = []
    layout: List[_LayoutBox] = []

    for box in boxes:
        if is_full_span_box(box, page_width, full_span_width_ratio):
            layout.append(_LayoutBox(result=box, zone="full_span", column_index=-1))
        else:
            column_candidates.append(box)

    if not column_candidates:
        return layout

    use_columns = should_use_column_reading_order(
        boxes, page_width, page_height, full_span_width_ratio=full_span_width_ratio
    )
    if not use_columns:
        for box in column_candidates:
            layout.append(_LayoutBox(result=box, zone="column", column_index=0))
        return layout

    # ── Structural gutter-based column assignment (primary path) ──────────────
    # Use the fine-grained gutter scan to find exact column split x-coordinates.
    # This is more reliable than centre-gap clustering for narrow gutters (e.g. a
    # 1.9 % gutter on a two-page spread whose word-centre gap of 2.5 % is below the
    # 4 % centre-gap threshold used in the fallback path).
    split_xpoints = detect_column_split_xpoints(
        boxes, page_width, page_height, full_span_width_ratio=full_span_width_ratio
    )

    if split_xpoints:
        for box in column_candidates:
            center_x = float(box.left) + float(box.width) / 2.0
            left_x = float(box.left)
            right_x = left_x + float(box.width)
            box_top = float(box.top)
            # Only apply splits whose qualifying gutter cluster started at or before
            # this box's vertical position.  This prevents a narrow sub-column gutter
            # (detected further down the page) from mis-splitting content that sits
            # above the two-column section (e.g. a full-width intro paragraph).
            active = [xp for xp, y_min in split_xpoints if box_top >= y_min]
            # A box genuinely spans a column gutter only if it extends substantially
            # to BOTH sides of the split point (margin = 1 % of page width).  A simple
            # left_x < xp < right_x test incorrectly marks single words that sit at the
            # right margin of a column (e.g. "when" ending at right=0.242 with split at
            # 0.238 from line-level box gutter detection) as full-span, displacing them
            # to the end of the column sequence.
            margin = 0.01 * page_width
            spans_gutter = any(
                left_x < xp - margin and right_x > xp + margin for xp in active
            )
            if spans_gutter:
                layout.append(_LayoutBox(result=box, zone="full_span", column_index=-1))
            else:
                col_idx = sum(1 for xp in active if center_x >= xp)
                layout.append(
                    _LayoutBox(result=box, zone="column", column_index=col_idx)
                )
        return layout

    # ── Fallback: centre-gap clustering ───────────────────────────────────────
    median_w = _median_box_width(column_candidates)
    # Wide lines (partial headers) bridge x-clusters; treat as full-span for ordering.
    bridge_width = max(0.2 * page_width, 1.25 * median_w)
    narrow_candidates = [b for b in column_candidates if float(b.width) <= bridge_width]
    for box in column_candidates:
        if float(box.width) > bridge_width:
            layout.append(_LayoutBox(result=box, zone="full_span", column_index=-1))

    if not narrow_candidates:
        return layout

    column_gap = compute_column_gap_threshold(
        page_width, narrow_candidates, column_gap_min_fraction
    )
    centers = [(float(b.left) + float(b.width) / 2.0, b) for b in narrow_candidates]
    centers.sort(key=lambda item: item[0])
    clusters: List[List[Any]] = [[centers[0][1]]]
    # Track the rightmost (max) x-centre in each cluster.  Using the rolling mean
    # instead caused the mean to drift far behind the rightmost box already added,
    # making the gap to the next box look artificially large and triggering false
    # column splits deep inside a single physical column when word-level boxes are
    # used (many short words with varying centres across successive text rows).
    cluster_max_x: List[float] = [centers[0][0]]

    for x_center, box in centers[1:]:
        if x_center - cluster_max_x[-1] > column_gap:
            clusters.append([box])
            cluster_max_x.append(x_center)
        else:
            clusters[-1].append(box)
            cluster_max_x[-1] = max(cluster_max_x[-1], x_center)

    cluster_mean_x = [
        statistics.mean(float(b.left) + float(b.width) / 2.0 for b in cluster)
        for cluster in clusters
    ]
    cluster_order = sorted(range(len(clusters)), key=lambda i: cluster_mean_x[i])
    cluster_to_column = {cluster_order[i]: i for i in range(len(clusters))}

    for cluster_idx, cluster in enumerate(clusters):
        col_idx = cluster_to_column[cluster_idx]
        for box in cluster:
            layout.append(_LayoutBox(result=box, zone="column", column_index=col_idx))

    return layout


def sort_reading_order(
    boxes: Sequence[Any],
    page_width: float | None = None,
    page_height: float | None = None,
    full_span_width_ratio: float = OCR_FULL_SPAN_WIDTH_RATIO,
    column_gap_min_fraction: float = OCR_COLUMN_GAP_MIN_FRACTION,
) -> List[Any]:
    """
    Order boxes for human reading: full-width lines first (by top), then each text
    column left-to-right with lines sorted top-to-bottom within the column.
    """
    if not boxes:
        return []

    if page_width is None or page_height is None:
        page_width, page_height = infer_page_dimensions(boxes)

    if not should_use_column_reading_order(boxes, page_width, page_height):
        return _sort_single_column_reading_order(
            boxes, page_width, full_span_width_ratio=full_span_width_ratio
        )

    layout_boxes = assign_layout_boxes(
        boxes,
        page_width,
        page_height,
        full_span_width_ratio=full_span_width_ratio,
        column_gap_min_fraction=column_gap_min_fraction,
    )

    full_span_lbs = sorted(
        [lb for lb in layout_boxes if lb.zone == "full_span"],
        key=lambda lb: (float(lb.result.top), float(lb.result.left)),
    )
    column_lbs = [lb for lb in layout_boxes if lb.zone == "column"]
    num_columns = 0
    for lb in column_lbs:
        num_columns = max(num_columns, lb.column_index + 1)

    ordered: List[Any] = []
    ordered.extend(lb.result for lb in full_span_lbs)

    for col_idx in range(num_columns):
        col_boxes = sorted(
            [lb for lb in column_lbs if lb.column_index == col_idx],
            key=lambda lb: (float(lb.result.top), float(lb.result.left)),
        )
        ordered.extend(lb.result for lb in col_boxes)

    return ordered


def _finalize_line(
    line: List[Any],
    gap_threshold: float | None,
) -> List[List[Any]]:
    """Sort a collected line by left position and split it wherever consecutive
    boxes have a horizontal gap larger than ``gap_threshold``.

    Returns a list of one or more sub-lines.  When ``gap_threshold`` is None the
    line is returned as-is (just sorted), with no splitting.

    This post-processing step handles the case where boxes from two distinct
    layout elements (e.g. two signature blocks on opposite sides of the page)
    were placed in the same y-band and collected into one line group regardless
    of their x-positions.  Sorting first guarantees that the gap check is
    applied left-to-right even when boxes arrived in a different order.
    """
    if not line:
        return []
    sorted_line = sorted(line, key=lambda b: float(b.left))
    if not gap_threshold:
        return [sorted_line]
    sub_lines: List[List[Any]] = []
    current: List[Any] = [sorted_line[0]]
    for box in sorted_line[1:]:
        prev_right = max(float(b.left) + float(b.width) for b in current)
        if float(box.left) - prev_right > gap_threshold:
            sub_lines.append(current)
            current = [box]
        else:
            current.append(box)
    sub_lines.append(current)
    return sub_lines


def group_into_lines_legacy(
    ocr_results: Sequence[Any],
    y_threshold: float = 12.0,
    page_width: float | None = None,
    line_split_gap_fraction: float = OCR_LINE_SPLIT_GAP_FRACTION,
) -> List[List[Any]]:
    """Original top-then-left grouping (single-column assumption).

    If ``page_width`` is supplied, two gap guards prevent side-by-side elements
    from being concatenated into a single line:

    1. **Rightward gap (build time)** — if the next box starts more than
       ``line_split_gap_fraction * page_width`` to the right of the current line's
       rightmost edge, a new line is started immediately.

    2. **Post-processing split** — after all boxes sharing a y-band have been
       collected into one group, the group is sorted by left position and then
       split wherever consecutive boxes have an internal gap exceeding the same
       threshold.  This catches cases where two elements on opposite sides of the
       page (e.g. "Giuliani" on the left and "Ken" on the right) share the same
       nominal y-position but arrive in right-before-left sort order, which the
       build-time check alone cannot handle.
    """
    lines: List[List[Any]] = []
    current_line: List[Any] = []
    gap_threshold = (line_split_gap_fraction * page_width) if page_width else None

    for result in sorted(ocr_results, key=lambda x: (float(x.top), float(x.left))):
        if not current_line:
            current_line.append(result)
            continue

        top_aligned = abs(float(result.top) - float(current_line[0].top)) <= y_threshold

        # Build-time rightward gap: if the next box starts far to the right of
        # everything collected so far, treat it as a new line immediately.
        if top_aligned and gap_threshold is not None:
            prev_right = max(float(b.left) + float(b.width) for b in current_line)
            if float(result.left) - prev_right > gap_threshold:
                top_aligned = False

        if top_aligned:
            current_line.append(result)
        else:
            lines.extend(_finalize_line(current_line, gap_threshold))
            current_line = [result]

    if current_line:
        lines.extend(_finalize_line(current_line, gap_threshold))
    return lines


def group_into_lines(
    ordered_boxes: Sequence[Any],
    page_width: float,
    page_height: float,
    y_threshold: float | None = None,
    line_split_gap_fraction: float = OCR_LINE_SPLIT_GAP_FRACTION,
) -> List[List[Any]]:
    """
    Group ordered boxes into lines without merging across columns or full-span zones.

    An additional horizontal gap guard (``line_split_gap_fraction * page_width``)
    prevents boxes that are vertically aligned but horizontally far apart from being
    merged into the same line, even when they share the same column assignment.
    """
    if not ordered_boxes:
        return []

    if y_threshold is None:
        y_threshold = compute_y_threshold(
            page_height, ordered_boxes, page_width=page_width
        )

    page_width, page_height = page_width or 1.0, page_height or 1.0
    gap_threshold = line_split_gap_fraction * page_width
    layout_boxes = assign_layout_boxes(ordered_boxes, page_width, page_height)
    layout_by_result = {id(lb.result): lb for lb in layout_boxes}

    lines: List[List[Any]] = []
    current_line: List[Any] = []
    current_lb: _LayoutBox | None = None

    for box in ordered_boxes:
        lb = layout_by_result.get(id(box))
        if lb is None:
            lb = _LayoutBox(result=box, zone="column", column_index=0)

        if not current_line:
            current_line = [box]
            current_lb = lb
            continue

        same_zone = lb.zone == current_lb.zone
        same_column = lb.column_index == current_lb.column_index
        top_aligned = abs(float(box.top) - float(current_line[0].top)) <= y_threshold

        if same_zone and same_column and top_aligned:
            prev_right = max(float(b.left) + float(b.width) for b in current_line)
            if float(box.left) - prev_right > gap_threshold:  # build-time rightward gap
                lines.extend(_finalize_line(current_line, gap_threshold))
                current_line = [box]
                current_lb = lb
            else:
                current_line.append(box)
        else:
            lines.extend(_finalize_line(current_line, gap_threshold))
            current_line = [box]
            current_lb = lb

    if current_line:
        lines.extend(_finalize_line(current_line, gap_threshold))
    return lines


def group_boxes_preserving_lines(ordered_boxes: Sequence[Any]) -> List[List[Any]]:
    """One input box per output line (Paddle line-level fast path)."""
    return [[box] for box in ordered_boxes]


def _reorder_lines_column_major(
    lines: List[List[Any]],
    page_width: float,
    word_median_w: float,
) -> List[List[Any]]:
    """Secondary sort: cluster line groups by their leftmost x-position so that any
    sub-column structure that was too narrow for word-level cluster detection (the
    primary ``assign_layout_boxes`` pass) is resolved into proper column-major order.

    This is needed when two adjacent text columns share the same macro-column cluster
    (because their word-centre gap is below ``column_gap_threshold``).  After
    ``_finalize_line`` has already split y-bands horizontally, the resulting sub-lines
    have well-separated *left* starting positions — far larger than the
    ``OCR_LINE_SPLIT_GAP_FRACTION`` used for splitting.  Clustering on those left
    positions with that fraction as the gap threshold reliably groups them into the
    correct sub-columns and re-orders the output.
    """
    if len(lines) <= 1:
        return lines

    gap = max(OCR_LINE_SPLIT_GAP_FRACTION * page_width, 1.2 * word_median_w)

    # Sort lines by their leftmost word, then cluster runs of similar left positions.
    indexed = sorted(enumerate(lines), key=lambda x: min(float(b.left) for b in x[1]))
    clusters: List[List[int]] = [[indexed[0][0]]]
    cluster_max_left: List[float] = [min(float(b.left) for b in indexed[0][1])]

    for orig_idx, line in indexed[1:]:
        left = min(float(b.left) for b in line)
        if left - cluster_max_left[-1] > gap:
            clusters.append([orig_idx])
            cluster_max_left.append(left)
        else:
            clusters[-1].append(orig_idx)
            cluster_max_left[-1] = max(cluster_max_left[-1], left)

    if len(clusters) == 1:
        return lines  # Single sub-column — original order already correct.

    # Sort clusters left-to-right, then within each cluster sort lines top-to-bottom.
    cluster_mean_left = [
        statistics.mean(min(float(b.left) for b in lines[i]) for i in c)
        for c in clusters
    ]
    ordered: List[List[Any]] = []
    for ci in sorted(range(len(clusters)), key=lambda i: cluster_mean_left[i]):
        col_lines = sorted(
            [lines[i] for i in clusters[ci]],
            key=lambda ln: min(float(b.top) for b in ln),
        )
        ordered.extend(col_lines)
    return ordered


def build_line_groups(
    ocr_results: Sequence[Any],
    reading_order_mode: str | None = None,
    preserve_line_boxes: bool = False,
    y_threshold: float | None = None,
) -> Tuple[List[List[Any]], float, float]:
    """
    Return line groups and inferred page dimensions.
    """
    if not ocr_results:
        return [], 1000.0, 1000.0

    page_width, page_height = infer_page_dimensions(ocr_results)
    mode = reading_order_mode or LOCAL_OCR_READING_ORDER

    use_columns = should_use_column_reading_order(
        ocr_results, page_width, page_height, reading_order_mode=mode
    )

    if not use_columns:
        if y_threshold is not None:
            yt = y_threshold
        elif _coordinates_likely_normalized(page_width, page_height):
            yt = compute_y_threshold(page_height, ocr_results, page_width=page_width)
        else:
            yt = OCR_LINE_Y_THRESHOLD_MIN_PX
        return (
            group_into_lines_legacy(ocr_results, y_threshold=yt, page_width=page_width),
            page_width,
            page_height,
        )

    ordered = sort_reading_order(ocr_results, page_width, page_height)

    if preserve_line_boxes:
        return group_boxes_preserving_lines(ordered), page_width, page_height

    yt = y_threshold
    if yt is None:
        yt = compute_y_threshold(page_height, ocr_results, page_width=page_width)
    lines = group_into_lines(ordered, page_width, page_height, y_threshold=yt)

    # Secondary pass: re-sort the produced line groups by their left-edge cluster so
    # that sub-columns which were too narrow to split at word level are still output in
    # proper column-major order (all left sub-column lines before right sub-column).
    if lines:
        word_mw = _median_box_width(list(ocr_results))
        lines = _reorder_lines_column_major(lines, page_width, word_mw)

    return lines, page_width, page_height


def _order_line_boxes_for_reading(
    line_results: Sequence[Any],
    page_width: float,
    page_height: float,
    reading_order_mode: str | None = None,
) -> List[Any]:
    """Return line boxes in reading order (column-aware or legacy top-left)."""
    if not line_results:
        return []
    if not should_use_column_reading_order(
        line_results,
        page_width,
        page_height,
        reading_order_mode=reading_order_mode,
    ):
        return _sort_single_column_reading_order(line_results, page_width)
    return sort_reading_order(
        line_results,
        page_width=page_width,
        page_height=page_height,
    )


def reorder_structured_text_lines(
    line_results: Sequence[Any],
    lines_char_groups: Sequence[Any],
    page_data: dict,
    *,
    page_width: float,
    page_height: float,
    start_line_number: int = 1,
    reading_order_mode: str | None = None,
) -> Tuple[List[Any], List[Any], dict]:
    """
    Reorder PyMuPDF/pdfminer structured page lines for multi-column reading order.

    Mirrors the local OCR pipeline (build_line_groups) as closely as possible:

    1. Column-aware sort via sort_reading_order / assign_layout_boxes /
       detect_column_split_xpoints (or legacy top-left for single-column pages).
    2. Y-band grouping via group_into_lines — merges any same-row split boxes
       (e.g. mixed-font spans that PyMuPDF emitted as separate lines) and splits
       horizontally-disparate boxes via _finalize_line.
    3. Secondary sub-column pass via _reorder_lines_column_major — ensures proper
       column-major order when sub-columns sit within a single macro-column.
    4. Each resulting line group is merged into a single box (OCRResult), with
       char_groups and page_data["results"] kept in sync.

    When a group contains only one box (the common case for clean PDF text), no
    merge is performed — the original box is just renumbered in-place.
    """
    if not line_results:
        return list(line_results), list(lines_char_groups), page_data

    if len(line_results) != len(lines_char_groups):
        raise ValueError(
            "line_results and lines_char_groups must have the same length "
            f"({len(line_results)} != {len(lines_char_groups)})"
        )

    # Build lookups keyed by object identity (survives sort/group passes).
    id_to_char_group = {
        id(line): lines_char_groups[i] for i, line in enumerate(line_results)
    }
    results_by_old_line = dict(page_data.get("results") or {})

    # ── Step 1: column-aware sort ──────────────────────────────────────────────
    # Determine column mode once so we can share it with steps 2 & 3.
    use_column = should_use_column_reading_order(
        line_results,
        page_width,
        page_height,
        reading_order_mode=reading_order_mode,
    )
    if use_column:
        ordered = sort_reading_order(
            line_results, page_width=page_width, page_height=page_height
        )
    else:
        ordered = _sort_single_column_reading_order(line_results, page_width)

    # ── Steps 2 & 3: y-band grouping + secondary sub-column pass ──────────────
    # These match build_line_groups (local OCR route) and are applied in column
    # mode only.  For single-column / legacy pages PyMuPDF lines are already
    # formed, so a simple per-line group is sufficient.
    if use_column:
        yt = compute_y_threshold(page_height, list(line_results), page_width=page_width)
        groups = group_into_lines(ordered, page_width, page_height, y_threshold=yt)
        if groups:
            word_mw = _median_box_width(list(line_results))
            groups = _reorder_lines_column_major(groups, page_width, word_mw)
    else:
        groups = [[box] for box in ordered]

    # ── Step 4: reconstruct output, merging boxes within each group ───────────
    new_line_results: List[Any] = []
    new_char_groups: List[Any] = []
    new_results: dict = {}

    for i, group in enumerate(groups):
        if not group:
            continue
        new_line_no = start_line_number + i

        if len(group) == 1:
            # Fast path: single-box group — renumber in-place, no merge needed.
            box = group[0]
            old_line_no = box.line
            box.line = new_line_no
            merged_chars = id_to_char_group.get(id(box), [])
            old_entry = results_by_old_line.get(f"text_line_{old_line_no}")
            if old_entry is not None:
                new_entry = dict(old_entry)
                new_entry["line"] = new_line_no
                new_results[f"text_line_{new_line_no}"] = new_entry
            else:
                new_results[f"text_line_{new_line_no}"] = {
                    "line": new_line_no,
                    "text": box.text,
                    "bounding_box": [
                        box.left,
                        box.top,
                        box.left + box.width,
                        box.top + box.height,
                    ],
                    "words": [],
                    "conf": getattr(box, "conf", 100.0),
                }
        else:
            # Merge group: combine text, union bounding-box, concatenate chars/words.
            # Capture old line numbers BEFORE mutating any box.
            old_line_nos_group = [b.line for b in group]
            group_text = " ".join(b.text for b in group)
            group_left = min(float(b.left) for b in group)
            group_top = min(float(b.top) for b in group)
            group_right = max(float(b.left) + float(b.width) for b in group)
            group_bottom = max(float(b.top) + float(b.height) for b in group)
            group_conf = sum(getattr(b, "conf", 100.0) for b in group) / len(group)

            # Mutate the first box as the merged representative (OCRResult is a
            # mutable dataclass, so field assignment is safe).
            box = group[0]
            box.text = group_text
            box.left = group_left
            box.top = group_top
            box.width = group_right - group_left
            box.height = group_bottom - group_top
            box.conf = group_conf
            box.line = new_line_no

            merged_chars: List[Any] = []
            merged_words: List[Any] = []
            for b, old_ln in zip(group, old_line_nos_group):
                merged_chars.extend(id_to_char_group.get(id(b), []))
                old_entry = results_by_old_line.get(f"text_line_{old_ln}")
                if old_entry:
                    merged_words.extend(old_entry.get("words", []))

            new_results[f"text_line_{new_line_no}"] = {
                "line": new_line_no,
                "text": group_text,
                "bounding_box": [group_left, group_top, group_right, group_bottom],
                "words": merged_words,
                "conf": group_conf,
            }

        new_line_results.append(box)
        new_char_groups.append(merged_chars)

    new_page_data = dict(page_data)
    new_page_data["results"] = new_results
    return new_line_results, new_char_groups, new_page_data
