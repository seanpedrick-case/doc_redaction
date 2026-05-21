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
    OCR_COLUMN_GAP_MIN_FRACTION,
    OCR_COLUMN_GUTTER_MIN_FRACTION,
    OCR_FULL_SPAN_WIDTH_RATIO,
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
) -> bool:
    """
    True when at least one text row has a clear gutter between consecutive boxes.

    Only adjacent left-to-right gaps count (not distant fragments on the same band).
    Single-column prose can place many boxes on one row with small word spacing; true
    columns have a wide empty gap between consecutive regions on the same row.
    """
    column_boxes = [
        b for b in boxes if not is_full_span_box(b, page_width, full_span_width_ratio)
    ]
    if len(column_boxes) < 2:
        return False

    y_thresh = compute_y_threshold(page_height, column_boxes, page_width=page_width)
    gutter_min = max(
        gutter_min_fraction * page_width,
        0.2 * _median_box_width(column_boxes),
    )

    for row in _group_boxes_into_rows(column_boxes, y_thresh):
        if len(row) < 2:
            continue
        row_sorted = sorted(row, key=lambda b: float(b.left))
        for j in range(len(row_sorted) - 1):
            right = float(row_sorted[j].left) + float(row_sorted[j].width)
            next_left = float(row_sorted[j + 1].left)
            gap = next_left - right
            if next_left >= right and gap + 1e-6 >= gutter_min:
                return True
    return False


def should_use_column_reading_order(
    boxes: Sequence[Any],
    page_width: float,
    page_height: float,
    reading_order_mode: str | None = None,
    full_span_width_ratio: float = OCR_FULL_SPAN_WIDTH_RATIO,
    gutter_min_fraction: float = OCR_COLUMN_GUTTER_MIN_FRACTION,
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
    cluster_centers = [centers[0][0]]

    for x_center, box in centers[1:]:
        if x_center - cluster_centers[-1] > column_gap:
            clusters.append([box])
            cluster_centers.append(x_center)
        else:
            clusters[-1].append(box)
            cluster_centers[-1] = statistics.mean(
                float(b.left) + float(b.width) / 2.0 for b in clusters[-1]
            )

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


def group_into_lines_legacy(
    ocr_results: Sequence[Any],
    y_threshold: float = 12.0,
) -> List[List[Any]]:
    """Original top-then-left grouping (single-column assumption)."""
    lines: List[List[Any]] = []
    current_line: List[Any] = []

    for result in sorted(ocr_results, key=lambda x: (float(x.top), float(x.left))):
        if (
            not current_line
            or abs(float(result.top) - float(current_line[0].top)) <= y_threshold
        ):
            current_line.append(result)
        else:
            lines.append(sorted(current_line, key=lambda x: float(x.left)))
            current_line = [result]
    if current_line:
        lines.append(sorted(current_line, key=lambda x: float(x.left)))
    return lines


def group_into_lines(
    ordered_boxes: Sequence[Any],
    page_width: float,
    page_height: float,
    y_threshold: float | None = None,
) -> List[List[Any]]:
    """
    Group ordered boxes into lines without merging across columns or full-span zones.
    """
    if not ordered_boxes:
        return []

    if y_threshold is None:
        y_threshold = compute_y_threshold(
            page_height, ordered_boxes, page_width=page_width
        )

    page_width, page_height = page_width or 1.0, page_height or 1.0
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
            current_line.append(box)
        else:
            lines.append(sorted(current_line, key=lambda x: float(x.left)))
            current_line = [box]
            current_lb = lb

    if current_line:
        lines.append(sorted(current_line, key=lambda x: float(x.left)))
    return lines


def group_boxes_preserving_lines(ordered_boxes: Sequence[Any]) -> List[List[Any]]:
    """One input box per output line (Paddle line-level fast path)."""
    return [[box] for box in ordered_boxes]


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
            group_into_lines_legacy(ocr_results, y_threshold=yt),
            page_width,
            page_height,
        )

    ordered = sort_reading_order(ocr_results, page_width, page_height)

    if preserve_line_boxes:
        return group_boxes_preserving_lines(ordered), page_width, page_height

    yt = y_threshold
    if yt is None:
        yt = compute_y_threshold(page_height, ocr_results, page_width=page_width)
    return (
        group_into_lines(ordered, page_width, page_height, y_threshold=yt),
        page_width,
        page_height,
    )


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

    Keeps line_results, lines_char_groups, and page_data["results"] aligned while
    reassigning line numbers from start_line_number.
    """
    if not line_results:
        return list(line_results), list(lines_char_groups), page_data

    if len(line_results) != len(lines_char_groups):
        raise ValueError(
            "line_results and lines_char_groups must have the same length "
            f"({len(line_results)} != {len(lines_char_groups)})"
        )

    id_to_char_group = {
        id(line): lines_char_groups[i] for i, line in enumerate(line_results)
    }
    results_by_old_line = dict(page_data.get("results") or {})

    ordered = _order_line_boxes_for_reading(
        line_results,
        page_width,
        page_height,
        reading_order_mode=reading_order_mode,
    )

    new_line_results: List[Any] = []
    new_char_groups: List[Any] = []
    new_results: dict = {}

    for i, line in enumerate(ordered):
        old_line_no = line.line
        new_line_no = start_line_number + i
        line.line = new_line_no
        new_line_results.append(line)
        new_char_groups.append(id_to_char_group[id(line)])

        old_entry = results_by_old_line.get(f"text_line_{old_line_no}")
        if old_entry is not None:
            new_entry = dict(old_entry)
            new_entry["line"] = new_line_no
            new_results[f"text_line_{new_line_no}"] = new_entry
        else:
            new_results[f"text_line_{new_line_no}"] = {
                "line": new_line_no,
                "text": line.text,
                "bounding_box": [
                    line.left,
                    line.top,
                    line.left + line.width,
                    line.top + line.height,
                ],
                "words": [],
                "conf": getattr(line, "conf", 100.0),
            }

    new_page_data = dict(page_data)
    new_page_data["results"] = new_results
    return new_line_results, new_char_groups, new_page_data
