"""Tests for multi-column local OCR reading order."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from tools.ocr_reading_order import (
    assign_layout_boxes,
    build_line_groups,
    group_into_lines_legacy,
    has_side_by_side_columns,
    reorder_structured_text_lines,
    should_use_column_reading_order,
    sort_reading_order,
)

COMPLAINT_CSV = (
    Path(__file__).resolve().parent.parent
    / "doc_redaction"
    / "example_data"
    / "example_outputs"
    / "example_complaint_letter_ocr_output_local_ocr.csv"
)

PARTNERSHIP_CSV = (
    Path(__file__).resolve().parent.parent
    / "doc_redaction"
    / "example_data"
    / "example_outputs"
    / "Partnership-Agreement-Toolkit_0_0_ocr_output_local_ocr.csv"
)

PARTNERSHIP_V2_CSV = (
    Path(__file__).resolve().parent.parent
    / "doc_redaction"
    / "example_data"
    / "example_outputs"
    / "Partnership-Agreement-Toolkit_0_0_ocr_output_local_ocr_v2.csv"
)

FOREWORD_WORDS_CSV = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "foreword_ocr"
    / "Lambeth 2030 FINAL ACC_Ver_Dec.pdf_foreword_ocr_results_with_words_local_ocr.csv"
)


@dataclass
class _OCRBox:
    text: str
    left: float
    top: float
    width: float
    height: float
    conf: float = 99.0
    line: int | None = None
    model: str | None = "Paddle"


def _ocr(text, left, top, width=0.15, height=0.01):
    return _OCRBox(
        text=text,
        left=left,
        top=top,
        width=width,
        height=height,
    )


def _three_column_page_boxes():
    """Synthetic 3-column layout (normalized 0-1 coords), mimicking foreword geometry."""
    header = _ocr("04 Lambeth 2030 banner", 0.02, 0.03, width=0.95, height=0.01)
    left_col = [
        _ocr("left A", 0.05, 0.30),
        _ocr("left B", 0.05, 0.32),
        _ocr("left C", 0.05, 0.34),
    ]
    mid_col = [
        _ocr("mid A", 0.26, 0.30),
        _ocr("mid B", 0.26, 0.32),
        _ocr("mid C", 0.26, 0.34),
    ]
    right_col = [
        _ocr("right A", 0.55, 0.30),
        _ocr("right B", 0.55, 0.32),
        _ocr("right C", 0.55, 0.34),
    ]
    # Interleaved tops (legacy sort would mix columns)
    body = [
        left_col[0],
        mid_col[0],
        left_col[1],
        right_col[0],
        mid_col[1],
        left_col[2],
        mid_col[2],
        right_col[1],
        right_col[2],
    ]
    return [header] + body


def test_detect_three_columns():
    boxes = _three_column_page_boxes()[1:]  # exclude full-span header
    layout = assign_layout_boxes(
        boxes, page_width=1.0, page_height=1.0, reading_order_mode="column"
    )
    column_indices = {lb.column_index for lb in layout if lb.zone == "column"}
    assert column_indices == {0, 1, 2}


def test_sort_reading_order_column_major():
    boxes = _three_column_page_boxes()
    ordered = sort_reading_order(
        boxes, page_width=1.0, page_height=1.0, reading_order_mode="column"
    )
    texts = [b.text for b in ordered]
    assert texts[0] == "04 Lambeth 2030 banner"
    assert texts[1:4] == ["left A", "left B", "left C"]
    assert texts[4:7] == ["mid A", "mid B", "mid C"]
    assert texts[7:10] == ["right A", "right B", "right C"]


def test_group_into_lines_no_cross_column_merge():
    boxes = _three_column_page_boxes()
    words = []
    for line_box in boxes[1:]:
        for word in line_box.text.split():
            words.append(
                _ocr(
                    word,
                    line_box.left,
                    line_box.top,
                    width=line_box.width / 2,
                    height=line_box.height,
                )
            )

    line_groups, _, _ = build_line_groups(
        words,
        reading_order_mode="column",
        preserve_line_boxes=False,
    )
    assert len(line_groups) == 9
    for group in line_groups:
        line_width = max(w.left + w.width for w in group) - min(w.left for w in group)
        if line_width > 0.5:
            continue  # full-span header line
        assert line_width < 0.35


def test_legacy_order_interleaves_columns():
    # Slightly staggered tops so legacy (top,left) sort interleaves columns.
    boxes = [
        _ocr("left A", 0.05, 0.30),
        _ocr("mid A", 0.26, 0.31),
        _ocr("left B", 0.05, 0.34),
        _ocr("mid B", 0.26, 0.35),
    ]
    legacy_lines = group_into_lines_legacy(boxes, y_threshold=0.005)
    first_texts = [line[0].text for line in legacy_lines[:3]]
    assert first_texts == ["left A", "mid A", "left B"]


def test_preserve_line_boxes_one_line_per_box():
    boxes = _three_column_page_boxes()[1:]
    lines, _, _ = build_line_groups(
        boxes,
        reading_order_mode="column",
        preserve_line_boxes=True,
    )
    assert len(lines) == len(boxes)
    assert all(len(group) == 1 for group in lines)


def test_column_and_legacy_single_column_same_order():
    single = [
        _ocr("one", 0.1, 0.1),
        _ocr("two", 0.1, 0.2),
        _ocr("three", 0.1, 0.3),
    ]
    col_groups, _, _ = build_line_groups(single, reading_order_mode="column")
    leg_groups, _, _ = build_line_groups(single, reading_order_mode="legacy")
    col_texts = [g[0].text for g in col_groups]
    leg_texts = [g[0].text for g in leg_groups]
    assert col_texts == leg_texts == ["one", "two", "three"]


def test_foreword_interleave_regression():
    """In a 2-column layout the left column precedes the right, even when the right box
    sits slightly higher on the page (tests column-major trumps raw top order).

    Three gutter rows are included so the minimum-gutter-rows check is satisfied.
    Forewords is full-span (width=0.60 >= OCR_FULL_SPAN_WIDTH_RATIO=0.6).
    """
    boxes = [
        _ocr("Forewords", 0.05, 0.18, width=0.60, height=0.05),
        # Row 1 – mid line sits slightly above left line on the page
        _ocr("left line", 0.05, 0.32, width=0.17, height=0.01),
        _ocr("mid line", 0.30, 0.317, width=0.18, height=0.01),
        # Rows 2 & 3 – needed to reach min_gutter_rows=3
        _ocr("left next", 0.05, 0.34, width=0.17, height=0.01),
        _ocr("mid next", 0.30, 0.34, width=0.18, height=0.01),
        _ocr("left third", 0.05, 0.36, width=0.17, height=0.01),
        _ocr("mid third", 0.30, 0.36, width=0.18, height=0.01),
    ]
    ordered = sort_reading_order(
        boxes, page_width=1.0, page_height=1.0, reading_order_mode="column"
    )
    texts = [b.text for b in ordered]
    assert texts.index("left line") < texts.index("mid line")
    # Column-major: both left-column lines precede the right column.
    assert texts.index("left next") < texts.index("mid line")


def test_build_line_groups_secondary_sort_column_major():
    """build_line_groups must output all left-sub-column lines before right-sub-column
    lines even when word-level gap between sub-columns is below the primary
    assign_layout_boxes column-gap threshold.

    Simulates a 2-sub-column layout where the sub-column gutter (0.025) is narrower
    than the word-based column_gap_threshold (≈0.04).  Each y-band is populated with
    one left word and one right word at the same top value so _finalize_line splits
    them; the secondary _reorder_lines_column_major pass must then group all left
    sub-column lines first.
    """
    # Build 4 y-rows with a left word (x≈0.05) and a right word (x≈0.28), separated
    # by a gap of 0.23 which is large enough for _finalize_line (threshold=0.025) to
    # split.  The macro column detection sees all words with centres 0.05-0.29 as one
    # cluster (gap 0.025 < threshold ≈0.04), so the primary sort alone would interleave.
    words = []
    for row in range(4):
        top = 0.30 + row * 0.02
        words.append(_ocr(f"L{row}", 0.05, top, width=0.14, height=0.012))
        # Right word starts at 0.28 — gap of 0.28-0.19=0.09 from right edge of left word,
        # far above OCR_LINE_SPLIT_GAP_FRACTION=0.025.
        words.append(_ocr(f"R{row}", 0.28, top, width=0.14, height=0.012))
    # Add enough gutter rows so has_side_by_side_columns returns True
    for row in range(4):
        top = 0.30 + row * 0.02
        words.append(_ocr(f"G{row}", 0.05, top, width=0.05, height=0.012))

    lines, _, _ = build_line_groups(words, reading_order_mode="column")
    line_texts = [" ".join(b.text for b in ln) for ln in lines]

    # All L* lines must precede all R* lines (column-major)
    l_indices = [
        i for i, t in enumerate(line_texts) if any(w.startswith("L") for w in t.split())
    ]
    r_indices = [
        i for i, t in enumerate(line_texts) if any(w.startswith("R") for w in t.split())
    ]
    if l_indices and r_indices:
        assert max(l_indices) < min(r_indices), (
            f"Left sub-column lines interleaved with right sub-column lines.\n"
            f"Line texts: {line_texts}"
        )


@pytest.mark.skipif(
    not FOREWORD_WORDS_CSV.exists(), reason="foreword word-level CSV not present"
)
def test_foreword_word_level_no_micro_column_fragmentation():
    """Word-level boxes from the Lambeth foreword spread must not be fragmented into
    many micro-columns.  The max-based cluster comparison in assign_layout_boxes must
    detect at most 4 columns (left-half + right-half of the spread is the minimum
    acceptable detection).  The old mean-based comparison caused 13+ spurious clusters.

    Also verifies that known clean body-text lines ("From William Blake to Olive Morris")
    and heading lines ("Lambeth has long been") are produced as coherent single lines,
    not as fragments of individual words.
    """
    from collections import namedtuple

    df = pd.read_csv(FOREWORD_WORDS_CSV)
    page = df[df["page"] == 1] if "page" in df.columns else df
    OCRResult = namedtuple(
        "OCRResult", ["left", "top", "width", "height", "text", "conf"]
    )
    boxes = [
        OCRResult(
            r.word_x0,
            r.word_y0,
            r.word_x1 - r.word_x0,
            r.word_y1 - r.word_y0,
            r.word_text,
            r.get("word_conf", 0),
        )
        for _, r in page.iterrows()
    ]

    layout = assign_layout_boxes(
        boxes, page_width=1.0, page_height=1.0, reading_order_mode="column"
    )
    num_columns = (
        max((lb.column_index for lb in layout if lb.zone == "column"), default=0) + 1
    )
    assert num_columns <= 4, (
        f"Too many micro-columns detected ({num_columns}); expected <= 4 "
        "(max-based clustering regression)"
    )

    lines, _, _ = build_line_groups(boxes)
    line_texts = [" ".join(b.text for b in line) for line in lines]

    # "Lambeth has long been" is row 1 of the bold left-column heading; it must appear
    # as a single coherent line, not split into individual word lines.
    heading_lines = [
        t for t in line_texts if "Lambeth" in t and "long" in t and "been" in t
    ]
    assert heading_lines, (
        "Expected a line containing 'Lambeth has long been' but none found.\n"
        f"Lines: {line_texts[:20]}"
    )
    # "From William Blake to Olive Morris" is the first body-text line in column A.
    body_lines = [
        t for t in line_texts if "William" in t and "Blake" in t and "Morris" in t
    ]
    assert body_lines, (
        "Expected a clean body-text line with 'William Blake ... Morris' but none found.\n"
        f"Lines: {line_texts[:30]}"
    )
    # Ensure "William" and "Blake" are not on separate tiny lines (fragmentation guard).
    william_lines = [t for t in line_texts if "William" in t]
    assert any("Blake" in t for t in william_lines), (
        f"'William' and 'Blake' ended up on different lines — word fragmentation detected.\n"
        f"Lines with 'William': {william_lines}"
    )


PAGE_W = 595.0
PAGE_H = 842.0


def _pymupdf_line(text, left, top, width=80.0, height=12.0, line_no=1):
    return _OCRBox(
        text=text,
        left=left,
        top=top,
        width=width,
        height=height,
        line=line_no,
        model="PyMuPDF",
    )


def _structured_page_from_lines(lines):
    """Build page_data / parallel lists as process_page_to_structured_ocr_pymupdf would."""
    line_results = []
    char_groups = []
    results = {}
    for line in lines:
        line_no = line.line
        line_results.append(line)
        char_groups.append([{"text": line.text, "bbox": [line.left, line.top, 10, 10]}])
        results[f"text_line_{line_no}"] = {
            "line": line_no,
            "text": line.text,
            "bounding_box": [
                line.left,
                line.top,
                line.left + line.width,
                line.top + line.height,
            ],
            "words": [
                {"text": line.text, "bounding_box": [line.left, line.top, 10, 10]}
            ],
            "conf": 100.0,
        }
    page_data = {"page": "1", "results": results}
    return line_results, char_groups, page_data


def _three_column_pymupdf_lines_interleaved():
    """PDF-point coords; block order interleaves columns."""
    lines = [
        _pymupdf_line("Banner", 20, 30, width=PAGE_W * 0.92, line_no=1),
        _pymupdf_line("left A", 50, 300, line_no=2),
        _pymupdf_line("mid A", 180, 301, line_no=3),
        _pymupdf_line("left B", 50, 320, line_no=4),
        _pymupdf_line("right A", 340, 300, line_no=5),
        _pymupdf_line("mid B", 180, 321, line_no=6),
        _pymupdf_line("left C", 50, 340, line_no=7),
        _pymupdf_line("mid C", 180, 341, line_no=8),
        _pymupdf_line("right B", 340, 320, line_no=9),
        _pymupdf_line("right C", 340, 340, line_no=10),
    ]
    return lines


def test_reorder_structured_text_lines_three_columns():
    lines = _three_column_pymupdf_lines_interleaved()
    lr, cg, pd = _structured_page_from_lines(lines)
    new_lr, new_cg, new_pd = reorder_structured_text_lines(
        lr,
        cg,
        pd,
        page_width=PAGE_W,
        page_height=PAGE_H,
        reading_order_mode="column",
    )
    texts = [ln.text for ln in new_lr]
    assert texts[0] == "Banner"
    assert texts[1:4] == ["left A", "left B", "left C"]
    assert texts[4:7] == ["mid A", "mid B", "mid C"]
    assert texts[7:10] == ["right A", "right B", "right C"]
    assert [ln.line for ln in new_lr] == list(range(1, 11))
    assert new_pd["results"]["text_line_2"]["text"] == "left A"
    assert len(new_cg) == len(new_lr)


def test_reorder_structured_text_lines_header_first():
    lines = [
        _pymupdf_line("Header", 10, 20, width=PAGE_W * 0.9, line_no=1),
        _pymupdf_line("body left", 50, 200, line_no=2),
        _pymupdf_line("body mid", 180, 201, line_no=3),
    ]
    lr, cg, pd = _structured_page_from_lines(lines)
    new_lr, _, _ = reorder_structured_text_lines(
        lr, cg, pd, page_width=PAGE_W, page_height=PAGE_H, reading_order_mode="column"
    )
    assert new_lr[0].text == "Header"
    assert new_lr[0].line == 1
    assert [ln.text for ln in new_lr[1:]] == ["body left", "body mid"]


def test_reorder_structured_text_lines_legacy():
    lines = [
        _pymupdf_line("left A", 50, 300, line_no=1),
        _pymupdf_line("mid A", 180, 301, line_no=2),
        _pymupdf_line("left B", 50, 320, line_no=3),
    ]
    lr, cg, pd = _structured_page_from_lines(lines)
    new_lr, _, _ = reorder_structured_text_lines(
        lr, cg, pd, page_width=PAGE_W, page_height=PAGE_H, reading_order_mode="legacy"
    )
    assert [ln.text for ln in new_lr] == ["left A", "mid A", "left B"]


def test_reorder_structured_text_lines_words_aligned():
    lines = [
        _pymupdf_line("mid first", 180, 300, line_no=1),
        _pymupdf_line("left second", 50, 300, line_no=2),
    ]
    lr, cg, pd = _structured_page_from_lines(lines)
    new_lr, _, new_pd = reorder_structured_text_lines(
        lr, cg, pd, page_width=PAGE_W, page_height=PAGE_H, reading_order_mode="column"
    )
    assert new_lr[0].text == "left second"
    assert new_pd["results"]["text_line_1"]["words"][0]["text"] == "left second"


def _boxes_from_csv(path: Path):
    df = pd.read_csv(path)
    boxes = []
    for _, r in df.iterrows():
        boxes.append(_ocr(r.text, r.left, r.top, r.width, r.height))
    return boxes


def test_complaint_letter_not_multi_column():
    """Single-column business letter must not use false column clustering."""
    boxes = _boxes_from_csv(COMPLAINT_CSV)
    assert should_use_column_reading_order(boxes, 1.0, 1.0) is False
    assert has_side_by_side_columns(boxes, 1.0, 1.0) is False
    layout = assign_layout_boxes(boxes, 1.0, 1.0)
    column_indices = {lb.column_index for lb in layout if lb.zone == "column"}
    assert column_indices == {0}


def test_complaint_letter_reading_order_puts_street_on_first_row():
    boxes = _boxes_from_csv(COMPLAINT_CSV)
    ordered = sort_reading_order(boxes, page_width=1.0, page_height=1.0)
    top_row = [b.text for b in ordered if abs(b.top - 0.109501) < 0.002]
    # Address line may be a merged token or separate words depending on OCR output
    address_tokens = {"123 Main Street", "123 Main", "Street"}
    assert any(
        t in top_row for t in address_tokens
    ), f"address not in top_row: {top_row}"
    address_box = next(
        (b for b in ordered if any(t in b.text for t in address_tokens)), None
    )
    assert address_box is not None
    assert ordered.index(address_box) < 20


def test_build_line_groups_complaint_merges_address_line():
    boxes = _boxes_from_csv(COMPLAINT_CSV)
    groups, _, _ = build_line_groups(boxes, reading_order_mode="column")
    first = groups[0]
    all_text = " ".join(w.text for w in first)
    # Address may be a single merged token or separate words
    assert "123 Main" in all_text and "Street" in all_text


# ---------------------------------------------------------------------------
# Partnership Agreement Toolkit page 1 – header-only side-by-side should not
# trigger multi-column mode for the single-column body that follows.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PARTNERSHIP_CSV.exists(), reason="partnership fixture not present"
)
def test_partnership_p1_header_does_not_trigger_column_mode():
    """Page 1 has logo+title side-by-side in the header but single-column body.

    Only 1 text-row group shows a gutter (the header band), so the page must
    not be classified as multi-column (requires >= OCR_COLUMN_MIN_GUTTER_ROWS=3).
    """
    df = pd.read_csv(PARTNERSHIP_CSV)
    df_p1 = df[df["page"] == 1]
    boxes = [
        _ocr(
            str(r["text"]),
            float(r["left"]),
            float(r["top"]),
            float(r["width"]),
            float(r["height"]),
        )
        for _, r in df_p1.iterrows()
    ]
    assert has_side_by_side_columns(boxes, 1.0, 1.0) is False
    assert should_use_column_reading_order(boxes, 1.0, 1.0) is False


@pytest.mark.skipif(
    not PARTNERSHIP_CSV.exists(), reason="partnership fixture not present"
)
def test_partnership_p1_body_all_in_single_column():
    """Column assignments for page 1 must all land in column 0 (no spurious split)."""
    df = pd.read_csv(PARTNERSHIP_CSV)
    df_p1 = df[df["page"] == 1]
    boxes = [
        _ocr(
            str(r["text"]),
            float(r["left"]),
            float(r["top"]),
            float(r["width"]),
            float(r["height"]),
        )
        for _, r in df_p1.iterrows()
    ]
    layout = assign_layout_boxes(boxes, 1.0, 1.0)
    column_indices = {lb.column_index for lb in layout if lb.zone == "column"}
    assert column_indices == {
        0
    }, f"Expected all column boxes in column 0, got indices {column_indices}"


# ---------------------------------------------------------------------------
# Partnership page 6 (v2 OCR) – city seal image misdetected as a tall text
# box must NOT count as a gutter row and must not trigger column mode.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PARTNERSHIP_V2_CSV.exists(), reason="partnership v2 fixture not present"
)
def test_partnership_p6_tall_image_box_excluded_from_gutter_detection():
    """Page 6 has a city-seal image OCR'd as a tall '?' box (height ~20× median).

    Without height filtering this box creates a spurious third gutter row that
    triggers column mode.  The height filter (OCR_COLUMN_MAX_BOX_HEIGHT_RATIO=4.0)
    must exclude it so the page stays in single-column mode.
    """
    df = pd.read_csv(PARTNERSHIP_V2_CSV)
    df_p6 = df[df["page"] == 6]
    boxes = [
        _ocr(
            str(r["text"]),
            float(r["left"]),
            float(r["top"]),
            float(r["width"]),
            float(r["height"]),
        )
        for _, r in df_p6.iterrows()
    ]
    assert has_side_by_side_columns(boxes, 1.0, 1.0) is False
    assert should_use_column_reading_order(boxes, 1.0, 1.0) is False


# ---------------------------------------------------------------------------
# Partnership page 4 (v2 OCR) – header + two signature rows must not trigger
# column mode on the single-column body text above the signature block.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PARTNERSHIP_V2_CSV.exists(), reason="partnership v2 fixture not present"
)
def test_partnership_p4_signature_block_does_not_trigger_column_mode():
    """Page 4 has a header gutter (logo | Toolkit, y≈0.07) and two side-by-side
    signature rows at the bottom (y≈0.80–0.82).  These three gutter rows would
    previously meet the old min_gutter_rows=3 threshold and incorrectly force
    column-major reading order, splitting body paragraphs into three apparent
    columns.

    The consecutive-cluster check must reject them because:
    * the header row and the signature rows are separated by a y-gap of ~73% of
      page height (>> OCR_COLUMN_MAX_CONSECUTIVE_GUTTER_GAP=0.06), so they form
      two distinct clusters of sizes 1 and 2, neither ≥ min_gutter_rows=3.
    * even if the signature cluster were large enough, its topmost row (y≈0.80)
      lies in the footer zone (≥ OCR_COLUMN_FOOTER_ZONE_FRACTION=0.75), so it
      must not trigger column mode on its own.
    """
    df = pd.read_csv(PARTNERSHIP_V2_CSV)
    df_p4 = df[df["page"] == 4]
    boxes = [
        _ocr(
            str(r["text"]),
            float(r["left"]),
            float(r["top"]),
            float(r["width"]),
            float(r["height"]),
        )
        for _, r in df_p4.iterrows()
    ]
    assert has_side_by_side_columns(boxes, 1.0, 1.0) is False
    assert should_use_column_reading_order(boxes, 1.0, 1.0) is False


# ---------------------------------------------------------------------------
# Line-split gap: side-by-side word boxes must not be merged into one line.
# ---------------------------------------------------------------------------


def test_group_into_lines_legacy_splits_side_by_side_names():
    """Two signature names on opposite sides of the page (e.g. Rudolph W. Giuliani
    on the left and Ken Livingstone on the right) must produce separate lines, not
    one merged line.

    A line break is triggered by either of two mechanisms:
    * Build-time rightward gap: the next box starts > OCR_LINE_SPLIT_GAP_FRACTION
      (10%) to the right of the current line's rightmost edge.
    * Post-processing split: after the y-band is closed, _finalize_line sorts
      the group by left position and splits it wherever consecutive boxes have an
      internal gap > the same threshold.  This handles the case where two elements
      on opposite sides of the page share a nearly identical y-coordinate, causing
      the right-side element to be sorted *before* the left-side element (smaller
      top value), after which the left-side element arrives as an apparent leftward
      step that the build-time check alone misses.
    """
    from tools.ocr_reading_order import group_into_lines_legacy

    # Row spacing is 0.022 between each logical row — comfortably above y_threshold
    # (0.02) so each row occupies its own y-band and the gap logic is the only
    # mechanism needed to split the left/right signature elements within a row.
    boxes = [
        _ocr("Rudolph", 0.256, 0.870, 0.072, 0.018),
        _ocr("W.", 0.330, 0.870, 0.024, 0.018),
        _ocr("Giuliani", 0.356, 0.870, 0.080, 0.018),
        _ocr("Ken", 0.706, 0.870, 0.040, 0.018),
        _ocr("Livingstone", 0.748, 0.870, 0.060, 0.018),
        _ocr("Mayor", 0.309, 0.895, 0.060, 0.016),
        _ocr("Mayor", 0.764, 0.895, 0.060, 0.016),
        _ocr("New", 0.284, 0.920, 0.032, 0.016),
        _ocr("York", 0.318, 0.920, 0.040, 0.016),
        _ocr("City", 0.360, 0.920, 0.040, 0.016),
        _ocr("London", 0.698, 0.920, 0.070, 0.016),
    ]

    lines = group_into_lines_legacy(boxes, y_threshold=0.02, page_width=1.0)
    line_texts = [" ".join(b.text for b in line) for line in lines]

    assert (
        "Rudolph W. Giuliani" in line_texts
    ), f"Expected separate Giuliani line, got: {line_texts}"
    assert (
        "Ken Livingstone" in line_texts
    ), f"Expected separate Livingstone line, got: {line_texts}"
    for text in line_texts:
        assert not (
            "Giuliani" in text and "Livingstone" in text
        ), f"Names were merged into one line: {text!r}"
    mayor_lines = [t for t in line_texts if "Mayor" in t]
    for m in mayor_lines:
        assert m.count("Mayor") == 1, f"Two 'Mayor' tokens merged into one line: {m!r}"
