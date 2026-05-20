"""Tests for multi-column local OCR reading order."""

from dataclasses import dataclass

from tools.ocr_reading_order import (
    assign_layout_boxes,
    build_line_groups,
    group_into_lines_legacy,
    reorder_structured_text_lines,
    sort_reading_order,
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
    layout = assign_layout_boxes(boxes, page_width=1.0, page_height=1.0)
    column_indices = {lb.column_index for lb in layout if lb.zone == "column"}
    assert column_indices == {0, 1, 2}


def test_sort_reading_order_column_major():
    boxes = _three_column_page_boxes()
    ordered = sort_reading_order(boxes, page_width=1.0, page_height=1.0)
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
    """Anonymized pattern from foreword CSV: mid column must not precede left at same band."""
    boxes = [
        _ocr("Forewords", 0.05, 0.18, width=0.24, height=0.05),
        _ocr("left line", 0.05, 0.32, width=0.17, height=0.01),
        _ocr("mid line", 0.26, 0.317, width=0.18, height=0.01),
        _ocr("left next", 0.05, 0.324, width=0.17, height=0.01),
    ]
    ordered = sort_reading_order(boxes, page_width=1.0, page_height=1.0)
    texts = [b.text for b in ordered]
    assert texts.index("left line") < texts.index("mid line")
    # Column-major: both left-column lines precede the middle column.
    assert texts.index("left next") < texts.index("mid line")


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
