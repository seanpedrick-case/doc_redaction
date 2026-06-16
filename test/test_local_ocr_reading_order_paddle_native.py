from __future__ import annotations

import pytest

pytest.importorskip("regex")

from tools.custom_image_analyser_engine import OCRResult, combine_ocr_results


def test_paddle_native_preserves_input_line_boxes_when_multicolumn() -> None:
    # Build a minimal multi-column pattern: 3 gutter rows (min required by config),
    # each with a left and right box separated by a wide gap.
    # This should trigger column mode and, with paddle_native, force preserve_line_boxes=True.
    boxes: list[OCRResult] = []
    for row, top in enumerate((0, 100, 200), start=1):
        boxes.append(
            OCRResult(
                text=f"L{row}",
                left=0,
                top=top,
                width=400,
                height=30,
                conf=90,
                model="Paddle",
            )
        )
        boxes.append(
            OCRResult(
                text=f"R{row}",
                left=600,
                top=top,
                width=400,
                height=30,
                conf=90,
                model="Paddle",
            )
        )

    page_level, _with_words = combine_ocr_results(
        boxes,
        page=1,
        preserve_line_boxes=False,  # should be overridden by paddle_native
        reading_order_mode="paddle_native",
    )

    # In preserve_line_boxes mode, each input box becomes its own line group,
    # so the number of produced line-level OCR results should match the inputs.
    assert len(page_level["results"]) == len(boxes)
