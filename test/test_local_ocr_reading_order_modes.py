from __future__ import annotations

from dataclasses import dataclass

from tools.ocr_reading_order import should_use_column_reading_order


@dataclass
class _Box:
    left: float
    top: float
    width: float
    height: float


def test_should_use_column_reading_order_accepts_paddle_native() -> None:
    # 3 gutter rows (default min) with a clear gap between left and right boxes.
    boxes: list[_Box] = []
    # Rows must be close enough to count as a single consecutive gutter cluster.
    for row, top in enumerate((0.10, 0.14, 0.18), start=1):
        boxes.append(_Box(left=0.05, top=top, width=0.20, height=0.02))
        boxes.append(_Box(left=0.60, top=top, width=0.20, height=0.02))

    assert should_use_column_reading_order(
        boxes,
        page_width=1.0,
        page_height=1.0,
        reading_order_mode="paddle_native",
    )
