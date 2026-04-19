"""Tests for Review-tab redaction overlay export (drawing + PNG write)."""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import pandas as pd

from tools.file_redaction import (
    add_redaction_label_legend,
    draw_rectangle_outline_pattern,
)
from tools.redaction_review import (
    build_label_to_pattern_map,
    visualise_review_redaction_boxes,
)


def test_build_label_to_pattern_map_order_stable():
    df = pd.DataFrame({"label": ["Zebra", "Alpha"]})
    m = build_label_to_pattern_map(df, [])
    assert m["Alpha"] == "solid"
    assert m["Zebra"] == "dashed"


def test_build_label_to_pattern_map_fallback_labels():
    m = build_label_to_pattern_map(pd.DataFrame(), ["B", "A"])
    assert m["A"] == "solid"
    assert m["B"] == "dashed"


def test_draw_rectangle_outline_pattern_solid_changes_edge_pixels():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    draw_rectangle_outline_pattern(
        img, 10, 10, 30, 30, (0, 255, 0), thickness=2, pattern="solid"
    )
    assert int(img[10, 10, 1]) > 200


def test_add_redaction_label_legend_modifies_top_right():
    img = np.ones((100, 160, 3), dtype=np.uint8) * 200
    corner_before = img[8:25, 135:155].copy()
    add_redaction_label_legend(
        img,
        [((0, 0, 255), "solid", "TestLabel")],
        title="Legend",
    )
    assert not np.array_equal(img[8:25, 135:155], corner_before)


def test_visualise_review_redaction_boxes_writes_png(tmp_path):
    rgb = np.full((100, 100, 3), 240, dtype=np.uint8)
    page = {
        "image": rgb,
        "boxes": [
            {
                "label": "PERSON",
                "color": "(255, 0, 0)",
                "xmin": 0.1,
                "ymin": 0.1,
                "xmax": 0.4,
                "ymax": 0.3,
            },
            {
                "label": "EMAIL",
                "color": "(0, 128, 0)",
                "xmin": 0.5,
                "ymin": 0.5,
                "xmax": 0.95,
                "ymax": 0.85,
            },
        ],
    }
    review_df = pd.DataFrame({"label": ["EMAIL", "PERSON"]})
    out = visualise_review_redaction_boxes(
        page,
        review_df=review_df,
        output_folder=str(tmp_path),
        page_number=2,
        doc_base_name="mydoc.pdf",
    )
    assert out is not None
    assert os.path.isfile(out)
    assert out.endswith("_page2_redaction_overlay.png")


def test_visualise_review_redaction_boxes_returns_none_without_boxes(tmp_path):
    rgb = np.full((20, 20, 3), 200, dtype=np.uint8)
    page = {"image": rgb, "boxes": []}
    assert visualise_review_redaction_boxes(page, output_folder=str(tmp_path)) is None
