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
    _box_coords_to_pixel_rect,
    build_label_to_pattern_map,
    visualise_review_redaction_boxes,
)


def test_box_coords_to_pixel_rect_gradio_pixels_vs_normalized():
    """Gradio uses pixel coords; review data uses 0–1."""
    assert _box_coords_to_pixel_rect(10, 20, 50, 60, 100, 100) == (10, 20, 50, 60)
    assert _box_coords_to_pixel_rect(0.1, 0.2, 0.5, 0.6, 100, 100) == (
        10,
        20,
        50,
        60,
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


def test_visualise_review_redaction_boxes_writes_jpeg_under_size_cap(tmp_path):
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
        label_abbrev_chars=0,
    )
    assert out is not None
    assert os.path.isfile(out)
    assert "redaction_overlay" in out.replace("\\", "/")
    assert out.endswith("_page2_redaction_overlay.jpg")
    assert os.path.getsize(out) <= 600 * 1024


def test_visualise_review_label_abbrev_drawn_when_requested(tmp_path):
    import cv2

    rgb = np.full((120, 120, 3), 250, dtype=np.uint8)
    page = {
        "image": rgb,
        "boxes": [
            {
                "label": "PERSON",
                "color": "(255, 0, 0)",
                "xmin": 20,
                "ymin": 40,
                "xmax": 90,
                "ymax": 90,
            }
        ],
    }
    out = visualise_review_redaction_boxes(
        page,
        output_folder=str(tmp_path),
        label_abbrev_chars=3,
    )
    assert out is not None
    bgr = cv2.imread(out)
    assert bgr is not None
    # Above-box region should differ from flat background once label is drawn
    assert bgr[25:38, 45:75].std() > 2.0


def test_visualise_review_gradio_style_pixel_boxes_draw_visible(tmp_path):
    """Absolute pixel boxes (Gradio) must produce a visible outline, not clip to edge."""
    import cv2

    rgb = np.full((100, 100, 3), 250, dtype=np.uint8)
    page = {
        "image": rgb,
        "boxes": [
            {
                "label": "R",
                "color": "(255, 0, 0)",
                "xmin": 10,
                "ymin": 10,
                "xmax": 45,
                "ymax": 40,
            }
        ],
    }
    out = visualise_review_redaction_boxes(
        page,
        output_folder=str(tmp_path),
        page_number=1,
        doc_base_name="t",
        label_abbrev_chars=0,
    )
    assert out is not None
    bgr = cv2.imread(out)
    assert bgr is not None
    # Outline at top-left corner should differ from flat 250 background
    assert not np.allclose(bgr[10, 10], [250, 250, 250], atol=3)


def test_visualise_review_redaction_boxes_returns_none_without_boxes(tmp_path):
    rgb = np.full((20, 20, 3), 200, dtype=np.uint8)
    page = {"image": rgb, "boxes": []}
    assert visualise_review_redaction_boxes(page, output_folder=str(tmp_path)) is None
