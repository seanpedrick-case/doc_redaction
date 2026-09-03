from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")

import pytest

pytest.importorskip("gradio_image_annotation_redaction")

import numpy as np

from tools.redaction_review import export_review_page_ocr_visualisation_for_gradio


def test_export_review_page_ocr_visualisation_writes_file(tmp_path):
    page = {
        "image": np.full((120, 160, 3), 255, dtype=np.uint8),
        "boxes": [],
    }
    ocr_with_words = [
        {
            "page": 1,
            "results": {
                "line_1": {
                    "line": 1,
                    "text": "Hello world",
                    "words": [
                        {
                            "text": "Hello",
                            "bounding_box": (10, 10, 60, 30),
                            "conf": 95,
                            "model": "Textract",
                        },
                        {
                            "text": "world",
                            "bounding_box": (70, 10, 120, 30),
                            "conf": 85,
                            "model": "Textract",
                        },
                    ],
                }
            },
        }
    ]

    out = export_review_page_ocr_visualisation_for_gradio(
        page,
        1,
        ocr_with_words,
        None,
        "doc.pdf",
        str(tmp_path),
    )
    assert out is not None
    resolved_out = os.path.realpath(out)
    safe_root = os.path.realpath(str(tmp_path))
    assert os.path.commonpath([safe_root, resolved_out]) == safe_root
    assert os.path.isfile(resolved_out)
    assert "review_ocr_visualisations" in out.replace("\\", "/")


def test_export_review_page_ocr_visualisation_prefers_state_image_path(
    tmp_path, monkeypatch
):
    """Stale Gradio tmp image paths should fall back to session state."""
    from PIL import Image

    import tools.config as config
    import tools.redaction_review as rr
    from tools.helper_functions import page_ocr_review_image_for_gradio

    output_root = str(tmp_path.resolve())
    monkeypatch.setattr(config, "OUTPUT_FOLDER", output_root + os.sep)
    monkeypatch.setattr(rr, "OUTPUT_FOLDER", output_root + os.sep)

    stable_png = tmp_path / "page1.png"
    Image.new("RGB", (160, 120), color=(255, 255, 255)).save(stable_png)

    client = {"image": "/tmp/gradio_tmp/stale.png", "boxes": []}
    state = [
        {
            "image": str(stable_png),
            "boxes": [],
        }
    ]
    ocr_with_words = [
        {
            "page": 1,
            "results": {
                "line_1": {
                    "line": 1,
                    "text": "Hello",
                    "words": [
                        {
                            "text": "Hello",
                            "bounding_box": (10, 10, 60, 30),
                            "conf": 95,
                        }
                    ],
                }
            },
        }
    ]

    out = page_ocr_review_image_for_gradio(
        client,
        1,
        ocr_with_words,
        None,
        "doc.pdf",
        output_root,
        "",
        state,
        [{"page": 1, "image_path": str(stable_png)}],
    )
    assert out is not None
    assert os.path.isfile(out)


def test_export_review_page_ocr_visualisation_draws_text_for_normalized_boxes(tmp_path):
    # Regression: some OCR pipelines provide bbox coords normalized to [0,1].
    # The visualisation should scale these into pixel space and render text.
    page = {
        "image": np.full((120, 160, 3), 255, dtype=np.uint8),
        "boxes": [],
    }
    ocr_with_words = [
        {
            "page": 1,
            "results": {
                "line_1": {
                    "line": 1,
                    "text": "Hello world",
                    "words": [
                        {
                            "text": "Hello",
                            "bounding_box": (0.10, 0.10, 0.40, 0.25),
                            "conf": 95,
                            "model": "Textract",
                        },
                        {
                            "text": "world",
                            "bounding_box": (0.45, 0.10, 0.80, 0.25),
                            "conf": 85,
                            "model": "Textract",
                        },
                    ],
                }
            },
        }
    ]

    out = export_review_page_ocr_visualisation_for_gradio(
        page,
        1,
        ocr_with_words,
        None,
        "doc.pdf",
        str(tmp_path),
    )
    assert out is not None
    resolved_out = os.path.realpath(out)
    safe_root = os.path.realpath(str(tmp_path))
    assert os.path.commonpath([safe_root, resolved_out]) == safe_root
    assert os.path.isfile(resolved_out)

    # Ensure there is non-white ink on the right-hand half (the text page).
    from PIL import Image

    img = Image.open(resolved_out).convert("RGB")
    w, h = img.size
    # Right half; skip a small top-left patch where only legend might appear.
    crop = img.crop((w // 2 + 5, 5, w - 5, h - 5))
    arr = np.asarray(crop)
    assert (arr < 250).any()
