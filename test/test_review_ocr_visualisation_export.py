from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")

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
    assert os.path.isfile(out)
    assert "review_ocr_visualisations" in out.replace("\\", "/")


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
    assert os.path.isfile(out)

    # Ensure there is non-white ink on the right-hand half (the text page).
    from PIL import Image

    img = Image.open(out).convert("RGB")
    w, h = img.size
    # Right half; skip a small top-left patch where only legend might appear.
    crop = img.crop((w // 2 + 5, 5, w - 5, h - 5))
    arr = np.asarray(crop)
    assert (arr < 250).any()
