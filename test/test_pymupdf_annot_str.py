"""Regression: PyMuPDF annot contents must be a Python str, not NaN."""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import pandas as pd
import pytest

from tools.file_conversion import convert_annotation_data_to_dataframe
from tools.helper_functions import pymupdf_annot_str


def test_pymupdf_annot_str_treats_missing_and_nan_as_empty():
    assert pymupdf_annot_str(None) == ""
    assert pymupdf_annot_str(float("nan")) == ""
    assert pymupdf_annot_str(np.nan) == ""
    assert pymupdf_annot_str(pd.NA) == ""
    assert pymupdf_annot_str("") == ""
    assert pymupdf_annot_str("PERSON") == "PERSON"
    assert pymupdf_annot_str(12345) == "12345"
    assert pymupdf_annot_str(None, "Redaction") == "Redaction"


def test_convert_annotation_data_missing_text_is_empty_string():
    df = convert_annotation_data_to_dataframe(
        [
            {
                "image": "page_0.png",
                "boxes": [
                    {
                        "label": "Whole page",
                        "xmin": 0.01,
                        "ymin": 0.01,
                        "xmax": 0.99,
                        "ymax": 0.99,
                    }
                ],
            }
        ]
    )
    assert "text" in df.columns
    assert df.iloc[0]["text"] == ""


def test_redact_single_box_accepts_nan_text_from_duplicate_page_rows():
    pymupdf = pytest.importorskip("pymupdf")
    file_redaction = pytest.importorskip("tools.file_redaction")

    doc = pymupdf.open()
    try:
        page = doc.new_page()
        rect = pymupdf.Rect(10, 10, 80, 40)
        box = {
            "label": "Whole page",
            "text": float("nan"),
            "color": (0, 0, 0),
        }
        file_redaction.redact_single_box(
            page,
            rect,
            box,
            retain_text=True,
            return_pdf_end_of_redaction=False,
        )
        annots = list(page.annots())
        assert annots
        assert annots[0].info.get("content") == ""
        assert box["text"] == ""
    finally:
        doc.close()


def test_redact_page_with_pymupdf_whole_page_nan_text():
    pymupdf = pytest.importorskip("pymupdf")
    file_redaction = pytest.importorskip("tools.file_redaction")

    doc = pymupdf.open()
    try:
        page = doc.new_page()
        page_annotations = {
            "image": "page_0.png",
            "boxes": [
                {
                    "label": "Whole page",
                    "text": np.nan,
                    "color": (0, 0, 0),
                    "xmin": 0.01,
                    "ymin": 0.01,
                    "xmax": 0.99,
                    "ymax": 0.99,
                }
            ],
        }
        file_redaction.redact_page_with_pymupdf(
            page=page,
            page_annotations=page_annotations,
            image=None,
            original_cropbox=page.cropbox,
            page_sizes_df=pd.DataFrame(),
            return_pdf_for_review=True,
            return_pdf_end_of_redaction=False,
        )
        annots = [a for a in page.annots() if a.type[0] == pymupdf.PDF_ANNOT_REDACT]
        assert annots
        assert annots[0].info.get("content") == ""
    finally:
        doc.close()


def test_annotation_records_from_review_df_coerces_nan_text():
    redaction_review = pytest.importorskip("tools.redaction_review")
    df = pd.DataFrame(
        {
            "label": ["Whole page"],
            "color": ["(0, 0, 0)"],
            "xmin": [0.01],
            "ymin": [0.01],
            "xmax": [0.99],
            "ymax": [0.99],
            "text": [np.nan],
            "id": [pd.NA],
        }
    )
    records = redaction_review._annotation_records_from_review_df(
        df, ["label", "color", "xmin", "ymin", "xmax", "ymax", "text", "id"]
    )
    assert records[0]["text"] == ""
    assert records[0]["id"] == ""
    assert records[0]["label"] == "Whole page"
