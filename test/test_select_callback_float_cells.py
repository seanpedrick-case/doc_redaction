"""Regression: table row entries arrive as floats after the Gradio JSON round-trip.

A dataframe column with any missing cell is rebuilt as float64 on the way back
from the frontend, so ``page``/``line``/``index`` cells arrive as ``3.0`` or
``None``/``NaN``. ``int(None)`` / ``int(float("nan"))`` raised in the select
callbacks after a text search, crashing the click handler. The callbacks must
coerce cells to nullable ints instead.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")

import pytest

pytest.importorskip("gradio")

import gradio as gr
import pandas as pd

from tools.redaction_review import (
    _coerce_select_cell_int,
    create_annotation_objects_from_filtered_ocr_results_with_words,
    df_select_callback_dataframe_row,
    df_select_callback_dataframe_row_ocr_with_words,
    df_select_callback_ocr,
)


def _evt(row):
    return gr.SelectData(
        target=None,
        data={"index": (0, 0), "value": row[0], "row_value": row, "col_value": row[0]},
    )


def test_coerce_select_cell_int_handles_float_str_and_missing():
    assert _coerce_select_cell_int(3) == 3
    assert _coerce_select_cell_int(3.0) == 3
    assert _coerce_select_cell_int("3") == 3
    assert _coerce_select_cell_int(3.7) == 4
    assert pd.isna(_coerce_select_cell_int(None))
    assert pd.isna(_coerce_select_cell_int(float("nan")))
    assert pd.isna(_coerce_select_cell_int("abc"))
    assert _coerce_select_cell_int(None, default=1) == 1


def test_ocr_with_words_callback_float_row_stays_int():
    df_sel, text = df_select_callback_dataframe_row_ocr_with_words(
        None, _evt([3.0, 1.0, "secret", 5.0])
    )
    assert str(df_sel["page"].dtype) == "Int64"
    assert str(df_sel["line"].dtype) == "Int64"
    assert str(df_sel["index"].dtype) == "Int64"
    assert df_sel["page"].iloc[0] == 3
    assert df_sel["line"].iloc[0] == 1
    assert df_sel["index"].iloc[0] == 5
    assert text == "secret"


def test_ocr_with_words_callback_missing_page_does_not_raise():
    for row in ([None, 11, "to", 1184], [float("nan"), 11.0, "to", 1184.0]):
        df_sel, text = df_select_callback_dataframe_row_ocr_with_words(None, _evt(row))
        assert pd.isna(df_sel["page"].iloc[0])
        assert df_sel["line"].iloc[0] == 11
        assert df_sel["index"].iloc[0] == 1184
        assert text == "to"


def test_ocr_with_words_callback_int_row_unchanged():
    df_sel, text = df_select_callback_dataframe_row_ocr_with_words(
        None, _evt([2, 1, "hello", 7])
    )
    assert df_sel.to_dict("records") == [
        {"page": 2, "line": 1, "word_text": "hello", "index": 7}
    ]
    assert text == "hello"


def test_dataframe_row_callback_coerces_float_page():
    df_sel, text = df_select_callback_dataframe_row(
        None, _evt([3.0, "PERSON", "John", "id1"])
    )
    assert str(df_sel["page"].dtype) == "Int64"
    assert df_sel["page"].iloc[0] == 3
    assert text == "John"


def test_ocr_callback_missing_page_falls_back_to_page_1():
    page, df_sel = df_select_callback_ocr(None, _evt([None, "some text"]))
    assert page == 1
    assert df_sel["page"].iloc[0] == 1
    page, _ = df_select_callback_ocr(None, _evt([4.0, "some text"]))
    assert page == 4


def test_redact_lookup_tolerates_float64_index_column_with_nan():
    base = pd.DataFrame(
        {
            "page": [1, 2, 3],
            "line": [1, 1, 1],
            "word_text": ["a", "b", "secret"],
            "word_x0": [0.1, 0.1, 0.1],
            "word_y0": [0.1, 0.1, 0.1],
            "word_x1": [0.2, 0.2, 0.2],
            "word_y1": [0.2, 0.2, 0.2],
            "index": [0, 1, 2],
        }
    )
    base.index = [0, 1, 2]
    # As reconstructed by gr.Dataframe.preprocess from JSON: float64 + NaN.
    selected = pd.DataFrame(
        [[3.0, 1.0, "secret", 2.0], [float("nan"), 1.0, "x", float("nan")]],
        columns=["page", "line", "word_text", "index"],
    )
    assert str(selected["index"].dtype) == "float64"
    page_sizes = [
        {"page": i, "image_path": f"p{i}.png", "image_width": 100, "image_height": 100}
        for i in (1, 2, 3)
    ]
    result = create_annotation_objects_from_filtered_ocr_results_with_words(
        selected,
        base,
        page_sizes,
        pd.DataFrame(),
        [],
        pd.DataFrame(),
        "Redaction",
        "(0, 0, 0)",
        1,
    )
    new_annotations_df = result[2]
    assert len(new_annotations_df) == 1
    assert new_annotations_df["text"].iloc[0] == "secret"
    assert int(new_annotations_df["page"].iloc[0]) == 3
