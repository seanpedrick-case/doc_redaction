"""Tests for word-level OCR dataframe conversion."""

from __future__ import annotations

from tools.file_conversion import (
    WORD_LEVEL_OCR_DF_COLUMNS,
    ensure_word_level_ocr_df_columns,
    word_level_ocr_output_to_dataframe,
)


def test_word_level_ocr_output_to_dataframe_empty_results_has_schema():
    df = word_level_ocr_output_to_dataframe([{"page": 1, "results": {}}])
    assert list(df.columns) == WORD_LEVEL_OCR_DF_COLUMNS
    assert df.empty


def test_ensure_word_level_ocr_df_columns_adds_missing_line_columns():
    import pandas as pd

    df = pd.DataFrame({"page": [1], "line": [1], "word_text": ["hi"]})
    out = ensure_word_level_ocr_df_columns(df)
    assert "line_x0" in out.columns
    assert out.loc[0, "line_x0"] == ""
