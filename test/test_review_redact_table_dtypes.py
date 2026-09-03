"""Regression: Review-tab Gradio string dtypes vs int/float columns."""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")

import pandas as pd
import pytest

pytest.importorskip("gradio_image_annotation_redaction")

from tools.redaction_review import (
    _align_merge_key_columns,
    _ensure_numeric_columns,
    _page_equals,
    create_annotation_objects_from_filtered_ocr_results_with_words,
    exclude_selected_items_from_redaction,
    update_selected_review_df_row_colour,
)


def test_align_merge_keys_string_page_vs_int_page():
    """OCR/Gradio frames often store page as str; review rows as int64."""
    new_df = pd.DataFrame(
        {
            "page": pd.Series(["1", "1"], dtype="string"),
            "label": ["Redaction", "Redaction"],
            "xmin": [0.1, 0.3],
            "ymin": [0.2, 0.2],
            "xmax": [0.2, 0.4],
            "ymax": [0.3, 0.3],
            "text": ["hello", "world"],
            "image": ["page1.png", "page1.png"],
            "color": ["(0, 0, 0)", "(0, 0, 0)"],
            "id": ["a", "b"],
        }
    )
    existing_df = pd.DataFrame(
        {
            "page": pd.Series([1], dtype="int64"),
            "label": ["Redaction"],
            "xmin": [0.1],
            "ymin": [0.2],
            "xmax": [0.2],
            "ymax": [0.3],
            "text": ["hello"],
        }
    )
    key_cols = ["page", "label", "xmin", "ymin", "xmax", "ymax", "text"]

    # pandas 3 StringDtype rejects this loc assignment (int64 into dtype 'str').
    # pandas 2 may coerce silently; either way the helper below must succeed.
    probe = new_df.copy()
    try:
        probe.loc[:, key_cols] = probe.loc[:, key_cols].astype(
            existing_df.loc[:, key_cols].dtypes
        )
    except TypeError:
        pass

    aligned_new, aligned_existing = _align_merge_key_columns(
        new_df, existing_df, key_cols
    )
    merged = pd.merge(
        aligned_new,
        aligned_existing,
        on=key_cols,
        how="left",
        indicator=True,
    )
    unique = merged[merged["_merge"] == "left_only"]
    assert len(unique) == 1
    assert unique.iloc[0]["text"] == "world"


def test_page_equals_treats_string_and_int_as_same_page():
    pages = pd.Series(["1", "2", "1"], dtype="string")
    assert _page_equals(pages, 1).tolist() == [True, False, True]
    assert _page_equals(pages, "2").tolist() == [False, True, False]


def test_ensure_numeric_columns_allows_loc_float_assignment():
    df = pd.DataFrame({"xmin": pd.Series(["0.1", "0.9"], dtype="string")})
    df = _ensure_numeric_columns(df, ["xmin"])
    df.loc[df["xmin"] > 0.5, "xmin"] = 0.4
    assert df.loc[1, "xmin"] == pytest.approx(0.4)


def test_redact_all_text_in_table_string_page_ocr_vs_int_review():
    """End-to-end: clicking 'Redact all text in table' with mismatched page dtypes."""
    n = 3
    ocr_base = pd.DataFrame(
        {
            "page": pd.Series(["1"] * n, dtype="string"),
            "line": [1, 1, 2],
            "word_x0": pd.Series(["0.10", "0.25", "0.10"], dtype="string"),
            "word_y0": pd.Series(["0.20", "0.20", "0.40"], dtype="string"),
            "word_x1": pd.Series(["0.20", "0.35", "0.20"], dtype="string"),
            "word_y1": pd.Series(["0.30", "0.30", "0.50"], dtype="string"),
            "word_text": ["alpha", "beta", "gamma"],
            "index": [0, 1, 2],
        }
    )
    filtered = ocr_base[["page", "line", "word_text", "index"]].copy()
    page_sizes = [{"page": 1, "image_path": "page1.png"}]
    existing = pd.DataFrame(
        {
            "image": ["page1.png"],
            "page": pd.Series([1], dtype="int64"),
            "label": ["Redaction"],
            "color": ["(0, 0, 0)"],
            "xmin": [0.9],
            "ymin": [0.9],
            "xmax": [0.95],
            "ymax": [0.95],
            "text": ["already"],
            "id": ["existing-1"],
        }
    )

    result = create_annotation_objects_from_filtered_ocr_results_with_words(
        filtered_ocr_results_with_words_df=filtered,
        ocr_results_with_words_df_base=ocr_base,
        page_sizes=page_sizes,
        existing_annotations_df=existing,
        existing_annotations_list=[],
        existing_recogniser_entity_df=pd.DataFrame(),
        redaction_label="Redaction",
        colour_label="(0, 0, 0)",
        annotate_current_page=1,
        progress=lambda *args, **kwargs: None,
    )
    updated_annotations_df = result[2]
    assert not updated_annotations_df.empty
    texts = set(updated_annotations_df["text"].astype(str))
    assert "already" in texts
    assert {"alpha", "beta", "gamma"} <= texts


def _review_two_rows():
    return pd.DataFrame(
        {
            "image": ["page1.png", "page1.png"],
            "page": pd.Series([1, 1], dtype="int64"),
            "label": ["Redaction", "Redaction"],
            "color": ["(0, 0, 0)", "(0, 0, 0)"],
            "xmin": [0.1, 0.3],
            "ymin": [0.2, 0.2],
            "xmax": [0.2, 0.4],
            "ymax": [0.3, 0.3],
            "text": ["keep", "drop"],
            "id": [pd.NA, pd.NA],
        }
    )


def test_exclude_matches_string_page_against_int_review():
    """Exclude-by-text uses page as a merge key when ids are missing."""
    selected = pd.DataFrame(
        {
            "page": pd.Series(["1"], dtype="string"),
            "label": ["Redaction"],
            "text": ["drop"],
            "id": [pd.NA],
        }
    )
    page_sizes = [
        {
            "page": 1,
            "image_path": "page1.png",
            "image_width": 100,
            "image_height": 100,
        }
    ]
    out_review, *_ = exclude_selected_items_from_redaction(
        _review_two_rows(),
        selected,
        ["page1.png"],
        page_sizes,
        [],
        pd.DataFrame(),
    )
    assert list(out_review["text"].astype(str)) == ["keep"]


def test_highlight_matches_string_page_selection():
    """Row highlight fallback merge is label/page/text when ids are missing."""
    selection = pd.DataFrame(
        {
            "page": pd.Series(["1"], dtype="string"),
            "label": ["Redaction"],
            "text": ["drop"],
            "id": [pd.NA],
        }
    )
    review, _, _ = update_selected_review_df_row_colour(
        selection,
        _review_two_rows(),
        colour="(1, 0, 255)",
    )
    drop_colour = str(review.loc[review["text"].astype(str) == "drop", "color"].iloc[0])
    keep_colour = str(review.loc[review["text"].astype(str) == "keep", "color"].iloc[0])
    assert drop_colour == "(1, 0, 255)"
    assert keep_colour == "(0, 0, 0)"


def test_duplicate_annotations_match_string_ocr_pages():
    """Duplicate-page boxes should still attach when OCR page is a string."""
    from tools.find_duplicate_pages import create_annotation_objects_from_duplicates

    duplicates = pd.DataFrame({"Page2_Page": [1]})
    ocr = pd.DataFrame(
        {
            "page": pd.Series(["1"], dtype="string"),
            "line": [1],
            "left": [0.1],
            "top": [0.2],
            "width": [0.1],
            "height": [0.05],
            "text": ["dup"],
        }
    )
    page_sizes = [{"page": 1, "image_path": "page1.png"}]
    result = create_annotation_objects_from_duplicates(
        duplicates, ocr, page_sizes, combine_pages=False
    )
    assert result
    assert result[0]["image"] == "page1.png"
    assert result[0]["boxes"][0]["text"] == "dup"
