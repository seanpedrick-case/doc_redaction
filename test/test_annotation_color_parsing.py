"""Regression tests for annotation colour normalisation in convert_annotation_data_to_dataframe."""

from __future__ import annotations

import pytest

from tools.file_conversion import convert_annotation_data_to_dataframe


def _minimal_annotations(color):
    """One page, one box with valid coords and the given colour."""
    return [
        {
            "image": r"C:\fake\page1.png",
            "boxes": [
                {
                    "xmin": 0.1,
                    "xmax": 0.2,
                    "ymin": 0.25,
                    "ymax": 0.35,
                    "label": "T",
                    "color": color,
                    "text": "x",
                    "id": "1",
                }
            ],
        }
    ]


@pytest.mark.parametrize(
    "color,expected_rgb_str",
    [
        ("rgba(0,0,0,1)", "(0, 0, 0)"),
        ("rgba(255, 128, 64, 0.5)", "(255, 128, 64)"),
        ("rgb(1, 2, 3)", "(1, 2, 3)"),
        ("rgb( 10 , 20 , 30 )", "(10, 20, 30)"),
        ("(128,128,128)", "(128, 128, 128)"),
        ("128,128,128", "(128, 128, 128)"),
        ("#abc", "(170, 187, 204)"),
        ("#aabbcc", "(170, 187, 204)"),
        ([10, 20, 30], "(10, 20, 30)"),
        ((1, 2, 3), "(1, 2, 3)"),
    ],
)
def test_convert_annotation_color_formats(color, expected_rgb_str):
    df = convert_annotation_data_to_dataframe(_minimal_annotations(color))
    assert len(df) == 1
    assert df.iloc[0]["color"] == expected_rgb_str
