"""Regression tests for annotation colour normalisation in convert_annotation_data_to_dataframe."""

from __future__ import annotations

import pytest

from tools.secure_regex_utils import safe_extract_rgb_values

pytest.importorskip("scipy")

from tools.file_conversion import convert_annotation_data_to_dataframe
from tools.file_redaction import define_box_colour


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
    "text,expected",
    [
        ("(255, 128, 0)", (255, 128, 0)),
        ("255, 128, 0", (255, 128, 0)),
        ("rgb(201, 13, 13)", (201, 13, 13)),
        ("rgb( 10 , 20 , 30 )", (10, 20, 30)),
        ("rgba(255, 128, 64, 0.5)", (255, 128, 64)),
        ("#c90d0d", (201, 13, 13)),
        ("#abc", (170, 187, 204)),
        ("[128 64 32]", (128, 64, 32)),
        ("not-a-colour", None),
        ("", None),
    ],
)
def test_safe_extract_rgb_values(text, expected):
    assert safe_extract_rgb_values(text) == expected


@pytest.mark.parametrize(
    "color,expected_rgb_str",
    [
        ("rgba(0,0,0,1)", "(0, 0, 0)"),
        ("rgba(255, 128, 64, 0.5)", "(255, 128, 64)"),
        ("rgb(1, 2, 3)", "(1, 2, 3)"),
        ("rgb( 10 , 20 , 30 )", "(10, 20, 30)"),
        ("rgb(201, 13, 13)", "(201, 13, 13)"),
        ("(128,128,128)", "(128, 128, 128)"),
        ("128,128,128", "(128, 128, 128)"),
        ("#abc", "(170, 187, 204)"),
        ("#aabbcc", "(170, 187, 204)"),
        ("#c90d0d", "(201, 13, 13)"),
        ([10, 20, 30], "(10, 20, 30)"),
        ((1, 2, 3), "(1, 2, 3)"),
    ],
)
def test_convert_annotation_color_formats(color, expected_rgb_str):
    df = convert_annotation_data_to_dataframe(_minimal_annotations(color))
    assert len(df) == 1
    assert df.iloc[0]["color"] == expected_rgb_str


@pytest.mark.parametrize(
    "color",
    [
        "rgb(201, 13, 13)",
        "rgba(201, 13, 13, 1)",
        "(201, 13, 13)",
        "#c90d0d",
        (201, 13, 13),
        [201, 13, 13],
    ],
)
def test_define_box_colour_accepts_annotator_css_rgb(color, capsys):
    """GUI colour picker emits CSS rgb(); review PDF always uses custom colours."""
    colour = define_box_colour(True, {"color": color}, (0, 0, 0))
    assert colour == pytest.approx((201 / 255, 13 / 255, 13 / 255))
    captured = capsys.readouterr()
    assert "Could not parse color string" not in captured.out
    assert "Could not parse color string" not in captured.err
