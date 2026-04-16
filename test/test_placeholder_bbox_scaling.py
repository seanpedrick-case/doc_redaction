import pandas as pd
import pytest


def test_review_navigation_renders_placeholder_and_rescales_boxes(monkeypatch):
    """
    Regression test for: after actions like bulk exclude rebuild state, some pages may
    still have placeholder images and boxes stored in relative (0-1) coords because
    image_width/height weren't known yet. When navigating to such a page, the review
    UI should render the page image on-demand and rescale boxes to pixel coords.
    """

    pytest.importorskip("defusedxml")
    from tools import redaction_review as rr

    # 3 pages, where page 3 is a placeholder with missing image dims.
    page_sizes = [
        {
            "page": 1,
            "image_path": r"C:\fake\page1.png",
            "image_width": 800,
            "image_height": 1000,
        },
        {
            "page": 2,
            "image_path": r"C:\fake\page2.png",
            "image_width": 800,
            "image_height": 1000,
        },
        {
            "page": 3,
            "image_path": "placeholder_image_2.png",
            "image_width": float("nan"),
            "image_height": float("nan"),
        },
    ]

    all_image_annotations = [
        {"image": r"C:\fake\page1.png", "boxes": []},
        {"image": r"C:\fake\page2.png", "boxes": []},
        {
            "image": "placeholder_image_2.png",
            "boxes": [
                {
                    "xmin": 0.10,
                    "xmax": 0.20,
                    "ymin": 0.25,
                    "ymax": 0.35,
                    "label": "TEST",
                    "color": "rgba(0,0,0,1)",
                    "text": "UK",
                    "id": "abc",
                }
            ],
        },
    ]

    # Pretend the page render succeeded and we now know the image size for page 3.
    def _fake_replace_placeholder_image_with_real_image(
        doc_full_file_name_textbox: str,
        current_image_path: str,
        page_sizes_df: pd.DataFrame,
        page_num_reported: int,
        input_folder: str,
    ):
        replaced = r"C:\fake\page3.png"
        mask = (
            pd.to_numeric(page_sizes_df["page"], errors="coerce") == page_num_reported
        )
        page_sizes_df.loc[mask, "image_path"] = replaced
        page_sizes_df.loc[mask, "image_width"] = 1000
        page_sizes_df.loc[mask, "image_height"] = 2000
        return replaced, page_sizes_df

    monkeypatch.setattr(
        rr,
        "replace_placeholder_image_with_real_image",
        _fake_replace_placeholder_image_with_real_image,
    )

    out_annotator, *_rest = rr.update_annotator_object_and_filter_df(
        all_image_annotations=all_image_annotations,
        gradio_annotator_current_page_number=3,
        recogniser_entities_dropdown_value="ALL",
        page_dropdown_value="ALL",
        page_dropdown_redaction_value="1",
        text_dropdown_value="ALL",
        recogniser_dataframe_base=pd.DataFrame(columns=["page", "label", "text", "id"]),
        zoom=100,
        review_df=pd.DataFrame(
            columns=[
                "image",
                "page",
                "label",
                "color",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "id",
                "text",
            ]
        ),
        page_sizes=page_sizes,
        doc_full_file_name_textbox=r"C:\fake\doc.pdf",
        input_folder=r"C:\fake",
    )

    assert out_annotator is not None
    assert out_annotator["image"] == r"C:\fake\page3.png"
    assert out_annotator["boxes"], "Expected boxes on page 3"

    b = out_annotator["boxes"][0]
    # Expect relative coords multiplied by 1000x2000.
    assert b["xmin"] == 100.0
    assert b["xmax"] == 200.0
    assert b["ymin"] == 500.0
    assert b["ymax"] == 700.0
