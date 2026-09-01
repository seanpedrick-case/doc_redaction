"""Regression: collapsed/empty annotator boxes must not wipe page redactions."""

from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")

import pytest

pytest.importorskip("gradio_image_annotation_redaction")

from tools.redaction_review import (
    _annotator_box_has_area,
    _annotator_boxes_are_collapsed,
    coerce_gradio_client_annotator_payload,
    persist_current_page_and_refresh_annotator,
    refresh_annotator_after_external_layout_reflow,
    refresh_annotator_if_review_document_loaded,
    update_all_page_annotation_object_based_on_previous_page,
)


def test_annotator_box_has_area():
    assert _annotator_box_has_area({"xmin": 0.1, "ymin": 0.2, "xmax": 0.3, "ymax": 0.4})
    assert not _annotator_box_has_area(
        {"xmin": 0.0, "ymin": 0.0, "xmax": 0.0, "ymax": 0.0}
    )
    assert not _annotator_box_has_area({"xmin": 10, "ymin": 20, "xmax": 10, "ymax": 25})


def test_annotator_boxes_are_collapsed_includes_empty():
    # Empty and all-zero payloads are both unsafe to write over existing boxes
    # (Gradio often returns [] briefly on page turns).
    assert _annotator_boxes_are_collapsed(None) is True
    assert _annotator_boxes_are_collapsed([]) is True
    assert (
        _annotator_boxes_are_collapsed([{"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}])
        is True
    )
    assert (
        _annotator_boxes_are_collapsed(
            [
                {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0},
                {"xmin": 0.1, "ymin": 0.2, "xmax": 0.3, "ymax": 0.4},
            ]
        )
        is False
    )


def test_update_all_page_keeps_existing_when_incoming_boxes_collapsed():
    existing = {
        "image": "page_0.png",
        "boxes": [
            {
                "xmin": 0.1,
                "ymin": 0.2,
                "xmax": 0.3,
                "ymax": 0.4,
                "label": "PERSON",
                "id": "abc",
            }
        ],
    }
    collapsed = {
        "image": "page_0.png",
        "boxes": [
            {
                "xmin": 0,
                "ymin": 0,
                "xmax": 0,
                "ymax": 0,
                "label": "PERSON",
                "id": "abc",
            }
        ],
        "orientation": 0,
    }
    page_sizes = [
        {
            "page": 1,
            "image_path": "page_0.png",
            "image_width": 100,
            "image_height": 200,
        }
    ]

    updated, current, bottom = update_all_page_annotation_object_based_on_previous_page(
        collapsed,
        current_page=1,
        previous_page=1,
        all_image_annotations=[existing],
        page_sizes=page_sizes,
    )

    assert current == 1
    assert bottom == 1
    assert len(updated[0]["boxes"]) == 1
    assert updated[0]["boxes"][0]["xmin"] == 0.1
    assert updated[0]["boxes"][0]["ymax"] == 0.4


def test_update_all_page_keeps_existing_when_incoming_boxes_empty():
    existing = {
        "image": "page_0.png",
        "boxes": [
            {
                "xmin": 0.1,
                "ymin": 0.2,
                "xmax": 0.3,
                "ymax": 0.4,
                "label": "PERSON",
                "id": "abc",
            }
        ],
    }
    cleared = {"image": "page_0.png", "boxes": [], "orientation": 0}
    page_sizes = [
        {
            "page": 1,
            "image_path": "page_0.png",
            "image_width": 100,
            "image_height": 200,
        },
        {
            "page": 2,
            "image_path": "page_1.png",
            "image_width": 100,
            "image_height": 200,
        },
    ]

    updated, _, _ = update_all_page_annotation_object_based_on_previous_page(
        cleared,
        current_page=2,
        previous_page=1,
        all_image_annotations=[existing, {"image": "page_1.png", "boxes": []}],
        page_sizes=page_sizes,
    )

    assert len(updated[0]["boxes"]) == 1
    assert updated[0]["boxes"][0]["id"] == "abc"


def test_update_all_page_previous_page_zero_does_not_overwrite_last_page():
    state = [
        {
            "image": "doc_0.png",
            "boxes": [
                {
                    "xmin": 0.1,
                    "ymin": 0.2,
                    "xmax": 0.3,
                    "ymax": 0.4,
                    "label": "PERSON",
                    "id": "a",
                }
            ],
        },
        {
            "image": "doc_1.png",
            "boxes": [
                {
                    "xmin": 0.2,
                    "ymin": 0.3,
                    "xmax": 0.4,
                    "ymax": 0.5,
                    "label": "PERSON",
                    "id": "c",
                }
            ],
        },
    ]
    annotator = {
        "image": "doc_0.png",
        "boxes": [
            {
                "xmin": 100,
                "ymin": 280,
                "xmax": 300,
                "ymax": 560,
                "label": "PERSON",
                "id": "a",
            }
        ],
        "orientation": 0,
    }
    page_sizes = [
        {
            "page": 1,
            "image_path": "doc_0.png",
            "image_width": 1000,
            "image_height": 1400,
        },
        {
            "page": 2,
            "image_path": "doc_1.png",
            "image_width": 1000,
            "image_height": 1400,
        },
    ]

    updated, _, _ = update_all_page_annotation_object_based_on_previous_page(
        annotator,
        current_page=2,
        previous_page=0,  # initial State used to be 0
        all_image_annotations=state,
        page_sizes=page_sizes,
    )

    # previous_page=0 must coerce to page 1, not index -1 (last page)
    assert updated[1]["boxes"][0]["id"] == "c"
    assert updated[0]["boxes"][0]["id"] == "a"
    assert updated[0]["boxes"][0]["xmin"] == pytest.approx(0.1)


def test_refresh_annotator_after_external_layout_reflow_skips_when_inactive():
    import gradio as gr

    result = refresh_annotator_after_external_layout_reflow(
        layout_reflow_trigger=False,
        all_image_annotations=[{"image": "p.png", "boxes": []}],
        gradio_annotator_current_page_number=1,
        page_sizes=[{"page": 1}],
    )
    assert len(result) == 12
    assert all(v == gr.skip() for v in result)


def test_refresh_annotator_if_review_document_loaded_skips_without_review_doc():
    import gradio as gr

    result = refresh_annotator_if_review_document_loaded(
        all_image_annotations=[],
        gradio_annotator_current_page_number=1,
        page_sizes=[],
    )
    assert len(result) == 12
    assert all(v == gr.skip() for v in result)


def test_persist_current_page_and_refresh_annotator_skips_without_review_doc():
    import gradio as gr

    result = persist_current_page_and_refresh_annotator(
        page_image_annotator_object={"image": "p.png", "boxes": []},
        gradio_annotator_current_page_number=1,
        all_image_annotations=[],
        page_sizes=[],
    )
    assert len(result) == 12
    assert all(v == gr.skip() for v in result)


def test_coerce_gradio_client_annotator_payload_replaces_stale_gradio_tmp(
    tmp_path, monkeypatch
):
    import tools.redaction_review as rr

    monkeypatch.setattr(rr, "INPUT_FOLDER", str(tmp_path))
    stable = tmp_path / "doc_1.png"
    stable.write_bytes(b"png")
    page_sizes = [{"page": 1, "image_path": str(stable)}]
    payload = {
        "image": {
            "path": "/tmp/gradio_tmp/abc/doc_1.png",
            "url": "https://example.cloudfront.net/gradio_api/file=...",
        },
        "boxes": [{"xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4, "label": "PERSON"}],
    }
    coerced = coerce_gradio_client_annotator_payload(
        payload, 1, [{"image": "/tmp/gradio_tmp/old.png", "boxes": []}], page_sizes
    )
    assert coerced["image"] == str(stable)
    assert coerced["boxes"][0]["label"] == "PERSON"
