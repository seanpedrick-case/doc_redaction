"""Tests for placeholder page image resolution before OCR visualization."""

from unittest.mock import MagicMock

import pytest
from PIL import Image


def test_open_page_image_for_pipeline_materialises_placeholder(monkeypatch, tmp_path):
    from tools import file_redaction as fr

    created_png = tmp_path / "doc.pdf_5.png"
    Image.new("RGB", (2, 2)).save(created_png)

    def _fake_convert(pdf_path, page_num, **kwargs):
        assert pdf_path == "/fake/doc.pdf"
        assert page_num == 5
        return page_num, str(created_png), 100.0, 200.0

    monkeypatch.setattr(fr, "process_single_page_for_image_conversion", _fake_convert)
    monkeypatch.setattr(fr, "validate_path_containment", lambda _p, _f: True)

    pil, resolved = fr.open_page_image_for_pipeline(
        "placeholder_image_5.png",
        "/fake/doc.pdf",
        5,
        input_folder=str(tmp_path),
    )

    assert resolved == str(created_png)
    assert pil is not None
    assert isinstance(pil, Image.Image)


def test_resolve_image_for_ocr_visualization_returns_pil_not_placeholder(
    monkeypatch, tmp_path
):
    from tools import file_redaction as fr

    created_png = tmp_path / "doc.pdf_2.png"
    Image.new("RGB", (2, 2)).save(created_png)

    monkeypatch.setattr(
        fr,
        "process_single_page_for_image_conversion",
        lambda *a, **k: (2, str(created_png), 10.0, 20.0),
    )
    monkeypatch.setattr(fr, "validate_path_containment", lambda _p, _f: True)

    result = fr.resolve_image_for_ocr_visualization(
        None,
        "placeholder_image_2.png",
        "/fake/doc.pdf",
        2,
        input_folder=str(tmp_path),
    )

    assert isinstance(result, Image.Image)
    assert not isinstance(result, str) or "placeholder" not in str(result)


def test_visualise_rejects_placeholder_without_context():
    from tools.file_redaction import visualise_ocr_words_bounding_boxes

    with pytest.raises(FileNotFoundError, match="placeholder"):
        visualise_ocr_words_bounding_boxes(
            "placeholder_image_0.png",
            {"line_1": {"text": "x", "bounding_box": (0, 0, 1, 1), "words": []}},
        )


def test_visualise_resolves_placeholder_when_file_path_given(monkeypatch, tmp_path):
    from tools import file_redaction as fr

    created_png = tmp_path / "doc.pdf_0.png"
    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    img.save(created_png)

    monkeypatch.setattr(
        fr,
        "process_single_page_for_image_conversion",
        lambda *a, **k: (0, str(created_png), 4.0, 4.0),
    )
    monkeypatch.setattr(fr, "validate_path_containment", lambda _p, _f: True)
    monkeypatch.setattr(fr, "cv2", MagicMock())
    monkeypatch.setattr(fr.np, "array", lambda x: __import__("numpy").array(x))
    monkeypatch.setattr(
        fr.cv2,
        "cvtColor",
        lambda arr, _code: arr,
    )
    monkeypatch.setattr(fr.cv2, "rectangle", lambda *a, **k: None)
    monkeypatch.setattr(fr.cv2, "putText", lambda *a, **k: None)
    monkeypatch.setattr(fr.cv2, "imwrite", lambda *a, **k: True)

    # Minimal path through visualise after open — only assert open succeeds
    resolved = fr.resolve_image_for_ocr_visualization(
        None,
        "placeholder_image_0.png",
        str(tmp_path / "doc.pdf"),
        0,
        input_folder=str(tmp_path),
    )
    assert isinstance(resolved, Image.Image)
