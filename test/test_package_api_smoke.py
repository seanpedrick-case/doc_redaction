"""
Smoke tests aligned with quarto_site/python_package_usage.qmd examples.

Keeps ``doc_redaction.api`` and ``merge_csv_files`` regressions from slipping in.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


def test_merge_csv_files_accepts_str_paths(repo_root: Path, tmp_path: Path) -> None:
    from tools.helper_functions import merge_csv_files

    f1 = (
        repo_root
        / "doc_redaction/example_data/example_outputs/Partnership-Agreement-Toolkit_0_0.pdf_review_file.csv"
    )
    f2 = (
        repo_root
        / "doc_redaction/example_data/example_outputs/example_of_emails_sent_to_a_professor_before_applying_review_file.csv"
    )
    assert f1.is_file() and f2.is_file()
    out = merge_csv_files([str(f1), str(f2)], output_folder=str(tmp_path) + os.sep)
    assert len(out) == 1
    assert Path(out[0]).is_file()


def test_merge_csv_files_accepts_gradio_like_named_objects(
    repo_root: Path, tmp_path: Path
) -> None:
    from tools.helper_functions import merge_csv_files

    class _Named:
        __slots__ = ("name",)

        def __init__(self, path: str) -> None:
            self.name = path

    f1 = (
        repo_root
        / "doc_redaction/example_data/example_outputs/Partnership-Agreement-Toolkit_0_0.pdf_review_file.csv"
    )
    f2 = (
        repo_root
        / "doc_redaction/example_data/example_outputs/example_of_emails_sent_to_a_professor_before_applying_review_file.csv"
    )
    out = merge_csv_files(
        [_Named(str(f1)), _Named(str(f2))], output_folder=str(tmp_path) + os.sep
    )
    assert Path(out[0]).is_file()


def test_combine_review_csvs_api(repo_root: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(repo_root)
    from doc_redaction.api import combine_review_csvs

    out = combine_review_csvs(
        input_files=[
            "doc_redaction/example_data/example_outputs/Partnership-Agreement-Toolkit_0_0.pdf_review_file.csv",
            "doc_redaction/example_data/example_outputs/example_of_emails_sent_to_a_professor_before_applying_review_file.csv",
        ],
        output_dir=str(tmp_path),
    )
    assert out
    assert Path(out[0]).exists()


def test_export_review_page_ocr_visualisation_dict_bbox(
    repo_root: Path, monkeypatch
) -> None:
    monkeypatch.chdir(repo_root)
    from doc_redaction.api import export_review_page_ocr_visualisation

    ocr_results = {
        "line_1": {
            "words": [
                {
                    "text": "Example",
                    "bounding_box": {
                        "left": 0.1,
                        "top": 0.1,
                        "width": 0.2,
                        "height": 0.05,
                    },
                    "conf": 0.99,
                }
            ]
        }
    }
    out = export_review_page_ocr_visualisation(
        page_image_path="doc_redaction/example_data/example_complaint_letter.jpg",
        ocr_results=ocr_results,
        page_number=1,
        doc_base_name="quarto_smoke_ocr_viz",
    )
    assert out
    assert Path(out[0]).exists()


def test_export_review_redaction_overlay_minimal(repo_root: Path, monkeypatch) -> None:
    pytest.importorskip(
        "gradio_image_annotation_redaction",
        reason="required by tools.redaction_review for overlay export",
    )
    monkeypatch.chdir(repo_root)
    from doc_redaction.api import export_review_redaction_overlay

    boxes = [
        {
            "label": "PERSON",
            "color": "#ff0000",
            "xmin": 0.1,
            "ymin": 0.1,
            "xmax": 0.4,
            "ymax": 0.2,
        }
    ]
    out = export_review_redaction_overlay(
        page_image_path="doc_redaction/example_data/example_complaint_letter.jpg",
        boxes=boxes,
        page_number=1,
        doc_base_name="quarto_smoke_overlay",
    )
    assert out
    assert Path(out[0]).exists()


def test_find_duplicate_pages_temp_output(repo_root: Path, monkeypatch) -> None:
    monkeypatch.chdir(repo_root)
    from doc_redaction.api import find_duplicate_pages

    out_dir = tempfile.mkdtemp(prefix="doc_redaction_dup_pages_smoke_")
    try:
        out_paths = find_duplicate_pages(
            input_files="doc_redaction/example_data/example_outputs/doubled_output_joined.pdf_ocr_output.csv",
            output_dir=out_dir,
            similarity_threshold=0.95,
        )
        assert isinstance(out_paths, list)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_load_and_prepare_documents_or_data_notimplemented() -> None:
    from doc_redaction.api import load_and_prepare_documents_or_data

    with pytest.raises(NotImplementedError):
        load_and_prepare_documents_or_data()


def test_word_level_ocr_text_search_notimplemented() -> None:
    from doc_redaction.api import word_level_ocr_text_search

    with pytest.raises(NotImplementedError):
        word_level_ocr_text_search()
