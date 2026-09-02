"""Tests for Gradio session memory release helpers."""

import os
from unittest.mock import MagicMock, patch

import pandas as pd

from tools.helper_functions import (
    ensure_ocr_word_results_list,
    release_document_session_state,
    release_post_workflow_ocr_state,
    release_session_document,
    reset_state_vars,
)


class _FakeDoc:
    def __init__(self, closed=False, page_count=1):
        self.is_closed = closed
        self.page_count = page_count
        self.close = MagicMock(side_effect=self._do_close)

    def _do_close(self):
        self.is_closed = True


def test_release_session_document_closes_open_doc():
    doc = _FakeDoc(closed=False)
    release_session_document(doc)
    doc.close.assert_called_once()
    assert doc.is_closed is True


def test_release_session_document_noop_for_empty_values():
    release_session_document(None)
    release_session_document([])
    already_closed = _FakeDoc(closed=True)
    release_session_document(already_closed)
    already_closed.close.assert_not_called()


def test_reset_state_vars_closes_doc_and_returns_legacy_tuple():
    doc = _FakeDoc()
    result = reset_state_vars(doc)
    doc.close.assert_called_once()
    assert len(result) == 22
    assert result[0] == []
    assert isinstance(result[1], pd.DataFrame) and result[1].empty


def test_release_document_session_state_full_tuple():
    doc = _FakeDoc(page_count=50)
    with patch("tools.file_conversion.clear_threadlocal_pdf_cache") as clear_cache:
        result = release_document_session_state(doc)
    doc.close.assert_called_once()
    clear_cache.assert_called_once()
    assert len(result) == 35
    assert result[10] == []
    assert isinstance(result[34], pd.DataFrame) and result[34].empty


def test_release_post_workflow_ocr_state_clears_lists_and_backups():
    result = release_post_workflow_ocr_state()
    assert result[0] == []
    assert result[1] == []
    assert isinstance(result[2], pd.DataFrame) and result[2].empty
    assert result[3] == []
    assert isinstance(result[4], pd.DataFrame) and result[4].empty
    assert isinstance(result[5], pd.DataFrame) and result[5].empty


def test_release_post_workflow_preserves_review_df_for_undo_baseline():
    """Post-workflow OCR release clears backup_* only; exclude/undo repopulates from review_df."""
    cleared = release_post_workflow_ocr_state()
    assert cleared[2].empty
    review_df = pd.DataFrame([{"page": 1, "text": "x", "label": "PERSON"}])
    fresh_backup = review_df.copy()
    assert not fresh_backup.empty


def test_ensure_ocr_word_results_list_returns_existing():
    existing = [{"page": 1, "results": []}]
    assert ensure_ocr_word_results_list(existing, "/missing.json") is existing


def test_ensure_ocr_word_results_list_rejects_unresolved_path():
    with patch(
        "tools.helper_functions.resolve_existing_io_path",
        side_effect=ValueError("Path not allowed"),
    ):
        assert ensure_ocr_word_results_list([], "/any/ocr.json") == []


def test_ensure_ocr_word_results_list_lazy_loads_under_output_folder(
    tmp_path, monkeypatch
):
    from tools import config

    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(config, "OUTPUT_FOLDER", str(output_root) + os.sep)
    monkeypatch.setattr(config, "INPUT_FOLDER", str(tmp_path / "input") + os.sep)

    ocr_path = output_root / "doc_ocr_results_with_words_tesseract.json"
    ocr_path.write_text(
        '[{"page": 1, "results": {"line_1": {"line": 1, "text": "hello", "words": []}}}]',
        encoding="utf-8",
    )
    loaded = ensure_ocr_word_results_list([], str(ocr_path))
    assert len(loaded) == 1
    assert loaded[0]["page"] == 1


def test_ensure_ocr_word_results_list_resolves_word_json_from_line_csv(
    tmp_path, monkeypatch
):
    from tools import config

    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(config, "OUTPUT_FOLDER", str(output_root) + os.sep)
    monkeypatch.setattr(config, "INPUT_FOLDER", str(tmp_path / "input") + os.sep)

    line_csv = output_root / "doc_ocr_output_tesseract_pages_6-6.csv"
    line_csv.write_text("page,line,text\n6,1,hello\n", encoding="utf-8")
    word_json = output_root / "doc_ocr_results_with_words_tesseract_pages_6-6.json"
    word_json.write_text(
        '[{"page": 6, "results": {"line_1": {"line": 1, "text": "hello", "words": []}}}]',
        encoding="utf-8",
    )
    loaded = ensure_ocr_word_results_list([], str(line_csv))
    assert len(loaded) == 1
    assert loaded[0]["page"] == 6


def test_ensure_ocr_word_results_list_skips_line_csv_without_word_json(
    tmp_path, monkeypatch
):
    from tools import config

    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(config, "OUTPUT_FOLDER", str(output_root) + os.sep)
    monkeypatch.setattr(config, "INPUT_FOLDER", str(tmp_path / "input") + os.sep)

    line_csv = output_root / "doc_ocr_output_tesseract.csv"
    line_csv.write_text("page,line,text\n6,1,hello\n", encoding="utf-8")
    assert ensure_ocr_word_results_list([], str(line_csv)) == []


def test_clear_threadlocal_pdf_cache_closes_cached_docs():
    from tools import file_conversion

    doc = _FakeDoc()
    file_conversion._PDF_DOC_CACHE.docs = {"/tmp/test.pdf": doc}
    file_conversion.clear_threadlocal_pdf_cache()
    doc.close.assert_called_once()
    assert file_conversion._PDF_DOC_CACHE.docs == {}
