"""Tests for staging Gradio HTTP uploads (tools.simplified_api)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from tools.simplified_api import (
    _is_gradio_ephemeral_upload_path,
    stage_gradio_upload_if_ephemeral,
)


def test_is_gradio_ephemeral_upload_path() -> None:
    assert _is_gradio_ephemeral_upload_path("/tmp/gradio_tmp/abc/file.pdf")
    assert _is_gradio_ephemeral_upload_path(r"C:\tmp\gradio_tmp\x\y.pdf")
    assert _is_gradio_ephemeral_upload_path("/tmp/gradio/something")
    assert not _is_gradio_ephemeral_upload_path("/home/user/app/input/doc.pdf")
    assert not _is_gradio_ephemeral_upload_path("")


def test_stage_gradio_upload_copies_gradio_tmp(tmp_path: Path) -> None:
    fake_gradio = tmp_path / "gradio_tmp" / "hash"
    fake_gradio.mkdir(parents=True)
    src = fake_gradio / "doc.pdf"
    src.write_bytes(b"%PDF-1.4 test")

    with patch("tools.simplified_api._api_upload_staging_dir") as mock_staging:
        staging = str(tmp_path / "staging")
        mock_staging.return_value = staging
        out = stage_gradio_upload_if_ephemeral(str(src))

    assert out != str(src)
    assert Path(out).is_file()
    assert Path(out).read_bytes() == b"%PDF-1.4 test"


def test_stage_gradio_upload_skips_normal_paths(tmp_path: Path) -> None:
    p = tmp_path / "regular.pdf"
    p.write_bytes(b"x")
    with patch("tools.simplified_api._api_upload_staging_dir") as mock_staging:
        out = stage_gradio_upload_if_ephemeral(str(p))
    mock_staging.assert_not_called()
    assert out == str(p)
