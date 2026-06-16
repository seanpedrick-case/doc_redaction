"""Tests for staging Gradio HTTP uploads (tools.simplified_api)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("regex")

from tools.simplified_api import (
    _is_gradio_ephemeral_upload_path,
    preview_boxes_api,
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


def test_preview_boxes_api_staging_uses_single_path_arg(tmp_path: Path) -> None:
    """Regression: preview_boxes must not pass a second arg to stage_gradio_upload_if_ephemeral."""
    pdf = tmp_path / "doc.pdf"
    review = tmp_path / "plan_review_file.csv"
    pdf.write_bytes(b"%PDF-1.4")
    review.write_bytes(b"x")
    fake_png = tmp_path / "p1.png"
    fake_png.write_bytes(b"png")
    out_root = tmp_path / "outputs"
    out_root.mkdir()

    staging_calls: list[tuple[Any, ...]] = []

    def recording_stage(src: str) -> str:
        staging_calls.append((src,))
        return src

    with patch("tools.simplified_api.OUTPUT_FOLDER", str(out_root)):
        with patch(
            "tools.preview_redaction_boxes.preview_redaction_boxes",
            return_value=[str(fake_png)],
        ):
            with patch(
                "tools.simplified_api.stage_gradio_upload_if_ephemeral",
                side_effect=recording_stage,
            ):
                zip_path, msg = preview_boxes_api(str(pdf), str(review))

    assert len(staging_calls) == 2
    assert all(len(c) == 1 for c in staging_calls)
    assert zip_path.endswith("preview_boxes.zip")
    assert "Preview complete" in msg
