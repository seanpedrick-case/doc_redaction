"""Tests for Gradio example path resolution (HF /tmp/gradio cache fallback)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import example_data_paths as edp


def test_bundled_example_asset_available_rejects_lfs_pointer(tmp_path: Path) -> None:
    stub = tmp_path / "demo.pdf"
    stub.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\n",
        encoding="utf-8",
    )
    assert edp.bundled_example_asset_available(str(stub)) is False


def test_resolve_prefers_bundled_when_gradio_temp_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "example_data"
    data_root.mkdir()
    bundled = data_root / "sample.pdf"
    bundled.write_bytes(b"%PDF-1.4 demo")

    monkeypatch.setattr(edp, "resolve_example_data_dirs", lambda: [data_root])

    missing_cache = tmp_path / "gradio" / "abc123" / "sample.pdf"
    resolved = edp.resolve_gradio_example_file_path(str(missing_cache))
    assert resolved == str(bundled.resolve())


def test_resolve_keeps_temp_when_no_bundled_match(tmp_path: Path) -> None:
    temp_copy = tmp_path / "upload.pdf"
    temp_copy.write_bytes(b"%PDF-upload")
    assert edp.resolve_gradio_example_file_path(str(temp_copy)) == str(
        temp_copy.resolve()
    )


def test_resolve_gradio_example_file_paths_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "example_data"
    data_root.mkdir()
    bundled = data_root / "a.csv"
    bundled.write_text("x\n", encoding="utf-8")
    stale = tmp_path / "gradio" / "hash" / "a.csv"
    monkeypatch.setattr(edp, "resolve_example_data_dirs", lambda: [data_root])

    resolved = edp.resolve_gradio_example_file_paths([str(stale)])
    assert len(resolved) == 1
    assert Path(resolved[0]).read_text(encoding="utf-8") == "x\n"


def test_bundled_candidate_finds_example_outputs_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edp.bundled_example_candidate_path.cache_clear()
    monkeypatch.setattr(
        edp,
        "resolve_example_data_dirs",
        lambda: [tmp_path / "example_data"],
    )
    data_root = tmp_path / "example_data"
    nested = data_root / "example_outputs"
    nested.mkdir(parents=True)
    target = nested / "doubled_output_joined.pdf_ocr_output.csv"
    target.write_text("page,line\n", encoding="utf-8")

    found = edp.bundled_example_candidate_path(
        "doubled_output_joined.pdf_ocr_output.csv"
    )
    assert found is not None
    assert found.resolve() == target.resolve()
