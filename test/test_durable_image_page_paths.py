"""Standalone JPG/PNG inputs should get durable page PNGs under input_folder."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image


def test_process_file_for_image_creation_copies_jpg_into_input_folder(tmp_path):
    from tools import file_conversion as fc

    # process_single_page_for_image_conversion requires input_folder under cwd.
    input_dir = Path.cwd() / "input" / f"_pytest_durable_jpg_{os.getpid()}"
    input_dir.mkdir(parents=True, exist_ok=True)

    source = tmp_path / "handwritten_note.jpg"
    Image.new("RGB", (64, 48), color=(200, 180, 160)).save(source, format="JPEG")

    try:
        img_paths, widths, heights, details = fc.process_file_for_image_creation(
            str(source),
            input_folder=str(input_dir),
            create_images=True,
        )

        assert len(img_paths) == 1
        durable = Path(img_paths[0])
        assert durable.is_file()
        assert durable.suffix.lower() == ".png"
        assert durable.resolve().is_relative_to(input_dir.resolve())
        assert durable.name.startswith(source.name)
        assert int(widths[0]) == 64
        assert int(heights[0]) == 48
        assert details and details[0][1] == str(durable)
        # Must not keep the original source / Gradio-temp path.
        assert os.path.normcase(str(durable.resolve())) != os.path.normcase(
            str(source.resolve())
        )
    finally:
        for leftover in input_dir.glob("*"):
            try:
                leftover.unlink()
            except OSError:
                pass
        try:
            input_dir.rmdir()
        except OSError:
            pass


def test_process_file_for_image_creation_placeholder_when_create_images_false(
    tmp_path,
):
    from tools import file_conversion as fc

    input_dir = Path.cwd() / "input" / f"_pytest_durable_jpg_ph_{os.getpid()}"
    input_dir.mkdir(parents=True, exist_ok=True)

    source = tmp_path / "note.png"
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(source)

    try:
        img_paths, _widths, _heights, details = fc.process_file_for_image_creation(
            str(source),
            input_folder=str(input_dir),
            create_images=False,
        )
        assert len(img_paths) == 1
        assert "placeholder_image_0.png" in img_paths[0]
        assert details
    finally:
        try:
            input_dir.rmdir()
        except OSError:
            pass
