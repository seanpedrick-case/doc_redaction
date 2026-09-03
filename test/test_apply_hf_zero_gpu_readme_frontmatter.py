"""Tests for the HF Zero GPU checkout patches applied before Space sync."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.apply_hf_zero_gpu_readme_frontmatter import (
    HF_ZERO_GPU_FRONT_MATTER,
    HF_ZERO_GPU_TORCH,
    HF_ZERO_GPU_TORCHVISION,
    patch_readme,
    patch_requirements,
)


def test_patch_readme_replaces_existing_front_matter(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text("---\nsdk: docker\n---\n# Title\n", encoding="utf-8")

    patch_readme(tmp_path)

    text = readme.read_text(encoding="utf-8")
    assert text.startswith(HF_ZERO_GPU_FRONT_MATTER)
    assert text.endswith("# Title\n")
    assert "sdk: docker" not in text


def test_patch_requirements_pins_zerogpu_torch_and_drops_cuda_index(tmp_path: Path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "\n".join(
            [
                "gradio==6.26.0",
                "",
                "# --- PyTorch (CUDA 12.8) ---",
                "--extra-index-url https://download.pytorch.org/whl/cu128",
                "torch==2.13.0",
                "torchvision>=0.28.0",
                "torchaudio==2.13.0",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    patch_requirements(tmp_path)

    text = requirements.read_text(encoding="utf-8")
    assert f"torch=={HF_ZERO_GPU_TORCH}" in text
    assert "torch==2.13.0" not in text
    assert f"torchvision=={HF_ZERO_GPU_TORCHVISION}" in text
    assert "torchvision>=0.28.0" not in text
    assert "download.pytorch.org/whl/cu128" not in text
    assert "ZeroGPU-compatible pin" in text
    assert "gradio==6.26.0" in text
    assert "torchaudio==2.13.0" in text


def test_patch_requirements_leaves_file_unchanged_if_torchvision_missing(
    tmp_path: Path,
):
    requirements = tmp_path / "requirements.txt"
    original = "torch==2.13.0\ntorchaudio==2.13.0\n"
    requirements.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="No torchvision pin"):
        patch_requirements(tmp_path)

    assert requirements.read_text(encoding="utf-8") == original
