"""
Rewrite README.md YAML front matter and pin torch for Hugging Face Spaces
(Zero GPU / Gradio).

Used by .github/workflows/sync_to_hf_zero_gpu.yml only. The committed README
and requirements.txt on GitHub are unchanged; this runs in CI on the checkout
before push to HF.

ZeroGPU currently accepts torch 2.8.0, 2.9.1, 2.10.0, and 2.11.0 only. The
repo pins a newer torch for security; this script rewrites the Space copy to
the newest supported version and a matching torchvision.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Must match what the Space expects (Gradio SDK metadata).
HF_ZERO_GPU_FRONT_MATTER = """---
title: Document OCR and redaction with VLMs
emoji: ⚡
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: true
license: agpl-3.0
short_description: OCR and redact PDF docs or images with VLMs
---
"""

# Newest torch ZeroGPU accepts (CONFIG_ERROR if requirements pin anything else).
HF_ZERO_GPU_TORCH = "2.11.0"
# torchvision minor that declares Requires-Dist: torch==2.11.0
HF_ZERO_GPU_TORCHVISION = "0.26.0"

_README_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", flags=re.DOTALL)
# Do not match torchvision / torchaudio / torchcodec.
_TORCH_PIN_RE = re.compile(
    r"^torch(?![\w.-])(?:==|<=|>=|~=|!=|>|<)[^\s#]+",
    flags=re.MULTILINE,
)
_TORCHVISION_PIN_RE = re.compile(
    r"^torchvision(?:==|<=|>=|~=|!=|>|<)[^\s#]+",
    flags=re.MULTILINE,
)
_PYTORCH_CUDA_INDEX_RE = re.compile(
    r"^--extra-index-url https://download\.pytorch\.org/whl/cu\d+\s*\n",
    flags=re.MULTILINE,
)
_PYTORCH_SECTION_COMMENT_RE = re.compile(
    r"^# --- PyTorch \(CUDA \d+(?:\.\d+)?\) ---",
    flags=re.MULTILINE,
)


def patch_readme(root: Path) -> None:
    readme = root / "README.md"
    if not readme.is_file():
        raise FileNotFoundError("README.md not found")
    text = readme.read_text(encoding="utf-8")
    if _README_FRONT_MATTER_RE.match(text):
        text = _README_FRONT_MATTER_RE.sub(HF_ZERO_GPU_FRONT_MATTER, text, count=1)
    else:
        text = HF_ZERO_GPU_FRONT_MATTER + text
    readme.write_text(text, encoding="utf-8")
    print("Patched README.md front matter for HF Zero GPU Space.")


def patch_requirements(root: Path) -> None:
    requirements = root / "requirements.txt"
    if not requirements.is_file():
        raise FileNotFoundError("requirements.txt not found")
    text = requirements.read_text(encoding="utf-8")

    text = _PYTORCH_CUDA_INDEX_RE.sub("", text)
    text = _PYTORCH_SECTION_COMMENT_RE.sub(
        "# --- PyTorch (ZeroGPU-compatible pin; CUDA provided by Space runtime) ---",
        text,
    )
    text, n_torch = _TORCH_PIN_RE.subn(f"torch=={HF_ZERO_GPU_TORCH}", text, count=1)
    if n_torch == 0:
        raise ValueError("No torch pin found in requirements.txt")
    text, n_vision = _TORCHVISION_PIN_RE.subn(
        f"torchvision=={HF_ZERO_GPU_TORCHVISION}",
        text,
        count=1,
    )
    if n_vision == 0:
        raise ValueError("No torchvision pin found in requirements.txt")

    requirements.write_text(text, encoding="utf-8")
    print(
        "Patched requirements.txt for HF Zero GPU Space: "
        f"torch=={HF_ZERO_GPU_TORCH}, torchvision=={HF_ZERO_GPU_TORCHVISION}."
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        patch_readme(root)
        patch_requirements(root)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
