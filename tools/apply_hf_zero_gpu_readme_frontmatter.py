"""
Rewrite README.md YAML front matter for Hugging Face Spaces (Zero GPU / Gradio).

Used by .github/workflows/sync_to_hf_zero_gpu.yml only. The committed README on
GitHub is unchanged; this runs in CI on the checkout before push to HF.
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
sdk_version: 6.9.0
app_file: app.py
pinned: true
license: agpl-3.0
short_description: OCR and redact PDF docs or images with VLMs
---
"""


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    readme = root / "README.md"
    if not readme.is_file():
        print("README.md not found", file=sys.stderr)
        return 1
    text = readme.read_text(encoding="utf-8")
    pattern = r"^---\s*\n.*?\n---\s*\n"
    if re.match(pattern, text, flags=re.DOTALL):
        text = re.sub(pattern, HF_ZERO_GPU_FRONT_MATTER, text, count=1, flags=re.DOTALL)
    else:
        text = HF_ZERO_GPU_FRONT_MATTER + text
    readme.write_text(text, encoding="utf-8")
    print("Patched README.md front matter for HF Zero GPU Space.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
