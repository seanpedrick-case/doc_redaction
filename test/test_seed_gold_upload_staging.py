"""Gold PDF staging into agent upload root."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "agent-redact" / "eval"
_SHARED = _REPO / "agent-redact" / "shared"
_PI = _REPO / "agent-redact" / "pi"
for p in (_EVAL, _SHARED, _PI):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_agent_regression import _stage_gold_pdf_under_upload_root  # noqa: E402


def test_stage_gold_pdf_copies_outside_upload_root(tmp_path, monkeypatch):
    upload = tmp_path / "uploads"
    upload.mkdir()
    outside = tmp_path / "gold" / "doc.pdf"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"%PDF-1.4 stub")

    monkeypatch.setenv("AGENT_UPLOAD_ROOT", str(upload))
    staged = _stage_gold_pdf_under_upload_root(outside, session_hash="bench123")
    assert staged.is_file()
    assert staged.parent.resolve() == upload.resolve()
    assert staged.name.startswith("bench123_")
