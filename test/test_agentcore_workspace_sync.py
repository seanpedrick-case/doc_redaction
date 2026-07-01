"""Tests for AgentCore workspace file sync helpers."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

_AGENTCORE = Path(__file__).resolve().parents[1] / "agent-redact" / "agentcore"
if str(_AGENTCORE) not in sys.path:
    sys.path.insert(0, str(_AGENTCORE))

from workspace_sync import (  # noqa: E402
    apply_workspace_files,
    collect_workspace_files_for_sync,
)


def test_apply_and_collect_workspace_files(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("PI_SESSION_WORKSPACE", "1")
    payload = base64.b64encode(b"pdf-bytes").decode("ascii")
    written = apply_workspace_files(
        "sess-1",
        [{"relative_path": "example.pdf", "content_base64": payload}],
    )
    assert written == ["example.pdf"]
    dest = tmp_path / "sess-1" / "example.pdf"
    assert dest.read_bytes() == b"pdf-bytes"

    redact_dir = tmp_path / "sess-1" / "redact" / "example" / "output_redact"
    redact_dir.mkdir(parents=True)
    (redact_dir / "out.pdf").write_bytes(b"redacted")
    synced = collect_workspace_files_for_sync("sess-1")
    assert len(synced) == 1
    assert synced[0]["relative_path"].endswith("out.pdf")
