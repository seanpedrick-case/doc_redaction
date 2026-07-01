"""Tests for AgentCore session store and packaging."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_AGENTCORE = _REPO / "agent-redact" / "agentcore"


def test_session_store_multi_turn():
    pytest = __import__("pytest")
    pytest.importorskip("langchain_core")
    if str(_AGENTCORE) not in sys.path:
        sys.path.insert(0, str(_AGENTCORE))
    from session_store import append_turn, clear_session, get_messages

    clear_session("abc")
    append_turn("abc", user_text="hello", assistant_text="hi there")
    messages = get_messages("abc")
    assert len(messages) == 2
    assert messages[0].content == "hello"
    assert messages[1].content == "hi there"
    clear_session("abc")
    assert get_messages("abc") == []


def test_bootstrap_runtime_env_loads_agentcore_env(tmp_path, monkeypatch):
    pytest = __import__("pytest")
    pytest.importorskip("dotenv")
    pytest.importorskip("langchain_core")
    if str(_AGENTCORE) not in sys.path:
        sys.path.insert(0, str(_AGENTCORE))
    from invoke_agent import bootstrap_runtime_env

    env_file = tmp_path / "agentcore.env"
    env_file.write_text(
        "DOC_REDACTION_GRADIO_URL=https://example.test\nPI_DEFAULT_PROVIDER=amazon-bedrock\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DOC_REDACTION_GRADIO_URL", raising=False)
    bootstrap_runtime_env(tmp_path)
    assert os.environ.get("DOC_REDACTION_GRADIO_URL") == "https://example.test"
    assert os.environ.get("PI_DEFAULT_PROVIDER") == "amazon-bedrock"


def test_package_runtime_dry_run(tmp_path):
    script = _AGENTCORE / "package_runtime.py"
    result = subprocess.run(
        [sys.executable, str(script), "--target", str(tmp_path), "--dry-run"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Packaging doc_redaction runtime" in result.stdout
