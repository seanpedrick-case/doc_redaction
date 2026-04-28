"""Tests for Gradio HTTP SSE completion parsing (mcp_doc_redaction.gradio_transport)."""

from __future__ import annotations

import pytest

from mcp_doc_redaction.gradio_transport import _parse_gradio_sse_final_payload


def test_sse_complete_returns_wrapped_list_payload() -> None:
    buf = """event: generating
data: ["x"]

event: complete
data: [{"path": "/tmp/out.pdf", "meta": {"_type": "gradio.FileData"}}]
"""
    out = _parse_gradio_sse_final_payload(buf)
    assert out is not None
    assert "data" in out
    assert isinstance(out["data"], list)
    assert out["data"][0]["path"] == "/tmp/out.pdf"


def test_sse_complete_dict_payload_returned_as_is() -> None:
    buf = """event: complete
data: {"path": "/tmp/single", "url": "https://example.com/f"}
"""
    out = _parse_gradio_sse_final_payload(buf)
    assert out is not None
    assert out["path"] == "/tmp/single"


def test_sse_incomplete_data_line_returns_none() -> None:
    buf = """event: complete
data: [{"path": "/tmp/x"
"""
    assert _parse_gradio_sse_final_payload(buf) is None


def test_sse_no_complete_returns_none() -> None:
    buf = """event: generating
data: ["hello"]
"""
    assert _parse_gradio_sse_final_payload(buf) is None


def test_sse_error_raises() -> None:
    buf = """event: error
data: "worker failed"
"""
    with pytest.raises(RuntimeError, match="event:error"):
        _parse_gradio_sse_final_payload(buf)


def test_sse_heartbeat_skipped() -> None:
    buf = """event: heartbeat
data: null

event: complete
data: ["done"]
"""
    out = _parse_gradio_sse_final_payload(buf)
    assert out == {"data": ["done"]}
