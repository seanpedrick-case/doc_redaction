from __future__ import annotations

from mcp_doc_redaction.gradio_transport import extract_file_like_paths


def test_extract_file_like_paths() -> None:
    payload = {
        "data": [
            "/tmp/gradio/a.pdf",
            {"path": "/tmp/gradio/b.csv", "orig_name": "b.csv"},
            [{"path": "/tmp/gradio/c.json"}],
            {"nested": {"path": "/tmp/gradio/d.png"}},
            "not/a/path",
            None,
        ]
    }
    paths = extract_file_like_paths(payload)
    assert paths == [
        "/tmp/gradio/a.pdf",
        "/tmp/gradio/b.csv",
        "/tmp/gradio/c.json",
        "/tmp/gradio/d.png",
    ]
