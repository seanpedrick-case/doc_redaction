from __future__ import annotations

from mcp_doc_redaction.artifact_bundle import bundle_artifacts


def test_bundle_artifacts_dedupes_by_hash() -> None:
    downloaded = {
        "/tmp/gradio/a/out1.txt": b"same",
        "/tmp/gradio/b/out2.txt": b"same",
        "/tmp/gradio/c/out3.txt": b"diff",
    }
    res = bundle_artifacts(
        produced_by="/x",
        base_url="https://example.com",
        downloaded=downloaded,
    )
    # 2 unique contents + manifest.json inside the zip
    assert len(res.manifest.files) == 2
    assert res.manifest.produced_by == "/x"
