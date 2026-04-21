from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_doc_redaction.artifact_bundle import bundle_artifacts, zip_bytes_to_base64
from mcp_doc_redaction.gradio_transport import GradioHttpClient, extract_file_like_paths
from mcp_doc_redaction.schemas import (
    ApplyReviewOptions,
    RedactDocumentOptions,
    RedactTabularOptions,
    SummariseOptions,
)


def _client() -> GradioHttpClient:
    base_url = os.environ.get("DOC_REDACTION_BASE_URL", "").strip()
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    return GradioHttpClient(base_url=base_url, hf_token=hf_token)


mcp = FastMCP("doc_redaction")


@mcp.tool()
def status() -> dict[str, Any]:
    """
    Check connectivity and list available short endpoints.
    """
    c = _client()
    try:
        info = c.info()
        named = info.get("named_endpoints") or {}
        keys = sorted(str(k) for k in named.keys())
        return {
            "base_url": c.base_url,
            "endpoint_count": len(keys),
            "endpoints": keys,
            "preferred_short_endpoints_present": {
                "/apply_review_redactions_from_uploads": "/apply_review_redactions_from_uploads"
                in named,
                "/summarise_document_from_upload": "/summarise_document_from_upload" in named,
                "/redact_data_from_upload": "/redact_data_from_upload" in named,
            },
        }
    finally:
        c.close()


@mcp.tool()
def apply_review_redactions(
    *,
    pdf_bytes: bytes,
    pdf_filename: str,
    review_csv_bytes: bytes,
    review_csv_filename: str,
    options: ApplyReviewOptions | None = None,
) -> dict[str, Any]:
    """
    Apply edited review CSV to a PDF (preferred endpoint: /apply_review_redactions_from_uploads).
    Returns a base64 zip and a manifest.
    """
    opt = options or ApplyReviewOptions()
    c = _client()
    try:
        api_name = "/apply_review_redactions_from_uploads"
        if not c.endpoint_exists(api_name):
            raise RuntimeError(f"Endpoint not available on server: {api_name}")

        pdf_path = c.upload_bytes(pdf_filename, pdf_bytes)
        csv_path = c.upload_bytes(review_csv_filename, review_csv_bytes)

        event_id = c.call(api_name, [pdf_path, csv_path, opt.output_dir])
        completed = c.poll(api_name, event_id)

        file_paths = extract_file_like_paths(completed.payload)
        downloaded: dict[str, bytes] = {}
        notes: list[str] = []
        for p in file_paths:
            try:
                downloaded[p] = c.download(p)
            except Exception as e:
                notes.append(f"Failed to download {p}: {e}")

        bundled = bundle_artifacts(
            produced_by=api_name,
            base_url=c.base_url,
            downloaded=downloaded,
            notes=notes,
            extra={"returned_paths": file_paths},
        )
        return {
            "zip_base64": zip_bytes_to_base64(bundled.zip_bytes),
            "manifest": bundled.manifest.model_dump(),
        }
    finally:
        c.close()


@mcp.tool()
def summarise_document(
    *,
    pdf_bytes: bytes,
    pdf_filename: str,
    options: SummariseOptions | None = None,
) -> dict[str, Any]:
    """
    Summarise a PDF (preferred endpoint: /summarise_document_from_upload).
    Returns a base64 zip and a manifest.
    """
    opt = options or SummariseOptions()
    c = _client()
    try:
        api_name = "/summarise_document_from_upload"
        if not c.endpoint_exists(api_name):
            raise RuntimeError(f"Endpoint not available on server: {api_name}")

        pdf_path = c.upload_bytes(pdf_filename, pdf_bytes)
        data = [
            pdf_path,
            opt.ocr_method,
            opt.summarisation_inference_method,
            opt.summarisation_format,
            opt.summarisation_context,
            opt.summarisation_additional_instructions,
            opt.summarisation_temperature,
            opt.summarisation_max_pages_per_group,
            opt.summarisation_api_key,
            opt.output_dir,
            opt.input_dir,
            opt.page_min,
            opt.page_max,
        ]
        event_id = c.call(api_name, data)
        completed = c.poll(api_name, event_id)

        file_paths = extract_file_like_paths(completed.payload)
        downloaded: dict[str, bytes] = {}
        notes: list[str] = []
        for p in file_paths:
            try:
                downloaded[p] = c.download(p)
            except Exception as e:
                notes.append(f"Failed to download {p}: {e}")

        bundled = bundle_artifacts(
            produced_by=api_name,
            base_url=c.base_url,
            downloaded=downloaded,
            notes=notes,
            extra={"returned_paths": file_paths},
        )
        return {
            "zip_base64": zip_bytes_to_base64(bundled.zip_bytes),
            "manifest": bundled.manifest.model_dump(),
        }
    finally:
        c.close()


@mcp.tool()
def redact_tabular(
    *,
    file_bytes: bytes,
    filename: str,
    entities: list[str],
    options: RedactTabularOptions | None = None,
) -> dict[str, Any]:
    """
    Redact a tabular file (preferred endpoint: /redact_data_from_upload).

    This tool FAILS if the simplified endpoint is not deployed on the target server.
    """
    opt = options or RedactTabularOptions()
    c = _client()
    try:
        api_name = "/redact_data_from_upload"
        if not c.endpoint_exists(api_name):
            raise RuntimeError(
                f"Endpoint not available on server: {api_name}. "
                "Redeploy the app with the simplified gr.api wrapper enabled, "
                "or use the long /redact_data endpoint manually."
            )

        path = c.upload_bytes(filename, file_bytes)
        data = [
            path,
            list(entities or []),
            opt.output_dir,
            opt.pii_method,
            list(opt.columns or []),
            opt.anon_strategy,
            list(opt.allow_list or []),
            list(opt.deny_list or []),
            opt.language,
            opt.max_fuzzy_spelling_mistakes_num,
            opt.do_initial_clean,
            opt.llm_instruction,
            list(opt.llm_entities or []),
            list(opt.comprehend_entities or []),
            opt.aws_access_key,
            opt.aws_secret_key,
        ]
        event_id = c.call(api_name, data)
        completed = c.poll(api_name, event_id)

        file_paths = extract_file_like_paths(completed.payload)
        downloaded: dict[str, bytes] = {}
        notes: list[str] = []
        for p in file_paths:
            try:
                downloaded[p] = c.download(p)
            except Exception as e:
                notes.append(f"Failed to download {p}: {e}")

        bundled = bundle_artifacts(
            produced_by=api_name,
            base_url=c.base_url,
            downloaded=downloaded,
            notes=notes,
            extra={"returned_paths": file_paths},
        )
        return {
            "zip_base64": zip_bytes_to_base64(bundled.zip_bytes),
            "manifest": bundled.manifest.model_dump(),
        }
    finally:
        c.close()


@mcp.tool()
def redact_document(
    *,
    file_bytes: bytes,
    filename: str,
    entities: list[str],
    options: RedactDocumentOptions | None = None,
) -> dict[str, Any]:
    """
    Redact a PDF/image using /redact_document. Returns a base64 zip and manifest.

    Note: /redact_document is a long, UI-wired handler in many deployments; this tool
    may require additional fields depending on the target server’s /gradio_api/info.
    """
    opt = options or RedactDocumentOptions()
    c = _client()
    try:
        api_name = "/redact_document"
        if not c.endpoint_exists(api_name):
            raise RuntimeError(f"Endpoint not available on server: {api_name}")

        path = c.upload_bytes(filename, file_bytes)

        # Minimal attempt: file + entities + method knobs. If the server requires
        # more args, users should call /gradio_api/info and extend this tool.
        data = [
            [path],  # file_paths is a list
            opt.ocr_method,
            opt.pii_method,
            [],  # placeholders for other UI inputs; may need extension per deployment
            list(entities or []),
        ]
        event_id = c.call(api_name, data)
        completed = c.poll(api_name, event_id)

        file_paths = extract_file_like_paths(completed.payload)
        downloaded: dict[str, bytes] = {}
        notes: list[str] = []
        for p in file_paths:
            try:
                downloaded[p] = c.download(p)
            except Exception as e:
                notes.append(f"Failed to download {p}: {e}")

        bundled = bundle_artifacts(
            produced_by=api_name,
            base_url=c.base_url,
            downloaded=downloaded,
            notes=notes,
            extra={"returned_paths": file_paths},
        )
        return {
            "zip_base64": zip_bytes_to_base64(bundled.zip_bytes),
            "manifest": bundled.manifest.model_dump(),
        }
    finally:
        c.close()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

