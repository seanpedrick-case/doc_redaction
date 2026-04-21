# doc_redaction MCP server

This MCP server runs **locally** (next to your agent) and calls a remote `doc_redaction` deployment (e.g. Hugging Face Space) via the Gradio HTTP API.

## Requirements

- Python 3.10+
- Environment variables:
  - `DOC_REDACTION_BASE_URL` (e.g. `https://seanpedrickcase-document-redaction.hf.space`)
  - `HF_TOKEN` (optional; set if the Space is private/gated)

## Run

If installed as a project script:

```bash
mcp_doc_redaction
```

Or directly:

```bash
python -m mcp_doc_redaction.server
```

## Tools

- `status()` → connectivity + available endpoints from `/gradio_api/info`
- `apply_review_redactions(pdf_bytes, review_csv_bytes, ...)` → calls `/review_apply`
- `summarise_document(pdf_bytes, ...)` → calls `/pdf_summarise`
- `redact_tabular(file_bytes, ...)` → calls `/tabular_redact` and **fails** if not deployed
- `redact_document(file_bytes, ...)` → calls `/redact_document` (may need extension for long-signature deployments)

## Output format

Each tool returns:

- `zip_base64`: a base64-encoded `outputs.zip` containing all downloaded artifacts plus `manifest.json`
- `manifest`: parsed manifest object (same as `manifest.json`)

