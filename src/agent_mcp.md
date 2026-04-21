## MCP (Model Context Protocol) option for doc_redaction

This app already exposes a usable remote surface via Gradio (`/gradio_api/info`, `/gradio_api/upload`, `/gradio_api/call/*`, `/gradio_api/file=*`). However, external LLM agents often struggle with:

- long, session-heavy endpoint signatures
- async queue/poll mechanics
- auth/cookie behavior (especially for `file=` downloads)
- container-only output paths

An MCP server can wrap these details behind a **small, stable tool interface**.

### Recommended MCP tool surface

Design the MCP server as a thin client for a deployment URL (e.g. HF Space), with tools that accept **bytes** and return a single **normalized bundle**.

All tools should return:
- `artifacts_zip_bytes` (or a download URL) containing outputs
- `manifest_json` describing files + sha256 hashes + key metadata

#### 1) `redact_document`

**Input**
- `pdf_or_image_bytes` (PDF/JPG/PNG)
- `entities` (list of labels)
- `options` (object): OCR method, PII method, allow/deny lists, page range, output knobs

**Implementation notes**
- Prefer `/doc_redact` when present (short signature)
- If `/doc_redact` is unavailable, fallback options are deployment-specific (many `/redact_document` routes are UI-chained and not reliably callable with a short payload)
- Always handle upload → call → poll → download internally

#### 2) `apply_review_redactions`

**Input**
- `pdf_bytes`
- `review_csv_bytes` (must be `*_review_file.csv` by app convention)
- `output_dir` (optional)

**Implementation notes**
- Prefer `/review_apply` when present (short signature)

#### 3) `redact_tabular`

**Input**
- `tabular_bytes` (CSV/XLSX/Parquet/DOCX)
- `entities`
- `options` (columns, method, allow/deny, anon strategy, etc.)

**Implementation notes**
- Prefer `/tabular_redact` when present (short signature)

#### 4) (Optional) `summarise_document`

**Input**
- `pdf_bytes`
- `options` (OCR method, format, context, max pages per group, etc.)

**Implementation notes**
- Prefer `/pdf_summarise`

### Authentication handling

The MCP server should accept:
- `base_url`
- `headers` (e.g. HF `Authorization: Bearer ...`)

and apply them consistently to every call. If `file=` requires cookies/session, the MCP server should prefer:
- retrieving outputs from the completed Gradio response payload and downloading immediately with the same client session, or
- reading outputs from a shared mount when the MCP server runs co-located with the app.

### Why MCP helps here

- Agents call **one tool** and receive a **single deterministic bundle**
- You can upgrade Gradio or adjust UI wiring without breaking agent callers
- Removes repeated “how do I call this endpoint?” guidance from skills/README

