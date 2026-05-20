---
name: doc-redaction-app
description: "Initial document redaction and downloading server outputs: gradio_client, `/doc_redact` first, path/download traps, and `/redact_document` when needed. Not for CSV review or reapply (see doc-redaction-modifications); parallel multi-page review orchestration → doc-redact-page-review."
version: 2.1.1
author: repo-maintained
license: AGPL-3.0-only
---

## Scope

This skill covers **running an initial redaction** and **getting artifacts onto the client**. For edited review CSVs, `/review_apply`, coordinate fixes, and visual QA, use [`../doc-redaction-modifications/SKILL.md`](../doc-redaction-modifications/SKILL.md). For **parallel per-page review** (spawn subagents, merge CSV, one `/review_apply`), use [`../doc-redact-page-review/SKILL.md`](../doc-redact-page-review/SKILL.md).

## Quick start

### 1) Access path

- Primary: `gradio_client`
- Fallback: raw `/gradio_api/*` HTTP
- `/agent/*` only when inputs resolve on the **server** (repo root, `INPUT_FOLDER`, or `OUTPUT_FOLDER`). If you see path validation errors, the client and app do not share a filesystem—use Gradio upload paths or MCP, not bare agent paths.

### 2) Prefer `/doc_redact` over `/redact_document`

Use `/doc_redact` for a normal PDF/image first pass. Use `/redact_document` only when you need the full Gradio control surface. Note that the `/doc_redact` endpoint may not be visible when calling `client.view_api()`.

### 2b) `/doc_redact` can succeed with no artifacts

Some deployments return a success message but **`[]`** for output paths. Treat that as **no deliverable** for automation.

- **Custom VLM entity types** (e.g. `CUSTOM_VLM_SIGNATURE`, `CUSTOM_VLM_FACES`) are a common trigger: processing may complete but paths stay empty. Fall back to `/redact_document` (or deployment-specific docs) if you need those entities.

### 2c) `/doc_redact` vs `/redact_document` parameter names

They are **not** interchangeable. Wrong kwargs raise errors such as `Parameter is not a valid keyword argument`.

- `/doc_redact`: e.g. `document_file`, `ocr_method`, `pii_method`, `redact_entities` (not `file_paths`, `chosen_local_ocr_model`, `chosen_redact_entities`).
- `/redact_document`: long-form names (`file_paths`, `chosen_redact_entities`, `chosen_local_ocr_model`, `pii_identification_method`, etc.). Use `client.view_api()` when in doubt.

### 2d) OCR / PII labels on `/doc_redact`

- `ocr_method`: high-level modes (`Local OCR`, `AWS Textract`, `Local text`) plus engine shortcuts (`tesseract`, `paddle`, `hybrid-paddle-vlm`, `vlm`, `inference-server`, …). Aliases like `textract`, `local` often work.
- `pii_method`: e.g. `Local`, `AWS Comprehend`, `LLM (AWS Bedrock)`, `Local inference server`, `Local transformers LLM`, `None`. Prefer exact labels from the deployment.

Optional **page window** (when exposed): `page_min` / `page_max` (1-based; `0` often means first/last page—confirm in `view_api()`).

### 3) `handle_file` (critical)

- **Local** file on the machine running the client: `handle_file("/local/path/file.pdf")`
- **Server** path (e.g. after Gradio upload): plain string—**do not** wrap in `handle_file(...)`

### 4) Downloads after `predict` (critical)

- Return values are **server paths** (and strings); nothing is written locally unless you fetch bytes.
- URL: `{BASE_URL}/gradio_api/file={urllib.parse.quote(path, safe="")}`. Always encode the path; spaces and special characters break naive URLs.
- Gated HF Spaces: send **`Authorization: Bearer <HF_TOKEN>`** on download requests as well as on the client.
- Paths may be strings or nested dicts with a `"path"` key; walk recursively if needed (see `extract_file_like_paths` in `mcp_doc_redaction/gradio_transport.py`).

### 5) `/redact_document` gotchas (initial run)

- `output_folder` must be **non-empty**. On many HF-style deployments, use the app’s real output directory (often something like `/home/user/app/output`) so returned paths are downloadable; paths under `/tmp/gradio/...` may still appear and can return **403** on `gradio_api/file=`—if downloads fail, check returned path prefixes and deployment README.
- `chosen_llm_entities` must contain **at least one** value even in Local PII mode (e.g. `["PERSON"]` or `["PERSON_NAME"]`).
- **`combined_out_message=""`** and **`ocr_review_files=[]`** are required in typical apps; omitting them raises `TypeError: No value provided for required argument`.
- Heavy **VLM / signature** jobs can run many minutes per page; use generous **`httpx.Timeout`** (e.g. `read=1800` or higher) to avoid false timeouts.
- When picking files by name, **`…_redacted.pdf`** is the redacted artifact; **`…_redactions_for_review.pdf`** is an overlay preview, not the final black-box PDF.

### 6) Client and network

- **`Client(BASE_URL)`** can hang on cold API info fetch; always pass **`httpx_kwargs`** with a **connect** timeout (e.g. 120s) and a long **read** timeout for big jobs.
- From **Docker** to a Gradio app on the host, use `http://host.docker.internal:<port>` instead of `localhost` on the client container.

## Known scripting limitations

- Prefer small Python scripts over fragile one-liners for anything touching CSVs or paths.
- CSV: quote colour tuples as strings (e.g. `"(0, 0, 0)"`); use `encoding="utf-8-sig"` when editing (BOM).
- PowerShell: `&&` is not a line separator; use `;` or separate commands.

## Full reference

### `gradio_client` pattern for `/doc_redact`

Use **`document_file`** (not `pdf_file`). Example download loop after `predict`:

```python
import os
from pathlib import Path
from urllib.parse import quote

import httpx
from gradio_client import Client, handle_file

BASE_URL = os.environ["DOC_REDACTION_BASE_URL"].rstrip("/")
HF_TOKEN = os.environ.get("HF_TOKEN")
httpx_kwargs = {
    "timeout": httpx.Timeout(connect=120.0, read=1800.0, write=120.0, pool=120.0),
}
client = (
    Client(BASE_URL, hf_token=HF_TOKEN, httpx_kwargs=httpx_kwargs)
    if HF_TOKEN
    else Client(BASE_URL, httpx_kwargs=httpx_kwargs)
)

result = client.predict(
    api_name="/doc_redact",
    document_file=handle_file("/local/path/document.pdf"),
)

headers = {}
if HF_TOKEN:
    headers["Authorization"] = f"Bearer {HF_TOKEN.strip()}"
out_dir = Path("output/run_001")
out_dir.mkdir(parents=True, exist_ok=True)
with httpx.Client(timeout=httpx_kwargs["timeout"], headers=headers) as http:
    for p in result[0]:
        if not isinstance(p, str) or not p.startswith("/"):
            continue
        url = f"{BASE_URL}/gradio_api/file={quote(p, safe='')}"
        dest = out_dir / Path(p).name
        dest.write_bytes(http.get(url).raise_for_status().content)
```

### Minimal `/redact_document` cold start

Expand with **`client.view_api()`** for your deployment; typical extra fields include `text_extraction_method`, `pii_identification_method`, `review_file_state`, etc.

```python
kwargs = {
    "file_paths": [handle_file("/local/path/document.pdf")],
    "chosen_redact_entities": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
    "chosen_redact_comprehend_entities": [],
    "chosen_llm_entities": ["PERSON"],
    "ocr_review_files": [],
    "combined_out_message": "",
    "output_folder": "/home/user/app/output",  # non-empty; adjust to deployment
}
result = client.predict(api_name="/redact_document", **kwargs)
```

### Pragmatic handling of `/redact_document` return shape

Returns are often a long tuple; indices can shift between versions. Prefer:

1. Collect strings ending in `.pdf`, `.csv`, `.json` (including nested lists).
2. Log human-readable status strings.
3. Re-run `view_api()` after upgrades if mapping breaks.

### Raw HTTP fallback

1. `GET /gradio_api/info`
2. Upload (if supported) → `POST /gradio_api/call/{api_name}` with `{"data":[...]}`
3. Poll `GET /gradio_api/call/{api_name}/{event_id}`
4. Download with encoded `gradio_api/file=` (and Bearer token if gated)

### MCP vs scripts

Use **`mcp_doc_redaction`** when the agent already has MCP tools and bundled zips help. For standalone automation, `gradio_client` is usually simpler.

### Auth

- HF gated/private: `HF_TOKEN` / `Authorization: Bearer …`
- Agent API: `X-Agent-API-Key` when configured

### Typical first-run artifacts

- `*_redacted.pdf`, `*_redactions_for_review.pdf`, `*_review_file.csv`
- OCR exports: `*_ocr_output_*.csv`, `*_ocr_results_with_words_*.{csv,json}`

Package downloads with a small manifest (name, size, hash, time) when automating.

Repo-wide API summary: [AGENTS.md](../../AGENTS.md).
