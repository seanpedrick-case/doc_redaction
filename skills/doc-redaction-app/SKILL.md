---
name: doc-redaction-app
description: "Agent skill for the Document Redaction app. Online-first: gradio_client examples; discover outputs via client.view_api() / runtime len(result); raw /gradio_api HTTP as fallback; optional /agent."
version: 1.2.3
author: repo-maintained
license: AGPL-3.0-only
changelog:
  - "v1.2.3 (Apr 20, 2026): `handle_file` caveat — server paths from `upload` must be plain strings, not handle_file-wrapped."
  - "v1.2.2 (Apr 20, 2026): Drop stale 41-index table; align with deployments where redact_document exposes ~12 API outputs—view_api + introspection only."
  - "v1.2.1 (Apr 20, 2026): Top-level `handle_file` import; `client.view_api()` wording; tuple-length caveats; DataFrame/bbox guidance; `client.upload` vs `handle_file`."
  - "v1.2.0 (Apr 20, 2026): Concrete gradio_client example; redact_document output indices; OCR JSON shape; /tmp/gradio collection + dedupe; clarify combined_out_message as Textbox; condense raw HTTP fallback; note /gradio_api vs OpenAPI."
  - "v1.1.0 (Apr 20, 2026): Prefer gradio_client over raw HTTP (file= 403 / cookie issues; fragile polling). Note shared-volume output paths; explicit empty session args for redact_document."
  - "v1.0 (Apr 20, 2026): Initial remote-first rewrite (Gradio HTTP API default; optional /agent; proposes future /agent upload/download)."
---

## Overview

This skill enables an LLM to redact a document **end-to-end without human intervention** when the app is deployed remotely (online-only; no shared filesystem). It supports three access surfaces, in **recommended order**:

1. **`gradio_client` (recommended)** — same Gradio app as the UI, but the official client handles **uploads, queueing, and SSE/streaming**; fewer moving parts than hand-written HTTP polling.
2. **Raw Gradio HTTP API** (`/gradio_api/*`) — works when you must use plain HTTP, but **file download and polling are deployment-sensitive** (see below).
3. **FastAPI `/agent` (optional)** — simpler JSON for some tasks, but expects **server-local file paths** unless you only use routes that accept uploads.

## Most efficient route (recommended): `gradio_client`

Use the Python package **`gradio_client`** with a version **compatible with the server’s Gradio** (install from the same major line as the deployment, e.g. Space or `requirements.txt`).

- **Discover**: instantiate `client = Client(...)`, then **`client.view_api()`** (or `GET /gradio_api/info`).
- **Upload**: two patterns work, depending on the endpoint signature: (1) pass **`handle_file("/path/to.pdf")`** inside `predict` kwargs when the file exists **on the client machine** (shown below); (2) call **`client.upload(path)`** and pass the returned **string** (e.g. `/tmp/gradio_tmp/...` on the server) **directly**—**do not** wrap server paths in **`handle_file`**, or you get `ValueError: File does not exist on local filesystem and is not a valid URL`. Same trap when switching from “local redaction” workflows to “already uploaded” review apply (see **doc-redaction-modifications** skill).
- **Call**: `client.predict(..., api_name="/redact_document", **kwargs)` (leading slash matches this app’s `gr.api` / `client.view_api()` names).
- **Outputs**: returned paths are often under `/tmp/gradio/...` on the server. If your test runner **shares a volume** with the app container, you can **read those paths from disk** instead of HTTP-downloading them.

### Prefer short endpoints when available

For programmatic agents, prefer the short `gr.api` routes when present in `client.view_api()` / `/gradio_api/info`:

- `/review_apply` — apply edited review CSV to a PDF
- `/pdf_summarise` — summarise a PDF
- `/tabular_redact` — redact a tabular file (CSV/XLSX/Parquet/DOCX)

### Working example (`gradio_client` + `redact_document`)

Pattern that has been exercised end-to-end (adjust kwargs to match **`client.view_api()`** for your deployment):

```python
import os
# `handle_file` is available at package root in gradio_client (also re-exported from gradio_client.utils).
from gradio_client import Client, handle_file

BASE_URL = os.environ["DOC_REDACTION_BASE_URL"].rstrip("/")
HF = os.environ.get("HF_TOKEN")

client = Client(BASE_URL, hf_token=HF) if HF else Client(BASE_URL)
# Optional alternative: uploaded = client.upload("/path/to/document.pdf") then pass [uploaded] if the API expects server paths.
kwargs = {
    "file_paths": [handle_file("/path/to/document.pdf")],
    "chosen_redact_entities": ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"],
    "chosen_redact_comprehend_entities": [],
    # Fresh run: clear session-carrying fields explicitly
    "ocr_review_files": [],
}
# This app wires `combined_out_message` to a **Gradio Textbox** → use a string (see below).
kwargs["combined_out_message"] = ""

result = client.predict(api_name="/redact_document", **kwargs)
```

If `predict` raises arity/preprocess errors, open `client.view_api()` (or `GET /gradio_api/info`) and supply **every** named parameter, using safe defaults from the docs for that type.

### `combined_out_message` type (this app)

In **`doc_redaction`** the `combined_out_message` input is connected to a **`gr.Textbox`**, not a list component. Use an **empty string** `""` for a clean run. If you fork the UI or change the component, re-check **`/gradio_api/info`** for that parameter’s type and send `[]`, `null`, or `""` accordingly.

### `redact_document` return value (discover; do not hard-code indices)

`client.predict` returns **one Python value per API output slot**, in the order shown by **`client.view_api()`** for `/redact_document` (and the same order in **`GET /gradio_api/info`**).

**Authoritative contract:** the **output count and types** come from **`client.view_api()`** and a probe call **`len(result)`**—not from counting `outputs=[...]` lines in `app.py`. Many live deployments (including current field tests) expose **about twelve** values for this endpoint, even though the UI handler in source lists more components. Gradio may surface a reduced or serialized subset over the client API; treat **~12 outputs** as normal unless your `view_api()` says otherwise.

**Never hard-code index 12, 20, 40, etc.** from old docs. Instead:

1. Read **`client.view_api()`** and note each **returned** parameter name and type for `/redact_document`.
2. After `result = client.predict(...)`, assert **`len(result)`** matches that output list.
3. Locate artifacts by **shape**, not magic indices:
   - **Summary / status** — usually an early **`str`**.
   - **Paths** — **`str`** ending in `.pdf` / `.csv` / `.json`, or **`list`** of such strings.
   - **Tabular review / bbox data** — **`dict`** with **`headers`** and **`data`** (Gradio **Dataframe** serialization). There may be **more than one** such dict (e.g. line OCR vs decision table); pick the one whose **headers** match review/bbox columns, or use the file paths in `result` and read **`*_review_file.csv`** on disk.
   - **Rich OCR** — **`list`** of per-page dicts when exposed; otherwise rely on **`*_ocr_results_with_words_*.json`** under **`/tmp/gradio`** or your download step.

**Bounding boxes:** encoded as **rows** inside those **`headers`/`data`** tables and/or in **`*_ocr_results_with_words_*.json`** (`words`, `bounding_box` per line). If a field test saw bbox-like data at **`result[6]`**, that only applies when **`len(result) > 6`** and that slot is a dataframe dict—**verify on your deployment**.

### Collecting artifacts under `/tmp/gradio` (shared volume)

On many Docker / self-hosted setups, Gradio writes under **`/tmp/gradio/<session-or-hash>/...`**. If your agent mounts that tree, you can **copy outputs from disk** instead of `gradio_api/file=`.

When several subfolders contain the same logical artifact name, **dedupe by file content hash** after collection:

```python
import hashlib
from pathlib import Path

def paths_by_content_hash(root: Path) -> dict[str, Path]:
    """First path wins for identical file bytes (walks all files under root)."""
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        if p.is_file():
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            out.setdefault(digest, p)
    return out
```

### Word-level OCR JSON (`*_ocr_results_with_words_*.json`)

Saved as a **JSON list** of per-page records. Each page is shaped like `{"page": "<n>", "results": {"text_line_<k>": {"line": int, "text": str, "bounding_box": [x0,y0,x1,y1], "conf": float, "words": [...]}}}`. Agents can walk `results` values for line-level boxes and per-word entries inside `words`.

### `redact_document` and remaining session-heavy arguments

`redact_document` wraps `choose_and_run_redactor` with **dozens** of inputs. Besides `combined_out_message` and `ocr_review_files`, fill any other session fields **`client.view_api()`** marks as required—often explicit **`[]`**, **`""`**, or **`0`** is safest for a cold start.

### Simpler `gr.api` endpoints (fewer inputs)

When they fit the task, prefer the dedicated wrappers (short `data[]`, less brittle):

- `review_apply` — PDF + `*_review_file.csv`.
- `pdf_summarise` — summarise-only path.
- `tabular_redact` — tabular files (CSV/XLSX/Parquet/DOCX) in one call.

## Alternative: raw Gradio HTTP API (fallback only)

Use when you cannot run `gradio_client` or must call from a non-Python stack.

**Stable contract:** `GET {BASE_URL}/gradio_api/info` (and upload/call/poll/download under the **`/gradio_api/...`** prefix). Gradio’s generated **OpenAPI** may also list these routes; names like `/run/...` in third-party docs are **not** a substitute for your server’s **`/gradio_api/info`**.

Minimal flow: **info → POST `/gradio_api/upload` (`files`) → POST `/gradio_api/call/{api_name}` (`{"data":[...]}`) → GET `/gradio_api/call/{api_name}/{event_id}` until done → GET `/gradio_api/file={path}`**.

**Caveats:** **`file=` may 403** without browser/session cookies; **polling** is version-sensitive. Prefer **`gradio_client`** or **read `/tmp/gradio/...`** on a shared volume.

## Decision tree

- **Use `gradio_client`** if you can run Python against the deployment URL (default recommendation).
- **Use raw Gradio HTTP API** only when necessary; plan for auth/cookie behaviour on `file=` and verify polling for your Gradio version.
- **Use browser automation** only if HTTP API calls are blocked by auth/network policy but the UI is reachable.
- **Use `/agent`** only if the agent can reference **server-visible file paths** (shared filesystem) or there is an additional upload mechanism that writes into the app’s allowed roots.

## Authentication (deployment-dependent)

Apply what your deployment requires:

- **Hugging Face Spaces** (private/gated): `Authorization: Bearer <HF_TOKEN>` on HTTP calls; with **`gradio_client`**, pass `hf_token=...` (or `headers=...`) per the client docs for your Gradio version.
- **FastAPI `/agent`** (if configured): `X-Agent-API-Key: <AGENT_API_KEY>`
- **Reverse proxy / SSO**: cookies/headers per environment

## Outputs (what to download)

The agent should download and return all produced artifacts, typically:

- `*_redacted.pdf`
- `*_redactions_for_review.pdf`
- `*_review_file.csv`
- `*_ocr_output_*.csv`
- `*_ocr_results_with_words_*.csv` and/or `.json`

Then package them:

- `manifest.json` (filenames, sizes, sha256 hashes, run metadata)
- `outputs.zip` containing everything

## Raw HTTP checklist (non-Python / legacy integrations)

1. `GET /gradio_api/info` → pick `api_name`, build `data[]` length and order.
2. `POST /gradio_api/upload` (`multipart` field **`files`**) → internal path(s).
3. `POST /gradio_api/call/{api_name}` with `{"data":[...]}` → `event_id`.
4. Poll `GET /gradio_api/call/{api_name}/{event_id}` until success payload.
5. Collect paths from the payload; download with `GET /gradio_api/file={path}` **or** read from disk if you share `/tmp/gradio`.

For Python, implement the above with **`httpx`** only if `gradio_client` is unavailable; do not assume alternate path prefixes without checking your server’s OpenAPI or `/gradio_api/info`.

## Optional: FastAPI `/agent` surface (shared filesystem only)

When FastAPI is enabled in this app, it mounts a router under **`/agent`**.

### Discover supported operations

- `GET {BASE_URL}/agent/operations`

### Key constraint (important)

Current `/agent/*` endpoints expect **server-local file paths** (validated to be under the repo root, `INPUT_FOLDER`, or `OUTPUT_FOLDER`). This means `/agent` is not online-only capable by itself.

## Proposed improvement (future work): make `/agent` online-only capable

Add two minimal endpoints (strict auth + allowlisted roots + size limits):

- `POST /agent/files/upload` → store under `INPUT_FOLDER/uploads/<job_id>/...` and return the safe server path.
- `GET /agent/jobs/{job_id}/artifacts.zip` → stream a single zip containing all outputs for the job.
