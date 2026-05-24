---
name: doc-redaction-app
description: "Initial document redaction and downloading server outputs: gradio_client, `/doc_redact` first, path/download traps, and `/redact_document` when needed. Not for CSV review or reapply (see doc-redaction-modifications); parallel multi-page review orchestration → doc-redact-page-review."
version: 2.3.1
author: repo-maintained
license: AGPL-3.0-only
---

## Scope

This skill covers **running an initial redaction** and **getting artifacts onto the client**. For review after a run: **Pass 1** (OCR/CSV edits, `/review_apply`) and optional **Pass 2** (visual VLM) — [`../doc-redaction-modifications/SKILL.md`](../doc-redaction-modifications/SKILL.md). For **parallel Pass 1 page review** (subagents, merge, one apply), use [`../doc-redact-page-review/SKILL.md`](../doc-redact-page-review/SKILL.md).

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

- `/doc_redact`: e.g. `document_file`, `ocr_method`, `pii_method`, `redact_entities`, `handwrite_signature_checkbox` (not `file_paths`, `chosen_local_ocr_model`, `chosen_redact_entities`, `custom_llm_instructions`).
- `/redact_document`: long-form names (`file_paths`, `chosen_redact_entities`, `chosen_local_ocr_model`, `pii_identification_method`, etc.). Use `client.view_api()` when in doubt.

### 2d) OCR / PII labels on `/doc_redact`

- `ocr_method`: high-level modes (`Local OCR`, `AWS Textract`, `Local text`) plus engine shortcuts (`tesseract`, `paddle`, `hybrid-paddle-vlm`, `vlm`, `inference-server`, …). Aliases like `textract`, `local` often work.
- `pii_method`: e.g. `Local`, `AWS Comprehend`, `LLM (AWS Bedrock)`, `Local inference server`, `Local transformers LLM`, `None`. Prefer exact labels from the deployment.

Optional **page window** (when exposed): `page_min` / `page_max` (1-based; `0` often means first/last page—confirm in `view_api()`).

**AWS Textract extraction** (`handwrite_signature_checkbox`): multiselect list passed when using `AWS Textract` (or hybrid Textract routes). Values must match the deployment UI labels exactly.

| Value | Always available? | Purpose |
|-------|-------------------|---------|
| `Extract handwriting` | Yes (base default) | Handwritten text via Textract |
| `Extract signatures` | Yes (base default) | Signature blocks via Textract |
| `Extract forms` | Only if deployment enables it | Key-value / form fields |
| `Extract layout` | Only if deployment enables it | Document layout structure |
| `Extract tables` | Only if deployment enables it | Table detection |
| `Face detection` | Only if deployment enables it | Face regions (often used with VLM routes) |

Base deployments expose only the first two (`HANDWRITE_SIGNATURE_TEXTBOX_FULL_OPTIONS` default). Extra rows appear when the server config sets `INCLUDE_FORM_EXTRACTION_TEXTRACT_OPTION`, `INCLUDE_LAYOUT_EXTRACTION_TEXTRACT_OPTION`, `INCLUDE_TABLE_EXTRACTION_TEXTRACT_OPTION`, or `INCLUDE_FACE_IDENTIFICATION_TEXTRACT_OPTION` to `True`. Confirm with the UI “AWS Textract extraction settings” checkbox group or `GET /gradio_api/info`.

Example:

```python
client.predict(
    api_name="/doc_redact",
    document_file=handle_file("/local/path/document.pdf"),
    ocr_method="AWS Textract",
    handwrite_signature_checkbox=["Extract handwriting", "Extract signatures"],
)
```

When omitted, server/CLI defaults apply (`DEFAULT_HANDWRITE_SIGNATURE_CHECKBOX`, often `[]`).

### 2e) PII entity lists — Local vs AWS Comprehend

**Which parameter to use depends on `pii_method`:**

| `pii_method` | `/doc_redact` | `/redact_document` |
|--------------|---------------|-------------------|
| `Local` (spaCy/Presidio) | `redact_entities` | `chosen_redact_entities` |
| `AWS Comprehend` | **Not exposed** — use `/redact_document` or CLI | `chosen_redact_comprehend_entities` |

`/doc_redact` maps `redact_entities` → CLI `local_redact_entities` only. For Comprehend entity selection, call `/redact_document` (or `cli_redact.py --aws_redact_entities …`).

**Local PII entities** (`FULL_ENTITY_LIST` in `tools/config.py`; deployment may extend via env):

`TITLES`, `PERSON`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `STREETNAME`, `UKPOSTCODE`, `CREDIT_CARD`, `CRYPTO`, `DATE_TIME`, `IBAN_CODE`, `IP_ADDRESS`, `NRP`, `LOCATION`, `MEDICAL_LICENSE`, `URL`, `UK_NHS`, `CUSTOM`, `CUSTOM_FUZZY`

When VLM face/signature detection is enabled on the deployment, the list may also include: `CUSTOM_VLM_FACES`, `CUSTOM_VLM_SIGNATURE`.

**Textract + `CUSTOM_VLM_SIGNATURE`:** when `ocr_method` is `AWS Textract` and `handwrite_signature_checkbox` includes `Extract signatures`, Textract signature analysis runs and **inline/post-pass `CUSTOM_VLM_SIGNATURE` VLM detection is skipped** (no duplicate signature finding). Keep `CUSTOM_VLM_SIGNATURE` in the entity list if you want those regions redacted; Textract supplies the boxes. Without `Extract signatures`, `CUSTOM_VLM_SIGNATURE` still uses VLM as before.

Default selection (`CHOSEN_REDACT_ENTITIES`): `TITLES`, `PERSON`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `STREETNAME`, `UKPOSTCODE`, `CUSTOM`.

**AWS Comprehend entities** (`FULL_COMPREHEND_ENTITY_LIST`):

`BANK_ACCOUNT_NUMBER`, `BANK_ROUTING`, `CREDIT_DEBIT_NUMBER`, `CREDIT_DEBIT_CVV`, `CREDIT_DEBIT_EXPIRY`, `PIN`, `EMAIL`, `ADDRESS`, `NAME`, `PHONE`, `SSN`, `DATE_TIME`, `PASSPORT_NUMBER`, `DRIVER_ID`, `URL`, `AGE`, `USERNAME`, `PASSWORD`, `AWS_ACCESS_KEY`, `AWS_SECRET_KEY`, `IP_ADDRESS`, `MAC_ADDRESS`, `LICENSE_PLATE`, `VEHICLE_IDENTIFICATION_NUMBER`, `UK_NATIONAL_INSURANCE_NUMBER`, `INTERNATIONAL_BANK_ACCOUNT_NUMBER`, `SWIFT_CODE`, `UK_NATIONAL_HEALTH_SERVICE_NUMBER`, `ALL`, `CUSTOM`, `CUSTOM_FUZZY`

Default selection (`CHOSEN_COMPREHEND_ENTITIES`): `EMAIL`, `ADDRESS`, `NAME`, `PHONE`, `PASSPORT_NUMBER`, `UK_NATIONAL_INSURANCE_NUMBER`, `UK_NATIONAL_HEALTH_SERVICE_NUMBER`, `CUSTOM`.

**Do not mix label namespaces:** Local uses `PERSON` / `EMAIL_ADDRESS`; Comprehend uses `NAME` / `EMAIL`. Pick the list that matches your `pii_method`.

#### Custom Presidio recognizers (Local **and** Comprehend)

These are **not** generic spaCy NER labels — they are app-specific Presidio recognizers (`tools/load_spacy_model_custom_recognisers.py`). They appear in **both** Local and Comprehend entity dropdowns because `CUSTOM_ENTITIES` is merged into the Comprehend lists (Comprehend can miss titles, UK postcodes, and street fragments).

| Entity | What it detects |
|--------|-----------------|
| `TITLES` | Honorifics/titles: Mr, Mrs, Ms, Miss, Dr, Professor, Sir, … |
| `UKPOSTCODE` | UK postcode patterns (e.g. `SW1A 1AA`, `GIR 0AA`) |
| `STREETNAME` | Street/road address fragments (number + street-type suffix, e.g. `… Road`, `… Lane`) |

Default deployment config (`CUSTOM_ENTITIES`): `TITLES`, `UKPOSTCODE`, `STREETNAME`. Admins may extend via the `CUSTOM_ENTITIES` env var.

On **`/doc_redact`**: include in `redact_entities`. On **`/redact_document`**: include in `chosen_redact_entities` (Local) **or** `chosen_redact_comprehend_entities` (Comprehend — these three labels are valid there too).

#### `CUSTOM` and `CUSTOM_FUZZY` (deny-list driven)

Both routes support explicit term lists via **`deny_list`** / **`allow_list`** on `/doc_redact`, or deny/allow list files on `/redact_document` / CLI.

| Entity | Matching | Typical use |
|--------|----------|-------------|
| `CUSTOM` | Exact (literal or regex in deny list) | Always redact specific names, orgs, or regex patterns |
| `CUSTOM_FUZZY` | Fuzzy deny-list match (spelling variants) | Same, tolerating typos; auto-included when CLI `fuzzy_mistakes > 0` |

**`/doc_redact` inline lists:** pass terms directly as `deny_list=["term1", "term2"]` (not a file path). When `deny_list` is non-empty, the API **auto-appends `CUSTOM`** to `redact_entities` if missing (deny-list matching requires that entity type). If you omit `redact_entities` entirely, CLI defaults already include `CUSTOM`.

For fuzzy deny-list matching via API, include `CUSTOM_FUZZY` in `redact_entities` explicitly (or use CLI/`/redact_document` with `fuzzy_mistakes > 0`, which auto-adds `CUSTOM_FUZZY`).

`allow_list` excludes terms from redaction even when another entity type would match.

Example — Local `/doc_redact` with UK-specific recognizers + custom deny terms:

```python
client.predict(
    api_name="/doc_redact",
    document_file=handle_file("/local/path/document.pdf"),
    pii_method="Local",
    redact_entities=[
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
        "TITLES", "STREETNAME", "UKPOSTCODE",
        "CUSTOM", "CUSTOM_FUZZY",
    ],
    deny_list=["Acme Corp", "Cora Fyller"],
)
```

Example — Comprehend `/redact_document` with Comprehend labels **plus** custom recognizers:

```python
client.predict(
    api_name="/redact_document",
    file_paths=[handle_file("/local/path/document.pdf")],
    pii_identification_method="AWS Comprehend",
    chosen_redact_entities=[],
    chosen_redact_comprehend_entities=[
        "NAME", "EMAIL", "PHONE", "ADDRESS",
        "TITLES", "UKPOSTCODE", "STREETNAME", "CUSTOM",
    ],
    chosen_llm_entities=["PERSON_NAME"],
    ocr_review_files=[],
    combined_out_message="",
    output_folder="/home/user/app/output",
)
```

Example — Local on `/doc_redact`:

```python
client.predict(
    api_name="/doc_redact",
    document_file=handle_file("/local/path/document.pdf"),
    pii_method="Local",
    redact_entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "STREETNAME", "UKPOSTCODE", "TITLES"],
)
```

Example — Comprehend on `/redact_document`:

```python
client.predict(
    api_name="/redact_document",
    file_paths=[handle_file("/local/path/document.pdf")],
    pii_identification_method="AWS Comprehend",
    chosen_redact_entities=[],  # unused when Comprehend is selected
    chosen_redact_comprehend_entities=["NAME", "EMAIL", "PHONE", "ADDRESS"],
    chosen_llm_entities=["PERSON_NAME"],
    ocr_review_files=[],
    combined_out_message="",
    output_folder="/home/user/app/output",
)
```

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
    "pii_identification_method": "Local",
    "chosen_redact_entities": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "STREETNAME", "UKPOSTCODE", "TITLES", "CUSTOM"],
    "chosen_redact_comprehend_entities": [],
    "chosen_llm_entities": ["PERSON_NAME"],
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
