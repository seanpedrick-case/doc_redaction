---
name: doc-redaction-app
description: "Operate the Document Redaction app with a practical default workflow: short Gradio endpoints via gradio_client, explicit handle_file rules, known failure traps, and output verification before sign-off."
version: 2.0.4
author: repo-maintained
license: AGPL-3.0-only
---


## Quick start (read this first)

### 1) Pick the right access path

- Primary: `gradio_client`
- Fallback: raw `/gradio_api/*` HTTP
- Use `/agent/*` only when you have server-local paths (shared filesystem)
- Browser UI only if APIs are blocked

### 2) Prefer short endpoints over `/redact_document`

Use these first:
- `/doc_redact` for PDF/image redaction
- `/review_apply` for applying edited review CSV
- `/preview_boxes` for rendering proposed CSV boxes onto the original PDF **without** applying redactions — use before `/review_apply` to verify coordinates (returns a ZIP of PNGs)
- `/pdf_summarise` for PDF summarization
- `/tabular_redact` for tabular files

Use `/redact_document` only when you need the full control surface.

### 2b) Important: `/doc_redact` can “succeed” but return no artifacts

Some deployments may return a success message (e.g. `"doc_redact completed"`) but **an empty output paths list** (`[]`).

- Treat **empty paths** as a failure for automation (there is nothing to download).
- Recommended fallback: immediately call `/redact_document` (or use raw `/gradio_api/*` HTTP) and continue from there.

### 2a) `/doc_redact` parameter values (important for agents)

`/doc_redact` accepts a simplified `ocr_method` input that maps to two CLI knobs:
- High-level OCR/text modes: `Local OCR`, `AWS Textract`, `Local text`
- Local OCR engine shortcuts (auto-mapped to `Local OCR` + `chosen_local_ocr_model`):
  - `tesseract`, `paddle`, `hybrid-paddle`, `hybrid-vlm`, `hybrid-paddle-vlm`, `hybrid-paddle-inference-server`
  - `vlm`, `inference-server`, `bedrock-vlm`, `gemini-vlm`, `azure-openai-vlm`
- Common aliases are accepted (for example `textract`, `local`, `simple text`, `hybrid paddle vlm`).

`/doc_redact` `pii_method` accepts configured labels and common aliases:
- `Local`
- `AWS Comprehend`
- `LLM (AWS Bedrock)`
- `Local inference server`
- `Local transformers LLM`
- `None` (plus aliases like `no redaction`)

Use exact configured labels where possible for maximum portability across deployments.

### 3) `handle_file` rule (critical)

- Local client file path: use `handle_file("/local/path/file.pdf")`
- Server path returned from upload (for example `/tmp/gradio_tmp/...`): pass as plain string
- Do not wrap server paths in `handle_file(...)`

### 3b) Local downloads after `predict` (critical)

- `client.predict` returns **server-side paths** (and status strings). It does **not** write files to your machine; fetch bytes yourself with HTTP unless you use MCP (bundled zip) or a shared filesystem.
- Endpoint: `{BASE_URL}/gradio_api/file={encoded_path}` where **`encoded_path` = `urllib.parse.quote(path, safe="")`**. Omitting encoding breaks when paths contain spaces or reserved characters.
- For gated/private HF Spaces, use the same **`Authorization: Bearer <HF_TOKEN>`** header on download requests as for the client.
- Paths may appear as plain strings or nested dicts with a `"path"` key; use recursive extraction (see full reference example or `extract_file_like_paths` in `mcp_doc_redaction/gradio_transport.py`).

### 4) Two high-impact gotchas

- For full `/redact_document`: `output_folder` must be non-empty.
- For full `/redact_document`: `chosen_llm_entities` must contain at least one value even when using Local PII mode.

### 5) `review_apply` image-name trap (critical)

When using `/review_apply`, do not submit review rows whose `image` values are fake placeholders (for example `placeholder_image_2.png`) unless they are valid for the active run context. In field failures this stripped redaction rows during apply. Keep `image` values aligned with current run artifacts.

## Known tool limitations (prominent)

Apply these constraints before writing scripts:

- Use full Python scripts instead of fragile one-liners for CSV editing and bbox generation.
- Avoid patch patterns that collapse line breaks in generated Python; verify written scripts before execution.
- Quote CSV color tuples as strings (for example `"(0, 0, 0)"`) to avoid comma-splitting issues.
- Use Python 3 explicitly.
- CSV files may have UTF-8 BOM; read/write with `encoding="utf-8-sig"` when editing.
- PowerShell note (Windows): `&&` is not a statement separator; use `;` or separate commands.

## Verification workflow (required before sign-off)

After every redaction/apply run:

1. Generate output artifacts (`*_redacted.pdf`, `*_review_file.csv`).
2. Render each PDF page to image with PyMuPDF.
3. Draw review CSV boxes on page images.
4. Review review images with a human or vision model for:
   - misses (sensitive text visible)
   - false positives (non-sensitive text boxed)
   - box drift (misaligned geometry)
5. Fix CSV page-by-page and re-apply using `/review_apply`.

### Multiple apply runs — always sort outputs by modification time

Each `/review_apply` call generates a **new hash-prefixed filename** (e.g. `a3f9..._ redacted.pdf`). After several iterations you will have multiple versions in the output folder. Scripts that glob for output files **must** sort by `st_mtime` descending and take the first result, or they will silently verify an older file:

```python
from pathlib import Path

candidates = sorted(
    Path("output_final").glob("*_redacted.pdf"),
    key=lambda f: f.stat().st_mtime,
    reverse=True,
)
latest = candidates[0]  # always the most recently applied version
```

Apply the same pattern to `*_review_file.csv` and `*_redactions_for_review.pdf`.

### Pre-apply coordinate preview (saves round-trips)

Before sending a modified CSV to `/review_apply`, render the proposed boxes on the **original PDF locally**. This lets you confirm geometry without a server round-trip and is especially valuable for manually placed boxes (signatures, decorative text, stamps).

Add a percentage grid to measure where boxes land relative to page content:

```python
import csv, fitz
from pathlib import Path
from PIL import Image, ImageDraw

PDF = Path("input/document.pdf")
CSV = Path("output/document_review_file_edited.csv")
OUT = Path("output/preview"); OUT.mkdir(exist_ok=True)

with CSV.open(encoding="utf-8-sig") as f:
    rows = {}
    for r in csv.DictReader(f):
        rows.setdefault(int(float(r.get("page", 0) or 0)), []).append(r)

doc = fitz.open(str(PDF))
for p in range(1, doc.page_count + 1):
    pix = doc[p - 1].get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img)
    for pct in range(0, 100, 5):           # percentage grid
        y = int(pct / 100 * pix.height)
        draw.line([(0, y), (60, y)], fill="red", width=1)
        draw.text((2, y), f"{pct}%", fill="red")
    for r in rows.get(p, []):
        draw.rectangle(
            [float(r["xmin"]) * pix.width, float(r["ymin"]) * pix.height,
             float(r["xmax"]) * pix.width, float(r["ymax"]) * pix.height],
            outline="orange", width=3,
        )
    img.save(OUT / f"page_{p:03d}.png")
```

Even a perfect local preview does not guarantee pixel-perfect alignment in the final redacted PDF — the server applies boxes in the original PDF coordinate space, which can render slightly differently to a local PyMuPDF render. Always verify the actual applied output after each apply run.



### Scanned-page warning (don’t rely on PDF text search)

Many PDFs contain scanned/image-like pages with **no reliable selectable text**.

- Do not rely on PDF text-search (e.g. PyMuPDF `search_for`) to find terms on those pages; it can silently miss.
- Prefer **OCR word outputs** (e.g. `*_ocr_results_with_words_*.csv`) to locate terms and build boxes.

Minimal review image generation script:

```python
import csv
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

PDF_PATH = Path("output/document_redacted.pdf")
REVIEW_CSV = Path("output/document.pdf_review_file.csv")
OUT_DIR = Path("output/review_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with REVIEW_CSV.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

rows_by_page = {}
for r in rows:
    p = int(float(r.get("page", "0") or 0))
    rows_by_page.setdefault(p, []).append(r)

doc = fitz.open(PDF_PATH)
for p in range(1, doc.page_count + 1):
    pix = doc[p - 1].get_pixmap(dpi=180)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img)
    for r in rows_by_page.get(p, []):
        x0 = float(r["xmin"]) * pix.width
        y0 = float(r["ymin"]) * pix.height
        x1 = float(r["xmax"]) * pix.width
        y1 = float(r["ymax"]) * pix.height
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
    img.save(OUT_DIR / f"page_{p:03d}.png")
```

For detailed review-editing procedures and scanned-page coordinate patterns, use [`../doc-redaction-modifications/SKILL.md`](../doc-redaction-modifications/SKILL.md) and [`../doc-redaction-modifications/TROUBLESHOOTING.md`](../doc-redaction-modifications/TROUBLESHOOTING.md).

## Full reference

### Recommended runtime order

1. `gradio_client` (default)
2. raw `/gradio_api/*` HTTP (fallback)
3. `/agent/*` only with server-local paths

### `gradio_client` default call pattern

Use **`document_file`** (not `pdf_file`) for `/doc_redact`. Hugging Face Spaces and slow TLS benefit from **long `httpx` timeouts**; defaults often raise `ConnectTimeout` on cold start or long jobs.

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
# client.view_api()  # prints endpoint signatures

result = client.predict(
    api_name="/doc_redact",
    document_file=handle_file("/local/path/document.pdf"),
)

# Then download each server path (see §3b). Example:
headers = {}
if HF_TOKEN:
    headers["Authorization"] = f"Bearer {HF_TOKEN.strip()}"
out_dir = Path("output/run_001")
out_dir.mkdir(parents=True, exist_ok=True)
with httpx.Client(timeout=httpx_kwargs["timeout"], headers=headers) as http:
    for p in result[0]:  # /doc_redact returns (output_paths, message)
        # If entries are dicts with "path", walk recursively (§3b / extract_file_like_paths).
        if not isinstance(p, str) or not p.startswith("/"):
            continue
        url = f"{BASE_URL}/gradio_api/file={quote(p, safe='')}"
        dest = out_dir / Path(p).name
        dest.write_bytes(http.get(url).raise_for_status().content)
```

### Full `/redact_document` cold-start template

Use this only when short routes are insufficient. Keep these defaults unless deployment docs require changes:

```python
kwargs = {
    "file_paths": [handle_file("/local/path/document.pdf")],
    "chosen_redact_entities": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
    "chosen_redact_comprehend_entities": [],
    "chosen_llm_entities": ["PERSON"],  # must be non-empty
    "ocr_review_files": [],
    "combined_out_message": "",
    "output_folder": "/tmp/gradio",  # must be non-empty
}
result = client.predict(api_name="/redact_document", **kwargs)
```

### Pragmatic return-value handling for `/redact_document`

Current tested deployments often return around twelve values. Practical strategy:

1. Treat `result` as tuple/list.
2. Extract all file-like values by suffix match (`.pdf`, `.csv`, `.json`).
3. Keep status strings for logging.
4. If deployment upgrades break shape, re-check `client.view_api()` and adjust mapping once.

Example:

```python
def extract_paths(result):
    out = []
    for item in result:
        if isinstance(item, str) and item.lower().endswith((".pdf", ".csv", ".json")):
            out.append(item)
        elif isinstance(item, list):
            out.extend(
                s for s in item
                if isinstance(s, str) and s.lower().endswith((".pdf", ".csv", ".json"))
            )
    return out
```

### Raw HTTP fallback checklist

1. `GET /gradio_api/info`
2. `POST /gradio_api/upload` with multipart `files`
3. `POST /gradio_api/call/{api_name}` with `{"data":[...]}`
4. Poll `GET /gradio_api/call/{api_name}/{event_id}`
5. Download outputs with `GET {BASE}/gradio_api/file={urllib.parse.quote(path, safe="")}` (always URL-encode `path`; use `Bearer` if the Space is gated). Or read shared disk.

### MCP usage guidance (when vs not when)

Use MCP (`mcp_doc_redaction`) when tools are already wired into an IDE agent runtime (for example Cursor) and you want structured tool calls plus bundled zip/manifest outputs.

Do not choose MCP as first step for standalone scripts or generic automation; `gradio_client` is simpler there.

### Authentication

- HF Spaces private/gated: `HF_TOKEN` bearer auth
- `/agent/*` when configured: `X-Agent-API-Key`
- Enterprise reverse proxy: deployment-specific cookies/headers

### Expected outputs

- `*_redacted.pdf`
- `*_redactions_for_review.pdf`
- `*_review_file.csv`
- `*_ocr_output_*.csv`
- `*_ocr_results_with_words_*.csv` and/or `.json`

Package artifacts with a manifest (`name`, `size`, `sha256`, timestamp, run metadata).
