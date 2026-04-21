---
name: doc-redaction-modifications
description: "Skill for modifying existing redactions using the Document Redaction app. Upload + apply via Gradio review_apply (3-param HTTP) or /agent when paths are allowed; Docker output paths; page-by-page review + verification exports."
version: 1.3.4
author: repo-maintained
license: AGPL-3.0-only
changelog:
  - "v1.3.4 (Apr 21, 2026): Add explicit `word_text` -> bbox -> `*_review_file.csv` row recipe; document tested `word_level_ocr_text_search` behavior (similarity range 0..1, requires in-session word-level OCR state)."
  - "v1.3.3 (Apr 21, 2026): Field test notes — `gradio_client` 2.x has no `Client.upload`; use `handle_file` for local PDF/CSV with `/review_apply`, or raw `/gradio_api/upload` then string paths."
  - "v1.3.2 (Apr 21, 2026): Rename short apply endpoint to `review_apply` (remove old route)."
  - "v1.3.1 (Apr 20, 2026): (Superseded) Previously used `apply_review_redactions_from_uploads` naming."
  - "v1.3.0 (Apr 20, 2026): Docker/output paths; page-scope note."
  - "v1.2.1 (Apr 20, 2026): (Superseded) Documented older route naming."
  - "v1.2 (Apr 20, 2026): Prefer `POST /agent/apply_review_redactions` when FastAPI + server paths are available; Gradio apply remains fallback for Spaces-only."
  - "v1.1 (Apr 20, 2026): Gradio 6 / gradio_client troubleshooting (explicit api_name, arity, async polling, annotator payload); cross-ref `/agent/operations` + richer 501 hints."
  - "v1.0 (Apr 20, 2026): Initial remote-first rewrite (Gradio HTTP API default; `/agent` limitations; optional word-level search path; page-by-page visual verification via export overlays)."
---

## Purpose

Modify an existing redaction run. Pick the surface that matches **where files actually live**:

- **Gradio: `review_apply`** (`/gradio_api/...`) — **three parameters** (PDF path, review CSV path, optional `output_dir` or `null`). Confirmed via `client.view_api()` / `GET /gradio_api/info`. **Raw HTTP** (`POST /gradio_api/call/review_apply`, poll until done) is often the **most reliable** headless apply after `POST /gradio_api/upload`, including **Docker** setups where outputs are written **inside** the container (see below).
- **`POST /agent/apply_review_redactions`** ([`AGENTS.md`](../../AGENTS.md)) — best when `RUN_FASTAPI` is on and both paths are **server-local paths the validator accepts** (repo root, `INPUT_FOLDER`, or `OUTPUT_FOLDER`). **Not** a substitute when your only copies of the PDF/CSV live at **ephemeral Gradio paths** (e.g. `/tmp/gradio_tmp/...` inside the container) that **do not** map to an allowed root **or** to a folder you can read from the host without `docker cp` / a bind mount.
- **Browser/GUI** — fallback.

`load_and_prepare_documents_or_data` and `word_level_ocr_text_search` remain Gradio-session-heavy; see `/agent` limitations below.

### Docker / container outputs (important)

**`gradio_client`** and the Gradio HTTP API both run apply **correctly**, but artifacts are usually written under the app’s configured **`OUTPUT_FOLDER`** (e.g. `/home/user/app/output/` in a Space or Docker image). That path is **inside** the container unless you **bind-mount** it, use **`docker cp`**, or **`docker exec` + archive (e.g. `tar`)**. Planning only “call the API from the host” without one of those is a common reason outputs feel “missing.” Prefer a **mounted output directory**, **`docker cp`**, or downloading via **`GET /gradio_api/file=...`** when your auth/cookies allow it (see the doc-redaction-app skill for `file=` caveats).

## Recommended posture: review page-by-page

For reliability and auditability, process the document **page-by-page**:

- Edit redactions for one page.
- Generate visual checks for that page.
- Only then move to the next page.

This reduces “silent misses” and makes it easy to explain what changed. For **small** documents, applying edits **in one pass** is acceptable; for **larger** PDFs, strict page-by-page review reduces error surface.

## Inputs and outputs

### Inputs (from a prior run)

- Original document (PDF)
- `*_review_file.csv` (the redaction plan)
- Word-level OCR output (recommended): `*_ocr_results_with_words_*.csv` (and/or `.json`)

### Outputs (after applying modifications)

- Updated redacted document (typically `*_redacted.pdf`)
- Updated review CSV — naming is **not** a bare `review_file.csv`. The pipeline derives the name from the **input PDF filename** by appending **`_review_file.csv`** to the name that includes the **`.pdf`** extension, e.g. `partnership_agreement.pdf_review_file.csv` (see `tools/file_redaction.py` / `tools/redaction_review.py`). The **input** CSV you upload must still be a `*_review_file.csv` whose basename contains **`_review_file`** (prepare pipeline requirement).
- Optional per-page visualisations:
  - OCR visualisation images
  - Redaction overlay images

## Default workflow (remote-first, API-driven)

### Step 1 — Download the prior run bundle

Obtain the original PDF + the outputs from the previous run (at minimum `*_review_file.csv`).

### Step 2 — Edit `*_review_file.csv` offline

Operate directly on the review CSV:

- **Remove false positives**: delete rows that should not be redacted.
- **Add missed redactions**: append rows for missed items.

#### Finding missed items (recommended: offline OCR search)

Search the word-level OCR output (`*_ocr_results_with_words_*.csv`) locally:

- Find occurrences of names/phrases/codes.
- For each match, derive a redaction box (normalized \(0..1\) coordinates) and add a row to `*_review_file.csv`.

Concrete mapping (field-tested on `example_of_emails_sent_to_a_professor_before_applying_ocr_results_with_words_*.csv`):

- OCR columns include: `page`, `word_text`, `word_x0`, `word_y0`, `word_x1`, `word_y1` (plus line-level fields).
- Review CSV target columns: `image,page,label,color,xmin,ymin,xmax,ymax,id,text`.
- Suggested row mapping for a `word_text` match:
  - `image` -> `placeholder_image_{page-1}.png`
  - `page` -> OCR `page`
  - `label` -> `CUSTOM` (or your policy label)
  - `color` -> `(0, 0, 0)` (or deployment default)
  - `xmin,ymin,xmax,ymax` -> `word_x0,word_y0,word_x1,word_y1`
  - `id` -> new unique token/string
  - `text` -> OCR `word_text`
- Then upload the modified `*_review_file.csv` and run `review_apply`.
- CSVs may include BOM-prefixed first headers (e.g. `\ufeffpage` / `\ufeffimage`), so read with `utf-8-sig` when scripting edits.

Minimal executable snippet (offline `word_text` search -> append row -> apply):

```python
import csv
import random
import string
import tempfile
from pathlib import Path
from gradio_client import Client, handle_file

BASE_URL = "https://seanpedrickcase-document-redaction.hf.space"
PDF_PATH = Path("example_data/example_of_emails_sent_to_a_professor_before_applying.pdf")
REVIEW_CSV = Path("example_data/example_outputs/example_of_emails_sent_to_a_professor_before_applying_review_file.csv")
OCR_WORDS_CSV = Path("example_data/example_outputs/example_of_emails_sent_to_a_professor_before_applying_ocr_results_with_words_textract.csv")
SEARCH_WORD = "professor"

# 1) Find a word match in OCR output (column: word_text)
with OCR_WORDS_CSV.open(newline="", encoding="utf-8-sig") as f:
    ocr_rows = csv.DictReader(f)
    match = next((r for r in ocr_rows if (r.get("word_text") or "").lower() == SEARCH_WORD), None)
if not match:
    raise RuntimeError(f"No word_text match for '{SEARCH_WORD}'")

# 2) Append a new review row using word bbox columns
with REVIEW_CSV.open(newline="", encoding="utf-8-sig") as f:
    existing = list(csv.DictReader(f))
fieldnames = list(existing[0].keys())  # preserves BOM-prefixed first header if present
image_col = fieldnames[0]
page = int(float(match["page"]))
new_row = {
    image_col: f"placeholder_image_{max(page - 1, 0)}.png",
    "page": str(page),
    "label": "CUSTOM",
    "color": "(0, 0, 0)",
    "xmin": match["word_x0"],
    "ymin": match["word_y0"],
    "xmax": match["word_x1"],
    "ymax": match["word_y1"],
    "id": "".join(random.choice(string.ascii_letters + string.digits) for _ in range(12)),
    "text": match["word_text"],
}

tmp_dir = Path(tempfile.mkdtemp(prefix="review_mod_"))
edited_review_csv = tmp_dir / REVIEW_CSV.name  # keep *_review_file.csv naming
with edited_review_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(existing)
    w.writerow(new_row)

# 3) Apply edits (short route: /review_apply)
client = Client(BASE_URL)
paths, message = client.predict(
    api_name="/review_apply",
    pdf_file=handle_file(str(PDF_PATH.resolve())),
    review_csv_file=handle_file(str(edited_review_csv.resolve())),
    output_dir=None,
)
print("apply message:", message)
print("output paths:", paths)
```

Minimal executable snippet (raw HTTP upload + `review_apply` polling):

```python
import json
from pathlib import Path
import httpx

BASE_URL = "https://seanpedrickcase-document-redaction.hf.space"
PDF_PATH = Path("example_data/example_of_emails_sent_to_a_professor_before_applying.pdf")
EDITED_REVIEW_CSV = Path("path/to/edited_review_file.csv")  # basename should contain _review_file

with httpx.Client(timeout=120.0) as client:
    # 1) Upload PDF + edited review CSV
    up_pdf = client.post(
        f"{BASE_URL}/gradio_api/upload",
        files={"files": (PDF_PATH.name, PDF_PATH.read_bytes())},
    )
    up_pdf.raise_for_status()
    pdf_internal = up_pdf.json()[0]

    up_csv = client.post(
        f"{BASE_URL}/gradio_api/upload",
        files={"files": (EDITED_REVIEW_CSV.name, EDITED_REVIEW_CSV.read_bytes())},
    )
    up_csv.raise_for_status()
    csv_internal = up_csv.json()[0]

    # 2) Call short apply route with 3-slot data payload
    call = client.post(
        f"{BASE_URL}/gradio_api/call/review_apply",
        content=json.dumps({"data": [pdf_internal, csv_internal, None]}),
        headers={"Content-Type": "application/json"},
    )
    call.raise_for_status()
    event_id = call.json()["event_id"]

    # 3) Poll until completion (SSE payload includes final `data`)
    poll = client.get(f"{BASE_URL}/gradio_api/call/review_apply/{event_id}")
    poll.raise_for_status()
    sse_text = poll.text

print("event_id:", event_id)
print("raw poll body starts with:", sse_text[:200])
# Parse the final `data:` JSON block to extract output file paths, then download with:
# GET {BASE_URL}/gradio_api/file={internal_path}
```

Notes:

- Keep edits **page-scoped** (finish page \(n\), validate, then proceed).
- Prefer exact matching first; use fuzzy/regex carefully.

### Step 3 — Upload edited artifacts to the app (Gradio HTTP API)

1. Upload the original PDF.
2. Upload the edited `*_review_file.csv`.
3. (Optional) Upload word-level OCR outputs if your deployment requires them to populate session state.

Endpoint:

- `POST {BASE_URL}/gradio_api/upload` with multipart key `files`

Store the returned internal paths (e.g. `/tmp/gradio/...`).

### Step 4 — Load the review session state

Call the Gradio endpoint:

- `api_name="load_and_prepare_documents_or_data"`

Important:

- Always build the `data` array by first calling `GET {BASE_URL}/gradio_api/info` and using the **actual parameter order** for your deployment.
- Provide the uploaded internal paths for the PDF and edited review CSV in the positions expected by the endpoint.

### Step 5 — (Optional but recommended) Visual verification for the current page

After load (and/or after apply), generate images to sanity-check the page:

- `api_name="review_ocr_export"`: shows OCR word boxes for a page (useful to confirm OCR alignment and that the target text exists where you think it does).
- `api_name="review_overlay_export"`: shows redaction boxes/legend for a page (useful to confirm box placement/coverage).

As with other calls, use `/gradio_api/info` to build the correct `data` array, then:

- Poll completion via `GET {BASE_URL}/gradio_api/call/{api_name}/{event_id}`
- Download images via `GET {BASE_URL}/gradio_api/file={internal_path}`

### Step 6 — Apply review redactions to generate updated outputs

**Practical default (uploads + short HTTP):** after `POST /gradio_api/upload` for the PDF and edited CSV, call:

- `POST {BASE_URL}/gradio_api/call/review_apply`
- Body: `{"data": [<pdf_internal_path>, <csv_internal_path>, <output_dir_or_null>]}`  
  Use JSON `null` for the third slot to keep the app’s default **`OUTPUT_FOLDER`** (still inside the container unless configured otherwise).

Poll `GET {BASE_URL}/gradio_api/call/review_apply/{event_id}` until complete. The response includes the **list of output paths** the server wrote; collect those paths (or copy from the mounted output dir / `docker cp`).

**`gradio_client`:** `Client.predict(..., api_name="/review_apply", ...)` with the same three arguments works, but you still must **retrieve bytes** from container-local paths unless the output tree is shared with the host.

**Return value (`gradio_client` / `predict`) — structured extraction**

The handler exposes **two** API outputs: a **list of file paths** and a **status string**. In Python you get a **two-tuple** (same idea as a list `[paths, message]`):

```text
(paths, message)
```

- **`paths`** — `list[str]`. Field-tested order for the **primary** artifacts is often:
  1. Original / source PDF path (as processed on the server)
  2. Redacted PDF (`*_redacted.pdf`)
  3. Review / overlay PDF (`*_redactions_for_review.pdf`)
  4. Updated review CSV (`*.pdf_review_file.csv` or equivalent `*_review_file.csv`)

  The pipeline may **append** extra paths (logs, OCR sidecars). Prefer **matching by filename suffix** (`*_redacted.pdf`, `*_review_file.csv`, etc.) over assuming exactly four entries or fixed indices.

- **`message`** — short human-readable summary (prepare/apply status).

Example (local PDF and `*_review_file.csv` on the agent machine — **field-tested** with `gradio_client` 2.4):

```python
from gradio_client import Client, handle_file

client = Client(BASE_URL)  # optional hf_token=... for private Spaces
paths, msg = client.predict(
    api_name="/review_apply",
    pdf_file=handle_file("/path/to/document.pdf"),
    review_csv_file=handle_file("/path/to/document.pdf_review_file.csv"),  # basename must contain `_review_file`
    output_dir=None,
)
# paths: list of server output paths (often 3–4+); scan for suffixes *_redacted.pdf, *_review_file.csv, etc.
```

Some **`gradio_client`** versions expose a helper **`upload`**; **`Client` in 2.4.x does not**. If you use **`POST /gradio_api/upload`** instead, pass the returned **internal path strings** as **`pdf_file`** / **`review_csv_file`** (no `handle_file` wrapper — see trap below).

Use **`client.view_api()`** (printed docs; return may be `None`) or **`GET /gradio_api/info`** to confirm output names/types for your Gradio version.

**`handle_file()` vs server paths (common trap)**

`handle_file("/local/doc.pdf")` is for files on **the client machine** (or real URLs). After **`POST /gradio_api/upload`**, you receive **server-internal** strings such as `/tmp/gradio_tmp/...`. Passing those strings into **`handle_file(...)`** raises **`ValueError: File does not exist on local filesystem and is not a valid URL`**, because the client cannot see the container filesystem.

- **Initial redaction** from a **local** PDF: `handle_file` is appropriate.
- **Review apply** using **uploaded** artifacts: pass **`str`** paths returned from **`client.upload`** (or from raw HTTP upload JSON) **directly** as `pdf_file` / `review_csv_file` — **do not** wrap them in `handle_file`.

**When `/agent` is actually preferable:** `POST {BASE_URL}/agent/apply_review_redactions` with JSON body ([`AGENTS.md`](../../AGENTS.md)) when **both** paths are real, allowed server paths (repo root, `INPUT_FOLDER`, or `OUTPUT_FOLDER`)—for example a shared volume or CI job that writes inputs under `/data/...`. If files exist only under **`/tmp/gradio_tmp/...`** after upload, they are often **not** in an allowed root or not visible on the host; use the **Gradio three-parameter** route above instead of fighting path validation.

Fields for `/agent`: `pdf_path`, `review_csv_path` (basename **must** contain `_review_file`), optional `output_dir` / `input_dir`, etc. Auth: `X-Agent-API-Key` when configured.

**Legacy / avoid unless necessary:** `api_name="apply_review_redactions"` — long `data[]` from a chained UI handler; use **`GET /gradio_api/info`** for exact arity or prefer **`review_apply`**.

### Step 7 — Download outputs and package results

From the completion payloads, extract all file paths and download them:

- `GET {BASE_URL}/gradio_api/file={internal_path}`

Produce:

- `manifest.json` including: filenames, byte sizes, sha256 hashes, timestamps, and the page-by-page decisions.
- `outputs.zip` containing all outputs + manifest.

## Optional helper path: `word_level_ocr_text_search`

The Gradio endpoint:

- `api_name="word_level_ocr_text_search"`

is useful when you want the app’s own in-session search UI logic (regex toggle, similarity threshold) applied to the **already-loaded** OCR-with-words dataframe.

Guidance:

- Treat this as **optional**: it depends on correct session state being loaded by `load_and_prepare_documents_or_data`.
- Prefer offline OCR CSV search if you want fewer moving parts and more deterministic control.
- Use this endpoint when:
  - You need **regex/fuzzy search** behaviour that should match the app UI’s semantics.
  - You want the app to return a filtered “candidates table” that can drive page-by-page review.
- Inputs are currently lightweight: `search_text`, `similarity_threshold`, `use_regex_flag`.
  - `similarity_threshold` is a float in **`0..1`** (passing `100` fails validation).
- Return shape (tested): `(filtered_df, artifact_paths)` where `filtered_df` has columns like `page,line,word_text,index`.
- Typical sequence:
  1. Ensure word-level OCR state exists in the same Gradio session (most reliable: run `redact_document` first, or load a state that includes OCR-with-words).
  2. Call `word_level_ocr_text_search` with your search text and options.
  3. Use results to decide what to add/remove in the offline `*_review_file.csv`.
  4. Re-upload edited review CSV and re-load, then call **`review_apply`** (or the long UI `apply_review_redactions` chain if you are mirroring the full Review tab).

## Gradio 6.x and `gradio_client` (avoiding endpoint and arity errors)

Deployments pin **Gradio ≤ 6.10** (`pyproject.toml`). The Python client often breaks in three ways:

1. **Ambiguous or wrong endpoint resolution** — Do not pass a **Python function name** or a fuzzy label to `predict()`. Always pass an explicit route name from **`GET /gradio_api/info`**:

   - For **headless apply after uploads**, prefer: `Client.predict(..., api_name="/review_apply", ...)` (three arguments).
   - The legacy Review-tab chain `apply_review_redactions` has a **long** `data` array; avoid it unless you truly need that UI wiring.

2. **`ValueError` on argument count** — The `data` array must match the schema. **`review_apply`** stays at **three** slots; **`apply_review_redactions`** and **`load_and_prepare_documents_or_data`** are **long** — build from **`/gradio_api/info`**, not from guesses.

3. **Async / event id** — `POST /gradio_api/call/<api_name>` returns an event id; you must **`GET /gradio_api/call/<api_name>/<event_id>`** until the result is ready. `gradio_client` does this for you when you use its high-level call API correctly; raw HTTP callers must poll.

4. **`handle_file` only for local client paths** — Server paths from **`upload`** (e.g. `/tmp/gradio_tmp/...`) must be passed as **plain strings** to `predict`, not wrapped in `handle_file` (see Step 6 above).

**Recommended stable pattern for headless “apply edited review CSV” (any language):**

- `GET {BASE_URL}/gradio_api/info` → confirm **`review_apply`** arity (expect **three** slots: PDF path, CSV path, optional output dir).
- `POST {BASE_URL}/gradio_api/upload` (×2) → internal paths for PDF and `*_review_file.csv`.
- `POST {BASE_URL}/gradio_api/call/review_apply` with `{"data":[pdf_path, csv_path, null]}` (or a string `output_dir`).
- Poll `GET {BASE_URL}/gradio_api/call/review_apply/{event_id}` until complete.
- Fetch files with `GET {BASE_URL}/gradio_api/file={path}` **or** copy from the container output directory / bind mount.

For **`load_and_prepare_documents_or_data`** (session UI), still read the full schema from `/gradio_api/info` — that route uses a **long** `data` array.

**`gradio_image_annotation` / annotator payloads:** The image annotator is serialized as a **dict** (not a bare file path). It must look like:

- `{"image": "<internal path from upload or server path>", "boxes": [<box dicts>]}`

Use `boxes: []` when you have no manual boxes; each box typically includes geometry plus string `label` and `color` (the frontend expects string colours). If preprocessing fails inside the annotator component, compare your payload to a working UI session (export from browser devtools or reduce to minimal `image` + empty `boxes`).

**Smaller API surface when stateless exports are enough:** `export_review_redaction_overlay` and `export_review_page_ocr_visualisation` are also exposed under **`/agent`** with small JSON bodies (`AGENTS.md`). Prefer those for one-off images if you already have paths and data.

**Multi-document merge (related):** `combine_review_csvs` / `combine_review_pdfs` are separate `api_name`s for packaging many review artifacts; they do not replace **`review_apply`** (or `/agent/apply_review_redactions` with valid paths) when you need updated PDFs from a review CSV.

**Discovery from FastAPI:** `GET /agent/operations` lists all `api_name` values and marks which routes are Gradio-session-only (**501**).

## `/agent` route limitations (important)

Even if the deployment exposes FastAPI under `/agent`, some review helpers are not implemented there:

- `/agent/load_and_prepare_documents_or_data` → **501**
- `/agent/word_level_ocr_text_search` → **501**

Reason: these depend on Gradio session state and in-memory dataframes. The **501 JSON body** includes `gradio_http` and `gradio_client_notes` for programmatic callers.

Implemented for headless review apply:

- `/agent/apply_review_redactions` — runs the same prepare + apply core as the Review tab (`tools/simplified_api.py`).

Where `/agent` *is* also useful:

- Initial redaction tasks that operate on server-visible paths (e.g. `/agent/redact_document`) in deployments with shared filesystem access.

## Browser/GUI fallback

Use only when the API surface is blocked:

- Load the PDF and review CSV in the Review workflow.
- Apply page-by-page modifications in the UI.
- Export/Download final redacted outputs.

## No-human success checklist

- [ ] All prior-run artifacts downloaded (PDF + `*_review_file.csv` + OCR-with-words)
- [ ] Page-by-page edits applied to `*_review_file.csv` with a recorded rationale per page
- [ ] Session loaded via `load_and_prepare_documents_or_data` (Gradio path) **or** PDF + CSV uploaded and applied via **`review_apply`** **or** paths on allowed roots for `POST /agent/apply_review_redactions`
- [ ] Visual verification images generated for sampled pages (or all pages for high-risk documents)
- [ ] Review apply completed (`POST /agent/apply_review_redactions` and/or **`POST /gradio_api/call/review_apply`**)
- [ ] Outputs recovered from **container output dir**, **HTTP `file=`**, or **docker cp** / mount — not assumed to exist on the host by magic
- [ ] Updated outputs downloaded and packaged (`outputs.zip` + `manifest.json`)
