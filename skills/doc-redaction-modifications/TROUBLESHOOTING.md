# Troubleshooting: Redaction Modifications

Use this file only when the standard `SKILL.md` workflow fails.

## 1) `/agent/apply_review_redactions` fails (404/501/path errors)

### Symptoms
- 404 on `/agent/apply_review_redactions`
- 501 or route not implemented
- Path validation rejects inputs

### Fix
- Switch to `review_apply` immediately:
  - `gradio_client` with `api_name="/review_apply"`, or
  - raw HTTP `/gradio_api/call/review_apply`.
- Use `/agent` only when both `pdf_path` and `review_csv_path` are server-local and accepted by route validation.

## 2) `gradio_client` call fails with wrong endpoint or arity

### Symptoms
- `ValueError` about argument count
- Endpoint name mismatch

### Fix
- Confirm endpoint shape first:
  - `GET /gradio_api/info` or `client.view_api()`.
- Use the short route:
  - `/review_apply` with exactly 3 inputs: `pdf_file`, `review_csv_file`, `output_dir`.
- Avoid legacy long Review UI-chain handlers unless specifically required.

## 3) `handle_file(...)` fails after upload

### Symptoms
- `ValueError: File does not exist on local filesystem...`

### Cause
- You wrapped a server-internal path (for example `/tmp/gradio_tmp/...`) with `handle_file(...)`.

### Fix
- `handle_file(...)` is for local client files only.
- If using `/gradio_api/upload`, pass returned server paths directly as plain strings in raw HTTP calls.

## 4) Outputs are "missing" after successful apply

### Symptoms
- API says success but files are not on host filesystem.

### Cause
- Outputs were written inside container path (for example `/home/user/app/output/...`).

### Fix
- Recover files via one of:
  - `GET /gradio_api/file={internal_path}`
  - bind-mounted output directory
  - `docker cp` from container

## 5) CSV edits corrupt headers or columns

### Symptoms
- First column appears as garbled header
- Parser misses expected fields

### Cause
- UTF-8 BOM in exported review CSV.

### Fix
- Read/write with `encoding="utf-8-sig"`.
- Preserve original field order from existing CSV before writing.

## 6) Scanned-page coordinate generation is unstable

### Symptoms
- Syntax errors in ad hoc one-liners
- Random box placement gives unreliable results

### Fix
- Use deterministic zone presets (see `SKILL.md`).
- Create boxes via explicit page+zone spec JSON.
- Verify with generated review images before applying to all pages.

## 7) Visual review endpoints are unreliable headlessly

### Symptoms
- `/page_ocr_review_image` or `/page_redaction_review_image` fails or returns unusable state errors.

### Cause
- These endpoints often require in-memory Gradio session state.

### Fix
- Use offline visual verification:
  - Render PDF pages with PyMuPDF.
  - Draw review CSV boxes locally.
  - Review review images with human or vision model.

## 8) Naming/input constraints cause silent apply failures

### Symptoms
- Apply runs but expected rows are ignored.
- Output CSV/PDF does not reflect inserted edits.
- Status text is generic and does not explain why rows were skipped.

### Cause
- Input CSV basename does not contain `_review_file`.
- `output_dir` is not `None` and not a valid server path.
- Inserted rows use page numbers that do not match the PDF page model (must be 1-based).

### Fix
- Ensure review CSV filename contains `_review_file` (for example `contract.pdf_review_file.csv`).
- Use `output_dir=None` unless you are certain the provided path exists and is writable on the server.
- Validate page numbers before apply:
  - First page is `1`, not `0`.
  - Max page value does not exceed source PDF page count.

## 9) Text layer leaks but word OCR shows 100% covered

### Symptoms
- Post-apply `verify_redaction_coverage` lists `text_layer_leaks` on `*_redacted.pdf`
- Word OCR overlap looks complete; agent concludes `/review_apply` “only draws overlays”

### Cause
- Wrong PDF tested (`*_redactions_for_review.pdf` retains text)
- CSV coordinates not normalized (pixel/point values >1) — boxes miss text silently on headless apply before validation was added
- Text baked into embedded images — text redaction cannot target it precisely
- Multi-line PyMuPDF blocks overlapped by one large box but substring positions still leak

### Fix
1. Confirm PDF is `*_redacted.pdf`.
2. Check coverage report `leak_likely_causes` per page.
3. Validate CSV: all bbox values in **[0, 1]**; normalize any PyMuPDF absolute coords before apply.
4. Add/widen `CUSTOM` boxes or use targeted Pass 2 VLM for image text — **do not** reimplement apply with PyMuPDF unless `/review_apply` itself errors.

## 10) `verify_redaction_coverage` path rejected on Agent API

### Symptoms
- `Path must be under the app repo, INPUT_FOLDER, or OUTPUT_FOLDER`
- Calling `verify_redaction_coverage()` from the Pi agent container fails on redaction-server paths
- `/tmp/gradio_tmp/...` paths from `/gradio_api/upload` are rejected

### Cause
- **Split-container deployment:** Pi agent and doc_redaction have **no shared filesystem**. Agent API path validation runs on the **redaction server** only.
- Pi workspace paths do not exist on the redaction container.
- Gradio upload temp paths are not under `OUTPUT_FOLDER`.
- Importing `verify_redaction_coverage` on the Pi container still applies path checks against the Pi filesystem.

### Fix
1. **Pre-apply** (CSV edited in Pi session workspace): download review CSV and OCR words CSV via `fetch_redaction_files`, then run:
   ```bash
   python tools/verify_redaction_coverage.py <local_review_csv> <local_ocr_words_csv> \
     --must-redact "..." --must-not-redact "..."
   ```
2. **Post-apply** (after `/review_apply`): call `POST {gradio_url}/agent/verify_redaction_coverage` with **server paths** from `extract_server_paths(review_apply result)`:
   - `review_csv_path` — post-apply review CSV on redaction server
   - `ocr_words_csv_path` — from the same `/doc_redact` run (already on server)
   - `redacted_pdf_path` — post-apply `*_redacted.pdf` on redaction server
3. **Do not** pass Pi workspace paths, `/tmp/gradio_tmp/...` upload paths, or call the Python API from the Pi container with redaction-server path strings.
