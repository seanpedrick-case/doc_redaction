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
