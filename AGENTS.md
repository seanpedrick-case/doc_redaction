# AGENTS.md

Context for AI coding agents working on **doc_redaction** (PII redaction for PDFs, images, Word, and tabular files). Human-oriented docs: [README.md](README.md). User guide: [doc_redaction user guide](https://seanpedrick-case.github.io/doc_redaction/src/user_guide.html).

## Project overview

- **Stack**: Python 3.10+, Gradio UI ([app.py](app.py)), optional FastAPI when `RUN_FASTAPI` is enabled, AWS/LLM integrations via [tools/config.py](tools/config.py) and env files under `config/`.
- **License**: AGPL-3.0-only (see [pyproject.toml](pyproject.toml)). Respect license terms when adding dependencies.
- **Accuracy**: Outputs are not guaranteed complete; downstream use should assume **human review** of redacted material.

## Cursor skills: redaction workflow (optional)

For agents operating the deployed app (Gradio Client, review CSV, `/review_apply`), these repo-local playbooks are a suggested ladder:

1. **[`skills/doc-redaction-app/SKILL.md`](skills/doc-redaction-app/SKILL.md)** — first-pass redaction (`/doc_redact` / `/redact_document`) and downloading artifacts.
2. **[`skills/doc-redact-page-review/SKILL.md`](skills/doc-redact-page-review/SKILL.md)** — after outputs exist: **parallel per-page** child agents, merge into one full-document `*_review_file.csv`, **single** `/review_apply` from the parent.
3. **[`skills/doc-redaction-modifications/SKILL.md`](skills/doc-redaction-modifications/SKILL.md)** — CSV mechanics, `preview_redaction_boxes`, `/review_apply` patterns, verification, VLM and PyMuPDF fallbacks (single-thread edits and the **technical** reference for page-review children).

## Setup

1. **System**: Install **Tesseract** and **Poppler** (required for OCR/PDF). See [README.md](README.md) (Windows/Linux sections).
2. **Python**: Create a venv, then install the project (e.g. `pip install -e ".[dev]"` or follow README).
3. **Configuration**: Copy or edit environment/config as described in README / `config/` (e.g. `app_config.env`). Do not commit secrets.

## Run locally

- Gradio/FastAPI entrypoint is [app.py](app.py). With FastAPI enabled, typical pattern is `uvicorn app:app --host 0.0.0.0 --port 7860` (exact host/port from your config).
- OpenAPI docs: `/docs` when the FastAPI app is mounted.

## Tests

- Run from repo root: `pytest` (optional: `pytest test/`).
- Fix failures related to your changes before opening a PR.

## Line order (local OCR and simple text extraction)

Multi-column layouts use shared logic in [`tools/ocr_reading_order.py`](tools/ocr_reading_order.py). Controlled by **`LOCAL_OCR_READING_ORDER`** (`column` default, `legacy` for previous top-left behaviour).

### Local OCR (Paddle/Tesseract)

Word boxes are merged into line-level CSV rows in [`combine_ocr_results`](tools/custom_image_analyser_engine.py).

- **`column`**: detect text columns, assign line numbers down each column left-to-right; full-width lines (headers) first. Stops cross-column merging that produced wide erroneous lines on multi-column PDFs. **Auto-fallback**: the page is treated as single-column unless a *consecutive cluster* of gutter rows (y-gap between adjacent rows ≤ `OCR_COLUMN_MAX_CONSECUTIVE_GUTTER_GAP`, default `0.06` of page height) has ≥ `OCR_COLUMN_MIN_GUTTER_ROWS` (default `3`) rows **and** the cluster's topmost row is above the footer zone (`OCR_COLUMN_FOOTER_ZONE_FRACTION`, default `0.75`). This prevents isolated header bands (logo | title, 1 gutter row), signature-only blocks at the page bottom (cluster starts at y ≥ 0.75), or the combination of both, from forcing column mode on the single-column body text between them.
- **`PADDLE_PRESERVE_LINE_BOXES=True`** or **`CONVERT_LINE_TO_WORD_LEVEL=False`** with Paddle: keep Paddle line boxes (skip word split + regrouping); line numbers still use column reading order.

### Simple text extraction (PyMuPDF)

[`redact_text_pdf`](tools/file_redaction.py) → [`process_page_to_structured_ocr_pymupdf`](tools/file_redaction.py) calls [`reorder_structured_text_lines`](tools/ocr_reading_order.py) after collecting lines, using **`page.mediabox`** width/height for full-span header detection.

`reorder_structured_text_lines` now mirrors `build_line_groups` (local OCR route):

1. **Column-aware sort** (`sort_reading_order` / `assign_layout_boxes` / `detect_column_split_xpoints`) — or legacy top-left for single-column pages.
2. **Y-band grouping** (`group_into_lines`) — merges any same-row PyMuPDF lines that were emitted as separate objects (e.g. mixed-font spans) and splits horizontally-disparate boxes via `_finalize_line`.  *Column mode only.*
3. **Secondary sub-column pass** (`_reorder_lines_column_major`) — ensures correct column-major order when sub-columns sit within a single macro-column.  *Column mode only.*
4. When a group contains more than one box, constituent boxes are **merged** into a single `OCRResult` (union bbox, joined text, concatenated chars/words).

In single-column / legacy mode only step 1 is applied; PyMuPDF lines are pre-formed so no merging is needed.

### Tunables (both routes)

`OCR_FULL_SPAN_WIDTH_RATIO`, `OCR_COLUMN_GAP_MIN_FRACTION`, `OCR_COLUMN_GUTTER_MIN_FRACTION`, `OCR_COLUMN_SUBGUTTER_MIN_FRACTION` (default `0.015` — fine-grained gutter scan in `assign_layout_boxes`; lower = detects narrower sub-column boundaries), `OCR_COLUMN_MIN_GUTTER_ROWS`, `OCR_COLUMN_MAX_BOX_HEIGHT_RATIO`, `OCR_COLUMN_MAX_CONSECUTIVE_GUTTER_GAP`, `OCR_COLUMN_FOOTER_ZONE_FRACTION`, `OCR_LINE_SPLIT_GAP_FRACTION` (default 0.025 — horizontal gap fraction that forces a line split; must be below the narrowest column gutter, ~0.030 for two-page spreads; also used as the gap threshold for the secondary sub-column sort in `build_line_groups`), `OCR_LINE_Y_THRESHOLD_FRACTION` (default 0.013 — row-alignment tolerance as a fraction of page height; reduced from 0.015 to correctly separate tightly-set 10 pt body text whose row spacing is ~0.014), `OCR_LINE_Y_THRESHOLD_MIN_PX`.

**Sub-column ordering** (`build_line_groups`): after the primary word-level column sort, a second pass (`_reorder_lines_column_major`) clusters the produced line groups by their leftmost x-position using `OCR_LINE_SPLIT_GAP_FRACTION` as the gap threshold. This ensures that adjacent narrow sub-columns whose word-level centre gap is below `column_gap_threshold` (e.g. two columns on a spread where each page is already one macro-column) are still output in left-to-right column-major order rather than interleaved by y-position.

**Fine-grained gutter-based column assignment** (`assign_layout_boxes`): before falling back to centre-gap clustering, `detect_column_split_xpoints` scans the page for structural gutters at the finer `OCR_COLUMN_SUBGUTTER_MIN_FRACTION` threshold (default 0.015). Each qualifying gutter cluster produces a `(split_x, y_min)` pair — the split point is only applied to boxes whose `top ≥ y_min`, preventing a narrow sub-column gutter (visible only in the lower two-column section) from mis-splitting a full-width introductory paragraph that sits above it. This correctly separates narrow adjacent columns (e.g. 1.9 % gutter on a two-page spread) without fragmenting full-width headings or paragraphs.

Changing line order affects PII page text, duplicate-page detection, and review CSV line indices on multi-column documents; re-review after upgrading.

## Agentic / programmatic access (two surfaces)

### 1. FastAPI Agent API (recommended for LLM agents: small JSON bodies)

When `RUN_FASTAPI` is true, routes are mounted under **`/agent`** ([agent_routes.py](agent_routes.py)).

- **Catalog**: `GET /agent/operations` — maps each Gradio `api_name` to an HTTP path and notes whether the route is implemented via CLI or returns HTTP 501 for Gradio-only flows.
- **Implemented POST routes** (CLI- or [tools/simplified_api.py](tools/simplified_api.py)-backed where noted):  
  `redact_document`, `redact_data`, `find_duplicate_pages`, `find_duplicate_tabular`, `summarise_document`, `combine_review_pdfs`, `combine_review_csvs`, `export_review_redaction_overlay`, `export_review_page_ocr_visualisation`, `apply_review_redactions`, **`verify_redaction_coverage`** (Pass 1 programmatic QA: `review_csv_path`, `ocr_words_csv_path`, optional `must_redact` / `must_not_redact` regex lists, optional `redacted_pdf_path`), **`word_level_ocr_text_search`** (headless search in `*_ocr_results_with_words_*.csv` with optional review-box coverage flags).  
  Note: on Gradio ([app.py](app.py)), the Review-tab visual exports use `api_name` **`page_redaction_review_image`** and **`page_ocr_review_image`**; the **`/agent`** routes above keep the explicit `export_review_*` names for the same operations.
- **Gradio-only stubs** (501 + JSON hint): `load_and_prepare_documents_or_data`.
- **Auth**: If `AGENT_API_KEY` is set in the environment, send header `X-Agent-API-Key` with that value.
- **Paths**: Inputs must resolve to files under the repo root, `INPUT_FOLDER`, or `OUTPUT_FOLDER` (see router validation).

Implementation uses **`cli_redact.main(direct_mode_args=...)`** where a CLI task exists (same behaviour as [cli_redact.py](cli_redact.py)); `apply_review_redactions` calls [tools/simplified_api.py](tools/simplified_api.py) instead.

### 2. Gradio Client API (e.g. Hugging Face Spaces)

For remote Spaces or any Gradio deployment exposing the HTTP API:

- **Schema**: `GET https://<host>/gradio_api/info`
- **Call**: `POST https://<host>/gradio_api/call/{api_name}` with body `{"data":[...]}` (argument order matches the named endpoint’s component list).
- **Poll**: `GET https://<host>/gradio_api/call/{api_name}/{event_id}`
- **Hugging Face**: `Authorization: Bearer $HF_TOKEN`

Named `api_name` values in this app include: `redact_document`, `load_and_prepare_documents_or_data`, `apply_review_redactions`, **`doc_redact`** (simple `gr.api`: one PDF/image + optional OCR/PII knobs; returns `(output_paths, message)`; `api_name='/doc_redact'`; parameters include `document_file`, `redact_entities`, `output_dir`, `ocr_method`, `pii_method`, `allow_list`, `deny_list`, `page_min`, `page_max`, **`handwrite_signature_checkbox`** — AWS Textract extraction options such as `Extract handwriting` / `Extract signatures`), **`review_apply`** (simple `gr.api`: PDF + `*_review_file.csv`; returns `(output_paths, message)`; `api_name='/review_apply'`), **`preview_boxes`** (simple `gr.api`: PDF + `*_review_file.csv`; renders proposed boxes onto the original PDF and returns `(zip_path, message)` — use to verify coordinates *before* calling `review_apply`, no redaction applied; `api_name='/preview_boxes'`), **`pdf_summarise`** (simple `gr.api`: PDF + optional summarisation/OCR knobs; returns `(output_paths, status_message, summary_text)`; `api_name='/pdf_summarise'`), **`tabular_redact`** (simple `gr.api`: one tabular file (CSV/XLSX/Parquet/DOCX) + optional knobs; returns `(output_paths, message)`; `api_name='/tabular_redact'`), **`page_redaction_review_image`** (short review overlay export; `api_name='/page_redaction_review_image'`), **`page_ocr_review_image`** (short OCR visualisation export; `api_name='/page_ocr_review_image'`), `word_level_ocr_text_search`, `redact_data`, `find_duplicate_pages`, `find_duplicate_tabular`, `summarise_document`, `combine_review_csvs`, `combine_review_pdfs`. The matching **`POST /agent`** names for those two visual exports are `export_review_redaction_overlay` and `export_review_page_ocr_visualisation` (§1). Many endpoints require **many positional arguments** (full Gradio state); prefer the short `gr.api` routes above or **`POST /agent/apply_review_redactions`** where applicable instead of building the full `data` array from `/gradio_api/info`.

## CLI parity

For scripting and tests, `python cli_redact.py` with flags is authoritative; programmatic merges use `get_cli_default_args_dict()` in [cli_redact.py](cli_redact.py).

## Security and data handling

- Do not commit API keys, tokens, or customer data.
- Treat paths as untrusted outside validated roots (see [tools/secure_path_utils.py](tools/secure_path_utils.py)).
- Optional `instruction` / LLM fields must not be passed into shell or unconstrained config keys.

## Conventions for PRs

- Keep changes focused; avoid drive-by refactors.
- Match existing naming and patterns in [app.py](app.py) and [tools/](tools/).
- Update tests when behaviour changes; run `pytest` before merge.
