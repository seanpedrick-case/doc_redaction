# AGENTS.md

Context for AI coding agents working on **doc_redaction** (PII redaction for PDFs, images, Word, and tabular files). Human-oriented docs: [README.md](README.md). User guide: [doc_redaction user guide](https://seanpedrick-case.github.io/doc_redaction/src/user_guide.html).

## Project overview

- **Stack**: Python 3.10+, Gradio UI ([app.py](app.py)), optional FastAPI when `RUN_FASTAPI` is enabled, AWS/LLM integrations via [tools/config.py](tools/config.py) and env files under `config/`.
- **License**: AGPL-3.0-only (see [pyproject.toml](pyproject.toml)). Respect license terms when adding dependencies.
- **Accuracy**: Outputs are not guaranteed complete; downstream use should assume **human review** of redacted material.

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

## Agentic / programmatic access (two surfaces)

### 1. FastAPI Agent API (recommended for LLM agents: small JSON bodies)

When `RUN_FASTAPI` is true, routes are mounted under **`/agent`** ([agent_routes.py](agent_routes.py)).

- **Catalog**: `GET /agent/operations` — maps each Gradio `api_name` to an HTTP path and notes whether the route is implemented via CLI or returns HTTP 501 for Gradio-only flows.
- **Implemented POST routes** (names align with `api_name` in [app.py](app.py)):  
  `redact_document`, `redact_data`, `find_duplicate_pages`, `find_duplicate_tabular`, `summarise_document`, `combine_review_pdfs`, `combine_review_csvs`, `export_review_redaction_overlay`, `export_review_page_ocr_visualisation`.
- **Gradio-only stubs** (501 + JSON hint): `load_and_prepare_documents_or_data`, `apply_review_redactions`, `word_level_ocr_text_search`.
- **Auth**: If `AGENT_API_KEY` is set in the environment, send header `X-Agent-API-Key` with that value.
- **Paths**: Inputs must resolve to files under the repo root, `INPUT_FOLDER`, or `OUTPUT_FOLDER` (see router validation).

Implementation uses **`cli_redact.main(direct_mode_args=...)`** where a CLI task exists; same behaviour as [cli_redact.py](cli_redact.py).

### 2. Gradio Client API (e.g. Hugging Face Spaces)

For remote Spaces or any Gradio deployment exposing the HTTP API:

- **Schema**: `GET https://<host>/gradio_api/info`
- **Call**: `POST https://<host>/gradio_api/call/{api_name}` with body `{"data":[...]}` (argument order matches the named endpoint’s component list).
- **Poll**: `GET https://<host>/gradio_api/call/{api_name}/{event_id}`
- **Hugging Face**: `Authorization: Bearer $HF_TOKEN`

Named `api_name` values in this app include: `redact_document`, `load_and_prepare_documents_or_data`, `apply_review_redactions`, `word_level_ocr_text_search`, `redact_data`, `find_duplicate_pages`, `find_duplicate_tabular`, `summarise_document`, `combine_review_csvs`, `combine_review_pdfs`, `export_review_redaction_overlay`, `export_review_page_ocr_visualisation`. Many endpoints require **many positional arguments** (full Gradio state); prefer the **FastAPI `/agent`** routes for agent workflows unless you are generating the full `data` array from `/gradio_api/info`.

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
