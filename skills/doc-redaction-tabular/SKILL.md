---
name: doc-redaction-tabular
description: "Skill for redacting tabular files (CSV/XLSX/Parquet/DOCX) using the Document Redaction app. gradio_client-first with discovery via view_api()."
version: 1.4.0
author: repo-maintained
license: AGPL-3.0-only
changelog:
  - "v1.4.0 (Apr 21, 2026): Added /find_duplicate_tabular docs with handle_file caveat. Clarified HF Space '0 entities' quirk.
Added Excel multi-sheet handling. Output file naming convention section expanded strategy
comparison. Added error handling patterns and edge case notes."
---

## Purpose

Redact **tabular and semi-tabular** files using the app:
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- Parquet (`.parquet`)
- Word (`.docx`)

Registered endpoints:
- **`/redact_data`** — main tabular redaction
- **`/find_duplicate_tabular`** — detect and remove duplicate rows

## When to use this skill

Use when the document is **not a PDF/image** and you want redaction applied directly to
a table-like file.
For PDFs/images, use `doc-redaction-app` with `api_name="/redact_document"` instead.

## Decision tree (recommended)

1. **Try `gradio_client` first** — confirmed working for both endpoints.
2. **`docker cp` + raw Gradio HTTP** — fallback if gradio_client fails.
3. **Raw Gradio HTTP API** — last resort: `/gradio_api/upload` +
   `/gradio_api/call/...` + poll.

## Tabular Redaction (`/redact_data`)

### Quick example (CSV)

```python
from gradio_client import Client, handle_file
import os

BASE_URL = "http://host.docker.internal:7861"
client = Client(BASE_URL)
csv_path = "/path/to/table.csv"
output_folder = "/home/user/app/output/"   # CONTAINER path

wrapped_files = [handle_file(csv_path)]

result = client.predict(
    file_paths=wrapped_files,
    in_text="",                               # required str
    anon_strategy="replace with 'REDACTED'",  # see strategies
    chosen_cols=["Name", "Email"],            # columns to anonymize
    chosen_redact_entities=[
        "PERSON", "PHONE_NUMBER",
        "EMAIL_ADDRESS"
    ],
    in_allow_list=[],
    latest_file_completed=0,                  # 0 for single file
    out_message="Tabular redaction test",     # required str
    in_excel_sheets=[],                       # not Excel; see multi-sheet below
    first_loop_state=True,                    # always True for fresh run
    output_folder=output_folder,
    in_deny_list=[],
    max_fuzzy_spelling_mistakes_num=0,
    pii_identification_method="Local",
    chosen_redact_comprehend_entities=[],
    aws_access_key_textbox="",
    aws_secret_key_textbox="",
    do_initial_clean=True,
    language="en",
    progress="",
    custom_llm_instructions=["PERSON_NAME"],
    api_name="/redact_data"
)

# Returns 4 outputs: (str status, list[filepath], int count, list[filepath])
print(f"Status: {result[0]}")
print(f"Output file: {result[1][0] if result[1] else 'N/A'}")
print(f"Files redacted: {result[2]}")
```

### Excel multi-sheet handling

To discover sheet names, read the file locally first:
```python
import pandas as pd
sheets = pd.ExcelFile(csv_path).sheet_names
# e.g. ['Sheet1', 'Q1_Data', 'Summary']
```

Pass discovered names to `in_excel_sheets`:
```python
result = client.predict(
    ...,
    in_excel_sheets=["Sheet1", "Q1_Data"],  # sheet name strings
    ...
)
```

If the file has no sheets or the list is empty, the app processes all available sheets.

### HF Space (Hugging Face)

Public deployment: `https://seanpedrickcase-document-redaction.hf.space`
Same API. Key differences from local Docker:
- No "Local Inference Server" option
- No VLM face/signature entities
- Stricter file validation
- ~2–3× slower (free-tier CPU)

#### HF Space output access

`result[1]` and `result[3]` are **local temp paths** that
`gradio_client` already downloaded. Read directly:
```python
output_file = result[1][0]
with open(output_file, "r", encoding="utf-8-sig") as f:
    print(f.readline().strip())   # first row (may have BOM)
```

> **`client.download_file()` (singular) does NOT exist** on some
> `gradio_client` versions. Use the output paths directly.

#### HF Space "0 entities" quirk

On HF Space the entity count (`result[2]`) may show `0` even when
redaction is applied correctly. **Always verify by reading the
output file**, not the log count. This is a known display issue
with the spaCy model on HF.

### Container path caveat (Docker)

| Path | Works? |
|------|--------|
| `/home/user/app/output/` | Yes — container-internal, app's OUTPUT_FOLDER
| `/tmp/test_data/output/` (host) | No — server runs inside the container

**Workaround**: Use container paths. Retrieve files via `docker cp`:
```bash
# Find the file
docker exec doc_redaction-redaction-app-llama-1 \
  ls -lt /home/user/app/output/ | grep combined_case_notes

# Download
docker cp doc_redaction-redaction-app-llama-1:/home/user/app/output/combined_case_notes_anon_redact_replace.csv ./output.csv
```

## Anonymization strategies (all 4 tested)

| Strategy | Behavior | Example ("Jane Smith") |
|----------|----------|------------------------|
| `replace with 'REDACTED'` | Replaces with literal text `REDACTED` | `REDACTED` (8 chars)
| `redact completely` | Removes content (empty cell) | *(blank)*
| `mask` | Replaces with asterisks matching original length | `**********` (10 chars)
| `hash` | SHA-256 hash of original value (consistent per entity) | `ca85b082d2e6...` (94 chars)

### Choosing a strategy — use case guide

| Use case | Recommended | Why |
|----------|-------------|-----|
| GDPR compliance | `redact completely` | Maximum privacy, no length leakage |
| Audit trail | `replace with 'REDACTED'` | Clear indication of redaction occurred |
| Data science (preserve structure) | `mask` or `hash` | Keeps row/column dimensions intact |
| Cross-row correlation analysis | `hash` | Same entity → same hash across all rows |
| Legal review (need to know what was there) | `replace with 'REDACTED'` + log | Redaction text + decision log |
| Machine learning (feature engineering) | `mask` or `hash` | Preserves data shape for pipelines |
| Maximum privacy (no length info leaked) | `redact completely` | No clue about original value length |

## Key parameters

| Parameter | Type | Notes |
|-----------|------|-------|
| `chosen_cols` | list[str] | Column names to anonymize. If empty, all columns with PII are processed. Use exact column header names from the CSV/XLSX. |
| `chosen_redact_entities` | list[str] | Entity types for spaCy NER: `PERSON`, `PHONE_NUMBER`, `EMAIL_ADDRESS`, `STREETNAME`, `UKPOSTCODE` |
| `max_fuzzy_spelling_mistakes_num` | int | 0 = exact match. 1–2 allows fuzzy matching (useful for typos in PII). |
| `do_initial_clean` | bool | True: strips whitespace, normalizes text. Recommended for dirty data. |
| `pii_identification_method` | str | `"Local"` (spaCy, no API keys) or `"AWS Comprehend"` (requires AWS creds). |
| `in_excel_sheets` | list[str] | For XLSX files: specific sheet names to process. Empty = all sheets. |
| `custom_llm_instructions` | list[str] | Entity types for LLM fallback (e.g., `["PERSON_NAME", "EMAIL_ADDRESS"]`). |
| `language` | str | `"en"`, `"fr"`, `"de"`, etc. Affects spaCy model language settings. |
| `latest_file_completed` | float | `0` for single file, `1.0` when last of a batch (for multi-file runs). |

## spaCy NER limitation

The default `Local` PII detection uses spaCy's English model. It **may miss short
names with initials** like "Alex D." — tested on `combined_case_notes.csv` (18 rows)
showed only "Jane Smith" (10 chars, full name) was detected in the Social Worker column,
but "Alex D." (7 chars, with initial) in the Client column was not. This is a known
spaCy limitation with short/abbreviated names.

**Workarounds:**
- `max_fuzzy_spelling_mistakes_num=1` for partial matches (won't help with initials)
- `custom_llm_instructions` with entity types for LLM-based fallback detection (requires external LLM)
- `in_deny_list` with known names to exclude from redaction (e.g., your own company name)
- `do_initial_clean=False` if cleaning is removing useful context

## Output file naming convention

The app generates files with predictable naming:
```
{original_name}_anon_{strategy}.csv
{original_name}_anon_{strategy}.csv_log.csv   (decision log)
```

| Original | Strategy | Output file |
|----------|----------|-------------|
| `combined_case_notes` | `replace with 'REDACTED'` | `combined_case_notes_anon_redact_replace.csv`
| `combined_case_notes` | `redact completely` | `combined_case_notes_anon_redact_remove.csv`
| `combined_case_notes` | `mask` | `combined_case_notes_anon_mask.csv`
| `combined_case_notes` | `hash` | `combined_case_notes_anon_hash.csv`

## Retrieving output files (Docker)

Outputs are written under the container's `OUTPUT_FOLDER`. Retrieve via:

```bash
# Find the file in container
docker exec doc_redaction-redaction-app-llama-1 \
  ls -lt /home/user/app/output/ | grep combined_case_notes | head -10
```

```bash
# Download the redacted CSV
docker cp doc_redaction-redaction-app-llama-1:/home/user/app/output/combined_case_notes_anon_redact_replace.csv ./output.csv
```

## Log file format (decision log)

The `_log.csv` file contains one row per PII entity detected:
```csv
entity_type,start,end,data_row,column,entity
PERSON,0,10,0,Social Worker,Jane Smith
```

- `entity_type` — spaCy entity label (PERSON, PHONE_NUMBER, etc.)
- `start,end` — character offsets in the cell value
- `data_row` — 0-indexed row number
- `column` — column name
- `entity` — the detected PII text

**Read with `encoding="utf-8-sig"` in Python** to strip the BOM
(`\ufeff`) that may appear at the start of CSV files.

### HF Space log caveat

On HF Space the log file path is returned in `result[3]` (not `result[2]`).
The entity count (`result[2]`) may show `0` even when redaction IS applied
correctly. **Always verify by reading the output file**, not the log count.

## Deployment variants

### HF Space (Hugging Face)

Public deployment: `https://seanpedrickcase-document-redaction.hf.space`
Same API. Key differences from local Docker:

| Aspect | Local Docker | HF Space |
|--------|-------------|----------|
| PII detection | Local, **Local Inference Server**, AWS Comprehend | Local, AWS Comprehend (no inference server) |
| OCR models | tesseract, paddle, **hybrid-paddle-inference-server**, inference-server | tesseract, paddle only |
| VLM entities | CUSTOM_VLM_FACES, CUSTOM_VLM_SIGNATURE available | **NOT** available (no GPU/VLM support) |
| efficient_ocr default | True | False (saves compute on free tier) |
| Speed | ~1.5s per request (GPU machine) | ~3–4.5s per request (~2–3× slower, CPU free tier) |
| File validation | Accepts any file type via API | Strict: only `.pdf, .jpg, .png, .json, .zip` for `/redact_document` |
| Output access | `docker cp` from container | Read `gradio_client` output paths directly (already cached locally) |

### Speed comparison (tested Apr 21, 2026)

| Strategy | Local Docker | HF Space |
|----------|-------------|----------|
| replace with REDACTED | ~1.5s | ~3.5s |
| redact completely | ~1.5s | ~3.4s |
| mask | ~1.5s | ~3.6s |
| hash | ~1.5s | ~4.5s |

## Error handling patterns

### Common errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `Cannot save file into a non-existent directory` | `output_folder` path doesn't exist in container | Use `/home/user/app/output/` or ensure the directory exists |
| `'meta' field must be explicitly provided. | Raw string passed to `files=` | Wrap with `handle_file(path)` |
| `Invalid file type. Please upload. | CSVs rejected by `/redact_document | Use `/redact_data` for tabular files |
| Timeout after ~3s | HF Space free tier spinning down | Accept ~3.5s runtime. No fix. |
| Entity count shows 0 | HF Space display issue | Read output file to verify. |

### Defensive coding pattern

```python
import os
from gradio_client import Client, handle_file

def safe_redact(client, csv_path, strategy="replace with 'REDACTED'"):
    """Redact a CSV file with error handling."""
    wrapped = [handle_file(csv_path)]
    
    result = client.predict(
        file_paths=wrapped,
        in_text="",
        anon_strategy=strategy,
        chosen_cols=["Name", "Email"],  # adjust to your columns
        out_message="Tabular redaction test",
        output_folder="/home/user/app/output/",
        api_name="/redact_data"
    )
    
    if not result[1]:
        raise RuntimeError("No output file returned")
    
    return result

# Usage with error handling
try:
    result = safe_redact(client, "/path/to/table.csv", "mask")
    print(f"Output: {result[1][0]}")
except Exception as e:
    print(f"Redaction failed: {e}")
```

## Edge cases & gotchas

- **CSV BOM**: CSV files may have a UTF-8 BOM (`\ufeff`) at the start. Read with `encoding="utf-8-sig"` in Python.
- **Empty columns**: If a column has no PII detected, it's left unchanged — verify your `chosen_cols` match actual headers.
- **Excel sheet names with special chars**: Use exact sheet names as they appear (e.g., `"Q1 '23 Data"`).
- **Large files**: spaCy NER on large CSVs can be slow. Test with a subset first.
- **DOCX tables**: Only tabular content in DOCX is redacted — body text paragraphs are NOT processed by `/redact_data`. Use `/redact_document` for full DOCX processing.
- **Parquet encoding**: Ensure Parquet files use UTF-8 string columns. Non-string types may be skipped by spaCy.
- **Multi-file batches**: Set `latest_file_completed=1.0` only on the last file in a batch to trigger final processing.
- **HF Space cold starts**: HF Space may take 10-30s on first call after idle. Subsequent calls are ~3-4s. |