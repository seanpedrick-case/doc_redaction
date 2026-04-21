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
| `combined_case_notes` | `redact completely` |