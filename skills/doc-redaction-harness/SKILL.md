---
name: doc-redaction-harness
description: "Agentic redaction inside an AWS AgentCore Harness invoked by the doc_redaction app: first-pass redaction, Pass 1 CSV review (optionally parallel per-page subagents via presigned S3 URLs), coverage QA and a single review_apply, all through the app's inline tools."
version: 1.0.0
author: repo-maintained
license: AGPL-3.0-only
---

# doc-redaction-harness

Use this skill when the task prompt says you are running in an **AgentCore Harness invoked by the
doc_redaction app** (the "Redact with AI agent" button or the `agentic_redact` API). It replaces
the Gradio-client mechanics of `doc-redaction-app`, `doc-redact-page-review` and
`doc-redaction-modifications`: the *quality rules* in those skills still apply, the *transport*
does not.

## Ground rules

- The app's redaction functions are **inline tools**. Calling one pauses your turn; the app runs it
  and returns a JSON result. There is no app URL: never use `gradio_client`, `/gradio_api/*`, or
  curl to the app.
- Your shell / filesystem is a **separate sandbox**. The app's files (source PDF, review CSV, OCR
  CSVs, outputs) are only reachable via inline tools, or via presigned S3 URLs that the tools
  return.
- Every path is **relative to the session folder**. Start with `list_session_files`.
- **Subagents cannot call inline tools.** Only the parent agent calls them; subagents work with
  presigned URLs and `curl`.
- The user requirements at the end of the task prompt are **authoritative**.

## Inline tools

| Tool | Use |
|---|---|
| `list_session_files` | Discover inputs and outputs (relative paths). |
| `doc_redact` | First-pass redaction of the source document. App defaults (OCR, PII method, entities, allow/deny lists, page range) apply; pass only overrides the instructions require. Writes to `output_redact/`. |
| `read_session_text` | Read CSV / text in chunks (`offset`, `max_chars`); follow `next_offset` until it is `null`. |
| `write_session_text` | Write a CSV / text file. Review CSVs are validated (bbox floats in `[0, 1]`, colour `"(R, G, B)"`); invalid content is rejected with the reason. Cannot overwrite the source document. |
| `verify_redaction_coverage` | Pass 1 gate: `must_redact` / `must_not_redact` regex lists; optional `redacted_pdf_path` after apply; optional `auto_prune_suspicious`. |
| `preview_redaction_boxes` | Render proposed boxes on the original PDF (no redaction). Returns PNG paths (and presigned URLs when staging is configured). |
| `stage_page_review_bundles` | Upload per-page slices (review rows, word OCR, line OCR) and return GET URLs plus one PUT URL per page for subagents. |
| `merge_page_reviews` | Download each page's uploaded result, validate it and merge it into `<review>_merged.csv`. Pages with no upload keep their master rows. |
| `review_apply` | Apply a review CSV to the original PDF. Writes to `review/output_review_final/`. |
| `finish_task` | Call **exactly once, last**, with a short summary and `final_files` (deliverables plus Pass 1 review aids: `*_redactions_for_review*.pdf`, `*_ocr_results_with_words*.csv` when present). |

## Workflow

1. **Discover**: `list_session_files`. The source document is named in the task prompt.
2. **First pass**: call `doc_redact` once. Extend `deny_list` / `redact_entities` only when the user
   requirements clearly call for it (for example, named people to always redact). The result lists
   `review_csv_path`, `ocr_words_csv_path` and `ocr_lines_csv_path`.
3. **Pass 1 review**: choose one of the modes below. Never apply per page.
4. **QA gate**: `verify_redaction_coverage` on the edited review CSV with the policy regexes
   derived from the user requirements (`must_redact`) and things that must stay visible
   (`must_not_redact`). Fix the CSV until `pass_strict` is true, or until the remaining items are
   explained false alarms. `auto_prune_suspicious` can drop obvious OCR-fragment rows.
5. **Optional preview**: `preview_redaction_boxes` for the pages you changed most.
6. **Apply once**: `review_apply` with the final review CSV (basename must contain `_review_file`;
   `*_review_file_merged.csv` is fine).
7. **Post-apply check**: `verify_redaction_coverage` with `redacted_pdf_path` from the apply
   result. Text-layer leaks mean a box is missing or misplaced: fix the CSV, then apply again (at
   most one more time).
8. **Finish**: `finish_task` with a 3–6 line summary (what was redacted, what was kept, any pages
   needing human review) and `final_files`: post-apply `*_redacted.pdf`, final `*_review_file*.csv`, and
   when they exist the Pass 1 review overlay `output_redact/*_redactions_for_review*.pdf` and
   `output_redact/*_ocr_results_with_words*.csv` so the user can spot-check boxes and OCR.

### Pass 1, mode A: single agent (short documents, or page review unavailable)

- `read_session_text` the review CSV (all chunks) and the line OCR CSV for context.
- Edit rows to match the user requirements: remove false positives, add missing rows, fix labels.
- `write_session_text` the edited CSV **next to the original** as `<stem>_edited.csv`, keeping
  `_review_file` in the name. Keep the full document: rows for pages you did not touch stay as
  they were.

### Pass 1, mode B: parallel page review with subagents

Use this for documents of more than about 5 pages with dense PII, when `stage_page_review_bundles`
succeeds. If it returns an error saying page review is disabled, use mode A.

1. Parent: `stage_page_review_bundles` (optionally `pages=[...]` to limit review to pages with
   rows or with likely misses).
2. Parent: spawn **one `subagent` per page** (batch about 4–8 at a time). Give each one:
   - the page number, the user requirements, and the `review_csv_columns`;
   - its `review_rows_url`, `ocr_words_url`, `ocr_lines_url` and `result_put_url`;
   - the child instructions below.
3. Parent: when all subagents report back, call `merge_page_reviews` (same `pages`). Check
   `pages_without_upload_kept_master_rows`; re-run those pages once or review them yourself.
4. Continue with the QA gate (step 4) on `merged_review_csv_path`.

**Child instructions (copy into each subagent task):**

```text
You review ONE page of a redaction review CSV. You cannot call app tools; use the shell.
1. Download:  curl -fsSL '<review_rows_url>' -o review_rows.csv
              curl -fsSL '<ocr_words_url>'  -o ocr_words.csv
              curl -fsSL '<ocr_lines_url>'  -o ocr_lines.csv   (if provided)
2. Apply the user requirements to THIS page only:
   - delete rows that are false positives (kept-visible terms, job titles, OCR fragments);
   - add rows for missed items using word-level OCR boxes from ocr_words.csv;
   - keep every column; page = <N>; reuse this page's existing 'image' value;
     color "(0, 0, 0)"; leave id empty for new rows (the app fills it);
   - xmin, ymin, xmax, ymax are floats normalised to 0-1 (never points or pixels).
3. Write the complete final row set for the page to result_review.csv (UTF-8, header row).
   Zero rows is valid when nothing on the page should be redacted.
4. Upload:  curl -fsS -X PUT -H 'Content-Type: text/csv' --upload-file result_review.csv '<result_put_url>'
5. Reply with: page number, rows added, rows removed, anything uncertain.
```

Presigned URLs expire (see `expires_in_seconds`); if uploads fail with 403 after a long run, call
`stage_page_review_bundles` again for the affected pages.

## Review CSV rules (from doc-redaction-modifications)

- Header: `image,page,label,color,xmin,ymin,xmax,ymax,text,id`.
- Boxes are **normalised 0–1**. Word OCR rows from `*_ocr_results_with_words*.csv` are the best
  source for precise boxes; line OCR (`*_ocr_output_*.csv`) gives reading-order context.
- New rows reuse an existing `image` value for the same page; `color` is `"(0, 0, 0)"`.
- Signatures are rarely caught by PII models: anchor `SIGNATURE` boxes near "Signed" / printed
  names; typed names are separate `PERSON` rows.
- Watch for false positives: places or organisations labelled `PERSON`, bare job titles, OCR
  fragments.

## Failure handling

- If a tool result is an error (`{"error": "..."}`), read it, fix the arguments and retry once. Do not loop on
  the same failing call.
- If `doc_redact` fails, report it via `finish_task` (empty `final_files`) and do not continue.
- Do not invent paths: use paths returned by tools or by `list_session_files`.
