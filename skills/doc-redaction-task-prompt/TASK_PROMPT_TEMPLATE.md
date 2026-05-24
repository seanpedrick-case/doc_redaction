# Document redaction task prompt (template)

Copy everything below the line into a new agent chat. Replace `{placeholders}` in the **Setup** section, then fill in **User redaction requirements** at the end with your case-specific policy.

Skills for the agent: [`doc-redaction-app`](../doc-redaction-app/SKILL.md) → [`doc-redaction-modifications`](../doc-redaction-modifications/SKILL.md) (Pass 1) → optional Pass 2; parallel Pass 1 pages → [`doc-redact-page-review`](../doc-redact-page-review/SKILL.md).

---

## Setup (fill in before sending)

| Placeholder | Example |
|-------------|---------|
| `{FILE_NAME}` | `20211004 Cora Fyller UNREDACTED 0507.pdf` |
| `{INPUT_PATH}` | `workspace/{FILE_NAME}` |
| `{OUTPUT_BASE}` | `workspace/redact/{FILE_NAME}/` |
| `{GRADIO_URL}` | `http://host.docker.internal:7861` |
| `{PAGE_RANGE}` | `all` or `1-56` |

---

## Agent task (fixed workflow — do not skip)

Redact and review **`{FILE_NAME}`** from **`{INPUT_PATH}`**.

### Two-pass model (Pass 1 is the deliverable)

**Default: Pass 1 only.** Pass 1 must be sufficient for delivery unless Pass 2 criteria below are met after Pass 1 completes.

**Do not run VLM on every page.** Do not spawn per-page VLM subagents unless I explicitly request Pass 2 or Pass 1 leaves pages in `pages_flagged_for_vlm`.

---

### Pass 1 (required — complete end-to-end)

1. **Initial redaction** — `POST /doc_redact` (or Gradio `api_name="/doc_redact"`) with settings from **User redaction requirements** below. Save artifacts to **`{OUTPUT_BASE}output_redact/`**.

2. **Review all pages in Pass 1** using **OCR / CSV / text only** (no VLM):
   - Load `*_review_file.csv`, `*_ocr_results_with_words_*.csv`, `*_ocr_output_*.csv`, original PDF from the same run.
   - Apply **User redaction requirements** (must redact / must not redact) programmatically to the review CSV.
   - Align missing boxes from word OCR (merge same-line tokens; separate boxes across lines).
   - Run **`verify_redaction_coverage`** (CLI or `POST /agent/verify_redaction_coverage`) with `must_redact` / `must_not_redact` regex lists derived from user requirements.
   - Fix policy issues until **`pass_strict: true`** (`uncovered_terms`, `over_redacted`, `text_layer_leaks` cleared).
   - **Prune suspicious rows** — short OCR fragments that do not match `must_redact` (`auto_prune_suspicious: true` or `--prune-suspicious`). Re-run coverage; target **`pass_with_cleanup: true`**.
   - Optional: `/preview_boxes` on highest-risk pages only (not every page).

3. **Single apply** — **one** `/review_apply` from the parent agent (original PDF + merged/pruned CSV). Download newest outputs by `st_mtime` to **`{OUTPUT_BASE}review/output_review_final/`**.

4. **Post-apply verification**
   - Re-run `verify_redaction_coverage` with `redacted_pdf_path`.
   - Optional term search (`POST /agent/word_level_ocr_text_search`) for key names from user requirements.

5. **Pass 1 completion criteria**
   - Post-apply coverage: **`pass_strict: true`**
   - Practitioner / allow-list names not over-redacted (per user requirements)
   - Write a brief summary markdown under **`{OUTPUT_BASE}review/`** (what was done, coverage results, any pages still needing optional Pass 2)

---

### Pass 2 (optional — strict gate)

**Do not start Pass 2 unless one of the following is true:**

| Criterion | Action |
|-----------|--------|
| I explicitly ask for visual / VLM review | Run Pass 2 only on pages or range I specify, or on `pages_flagged_for_vlm` if I say “flagged pages only” |
| Post-apply coverage lists **`pages_flagged_for_vlm`** | VLM **only those pages** (sequential, max 1 concurrent on local VLM) |
| Page has **`uncovered_terms`** for must-redact regex after Pass 1 fixes | Targeted Pass 2 on that page |
| Page has **`text_layer_leaks`** or **`pixel_failures`** | Targeted Pass 2 on that page |
| Handwriting, stamps, signatures, or ink **absent from word OCR** and suspected to contain policy PII | Targeted Pass 2 on that page after noting why in the summary |

**Do not run Pass 2 for:**

- **`pages_needing_csv_cleanup` alone** — fix with suspicious-row prune, not VLM
- Suspicious short OCR rows (`"-"`, `"."`, `"Ho"`, etc.)
- “Review every page visually for completeness” on large documents
- Bulk-adding every OCR token from VLM output (conservative CSV edits only)

If Pass 2 runs: render PNGs for flagged pages only → one VLM call per page → conservative CSV patch → **at most one** additional `/review_apply` → re-run Pass 1 coverage.

---

### Technical constraints

- Gradio: **`{GRADIO_URL}`**
- Page scope: **`{PAGE_RANGE}`**
- Review CSV basename must contain `_review_file`
- CSV encoding: **`utf-8-sig`**
- Reuse same-page `image` column value when adding rows
- Long `httpx` read timeout for large PDFs (e.g. 1800s+)
- Human review of redacted material is still assumed downstream

---

## User redaction requirements (fill in — authoritative for this task)

> **Instructions for the user:** Replace the placeholder bullets below with your case-specific policy. The agent must treat this section as the source of truth for what to redact, what to leave visible, and any run settings not covered above.

### Document and scope

- **File:** `{FILE_NAME}`
- **Pages to process:** `{PAGE_RANGE}`
- **Output folder:** `{OUTPUT_BASE}`

### OCR and PII detection

- **OCR method:** _(e.g. `hybrid-paddle-inference-server`, fallback `paddle`)_
- **PII method:** _(e.g. `Local`)_
- **Other run options:** _(e.g. efficient OCR off, no batching, custom entities — or “defaults”)_

### Must redact (family / sensitive — add boxes if missing)

_List every name, relationship, address pattern, phone/postcode pattern, or other PII that must be blacked out. Be explicit about spelling variants and OCR fragments._

- _(e.g. Cora, Stephen, Peter, Rhett, Yazmin, Elliot)_
- _(e.g. surname variants: Fuller, Fyller)_
- _(e.g. Romola)_
- _(e.g. family addresses, phone numbers, postcodes)_
- _(add rows as needed)_

### Must NOT redact (practitioners / allow list — remove false-positive boxes)

_List professionals, organisations, and generic terms that must remain visible unless they are also in “must redact”._

- _(e.g. Dr / Doctor / social worker / nurse titles when not family)_
- _(e.g. named practitioners: Macrae Gibson, Syred, Allsop, Wright, …)_
- _(e.g. Lambeth Council, borough, department names when not personal data)_
- _(add rows as needed)_

### Coverage regex hints (optional — agent may derive from lists above)

- **`must_redact` patterns:** _(e.g. `\b(cora|stephen|fuller|fyller)\b`)_
- **`must_not_redact` patterns:** _(e.g. `\bdr\.?\b|\bmacrae\b|\bgibson\b`)_

### Signatures, handwriting, and non-OCR content

- _(e.g. redact handwritten signatures near “Signed”; add CUSTOM boxes if OCR blind)_
- _(e.g. skip Pass 2 unless flagged)_

### Pass 2 preference

- _(e.g. “Pass 1 only — do not run VLM unless I ask”)_
- _(e.g. “Run Pass 2 only on pages_flagged_for_vlm after Pass 1”)_

### Additional instructions

_(Any other constraints: duplicate pages, specific pages to spot-check, files not to commit, delivery format, etc.)_

---

## Example (minimal filled user section)

```markdown
### Must redact
- Cora, Stephen, Peter, Rhett, Yazmin, Elliot; Fuller/Fyller; Romola
- Phone numbers and UK postcodes associated with the family

### Must NOT redact
- Practitioners: Dr, doctors, social workers, Macrae, Gibson, Syred, Allsop, Wright
- Generic org text: Lambeth Council, borough, department (unless containing family PII)

### OCR and PII
- OCR: hybrid-paddle-inference-server (fallback paddle)
- PII: Local

### Pass 2 preference
- Pass 1 only unless pages_flagged_for_vlm remain after prune and post-apply coverage
```
