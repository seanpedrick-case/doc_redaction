---
name: doc-redact-page-review
description: "After initial redaction (doc-redaction-app): Pass 1 parallel per-page OCR/CSV review (subagents), merge, single /review_apply; optional Pass 2 VLM visual check after Pass 1 outputs."
version: 1.4.1
author: repo-maintained
license: AGPL-3.0-only
---

## Where this fits

1. **Initial redaction** — [`../doc-redaction-app/SKILL.md`](../doc-redaction-app/SKILL.md).
2. **This skill** — orchestrate **Pass 1** (parallel per-page OCR/CSV review) → merge → **one** `/review_apply`.
3. **Pass 2 (optional)** — visual VLM page checks **after** Pass 1 outputs; see [`../doc-redaction-modifications/SKILL.md`](../doc-redaction-modifications/SKILL.md) § Pass 2.
4. **Mechanics** (CSV, word/line OCR, preview, apply, VLM details) — modifications skill.

## Two-pass model (orchestration view)

| Phase | Who | Method |
|-------|-----|--------|
| **Pass 1** | Parallel child agents (one page each) + parent merge/apply | OCR line CSV, word OCR CSV, review CSV, rules, local preview — **no VLM** |
| **Pass 2** | Parent (or sequential subagents, max 1 at a time on local VLM) | Visual check of Pass 1 preview/redacted PNGs — **only if user requests** |

**Default:** complete Pass 1 and deliver. Do **not** spawn VLM subagents for every page unless explicitly asked.

## Goal

User gives a **folder** of a completed run, a **page range**, and **review rules**. Parent spawns **Pass 1 children (one page each)**. Children return JSON deltas; parent **merges** full `*_review_file.csv`, **one** `/review_apply`, text verify. Optional **Pass 2** VLM after that.

## Trigger

- “Review pages X–Y”, “check pages X through Y in [folder]”, “parallel page review”.
- Pass 2 trigger: “VLM check”, “visual review”, “verify redactions visually” — run **after** Pass 1 apply.

## Inputs

| Input | Notes |
|--------|--------|
| **Folder** | `*_review_file.csv`, `*_ocr_results_with_words*.csv`, `*_ocr_output_*.csv` (line), original PDF |
| **Page range** | e.g. `4-7` (1-based) |
| **Rules** | Removals, additions, phrases, false-positive policy, signatures |
| **App URL** | Optional; e.g. `http://host.docker.internal:7861` |
| **Output layout** | e.g. `output_review_p{N}/`, `output_review_final/` |
| **Pass 2** | Optional VLM base URL; flag pages or full doc |

## Pass 1 — Discover files

Resolve: one `*_review_file.csv`, one `*_ocr_results_with_words*.csv`, one `*_ocr_output_*.csv`, one **original** PDF.

## Pass 1 — Keep a full baseline (CRITICAL)

- Load **master** review CSV once (`utf-8-sig`). Source of truth for pages **outside** the review range.
- **Never** let a child overwrite the master with a partial CSV.

**Per-child input:** page slice only + paths to PDF, word OCR CSV, **line OCR CSV**, rules. No VLM in Pass 1 children.

```python
import csv
from pathlib import Path

def page_int(row: dict) -> int:
    return int(float(row.get("page", 0) or 0))

def rows_for_page(csv_path: Path, page: int) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if page_int(r) == page]
```

## Pass 1 — Parallel children (one page each)

- **One child per page**; batch concurrency (e.g. 3–5) for Gradio — **not** for VLM.
- Each child: review slice + word OCR + **line OCR** → apply rules → return **JSON only** (no `/review_apply`, **no VLM**).
- Absolute paths for all inputs.

Suggested JSON: `page`, `removals`, `additions`, `modifications`, or **`final_rows`** for that page only.

### Pass 1 child checklist

Follow modifications skill **Pass 1**: word OCR merge rules, line OCR for context, `image` column reuse, `utf-8-sig`, false positives, signatures, optional local preview PNG for parent spot-check (not VLM).

**Hard rule:** children **do not** call `/review_apply`, `/preview_boxes`, or VLM.

## Pass 1 — Merge (parent only)

1. Master rows for pages **not** in reviewed set → unchanged.
2. Replace each reviewed page with child **verified** row set.
3. Sort by `page`, `ymin`, `xmin`.
4. Write **`{document}_review_file_merged.csv`** (`_review_file` in basename).
5. Validate: no duplicate `id`s; boxes in [0, 1].

## Pass 1 — Preview and single `/review_apply`

1. Run **`verify_redaction_coverage`** on merged CSV; fix policy issues until `pass_strict`.
2. **Prune suspicious rows** (`--prune-suspicious` or `auto_prune_suspicious: true`); re-run until `pass_with_cleanup` (or accept cleanup debt).
3. Optional parent preview on merged/pruned CSV (`preview_redaction_boxes` or `/preview_boxes`).
4. **One** `/review_apply` — original PDF + pruned merged CSV.
5. Download; newest by `mtime`.

## Pass 1 — Verify (parent)

- Re-run **`verify_redaction_coverage`** after apply with `redacted_pdf_path` for text-layer checks.
- **Split container (Pi + separate redaction service):** pre-apply coverage on the **merged CSV downloaded locally** via CLI; post-apply coverage via **`POST /agent/verify_redaction_coverage`** with server paths from `/review_apply`. See [`../doc-redaction-modifications/SKILL.md`](../doc-redaction-modifications/SKILL.md) § Split-container verify.
- Deliver when `pass_strict` is true. **`pages_flagged_for_vlm`** = policy/visual risk → optional Pass 2. **`pages_needing_csv_cleanup`** = run prune, not VLM.

**Pass 1 deliverables:** per-page JSON/CSVs under `output_review_p{N}/`; coverage JSON; merged PDF/CSV under `output_review_final/`.

## Pass 2 — Optional visual VLM (flagged pages only)

Run only when user requests visual QA **or** Pass 1 coverage report lists `pages_flagged_for_vlm`.

1. Use Pass 1 merged CSV + original PDF (or Pass 1 `*_redacted.pdf`).
2. Render preview/redacted PNGs **only for flagged pages**.
3. **Sequential** VLM calls if using a local model (max **1 concurrent** VLM subagent).
4. Parse findings → **conservative** CSV edits (see modifications § Pass 2).
5. **One** more `/review_apply` if CSV changed → re-run coverage report.

## Error handling

| Scenario | Action |
|----------|--------|
| Pass 1 child fails | Retry once; else leave master rows for that page |
| `/review_apply` fails | PyMuPDF local apply (TROUBLESHOOTING) |
| Missing OCR | Flag in JSON; zone heuristics (modifications) |
| VLM timeout / empty response | Skip page or retry once; do not block Pass 1 delivery |

## Pitfalls

- **VLM in Pass 1 children** — expensive; defer to Pass 2.
- **Multiple `/review_apply`** within one pass — races; parent only, once per pass.
- **Partial CSV as full document** — always merge first.
- **VLM heuristic over-addition** — prefer explicit name matches; see modifications Pass 2.

## Example user prompts

**Pass 1 only (default):**
```
Review pages 4–7 in /workspace/output_redact/. One subagent per page. OCR/CSV only — no VLM.
Rules: [policy bullets]. Save merged outputs under output_review_final/
```

**Pass 1 + Pass 2:**
```
Review pages 1–56 (Pass 1: OCR/CSV subagents, merge, apply). Then Pass 2: VLM visual check
each page sequentially against output_review_final/ previews. Re-apply once if VLM finds fixes.
```

## Completion checklist

**Pass 1**
- [ ] Master CSV backed up
- [ ] One OCR/CSV child per page (no VLM)
- [ ] Merged full-document `*_review_file.csv`
- [ ] Single `/review_apply`
- [ ] Text/OCR verification

**Pass 2 (if requested)**
- [ ] VLM on Pass 1 outputs only
- [ ] Conservative CSV patches
- [ ] Single re-apply if needed
- [ ] Re-verify
