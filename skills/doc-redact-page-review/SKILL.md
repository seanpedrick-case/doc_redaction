---
name: doc-redact-page-review
description: "After initial redaction (doc-redaction-app): orchestrate parallel per-page child agents, merge CSV changes, preview once, single /review_apply. Use when the user asks to review a page range and spawn subagents."
version: 1.1.0
author: repo-maintained
license: AGPL-3.0-only
---

## Where this fits

1. **Initial redaction** — [`../doc-redaction-app/SKILL.md`](../doc-redaction-app/SKILL.md) (`/doc_redact` or `/redact_document`) and download outputs.
2. **This skill** — parallel **per-page** review with **one** merged `/review_apply` from the **parent** agent.
3. **Mechanics** (CSV columns, OCR word merge rules, `preview_redaction_boxes`, `/review_apply` snippets, VLM, PyMuPDF fallback) — [`../doc-redaction-modifications/SKILL.md`](../doc-redaction-modifications/SKILL.md).

## Goal

User gives a **folder** of a completed run, a **page range**, and **review rules**. The **parent** agent spawns **parallel child agents (one page each)**. Children return structured deltas (e.g. JSON); the parent **merges** into a **full-document** `*_review_file.csv`, runs **one** local or server **preview** if needed, then **one** `/review_apply`. Children **must not** call `/review_apply`.

## Trigger

- Phrases like “review pages X–Y”, “check pages X through Y in [folder]”, “parallel page review”.
- User supplies: output **folder**, **page range**, **rules**; optionally **app base URL** and output dir names.

## Inputs

| Input | Notes |
|--------|--------|
| **Folder** | Contains `*_review_file.csv`, `*_ocr_results_with_words*.csv` (word columns for that run), and the **original** unredacted `.pdf`. |
| **Page range** | e.g. `4-7` or `[4,5,6,7]` (1-based page indices as in the review CSV). |
| **Rules** | Bullet list: removals, additions, phrases to redact, false-positive policy, signatures, etc. |
| **App URL** | Optional; e.g. `http://host.docker.internal:7861` for local Docker. |
| **Output layout** | Optional; e.g. per-page artifacts under `output_review_p{N}/`, merged under `output_review_final/`. |

## Step 1 — Discover files

Resolve paths under the folder: one `*_review_file.csv`, one `*_ocr_results_with_words*.csv`, one **original** PDF (same document the review CSV belongs to).

## Step 1b — Keep a full baseline (CRITICAL)

- Load the **master** review CSV once (`encoding="utf-8-sig"`). Treat it as the **source of truth** for pages **outside** the review range.
- **Never** let a child overwrite the whole master with a partial CSV.

**Per-child input:** give each child **only rows for its assigned page** (a slice or small temp CSV), plus paths to the PDF and OCR CSV and the full rules list. That limits cross-page mistakes and keeps prompts small.

Normalize page numbers when filtering:

```python
import csv
from pathlib import Path

def page_int(row: dict) -> int:
    return int(float(row.get("page", 0) or 0))

def rows_for_page(csv_path: Path, page: int) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if page_int(r) == page]
```

## Step 2 — Parallel children (one page each)

Use your **agent platform’s parallel child tasks** (subagents, delegated tasks, etc.):

- **One child per page** in the range; run up to **N** at a time (e.g. **3–5**) so Gradio is not overloaded; queue the rest.
- Each child: **read** slice + OCR for that page → apply rules → return **JSON only** (no `/review_apply`).
- Pass **absolute paths**; children do not share cwd unless your platform guarantees it.

Suggested JSON shape (adapt if needed): `page`, `removals` (ids), `additions` (full row dicts), `modifications` (id + field patches), or **`final_rows`** for that page only—parent must still merge with **unreviewed** pages from master.

## Step 3 — Child checklist (delegate to modifications rules)

Children should follow [`../doc-redaction-modifications/SKILL.md`](../doc-redaction-modifications/SKILL.md): word-level OCR (`word_x0`…), same-line merge vs split lines, `image` column reuse, `utf-8-sig`, false positives, signatures.

**Hard rule:** children **do not** call `/review_apply` or `/preview_boxes` unless you intentionally centralize preview on the parent (recommended: **parent** runs `preview_redaction_boxes` on the **merged** CSV once).

## Step 4 — Merge (parent only)

1. Start from **master** rows where `page` **not in** the reviewed set → copy unchanged.
2. For each reviewed page, replace with that page’s **verified** row set from the child (after applying removals/additions/patches in order).
3. Sort rows if the app expects stable ordering (e.g. by `page`, then `ymin`, then `xmin`).
4. Write **`{document}_review_file_merged.csv`** (basename must still contain **`_review_file`** for `/review_apply`).
5. Optionally validate: no duplicate `id` collisions across pages; every row has valid `xmin`…`ymax` in [0, 1].

**Invariant:** merged file = **all pages** present → one `/review_apply` updates the **entire** PDF consistently.

## Step 5 — Preview and single `/review_apply`

1. Parent runs local preview (see modifications) on **original PDF + merged CSV** for spot-check pages if needed.
2. **One** `gradio_client` `predict(..., api_name="/review_apply")` with **original PDF** + merged CSV.
3. Download outputs; **sort by `mtime`** for `*_redacted.pdf` / `*_review_file.csv` (see modifications).

## Step 6 — Verify (parent)

Minimum: **text extraction** on redacted PDF for rules-driven strings; **page count** vs original; optional **pixel sample** at box centers (expect near-black). Optional VLM: see modifications. On failure: PyMuPDF fallback per modifications / TROUBLESHOOTING.

## Step 7 — Deliverables (convention)

Adjust to user request; typical layout:

- Optional per-page notes or PNGs: `output_review_p{N}/`
- Merged artifacts: `output_review_final/` (redacted PDF, updated review CSV, overlay PDF if produced)

## Error handling

| Scenario | Action |
|----------|--------|
| Child fails | Retry that page once; if still failing, log and either skip page (leave master rows) or escalate. |
| `/review_apply` fails / host down | PyMuPDF local apply from merged CSV (modifications / TROUBLESHOOTING). |
| Missing OCR for a page | Child flags in JSON; parent may use grid/zone heuristics from modifications. |

## Pitfalls

- **Multiple `/review_apply` calls** (e.g. one per child) → races, inconsistent hashes, wasted GPU; **parent only, once.**
- **Partial CSV uploaded** as if it were the full document → unreviewed pages lose boxes. Always merge to full CSV first.
- **Concurrency**: batch pages if the queue or GPU thrashes.
- **Trust but verify** child JSON against OCR before merge.

## Example user prompt

```
Review pages 4–7 of outputs in /workspace/output_redact_sig/ for Partnership-Agreement-Toolkit_0_0.pdf. Spawn one subagent per page in parallel. Rules:
1. Remove boxes for general country names
2. Remove redactions for [name] per policy
3. Tune box sizes vs OCR word boxes
4. Add SIGNATURE rows where missing
5. Redact London and Sister City / Sister Cities per word-OCR rules
6. No false positives; no missed PII

Save merged outputs under output_review_final/
```

## Completion checklist

- [ ] Master CSV backed up or reproducible from folder
- [ ] One child per reviewed page (or explicit batching)
- [ ] Merged full-document `*_review_file.csv` with `_review_file` in name
- [ ] Preview (if used) on merged file before apply
- [ ] Single `/review_apply` (or documented PyMuPDF fallback)
- [ ] Verification + newest artifacts by mtime
