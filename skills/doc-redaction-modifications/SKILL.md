---
name: doc-redaction-modifications
description: "Review and reapply: two-pass workflow — Pass 1 (OCR/CSV/text, default) then optional Pass 2 (VLM per page). Edit *_review_file.csv, preview, /review_apply, verify. Parallel page orchestration → doc-redact-page-review. Initial redaction → doc-redaction-app."
version: 2.5.2
author: repo-maintained
license: AGPL-3.0-only
---

## Goal

Repeatable **review → edit CSV → preview → `/review_apply` → download → verify** when you already have the **original PDF** and a matching `*_review_file.csv`.

Initial redaction only: [`../doc-redaction-app/SKILL.md`](../doc-redaction-app/SKILL.md).

**Parallel page-range review (subagents):** [`../doc-redact-page-review/SKILL.md`](../doc-redact-page-review/SKILL.md) — one child per page for **Pass 1**, parent merges and applies once; **Pass 2** (optional VLM) runs after Pass 1 outputs exist.

## Two-pass review model

Review is split into a **fast, text-based first pass** (default) and an **optional visual second pass**. Do **not** run VLM on every page unless the user explicitly requests Pass 2 or you are re-checking specific flagged pages.

| | **Pass 1 — OCR / CSV / text (default)** | **Pass 2 — Visual VLM (optional)** |
|---|----------------------------------------|-------------------------------------|
| **Inputs** | `*_review_file.csv`, `*_ocr_output_*` (line), `*_ocr_results_with_words_*` (word), original PDF | Pass 1 outputs: merged CSV, `*_redacted.pdf`, preview/redaction overlay PNGs |
| **Methods** | Row edits, word/line OCR alignment, regex/policy rules, overlap checks, PyMuPDF text extraction, local `preview_redaction_boxes` | OpenAI-compatible VLM (`/v1/chat/completions` + page PNG) |
| **Cost / time** | Low — no image tokens | High — ~1–2 min/page on local VLMs; scales with page count |
| **When** | Always — completes the first reviewed apply | User asks for visual QA; high-risk pages; Pass 1 text checks inconclusive |
| **Apply** | **One** `/review_apply` after Pass 1 merge | Edit CSV from VLM findings → **one** more `/review_apply` if changes made |

**Default workflow:** Pass 1 only → deliver. Run Pass 2 only when needed.

## Endpoint semantics (do not get this wrong)

| Endpoint | Applies redaction? | Text layer stripped? | Output to use |
|----------|-------------------|----------------------|---------------|
| `/doc_redact` | Proposes boxes; may emit an early `*_redacted.pdf` | Run-dependent | Review CSV + OCR — treat as **draft** until Pass 1 review + apply |
| `/preview_boxes` | **No** — draws boxes on rendered page images only | N/A | Coordinate QA only |
| `/review_apply` | **Yes** — from **original PDF + `*_review_file.csv`** | **Yes** — PyMuPDF redaction + `apply_redactions()` | `*_redacted.pdf` |

**Never** implement a custom “true redaction” PyMuPDF script because `text_layer_leaks` appeared. Fix CSV coverage and coordinates first, then call `/review_apply` again. **Exception:** reading PyMuPDF word positions to **add normalized CSV rows** is allowed — writing the final PDF without `/review_apply` is not (see task template § Agent anti-confusion rules).

**Never** run post-apply text-layer checks on `*_redactions_for_review.pdf` — that file **retains text** for human review. `/review_apply` returns **both** files; only `*_redacted.pdf` is the deliverable.

Use [`tools/verify_redaction_coverage.py`](../../tools/verify_redaction_coverage.py) or `POST /agent/verify_redaction_coverage` — **do not** reimplement coverage logic ad hoc (pandas/regex scripts). On split-container Pi deployments, **pre-apply** checks use the **CLI on downloaded artifacts** (edited CSV is local); **post-apply** checks use the **Agent API with server paths** — see § Split-container verify below.

### Text-layer leak troubleshooting

When `text_layer_leaks` appear on `*_redacted.pdf`:

1. Confirm the PDF basename ends with `_redacted.pdf` (not `_redactions_for_review.pdf`).
2. **Pre-apply:** all `xmin/ymin/xmax/ymax` must be **normalized 0–1** (`df[bbox_cols].max().max() <= 1`). Never paste PyMuPDF absolute points without dividing by page width/height.
3. Read `leak_likely_causes` on each page in the coverage report:

| Cause | Meaning | Fix |
|-------|---------|-----|
| `missing_page_boxes` | No review rows on that page | Add boxes from word/line OCR or PyMuPDF text positions (normalized) |
| `missing_review_boxes` | Word OCR hits not intersecting any box | Add/extend review rows |
| `coord_not_normalized` | CSV rows use pixel/point coords (>1) | Normalize to 0–1; re-apply (headless apply now rejects invalid coords) |
| `coord_mismatch_or_image_text` | Word OCR covered but text still extractable | Widen boxes, split multi-line blocks, or redact image areas (`CUSTOM`); image-baked text cannot be stripped by text redaction alone. If **`pixel_failures` is empty** after apply, stop adding full-span boxes — document limitation or use Pass 2 visual check on those pages. |

Word OCR can show **100% covered** while the text layer still leaks — that is usually **coordinates** or **image text**, not a broken apply endpoint.

## Pass 1 — OCR / CSV / text review

Use artefacts from the **same redaction run**. No VLM in this pass.

### Inputs (discover in output folder)

| File | Use |
|------|-----|
| `*_review_file.csv` | Master list of proposed boxes — add/remove/relabel rows |
| `*_ocr_results_with_words_*.csv` | Word boxes (`word_x0`…`word_y1`) for precise coordinates |
| `*_ocr_output_*.csv` | **Line-level** OCR — reading order, line text, line indices; use for context and same-line grouping |
| Original (unredacted) `.pdf` | Preview overlays; text extraction sanity checks |
| `*_ocr_results_with_words_*.json` | Optional — same word data as CSV when easier to parse |

### Pass 1 loop (per page or whole document)

1. Load `*_review_file.csv` (`encoding="utf-8-sig"`).
2. **Policy edits** — remove false positives, add missing PII rows, relabel (programmatically, not Excel-only).
3. **Word OCR** — match `page` + `word_text`; merge words on the same line (`|Δy0| < ~0.01`); separate boxes across lines.
4. **Line OCR** — use line CSV for phrase context, line numbers, and confirming reading order when word boxes fragment a name or address.
5. **Coverage report (mandatory before apply)** — run [`tools/verify_redaction_coverage.py`](../../tools/verify_redaction_coverage.py) or **`POST /agent/verify_redaction_coverage`** with `must_redact` / `must_not_redact` regex lists. Fix **policy** flags (`uncovered_terms`, `over_redacted`, `text_layer_leaks`); re-run until `pass_strict` is true.
6. **Suspicious-row prune (standard Pass 1 cleanup)** — remove short OCR-fragment boxes (`"-"`, `"."`, `"Ho"`, etc.) that do **not** match `must_redact`. CLI: `--prune-suspicious --pruned-output merged_pruned.csv` or API: `auto_prune_suspicious: true`. Re-run coverage; target `pass_with_cleanup: true`.
7. **Preview** — `preview_redaction_boxes` or `/preview_boxes` on edited CSV (spot-check worst pages from the report).
8. **Merge** full-document CSV (all pages) if reviewing a subset — see page-review skill.
9. **One** `/review_apply` → download newest `*_redacted.pdf` / `*_review_file.csv` (sort by `st_mtime`).
10. **Coverage report (after apply)** — re-run with `redacted_pdf_path` for text-layer leak checks; optional `sample_pixels=true`.
11. **Term search (optional)** — `POST /agent/word_level_ocr_text_search` or `word_level_ocr_text_search` to find policy phrases in word OCR and whether each hit is boxed.

Pass 1 is **complete** when `pass_strict` is true (policy satisfied). **`pass_with_cleanup`** also requires no suspicious short rows. Run **Pass 2 VLM only on `pages_flagged_for_vlm`** (policy/visual risk — not `pages_needing_csv_cleanup` alone).

### Automatic post-redaction QA (optional — main app)

When `POST_REDACT_PASS1_QA=True` ([`tools/config.py`](../../tools/config.py)), initial redaction (Gradio / CLI / `/doc_redact`) runs [`tools/post_redaction_pass1_qa.py`](../../tools/post_redaction_pass1_qa.py) after writing `*_review_file.csv`:

- Emits `*_coverage_report.json` next to the review CSV
- Optionally emits sibling `*_review_file_pruned.csv` when `POST_REDACT_PASS1_AUTO_PRUNE=True` (does **not** replace the original CSV)
- Maps run **deny list → must_redact**, **allow list → must_not_redact** when `POST_REDACT_PASS1_USE_DENY_ALLOW_LISTS=True`
- Appends a one-line QA summary to the redaction status message

This is **pre-review-apply** deployment QA only. **Agent Pass 1** (policy edits, merge, `/review_apply`, post-apply coverage) is still required for case-specific review workflows.

### Coverage verification (Pass 1 — no VLM)

Programmatic QA replacing per-page visual review for most cases.

**CLI:**

```bash
python tools/verify_redaction_coverage.py merged_review_file.csv ocr_words.csv \
  --must-redact "cora|fuller|fyller" \
  --must-not-redact "dr\\.|macrae|gibson|social worker" \
  --prune-suspicious --pruned-output merged_pruned.csv \
  --redacted-pdf output_redacted.pdf \
  --output-json coverage_report.json
```

**Report fields:**

| Field | Meaning |
|-------|---------|
| `pass` / `pass_strict` | Policy satisfied: no uncovered terms, over-redactions, text leaks, or pixel failures |
| `pass_with_cleanup` | Also no suspicious short OCR-fragment rows |
| `pages_flagged_for_vlm` | Policy/visual failures → optional Pass 2 |
| `pages_needing_csv_cleanup` | Suspicious rows only → run prune step, not VLM |
| `leak_likely_causes` (per page) | Why `text_layer_leaks` appeared — see troubleshooting table above |

**Agent API:** `POST /agent/verify_redaction_coverage`

```json
{
  "review_csv_path": "path/to/doc_review_file.csv",
  "ocr_words_csv_path": "path/to/doc_ocr_results_with_words_local_ocr.csv",
  "must_redact": ["cora|fuller|fyller", "stephen|peter|rhett|yazmin"],
  "must_not_redact": ["dr\\.|doctor|social worker|macrae|gibson"],
  "redacted_pdf_path": "path/to/doc_redacted.pdf",
  "auto_prune_suspicious": true,
  "pruned_output_path": "path/to/doc_review_file_pruned.csv",
  "sample_pixels": false
}
```

Response includes `coverage_pass_strict`, `coverage_pass_with_cleanup`, `pruned_csv_path`, `prune_log`, per-page issues, and `pages_flagged_for_vlm` vs `pages_needing_csv_cleanup`.

### Split-container verify (Pi agent + separate redaction service)

When the Pi agent and doc_redaction run in **separate containers** (e.g. `http://redaction:7860`), path validation in `secure_path_utils` runs on the **redaction server**. Agent API paths must resolve under repo root, `INPUT_FOLDER`, or `OUTPUT_FOLDER` on that server.

| Phase | Where files live | How to verify |
|-------|------------------|---------------|
| **Pre-apply** | Edited `*_review_file.csv` in Pi session workspace; OCR CSV downloaded locally | `python tools/verify_redaction_coverage.py <local_review_csv> <local_ocr_words_csv> ...` — official CLI on downloaded copies |
| **Post-apply** | `*_redacted.pdf` and review CSV on redaction server from `/review_apply` | `POST {gradio_url}/agent/verify_redaction_coverage` with server paths from `extract_server_paths(review_apply result)` plus OCR words path from `/doc_redact` |

**Rejected paths (common mistakes):**

- Pi workspace paths (e.g. `/home/user/app/workspace/sess/...`)
- `/tmp/gradio_tmp/...` from `/gradio_api/upload` (not under `OUTPUT_FOLDER`)
- Calling `verify_redaction_coverage()` from the Pi container with redaction-server path strings

**Word search:** `POST /agent/word_level_ocr_text_search` with `ocr_words_csv_path`, `search_text`, optional `review_csv_path`.

`covered_by_review_box` uses **intersecting** review boxes (not strict containment). A hit marked `false` may still be visually redacted if a larger box overlaps — inspect coordinates before adding rows.

**Python:**

```python
from doc_redaction import verify_redaction_coverage, word_level_ocr_text_search

report = verify_redaction_coverage(
    "doc_review_file.csv",
    "doc_ocr_results_with_words_local_ocr.csv",
    must_redact=[r"cora|fuller"],
    must_not_redact=[r"dr\."],
    redacted_pdf_path="doc_redacted.pdf",
)
hits = word_level_ocr_text_search(
    "doc_ocr_results_with_words_local_ocr.csv",
    "Fuller",
    review_csv_path="doc_review_file.csv",
)
```

**Reference orchestrator:** [`workspace/run_pass1_cora_fyller.py`](../../workspace/run_pass1_cora_fyller.py) — policy edits → coverage fix → **prune suspicious rows** → single `/review_apply` → post coverage → term search.

### Word-level OCR (precise boxes)

Typical word columns: **`word_x0`, `word_y0`, `word_x1`, `word_y1`** (normalized 0–1). Match **`page`** and **`word_text`**:

- **Same line** (small vertical gap, e.g. `|Δy0| < 0.01`): merge to one box (`min`/`max` of coordinates).
- **Different lines**: **separate boxes** — one merged box spanning lines wipes unrelated text.

### Line-level OCR (context and grouping)

Line CSV rows usually include **`page`**, **`line_number`**, **`text`**, and line bbox columns. Use to:

- Find phrases split across word boxes or mis-merged review rows.
- Confirm which line a policy phrase belongs to before adding/removing boxes.
- Cross-check review CSV `text` against line `text` for false positives (e.g. org names, bare titles).

Prefer **word OCR for coordinates**, **line OCR for text context**.

## Pass 2 — Optional visual VLM review

Run **after Pass 1** has produced reviewed outputs. Checks whether black boxes match policy on **rendered pages** — catches handwriting, stamps, and OCR misses.

### When to run Pass 2

- User explicitly requests visual / VLM check of all pages or a page range.
- **`pages_flagged_for_vlm`** from coverage report after Pass 1 (preferred — targeted, not full doc). These are **policy/visual** failures only (`uncovered_terms`, text leaks, pixel failures) — **not** pages that only have suspicious short OCR rows (use prune instead).
- Pass 1 text/coverage verification inconclusive on scanned pages (handwriting, stamps, OCR-blind ink).

### When to skip Pass 2

- User did not ask for visual QA.
- Large documents where full-page VLM would exhaust time/token budget — prefer Pass 1 + targeted Pass 2 on flagged pages only.
- Pass 1 preview PNGs and text checks already sufficient.

### Pass 2 inputs

- **Preview PNGs** from `preview_redaction_boxes(original.pdf, pass1_merged.csv)` — proposed boxes on the original; **or**
- **Redacted page PNGs** rasterized from Pass 1 `*_redacted.pdf` — verify applied black boxes.
- Pass 1 merged `*_review_file.csv` as baseline for edits.

### Pass 2 loop

1. For each page (or flagged subset): render PNG at **moderate DPI** (≈100–120; `max_width` ≈1200 — huge tiles timeout).
2. **One VLM call per page** (or one focused question per call); sequential if using a local model (avoid parallel VLM overload).
3. Parse response → structured deltas: uncovered PII, practitioner boxes to remove, false positives.
4. **Conservative CSV edits only** — prefer explicit name/phrase matches from VLM; do **not** bulk-add every OCR token on a page (reasoning models may over-trigger additions).
5. Merge full CSV → **one** `/review_apply` if any changes → download → brief text re-verify.

Log per-page VLM results (e.g. `vlm_checks/p{N}/vlm_result.json`) when automating.

### OpenAI-compatible VLM

`POST {base_url}/v1/chat/completions` with multimodal `image_url` (`data:image/png;base64,...`) + short policy prompt.

- **`max_tokens`**: **≈1000–2500+** — low values often yield empty `content` with `finish_reason: length`.
- **`temperature`**: **≈0.1** for repeatable checks.
- **Reasoning models** (e.g. some Qwen variants): read **`content` and `reasoning_content`** — answers may be only in `reasoning_content`.
- **Prompts**: one focused question per call; state what must be **visible** vs **black-boxed**.
- **Structured output**: prefer explicit YES/NO lines or JSON; if the model returns prose-only reasoning, parse conservatively — avoid mass OCR additions from heuristic triggers.
- **Timeout**: **≈180–240 s** per page for local VLMs.

```python
import base64
import json
from pathlib import Path

import httpx

def vlm_review(image_path: str, prompt: str, base_url: str, model: str, max_tokens: int = 2000) -> str:
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    r = httpx.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        content=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        timeout=240.0,
    )
    if r.status_code != 200:
        return f"ERROR {r.status_code}: {r.text[:500]}"
    msg = r.json()["choices"][0]["message"]
    return (msg.get("content") or "") + (msg.get("reasoning_content") or "")
```

## Primary path — `/review_apply`

`gradio_client` with **`api_name="/review_apply"`**: `pdf_file`, `review_csv_file`, `output_dir` (`None` for server default).

Prefer **positional** args: `client.predict(handle_file(pdf), handle_file(csv), None, api_name="/review_apply")`.

Do not default to `/agent/apply_review_redactions` unless paths resolve **on the server** (see Fallbacks).

**Apply cadence:** at most **one apply per pass** (Pass 1 apply, then optional Pass 2 apply after VLM edits). Do not apply per page.

## Critical constraints

- Review CSV **basename** must contain `_review_file`.
- **Bounding boxes:** `xmin`, `ymin`, `xmax`, `ymax` must be **normalized 0–1** (not PDF points or pixels). Pre-apply sanity check: `df[["xmin","ymin","xmax","ymax"]].max().max() <= 1`.
- **`image` column**: reuse an **existing row’s `image` value for the same page** when adding rows.
- **`handle_file`**: local paths → `handle_file(...)`; server paths from prior upload → plain string.
- CSV: **`encoding="utf-8-sig"`** (BOM).
- Download: `GET {BASE}/gradio_api/file={urllib.parse.quote(path, safe="")}`; Bearer token on gated Spaces.
- **`httpx.Timeout`**: long **read** timeout (e.g. 1800s+) for large PDFs.
- Docker → host Gradio: `http://host.docker.internal:<port>`.

## Pre-apply preview

### A — Local (preferred, Pass 1 and Pass 2)

```bash
python tools/preview_redaction_boxes.py original.pdf edited_review_file.csv --pages 5,6 --grid
```

### B — Server `/preview_boxes`

Upload original PDF + CSV → ZIP of PNGs; no redaction applied.

### C — Fallback

PyMuPDF + Pillow: draw normalized `xmin`…`ymax` rectangles on rasterized pages.

## Picking the latest outputs

```python
from pathlib import Path

def latest_match(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]
```

## Minimal apply + download

```python
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import httpx
from gradio_client import Client, handle_file

BASE_URL = "https://example.hf.space".rstrip("/")
HF_TOKEN = None

httpx_kwargs = {"timeout": httpx.Timeout(connect=120.0, read=1800.0, write=120.0, pool=120.0)}
client = Client(BASE_URL, hf_token=HF_TOKEN, httpx_kwargs=httpx_kwargs) if HF_TOKEN else Client(BASE_URL, httpx_kwargs=httpx_kwargs)

pdf = Path("original.pdf")
csv_in = Path("document_review_file.csv")

raw = client.predict(handle_file(str(pdf)), handle_file(str(csv_in)), None, api_name="/review_apply")
paths, message = (raw[0], raw[1]) if isinstance(raw, (list, tuple)) and len(raw) >= 2 else (raw, "")

headers = {"Authorization": f"Bearer {HF_TOKEN.strip()}"} if HF_TOKEN else {}
out_dir = Path("downloads")
out_dir.mkdir(parents=True, exist_ok=True)
with httpx.Client(timeout=httpx_kwargs["timeout"], headers=headers) as http:
    for p in paths:
        if isinstance(p, str) and p.startswith("/"):
            url = f"{BASE_URL}/gradio_api/file={quote(p, safe='')}"
            (out_dir / Path(p).name).write_bytes(http.get(url).raise_for_status().content)
```

## CSV edits that come up often (Pass 1)

### Signatures

PII pipelines rarely catch ink. Use word/line OCR + grid preview; anchor **`SIGNATURE`** near “Signed” / printed name; separate **`PERSON`** rows for typed names.

### OCR-invisible content

Add **`CUSTOM`** rows from percentage-grid estimates; iterate preview → apply.

### Scanned pages without word boxes

Zone presets (see prior versions) or line OCR text + grid estimate.

When appending rows: same-page **`image`**, **`color`** as `"(0, 0, 0)"`, unique **`id`**.

## Verification

### Pass 1 (required)

1. **Coverage report** — `pass_strict` (policy terms covered, no over-redactions, no text leaks).
2. **Text layer** — PyMuPDF on `*_redacted.pdf`; policy strings should be absent where boxed.
3. **Word OCR overlap** — target terms intersect review boxes on each page.
4. **Preview PNGs** — spot-check worst pages locally.

### Pass 2 (when run)

1. VLM per page against preview or redacted PNG.
2. Conservative CSV patch → single re-apply if needed.
3. Re-run Pass 1 text/OCR checks on updated PDF.

Watch **false positives**: geography/org as **PERSON**, bare job titles, OCR fragments — trim in Pass 1; VLM may flag in Pass 2.

## Fallbacks

1. Raw **`/gradio_api/*`**
2. **`/agent/apply_review_redactions`** — server-local paths only
3. Browser UI
4. Local PyMuPDF apply — [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — **only when `/review_apply` fails** (HTTP/Gradio error), **not** when coverage reports `text_layer_leaks`

## Checklists

**Pass 1 (each page):** policy removals/additions; word OCR box alignment; line OCR context; false positives; signatures; **coverage report `pass_strict`**; **suspicious-row prune**; preview spot-check; merge; single apply; post-apply coverage report.

**Pass 2 (optional, flagged pages only):** VLM on `pages_flagged_for_vlm`; conservative CSV patch; single re-apply; re-run coverage report.

## When stuck

[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)

Repo API overview: [AGENTS.md](../../AGENTS.md).
