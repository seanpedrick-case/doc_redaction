---
name: doc-redaction-modifications
description: "Review and reapply: edit *_review_file.csv, preview boxes, call Gradio /review_apply, download newest outputs, and verify. Parallel multi-page orchestration (subagents) → doc-redact-page-review. Initial redaction → doc-redaction-app."
version: 2.1.2
author: repo-maintained
license: AGPL-3.0-only
---

## Goal

Repeatable **review → edit CSV → preview → `/review_apply` → download → verify** loop when you already have the **original PDF** and a matching `*_review_file.csv`.

Initial redaction only: [`../doc-redaction-app/SKILL.md`](../doc-redaction-app/SKILL.md).

**Parallel page-range review (subagents):** after a first-pass run, if the user wants **many pages reviewed in parallel** with a **single** merged `/review_apply`, use the orchestration playbook [`../doc-redact-page-review/SKILL.md`](../doc-redact-page-review/SKILL.md) (one child per page, parent merges full CSV, parent calls `/review_apply` once). This skill stays the reference for CSV/OCR/preview/API details.

## Primary path

`gradio_client` with **`api_name="/review_apply"`** and three arguments: `pdf_file`, `review_csv_file`, `output_dir` (use `None` for server default).

Prefer **positional** arguments when automating (`client.predict(handle_file(pdf), handle_file(csv), None, api_name="/review_apply")`)—named kwargs can break endpoint inference on some Gradio multi-route apps.

Do not default to `/agent/apply_review_redactions` unless paths are valid **on the server** (see Fallbacks).

## Critical constraints

- Review CSV **basename** must contain `_review_file`.
- **`image` column**: reuse an **existing row’s `image` value for the same page** when adding rows; arbitrary placeholders can cause rows to be dropped on apply.
- **`handle_file`**: local paths → `handle_file(...)`; server paths from a prior upload → plain string, not wrapped.
- CSV: read/write with **`encoding="utf-8-sig"`** (BOM).
- Returned paths live **inside the server/container**; fetch with **`GET {BASE}/gradio_api/file={urllib.parse.quote(path, safe="")}`** (encode the full path), bind mount, or `docker cp`. Use the same **`Authorization: Bearer`** as the client on gated HF Spaces.
- **`httpx.Timeout`**: long **read** timeout for large PDFs (e.g. 1800s+).
- From a **Docker client** to Gradio on the host, use `http://host.docker.internal:<port>` instead of `localhost`.
- If **`/review_apply` HTTP responses** look like HTML (even with status 200), treat as server error—inspect body; do not assume JSON.
- Smoke-check the Space/app is up (e.g. `GET` base URL) before long runs.

## Execution loop

For a full pass, edit **one** consolidated CSV and apply **once**. Page-by-page apply only when you need intermediate diffs.

1. Load `*_review_file.csv` in a script (not by hand in Excel only).
2. Add/remove/relabel rows; fix coordinates programmatically.
3. **Preview** box geometry (local tool or `/preview_boxes`) before calling the server.
4. **`/review_apply`** when the preview is acceptable.
5. **Download** artifacts; **sort by `st_mtime`** and take the newest `*_redacted.pdf` / `*_review_file.csv` (each apply adds hash-prefixed names).
6. **Verify** (text extraction and/or review PNGs); repeat from 2 if needed.

## Word-level OCR (for precise boxes)

Use `*_ocr_results_with_words_*.csv` from the same run. Typical word columns include **`word_x0`, `word_y0`, `word_x1`, `word_y1`** (normalized 0–1, same convention as the review CSV). Match **`page`** and **`word_text`**, then:

- **Same line** (small vertical gap between word rows, e.g. `|Δy0| < 0.01`): merge to one box with `min`/`max` of coordinates.
- **Different lines**: **separate boxes**—one merged box spanning lines wipes unrelated text.

Optional: overlap check—confirm each target word’s rectangle intersects some review row on that page before applying.

## Pre-apply preview

### A — Local (preferred)

```bash
python tools/preview_redaction_boxes.py original.pdf edited_review_file.csv --pages 5,6 --grid
```

Or `from tools.preview_redaction_boxes import preview_redaction_boxes` (see tool docstring for `dpi`, `draw_grid`, `pages`).

### B — Server `/preview_boxes`

When you lack local repo tools but can upload: returns a **ZIP** of PNGs; no redaction applied. Download the returned path like any other `gradio_api/file=` artifact.

### C — No `preview_redaction_boxes` available

Render the original PDF with PyMuPDF/Pillow at chosen DPI and draw rectangles using normalized `xmin`…`ymax` × pixel width/height—same math as inside `preview_redaction_boxes`.

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
HF_TOKEN = None  # set if Space is gated

httpx_kwargs = {"timeout": httpx.Timeout(connect=120.0, read=1800.0, write=120.0, pool=120.0)}
client = Client(BASE_URL, hf_token=HF_TOKEN, httpx_kwargs=httpx_kwargs) if HF_TOKEN else Client(BASE_URL, httpx_kwargs=httpx_kwargs)

pdf = Path("original.pdf")
csv_in = Path("document_review_file.csv")  # basename must contain _review_file

raw = client.predict(
    handle_file(str(pdf)),
    handle_file(str(csv_in)),
    None,
    api_name="/review_apply",
)
paths, message = (raw[0], raw[1]) if isinstance(raw, (list, tuple)) and len(raw) >= 2 and isinstance(raw[-1], str) else (raw if isinstance(raw, list) else [raw], "")
print(message, paths)

headers = {}
if HF_TOKEN:
    headers["Authorization"] = f"Bearer {HF_TOKEN.strip()}"
out_dir = Path("downloads")
out_dir.mkdir(parents=True, exist_ok=True)
manifest = []
with httpx.Client(timeout=httpx_kwargs["timeout"], headers=headers) as http:
    for p in paths:
        if not isinstance(p, str) or not p.startswith("/"):
            continue
        url = f"{BASE_URL}/gradio_api/file={quote(p, safe='')}"
        data = http.get(url).raise_for_status().content
        dest = out_dir / Path(p).name
        dest.write_bytes(data)
        manifest.append({"server_path": p, "local": str(dest), "sha256": hashlib.sha256(data).hexdigest()})
(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
```

## CSV edits that come up often

### Signatures (always treat as manual)

PII pipelines rarely catch ink. Locate ink vs printed name in a grid preview; anchor **`SIGNATURE`** boxes using nearby OCR words (“Signed”, titles). Rough anchors: signature band often **just above** the printed name line; use separate **`PERSON`** (or rule-appropriate label) rows for typed names if they are sensitive. With **word-level OCR only**, a **vertical gap** between last body line and first footer/name line can indicate a signature band—still confirm in a preview before apply.

### OCR-invisible content (stamps, calligraphy)

If it is visible on the page but absent from word OCR, add **`CUSTOM`** (or rule label) rows from a **percentage-grid** estimate; expect a few preview/apply iterations—local rasterization can differ slightly from server PDF space.

### Scanned pages without word boxes

Use deterministic **zone presets** as starting guesses, then refine (example normalized rectangles):

| Zone         | xmin | ymin | xmax | ymax |
|-------------|------|------|------|------|
| top_left    | 0.05 | 0.08 | 0.45 | 0.18 |
| top_right   | 0.55 | 0.08 | 0.95 | 0.18 |
| mid_left    | 0.05 | 0.40 | 0.45 | 0.52 |
| mid_right   | 0.55 | 0.40 | 0.95 | 0.52 |
| bottom_left | 0.05 | 0.78 | 0.45 | 0.90 |
| bottom_right | 0.55 | 0.78 | 0.95 | 0.90 |

When appending rows, set **`image`** from an existing same-page row; **`color`** as a quoted string, e.g. `"(0, 0, 0)"`; unique **`id`**.

```python
import csv
import secrets
from pathlib import Path

ZONE = {
    "top_left": (0.05, 0.08, 0.45, 0.18),
    "top_right": (0.55, 0.08, 0.95, 0.18),
    "mid_left": (0.05, 0.40, 0.45, 0.52),
    "mid_right": (0.55, 0.40, 0.95, 0.52),
    "bottom_left": (0.05, 0.78, 0.45, 0.90),
    "bottom_right": (0.55, 0.78, 0.95, 0.90),
}
# spec = [{"page": 1, "zone": "bottom_right", "label": "SIGNATURE", "text": "signature"}, ...]


def append_zones(review_csv: Path, spec: list, out_csv: Path) -> None:
    rows = list(csv.DictReader(review_csv.open(encoding="utf-8-sig")))
    fieldnames = list(rows[0].keys())
    img_col = fieldnames[0]
    for item in spec:
        page = int(item["page"])
        xmin, ymin, xmax, ymax = ZONE[item["zone"]]
        same_page = [r for r in rows if int(float(r.get("page", 0) or 0)) == page]
        image_val = (
            same_page[0].get(img_col)
            if same_page and same_page[0].get(img_col)
            else f"placeholder_image_{max(page - 1, 0)}.png"
        )
        new_row = {k: "" for k in fieldnames}
        new_row.update(
            {
                img_col: image_val,
                "page": str(page),
                "label": item.get("label", "CUSTOM"),
                "color": "(0, 0, 0)",
                "xmin": f"{xmin:.4f}",
                "ymin": f"{ymin:.4f}",
                "xmax": f"{xmax:.4f}",
                "ymax": f"{ymax:.4f}",
                "id": secrets.token_hex(6),
                "text": item.get("text", ""),
            }
        )
        rows.append(new_row)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
```

## Verification (after reapply)

1. **Text layer**: `PyMuPDF` `page.get_text()` on the **`*_redacted.pdf`**—sensitive strings present in the original should be **gone** under successful blackouts. Useless for purely scanned pages with no text.
2. **Scanned / hybrid**: check overlap between review boxes and **`*_ocr_results_with_words_*.csv`** positions for the terms you care about.
3. **Optional — VLM spot-check**: export a **moderate-DPI** page PNG (downscale very large rasterizations—huge tiles time out). Many OpenAI-compatible servers are **text-only**; confirm vision with a **real page crop**, not a tiny placeholder image (1×1 probes mislead).

### OpenAI-compatible VLM (when the endpoint supports vision)

Use the usual **`/v1/chat/completions`** multimodal body: one `image_url` with a **`data:image/png;base64,...`** URL plus a short text prompt.

- **`max_tokens`**: set **high (≈1000–2500+)**. Image tokens count toward the limit; values like 50–200 often return **empty `content` with HTTP 200** and `finish_reason: length`. Budget for both encoding and the answer.
- **`temperature`**: keep **low** (e.g. **0.1**) for repeatable yes/no checks.
- **Reasoning-style models** (e.g. some Qwen variants): read **`content` and `reasoning_content`** and concatenate—analysis may live only in **`reasoning_content`** while `content` stays empty on success.
- **Prompts**: one focused question per call (“Is the name X still readable?”); long multi-part prompts often come back empty—**split** into separate requests. Say what should be **visible** vs **covered by black boxes**.
- **Client**: `httpx.post(..., content=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, timeout=180)` (raise timeout for slow hosts / large images).

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
        timeout=180.0,
    )
    if r.status_code != 200:
        return f"ERROR {r.status_code}: {r.text[:500]}"
    msg = r.json()["choices"][0]["message"]
    return (msg.get("content") or "") + (msg.get("reasoning_content") or "")
```

Watch **false positives** when trimming boxes: geography/org phrases mis-tagged as **PERSON**, bare job titles, OCR gibberish next to signatures mistaken for names—remove or relabel per policy.

## Fallbacks

1. Raw **`/gradio_api/*`**: upload (if enabled) → `call/review_apply` → poll → `file=` download with encoding + auth.
2. **`/agent/apply_review_redactions`**: only if both paths resolve under the server’s allowed roots.
3. **Browser UI** if APIs are blocked.
4. **Container / path isolation** (client cannot upload, agent paths rejected): apply the same CSV boxes **locally** with PyMuPDF (`add_redact_annot` + `apply_redactions`, or equivalent)—see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

## Gradio vs Agent names

Review-tab Gradio exports use names like **`page_redaction_review_image`** / **`page_ocr_review_image`**. FastAPI **`/agent/export_review_*`** routes are different; do not use agent path strings as Gradio `api_name`. The short apply route for agents is **`/review_apply`**, not the full Review-tab **`apply_review_redactions`** event (many positional inputs).

## Checklists

**Per page (policy-dependent):** names, signatures + printed names, target phrases, OCR-invisible stamps/headings, box size/alignment, false positives removed.

**Run:** `utf-8-sig` CSV; preview before apply; **`/review_apply`** as default; newest outputs by mtime; verify with extraction and/or images.

## When stuck

[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — path validation, 403 downloads, and PyMuPDF fallback steps.

Repo API overview: [AGENTS.md](../../AGENTS.md).
