---
name: doc-redaction-modifications
description: "Modify existing PDF redactions using a single default path: Gradio `review_apply` with `gradio_client`. Use this when editing `*_review_file.csv`, adding scanned-page boxes, applying one page at a time, and verifying outputs."
version: 2.0.2
author: repo-maintained
license: AGPL-3.0-only
---

## Goal

Apply targeted fixes to an existing redaction run with a repeatable, page-by-page workflow.

Use this skill when you already have:
- The original PDF.
- A corresponding `*_review_file.csv`.

## Primary path (default)

Use `gradio_client` with `api_name="/review_apply"` and these 3 inputs:
1. `pdf_file`
2. `review_csv_file`
3. `output_dir` (`None` to use server default)

Do not start with `/agent` or long UI-chain endpoints unless the fallback section says to.

## Critical constraints

- `review_apply` is the stable default for headless review modifications.
- Prefer calling `/review_apply` with **positional arguments** (`client.predict(pdf, csv, None, api_name="/review_apply")`) when automating across deployments; named keyword arguments may be brittle in some Gradio 6.x multi-endpoint apps.
- `/agent/apply_review_redactions` can return 404 or be unusable depending on deployment; treat it as fallback.
- Local client file path: use `handle_file("/local/path/file.pdf")`. Server path returned from upload (for example `/tmp/gradio_tmp/...`): pass as plain string. Do not wrap server paths in `handle_file(...)`.
- CSV files may have UTF-8 BOM; read/write with `encoding="utf-8-sig"` when editing.
- Container output paths (for example `/home/user/app/output/...`) are inside the container. Retrieve outputs via `GET /gradio_api/file=...`, a bind mount, or `docker cp`.
- Input review CSV basename must contain `_review_file`.
- When using `/review_apply`, do not submit review rows whose `image` values are fake placeholders (for example `placeholder_image_2.png`) unless they are valid for the active run context. In field failures this stripped redaction rows during apply. Keep `image` values aligned with current run artifacts.
  - Practical rule: when adding new rows, set `image` to an **existing row’s** `image` value **for the same page** whenever possible; only fall back to `placeholder_image_{page-1}.png` if you know the run uses that convention.

These constraints are intentionally aligned with `doc-redaction-app/SKILL.md` so both skills use the same operational rules.

## One-page execution loop

Run this loop for each page:
1. Edit only one target page in `*_review_file.csv`.
2. Apply with `/review_apply`.
3. Download outputs.
4. Verify page review export image visually.
5. Move to next page.

## Minimal end-to-end example (copy/paste)

This script performs one full cycle: edit page rows -> apply -> download artifacts.

```python
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import httpx
from gradio_client import Client, handle_file

BASE_URL = "https://seanpedrickcase-document-redaction.hf.space"
PDF_PATH = Path("example_data/example_of_emails_sent_to_a_professor_before_applying.pdf").resolve()
REVIEW_CSV = Path("example_data/example_outputs/example_of_emails_sent_to_a_professor_before_applying_review_file.csv").resolve()
WORK_DIR = Path("tmp/review_cycle").resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PAGE = 1

edited_csv = WORK_DIR / REVIEW_CSV.name

# 1) Edit CSV for one page (example: remove one row on TARGET_PAGE)
with REVIEW_CSV.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
fieldnames = list(rows[0].keys())
page_rows = [r for r in rows if int(float(r.get("page", "0") or 0)) == TARGET_PAGE]
other_rows = [r for r in rows if int(float(r.get("page", "0") or 0)) != TARGET_PAGE]
if page_rows:
    page_rows = page_rows[1:]  # delete one redaction on this page
updated_rows = other_rows + page_rows

with edited_csv.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(updated_rows)

# 2) Apply edited review CSV via /review_apply
client = Client(BASE_URL)
paths, message = client.predict(
    api_name="/review_apply",
    pdf_file=handle_file(str(PDF_PATH)),
    review_csv_file=handle_file(str(edited_csv)),
    output_dir=None,
)
print("Apply message:", message)
print("Returned paths:", paths)

# 3) Download returned server files
download_dir = WORK_DIR / "downloads"
download_dir.mkdir(exist_ok=True)

manifest = []
with httpx.Client(timeout=120.0) as http:
    for p in paths:
        name = Path(p).name or "artifact.bin"
        url = f"{BASE_URL}/gradio_api/file={quote(p, safe='/')}"
        r = http.get(url)
        r.raise_for_status()
        out = download_dir / name
        out.write_bytes(r.content)
        manifest.append(
            {
                "server_path": p,
                "local_file": str(out),
                "size": out.stat().st_size,
                "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            }
        )

(WORK_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Downloaded files to:", download_dir)
```

## Adding redactions on scanned/image pages (no OCR boxes)

Use deterministic zone-based coordinates, not random values.
Use zone presets as a starting estimate, not a final geometry guarantee.
For variable signatures/stamps, refine coordinates with vision review from page review export images before final apply.

### Zone presets

Use these normalized boxes:
- `top_left`: `xmin=0.05, ymin=0.08, xmax=0.45, ymax=0.18`
- `top_right`: `xmin=0.55, ymin=0.08, xmax=0.95, ymax=0.18`
- `mid_left`: `xmin=0.05, ymin=0.40, xmax=0.45, ymax=0.52`
- `mid_right`: `xmin=0.55, ymin=0.40, xmax=0.95, ymax=0.52`
- `bottom_left`: `xmin=0.05, ymin=0.78, xmax=0.45, ymax=0.90`
- `bottom_right`: `xmin=0.55, ymin=0.78, xmax=0.95, ymax=0.90`

### Script: page-image spec -> review CSV rows

Inputs:
- Existing `*_review_file.csv`
- A JSON spec listing page and zone for each missing redaction

```python
import csv
import json
import secrets
from pathlib import Path

REVIEW_CSV = Path("example_data/example_outputs/example_of_emails_sent_to_a_professor_before_applying_review_file.csv")
SPEC_JSON = Path("tmp/scan_zone_spec.json")
OUT_CSV = Path("tmp/scan_zone_review_file.csv")

ZONE = {
    "top_left": (0.05, 0.08, 0.45, 0.18),
    "top_right": (0.55, 0.08, 0.95, 0.18),
    "mid_left": (0.05, 0.40, 0.45, 0.52),
    "mid_right": (0.55, 0.40, 0.95, 0.52),
    "bottom_left": (0.05, 0.78, 0.45, 0.90),
    "bottom_right": (0.55, 0.78, 0.95, 0.90),
}

# SPEC format:
# [
#   {"page": 1, "zone": "bottom_right", "label": "SIGNATURE", "text": "signature"},
#   {"page": 2, "zone": "top_left", "label": "CUSTOM", "text": "address"}
# ]
spec = json.loads(SPEC_JSON.read_text(encoding="utf-8"))

with REVIEW_CSV.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
fieldnames = list(rows[0].keys())
image_col = fieldnames[0]

for item in spec:
    page = int(item["page"])
    xmin, ymin, xmax, ymax = ZONE[item["zone"]]
    # Keep image values aligned with this run: prefer an existing row's image for the same page.
    same_page = [r for r in rows if int(float(r.get("page", "0") or 0)) == page]
    image_value = same_page[0].get(image_col) if same_page and same_page[0].get(image_col) else f"placeholder_image_{max(page - 1, 0)}.png"
    rows.append(
        {
            image_col: image_value,
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

with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print("Wrote:", OUT_CSV)
```

After writing `OUT_CSV`, apply it with the minimal end-to-end script above.

## Practical verification (headless-safe)

### Step 1: Programmatic text extraction (primary — fast, reliable)

Extract visible text from the redacted PDF and confirm sensitive terms are NOT extractable. Black redaction boxes suppress text extraction, so if a term still appears in extracted text, its box is misaligned or missing. This is the primary verification method — faster than visual checks and catches missed PII definitively.

```python
import fitz  # PyMuPDF

REDACTED = "output/document_redacted.pdf"
ORIGINAL = "input/document.pdf"
SENSITIVE_TERMS = ["London", "Sister City", "rudy.giuliani@email.com"]  # adjust per rules

def extract_text(path):
    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)

redacted_text = extract_text(REDACTED).lower()
original_text = extract_text(ORIGINAL).lower()

for term in SENSITIVE_TERMS:
    t = term.lower()
    orig_count = original_text.count(t)
    redact_count = redacted_text.count(t)
    status = "PASS" if redact_count == 0 else "FAIL"
    print(f"[{status}] '{term}': {orig_count} in original, {redact_count} in redacted")
```

For scanned/image pages (no text layer), cross-reference box coordinates with OCR output CSVs (`*_ocr_results_with_words_*.csv`) to verify boxes overlap actual word positions.

### Step 2: Review images (optional supplement)

Prefer this method over fragile in-session review export endpoints for visual spot-checking. **CAUTION:** `vision_analyze` times out on large PDF page images (~4096x5760 pixels). If using vision models, downscale to 1280px max width first. For most cases, Step 1 alone is sufficient.

```python
import csv
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

PDF_PATH = Path("tmp/review_cycle/downloads/example_redacted.pdf")
REVIEW_CSV = Path("tmp/review_cycle/downloads/example_review_file.csv")
OUT_DIR = Path("tmp/review_cycle/page_review_exports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with REVIEW_CSV.open("r", newline="", encoding="utf-8-sig") as f:
    review_rows = list(csv.DictReader(f))

rows_by_page = {}
for r in review_rows:
    page = int(float(r.get("page", "0") or 0))
    rows_by_page.setdefault(page, []).append(r)

doc = fitz.open(PDF_PATH)
for p in range(1, doc.page_count + 1):
    page_obj = doc[p - 1]
    pix = page_obj.get_pixmap(dpi=180)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Downscale for vision compatibility if needed
    max_w = 1280
    if img.width > max_w:
        scale = max_w / img.width
        new_size = (max_w, int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    draw = ImageDraw.Draw(img)
    for r in rows_by_page.get(p, []):
        x0 = float(r["xmin"]) * pix.width
        y0 = float(r["ymin"]) * pix.height
        x1 = float(r["xmax"]) * pix.width
        y1 = float(r["ymax"]) * pix.height
        if img.width != pix.width:
            scale_x, scale_y = img.width / pix.width, img.height / pix.height
            x0 *= scale_x; y0 *= scale_y; x1 *= scale_x; y1 *= scale_y
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
    img.save(OUT_DIR / f"page_{p:03d}_review.png")

print("Review images in:", OUT_DIR)
```

## Fallbacks (use only if default path is blocked)

1. Raw Gradio HTTP:
   - `POST /gradio_api/upload` (PDF + review CSV)
   - `POST /gradio_api/call/review_apply` with `{"data":[pdf_path, csv_path, null]}`
   - Poll `GET /gradio_api/call/review_apply/{event_id}`
2. `/agent/apply_review_redactions`:
   - Use only when both file paths are server-local and accepted by route validation.
3. Browser UI:
   - Last resort when API access is unavailable.

## Completion checklist

- [ ] Worked page-by-page, not full-document edits first.
- [ ] Read and wrote CSV with `utf-8-sig`.
- [ ] Used `/review_apply` as primary path.
- [ ] Recovered container outputs via `file=` download, bind mount, or `docker cp`.
- [ ] Produced visual review images for verification before sign-off.

## Known Issues & Workarounds

### gradio_client Endpoint Inference Bug (Gradio 6.x)
When calling endpoints via `gradio_client.Client().predict()`, the client fails to dispatch if **named keyword arguments** are used with multi-endpoint apps:
```python
# FAILS - endpoint inference can't match named kwargs:
client.predict(annotated_image=..., current_page=1, review_file_state=modified_df)

# WORKAROUND 1 - positional args only (no names):
client.predict(image_dict, 1, modified_df, "/home/user/app/output/", True)

# WORKAROUND 2 - raw HTTP POST + SSE polling:
POST /gradio_api/call/apply_review_redactions with JSON payload
GET /gradio_api/call/apply_review_redactions/{event_id} for results (requires async queue handling)
```

### Review Dataframe Modification Pattern
Review data from `result[6]` of `/choose_and_run_redactor` is a dict: `{headers, data, metadata}`. To modify:
- **Remove rows**: Filter out unwanted entries by text/label match before passing to apply endpoint
- **Reclassify labels**: Change label strings in data rows (e.g., "STREETNAME" → "ADDRESS")
- Modified dataframe must maintain same structure with headers intact

### Affected Endpoints (endpoint inference bug)
| Endpoint | Purpose | Status |
|----------|---------|--------|
| `/apply_review_redactions` | Apply modified review data → PDF + log + updated df | ⚠️ Works with positional args only |
| `/export_review_page_ocr_visualisation` | Export OCR overlay image via AnnotatedImageData input | ✅ Confirmed working pattern from initial redaction |
| `/export_review_redaction_overlay` | Export redaction overlay with review_df → PNG + log paths | ⚠️ Same endpoint inference issue expected |
| `/combine_review_csvs`, `/combine_review_pdfs` | Combine results across multiple review sessions | Not tested — likely same positional-args requirement |

## When the default flow breaks

Use [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for deeper failure-mode diagnosis and recovery steps.
