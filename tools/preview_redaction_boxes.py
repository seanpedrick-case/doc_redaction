"""
preview_redaction_boxes.py
==========================
Local-first coordinate preview tool for the Document Redaction app.

Purpose
-------
Render proposed redaction boxes from a ``*_review_file.csv`` onto the
**original** (un-redacted) PDF pages and save the result as PNG images.
Because this runs entirely locally with PyMuPDF + Pillow, iteration is
instantaneous — no server round-trip, no waiting for ``/review_apply``.

Primary use-case
----------------
Called by agents or humans **between CSV edits and the API call to
``/review_apply``**.  Iterate until the preview looks right, *then*
send to the server.  This avoids the expensive cycle of:

    guess coordinates → apply → download → render → spot the miss → repeat

Typical agent workflow
----------------------
1. Edit ``*_review_file_edited.csv`` (remove FPs, add signatures, etc.).
2. Call ``preview_redaction_boxes(pdf_path, csv_path, out_dir)`` locally.
3. Inspect the saved PNGs.
4. If anything is wrong, adjust the CSV and go to step 2.
5. Only when satisfied, call ``/review_apply`` on the server.

API endpoint (server-side fallback)
------------------------------------
When the agent does not have a local copy of the original PDF,
``preview_boxes_api()`` exposes the same logic as a short ``gr.api``
endpoint registered as ``/preview_boxes`` in ``app.py``.  The caller
uploads the original PDF and the edited review CSV; the server returns a
ZIP of preview PNGs.

CLI usage
---------
    python tools/preview_redaction_boxes.py original.pdf review_file.csv

    # Optional flags:
    python tools/preview_redaction_boxes.py original.pdf review_file.csv \\
        --out-dir output/preview \\
        --dpi 150 \\
        --max-width 1280 \\
        --grid            # draw percentage-grid lines
        --pages 1,3,5     # only render specific pages (1-indexed)
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Sequence

import pymupdf
from PIL import Image, ImageDraw, ImageFont

# ── Colour palette per label type ──────────────────────────────────────────
_LABEL_COLOURS: dict[str, str] = {
    "PERSON": "#e74c3c",  # red
    "SIGNATURE": "#8e44ad",  # purple
    "LOCATION": "#2980b9",  # blue
    "EMAIL_ADDRESS": "#e67e22",  # orange
    "PHONE_NUMBER": "#27ae60",  # green
    "CUSTOM": "#f39c12",  # amber
    "DATE_TIME": "#16a085",  # teal
    "ORG": "#7f8c8d",  # grey
}
_DEFAULT_COLOUR = "#c0392b"

# ── Grid style ─────────────────────────────────────────────────────────────
_GRID_COLOUR = "#cc0000"
_GRID_STEP = 5  # percentage intervals


def _label_colour(label: str) -> str:
    for key, colour in _LABEL_COLOURS.items():
        if key in label.upper():
            return colour
    return _DEFAULT_COLOUR


def _load_font(size: int = 11) -> ImageFont.ImageFont:
    """Return a PIL font; fall back to the default if no TTF is available."""
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def preview_redaction_boxes(
    pdf_path: str | Path,
    csv_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    dpi: int = 150,
    max_width: int = 1280,
    draw_grid: bool = True,
    pages: Sequence[int] | None = None,
) -> list[Path]:
    """
    Render proposed redaction boxes from *csv_path* onto the original PDF
    at *pdf_path* and save one PNG per page to *out_dir*.

    Parameters
    ----------
    pdf_path:
        Path to the original (un-redacted) PDF.
    csv_path:
        Path to the ``*_review_file.csv`` (original or edited).
    out_dir:
        Directory for output PNGs.  Defaults to a ``preview/`` subfolder
        next to the CSV.
    dpi:
        Render resolution.  150 is a good balance of speed vs. detail.
        Use 200-300 for detailed inspection of small text.
    max_width:
        Downscale rendered pages to at most this width (pixels) before
        drawing boxes, to keep file sizes manageable.
    draw_grid:
        If True, overlay horizontal lines at every *_GRID_STEP* percent of
        page height with percentage labels so you can read off normalized
        y-coordinates by eye.
    pages:
        If given, only render these 1-indexed page numbers.  Useful when
        you are iterating on a single page and don't want to wait for the
        whole document.

    Returns
    -------
    list[Path]
        Sorted list of saved PNG paths.
    """
    pdf_path = Path(pdf_path)
    csv_path = Path(csv_path)

    if out_dir is None:
        out_dir = csv_path.parent / "preview"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load CSV ────────────────────────────────────────────────────────────
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    rows_by_page: dict[int, list[dict]] = {}
    for row in rows:
        try:
            page_num = int(float(row.get("page", "0") or 0))
        except ValueError:
            continue
        rows_by_page.setdefault(page_num, []).append(row)

    # ── Render pages ────────────────────────────────────────────────────────
    doc = pymupdf.open(str(pdf_path))
    font = _load_font(11)
    saved: list[Path] = []

    page_range = range(1, doc.page_count + 1)
    if pages:
        page_range = [p for p in pages if 1 <= p <= doc.page_count]

    for page_num in page_range:
        pix = doc[page_num - 1].get_pixmap(dpi=dpi)
        render_w, render_h = pix.width, pix.height

        img = Image.frombytes("RGB", [render_w, render_h], pix.samples)

        # ── Downscale if needed ──────────────────────────────────────────
        if render_w > max_width:
            scale = max_width / render_w
            img = img.resize((max_width, int(render_h * scale)), Image.LANCZOS)
        draw_w, draw_h = img.size

        draw = ImageDraw.Draw(img, "RGBA")

        # ── Percentage grid ──────────────────────────────────────────────
        if draw_grid:
            for pct in range(0, 101, _GRID_STEP):
                y = int(pct / 100 * draw_h)
                draw.line([(0, y), (draw_w, y)], fill=_GRID_COLOUR + "55", width=1)
                draw.text((3, max(0, y - 11)), f"{pct}%", fill=_GRID_COLOUR, font=font)

        # ── Redaction boxes ──────────────────────────────────────────────
        for row in rows_by_page.get(page_num, []):
            try:
                x0 = float(row["xmin"]) * draw_w
                y0 = float(row["ymin"]) * draw_h
                x1 = float(row["xmax"]) * draw_w
                y1 = float(row["ymax"]) * draw_h
            except (KeyError, ValueError):
                continue

            label = row.get("label", "CUSTOM")
            colour = _label_colour(label)
            text_snippet = (row.get("text", "") or "")[:30]

            # Semi-transparent fill
            draw.rectangle(
                [x0, y0, x1, y1], fill=colour + "33", outline=colour, width=2
            )

            # Label text
            tag = f"{label}: {text_snippet}" if text_snippet else label
            draw.text((x0 + 3, y0 + 2), tag, fill=colour, font=font)

        # ── Legend (top-right corner) ────────────────────────────────────
        legend_labels = sorted(
            {r.get("label", "CUSTOM") for r in rows_by_page.get(page_num, [])}
        )
        lx, ly = draw_w - 200, 8
        for lbl in legend_labels:
            col = _label_colour(lbl)
            draw.rectangle(
                [lx, ly, lx + 14, ly + 14], fill=col + "cc", outline=col, width=1
            )
            draw.text((lx + 18, ly + 1), lbl, fill=col, font=font)
            ly += 17

        out_path = out_dir / f"page_{page_num:03d}_preview.png"
        img.save(out_path)
        saved.append(out_path)

    doc.close()
    print(f"Saved {len(saved)} preview image(s) to: {out_dir}")
    return sorted(saved)


def preview_redaction_boxes_to_zip(
    pdf_path: str | Path,
    csv_path: str | Path,
    *,
    dpi: int = 150,
    max_width: int = 1280,
    draw_grid: bool = True,
    pages: Sequence[int] | None = None,
) -> bytes:
    """
    Same as ``preview_redaction_boxes`` but returns a ZIP of PNGs as bytes.

    Used by the ``preview_boxes_api`` server endpoint so callers receive
    all preview images in a single response without needing a shared
    filesystem.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        paths = preview_redaction_boxes(
            pdf_path,
            csv_path,
            out_dir=tmp,
            dpi=dpi,
            max_width=max_width,
            draw_grid=draw_grid,
            pages=pages,
        )
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                zf.write(p, arcname=Path(p).name)
        return buf.getvalue()


# ── CLI entry-point ─────────────────────────────────────────────────────────
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Render proposed redaction boxes from a review CSV onto the original PDF."
    )
    parser.add_argument("pdf", help="Path to the original (un-redacted) PDF")
    parser.add_argument("csv", help="Path to the *_review_file.csv")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for PNGs (default: <csv-dir>/preview/)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="Render DPI (default: 150)"
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Max image width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        default=True,
        help="Draw percentage grid (default: on)",
    )
    parser.add_argument(
        "--no-grid", dest="grid", action="store_false", help="Disable percentage grid"
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="Comma-separated 1-indexed page numbers to render, e.g. 1,3,5 (default: all)",
    )
    args = parser.parse_args()

    pages = None
    if args.pages:
        pages = [int(p.strip()) for p in args.pages.split(",")]

    preview_redaction_boxes(
        args.pdf,
        args.csv,
        out_dir=args.out_dir,
        dpi=args.dpi,
        max_width=args.max_width,
        draw_grid=args.grid,
        pages=pages,
    )


if __name__ == "__main__":
    _main()
