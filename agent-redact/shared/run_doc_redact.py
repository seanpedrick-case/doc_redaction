#!/usr/bin/env python3
"""One-shot ``/doc_redact`` CLI for Pi agents (HF Space / split-container backends)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow ``python3 …/run_doc_redact.py`` without installing the package.
_HELPERS = Path(__file__).resolve().parent
if str(_HELPERS) not in sys.path:
    sys.path.insert(0, str(_HELPERS))

from remote_redaction import call_doc_redact  # noqa: E402


def _parse_list(raw: str | None) -> list[str] | None:
    if raw is None or not str(raw).strip():
        return None
    text = str(raw).strip()
    if text.startswith("["):
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [part.strip() for part in text.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run /doc_redact via remote_redaction.make_redaction_client()."
    )
    parser.add_argument(
        "--pdf", required=True, help="Local PDF path (session workspace)."
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Directory for downloaded artifacts (e.g. …/output_redact/).",
    )
    parser.add_argument("--ocr-method", default=None)
    parser.add_argument("--pii-method", default=None)
    parser.add_argument(
        "--deny-list",
        default=None,
        help="Comma-separated or JSON list for CUSTOM deny terms.",
    )
    parser.add_argument(
        "--allow-list",
        default=None,
        help="Comma-separated or JSON list for allow terms.",
    )
    parser.add_argument(
        "--redact-entities",
        default=None,
        help="Comma-separated or JSON list (default: PERSON, EMAIL, …, CUSTOM).",
    )
    parser.add_argument("--page-min", type=int, default=None)
    parser.add_argument("--page-max", type=int, default=None)
    args = parser.parse_args(argv)

    pdf = Path(args.pdf).expanduser().resolve()
    if not pdf.is_file():
        print(f"PDF not found: {pdf}", file=sys.stderr)
        return 2

    result, saved = call_doc_redact(
        pdf,
        args.dest,
        ocr_method=args.ocr_method,
        pii_method=args.pii_method,
        deny_list=_parse_list(args.deny_list),
        allow_list=_parse_list(args.allow_list),
        redact_entities=_parse_list(args.redact_entities),
        page_min=args.page_min,
        page_max=args.page_max,
    )
    message = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else ""
    print(message or "doc_redact completed.")
    for path in saved:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
