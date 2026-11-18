#!/usr/bin/env python3
"""Offline replay: score an output folder against gold (main app settings JSON)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from derive_app_eval_instructions import settings_from_json  # noqa: E402
from post_run_app_eval import write_app_post_run_eval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True, help="Input PDF path")
    parser.add_argument(
        "--settings-json",
        type=Path,
        required=True,
        help="AppEvalRedactionSettings JSON",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=None,
        help="Directory to discover review/OCR outputs (default: PDF parent)",
    )
    args = parser.parse_args()
    settings = settings_from_json(str(args.settings_json))
    search = args.search_root or args.pdf.parent
    path, status = write_app_post_run_eval(
        pdf_path=args.pdf.resolve(),
        document_name=args.pdf.name,
        settings=settings,
        search_root=search.resolve(),
    )
    print(status)
    if path:
        print(json.dumps({"eval_summary": str(path)}, indent=2))
    return 0 if path else 1


if __name__ == "__main__":
    raise SystemExit(main())
