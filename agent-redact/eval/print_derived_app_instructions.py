#!/usr/bin/env python3
"""Print derived app eval instructions for pasting into gold ``instructions.txt``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from derive_app_eval_instructions import (  # noqa: E402
    AppEvalRedactionSettings,
    derive_app_eval_instructions,
    settings_from_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings-json",
        type=str,
        help="JSON file with AppEvalRedactionSettings fields",
    )
    parser.add_argument(
        "--entities",
        type=str,
        default="",
        help="Comma-separated entity types (e.g. PERSON,ADDRESS)",
    )
    parser.add_argument(
        "--deny",
        action="append",
        default=[],
        help="Deny list term (repeatable)",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        help="Allow list term (repeatable)",
    )
    parser.add_argument(
        "--text-extraction-only",
        action="store_true",
        help="Extract text only mode",
    )
    parser.add_argument(
        "--ocr-method",
        type=str,
        default="",
        help="Stored in settings JSON sidecar only (not in instruction bullets)",
    )
    parser.add_argument(
        "--pii-method",
        type=str,
        default="",
        help="Stored in settings JSON sidecar only",
    )
    args = parser.parse_args()

    if args.settings_json:
        settings = settings_from_json(args.settings_json)
    else:
        entities = [e.strip() for e in args.entities.split(",") if e.strip()]
        settings = AppEvalRedactionSettings(
            chosen_redact_entities=entities,
            deny_list=list(args.deny),
            allow_list=list(args.allow),
            text_extraction_only=args.text_extraction_only,
            text_extraction_method=args.ocr_method,
            pii_identification_method=args.pii_method,
        )

    print(derive_app_eval_instructions(settings))
    if args.settings_json is None and (args.ocr_method or args.pii_method):
        sidecar = settings.to_settings_dict()
        print("\n# settings sidecar (not part of instructions.txt):", file=sys.stderr)
        print(json.dumps(sidecar, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
