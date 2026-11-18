#!/usr/bin/env python3
"""Write input_pdf_sha256 and instructions_sha256 into a gold case manifest.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gold_match import normalize_instructions_text, sha256_file
from run_agent_regression import resolve_gold_case_pdf


def update_manifest_hashes(gold_case_dir: Path) -> dict[str, str]:
    root = gold_case_dir.resolve()
    pdf_path = resolve_gold_case_pdf(root)
    instr_path = root / "instructions.txt"
    if not instr_path.is_file():
        raise FileNotFoundError(f"Missing instructions.txt in {root}")

    pdf_hash = sha256_file(pdf_path)
    instr_bytes = normalize_instructions_text(
        instr_path.read_text(encoding="utf-8")
    ).encode("utf-8")
    import hashlib

    instr_hash = hashlib.sha256(instr_bytes).hexdigest()

    manifest_path = root / "manifest.json"
    data: dict = {}
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    data["input_pdf_sha256"] = pdf_hash
    data["instructions_sha256"] = instr_hash
    if not str(data.get("input_pdf") or "").strip():
        data["input_pdf"] = pdf_path.name if pdf_path.parent == root else pdf_path.name
    manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return {"input_pdf_sha256": pdf_hash, "instructions_sha256": instr_hash}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Pin PDF + instruction hashes in manifest.json"
    )
    p.add_argument(
        "gold_case_dir",
        type=Path,
        help="Gold case directory (e.g. agent-redact/eval/gold/example_live)",
    )
    args = p.parse_args(argv)
    try:
        out = update_manifest_hashes(args.gold_case_dir)
    except (OSError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
