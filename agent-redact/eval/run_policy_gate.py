#!/usr/bin/env python3
"""Run the non-gold policy quality gate (``verify_redaction_coverage``).

Uses the same Pass 1 checks as the LangGraph ``verify_coverage`` tool:
uncovered ``must_redact`` terms, over-redacted ``must_not_redact`` terms,
optional post-apply text-layer checks on ``*_redacted.pdf``.

Fixture mode (recommended)::

    python agent-redact/eval/run_policy_gate.py --fixture agent-redact/eval/fixtures/example_smoke

Direct mode::

    python agent-redact/eval/run_policy_gate.py \\
        --review-csv path/review_file.csv \\
        --ocr-words-csv path/ocr_results_with_words.csv \\
        --must-redact-file must_redact.txt \\
        --redacted-pdf path/doc_redacted.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repo_paths() -> tuple[Path, Path]:
    eval_dir = Path(__file__).resolve().parent
    agent_redact = eval_dir.parent
    repo_root = agent_redact.parent
    return repo_root, eval_dir


def _ensure_tools_importable() -> None:
    repo_root, _ = _repo_paths()
    tools_parent = str(repo_root)
    if tools_parent not in sys.path:
        sys.path.insert(0, tools_parent)


def _read_pattern_file(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _discover_fixture_files(fixture_dir: Path) -> dict[str, Path | None]:
    """Resolve standard fixture filenames under *fixture_dir*."""
    root = fixture_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Fixture directory not found: {root}")

    def first(*names: str) -> Path | None:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate
        # Prefer any matching glob for review / ocr when exact names missing.
        return None

    review = first(
        "agent_review_file.csv",
        "ideal_review_file.csv",
        "review_file.csv",
    )
    if review is None:
        matches = sorted(root.glob("*review_file*.csv"))
        review = matches[0] if matches else None

    ocr = first(
        "ocr_results_with_words.csv",
        "ocr_words.csv",
    )
    if ocr is None:
        matches = sorted(root.glob("*ocr_results_with_words*"))
        ocr = matches[0] if matches else None

    redacted = first("redacted.pdf")
    if redacted is None:
        matches = sorted(root.glob("*_redacted.pdf"))
        redacted = matches[0] if matches else None

    return {
        "review_csv": review,
        "ocr_words_csv": ocr,
        "must_redact_file": first("must_redact.txt"),
        "must_not_redact_file": first("must_not_redact.txt"),
        "redacted_pdf": redacted,
        "instructions": first("instructions.txt"),
        "tool_args": first("tool_args.json", "observed_tool_args.json"),
    }


def run_policy_gate(
    *,
    review_csv: Path,
    ocr_words_csv: Path,
    must_redact: list[str] | None = None,
    must_not_redact: list[str] | None = None,
    redacted_pdf: Path | None = None,
    sample_pixels: bool = False,
) -> dict[str, Any]:
    """Execute coverage verification and return a JSON-serialisable summary."""
    _ensure_tools_importable()
    from tools.verify_redaction_coverage import verify_redaction_coverage

    report = verify_redaction_coverage(
        review_csv,
        ocr_words_csv,
        must_redact=must_redact or [],
        must_not_redact=must_not_redact or [],
        redacted_pdf_path=redacted_pdf,
        sample_pixels=sample_pixels,
    )
    payload = report.to_dict()
    summary = {
        "pass_strict": bool(report.pass_strict),
        "pass_with_cleanup": bool(report.pass_with_cleanup),
        "pages_flagged_for_vlm": list(report.pages_flagged_for_vlm or []),
        "pages_needing_csv_cleanup": list(
            getattr(report, "pages_needing_csv_cleanup", []) or []
        ),
        "review_csv": str(review_csv),
        "ocr_words_csv": str(ocr_words_csv),
        "redacted_pdf": str(redacted_pdf) if redacted_pdf else None,
        "must_redact_count": len(must_redact or []),
        "must_not_redact_count": len(must_not_redact or []),
        "report": payload,
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Policy quality gate via verify_redaction_coverage."
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Fixture directory (see agent-redact/eval/fixtures/README.md).",
    )
    parser.add_argument(
        "--review-csv", type=Path, help="Agent or candidate review CSV."
    )
    parser.add_argument(
        "--ocr-words-csv",
        type=Path,
        help="Word-level OCR CSV (*ocr_results_with_words*).",
    )
    parser.add_argument(
        "--must-redact-file",
        type=Path,
        help="Text file: one regex per line (must_redact).",
    )
    parser.add_argument(
        "--must-not-redact-file",
        type=Path,
        help="Text file: one regex per line (must_not_redact).",
    )
    parser.add_argument(
        "--redacted-pdf",
        type=Path,
        help="Post-apply *_redacted.pdf for text-layer leak checks.",
    )
    parser.add_argument(
        "--sample-pixels",
        action="store_true",
        help="Enable optional centre-pixel sampling on redacted PDF.",
    )
    parser.add_argument(
        "--require-pass-strict",
        action="store_true",
        default=True,
        help="Exit 1 when pass_strict is false (default).",
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Always exit 0 (still print JSON); useful for collecting baselines.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Write full summary JSON to this path.",
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help=(
            "Run twice: gold lists (fixture files) and agent lists from "
            "--agent-tool-args / fixture tool_args.json. Compliance uses gold; "
            "agent run is diagnostic."
        ),
    )
    parser.add_argument(
        "--agent-tool-args",
        type=Path,
        help="JSON tool_calls recording for --dual agent-list gate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    review_csv = args.review_csv
    ocr_words_csv = args.ocr_words_csv
    must_redact_file = args.must_redact_file
    must_not_redact_file = args.must_not_redact_file
    redacted_pdf = args.redacted_pdf
    tool_args_path = args.agent_tool_args

    if args.fixture:
        discovered = _discover_fixture_files(args.fixture)
        review_csv = review_csv or discovered["review_csv"]
        ocr_words_csv = ocr_words_csv or discovered["ocr_words_csv"]
        must_redact_file = must_redact_file or discovered["must_redact_file"]
        must_not_redact_file = (
            must_not_redact_file or discovered["must_not_redact_file"]
        )
        redacted_pdf = redacted_pdf or discovered["redacted_pdf"]
        tool_args_path = tool_args_path or discovered.get("tool_args")

    if review_csv is None or not Path(review_csv).is_file():
        print("review CSV is required (--review-csv or --fixture)", file=sys.stderr)
        return 2
    if ocr_words_csv is None or not Path(ocr_words_csv).is_file():
        print(
            "OCR words CSV is required (--ocr-words-csv or --fixture "
            "with ocr_results_with_words.csv)",
            file=sys.stderr,
        )
        return 2

    must_redact = _read_pattern_file(must_redact_file)
    must_not_redact = _read_pattern_file(must_not_redact_file)
    redacted = Path(redacted_pdf) if redacted_pdf else None

    try:
        gold_summary = run_policy_gate(
            review_csv=Path(review_csv),
            ocr_words_csv=Path(ocr_words_csv),
            must_redact=must_redact,
            must_not_redact=must_not_redact,
            redacted_pdf=redacted,
            sample_pixels=bool(args.sample_pixels),
        )
    except Exception as exc:
        print(f"policy gate failed: {exc}", file=sys.stderr)
        return 2

    output: dict[str, Any] = {
        "pass_strict": gold_summary["pass_strict"],
        "pass_with_cleanup": gold_summary["pass_with_cleanup"],
        "pages_flagged_for_vlm": gold_summary["pages_flagged_for_vlm"],
        "pages_needing_csv_cleanup": gold_summary["pages_needing_csv_cleanup"],
        "review_csv": gold_summary["review_csv"],
        "lists_source": "gold",
    }

    if args.dual:
        if tool_args_path is None or not Path(tool_args_path).is_file():
            print(
                "--dual requires --agent-tool-args or fixture tool_args.json",
                file=sys.stderr,
            )
            return 2
        # Local import keeps eval usable without path hacks when run as script.
        eval_dir = Path(__file__).resolve().parent
        if str(eval_dir) not in sys.path:
            sys.path.insert(0, str(eval_dir))
        from compare_policy_lists import (  # noqa: WPS433
            extract_lists_from_tool_calls,
            load_observed_tool_args,
        )

        observed = extract_lists_from_tool_calls(
            load_observed_tool_args(Path(tool_args_path)), aggregate="union"
        )
        try:
            agent_summary = run_policy_gate(
                review_csv=Path(review_csv),
                ocr_words_csv=Path(ocr_words_csv),
                must_redact=observed["must_redact"],
                must_not_redact=observed["must_not_redact"],
                redacted_pdf=redacted,
                sample_pixels=bool(args.sample_pixels),
            )
        except Exception as exc:
            print(f"agent-list policy gate failed: {exc}", file=sys.stderr)
            return 2
        output = {
            "gold": {
                "pass_strict": gold_summary["pass_strict"],
                "pass_with_cleanup": gold_summary["pass_with_cleanup"],
                "must_redact_count": gold_summary["must_redact_count"],
                "must_not_redact_count": gold_summary["must_not_redact_count"],
            },
            "agent": {
                "pass_strict": agent_summary["pass_strict"],
                "pass_with_cleanup": agent_summary["pass_with_cleanup"],
                "must_redact_count": agent_summary["must_redact_count"],
                "must_not_redact_count": agent_summary["must_not_redact_count"],
                "must_redact": observed["must_redact"],
                "must_not_redact": observed["must_not_redact"],
                "deny_list": observed["deny_list"],
            },
            "diagnostic": {
                "gold_fail_agent_pass": (
                    not gold_summary["pass_strict"] and agent_summary["pass_strict"]
                ),
                "gold_pass_agent_empty_must_redact": (
                    gold_summary["pass_strict"] and not observed["must_redact"]
                ),
            },
            # Compliance signal remains gold-based.
            "pass_strict": gold_summary["pass_strict"],
            "review_csv": gold_summary["review_csv"],
        }

    print(json.dumps(output, indent=2))
    full_out = {"gold_report": gold_summary, **output} if args.dual else gold_summary
    if args.dual:
        full_out["agent_report"] = agent_summary  # type: ignore[name-defined]
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(full_out, indent=2, default=str), encoding="utf-8"
        )

    if args.allow_fail:
        return 0
    compliance = (
        gold_summary["pass_strict"]
        if args.dual
        else output.get("pass_strict", gold_summary["pass_strict"])
    )
    if args.require_pass_strict and not compliance:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
