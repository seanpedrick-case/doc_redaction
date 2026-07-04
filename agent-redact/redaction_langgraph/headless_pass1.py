#!/usr/bin/env python3
"""Headless Pass 1 spike: LangGraph doc_redact without Gradio UI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AGENT_REDACT = Path(__file__).resolve().parents[1]
_PI = _AGENT_REDACT / "pi"
for path in (_REPO_ROOT, _AGENT_REDACT, _PI):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bootstrap_pi_config import ensure_pi_config_env  # noqa: E402

ensure_pi_config_env(_REPO_ROOT)

from langchain_core.messages import HumanMessage  # noqa: E402

from redaction_langgraph.graph import build_redaction_agent  # noqa: E402
from redaction_langgraph.tools import run_doc_redact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headless LangGraph Pass 1 spike.")
    parser.add_argument(
        "--pdf", required=True, help="PDF path (absolute or under workspace)."
    )
    parser.add_argument(
        "--dest",
        default="redact/output_redact",
        help="Destination directory relative to workspace (default: redact/output_redact).",
    )
    parser.add_argument(
        "--direct-tool",
        action="store_true",
        help="Call run_doc_redact directly instead of the LangGraph agent.",
    )
    parser.add_argument("--ocr-method", default=None)
    parser.add_argument("--pii-method", default=None)
    args = parser.parse_args(argv)

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    if args.direct_tool:
        result = run_doc_redact(
            str(pdf_path.name),
            args.dest,
            session_hash=None,
            ocr_method=args.ocr_method,
            pii_method=args.pii_method,
        )
        print(result)
        return 0

    graph, system_message = build_redaction_agent(session_hash=None)
    prompt = (
        f"Run Pass 1 redaction on `{pdf_path.name}` and save outputs under `{args.dest}/`. "
        "Use doc_redact with workspace-relative paths."
    )
    final_messages: list = []
    for event in graph.stream(
        {"messages": [system_message, HumanMessage(content=prompt)]},
        stream_mode="updates",
    ):
        for _node, update in event.items():
            final_messages.extend(update.get("messages") or [])

    print(json.dumps({"messages": [str(m) for m in final_messages]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
