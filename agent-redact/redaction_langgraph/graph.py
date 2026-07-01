"""LangGraph ReAct agent for document redaction orchestration."""

from __future__ import annotations

import os

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

_SYSTEM_PROMPT = """You are a document redaction assistant for the doc_redaction application.

Use only the provided tools — never run shell commands or access paths outside the session workspace.
**Do not read `.pi/skills/` or `skills/` files** — skill playbooks are for the Pi coding agent only.
Start with `list_workspace_files` and `doc_redact` when the user prompt includes a document path.

**Pass 1 is not complete after doc_redact.** You must finish the full workflow in this turn unless the user
explicitly asks to stop:

1. list_workspace_files — locate the uploaded PDF
2. doc_redact — initial redaction; artifacts land under redact/<document>/output_redact/
3. Edit the review CSV to satisfy **User redaction requirements**:
   - read_workspace_text / write_workspace_text for small edits, or
   - write_workspace_text a fix_policy.py script, then run_workspace_python_script
   - Preserve CSV headers, utf-8-sig encoding, and bbox values in [0, 1]
4. verify_coverage — pre-apply check on the review CSV (+ auto-discovered word OCR CSV).
   Fix issues until pass_strict is true (or report why it cannot be reached).
5. review_apply — **once** on the original PDF + edited review CSV; save under
   redact/<document>/review/output_review_final/
6. verify_coverage again on the **post-apply** *_redacted.pdf from review_apply

Do not stop after step 2 or after a failed verify_coverage — read tool errors, fix paths/CSV, and continue.
Prefer relative paths within the session workspace. Download artifacts via tool results; never assume shared disk
with the remote doc_redaction server except files already saved in your workspace.
"""


def _build_llm():
    from langchain_openai import ChatOpenAI

    provider = (os.environ.get("PI_DEFAULT_PROVIDER") or "llama-cpp").strip().lower()
    if provider in {"amazon-bedrock", "bedrock"}:
        from langchain_aws import ChatBedrockConverse

        model_id = (
            os.environ.get("PI_DEFAULT_MODEL") or "anthropic.claude-sonnet-4-6"
        ).strip()
        return ChatBedrockConverse(
            model=model_id, region_name=os.environ.get("AWS_REGION")
        )
    if provider in {"google-gemini", "gemini"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_id = (os.environ.get("PI_DEFAULT_MODEL") or "gemini-flash-latest").strip()
        return ChatGoogleGenerativeAI(
            model=model_id, google_api_key=os.environ.get("GEMINI_API_KEY")
        )

    base_url = (
        os.environ.get("PI_LLAMA_BASE_URL") or "http://127.0.0.1:8080/v1"
    ).rstrip("/")
    model_id = (
        os.environ.get("PI_LLAMA_MODEL_ID")
        or os.environ.get("PI_DEFAULT_MODEL")
        or "local"
    ).strip()
    return ChatOpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        model=model_id,
        temperature=0.2,
    )


def build_redaction_agent(session_hash: str | None):
    """Compile a ReAct agent with session-scoped tools."""
    from redaction_langgraph.tools import build_langgraph_tools

    llm = _build_llm()
    tools = build_langgraph_tools(session_hash)
    graph = create_react_agent(llm, tools)
    return graph, SystemMessage(content=_SYSTEM_PROMPT)


def graph_recursion_limit() -> int:
    raw = (os.environ.get("LANGGRAPH_RECURSION_LIMIT") or "50").strip()
    try:
        return max(10, int(raw))
    except ValueError:
        return 50
