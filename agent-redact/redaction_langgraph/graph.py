"""LangGraph ReAct agent for document redaction orchestration."""

from __future__ import annotations

import os

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

_SYSTEM_PROMPT = """You are a document redaction assistant for the doc_redaction application.

Use only the provided tools — never run shell commands or access paths outside the session workspace.
**Do not read `.pi/skills/` or `skills/` files** — skill playbooks are for the Pi coding agent only.
Start with `list_workspace_files` and `doc_redact` when the user prompt includes a document path.

TOOL ARGUMENT FORMAT (critical — wrong format wastes the whole turn):
  Args are flat JSON strings. Nesting is WRONG.
    Correct:  {"pdf_relative_path": "file.pdf"}
    Correct:  {"relative_path": "fix_review.py", "content": "import csv\\n..."}
    Wrong:    {"pdf_relative_path": {}}
    Wrong:    {"pdf_relative_path": {"relative_path": "file.pdf"}}
    Wrong:    {"relative_path": {"relative_path": "fix_review.py"}}
  After the same tool error twice, stop and rebuild args from scratch using flat strings.

**Pass 1 is not complete after doc_redact.** You must finish the full workflow in this turn unless the user
explicitly asks to stop:

1. list_workspace_files — locate the uploaded PDF
2. doc_redact — initial redaction; artifacts land under redact/<document>/output_redact/
   (result includes review_csv_relative_path and ocr_words_csv_relative_path when available)
   Cover **all** User redaction requirements on this call — not only faces/signatures:
   - Names: default entities already include PERSON (and related types). You do not need to
     replace the entity list with faces alone.
   - Explicit org/place/phrase terms from the user (e.g. "Lambeth", "Lambeth 2030"): pass them
     in deny_list (CUSTOM is already in the defaults). Example:
     {"pdf_relative_path": "file.pdf", "deny_list": ["Lambeth", "Lambeth 2030"],
      "redact_entities": ["CUSTOM_VLM_FACES"]}
   - Faces/photos: append CUSTOM_VLM_FACES. Signatures: append CUSTOM_VLM_SIGNATURE
     (AWS Textract OCR: also handwrite_signature_checkbox including "Extract signatures").
   Passing only the VLM extras is fine — they are merged onto the default entity list.
   Do not add VLM entities unless the user asked for faces/signatures (they are slower).
   Always use flat string paths: {"pdf_relative_path": "file.pdf"} — never nest relative_path.
3. Edit the review CSV to satisfy **User redaction requirements**:
   - Write ONE compact fix_policy.py (derive rows from OCR/review CSV — do not hard-code bboxes)
   - Call run_workspace_python_script IMMEDIATELY — never rewrite the same .py without running it
   - Keep each write_workspace_text body modest (under ~80 lines / ~24KB)
   - Preserve CSV headers, utf-8-sig encoding, and numeric bbox values in [0, 1]
     (never "placeholder", "N/A", or empty strings for xmin/xmax/ymin/ymax)
   - color column must be a tuple string like "(0, 0, 0)" with ints 0–255
     (invalid colors are auto-repaired to black when possible)
4. verify_coverage — pre-apply check on the review CSV (word OCR CSV auto-discovered;
   pass ocr_words_csv_relative_path only if discovery fails). Fix until pass_strict is true.
5. review_apply — **once** on the original PDF + edited review CSV; save under
   redact/<document>/review/output_review_final/
6. verify_coverage again on the **post-apply** *_redacted.pdf from review_apply
   (pass redacted_pdf_relative_path as that PDF only — never the review CSV; omit it for pre-apply)

Do not stop after step 2 or after a failed verify_coverage — read tool errors, fix paths/CSV, and continue.
Prefer relative paths within the session workspace. Download artifacts via tool results; never assume shared disk
with the remote doc_redaction server except files already saved in your workspace.
"""


def _build_llm():
    from langchain_openai import ChatOpenAI

    provider = (os.environ.get("AGENT_DEFAULT_PROVIDER") or "llama-cpp").strip().lower()
    if provider in {"amazon-bedrock", "bedrock"}:
        from langchain_aws import ChatBedrockConverse

        model_id = (
            os.environ.get("AGENT_DEFAULT_MODEL") or "anthropic.claude-sonnet-4-6"
        ).strip()
        return ChatBedrockConverse(
            model=model_id, region_name=os.environ.get("AWS_REGION")
        )
    if provider in {"google-gemini", "gemini"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_id = (
            os.environ.get("AGENT_DEFAULT_MODEL") or "gemini-flash-latest"
        ).strip()
        return ChatGoogleGenerativeAI(
            model=model_id, google_api_key=os.environ.get("GEMINI_API_KEY")
        )

    base_url = (
        os.environ.get("AGENT_LLAMA_BASE_URL") or "http://127.0.0.1:8080/v1"
    ).rstrip("/")
    model_id = (
        os.environ.get("AGENT_LLAMA_MODEL_ID")
        or os.environ.get("AGENT_DEFAULT_MODEL")
        or "local"
    ).strip()
    from redaction_langgraph.message_context import langgraph_max_output_tokens

    return ChatOpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        model=model_id,
        temperature=0.2,
        max_tokens=langgraph_max_output_tokens(),
    )


def build_redaction_agent(
    session_hash: str | None,
    *,
    aggressive_compaction: bool = False,
):
    """Compile a ReAct agent with session-scoped tools.

    When compaction is enabled (default), attaches a ``pre_model_hook`` that
    trims LLM input to fit ``AGENT_LLAMA_CONTEXT_WINDOW`` without overwriting
    the full graph message history. Pass ``aggressive_compaction=True`` for
    the one-shot overflow retry path (halved token budget).
    """
    from redaction_langgraph.message_context import (
        build_pre_model_hook,
        langgraph_compaction_enabled,
    )
    from redaction_langgraph.tools import build_langgraph_tools

    llm = _build_llm()
    tools = build_langgraph_tools(session_hash)
    hook = None
    if langgraph_compaction_enabled():
        hook = build_pre_model_hook(aggressive=aggressive_compaction)
    graph = create_react_agent(llm, tools, pre_model_hook=hook)
    return graph, SystemMessage(content=_SYSTEM_PROMPT)


def graph_recursion_limit() -> int:
    raw = (os.environ.get("LANGGRAPH_RECURSION_LIMIT") or "50").strip()
    try:
        return max(10, int(raw))
    except ValueError:
        return 50
