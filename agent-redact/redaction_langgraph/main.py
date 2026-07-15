"""Bedrock AgentCore runtime entrypoint wrapping the LangGraph redaction agent."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AGENT_REDACT = Path(__file__).resolve().parents[1]
for path in (_REPO_ROOT, _AGENT_REDACT, _AGENT_REDACT / "shared"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bootstrap_pi_config import ensure_pi_config_env  # noqa: E402

ensure_pi_config_env(_REPO_ROOT)

# Must run before LangChain imports (OpenInference patch order).
try:
    from eval.arize_monitoring import setup_arize_ax_tracing  # noqa: E402

    setup_arize_ax_tracing()
except ImportError:
    pass

from bedrock_agentcore import BedrockAgentCoreApp  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from redaction_langgraph.graph import build_redaction_agent  # noqa: E402

app = BedrockAgentCoreApp()


@app.entrypoint
async def handler(request: dict):
    """Stream LangGraph agent events for one user prompt."""
    prompt = str(request.get("prompt") or request.get("message") or "").strip()
    session_hash = str(request.get("session_hash") or "").strip() or None
    if not prompt:
        yield {"type": "error", "message": "prompt is required"}
        return

    from eval.arize_monitoring import arize_session_context, langgraph_trace_config

    from redaction_langgraph.graph import graph_recursion_limit

    graph, system_message = build_redaction_agent(session_hash)
    inputs = {"messages": [system_message, HumanMessage(content=prompt)]}
    yield {"type": "agent_start"}

    stream_config = langgraph_trace_config(
        session_hash, recursion_limit=graph_recursion_limit()
    )
    with arize_session_context(session_hash):
        for event in graph.stream(inputs, stream_mode="updates", config=stream_config):
            for node, update in event.items():
                messages = update.get("messages") or []
                for message in messages:
                    content = getattr(message, "content", "")
                    yield {
                        "type": "message_update",
                        "node": node,
                        "role": getattr(message, "type", "unknown"),
                        "content": content,
                    }

    yield {"type": "agent_end", "message": "Agent finished."}


if __name__ == "__main__":
    app.run()
