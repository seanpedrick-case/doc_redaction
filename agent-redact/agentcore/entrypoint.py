"""Bedrock AgentCore runtime entrypoint wrapping the LangGraph redaction agent."""

from __future__ import annotations

import sys
from pathlib import Path

_AGENTCORE_DIR = Path(__file__).resolve().parent
_AGENT_REDACT = _AGENTCORE_DIR.parent
_REPO_ROOT = _AGENT_REDACT.parent
for path in (_REPO_ROOT, _AGENT_REDACT, _AGENT_REDACT / "pi", _AGENTCORE_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from invoke_agent import bootstrap_runtime_env, invoke_redaction_agent  # noqa: E402

bootstrap_runtime_env(_REPO_ROOT)

from bedrock_agentcore import BedrockAgentCoreApp  # noqa: E402

app = BedrockAgentCoreApp()


@app.entrypoint
async def handler(request: dict):
    """Stream LangGraph agent events for one user prompt."""
    async for event in invoke_redaction_agent(request):
        yield event


if __name__ == "__main__":
    app.run()
