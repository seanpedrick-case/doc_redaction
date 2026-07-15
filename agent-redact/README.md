# Pi-based agentic document redaction: local Docker orchestration and Hugging Face Space packaging.

Supports three orchestration backends via `AGENT_ORCHESTRATOR` (`pi` default, `langgraph`, `agentcore`). See [`pi/agent/README.md`](pi/agent/README.md).

| Path | Purpose |
|------|---------|
| [`shared/`](shared/) | Gradio UI, runtime factory, session workspace, remote redaction helpers |
| [`pi/`](pi/) | Pi RPC client, Pi config generation, skills sync, `agent/*.json` templates |
| [`redaction_langgraph/`](redaction_langgraph/) | LangGraph ReAct agent (curated tools, no shell) |
| [`agentcore/`](agentcore/) | Bedrock AgentCore runtime + Gradio AgentCore clients + **[install guide](agentcore/README.md)** |
| [`eval/`](eval/) | Arize AX / Phoenix tracing for LangGraph |
| [`pi-agent/`](pi-agent/) | Docker image (`dev` + `runtime` targets), sync script, and manifest |
| [`requirements_agent.txt`](requirements_agent.txt) | Python deps for agent-redact Docker images |

Per-user output isolation uses Gradio `session_hash` subfolders under `AGENT_WORKSPACE_DIR` (see `agent-redact/shared/session_workspace.py`). Enabled by default locally and on HF Spaces. Set `AGENT_SESSION_WORKSPACE=false` only if you want one shared workspace tree for all sessions.

## Local Docker

Use the `pi-agent` service in [`docker-compose_llama_agentic.yml`](../docker-compose_llama_agentic.yml) (profile `27b_36`). See [`pi/agent/README.md`](pi/agent/README.md).

## Hugging Face Space

Build from repo root:

```bash
# Production (HF Space / ECS)
docker build -f agent-redact/pi-agent/Dockerfile --target runtime .

# Local compose (bind-mounted repo)
docker build -f agent-redact/pi-agent/Dockerfile --target dev .
```

Sync to Space on pushes to `dev` via [`.github/workflows/sync-pi-agent-space.yml`](../.github/workflows/sync-pi-agent-space.yml).
