# Pi-based agentic document redaction: local Docker orchestration and Hugging Face Space packaging.

Supports three orchestration backends via `AGENT_ORCHESTRATOR` (`pi` default, `langgraph`, `agentcore`). See [`pi/agent/README.md`](pi/agent/README.md).

| Path | Purpose |
|------|---------|
| [`pi/`](pi/) | Gradio UI, Pi RPC client, remote redaction helpers, runtime config |
| [`agentcore/`](agentcore/) | Bedrock AgentCore runtime entrypoint + **[install guide](agentcore/README.md)** |
| [`pi-agent/`](pi-agent/) | Pi Docker image (`dev` + `runtime` targets), sync script, and manifest |
| [`requirements_pi_agent.txt`](requirements_pi_agent.txt) | Python deps for the Pi agent image |

Per-user output isolation uses Gradio `session_hash` subfolders under `PI_WORKSPACE_DIR` (see `agent-redact/pi/session_workspace.py`). Enabled by default locally and on HF Spaces. Set `PI_SESSION_WORKSPACE=false` only if you want one shared workspace tree for all sessions.

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
