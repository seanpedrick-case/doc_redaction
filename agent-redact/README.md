# Agent redaction (Pi)

Pi-based agentic document redaction: local Docker orchestration and Hugging Face Space packaging.

| Path | Purpose |
|------|---------|
| [`pi/`](pi/) | Gradio UI, Pi RPC client, remote redaction helpers, runtime config |
| [`pi-agent/`](pi-agent/) | HF Space Dockerfile, sync script, and manifest |
| [`requirements_pi_agent.txt`](requirements_pi_agent.txt) | Python deps for the Pi agent image |

Per-user output isolation on HF Spaces uses Gradio `session_hash` subfolders under `PI_WORKSPACE_DIR` (see `agent-redact/pi/session_workspace.py`). Set `PI_SESSION_WORKSPACE=false` to use a single shared workspace (local dev only).

## Local Docker

Use the `pi-agent` service in [`docker-compose_llama_agentic.yml`](../docker-compose_llama_agentic.yml) (profile `27b_36`). See [`pi/agent/README.md`](pi/agent/README.md).

## Hugging Face Space

Build from repo root:

```bash
docker build -f agent-redact/pi-agent/Dockerfile .
```

Sync to Space on pushes to `dev` via [`.github/workflows/sync-pi-agent-space.yml`](../.github/workflows/sync-pi-agent-space.yml).
