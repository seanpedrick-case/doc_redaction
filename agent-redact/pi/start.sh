#!/usr/bin/env bash
# Start Gradio Pi chat UI in the background; keep container alive for `docker compose exec pi-agent pi`.
set -euo pipefail

export HOME="${HOME:-/home/node}"
export PI_WORKDIR="${PI_WORKDIR:-/workspace/doc_redaction}"
export PYTHONPATH="${PI_WORKDIR}/agent-redact/pi:${PYTHONPATH:-}"

cd "$PI_WORKDIR"

mkdir -p "${PI_WORKSPACE_DIR:-/home/user/app/workspace}"
if [ -n "${PI_SESSION_DIR:-}" ]; then
  mkdir -p "${PI_SESSION_DIR}"
fi
python3 agent-redact/pi/pi_agent_config.py
python3 agent-redact/pi/gradio_app.py &

wait -n
