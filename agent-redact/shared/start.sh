#!/usr/bin/env bash
# Start Gradio Pi chat UI in the background; keep container alive for `docker compose exec pi-agent pi`.
set -euo pipefail

export HOME="${HOME:-/home/user}"
export AGENT_WORKDIR="${AGENT_WORKDIR:-/workspace/doc_redaction}"
export PYTHONPATH="${AGENT_WORKDIR}:${AGENT_WORKDIR}/agent-redact:${AGENT_WORKDIR}/agent-redact/shared:${AGENT_WORKDIR}/agent-redact/pi:${AGENT_WORKDIR}/agent-redact/agentcore:${PYTHONPATH:-}"

cd "$AGENT_WORKDIR"

export APP_TYPE="${APP_TYPE:-agent}"
# Config file renamed agent.env (legacy: pi_agent.env). Prefer the new name.
if [ -z "${APP_CONFIG_PATH:-}" ]; then
  if [ -f "$AGENT_WORKDIR/config/agent.env" ]; then
    export APP_CONFIG_PATH="$AGENT_WORKDIR/config/agent.env"
  elif [ -f "$AGENT_WORKDIR/config/pi_agent.env" ]; then
    export APP_CONFIG_PATH="$AGENT_WORKDIR/config/pi_agent.env"
  else
    export APP_CONFIG_PATH="$AGENT_WORKDIR/config/agent.env"
  fi
fi

mkdir -p "${AGENT_WORKSPACE_DIR:-/home/user/app/workspace}"
python3 agent-redact/pi/pi_agent_config.py

if [ "${RUN_FASTAPI:-False}" = "True" ]; then
  exec uvicorn gradio_app:app \
    --app-dir agent-redact/shared \
    --host "${GRADIO_SERVER_NAME:-0.0.0.0}" \
    --port "${AGENT_GRADIO_PORT:-${GRADIO_SERVER_PORT:-7862}}"
else
  python3 agent-redact/shared/gradio_app.py &
fi

wait -n
