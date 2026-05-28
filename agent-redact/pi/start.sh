#!/usr/bin/env bash
# Start Gradio Pi chat UI in the background; keep container alive for `docker compose exec pi-agent pi`.
set -euo pipefail

export HOME="${HOME:-/home/node}"
export PI_WORKDIR="${PI_WORKDIR:-/workspace/doc_redaction}"
export PYTHONPATH="${PI_WORKDIR}:${PI_WORKDIR}/agent-redact/pi:${PYTHONPATH:-}"

cd "$PI_WORKDIR"

mkdir -p "${PI_WORKSPACE_DIR:-/home/user/app/workspace}"
python3 agent-redact/pi/pi_agent_config.py

if [ "${RUN_FASTAPI:-False}" = "True" ]; then
  exec uvicorn gradio_app:app \
    --app-dir agent-redact/pi \
    --host "${GRADIO_SERVER_NAME:-0.0.0.0}" \
    --port "${GRADIO_SERVER_PORT:-7862}"
else
  python3 agent-redact/pi/gradio_app.py &
fi

wait -n
