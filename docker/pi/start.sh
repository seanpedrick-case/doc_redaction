#!/usr/bin/env bash
# Start Gradio Pi chat UI in the background; keep container alive for `docker compose exec pi-agent pi`.
set -euo pipefail
cd /workspace/doc_redaction
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7862}"
export PYTHONPATH="/workspace/doc_redaction/docker/pi:${PYTHONPATH:-}"
python3 docker/pi/gradio_app.py &
echo "Pi Gradio UI on http://${GRADIO_SERVER_NAME}:${GRADIO_SERVER_PORT}"
exec sleep infinity
