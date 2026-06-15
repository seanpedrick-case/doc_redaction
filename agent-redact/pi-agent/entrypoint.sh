#!/bin/sh
set -e

echo "Starting Pi agent (profile=${PI_DEPLOYMENT_PROFILE:-unknown})"

for dir in \
    "${PI_CODING_AGENT_DIR:-/tmp/pi-agent}" \
    "${PI_WORKSPACE_DIR:-/home/user/app/workspace}" \
    "${PI_UPLOAD_ROOT:-/tmp/gradio}" \
    "${PI_SESSION_DIR:-/tmp/pi-sessions}" \
    "${ACCESS_LOGS_FOLDER:-/tmp/pi-logs}" \
    "${USAGE_LOGS_FOLDER:-/tmp/pi-usage}" \
    "${FEEDBACK_LOGS_FOLDER:-/tmp/pi-feedback}" \
    "${MPLCONFIGDIR:-/tmp/matplotlib_cache}" \
    "${XDG_CACHE_HOME:-/tmp/xdg_cache/user_1000}"; do
    mkdir -p "$dir" 2>/dev/null || true
    if [ ! -w "$dir" ]; then
        echo "WARNING: Directory $dir is not writable by current user (uid=$(id -u)). File I/O may fail." >&2
    fi
done

cd "${PI_WORKDIR:-/workspace/doc_redaction}"

echo "Entrypoint environment: PI_WORKSPACE_DIR=${PI_WORKSPACE_DIR:-} PI_UI_HOST=${PI_UI_HOST:-} PI_UI_PORT=${PI_UI_PORT:-} PI_GRADIO_PORT=${PI_GRADIO_PORT:-} GRADIO_SERVER_NAME=${GRADIO_SERVER_NAME:-} GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT:-} RUN_FASTAPI=${RUN_FASTAPI:-}"

python3 agent-redact/pi/pi_agent_config.py
if [ "${RUN_FASTAPI:-False}" = "True" ]; then
  exec uvicorn gradio_app:app \
    --app-dir agent-redact/pi \
    --host "${GRADIO_SERVER_NAME:-0.0.0.0}" \
    --port "${PI_GRADIO_PORT:-${GRADIO_SERVER_PORT:-7860}}" \
    --proxy-headers \
    --forwarded-allow-ips "*"
else
  exec python3 agent-redact/pi/gradio_app.py
fi
