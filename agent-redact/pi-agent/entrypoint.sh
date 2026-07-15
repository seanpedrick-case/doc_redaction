#!/bin/sh
set -e

echo "Starting Pi agent (profile=${AGENT_DEPLOYMENT_PROFILE:-unknown})"

for dir in \
    "${AGENT_CODING_AGENT_DIR:-/tmp/agent-coding}" \
    "${AGENT_WORKSPACE_DIR:-/home/user/app/workspace}" \
    "${AGENT_UPLOAD_ROOT:-/tmp/gradio}" \
    "${AGENT_SESSION_DIR:-/tmp/agent-sessions}" \
    "${ACCESS_LOGS_FOLDER:-/tmp/agent-logs}" \
    "${USAGE_LOGS_FOLDER:-/tmp/agent-usage}" \
    "${FEEDBACK_LOGS_FOLDER:-/tmp/agent-feedback}" \
    "${MPLCONFIGDIR:-/tmp/matplotlib_cache}" \
    "${XDG_CACHE_HOME:-/tmp/xdg_cache/user_1000}"; do
    mkdir -p "$dir" 2>/dev/null || true
    if [ ! -w "$dir" ]; then
        echo "WARNING: Directory $dir is not writable by current user (uid=$(id -u)). File I/O may fail." >&2
    fi
done

cd "${AGENT_WORKDIR:-/workspace/doc_redaction}"

echo "Entrypoint environment: AGENT_WORKSPACE_DIR=${AGENT_WORKSPACE_DIR:-} AGENT_UI_HOST=${AGENT_UI_HOST:-} AGENT_UI_PORT=${AGENT_UI_PORT:-} AGENT_GRADIO_PORT=${AGENT_GRADIO_PORT:-} GRADIO_SERVER_NAME=${GRADIO_SERVER_NAME:-} GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT:-} RUN_FASTAPI=${RUN_FASTAPI:-}"

python3 agent-redact/pi/pi_agent_config.py
if [ "${RUN_FASTAPI:-False}" = "True" ]; then
  exec uvicorn gradio_app:app \
    --app-dir agent-redact/shared \
    --host "${GRADIO_SERVER_NAME:-0.0.0.0}" \
    --port "${AGENT_GRADIO_PORT:-${GRADIO_SERVER_PORT:-7860}}" \
    --proxy-headers \
    --forwarded-allow-ips "*"
else
  exec python3 agent-redact/shared/gradio_app.py
fi
