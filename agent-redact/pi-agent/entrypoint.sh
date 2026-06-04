#!/bin/sh
set -e

echo "Starting Pi agent (profile=${PI_DEPLOYMENT_PROFILE:-unknown})"

for dir in \
    "${PI_CODING_AGENT_DIR:-/tmp/pi-agent}" \
    "${PI_WORKSPACE_DIR:-/home/user/app/workspace}" \
    "${PI_UPLOAD_ROOT:-/tmp/gradio}" \
    "${PI_SESSION_DIR:-/tmp/pi-sessions}" \
    "${MPLCONFIGDIR:-/tmp/matplotlib_cache}" \
    "${XDG_CACHE_HOME:-/tmp/xdg_cache/user_1000}"; do
    mkdir -p "$dir" 2>/dev/null || true
    if [ ! -w "$dir" ]; then
        echo "WARNING: Directory $dir is not writable by current user (uid=$(id -u)). File I/O may fail." >&2
    fi
done

cd "${PI_WORKDIR:-/workspace/doc_redaction}"

python3 agent-redact/pi/pi_agent_config.py
exec python3 agent-redact/pi/gradio_app.py
