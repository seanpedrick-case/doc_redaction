#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting in APP_MODE: $APP_MODE"

# --- Ensure application directories are writable by the current user ---
# This is important when Docker volumes are bind-mounted from the host and
# the host directory may be owned by root (uid 0), which would prevent the
# non-root container user (uid 1000) from writing output/input files.
for dir in \
    "${GRADIO_OUTPUT_FOLDER:-/home/user/app/output}" \
    "${GRADIO_INPUT_FOLDER:-/home/user/app/input}" \
    "${GRADIO_TEMP_DIR:-/tmp/gradio_tmp}" \
    "${ACCESS_LOGS_FOLDER:-/home/user/app/logs}" \
    "${USAGE_LOGS_FOLDER:-/home/user/app/usage}" \
    "${FEEDBACK_LOGS_FOLDER:-/home/user/app/feedback}" \
    "${CONFIG_FOLDER:-/home/user/app/config}"; do
    mkdir -p "$dir" 2>/dev/null || true
    if [ ! -w "$dir" ]; then
        echo "WARNING: Directory $dir is not writable by current user (uid=$(id -u)). File I/O will fail." >&2
    fi
done

# --- Start the app based on mode ---

if [ "$APP_MODE" = "lambda" ]; then
    echo "Starting in Lambda mode..."
    # The CMD from Dockerfile will be passed as "$@"
    exec python -m awslambdaric "$@"
else
    echo "Starting in Gradio/FastAPI mode..."

    if [ "$RUN_FASTAPI" = "True" ]; then
        echo "Starting in FastAPI mode..."
        
        GRADIO_SERVER_NAME=${GRADIO_SERVER_NAME:-0.0.0.0}
        GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT:-7860}

        # Start uvicorn server.
        echo "Starting with Uvicorn on $GRADIO_SERVER_NAME:$GRADIO_SERVER_PORT"
        exec uvicorn app:app \
            --host $GRADIO_SERVER_NAME \
            --port $GRADIO_SERVER_PORT \
            --proxy-headers \
            --forwarded-allow-ips "*"
    else
        echo "Starting in Gradio mode..."
        exec python app.py
    fi    
fi