#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting in APP_MODE: $APP_MODE"

# --- STEP 1: Load environment file from S3 ---
if [ -n "$CONFIG_ENV_S3_URI" ]; then
    echo "Downloading environment config from $CONFIG_ENV_S3_URI"
    aws s3 cp "$CONFIG_ENV_S3_URI" /tmp/config.env
    set -a  # Automatically export all variables
    . /tmp/config.env
    set +a
else
    echo "No CONFIG_ENV_S3_URI provided; using default env vars"
fi

# --- STEP 2: Start the app based on mode ---

if [ "$APP_MODE" = "lambda" ]; then
    echo "Starting in Lambda mode..."
    # The CMD from Dockerfile will be passed as "$@"
    exec python -m awslambdaric "$@"
else
    echo "Starting in Gradio/FastAPI mode..."

    if [ "$RUN_FASTAPI" = "1" ]; then
        echo "Starting in FastAPI mode..."
        
        GRADIO_SERVER_NAME=${GRADIO_SERVER_NAME:-0.0.0.0}
        GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT:-7860}
        ROOT_PATH=${ROOT_PATH:-/}

        # Start uvicorn server.
        echo "Starting with Uvicorn on $GRADIO_SERVER_NAME:$GRADIO_SERVER_PORT with root path $ROOT_PATH"
        exec uvicorn app:app \
            --host $GRADIO_SERVER_NAME \
            --port $GRADIO_SERVER_PORT \
            --proxy-headers \
            --root-path $ROOT_PATH
    else
        echo "Starting in Gradio mode..."
        exec python app.py
    fi    
fi