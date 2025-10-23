#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting in APP_MODE: $APP_MODE"

# --- Start the app based on mode ---

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

        # Start uvicorn server.
        echo "Starting with Uvicorn on $GRADIO_SERVER_NAME:$GRADIO_SERVER_PORT"
        exec su -s /bin/sh user -c "uvicorn app:app \
            --host $GRADIO_SERVER_NAME \
            --port $GRADIO_SERVER_PORT \
            --proxy-headers"
    else
        echo "Starting in Gradio mode..."
        exec su -s /bin/sh user -c "python app.py"
    fi    
fi