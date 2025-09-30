#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting in APP_MODE: $APP_MODE"

if [ "$APP_MODE" = "lambda" ]; then
    echo "Starting in Lambda mode..."
    # The CMD from Dockerfile will be passed as "$@"
    exec python -m awslambdaric "$@"
else
    echo "Starting in Gradio/FastAPI mode..."

    if [ "$RUN_FASTAPI" = "1" ]; then
        echo "Starting in FastAPI mode..."
        # Set default values for Gunicorn configuration if they are not provided
        # This makes your container more flexible.
        GUNICORN_HOST=${GRADIO_SERVER_NAME:-0.0.0.0}
        GUNICORN_PORT=${GRADIO_SERVER_PORT:-7860}
        GUNICORN_WORKERS=${DEFAULT_CONCURRENCY_LIMIT:-3}

        echo "Starting Gunicorn with $GUNICORN_WORKERS workers, binding to $GUNICORN_HOST:$GUNICORN_PORT"

        # Start the Gunicorn server managing Uvicorn workers
        # `exec` is important as it replaces the shell process with the Gunicorn process,
        # allowing it to receive signals (like SIGTERM for shutdown) correctly.
        exec gunicorn \
            --workers "$GUNICORN_WORKERS" \
            --worker-class "uvicorn.workers.UvicornWorker" \
            --bind "$GUNICORN_HOST:$GUNICORN_PORT" \
            app:app
    else
        echo "Starting in Gradio mode..."
        exec python app.py
    fi    
fi