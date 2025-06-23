#!/bin/sh

echo "Starting in APP_MODE: $APP_MODE"

if [ "$APP_MODE" = "lambda" ]; then
    echo "Starting in Lambda mode..."
    exec python -m awslambdaric "$@"
else
    echo "Starting in Gradio mode..."
    exec python app.py "$@"
fi