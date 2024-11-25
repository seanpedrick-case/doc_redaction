#!/bin/sh

if [ "$APP_MODE" = "lambda" ]; then
    echo "Starting in Lambda mode..."
    exec python -m awslambdaric "$@"
else
    echo "Starting in Gradio mode..."
    exec python app.py "$@"
fi