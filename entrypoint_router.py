import os
import subprocess

if __name__ == "__main__":
    run_direct_mode = os.getenv("RUN_DIRECT_MODE", "0")

    if run_direct_mode == "1":
        # Lambda execution or CLI invocation (Direct Mode)
        from lambda_entrypoint import lambda_handler

        # Simulate the Lambda event and context for local testing
        event = os.getenv("LAMBDA_TEST_EVENT", '{}')
        context = None  # Add mock context if needed
        response = lambda_handler(eval(event), context)
        print(response)
    else:
        # Gradio App execution
        from app import app, max_queue_size, max_file_size  # Replace with actual import if needed
        from tools.auth import authenticate_user

        if os.getenv("COGNITO_AUTH", "0") == "1":
            app.queue(max_size=max_queue_size).launch(show_error=True, auth=authenticate_user, max_file_size=max_file_size)
        else:
            app.queue(max_size=max_queue_size).launch(show_error=True, inbrowser=True, max_file_size=max_file_size)