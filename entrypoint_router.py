import os
import subprocess

if __name__ == "__main__":
    run_direct_mode = os.getenv("RUN_DIRECT_MODE", "0")

    if run_direct_mode == "1":

        print("Attempting to import lambda_handler from lambda_entrypoint")
        # Invoke the lambda handler
        from lambda_entrypoint import lambda_handler

        print("Imported lambda_handler from lambda_entrypoint")

    else:
        # Gradio App execution
        from app import app, max_queue_size, max_file_size  # Replace with actual import if needed
        from tools.auth import authenticate_user

        if os.getenv("COGNITO_AUTH", "0") == "1":
            app.queue(max_size=max_queue_size).launch(show_error=True, auth=authenticate_user, max_file_size=max_file_size)
        else:
            app.queue(max_size=max_queue_size).launch(show_error=True, inbrowser=True, max_file_size=max_file_size)