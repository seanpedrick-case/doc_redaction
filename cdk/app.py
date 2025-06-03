# --- App Entry Point (typically in app.py) ---
# In your app.py, you would have something like this:
import json
import os
from aws_cdk import (
    App, # Use App directly from aws_cdk
    Stack, # Use Stack directly from aws_cdk
    Environment, # Use Environment directly from aws_cdk
)

from check_resources import check_and_set_context, CONTEXT_FILE
from cdk_config import AWS_ACCOUNT_ID, AWS_REGION
from cdk_stack import CdkStack

# --- Function to load context from file ---
def load_context_from_file(app: App, file_path: str):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            context_data = json.load(f)
            # Manually set context values on the App object
            for key, value in context_data.items():
                app.node.set_context(key, value)
            print(f"Loaded context from {file_path}")
    else:
        print(f"Context file not found: {file_path}")

# Optional: Clean up previous context file
#if os.path.exists(CONTEXT_FILE):
#     os.remove(CONTEXT_FILE)

# Initialize the CDK app
app = App()

# Run the pre-check script to generate the context file
if not os.path.exists(CONTEXT_FILE):
    print("Context file not found, loading in again")
    check_and_set_context()
else:
    print("Loading context from file")
    # --- Programmatically load context from the file FOR TESTING WITHIN PYTHON ONLY, OTHERWISE COMMENT OUT ---
    load_context_from_file(app, CONTEXT_FILE)

# Define the environment - Option 2: From environment variables (Recommended)
# This will use the AWS_ACCOUNT_ID and AWS_REGION environment variables
# or resolve from your AWS credentials.
aws_env = Environment(account=AWS_ACCOUNT_ID, region=AWS_REGION)

# Create the stack, which will automatically load context from cdk.context.json
CdkStack(app, "RedactionStack", env=aws_env)

# Synthesize the CloudFormation template (or deploy directly)
app.synth(validate_on_synthesis=True)

# Or deploy:
# subprocess.run(["cdk", "deploy", "--all", "--require-approval", "never"], check=True)