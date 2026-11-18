"""
CDK app entrypoint: **harness only** (for a locally running main app).

Synth / deploy without the full RedactionStack pre-checks::

    cd cdk
    cdk deploy DocRedactionHarnessLocal -a "python app_harness_local.py"

Or use ``deploy_harness_local.py`` from the repo root.
"""

from aws_cdk import App, Environment
from harness_local_config import AWS_ACCOUNT_ID, AWS_REGION, STACK_NAME
from harness_local_stack import HarnessLocalStack

app = App()

if not AWS_ACCOUNT_ID:
    raise RuntimeError(
        "Set AWS_ACCOUNT_ID or CDK_DEFAULT_ACCOUNT (deploy_harness_local.py does this via STS)."
    )

env = Environment(account=AWS_ACCOUNT_ID, region=AWS_REGION)

HarnessLocalStack(app, STACK_NAME, env=env)

app.synth()
