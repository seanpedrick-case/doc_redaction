import os
import time

from cdk_config import (
    CLUSTER_NAME,
    CODEBUILD_PI_PROJECT_NAME,
    CODEBUILD_PROJECT_NAME,
    ECS_EXPRESS_SC_PORT_NAME,
    ECS_EXPRESS_SERVICE_NAME,
    ECS_PI_EXPRESS_SERVICE_NAME,
    ECS_PI_SERVICE_NAME,
    ECS_SERVICE_CONNECT_DISCOVERY_NAME,
    ECS_SERVICE_CONNECT_NAMESPACE,
    ECS_SERVICE_NAME,
    ENABLE_HEADLESS_DEPLOYMENT,
    ENABLE_PI_AGENT_ECS_SERVICE,
    ENABLE_PI_AGENT_EXPRESS_SERVICE,
    GRADIO_SERVER_PORT,
    PI_AGENT_ENV_S3_KEY,
    S3_LOG_CONFIG_BUCKET_NAME,
    USE_ECS_EXPRESS_MODE,
)
from cdk_functions import create_basic_config_env

# boto3-only module (does not import aws-cdk / Node.js)
from cdk_post_deploy import (
    configure_express_pi_service_connect,
    start_codebuild_build,
    start_ecs_task,
    start_express_gateway_service,
    upload_file_to_s3,
)
from tqdm import tqdm

# Create basic config.env file that user can use to run the app later. Input is the folder it is saved into.
create_basic_config_env("config", headless=ENABLE_HEADLESS_DEPLOYMENT == "True")

# Start CodeBuild for the main app image
print("Starting main app CodeBuild project.")
start_codebuild_build(project_name=CODEBUILD_PROJECT_NAME)

_enable_pi_image_build = (
    ENABLE_PI_AGENT_ECS_SERVICE == "True" or ENABLE_PI_AGENT_EXPRESS_SERVICE == "True"
)
if _enable_pi_image_build:
    print("Starting Pi agent CodeBuild project.")
    start_codebuild_build(project_name=CODEBUILD_PI_PROJECT_NAME)

# Upload config.env file to S3 bucket
upload_file_to_s3(
    local_file_paths="config/config.env", s3_key="", s3_bucket=S3_LOG_CONFIG_BUCKET_NAME
)

if _enable_pi_image_build:
    pi_env_local = os.path.join("config", "pi_agent.env")
    if os.path.isfile(pi_env_local):
        print(
            f"Uploading {pi_env_local} to s3://{S3_LOG_CONFIG_BUCKET_NAME}/{PI_AGENT_ENV_S3_KEY}"
        )
        upload_file_to_s3(
            local_file_paths=pi_env_local,
            s3_key="",
            s3_bucket=S3_LOG_CONFIG_BUCKET_NAME,
        )
    else:
        print(
            f"Skipping Pi env upload: {pi_env_local} not found. "
            f"Create it (from config/pi_agent.env.example) and upload to "
            f"s3://{S3_LOG_CONFIG_BUCKET_NAME}/{PI_AGENT_ENV_S3_KEY} before scaling the Pi service."
        )

total_seconds = 480  # 8 minutes
update_interval = 1  # Update every second

print("Waiting 8 minutes for CodeBuild container image(s) to build.")

# tqdm iterates over a range, and you perform a small sleep in each iteration
for i in tqdm(range(total_seconds), desc="Building container"):
    time.sleep(update_interval)

# Scale main ECS service to one task (skipped for headless batch-only deployments)
if ENABLE_HEADLESS_DEPLOYMENT != "True":
    if USE_ECS_EXPRESS_MODE == "True":
        print(f"Starting Express ECS service {ECS_EXPRESS_SERVICE_NAME}")
        start_express_gateway_service(
            cluster_name=CLUSTER_NAME, service_name=ECS_EXPRESS_SERVICE_NAME
        )
    else:
        print(f"Starting ECS service {ECS_SERVICE_NAME}")
        start_ecs_task(cluster_name=CLUSTER_NAME, service_name=ECS_SERVICE_NAME)
else:
    print(
        "Headless deployment: skipping always-on ECS service start "
        "(tasks are started by the S3 batch Lambda)."
    )

if ENABLE_PI_AGENT_ECS_SERVICE == "True":
    print(f"Starting Pi agent ECS service {ECS_PI_SERVICE_NAME}")
    start_ecs_task(cluster_name=CLUSTER_NAME, service_name=ECS_PI_SERVICE_NAME)

if ENABLE_PI_AGENT_EXPRESS_SERVICE == "True":
    print(
        "Configuring Service Connect for main Express (server) and Pi Express (client)."
    )
    configure_express_pi_service_connect(
        cluster_name=CLUSTER_NAME,
        main_service_name=ECS_EXPRESS_SERVICE_NAME,
        pi_service_name=ECS_PI_EXPRESS_SERVICE_NAME,
        namespace=ECS_SERVICE_CONNECT_NAMESPACE,
        main_port_name=ECS_EXPRESS_SC_PORT_NAME,
        discovery_name=ECS_SERVICE_CONNECT_DISCOVERY_NAME,
        main_port=int(GRADIO_SERVER_PORT),
    )
    print(f"Starting Pi Express ECS service {ECS_PI_EXPRESS_SERVICE_NAME}")
    start_express_gateway_service(
        cluster_name=CLUSTER_NAME, service_name=ECS_PI_EXPRESS_SERVICE_NAME
    )
