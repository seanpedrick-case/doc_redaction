import time

from cdk_config import (
    CLUSTER_NAME,
    CODEBUILD_PROJECT_NAME,
    ECS_SERVICE_NAME,
    S3_LOG_CONFIG_BUCKET_NAME,
)
from cdk_functions import (
    create_basic_config_env,
    start_codebuild_build,
    start_ecs_task,
    upload_file_to_s3,
)
from tqdm import tqdm

# Create basic config.env file that user can use to run the app later. Input is the folder it is saved into.
create_basic_config_env("config")

# Start codebuild build
print("Starting CodeBuild project.")
start_codebuild_build(PROJECT_NAME=CODEBUILD_PROJECT_NAME)

# Upload config.env file to S3 bucket
upload_file_to_s3(
    local_file_paths="config/config.env", s3_key="", s3_bucket=S3_LOG_CONFIG_BUCKET_NAME
)

total_seconds = 660  # 11 minutes
update_interval = 1  # Update every second

print("Waiting 11 minutes for the CodeBuild container to build.")

# tqdm iterates over a range, and you perform a small sleep in each iteration
for i in tqdm(range(total_seconds), desc="Building container"):
    time.sleep(update_interval)

# Start task on ECS
print("Starting ECS task")
start_ecs_task(cluster_name=CLUSTER_NAME, service_name=ECS_SERVICE_NAME)
