import os
from typing import List, Type

import boto3
import pandas as pd

from tools.config import (
    AWS_REGION,
    DOCUMENT_REDACTION_BUCKET,
    RUN_AWS_FUNCTIONS,
    SAVE_LOGS_TO_CSV,
)
from tools.secure_path_utils import secure_join

PandasDataFrame = Type[pd.DataFrame]


def get_assumed_role_info():
    sts_endpoint = "https://sts." + AWS_REGION + ".amazonaws.com"
    sts = boto3.client("sts", region_name=AWS_REGION, endpoint_url=sts_endpoint)
    response = sts.get_caller_identity()

    # Extract ARN of the assumed role
    assumed_role_arn = response["Arn"]

    # Extract the name of the assumed role from the ARN
    assumed_role_name = assumed_role_arn.split("/")[-1]

    return assumed_role_arn, assumed_role_name


if RUN_AWS_FUNCTIONS == "1":
    try:
        session = boto3.Session(region_name=AWS_REGION)

    except Exception as e:
        print("Could not start boto3 session:", e)

    try:
        assumed_role_arn, assumed_role_name = get_assumed_role_info()

        print("Successfully assumed ARN role")
        # print("Assumed Role ARN:", assumed_role_arn)
        # print("Assumed Role Name:", assumed_role_name)

    except Exception as e:
        print("Could not get assumed role from STS:", e)


# Download direct from S3 - requires login credentials
def download_file_from_s3(
    bucket_name: str,
    key: str,
    local_file_path_and_name: str,
    RUN_AWS_FUNCTIONS: str = RUN_AWS_FUNCTIONS,
):

    if RUN_AWS_FUNCTIONS == "1":

        try:
            # Ensure the local directory exists
            os.makedirs(os.path.dirname(local_file_path_and_name), exist_ok=True)

            s3 = boto3.client("s3", region_name=AWS_REGION)
            s3.download_file(bucket_name, key, local_file_path_and_name)
            print(
                f"File downloaded from s3://{bucket_name}/{key} to {local_file_path_and_name}"
            )
        except Exception as e:
            print("Could not download file:", key, "from s3 due to", e)


def download_folder_from_s3(
    bucket_name: str,
    s3_folder: str,
    local_folder: str,
    RUN_AWS_FUNCTIONS: str = RUN_AWS_FUNCTIONS,
):
    """
    Download all files from an S3 folder to a local folder.
    """
    if RUN_AWS_FUNCTIONS == "1":
        if bucket_name and s3_folder and local_folder:

            s3 = boto3.client("s3", region_name=AWS_REGION)

            # List objects in the specified S3 folder
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)

            # Download each object
            for obj in response.get("Contents", []):
                # Extract object key and construct local file path
                object_key = obj["Key"]
                local_file_path = secure_join(
                    local_folder, os.path.relpath(object_key, s3_folder)
                )

                # Create directories if necessary
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the object
                try:
                    s3.download_file(bucket_name, object_key, local_file_path)
                    print(
                        f"Downloaded 's3://{bucket_name}/{object_key}' to '{local_file_path}'"
                    )
                except Exception as e:
                    print(f"Error downloading 's3://{bucket_name}/{object_key}':", e)
        else:
            print(
                "One or more required variables are empty, could not download from S3"
            )


def download_files_from_s3(
    bucket_name: str,
    s3_folder: str,
    local_folder: str,
    filenames: List[str],
    RUN_AWS_FUNCTIONS: str = RUN_AWS_FUNCTIONS,
):
    """
    Download specific files from an S3 folder to a local folder.
    """

    if RUN_AWS_FUNCTIONS == "1":
        if bucket_name and s3_folder and local_folder and filenames:

            s3 = boto3.client("s3", region_name=AWS_REGION)

            print("Trying to download file: ", filenames)

            if filenames == "*":
                # List all objects in the S3 folder
                print("Trying to download all files in AWS folder: ", s3_folder)
                response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)

                print("Found files in AWS folder: ", response.get("Contents", []))

                filenames = [
                    obj["Key"].split("/")[-1] for obj in response.get("Contents", [])
                ]

                print("Found filenames in AWS folder: ", filenames)

            for filename in filenames:
                object_key = secure_join(s3_folder, filename)
                local_file_path = secure_join(local_folder, filename)

                # Create directories if necessary
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the object
                try:
                    s3.download_file(bucket_name, object_key, local_file_path)
                    print(
                        f"Downloaded 's3://{bucket_name}/{object_key}' to '{local_file_path}'"
                    )
                except Exception as e:
                    print(f"Error downloading 's3://{bucket_name}/{object_key}':", e)

        else:
            print(
                "One or more required variables are empty, could not download from S3"
            )


def upload_file_to_s3(
    local_file_paths: List[str],
    s3_key: str,
    s3_bucket: str = DOCUMENT_REDACTION_BUCKET,
    RUN_AWS_FUNCTIONS: str = RUN_AWS_FUNCTIONS,
):
    """
    Uploads a file from local machine to Amazon S3.

    Args:
    - local_file_path: Local file path(s) of the file(s) to upload.
    - s3_key: Key (path) to the file in the S3 bucket.
    - s3_bucket: Name of the S3 bucket.

    Returns:
    - Message as variable/printed to console
    """
    final_out_message = []
    final_out_message_str = ""

    if RUN_AWS_FUNCTIONS == "1":
        try:
            if s3_bucket and s3_key and local_file_paths:

                s3_client = boto3.client("s3", region_name=AWS_REGION)

                if isinstance(local_file_paths, str):
                    local_file_paths = [local_file_paths]

                for file in local_file_paths:
                    if s3_client:
                        # print(s3_client)
                        try:
                            # Get file name off file path
                            file_name = os.path.basename(file)

                            s3_key_full = s3_key + file_name
                            print("S3 key: ", s3_key_full)

                            s3_client.upload_file(file, s3_bucket, s3_key_full)
                            out_message = (
                                "File " + file_name + " uploaded successfully!"
                            )
                            print(out_message)

                        except Exception as e:
                            out_message = f"Error uploading file(s): {e}"
                            print(out_message)

                        final_out_message.append(out_message)
                        final_out_message_str = "\n".join(final_out_message)

                    else:
                        final_out_message_str = "Could not connect to AWS."
            else:
                final_out_message_str = (
                    "At least one essential variable is empty, could not upload to S3"
                )
        except Exception as e:
            final_out_message_str = "Could not upload files to S3 due to: " + str(e)
            print(final_out_message_str)
    else:
        final_out_message_str = "App config will not run AWS functions"

    return final_out_message_str


def upload_log_file_to_s3(
    local_file_paths: List[str],
    s3_key: str,
    s3_bucket: str = DOCUMENT_REDACTION_BUCKET,
    RUN_AWS_FUNCTIONS: str = RUN_AWS_FUNCTIONS,
    SAVE_LOGS_TO_CSV: str = SAVE_LOGS_TO_CSV,
):
    """
    Uploads a log file from local machine to Amazon S3.

    Args:
    - local_file_path: Local file path(s) of the file(s) to upload.
    - s3_key: Key (path) to the file in the S3 bucket.
    - s3_bucket: Name of the S3 bucket.

    Returns:
    - Message as variable/printed to console
    """
    final_out_message = []
    final_out_message_str = ""

    if RUN_AWS_FUNCTIONS == "1" and SAVE_LOGS_TO_CSV == "True":
        try:
            if s3_bucket and s3_key and local_file_paths:

                s3_client = boto3.client("s3", region_name=AWS_REGION)

                if isinstance(local_file_paths, str):
                    local_file_paths = [local_file_paths]

                for file in local_file_paths:
                    if s3_client:
                        # print(s3_client)
                        try:
                            # Get file name off file path
                            file_name = os.path.basename(file)

                            s3_key_full = s3_key + file_name
                            print("S3 key: ", s3_key_full)

                            s3_client.upload_file(file, s3_bucket, s3_key_full)
                            out_message = (
                                "File " + file_name + " uploaded successfully!"
                            )
                            print(out_message)

                        except Exception as e:
                            out_message = f"Error uploading file(s): {e}"
                            print(out_message)

                        final_out_message.append(out_message)
                        final_out_message_str = "\n".join(final_out_message)

                    else:
                        final_out_message_str = "Could not connect to AWS."
            else:
                final_out_message_str = (
                    "At least one essential variable is empty, could not upload to S3"
                )
        except Exception as e:
            final_out_message_str = "Could not upload files to S3 due to: " + str(e)
            print(final_out_message_str)
    else:
        final_out_message_str = "App config will not run AWS functions"
        print(final_out_message_str)

    return final_out_message_str
