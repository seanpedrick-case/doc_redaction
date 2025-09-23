"""
CLI Usage Logger - A simplified version of the Gradio CSVLogger_custom for CLI usage logging.
This module provides functionality to log usage data from CLI operations to CSV files and optionally DynamoDB.
"""

import csv
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List

import boto3

from tools.aws_functions import upload_log_file_to_s3
from tools.config import (
    AWS_ACCESS_KEY,
    AWS_REGION,
    AWS_SECRET_KEY,
    CSV_USAGE_LOG_HEADERS,
    DISPLAY_FILE_NAMES_IN_LOGS,
    DOCUMENT_REDACTION_BUCKET,
    DYNAMODB_USAGE_LOG_HEADERS,
    HOST_NAME,
    RUN_AWS_FUNCTIONS,
    S3_USAGE_LOGS_FOLDER,
    SAVE_LOGS_TO_CSV,
    SAVE_LOGS_TO_DYNAMODB,
    USAGE_LOG_DYNAMODB_TABLE_NAME,
    USAGE_LOGS_FOLDER,
)


class CLIUsageLogger:
    """
    A simplified usage logger for CLI operations that mimics the functionality
    of the Gradio CSVLogger_custom class.
    """

    def __init__(self, dataset_file_name: str = "usage_log.csv"):
        """
        Initialize the CLI usage logger.

        Args:
            dataset_file_name: Name of the CSV file to store logs
        """
        self.dataset_file_name = dataset_file_name
        self.flagging_dir = Path(USAGE_LOGS_FOLDER)
        self.dataset_filepath = None
        self.headers = None

    def setup(self, headers: List[str]):
        """
        Setup the logger with the specified headers.

        Args:
            headers: List of column headers for the CSV file
        """
        self.headers = headers
        self._create_dataset_file()

    def _create_dataset_file(self):
        """Create the dataset CSV file with headers if it doesn't exist."""
        os.makedirs(self.flagging_dir, exist_ok=True)

        # Add ID and timestamp to headers (matching custom_csvlogger.py structure)
        full_headers = self.headers + ["id", "timestamp"]

        self.dataset_filepath = self.flagging_dir / self.dataset_file_name

        if not Path(self.dataset_filepath).exists():
            with open(
                self.dataset_filepath, "w", newline="", encoding="utf-8"
            ) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(full_headers)
            print(f"Created usage log file at: {self.dataset_filepath}")
        else:
            print(f"Using existing usage log file at: {self.dataset_filepath}")

    def log_usage(
        self,
        data: List[Any],
        save_to_csv: bool = None,
        save_to_dynamodb: bool = None,
        save_to_s3: bool = None,
        s3_bucket: str = None,
        s3_key_prefix: str = None,
        dynamodb_table_name: str = None,
        dynamodb_headers: List[str] = None,
        replacement_headers: List[str] = None,
    ) -> int:
        """
        Log usage data to CSV and optionally DynamoDB and S3.

        Args:
            data: List of data values to log
            save_to_csv: Whether to save to CSV (defaults to config setting)
            save_to_dynamodb: Whether to save to DynamoDB (defaults to config setting)
            save_to_s3: Whether to save to S3 (defaults to config setting)
            s3_bucket: S3 bucket name (defaults to config setting)
            s3_key_prefix: S3 key prefix (defaults to config setting)
            dynamodb_table_name: DynamoDB table name (defaults to config setting)
            dynamodb_headers: DynamoDB headers (defaults to config setting)
            replacement_headers: Replacement headers for CSV (defaults to config setting)

        Returns:
            Number of lines written
        """
        # Use config defaults if not specified
        if save_to_csv is None:
            save_to_csv = SAVE_LOGS_TO_CSV == "True"
        if save_to_dynamodb is None:
            save_to_dynamodb = SAVE_LOGS_TO_DYNAMODB == "True"
        if save_to_s3 is None:
            save_to_s3 = RUN_AWS_FUNCTIONS == "1" and SAVE_LOGS_TO_CSV == "True"
        if s3_bucket is None:
            s3_bucket = DOCUMENT_REDACTION_BUCKET
        if s3_key_prefix is None:
            s3_key_prefix = S3_USAGE_LOGS_FOLDER
        if dynamodb_table_name is None:
            dynamodb_table_name = USAGE_LOG_DYNAMODB_TABLE_NAME
        if dynamodb_headers is None:
            dynamodb_headers = DYNAMODB_USAGE_LOG_HEADERS
        if replacement_headers is None:
            replacement_headers = CSV_USAGE_LOG_HEADERS

        # Generate unique ID and add timestamp (matching custom_csvlogger.py structure)
        generated_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]  # Correct format for Amazon Athena
        csv_data = data + [generated_id, timestamp]

        line_count = 0

        # Save to CSV
        if save_to_csv and self.dataset_filepath:
            try:
                with open(
                    self.dataset_filepath, "a", newline="", encoding="utf-8-sig"
                ) as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_data)
                    line_count = 1
                print(f"Logged usage data to CSV: {self.dataset_filepath}")
            except Exception as e:
                print(f"Error writing to CSV: {e}")

        # Upload to S3 if enabled
        if save_to_s3 and self.dataset_filepath and s3_bucket and s3_key_prefix:
            try:
                # Upload the log file to S3
                upload_result = upload_log_file_to_s3(
                    local_file_paths=[str(self.dataset_filepath)],
                    s3_key=s3_key_prefix,
                    s3_bucket=s3_bucket,
                    RUN_AWS_FUNCTIONS=RUN_AWS_FUNCTIONS,
                    SAVE_LOGS_TO_CSV=SAVE_LOGS_TO_CSV,
                )
                print(f"S3 upload result: {upload_result}")
            except Exception as e:
                print(f"Error uploading log file to S3: {e}")

        # Save to DynamoDB
        if save_to_dynamodb and dynamodb_table_name and dynamodb_headers:
            try:
                # Initialize DynamoDB client
                if AWS_ACCESS_KEY and AWS_SECRET_KEY:
                    dynamodb = boto3.resource(
                        "dynamodb",
                        region_name=AWS_REGION,
                        aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY,
                    )
                else:
                    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

                table = dynamodb.Table(dynamodb_table_name)

                # Generate unique ID
                generated_id = str(uuid.uuid4())

                # Prepare the DynamoDB item
                item = {
                    "id": generated_id,
                    "timestamp": timestamp,
                }

                # Map the headers to values
                item.update(
                    {
                        header: str(value)
                        for header, value in zip(dynamodb_headers, data)
                    }
                )

                table.put_item(Item=item)
                print("Successfully uploaded usage log to DynamoDB")

            except Exception as e:
                print(f"Could not upload usage log to DynamoDB: {e}")

        return line_count


def create_cli_usage_logger() -> CLIUsageLogger:
    """
    Create and setup a CLI usage logger with the standard headers.

    Returns:
        Configured CLIUsageLogger instance
    """
    # Parse CSV headers from config
    import json

    try:
        headers = json.loads(CSV_USAGE_LOG_HEADERS)
    except Exception as e:
        print(f"Error parsing CSV usage log headers: {e}")
        # Fallback headers if parsing fails
        headers = [
            "session_hash_textbox",
            "doc_full_file_name_textbox",
            "data_full_file_name_textbox",
            "actual_time_taken_number",
            "total_page_count",
            "textract_query_number",
            "pii_detection_method",
            "comprehend_query_number",
            "cost_code",
            "textract_handwriting_signature",
            "host_name_textbox",
            "text_extraction_method",
            "is_this_a_textract_api_call",
            "task",
        ]

    logger = CLIUsageLogger()
    logger.setup(headers)
    return logger


def log_redaction_usage(
    logger: CLIUsageLogger,
    session_hash: str,
    doc_file_name: str,
    data_file_name: str,
    time_taken: float,
    total_pages: int,
    textract_queries: int,
    pii_method: str,
    comprehend_queries: int,
    cost_code: str,
    handwriting_signature: str,
    text_extraction_method: str,
    is_textract_call: bool,
    task: str,
    save_to_dynamodb: bool = None,
    save_to_s3: bool = None,
    s3_bucket: str = None,
    s3_key_prefix: str = None,
):
    """
    Log redaction usage data using the provided logger.

    Args:
        logger: CLIUsageLogger instance
        session_hash: Session identifier
        doc_file_name: Document file name (or placeholder if not displaying names)
        data_file_name: Data file name (or placeholder if not displaying names)
        time_taken: Time taken for processing in seconds
        total_pages: Total number of pages processed
        textract_queries: Number of Textract API calls made
        pii_method: PII detection method used
        comprehend_queries: Number of Comprehend API calls made
        cost_code: Cost code for the operation
        handwriting_signature: Handwriting/signature extraction options
        text_extraction_method: Text extraction method used
        is_textract_call: Whether this was a Textract API call
        task: The task performed (redact, deduplicate, textract)
        save_to_dynamodb: Whether to save to DynamoDB (overrides config default)
        save_to_s3: Whether to save to S3 (overrides config default)
        s3_bucket: S3 bucket name (overrides config default)
        s3_key_prefix: S3 key prefix (overrides config default)
    """
    # Use placeholder names if not displaying file names in logs
    if DISPLAY_FILE_NAMES_IN_LOGS != "True":
        if doc_file_name:
            doc_file_name = "document"
            data_file_name = ""
        if data_file_name:
            data_file_name = "data_file"
            doc_file_name = ""
    else:
        doc_file_name = doc_file_name
        data_file_name = data_file_name

    rounded_time_taken = round(time_taken, 2)

    data = [
        session_hash,
        doc_file_name,
        data_file_name,
        rounded_time_taken,
        total_pages,
        textract_queries,
        pii_method,
        comprehend_queries,
        cost_code,
        handwriting_signature,
        HOST_NAME,
        text_extraction_method,
        is_textract_call,
        task,
    ]

    logger.log_usage(
        data,
        save_to_dynamodb=save_to_dynamodb,
        save_to_s3=save_to_s3,
        s3_bucket=s3_bucket,
        s3_key_prefix=s3_key_prefix,
    )
