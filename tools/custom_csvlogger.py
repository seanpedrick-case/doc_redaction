from __future__ import annotations

import csv
import os
import time
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

# from multiprocessing import Lock
from threading import Lock
from typing import TYPE_CHECKING, Any

import boto3
import botocore
from gradio import utils
from gradio_client import utils as client_utils

from tools.config import AWS_ACCESS_KEY, AWS_REGION, AWS_SECRET_KEY, RUN_AWS_FUNCTIONS

if TYPE_CHECKING:
    from gradio.components import Component

from gradio.flagging import FlaggingCallback


class CSVLogger_custom(FlaggingCallback):
    """
    The default implementation of the FlaggingCallback abstract class in gradio>=5.0. Each flagged
    sample (both the input and output data) is logged to a CSV file with headers on the machine running
    the gradio app. Unlike ClassicCSVLogger, this implementation is concurrent-safe and it creates a new
    dataset file every time the headers of the CSV (derived from the labels of the components) change. It also
    only creates columns for "username" and "flag" if the flag_option and username are provided, respectively.

    Example:
        import gradio as gr
        def image_classifier(inp):
            return {'cat': 0.3, 'dog': 0.7}
        demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label",
                            flagging_callback=CSVLogger())
    Guides: using-flagging
    """

    def __init__(
        self,
        simplify_file_data: bool = True,
        verbose: bool = True,
        dataset_file_name: str | None = None,
    ):
        """
        Parameters:
            simplify_file_data: If True, the file data will be simplified before being written to the CSV file. If CSVLogger is being used to cache examples, this is set to False to preserve the original FileData class
            verbose: If True, prints messages to the console about the dataset file creation
            dataset_file_name: The name of the dataset file to be created (should end in ".csv"). If None, the dataset file will be named "dataset1.csv" or the next available number.
        """
        self.simplify_file_data = simplify_file_data
        self.verbose = verbose
        self.dataset_file_name = dataset_file_name
        self.lock = Lock()

    def setup(
        self,
        components: Sequence[Component],
        flagging_dir: str | Path,
    ):
        self.components = components
        self.flagging_dir = Path(flagging_dir)
        self.first_time = True

    def _create_dataset_file(
        self,
        additional_headers: list[str] | None = None,
        replacement_headers: list[str] | None = None,
    ):
        os.makedirs(self.flagging_dir, exist_ok=True)

        if replacement_headers:
            if additional_headers is None:
                additional_headers = list()

            if len(replacement_headers) != len(self.components):
                raise ValueError(
                    f"replacement_headers must have the same length as components "
                    f"({len(replacement_headers)} provided, {len(self.components)} expected)"
                )
            headers = replacement_headers + additional_headers + ["timestamp"]
        else:
            if additional_headers is None:
                additional_headers = []
            headers = (
                [
                    getattr(component, "label", None) or f"component {idx}"
                    for idx, component in enumerate(self.components)
                ]
                + additional_headers
                + ["timestamp"]
            )

        headers = utils.sanitize_list_for_csv(headers)
        dataset_files = list(Path(self.flagging_dir).glob("dataset*.csv"))

        if self.dataset_file_name:
            self.dataset_filepath = self.flagging_dir / self.dataset_file_name
        elif dataset_files:
            try:
                from tools.secure_regex_utils import (
                    safe_extract_latest_number_from_filename,
                )

                latest_file = max(
                    dataset_files,
                    key=lambda f: safe_extract_latest_number_from_filename(f.stem) or 0,
                )
                latest_num = (
                    safe_extract_latest_number_from_filename(latest_file.stem) or 0
                )

                with open(latest_file, newline="", encoding="utf-8") as csvfile:
                    reader = csv.reader(csvfile)
                    existing_headers = next(reader, None)

                if existing_headers != headers:
                    new_num = latest_num + 1
                    self.dataset_filepath = self.flagging_dir / f"dataset{new_num}.csv"
                else:
                    self.dataset_filepath = latest_file
            except Exception:
                self.dataset_filepath = self.flagging_dir / "dataset1.csv"
        else:
            self.dataset_filepath = self.flagging_dir / "dataset1.csv"

        if not Path(self.dataset_filepath).exists():
            with open(
                self.dataset_filepath, "w", newline="", encoding="utf-8"
            ) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(utils.sanitize_list_for_csv(headers))
            if self.verbose:
                print("Created dataset file at:", self.dataset_filepath)
        elif self.verbose:
            print("Using existing dataset file at:", self.dataset_filepath)

    def flag(
        self,
        flag_data: list[Any],
        flag_option: str | None = None,
        username: str | None = None,
        save_to_csv: bool = True,
        save_to_dynamodb: bool = False,
        dynamodb_table_name: str | None = None,
        dynamodb_headers: list[str] | None = None,  # New: specify headers for DynamoDB
        replacement_headers: list[str] | None = None,
    ) -> int:
        if self.first_time:
            additional_headers = list()
            if flag_option is not None:
                additional_headers.append("flag")
            if username is not None:
                additional_headers.append("username")
            additional_headers.append("id")
            self._create_dataset_file(
                additional_headers=additional_headers,
                replacement_headers=replacement_headers,
            )
            self.first_time = False

        csv_data = list()
        for idx, (component, sample) in enumerate(
            zip(self.components, flag_data, strict=False)
        ):
            save_dir = (
                self.flagging_dir
                / client_utils.strip_invalid_filename_characters(
                    getattr(component, "label", None) or f"component {idx}"
                )
            )
            if utils.is_prop_update(sample):
                csv_data.append(str(sample))
            else:
                data = (
                    component.flag(sample, flag_dir=save_dir)
                    if sample is not None
                    else ""
                )
                if self.simplify_file_data:
                    data = utils.simplify_file_data_in_str(data)
                csv_data.append(data)

        if flag_option is not None:
            csv_data.append(flag_option)
        if username is not None:
            csv_data.append(username)

        generated_id = str(uuid.uuid4())
        csv_data.append(generated_id)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]  # Correct format for Amazon Athena
        csv_data.append(timestamp)

        # Build the headers
        headers = [
            getattr(component, "label", None) or f"component {idx}"
            for idx, component in enumerate(self.components)
        ]
        if flag_option is not None:
            headers.append("flag")
        if username is not None:
            headers.append("username")
        headers.append("id")
        headers.append("timestamp")

        line_count = -1

        if save_to_csv:
            with self.lock:
                with open(
                    self.dataset_filepath, "a", newline="", encoding="utf-8"
                ) as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(utils.sanitize_list_for_csv(csv_data))
                with open(self.dataset_filepath, encoding="utf-8") as csvfile:
                    line_count = len(list(csv.reader(csvfile))) - 1

        if save_to_dynamodb is True:

            if RUN_AWS_FUNCTIONS:
                try:
                    print("Connecting to DynamoDB via existing SSO connection")
                    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)

                    dynamodb.meta.client.list_tables()

                except Exception as e:
                    print("No SSO credentials found:", e)
                    if AWS_ACCESS_KEY and AWS_SECRET_KEY:
                        print(
                            "Trying to get DynamoDB credentials from environment variables"
                        )
                        dynamodb = boto3.resource(
                            "dynamodb",
                            aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY,
                            region_name=AWS_REGION,
                        )
                    else:
                        raise Exception(
                            "AWS credentials for DynamoDB logging not found"
                        )
            else:
                raise Exception("AWS credentials for DynamoDB logging not found")

            if dynamodb_table_name is None:
                raise ValueError(
                    "You must provide a dynamodb_table_name if save_to_dynamodb is True"
                )

            if dynamodb_headers:
                dynamodb_headers = dynamodb_headers
            if not dynamodb_headers and replacement_headers:
                dynamodb_headers = replacement_headers
            elif headers:
                dynamodb_headers = headers
            elif not dynamodb_headers:
                raise ValueError(
                    "Headers not found. You must provide dynamodb_headers or replacement_headers to create a new table."
                )

            if flag_option is not None:
                if "flag" not in dynamodb_headers:
                    dynamodb_headers.append("flag")
                if username is not None:
                    if "username" not in dynamodb_headers:
                        dynamodb_headers.append("username")
                if "timestamp" not in dynamodb_headers:
                    dynamodb_headers.append("timestamp")
                if "id" not in dynamodb_headers:
                    dynamodb_headers.append("id")

            # Table doesn't exist — create it
            try:
                table = dynamodb.Table(dynamodb_table_name)
                table.load()
            except botocore.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":

                    attribute_definitions = [
                        {
                            "AttributeName": "id",
                            "AttributeType": "S",
                        }  # Only define key attributes here
                    ]

                    table = dynamodb.create_table(
                        TableName=dynamodb_table_name,
                        KeySchema=[
                            {"AttributeName": "id", "KeyType": "HASH"}  # Partition key
                        ],
                        AttributeDefinitions=attribute_definitions,
                        BillingMode="PAY_PER_REQUEST",
                    )
                    # Wait until the table exists
                    table.meta.client.get_waiter("table_exists").wait(
                        TableName=dynamodb_table_name
                    )
                    time.sleep(5)
                    print(f"Table '{dynamodb_table_name}' created successfully.")
                else:
                    raise

            # Prepare the DynamoDB item to upload

            try:
                item = {
                    "id": str(generated_id),  # UUID primary key
                    "timestamp": timestamp,
                }

                # Map the headers to values
                item.update(
                    {
                        header: str(value)
                        for header, value in zip(dynamodb_headers, csv_data)
                    }
                )

                table.put_item(Item=item)

                print("Successfully uploaded log to DynamoDB")
            except Exception as e:
                print("Could not upload log to DynamobDB due to", e)

        return line_count
