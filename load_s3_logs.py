import argparse
import os
from datetime import datetime, timedelta
from io import StringIO

import boto3
import pandas as pd

from tools.config import (
    AWS_ACCESS_KEY,
    AWS_REGION,
    AWS_SECRET_KEY,
    DOCUMENT_REDACTION_BUCKET,
    OUTPUT_FOLDER,
)

# Combine together log files that can be then used for e.g. dashboarding and financial tracking.


def parse_args():
    """Parse command-line arguments; config values are used as defaults."""
    today = datetime.now()
    one_year_ago = (today - timedelta(days=365)).strftime("%Y%m%d")
    today_str = today.strftime("%Y%m%d")

    parser = argparse.ArgumentParser(
        description="Combine S3 usage log CSVs in a date range into a single CSV."
    )
    parser.add_argument(
        "--bucket",
        default=DOCUMENT_REDACTION_BUCKET,
        help=f"S3 bucket name (default from config: {DOCUMENT_REDACTION_BUCKET!r})",
    )
    parser.add_argument(
        "--region",
        default=AWS_REGION,
        help=f"AWS region (default from config: {AWS_REGION!r})",
    )
    parser.add_argument(
        "--prefix",
        default="usage/",
        help="S3 prefix / top-level folder where logs are stored (default: usage/)",
    )
    parser.add_argument(
        "--from-date",
        dest="earliest_date",
        default=one_year_ago,
        metavar="YYYYMMDD",
        help=f"Earliest date of logs to include (default: one year ago, {one_year_ago})",
    )
    parser.add_argument(
        "--to-date",
        dest="latest_date",
        default=today_str,
        metavar="YYYYMMDD",
        help=f"Latest date of logs to include (default: today, {today_str})",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Full output CSV path (overrides --output-folder and --output-filename if set)",
    )
    parser.add_argument(
        "--output-folder",
        default=OUTPUT_FOLDER,
        metavar="DIR",
        help=f"Output folder for the CSV (default from config: {OUTPUT_FOLDER!r})",
    )
    parser.add_argument(
        "--output-filename",
        default="consolidated_s3_logs.csv",
        metavar="NAME",
        help="Output CSV file name (default: consolidated_s3_logs.csv)",
    )
    parser.add_argument(
        "--s3-output-bucket",
        default=None,
        metavar="BUCKET",
        help="If set (with --s3-output-key), upload the output CSV to this S3 bucket",
    )
    parser.add_argument(
        "--s3-output-key",
        default=None,
        metavar="KEY",
        help="S3 object key (path) for the output CSV when using --s3-output-bucket",
    )
    return parser.parse_args()


# Function to list all files in a folder
def list_files_in_s3(s3_client, bucket, prefix):
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" in response:
        return [content["Key"] for content in response["Contents"]]
    return []


# Function to filter date range
def is_within_date_range(date_str, start_date, end_date):
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    return start_date <= date_obj <= end_date


def main():
    args = parse_args()
    bucket_name = args.bucket
    region = args.region
    prefix = args.prefix
    earliest_date = args.earliest_date
    latest_date = args.latest_date
    if args.output is not None:
        output_path = args.output
    else:
        output_path = os.path.join(
            args.output_folder.rstrip(r"\/"), args.output_filename
        )

    # S3 setup. Use provided keys if in config, otherwise assume AWS SSO/default credentials
    if AWS_ACCESS_KEY and AWS_SECRET_KEY and region:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=region,
        )
    else:
        s3 = boto3.client("s3", region_name=region if region else None)

    # Define the date range
    start_date = datetime.strptime(earliest_date, "%Y%m%d")
    end_date = datetime.strptime(latest_date, "%Y%m%d")

    # List all files under prefix
    all_files = list_files_in_s3(s3, bucket_name, prefix)

    # Filter based on date range (expects structure like prefix/YYYYMMDD/.../log.csv)
    log_files = []
    for file in all_files:
        parts = file.split("/")
        if len(parts) >= 3:
            date_str = parts[1]
            if (
                is_within_date_range(date_str, start_date, end_date)
                and parts[-1] == "log.csv"
            ):
                log_files.append(file)

    # Download, read and concatenate CSV files into a pandas DataFrame
    df_list = []
    for log_file in log_files:
        obj = s3.get_object(Bucket=bucket_name, Key=log_file)
        try:
            csv_content = obj["Body"].read().decode("utf-8")
        except Exception as e:
            print("Could not load in log file:", log_file, "due to:", e)
            csv_content = obj["Body"].read().decode("latin-1")

        try:
            df = pd.read_csv(StringIO(csv_content))
        except Exception as e:
            print("Could not load in log file:", log_file, "due to:", e)
            continue

        df_list.append(df)

    if df_list:
        concatenated_df = pd.concat(df_list, ignore_index=True)
        concatenated_df.to_csv(output_path, index=False)
        print(f"Consolidated CSV saved to {output_path}")

        if args.s3_output_bucket and args.s3_output_key:
            try:
                s3.upload_file(output_path, args.s3_output_bucket, args.s3_output_key)
                print(f"Uploaded to s3://{args.s3_output_bucket}/{args.s3_output_key}")
            except Exception as e:
                print(f"Failed to upload to S3: {e}")
        elif args.s3_output_bucket or args.s3_output_key:
            print(
                "Warning: both --s3-output-bucket and --s3-output-key are required for S3 upload; skipping."
            )
    else:
        print("No log files found in the given date range.")


if __name__ == "__main__":
    main()
