import argparse
import csv
import datetime
import os
from decimal import Decimal

import boto3

from tools.config import (
    AWS_ACCESS_KEY,
    AWS_REGION,
    AWS_SECRET_KEY,
    OUTPUT_FOLDER,
    USAGE_LOG_DYNAMODB_TABLE_NAME,
)


def parse_args():
    """Parse command-line arguments; config values are used as defaults."""
    parser = argparse.ArgumentParser(
        description="Export DynamoDB usage log table to CSV."
    )
    parser.add_argument(
        "--table",
        default=USAGE_LOG_DYNAMODB_TABLE_NAME,
        help=f"DynamoDB table name (default from config: {USAGE_LOG_DYNAMODB_TABLE_NAME!r})",
    )
    parser.add_argument(
        "--region",
        default=AWS_REGION,
        help=f"AWS region (default from config: {AWS_REGION!r})",
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
        default="dynamodb_logs_export.csv",
        metavar="NAME",
        help="Output CSV file name (default: dynamodb_logs_export.csv)",
    )
    parser.add_argument(
        "--from-date",
        dest="from_date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only include entries on or after this date (optional)",
    )
    parser.add_argument(
        "--to-date",
        dest="to_date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only include entries on or before this date (optional)",
    )
    parser.add_argument(
        "--date-attribute",
        default="timestamp",
        help="DynamoDB attribute name used for date filtering (default: timestamp)",
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


# Helper function to convert Decimal to float or int
def convert_types(item):
    new_item = {}
    for key, value in item.items():
        # Handle Decimals first
        if isinstance(value, Decimal):
            new_item[key] = int(value) if value % 1 == 0 else float(value)
        # Handle Strings that might be dates
        elif isinstance(value, str):
            try:
                # Attempt to parse a common ISO 8601 format.
                # The .replace() handles the 'Z' for Zulu/UTC time.
                dt_obj = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
                # Now that we have a datetime object, format it as desired
                new_item[key] = dt_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            except (ValueError, TypeError):
                # If it fails to parse, it's just a regular string
                new_item[key] = value
        # Handle all other types
        else:
            new_item[key] = value
    return new_item


def _parse_item_date(value):
    """Parse a DynamoDB attribute value to datetime for comparison. Returns None if unparseable."""
    if value is None:
        return None
    # Decimal (DynamoDB number type, e.g. Unix timestamp)
    if isinstance(value, Decimal):
        try:
            ts = float(value)
            return datetime.datetime.utcfromtimestamp(ts)
        except (ValueError, OSError):
            return None
    if isinstance(value, (int, float)):
        try:
            return datetime.datetime.utcfromtimestamp(float(value))
        except (ValueError, OSError):
            return None
    # String: try ISO and common formats
    if isinstance(value, str):
        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                return datetime.datetime.strptime(value, fmt)
            except (ValueError, TypeError):
                continue
        try:
            # Handles ISO with Z or +00:00
            return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass
    return None


def filter_items_by_date(items, from_date, to_date, date_attribute: str):
    """Return items whose date attribute falls within [from_date, to_date] (inclusive)."""
    if from_date is None and to_date is None:
        return items
    start = datetime.datetime.min
    end = datetime.datetime.max
    if from_date is not None:
        start = datetime.datetime.combine(from_date, datetime.time.min)
    if to_date is not None:
        end = datetime.datetime.combine(to_date, datetime.time.max)
    filtered = []
    for item in items:
        raw = item.get(date_attribute)
        dt = _parse_item_date(raw)
        if dt is None:
            continue
        # Normalize to naive for comparison if needed
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)
        if start <= dt <= end:
            filtered.append(item)
    return filtered


# Paginated scan
def scan_table(table):
    items = []
    response = table.scan()
    items.extend(response["Items"])

    while "LastEvaluatedKey" in response:
        response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
        items.extend(response["Items"])

    return items


# Export to CSV
def export_to_csv(items, output_path, fields_to_drop: list = None):
    if not items:
        print("No items found.")
        return

    # Use a set for efficient lookup
    drop_set = set(fields_to_drop or [])

    # Get a comprehensive list of all possible headers from all items
    all_keys = set()
    for item in items:
        all_keys.update(item.keys())

    # Determine the final fieldnames by subtracting the ones to drop
    fieldnames = sorted(list(all_keys - drop_set))

    print("Final CSV columns will be:", fieldnames)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        # The key fix is here: extrasaction='ignore'
        # restval='' is also good practice to handle rows that are missing a key
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames, extrasaction="ignore", restval=""
        )
        writer.writeheader()

        for item in items:
            # The convert_types function can now return the full dict,
            # and the writer will simply ignore the extra fields.
            writer.writerow(convert_types(item))

    print(f"Exported {len(items)} items to {output_path}")


def main():
    args = parse_args()
    table_name = args.table
    region = args.region
    if args.output is not None:
        csv_output = args.output
    else:
        csv_output = os.path.join(
            args.output_folder.rstrip(r"\/"), args.output_filename
        )

    today = datetime.datetime.now().date()
    one_year_ago = today - datetime.timedelta(days=365)

    from_date = None
    to_date = None
    if args.from_date:
        from_date = datetime.datetime.strptime(args.from_date, "%Y-%m-%d").date()
    if args.to_date:
        to_date = datetime.datetime.strptime(args.to_date, "%Y-%m-%d").date()
    # Default date range: one year ago to today
    if from_date is None and to_date is None:
        from_date = one_year_ago
        to_date = today
    elif from_date is None:
        from_date = one_year_ago
    elif to_date is None:
        to_date = today
    if from_date > to_date:
        raise ValueError("--from-date must be on or before --to-date")

    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(table_name)

    items = scan_table(table)
    items = filter_items_by_date(items, from_date, to_date, args.date_attribute)
    print(f"Filtered to {len(items)} items in date range {from_date} to {to_date}.")
    export_to_csv(items, csv_output, fields_to_drop=[])

    if args.s3_output_bucket and args.s3_output_key:
        if AWS_ACCESS_KEY and AWS_SECRET_KEY and region:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
                region_name=region,
            )
        else:
            s3_client = boto3.client("s3", region_name=region if region else None)
        try:
            s3_client.upload_file(csv_output, args.s3_output_bucket, args.s3_output_key)
            print(f"Uploaded to s3://{args.s3_output_bucket}/{args.s3_output_key}")
        except Exception as e:
            print(f"Failed to upload to S3: {e}")
    elif args.s3_output_bucket or args.s3_output_key:
        print(
            "Warning: both --s3-output-bucket and --s3-output-key are required for S3 upload; skipping."
        )


if __name__ == "__main__":
    main()
