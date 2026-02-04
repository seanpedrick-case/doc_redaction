"""
Lambda handler to export DynamoDB usage log table to CSV and upload to S3.

All inputs are read from environment variables (no argparse).
Intended to run as an AWS Lambda function; can also be invoked locally
by setting env vars and calling lambda_handler({}, None).

Environment variables (same semantics as load_dynamo_logs.py CLI):
  DYNAMODB_TABLE_NAME       - DynamoDB table name (default: redaction_usage)
  AWS_REGION                - AWS region (Lambda often sets this)
  OUTPUT_FOLDER             - Local output directory, e.g. /tmp (optional)
  OUTPUT_FILENAME           - Local output file name (default: dynamodb_logs_export.csv)
  OUTPUT                    - Full local output path (overrides folder + filename if set)
  FROM_DATE                 - Only include entries on/after this date YYYY-MM-DD (optional)
  TO_DATE                   - Only include entries on/before this date YYYY-MM-DD (optional)
  DATE_ATTRIBUTE            - Attribute name for date filtering (default: timestamp)
  S3_OUTPUT_BUCKET          - S3 bucket for the output CSV (required for upload)
  S3_OUTPUT_KEY             - S3 object key/path for the output CSV (required for upload)
"""

import csv
import datetime
import os
from decimal import Decimal
from io import StringIO

import boto3


def get_config_from_env():
    """Read all settings from environment variables (same inputs as load_dynamo_logs.py)."""
    today = datetime.datetime.now().date()
    one_year_ago = today - datetime.timedelta(days=365)

    table_name = os.environ.get("DYNAMODB_TABLE_NAME") or os.environ.get(
        "USAGE_LOG_DYNAMODB_TABLE_NAME", "redaction_usage"
    )
    region = os.environ.get("AWS_REGION", "")
    output = os.environ.get("OUTPUT")
    output_folder = os.environ.get("OUTPUT_FOLDER", "output/")
    output_filename = os.environ.get("OUTPUT_FILENAME", "dynamodb_logs_export.csv")
    from_date_str = os.environ.get("FROM_DATE")
    to_date_str = os.environ.get("TO_DATE")
    date_attribute = os.environ.get("DATE_ATTRIBUTE", "timestamp")
    s3_output_bucket = os.environ.get("S3_OUTPUT_BUCKET")
    s3_output_key = os.environ.get("S3_OUTPUT_KEY")

    if output:
        local_output_path = output
    else:
        folder = output_folder.rstrip("/").rstrip("\\")
        local_output_path = os.path.join(folder, output_filename)

    from_date = None
    to_date = None
    if from_date_str:
        from_date = datetime.datetime.strptime(from_date_str, "%Y-%m-%d").date()
    if to_date_str:
        to_date = datetime.datetime.strptime(to_date_str, "%Y-%m-%d").date()
    if from_date is None and to_date is None:
        from_date = one_year_ago
        to_date = today
    elif from_date is None:
        from_date = one_year_ago
    elif to_date is None:
        to_date = today

    return {
        "table_name": table_name,
        "region": region,
        "local_output_path": local_output_path,
        "from_date": from_date,
        "to_date": to_date,
        "date_attribute": date_attribute,
        "s3_output_bucket": s3_output_bucket,
        "s3_output_key": s3_output_key,
    }


# Helper function to convert Decimal to float or int
def convert_types(item):
    new_item = {}
    for key, value in item.items():
        if isinstance(value, Decimal):
            new_item[key] = int(value) if value % 1 == 0 else float(value)
        elif isinstance(value, str):
            try:
                dt_obj = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
                new_item[key] = dt_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            except (ValueError, TypeError):
                new_item[key] = value
        else:
            new_item[key] = value
    return new_item


def _parse_item_date(value):
    """Parse a DynamoDB attribute value to datetime for comparison. Returns None if unparseable."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        try:
            return datetime.datetime.utcfromtimestamp(float(value))
        except (ValueError, OSError):
            return None
    if isinstance(value, (int, float)):
        try:
            return datetime.datetime.utcfromtimestamp(float(value))
        except (ValueError, OSError):
            return None
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
            return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass
    return None


def filter_items_by_date(items, from_date, to_date, date_attribute: str):
    """Return items whose date attribute falls within [from_date, to_date] (inclusive)."""
    if from_date is None and to_date is None:
        return items
    start = datetime.datetime.combine(from_date, datetime.time.min)
    end = datetime.datetime.combine(to_date, datetime.time.max)
    filtered = []
    for item in items:
        raw = item.get(date_attribute)
        dt = _parse_item_date(raw)
        if dt is None:
            continue
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)
        if start <= dt <= end:
            filtered.append(item)
    return filtered


def scan_table(table):
    """Paginated scan of DynamoDB table."""
    items = []
    response = table.scan()
    items.extend(response["Items"])
    while "LastEvaluatedKey" in response:
        response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
        items.extend(response["Items"])
    return items


def export_to_csv_buffer(items, fields_to_drop=None):
    """
    Write items to a CSV in memory; return (csv_string, fieldnames).
    Use for uploading to S3 without writing to disk.
    """
    if not items:
        return "", []

    drop_set = set(fields_to_drop or [])
    all_keys = set()
    for item in items:
        all_keys.update(item.keys())
    fieldnames = sorted(list(all_keys - drop_set))

    buf = StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=fieldnames, extrasaction="ignore", restval=""
    )
    writer.writeheader()
    for item in items:
        writer.writerow(convert_types(item))
    return buf.getvalue(), fieldnames


def export_to_csv_file(items, output_path, fields_to_drop=None):
    """Write items to a CSV file (for optional /tmp or local path)."""
    csv_string, _ = export_to_csv_buffer(items, fields_to_drop)
    if not csv_string:
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        f.write(csv_string)


def run_export(config):
    """
    Run the full export: scan DynamoDB, filter by date, write CSV (buffer and/or file), upload to S3.
    """
    table_name = config["table_name"]
    region = config["region"]
    local_output_path = config["local_output_path"]
    from_date = config["from_date"]
    to_date = config["to_date"]
    date_attribute = config["date_attribute"]
    s3_output_bucket = config["s3_output_bucket"]
    s3_output_key = config["s3_output_key"]

    if from_date > to_date:
        raise ValueError("FROM_DATE must be on or before TO_DATE")

    dynamodb = boto3.resource("dynamodb", region_name=region or None)
    table = dynamodb.Table(table_name)

    items = scan_table(table)
    items = filter_items_by_date(items, from_date, to_date, date_attribute)

    csv_string, fieldnames = export_to_csv_buffer(items, fields_to_drop=[])
    result = {
        "item_count": len(items),
        "from_date": str(from_date),
        "to_date": str(to_date),
        "columns": fieldnames,
    }

    if csv_string:
        # Optional: write to local path (e.g. /tmp in Lambda)
        try:
            export_to_csv_file(items, local_output_path, fields_to_drop=[])
            result["local_path"] = local_output_path
        except Exception as e:
            result["local_write_error"] = str(e)

        # Upload to S3 if bucket and key are set
        if s3_output_bucket and s3_output_key:
            s3 = boto3.client("s3", region_name=region or None)
            s3.put_object(
                Bucket=s3_output_bucket,
                Key=s3_output_key,
                Body=csv_string.encode("utf-8-sig"),
                ContentType="text/csv; charset=utf-8",
            )
            result["s3_uri"] = f"s3://{s3_output_bucket}/{s3_output_key}"
        elif s3_output_bucket or s3_output_key:
            result["s3_skip_reason"] = (
                "Both S3_OUTPUT_BUCKET and S3_OUTPUT_KEY must be set"
            )

    return result


def lambda_handler(event, context):
    """
    AWS Lambda entrypoint. Config is read from environment variables.

    Event is not required for config; it can be used to override env vars
    (e.g. pass table_name, from_date, to_date, s3_output_bucket, s3_output_key).
    """
    config = get_config_from_env()

    # Optional: allow event to override env-based config
    if isinstance(event, dict):
        if event.get("table_name"):
            config["table_name"] = event["table_name"]
        if event.get("region"):
            config["region"] = event["region"]
        if event.get("from_date"):
            config["from_date"] = datetime.datetime.strptime(
                event["from_date"], "%Y-%m-%d"
            ).date()
        if event.get("to_date"):
            config["to_date"] = datetime.datetime.strptime(
                event["to_date"], "%Y-%m-%d"
            ).date()
        if event.get("date_attribute"):
            config["date_attribute"] = event["date_attribute"]
        if event.get("s3_output_bucket"):
            config["s3_output_bucket"] = event["s3_output_bucket"]
        if event.get("s3_output_key"):
            config["s3_output_key"] = event["s3_output_key"]

    result = run_export(config)
    return {"statusCode": 200, "body": result}


if __name__ == "__main__":
    # Allow running locally with env vars set
    import json

    result = lambda_handler({}, None)
    print(json.dumps(result, indent=2))
