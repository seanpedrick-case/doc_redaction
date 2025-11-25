import csv
import datetime
from decimal import Decimal

import boto3

from tools.config import (
    AWS_REGION,
    OUTPUT_FOLDER,
    USAGE_LOG_DYNAMODB_TABLE_NAME,
)

# Replace with your actual table name and region
TABLE_NAME = USAGE_LOG_DYNAMODB_TABLE_NAME  # Choose as appropriate
REGION = AWS_REGION
CSV_OUTPUT = OUTPUT_FOLDER + "dynamodb_logs_export.csv"

# Create DynamoDB resource
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)


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


# Paginated scan
def scan_table():
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


# Run export
items = scan_table()
export_to_csv(items, CSV_OUTPUT, fields_to_drop=[])
