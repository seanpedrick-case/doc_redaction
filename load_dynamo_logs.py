import boto3
import csv
from decimal import Decimal
from boto3.dynamodb.conditions import Key

from tools.config import AWS_REGION, ACCESS_LOG_DYNAMODB_TABLE_NAME, FEEDBACK_LOG_DYNAMODB_TABLE_NAME, USAGE_LOG_DYNAMODB_TABLE_NAME, OUTPUT_FOLDER

# Replace with your actual table name and region
TABLE_NAME = USAGE_LOG_DYNAMODB_TABLE_NAME # Choose as appropriate
REGION = AWS_REGION
CSV_OUTPUT = OUTPUT_FOLDER + 'dynamodb_logs_export.csv'

# Create DynamoDB resource
dynamodb = boto3.resource('dynamodb', region_name=REGION)
table = dynamodb.Table(TABLE_NAME)

# Helper function to convert Decimal to float or int
def convert_types(item):
    for key, value in item.items():
        if isinstance(value, Decimal):
            # Convert to int if no decimal places, else float
            item[key] = int(value) if value % 1 == 0 else float(value)
    return item

# Paginated scan
def scan_table():
    items = []
    response = table.scan()
    items.extend(response['Items'])

    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response['Items'])

    return items

# Export to CSV
def export_to_csv(items, output_path):
    if not items:
        print("No items found.")
        return

    fieldnames = sorted(items[0].keys())

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in items:
            writer.writerow(convert_types(item))

    print(f"Exported {len(items)} items to {output_path}")

# Run export
items = scan_table()
export_to_csv(items, CSV_OUTPUT)