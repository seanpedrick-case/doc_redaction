import boto3
import pandas as pd
from io import StringIO
from datetime import datetime
from tools.config import DOCUMENT_REDACTION_BUCKET

# S3 setup
s3 = boto3.client('s3')
bucket_name = DOCUMENT_REDACTION_BUCKET
prefix = 'logs'# 'usage/' # 'feedback/'  # Change as needed - top-level folder where logs are stored
earliest_date = '20250401' # Earliest date of logs folder retrieved
latest_date = '20250412' # Latest date of logs folder retrieved

# Function to list all files in a folder
def list_files_in_s3(bucket, prefix):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' in response:
        return [content['Key'] for content in response['Contents']]
    return []

# Function to filter date range
def is_within_date_range(date_str, start_date, end_date):
    date_obj = datetime.strptime(date_str, '%Y%m%d')
    return start_date <= date_obj <= end_date

# Define the date range
start_date = datetime.strptime('20250401', '%Y%m%d')  # Replace with your start date
end_date = datetime.strptime('20250412', '%Y%m%d')    # Replace with your end date

# List all subfolders under 'usage/'
all_files = list_files_in_s3(bucket_name, prefix)

# Filter based on date range
log_files = []
for file in all_files:
    parts = file.split('/')
    if len(parts) >= 3:
        date_str = parts[1]
        if is_within_date_range(date_str, start_date, end_date) and parts[-1] == 'log.csv':
            log_files.append(file)

# Download, read and concatenate CSV files into a pandas DataFrame
df_list = []
for log_file in log_files:
    # Download the file
    obj = s3.get_object(Bucket=bucket_name, Key=log_file)
    csv_content = obj['Body'].read().decode('utf-8')

    # Read CSV content into pandas DataFrame
    try:
        df = pd.read_csv(StringIO(csv_content))
    except Exception as e:
        print("Could not load in log file:", log_file, "due to:", e)
        continue

    df_list.append(df)

# Concatenate all DataFrames
if df_list:
    concatenated_df = pd.concat(df_list, ignore_index=True)

    # Save the concatenated DataFrame to a CSV file
    concatenated_df.to_csv('consolidated_logs.csv', index=False)
    print("Consolidated CSV saved as 'consolidated_logs.csv'")
else:
    print("No log files found in the given date range.")
