import boto3
import os
import subprocess
from urllib.parse import unquote_plus

s3_client = boto3.client("s3")

def download_file_from_s3(bucket_name, key, download_path):
    """Download a file from S3 to the local filesystem."""
    s3_client.download_file(bucket_name, key, download_path)
    print(f"Downloaded {key} to {download_path}")

def upload_file_to_s3(file_path, bucket_name, key):
    """Upload a file to S3."""
    s3_client.upload_file(file_path, bucket_name, key)
    print(f"Uploaded {file_path} to {key}")

def lambda_handler(event, context):
    """Main Lambda function handler."""
    # Parse the S3 event
    for record in event["Records"]:
        bucket_name = record["s3"]["bucket"]["name"]
        input_key = unquote_plus(record["s3"]["object"]["key"])
        print(f"Processing file {input_key} from bucket {bucket_name}")

        # Prepare paths
        input_file_path = f"/tmp/{os.path.basename(input_key)}"
        allow_list_path = f"/tmp/allow_list.csv"  # Adjust this as needed
        output_dir = "/tmp/output"
        os.makedirs(output_dir, exist_ok=True)

        # Download input file
        download_file_from_s3(bucket_name, input_key, input_file_path)

        # (Optional) Download allow_list if needed
        allow_list_key = "path/to/allow_list.csv"  # Adjust path as required
        download_file_from_s3(bucket_name, allow_list_key, allow_list_path)

        # Construct and run the command
        command = [
            "python",
            "app.py",
            "--input_file", input_file_path,
            "--ocr_method", "Complex image analysis - docs with handwriting/signatures (AWS Textract)",
            "--pii_detector", "AWS Comprehend",
            "--page_min", "0",
            "--page_max", "0",
            "--allow_list", allow_list_path,
            "--output_dir", output_dir,
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("Processing succeeded:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error during processing:", e.stderr)
            raise e

        # Upload output files back to S3
        for root, _, files in os.walk(output_dir):
            for file_name in files:
                local_file_path = os.path.join(root, file_name)
                output_key = f"{os.path.dirname(input_key)}/output/{file_name}"
                upload_file_to_s3(local_file_path, bucket_name, output_key)

    return {"statusCode": 200, "body": "Processing complete."}