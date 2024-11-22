import boto3
import os
import subprocess

print("In lambda_entrypoint function")

try:
    s3_client = boto3.client("s3", region_name="eu-west-2")
    print("s3_client is initialized:", s3_client)
except Exception as e:
    print(f"Error initializing s3_client: {e}")
    raise e

TMP_DIR = "/tmp/"

run_direct_mode = os.getenv("RUN_DIRECT_MODE", "0")

def download_file_from_s3(bucket_name, key, download_path):
    """Download a file from S3 to the local filesystem."""
    s3_client.download_file(bucket_name, key, download_path)
    print(f"Downloaded {key} to {download_path}")

def upload_file_to_s3(file_path, bucket_name, key):
    """Upload a file to S3."""
    s3_client.upload_file(file_path, bucket_name, key)
    print(f"Uploaded {file_path} to {key}")

def lambda_handler(event, context):

    print("In lambda_handler function")

    if run_direct_mode == "0":
        # Gradio App execution
        from app import app, max_queue_size, max_file_size  # Replace with actual import if needed
        from tools.auth import authenticate_user

        if os.getenv("COGNITO_AUTH", "0") == "1":
            app.queue(max_size=max_queue_size).launch(show_error=True, auth=authenticate_user, max_file_size=max_file_size)
        else:
            app.queue(max_size=max_queue_size).launch(show_error=True, inbrowser=True, max_file_size=max_file_size)

    else:
   
        # Create necessary directories
        os.makedirs(os.path.join(TMP_DIR, "input"), exist_ok=True)
        os.makedirs(os.path.join(TMP_DIR, "output"), exist_ok=True)

        print("Got to record loop")
        print("Event records is:", event["Records"])

        # Extract S3 bucket and object key from the Records
        for record in event.get("Records", [{}]):
            bucket_name = record.get("s3", {}).get("bucket", {}).get("name")
            input_key = record.get("s3", {}).get("object", {}).get("key")
            print(f"Processing file {input_key} from bucket {bucket_name}")

            # Extract additional arguments
            arguments = event.get("arguments", {})

            if not input_key:
                input_key = arguments.get("input_file", "")

            ocr_method = arguments.get("ocr_method", "Complex image analysis - docs with handwriting/signatures (AWS Textract)")
            pii_detector = arguments.get("pii_detector", "AWS Comprehend")
            page_min = str(arguments.get("page_min", 0))
            page_max = str(arguments.get("page_max", 0))
            allow_list = arguments.get("allow_list", None)
            output_dir = arguments.get("output_dir", os.path.join(TMP_DIR, "output"))
            
            print(f"OCR Method: {ocr_method}")
            print(f"PII Detector: {pii_detector}")
            print(f"Page Range: {page_min} - {page_max}")
            print(f"Allow List: {allow_list}")
            print(f"Output Directory: {output_dir}")

            # Download input file
            input_file_path = os.path.join(TMP_DIR, "input", os.path.basename(input_key))
            download_file_from_s3(bucket_name, input_key, input_file_path)

            # Construct command
            command = [
                "python",
                "app.py",
                "--input_file", input_file_path,
                "--ocr_method", ocr_method,
                "--pii_detector", pii_detector,
                "--page_min", page_min,
                "--page_max", page_max,
                "--output_dir", output_dir,
            ]

            # Add allow_list only if provided
            if allow_list:
                allow_list_path = os.path.join(TMP_DIR, "allow_list.csv")
                download_file_from_s3(bucket_name, allow_list, allow_list_path)
                command.extend(["--allow_list", allow_list_path])

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