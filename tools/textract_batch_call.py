import boto3
import time
import os
import json
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_pdf_with_textract(
    local_pdf_path: str,
    s3_bucket_name: str,
    s3_input_prefix: str,
    s3_output_prefix: str,
    local_output_dir: str,
    aws_region: str = None, # Optional: specify region if not default
    poll_interval_seconds: int = 5,
    max_polling_attempts: int = 120 # ~10 minutes total wait time
    ):
    """
    Uploads a local PDF to S3, starts a Textract analysis job (detecting text & signatures),
    waits for completion, and downloads the output JSON from S3 to a local directory.

    Args:
        local_pdf_path (str): Path to the local PDF file.
        s3_bucket_name (str): Name of the S3 bucket to use.
        s3_input_prefix (str): S3 prefix (folder) to upload the input PDF.
        s3_output_prefix (str): S3 prefix (folder) where Textract should write output.
        local_output_dir (str): Local directory to save the downloaded JSON results.
        aws_region (str, optional): AWS region name. Defaults to boto3 default region.
        poll_interval_seconds (int): Seconds to wait between polling Textract status.
        max_polling_attempts (int): Maximum number of times to poll Textract status.

    Returns:
        str: Path to the downloaded local JSON output file, or None if failed.

    Raises:
        FileNotFoundError: If the local_pdf_path does not exist.
        boto3.exceptions.NoCredentialsError: If AWS credentials are not found.
        Exception: For other AWS errors or job failures.
    """

    if not os.path.exists(local_pdf_path):
        raise FileNotFoundError(f"Input PDF not found: {local_pdf_path}")

    if not os.path.exists(local_output_dir):
        os.makedirs(local_output_dir)
        logging.info(f"Created local output directory: {local_output_dir}")

    # Initialize boto3 clients
    session = boto3.Session(region_name=aws_region)
    s3_client = session.client('s3')
    textract_client = session.client('textract')

    # --- 1. Upload PDF to S3 ---
    pdf_filename = os.path.basename(local_pdf_path)
    s3_input_key = os.path.join(s3_input_prefix, pdf_filename).replace("\\", "/") # Ensure forward slashes for S3

    logging.info(f"Uploading '{local_pdf_path}' to 's3://{s3_bucket_name}/{s3_input_key}'...")
    try:
        s3_client.upload_file(local_pdf_path, s3_bucket_name, s3_input_key)
        logging.info("Upload successful.")
    except Exception as e:
        logging.error(f"Failed to upload PDF to S3: {e}")
        raise

    # --- 2. Start Textract Document Analysis ---
    logging.info("Starting Textract document analysis job...")
    try:
        response = textract_client.start_document_analysis(
            DocumentLocation={
                'S3Object': {
                    'Bucket': s3_bucket_name,
                    'Name': s3_input_key
                }
            },
            FeatureTypes=['SIGNATURES', 'FORMS', 'TABLES'], # Analyze for signatures, forms, and tables
            OutputConfig={
                'S3Bucket': s3_bucket_name,
                'S3Prefix': s3_output_prefix
            }
            # Optional: Add NotificationChannel for SNS topic notifications
            # NotificationChannel={
            #     'SNSTopicArn': 'YOUR_SNS_TOPIC_ARN',
            #     'RoleArn': 'YOUR_IAM_ROLE_ARN_FOR_TEXTRACT_TO_ACCESS_SNS'
            # }
        )
        job_id = response['JobId']
        logging.info(f"Textract job started with JobId: {job_id}")

    except Exception as e:
        logging.error(f"Failed to start Textract job: {e}")
        raise

    # --- 3. Poll for Job Completion ---
    job_status = 'IN_PROGRESS'
    attempts = 0
    logging.info("Polling Textract for job completion status...")

    while job_status == 'IN_PROGRESS' and attempts < max_polling_attempts:
        attempts += 1
        try:
            response = textract_client.get_document_analysis(JobId=job_id)
            job_status = response['JobStatus']
            logging.info(f"Polling attempt {attempts}/{max_polling_attempts}. Job status: {job_status}")

            if job_status == 'IN_PROGRESS':
                time.sleep(poll_interval_seconds)
            elif job_status == 'SUCCEEDED':
                logging.info("Textract job succeeded.")
                break
            elif job_status in ['FAILED', 'PARTIAL_SUCCESS']:
                 status_message = response.get('StatusMessage', 'No status message provided.')
                 warnings = response.get('Warnings', [])
                 logging.error(f"Textract job ended with status: {job_status}. Message: {status_message}")
                 if warnings:
                     logging.warning(f"Warnings: {warnings}")
                 # Decide if PARTIAL_SUCCESS should proceed or raise error
                 # For simplicity here, we raise for both FAILED and PARTIAL_SUCCESS
                 raise Exception(f"Textract job {job_id} failed or partially failed. Status: {job_status}. Message: {status_message}")
            else:
                # Should not happen based on documentation, but handle defensively
                raise Exception(f"Unexpected Textract job status: {job_status}")

        except textract_client.exceptions.InvalidJobIdException:
             logging.error(f"Invalid JobId: {job_id}. This might happen if the job expired (older than 7 days) or never existed.")
             raise
        except Exception as e:
             logging.error(f"Error while polling Textract status for job {job_id}: {e}")
             raise

    if job_status != 'SUCCEEDED':
        raise TimeoutError(f"Textract job {job_id} did not complete successfully within the polling limit.")

    # --- 4. Download Output JSON from S3 ---
    # Textract typically creates output under s3_output_prefix/job_id/
    # There might be multiple JSON files if pagination occurred during writing.
    # Usually, for smaller docs, there's one file, often named '1'.
    # For robust handling, list objects and find the JSON(s).

    s3_output_key_prefix = os.path.join(s3_output_prefix, job_id).replace("\\", "/") + "/"
    logging.info(f"Searching for output files in s3://{s3_bucket_name}/{s3_output_key_prefix}")

    downloaded_file_path = None
    try:
        list_response = s3_client.list_objects_v2(
            Bucket=s3_bucket_name,
            Prefix=s3_output_key_prefix
        )

        output_files = list_response.get('Contents', [])
        if not output_files:
            # Sometimes Textract might take a moment longer to write the output after SUCCEEDED status
            logging.warning("No output files found immediately after job success. Waiting briefly and retrying list...")
            time.sleep(5)
            list_response = s3_client.list_objects_v2(
                Bucket=s3_bucket_name,
                Prefix=s3_output_key_prefix
            )
            output_files = list_response.get('Contents', [])

        if not output_files:
             logging.error(f"No output files found in s3://{s3_bucket_name}/{s3_output_key_prefix}")
             # You could alternatively try getting results via get_document_analysis pagination here
             # but sticking to the request to download from S3 output path.
             raise FileNotFoundError(f"Textract output files not found in S3 path: s3://{s3_bucket_name}/{s3_output_key_prefix}")

        # Usually, we only need the first/main JSON output file(s)
        # For simplicity, download the first one found. A more complex scenario might merge multiple files.
        # Filter out potential directory markers if any key ends with '/'
        json_files_to_download = [f for f in output_files if f['Key'] != s3_output_key_prefix and not f['Key'].endswith('/')]

        if not json_files_to_download:
            logging.error(f"No JSON files found (only prefix marker?) in s3://{s3_bucket_name}/{s3_output_key_prefix}")
            raise FileNotFoundError(f"Textract output JSON files not found in S3 path: s3://{s3_bucket_name}/{s3_output_key_prefix}")

        # Let's download the first JSON found. Often it's the only one or the main one.
        s3_output_key = json_files_to_download[0]['Key']
        output_filename_base = os.path.basename(pdf_filename).replace('.pdf', '')
        local_output_filename = f"{output_filename_base}_textract_output_{job_id}.json"
        local_output_path = os.path.join(local_output_dir, local_output_filename)

        logging.info(f"Downloading Textract output from 's3://{s3_bucket_name}/{s3_output_key}' to '{local_output_path}'...")
        s3_client.download_file(s3_bucket_name, s3_output_key, local_output_path)
        logging.info("Download successful.")
        downloaded_file_path = local_output_path

        # Log if multiple files were found, as user might need to handle them
        if len(json_files_to_download) > 1:
            logging.warning(f"Multiple output files found in S3 output location. Downloaded the first: '{s3_output_key}'. Other files exist.")

    except Exception as e:
        logging.error(f"Failed to download or process Textract output from S3: {e}")
        raise

    return downloaded_file_path

# --- Example Usage ---
if __name__ == '__main__':
    # --- Configuration --- (Replace with your actual values)
    MY_LOCAL_PDF = r"C:\path\to\your\document.pdf" # Use raw string for Windows paths
    MY_S3_BUCKET = "your-textract-demo-bucket-name" # MUST BE UNIQUE GLOBALLY
    MY_S3_INPUT_PREFIX = "textract-inputs"          # Folder in the bucket for uploads
    MY_S3_OUTPUT_PREFIX = "textract-outputs"        # Folder in the bucket for results
    MY_LOCAL_OUTPUT_DIR = "./textract_results"      # Local folder to save JSON
    MY_AWS_REGION = "us-east-1"                     # e.g., 'us-east-1', 'eu-west-1'

    # --- Create a dummy PDF for testing if you don't have one ---
    # Requires 'reportlab' library: pip install reportlab
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        if not os.path.exists(MY_LOCAL_PDF):
             print(f"Creating dummy PDF: {MY_LOCAL_PDF}")
             c = canvas.Canvas(MY_LOCAL_PDF, pagesize=letter)
             c.drawString(100, 750, "This is a test document for AWS Textract.")
             c.drawString(100, 700, "It includes some text and a placeholder for a signature.")
             c.drawString(100, 650, "Signed:")
             # Draw a simple line/scribble for signature placeholder
             c.line(150, 630, 250, 645)
             c.line(250, 645, 300, 620)
             c.save()
             print("Dummy PDF created.")
    except ImportError:
        if not os.path.exists(MY_LOCAL_PDF):
            print(f"Warning: reportlab not installed and '{MY_LOCAL_PDF}' not found. Cannot run example without an input PDF.")
            exit() # Exit if no PDF available for the example
    except Exception as e:
         print(f"Error creating dummy PDF: {e}")
         exit()


    # --- Run the analysis ---
    try:
        output_json_path = analyze_pdf_with_textract(
            local_pdf_path=MY_LOCAL_PDF,
            s3_bucket_name=MY_S3_BUCKET,
            s3_input_prefix=MY_S3_INPUT_PREFIX,
            s3_output_prefix=MY_S3_OUTPUT_PREFIX,
            local_output_dir=MY_LOCAL_OUTPUT_DIR,
            aws_region=MY_AWS_REGION
        )

        if output_json_path:
            print(f"\n--- Analysis Complete ---")
            print(f"Textract output JSON saved to: {output_json_path}")

            # Optional: Load and print some info from the JSON
            with open(output_json_path, 'r') as f:
                results = json.load(f)
            print(f"Detected {results.get('DocumentMetadata', {}).get('Pages', 'N/A')} page(s).")
            # Find signature blocks (Note: This is basic, real parsing might be more complex)
            signature_blocks = [block for block in results.get('Blocks', []) if block.get('BlockType') == 'SIGNATURE']
            print(f"Found {len(signature_blocks)} potential signature block(s).")
            if signature_blocks:
                 print(f"First signature confidence: {signature_blocks[0].get('Confidence', 'N/A')}")


    except FileNotFoundError as e:
        print(f"\nError: Input file not found. {e}")
    except Exception as e:
        print(f"\nAn error occurred during the process: {e}")

import boto3
import time
import os

def download_textract_output(job_id, output_bucket, output_prefix, local_folder):
    """
    Checks the status of a Textract job and downloads the output ZIP file if the job is complete.

    :param job_id: The Textract job ID.
    :param output_bucket: The S3 bucket where the output is stored.
    :param output_prefix: The prefix (folder path) in S3 where the output file is stored.
    :param local_folder: The local directory where the ZIP file should be saved.
    """
    textract_client = boto3.client('textract')
    s3_client = boto3.client('s3')

    # Check job status
    while True:
        response = textract_client.get_document_analysis(JobId=job_id)
        status = response['JobStatus']
        
        if status == 'SUCCEEDED':
            print("Job completed successfully.")
            break
        elif status == 'FAILED':
            print("Job failed:", response.get("StatusMessage", "No error message provided."))
            return
        else:
            print(f"Job is still {status}, waiting...")
            time.sleep(10)  # Wait before checking again

    # Find output ZIP file in S3
    output_file_key = f"{output_prefix}/{job_id}.zip"
    local_file_path = os.path.join(local_folder, f"{job_id}.zip")

    # Download file
    try:
        s3_client.download_file(output_bucket, output_file_key, local_file_path)
        print(f"Output file downloaded to: {local_file_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")

# Example usage:
# download_textract_output("your-job-id", "your-output-bucket", "your-output-prefix", "/path/to/local/folder")
