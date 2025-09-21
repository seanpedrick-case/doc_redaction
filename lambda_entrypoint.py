import boto3
import os
import json

# Import the main function from your CLI script
from cli_redact import main as cli_main

print("Lambda entrypoint loading...")

# Initialize S3 client outside the handler for connection reuse
s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION", "eu-west-2"))
print("S3 client initialised")

# Lambda's only writable directory
TMP_DIR = "/tmp"
INPUT_DIR = os.path.join(TMP_DIR, "input")
OUTPUT_DIR = os.path.join(TMP_DIR, "output")

def download_file_from_s3(bucket_name, key, download_path):
    """Download a file from S3 to the local filesystem."""
    try:
        s3_client.download_file(bucket_name, key, download_path)
        print(f"Successfully downloaded s3://{bucket_name}/{key} to {download_path}")
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        raise

def upload_directory_to_s3(local_directory, bucket_name, s3_prefix):
    """Upload all files from a local directory to an S3 prefix."""
    for root, _, files in os.walk(local_directory):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            # Create a relative path to maintain directory structure if needed
            relative_path = os.path.relpath(local_file_path, local_directory)
            output_key = os.path.join(s3_prefix, relative_path)
            
            try:
                s3_client.upload_file(local_file_path, bucket_name, output_key)
                print(f"Successfully uploaded {local_file_path} to s3://{bucket_name}/{output_key}")
            except Exception as e:
                print(f"Error uploading to S3: {e}")
                raise

def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event)}")

    # 1. Setup temporary directories
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Extract information from the event
    # Assumes the event is triggered by S3 and may contain an 'arguments' payload
    try:
        record = event['Records'][0]
        bucket_name = record['s3']['bucket']['name']
        input_key = record['s3']['object']['key']
        
        # The user metadata can be used to pass arguments
        # This is more robust than embedding them in the main event body
        response = s3_client.head_object(Bucket=bucket_name, Key=input_key)
        metadata = response.get('Metadata', {})
        # Arguments can be passed as a JSON string in metadata
        arguments = json.loads(metadata.get('arguments', '{}'))

    except (KeyError, IndexError) as e:
        print(f"Could not parse S3 event record: {e}. Checking for direct invocation payload.")
        # Fallback for direct invocation (e.g., from Step Functions or manual test)
        bucket_name = event.get('bucket_name')
        input_key = event.get('input_key')
        arguments = event.get('arguments', {})
        if not all([bucket_name, input_key]):
            raise ValueError("Missing 'bucket_name' or 'input_key' in direct invocation event.")

    print(f"Processing s3://{bucket_name}/{input_key}")
    print(f"With arguments: {arguments}")
    
    # 3. Download the main input file
    input_file_path = os.path.join(INPUT_DIR, os.path.basename(input_key))
    download_file_from_s3(bucket_name, input_key, input_file_path)

    # 4. Prepare arguments for the CLI function
    # This dictionary should mirror the one in your app.py's "direct mode"
    cli_args = {
        'task': arguments.get('task', 'redact'),
        'input_file': input_file_path,
        'output_dir': OUTPUT_DIR,
        'input_dir': INPUT_DIR,
        'language': arguments.get('language', 'en_core_web_lg'),
        'pii_detector': arguments.get('pii_detector', 'Local'), # Default to local
        'username': arguments.get('username', 'lambda_user'),
        'save_to_user_folders': arguments.get('save_to_user_folders', 'False'),
        'ocr_method': arguments.get('ocr_method', 'Tesseract OCR - all PDF types'),
        'page_min': int(arguments.get('page_min', 0)),
        'page_max': int(arguments.get('page_max', 0)),
        'handwrite_signature_extraction': arguments.get('handwrite_signature_checkbox', ['Extract handwriting', 'Extract signatures']),
        'extract_forms': arguments.get('extract_forms', False),
        'extract_tables': arguments.get('extract_tables', False),
        'extract_layout': arguments.get('extract_layout', False),
        
        # General arguments
        'local_redact_entities': arguments.get('local_redact_entities', []),
        'aws_redact_entities': arguments.get('aws_redact_entities', []),
        'cost_code': arguments.get('cost_code', ''),
        'save_logs_to_csv': arguments.get('save_logs_to_csv', 'False'),
        'save_logs_to_dynamodb': arguments.get('save_logs_to_dynamodb', 'False'),
        'display_file_names_in_logs': arguments.get('display_file_names_in_logs', 'True'),
        'upload_logs_to_s3': arguments.get('upload_logs_to_s3', 'False'),
        's3_logs_prefix': arguments.get('s3_logs_prefix', ''),
        'do_initial_clean': arguments.get('do_initial_clean', 'False'),
        
        # PDF/Image specific arguments
        'images_dpi': float(arguments.get('images_dpi', 300.0)),
        'chosen_local_ocr_model': arguments.get('chosen_local_ocr_model', 'tesseract'),
        'preprocess_local_ocr_images': arguments.get('preprocess_local_ocr_images', 'False'),
        
        # Handle optional files like allow/deny lists
        'allow_list_file': arguments.get('allow_list_file', ""),
        'deny_list_file': arguments.get('deny_list_file', ""),
        'redact_whole_page_file': arguments.get('redact_whole_page_file', ""),
        
        # Tabular/Anonymisation arguments
        'excel_sheets': arguments.get('excel_sheets', []),
        'fuzzy_mistakes': int(arguments.get('fuzzy_mistakes', 0)),
        'match_fuzzy_whole_phrase_bool': arguments.get('match_fuzzy_whole_phrase_bool', 'True'),
        
        # Deduplication specific arguments
        'duplicate_type': arguments.get('duplicate_type', 'pages'),
        'similarity_threshold': float(arguments.get('similarity_threshold', 0.95)),
        'min_word_count': int(arguments.get('min_word_count', 3)),
        'min_consecutive_pages': int(arguments.get('min_consecutive_pages', 1)),
        'greedy_match': arguments.get('greedy_match', 'False'),
        'combine_pages': arguments.get('combine_pages', 'True'),
        'search_query': arguments.get('search_query', ""),
        'text_columns': arguments.get('text_columns', []),
        'remove_duplicate_rows': arguments.get('remove_duplicate_rows', 'True'),
        'anon_strategy': arguments.get('anon_strategy', 'redact'),
        
        # Textract specific arguments
        'textract_action': arguments.get('textract_action', ''),
        'job_id': arguments.get('job_id', ''),
        'extract_signatures': arguments.get('extract_signatures', False),
        'textract_bucket': arguments.get('textract_bucket', ''),
        'textract_input_prefix': arguments.get('textract_input_prefix', ''),
        'textract_output_prefix': arguments.get('textract_output_prefix', ''),
        's3_textract_document_logs_subfolder': arguments.get('s3_textract_document_logs_subfolder', ''),
        'local_textract_document_logs_subfolder': arguments.get('local_textract_document_logs_subfolder', ''),
        'poll_interval': int(arguments.get('poll_interval', 30)),
        'max_poll_attempts': int(arguments.get('max_poll_attempts', 120)),
        
        # AWS credentials (use IAM Role instead of keys)
        'aws_access_key': None,
        'aws_secret_key': None,
        'aws_region': os.getenv("AWS_REGION", ""),
        's3_bucket': bucket_name,
        
        # Set defaults for boolean flags
        'prepare_images': arguments.get('prepare_images', True),
        'compress_redacted_pdf': arguments.get('compress_redacted_pdf', False),
        'return_pdf_end_of_redaction': arguments.get('return_pdf_end_of_redaction', True)
    }

    # Combine extraction options
    extraction_options = list(cli_args['handwrite_signature_extraction']) if cli_args['handwrite_signature_extraction'] else []
    if cli_args['extract_forms']:
        extraction_options.append('Extract forms')
    if cli_args['extract_tables']:
        extraction_options.append('Extract tables')
    if cli_args['extract_layout']:
        extraction_options.append('Extract layout')
    cli_args['handwrite_signature_extraction'] = extraction_options

    # Download optional files if they are specified
    allow_list_key = arguments.get('allow_list_file')
    if allow_list_key:
        allow_list_path = os.path.join(INPUT_DIR, 'allow_list.csv')
        download_file_from_s3(bucket_name, allow_list_key, allow_list_path)
        cli_args['allow_list_file'] = allow_list_path
        
    deny_list_key = arguments.get('deny_list_file')
    if deny_list_key:
        deny_list_path = os.path.join(INPUT_DIR, 'deny_list.csv')
        download_file_from_s3(bucket_name, deny_list_key, deny_list_path)
        cli_args['deny_list_file'] = deny_list_path

    # 5. Execute the main application logic
    try:
        print("--- Starting CLI Redact Main Function ---")
        print(f"Arguments passed to cli_main: {cli_args}")
        cli_main(direct_mode_args=cli_args)
        print("--- CLI Redact Main Function Finished ---")
    except Exception as e:
        print(f"An error occurred during CLI execution: {e}")
        # Optionally, re-raise the exception to make the Lambda fail
        raise

    # 6. Upload results back to S3
    output_s3_prefix = f"output/{os.path.splitext(os.path.basename(input_key))[0]}"
    print(f"Uploading contents of {OUTPUT_DIR} to s3://{bucket_name}/{output_s3_prefix}/")
    upload_directory_to_s3(OUTPUT_DIR, bucket_name, output_s3_prefix)

    return {
        "statusCode": 200,
        "body": json.dumps(f"Processing complete for {input_key}. Output saved to s3://{bucket_name}/{output_s3_prefix}/")
    }