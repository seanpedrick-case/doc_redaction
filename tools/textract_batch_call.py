import boto3
import time
import os
import pandas as pd
import json
import logging
import datetime
import gradio as gr
from gradio import FileData
from typing import List
from io import StringIO
from urllib.parse import urlparse
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError, TokenRetrievalError
from tools.config import TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET, OUTPUT_FOLDER, AWS_REGION, DOCUMENT_REDACTION_BUCKET, LOAD_PREVIOUS_TEXTRACT_JOBS_S3, TEXTRACT_JOBS_S3_LOC, TEXTRACT_JOBS_LOCAL_LOC, TEXTRACT_JOBS_S3_INPUT_LOC, RUN_AWS_FUNCTIONS, INPUT_FOLDER
from tools.aws_functions import download_file_from_s3
from tools.file_conversion import get_input_file_names
from tools.helper_functions import get_file_name_without_type

def analyse_document_with_textract_api(
    local_pdf_path: str,
    s3_input_prefix: str,
    s3_output_prefix: str,
    job_df:pd.DataFrame,
    s3_bucket_name: str = TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET,
    local_output_dir: str = OUTPUT_FOLDER,    
    analyse_signatures:List[str] = [],
    successful_job_number:int=0,
    total_document_page_count:int=1,
    general_s3_bucket_name: str = DOCUMENT_REDACTION_BUCKET,
    aws_region: str = AWS_REGION # Optional: specify region if not default
    ):
    """
    Uploads a local PDF to S3, starts a Textract analysis job (detecting text & signatures),
    waits for completion, and downloads the output JSON from S3 to a local directory.

    Args:
        local_pdf_path (str): Path to the local PDF file.
        s3_bucket_name (str): Name of the S3 bucket to use.
        s3_input_prefix (str): S3 prefix (folder) to upload the input PDF.
        s3_output_prefix (str): S3 prefix (folder) where Textract should write output.
        job_df (pd.DataFrame): Dataframe containing information from previous Textract API calls.
        s3_bucket_name (str, optional): S3 bucket in which to save API call outputs.
        local_output_dir (str, optional): Local directory to save the downloaded JSON results.        
        analyse_signatures (List[str], optional): Analyse signatures? Default is no.
        successful_job_number (int): The number of successful jobs that have been submitted in this session.
        total_document_page_count (int): The number of pages in the document
        aws_region (str, optional): AWS region name. Defaults to boto3 default region.

    Returns:
        str: Path to the downloaded local JSON output file, or None if failed.

    Raises:
        FileNotFoundError: If the local_pdf_path does not exist.
        boto3.exceptions.NoCredentialsError: If AWS credentials are not found.
        Exception: For other AWS errors or job failures.
    """

    # This is a variable that is written to logs to indicate that a Textract API call was made
    is_a_textract_api_call = True

    # Keep only latest pdf path if it's a list
    if isinstance(local_pdf_path, list):
        local_pdf_path = local_pdf_path[-1]

    if not os.path.exists(local_pdf_path):
        raise FileNotFoundError(f"Input document not found {local_pdf_path}")

    if not os.path.exists(local_output_dir):
        os.makedirs(local_output_dir)
        log_message = f"Created local output directory: {local_output_dir}"
        print(log_message)
        #logging.info(log_message)

    # Initialize boto3 clients
    session = boto3.Session(region_name=aws_region)
    s3_client = session.client('s3')
    textract_client = session.client('textract')

    # --- 1. Upload PDF to S3 ---
    pdf_filename = os.path.basename(local_pdf_path)
    s3_input_key = os.path.join(s3_input_prefix, pdf_filename).replace("\\", "/") # Ensure forward slashes for S3

    log_message = f"Uploading '{local_pdf_path}' to 's3://{s3_bucket_name}/{s3_input_key}'..."
    print(log_message)
    #logging.info(log_message)
    try:
        s3_client.upload_file(local_pdf_path, s3_bucket_name, s3_input_key)
        log_message = "Upload successful."
        print(log_message)
        #logging.info(log_message)
    except Exception as e:
        log_message = f"Failed to upload PDF to S3: {e}"
        print(log_message)
        #logging.error(log_message)
        raise

    # If job_df is not empty
    if not job_df.empty:
        if "file_name" in job_df.columns:
            matching_job_id_file_names = job_df.loc[(job_df["file_name"] == pdf_filename) & (job_df["signature_extraction"].astype(str) == str(analyse_signatures)), "file_name"]

            if len(matching_job_id_file_names) > 0:
                    raise Exception("Existing Textract outputs found. No need to re-analyse. Please download existing results from the list")

    # --- 2. Start Textract Document Analysis ---
    message = "Starting Textract document analysis job..."
    print(message)
    #logging.info("Starting Textract document analysis job...")

    try:
        if "Extract signatures" in analyse_signatures:
            response = textract_client.start_document_analysis(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': s3_bucket_name,
                        'Name': s3_input_key
                    }
                },
                FeatureTypes=['SIGNATURES'], # Analyze for signatures, forms, and tables
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
            job_type="document_analysis"

        else:
            response = textract_client.start_document_text_detection(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': s3_bucket_name,
                        'Name': s3_input_key
                    }
                },
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
            job_type="document_text_detection"

        job_id = response['JobId']
        print(f"Textract job started with JobId: {job_id}")
        #logging.info(f"Textract job started with JobId: {job_id}")

        # Write job_id to memory
        # Prepare CSV in memory
        log_csv_key_location = f"{s3_output_prefix}/textract_document_jobs.csv"
        job_location_full = f"s3://{s3_bucket_name}/{s3_output_prefix}/{job_id}/"

        csv_buffer = StringIO()
        log_df = pd.DataFrame([{
            'job_id': job_id,
            'file_name': pdf_filename,
            'job_type': job_type,
            'signature_extraction':analyse_signatures,
            's3_location': job_location_full,
            'job_date_time': datetime.datetime.now()
        }])

        # File path
        log_file_path = os.path.join(local_output_dir, "textract_document_jobs.csv")

        # Check if file exists
        file_exists = os.path.exists(log_file_path)

        # Append to CSV if it exists, otherwise write with header
        log_df.to_csv(log_file_path, mode='a', index=False, header=not file_exists)
        
        #log_df.to_csv(csv_buffer)

        # Upload the file
        s3_client.upload_file(log_file_path, general_s3_bucket_name, log_csv_key_location)

        # Upload to S3 (overwrite existing file)
        #s3_client.put_object(Bucket=general_s3_bucket_name, Key=log_csv_key_location, Body=csv_buffer.getvalue())
        print(f"Job ID written to {log_csv_key_location}")
        #logging.info(f"Job ID written to s3://{s3_bucket_name}/{s3_output_prefix}/textract_document_jobs.csv")

    except Exception as e:
        error = f"Failed to start Textract job: {e}"
        print(error)
        #logging.error(error)
        raise

    successful_job_number += 1
    total_number_of_textract_page_calls = total_document_page_count

    return f"Textract analysis job submitted, job ID:{job_id}", job_id, job_type, successful_job_number, is_a_textract_api_call, total_number_of_textract_page_calls

def return_job_status(job_id:str,
                     response:dict,
                     attempts:int,
                     poll_interval_seconds: int = 0,
                     max_polling_attempts: int = 1 # ~10 minutes total wait time
                     ):
    '''
    Poll Textract for the current status of a previously-submitted job.
    '''

    job_status = response['JobStatus']
    logging.info(f"Polling attempt {attempts}/{max_polling_attempts}. Job status: {job_status}")

    if job_status == 'IN_PROGRESS':
        pass
        #time.sleep(poll_interval_seconds)
    elif job_status == 'SUCCEEDED':
        logging.info("Textract job succeeded.")
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
    
    return job_status

def download_textract_job_files(s3_client:str,
                                s3_bucket_name:str,
                                s3_output_key_prefix:str,
                                pdf_filename:str,
                                job_id:str,
                                local_output_dir:str):    
    '''
    Download and combine selected job files from the AWS Textract service.
    '''

    #print("s3_output_key_prefix at download:", s3_output_key_prefix)

    list_response = s3_client.list_objects_v2(
        Bucket=s3_bucket_name,
        Prefix=s3_output_key_prefix
    )

    output_files = list_response.get('Contents', [])
    if not output_files:
        # Sometimes Textract might take a moment longer to write the output after SUCCEEDED status
        #logging.warning("No output files found immediately after job success. Waiting briefly and retrying list...")
        #time.sleep(5)
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
    json_files_to_download = [
    f for f in output_files 
    if f['Key'] != s3_output_key_prefix and not f['Key'].endswith('/') and 'access_check' not in f['Key']
]

    #print("json_files_to_download:", json_files_to_download)

    if not json_files_to_download:
        error = f"No JSON files found (only prefix marker?) in s3://{s3_bucket_name}/{s3_output_key_prefix}"
        print(error)
        #logging.error(error)
        raise FileNotFoundError(error)

    combined_blocks = []

    for f in sorted(json_files_to_download, key=lambda x: x['Key']):  # Optional: sort to ensure consistent order
        obj = s3_client.get_object(Bucket=s3_bucket_name, Key=f['Key'])
        data = json.loads(obj['Body'].read())
        
        # Assuming Textract-style output with a "Blocks" key
        if "Blocks" in data:
            combined_blocks.extend(data["Blocks"])
        else:
            logging.warning(f"No 'Blocks' key in file: {f['Key']}")

    # Build final combined JSON structure
    combined_output = {
        "DocumentMetadata": {
            "Pages": len(set(block.get('Page', 1) for block in combined_blocks))
        },
        "Blocks": combined_blocks,
        "JobStatus": "SUCCEEDED"
    }

    output_filename_base = os.path.basename(pdf_filename)
    output_filename_base_no_ext = os.path.splitext(output_filename_base)[0]
    local_output_filename = f"{output_filename_base_no_ext}_textract.json"
    local_output_path = os.path.join(local_output_dir, local_output_filename)

    with open(local_output_path, 'w') as f:
        json.dump(combined_output, f)

    print(f"Combined Textract output written to {local_output_path}")

    # logging.info(f"Downloading Textract output from 's3://{s3_bucket_name}/{s3_output_key}' to '{local_output_path}'...")
    # s3_client.download_file(s3_bucket_name, s3_output_key, local_output_path)
    # logging.info("Download successful.")
    downloaded_file_path = local_output_path

    # Log if multiple files were found, as user might need to handle them
    #if len(json_files_to_download) > 1:
    #    logging.warning(f"Multiple output files found in S3 output location. Downloaded the first: '{s3_output_key}'. Other files exist.")

    return downloaded_file_path

def check_for_provided_job_id(job_id:str):
    if not job_id:
        raise Exception("Please provide a job ID.")    
    return

def load_pdf_job_file_from_s3(
    load_s3_jobs_input_loc,
    pdf_filename,
    local_output_dir,
    s3_bucket_name,
    RUN_AWS_FUNCTIONS=RUN_AWS_FUNCTIONS):    

    try:
        print("load_s3_jobs_input_loc:", load_s3_jobs_input_loc)
        pdf_file_location = ''
        doc_file_name_no_extension_textbox = ''

        s3_input_key_prefix = os.path.join(load_s3_jobs_input_loc, pdf_filename).replace("\\", "/")
        s3_input_key_prefix = s3_input_key_prefix + ".pdf"
        print("s3_input_key_prefix:", s3_input_key_prefix)

        local_input_file_path = os.path.join(local_output_dir, pdf_filename)
        local_input_file_path = local_input_file_path + ".pdf"

        print("input to s3 download:", s3_bucket_name, s3_input_key_prefix, local_input_file_path)

        download_file_from_s3(s3_bucket_name, s3_input_key_prefix, local_input_file_path, RUN_AWS_FUNCTIONS= RUN_AWS_FUNCTIONS)
        
        pdf_file_location = [local_input_file_path]
        doc_file_name_no_extension_textbox = get_file_name_without_type(pdf_filename)
    except Exception as e:
        print("Could not download PDF job file from S3 due to:", e)        

    return pdf_file_location, doc_file_name_no_extension_textbox

def replace_existing_pdf_input_for_whole_document_outputs(    
    load_s3_jobs_input_loc:str,
    pdf_filename:str,
    local_output_dir:str,
    s3_bucket_name:str,
    in_doc_files:FileData=[],
    input_folder:str=INPUT_FOLDER,
    RUN_AWS_FUNCTIONS=RUN_AWS_FUNCTIONS,
    progress = gr.Progress(track_tqdm=True)):

    progress(0.1, "Loading PDF from s3")

    if in_doc_files:
        doc_file_name_no_extension_textbox, doc_file_name_with_extension_textbox, doc_full_file_name_textbox, doc_file_name_textbox_list, total_pdf_page_count = get_input_file_names(in_doc_files)

        if pdf_filename == doc_file_name_no_extension_textbox:
            print("Existing loaded PDF file has same name as file from S3")
            doc_file_name_no_extension_textbox = pdf_filename
            downloaded_pdf_file_location = in_doc_files
        else:
            downloaded_pdf_file_location, doc_file_name_no_extension_textbox = load_pdf_job_file_from_s3(load_s3_jobs_input_loc, pdf_filename, local_output_dir, s3_bucket_name, RUN_AWS_FUNCTIONS=RUN_AWS_FUNCTIONS)

            doc_file_name_no_extension_textbox, doc_file_name_with_extension_textbox, doc_full_file_name_textbox, doc_file_name_textbox_list, total_pdf_page_count = get_input_file_names(downloaded_pdf_file_location)
    else:               
        downloaded_pdf_file_location, doc_file_name_no_extension_textbox = load_pdf_job_file_from_s3(load_s3_jobs_input_loc, pdf_filename, local_output_dir, s3_bucket_name, RUN_AWS_FUNCTIONS=RUN_AWS_FUNCTIONS)

        doc_file_name_no_extension_textbox, doc_file_name_with_extension_textbox, doc_full_file_name_textbox, doc_file_name_textbox_list, total_pdf_page_count = get_input_file_names(downloaded_pdf_file_location)

    return downloaded_pdf_file_location, doc_file_name_no_extension_textbox, doc_file_name_with_extension_textbox, doc_full_file_name_textbox, doc_file_name_textbox_list, total_pdf_page_count

def poll_whole_document_textract_analysis_progress_and_download(
    job_id:str,
    job_type_dropdown:str,
    s3_output_prefix: str,
    pdf_filename:str,
    job_df:pd.DataFrame,
    s3_bucket_name: str = TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_BUCKET,
    local_output_dir: str = OUTPUT_FOLDER,
    load_s3_jobs_loc:str=TEXTRACT_JOBS_S3_LOC,
    load_local_jobs_loc:str=TEXTRACT_JOBS_LOCAL_LOC,    
    aws_region: str = AWS_REGION, # Optional: specify region if not default
    load_jobs_from_s3:str = LOAD_PREVIOUS_TEXTRACT_JOBS_S3,    
    poll_interval_seconds: int = 1,
    max_polling_attempts: int = 1, # ~10 minutes total wait time):
    progress = gr.Progress(track_tqdm=True)
    ):
    '''
    Poll AWS for the status of a Textract API job. Return status, and if finished, combine and download results into a locally-stored json file for further processing by the app.
    '''

    progress(0.1, "Querying AWS Textract for status of document analysis job")

    if job_id:
        # Initialize boto3 clients
        session = boto3.Session(region_name=aws_region)
        s3_client = session.client('s3')
        textract_client = session.client('textract')

        # --- 3. Poll for Job Completion ---
        job_status = 'IN_PROGRESS'
        attempts = 0

        message = "Polling Textract for job completion status..."
        print(message)
        #logging.info("Polling Textract for job completion status...")

        # Update Textract document history df
        try:
            job_df = load_in_textract_job_details(load_s3_jobs=load_jobs_from_s3,
                                        load_s3_jobs_loc=load_s3_jobs_loc,
                                        load_local_jobs_loc=load_local_jobs_loc)
        except Exception as e:
            #logging.error(f"Failed to update job details dataframe: {e}")
            print(f"Failed to update job details dataframe: {e}")
            #raise

        while job_status == 'IN_PROGRESS' and attempts <= max_polling_attempts:
            attempts += 1
            try:
                if job_type_dropdown=="document_analysis":
                    response = textract_client.get_document_analysis(JobId=job_id)
                    job_status = return_job_status(job_id, response, attempts, poll_interval_seconds, max_polling_attempts)
                elif job_type_dropdown=="document_text_detection":
                    response = textract_client.get_document_text_detection(JobId=job_id)
                    job_status = return_job_status(job_id, response, attempts, poll_interval_seconds, max_polling_attempts)
                else:
                    error = f"Unknown job type, cannot poll job"
                    print(error)
                    #logging.error(f"Invalid JobId: {job_id}. This might happen if the job expired (older than 7 days) or never existed.")
                    raise

            except textract_client.exceptions.InvalidJobIdException:
                error_message = f"Invalid JobId: {job_id}. This might happen if the job expired (older than 7 days) or never existed."
                print(error_message)
                logging.error(error_message)
                raise
            except Exception as e:
                error_message = f"Error while polling Textract status for job {job_id}: {e}"
                print(error_message)
                logging.error(error_message)
                raise

        downloaded_file_path = None
        if job_status == 'SUCCEEDED':
            #raise TimeoutError(f"Textract job {job_id} did not complete successfully within the polling limit.")
            # 3b - Replace PDF file name if it exists in the job dataframe   

            progress(0.5, "Document analysis task outputs found. Downloading from S3")                 

            # If job_df is not empty
            if not job_df.empty:
                if "file_name" in job_df.columns:
                    matching_job_id_file_names = job_df.loc[job_df["job_id"] == job_id, "file_name"]

                    if pdf_filename and not matching_job_id_file_names.empty:
                        if pdf_filename == matching_job_id_file_names.iloc[0]:
                            raise Exception("Existing Textract outputs found. No need to re-download.")

                    if not matching_job_id_file_names.empty:
                        pdf_filename = matching_job_id_file_names.iloc[0]
                    else:
                        pdf_filename = "unknown_file"

            # --- 4. Download Output JSON from S3 ---
            # Textract typically creates output under s3_output_prefix/job_id/
            # There might be multiple JSON files if pagination occurred during writing.
            # Usually, for smaller docs, there's one file, often named '1'.
            # For robust handling, list objects and find the JSON(s).    

            s3_output_key_prefix = os.path.join(s3_output_prefix, job_id).replace("\\", "/") + "/"
            logging.info(f"Searching for output files in s3://{s3_bucket_name}/{s3_output_key_prefix}")

            try:
                downloaded_file_path = download_textract_job_files(s3_client,
                                                s3_bucket_name,
                                                s3_output_key_prefix,
                                                pdf_filename,
                                                job_id,
                                                local_output_dir)

            except Exception as e:
                #logging.error(f"Failed to download or process Textract output from S3: {e}")
                print(f"Failed to download or process Textract output from S3: {e}")
                raise

    else:
        raise Exception("No Job ID provided.")        
    
    output_pdf_filename = get_file_name_without_type(pdf_filename)

    return downloaded_file_path, job_status, job_df, output_pdf_filename

def load_in_textract_job_details(load_s3_jobs:str=LOAD_PREVIOUS_TEXTRACT_JOBS_S3,
                                     load_s3_jobs_loc:str=TEXTRACT_JOBS_S3_LOC,
                                     load_local_jobs_loc:str=TEXTRACT_JOBS_LOCAL_LOC,
                                     document_redaction_bucket:str=DOCUMENT_REDACTION_BUCKET,
                                     aws_region:str=AWS_REGION):
    '''
    Load in a dataframe of jobs previous submitted to the Textract API service.
    '''
    job_df = pd.DataFrame(columns=['job_id','file_name','job_type','signature_extraction','s3_location','job_date_time'])

    # Initialize boto3 clients
    session = boto3.Session(region_name=aws_region)
    s3_client = session.client('s3')

    local_output_path = f'{load_local_jobs_loc}/textract_document_jobs.csv'

    if load_s3_jobs == 'True':
        s3_output_key = f'{load_s3_jobs_loc}/textract_document_jobs.csv'
                
        try:
            s3_client.head_object(Bucket=document_redaction_bucket, Key=s3_output_key)
            #print(f"File exists. Downloading from '{s3_output_key}' to '{local_output_path}'...")
            s3_client.download_file(document_redaction_bucket, s3_output_key, local_output_path)
            #print("Download successful.")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print("Log file does not exist in S3.")
            else:
                print(f"Unexpected error occurred: {e}")
        except (NoCredentialsError, PartialCredentialsError, TokenRetrievalError) as e:
            print(f"AWS credential issue encountered: {e}")
            print("Skipping S3 log file download.")

    # If the log path exists, load it in
    if os.path.exists(local_output_path):
        print("Found log file in local path")
        job_df = pd.read_csv(local_output_path)

        if "job_date_time" in job_df.columns:
            job_df["job_date_time"] = pd.to_datetime(job_df["job_date_time"], errors='coerce')
            # Keep only jobs that have been completed in the last 7 days
            cutoff_time = pd.Timestamp.now() - pd.Timedelta(days=7)
            job_df = job_df.loc[job_df["job_date_time"] >= cutoff_time,:]

    return job_df

def download_textract_output(job_id:str,
                             output_bucket:str,
                             output_prefix:str,
                             local_folder:str):
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
            print(f"Job is still {status}.")
            #time.sleep(10)  # Wait before checking again

    # Find output ZIP file in S3
    output_file_key = f"{output_prefix}/{job_id}.zip"
    local_file_path = os.path.join(local_folder, f"{job_id}.zip")

    # Download file
    try:
        s3_client.download_file(output_bucket, output_file_key, local_file_path)
        print(f"Output file downloaded to: {local_file_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")

def check_textract_outputs_exist(textract_output_found_checkbox):
        if textract_output_found_checkbox == True:
            print("Textract outputs found")
            return
        else: raise Exception("Relevant Textract outputs not found. Please ensure you have selected to correct results output and you have uploaded the relevant document file in 'Choose document or image file...' above")