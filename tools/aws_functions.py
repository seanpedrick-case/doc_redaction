from typing import Type, List
import pandas as pd
import boto3
import tempfile
import os
from tools.helper_functions import get_or_create_env_var

PandasDataFrame = Type[pd.DataFrame]

# Get AWS credentials
bucket_name=""

RUN_AWS_FUNCTIONS = get_or_create_env_var("RUN_AWS_FUNCTIONS", "0")
print(f'The value of RUN_AWS_FUNCTIONS is {RUN_AWS_FUNCTIONS}')

AWS_REGION = get_or_create_env_var('AWS_REGION', 'eu-west-2')
print(f'The value of AWS_REGION is {AWS_REGION}')

try:
    comprehend_client = boto3.client('comprehend', region_name=AWS_REGION) 
except Exception as e:
    print(e)
    comprehend_client = ""

def get_assumed_role_info():
        sts_endpoint = 'https://sts.' + AWS_REGION + '.amazonaws.com'
        sts = boto3.client('sts', region_name=AWS_REGION, endpoint_url=sts_endpoint)
        response = sts.get_caller_identity()

        # Extract ARN of the assumed role
        assumed_role_arn = response['Arn']
        
        # Extract the name of the assumed role from the ARN
        assumed_role_name = assumed_role_arn.split('/')[-1]
        
        return assumed_role_arn, assumed_role_name

if RUN_AWS_FUNCTIONS == "1":
    try:
        bucket_name = os.environ['DOCUMENT_REDACTION_BUCKET']
        session = boto3.Session()
        # Initialize the Boto3 client for Comprehend
        
            
    except Exception as e:
        print(e)    

    try:
        assumed_role_arn, assumed_role_name = get_assumed_role_info()

        print("Assumed Role ARN:", assumed_role_arn)
        print("Assumed Role Name:", assumed_role_name)

    except Exception as e:
        print(e)




# Download direct from S3 - requires login credentials
def download_file_from_s3(bucket_name, key, local_file_path):

    s3 = boto3.client('s3')
    s3.download_file(bucket_name, key, local_file_path)
    print(f"File downloaded from S3: s3://{bucket_name}/{key} to {local_file_path}")
                         
def download_folder_from_s3(bucket_name, s3_folder, local_folder):
    """
    Download all files from an S3 folder to a local folder.
    """
    s3 = boto3.client('s3')

    # List objects in the specified S3 folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)

    # Download each object
    for obj in response.get('Contents', []):
        # Extract object key and construct local file path
        object_key = obj['Key']
        local_file_path = os.path.join(local_folder, os.path.relpath(object_key, s3_folder))

        # Create directories if necessary
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the object
        try:
            s3.download_file(bucket_name, object_key, local_file_path)
            print(f"Downloaded 's3://{bucket_name}/{object_key}' to '{local_file_path}'")
        except Exception as e:
            print(f"Error downloading 's3://{bucket_name}/{object_key}':", e)

def download_files_from_s3(bucket_name, s3_folder, local_folder, filenames):
    """
    Download specific files from an S3 folder to a local folder.
    """
    s3 = boto3.client('s3')

    print("Trying to download file: ", filenames)

    if filenames == '*':
        # List all objects in the S3 folder
        print("Trying to download all files in AWS folder: ", s3_folder)
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)

        print("Found files in AWS folder: ", response.get('Contents', []))

        filenames = [obj['Key'].split('/')[-1] for obj in response.get('Contents', [])]

        print("Found filenames in AWS folder: ", filenames)

    for filename in filenames:
        object_key = os.path.join(s3_folder, filename)
        local_file_path = os.path.join(local_folder, filename)

        # Create directories if necessary
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the object
        try:
            s3.download_file(bucket_name, object_key, local_file_path)
            print(f"Downloaded 's3://{bucket_name}/{object_key}' to '{local_file_path}'")
        except Exception as e:
            print(f"Error downloading 's3://{bucket_name}/{object_key}':", e)

def load_data_from_aws(in_aws_keyword_file, aws_password="", bucket_name=bucket_name):

    temp_dir = tempfile.mkdtemp()
    local_address_stub = temp_dir + '/doc-redaction/'
    files = []

    if not 'LAMBETH_BOROUGH_PLAN_PASSWORD' in os.environ:
        out_message = "Can't verify password for dataset access. Do you have a valid AWS connection? Data not loaded."
        return files, out_message
    
    if aws_password:
        if "Lambeth borough plan" in in_aws_keyword_file and aws_password == os.environ['LAMBETH_BOROUGH_PLAN_PASSWORD']:

            s3_folder_stub = 'example-data/lambeth-borough-plan/latest/'

            local_folder_path = local_address_stub                

            # Check if folder exists
            if not os.path.exists(local_folder_path):
                print(f"Folder {local_folder_path} does not exist! Making folder.")

                os.mkdir(local_folder_path)

            # Check if folder is empty
            if len(os.listdir(local_folder_path)) == 0:
                print(f"Folder {local_folder_path} is empty")
                # Download data
                download_files_from_s3(bucket_name, s3_folder_stub, local_folder_path, filenames='*')

                print("AWS data downloaded")

            else:
                print(f"Folder {local_folder_path} is not empty")

            #files = os.listdir(local_folder_stub)
            #print(files)

            files = [os.path.join(local_folder_path, f) for f in os.listdir(local_folder_path) if os.path.isfile(os.path.join(local_folder_path, f))]

            out_message = "Data successfully loaded from AWS"
            print(out_message)

        else:
            out_message = "Data not loaded from AWS"
            print(out_message)
    else:
        out_message = "No password provided. Please ask the data team for access if you need this."
        print(out_message)

    return files, out_message

def upload_file_to_s3(local_file_paths:List[str], s3_key:str, s3_bucket:str=bucket_name):
    """
    Uploads a file from local machine to Amazon S3.

    Args:
    - local_file_path: Local file path(s) of the file(s) to upload.
    - s3_key: Key (path) to the file in the S3 bucket.
    - s3_bucket: Name of the S3 bucket.

    Returns:
    - Message as variable/printed to console
    """
    final_out_message = []

    s3_client = boto3.client('s3')

    if isinstance(local_file_paths, str):
        local_file_paths = [local_file_paths]

    for file in local_file_paths:
        if s3_client:
            #print(s3_client)
            try:
                # Get file name off file path
                file_name = os.path.basename(file)

                s3_key_full = s3_key + file_name
                print("S3 key: ", s3_key_full)

                s3_client.upload_file(file, s3_bucket, s3_key_full)
                out_message = "File " + file_name + " uploaded successfully!"
                print(out_message)
            
            except Exception as e:
                out_message = f"Error uploading file(s): {e}"
                print(out_message)

            final_out_message.append(out_message)
            final_out_message_str = '\n'.join(final_out_message)

        else: final_out_message_str = "Could not connect to AWS."

    return final_out_message_str
        
    
