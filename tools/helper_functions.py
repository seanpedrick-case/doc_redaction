import os
import re
import gradio as gr
import pandas as pd
import unicodedata
from typing import List
from gradio_image_annotation import image_annotator

def reset_state_vars():
    return [], [], pd.DataFrame(), pd.DataFrame(), 0, "", image_annotator(
            label="Modify redaction boxes",
            label_list=["Redaction"],
            label_colors=[(0, 0, 0)],
            show_label=False,
            sources=None,#["upload"],
            show_clear_button=False,
            show_share_button=False,
            show_remove_button=False,
            interactive=False
        )

def get_or_create_env_var(var_name, default_value):
    # Get the environment variable if it exists
    value = os.environ.get(var_name)
    
    # If it doesn't exist, set it to the default value
    if value is None:
        os.environ[var_name] = default_value
        value = default_value
    
    return value


# Names for options labels
text_ocr_option = "Local model - selectable text"
tesseract_ocr_option = "Local OCR model - PDFs without selectable text"
textract_option = "AWS Textract service - all PDF types"

local_pii_detector = "Local"
aws_pii_detector  = "AWS Comprehend"

output_folder = get_or_create_env_var('GRADIO_OUTPUT_FOLDER', 'output/')
print(f'The value of GRADIO_OUTPUT_FOLDER is {output_folder}')

input_folder = get_or_create_env_var('GRADIO_INPUT_FOLDER', 'input/')
print(f'The value of GRADIO_INPUT_FOLDER is {input_folder}')

def load_in_default_allow_list(allow_list_file_path):
    if isinstance(allow_list_file_path, str):
        allow_list_file_path = [allow_list_file_path]
    return allow_list_file_path


def get_file_path_end(file_path):
    # First, get the basename of the file (e.g., "example.txt" from "/path/to/example.txt")
    basename = os.path.basename(file_path)
    
    # Then, split the basename and its extension and return only the basename without the extension
    filename_without_extension, _ = os.path.splitext(basename)

    #print(filename_without_extension)
    
    return filename_without_extension

def detect_file_type(filename):
    """Detect the file type based on its extension."""
    if (filename.endswith('.csv')) | (filename.endswith('.csv.gz')) | (filename.endswith('.zip')):
        return 'csv'
    elif filename.endswith('.xlsx'):
        return 'xlsx'
    elif filename.endswith('.parquet'):
        return 'parquet'
    elif filename.endswith('.pdf'):
        return 'pdf'
    elif filename.endswith('.jpg'):
        return 'jpg'
    elif filename.endswith('.jpeg'):
        return 'jpeg'
    elif filename.endswith('.png'):
        return 'png'
    else:
        raise ValueError("Unsupported file type.")

def read_file(filename):
    """Read the file based on its detected type."""
    file_type = detect_file_type(filename)
    
    if file_type == 'csv':
        return pd.read_csv(filename, low_memory=False)
    elif file_type == 'xlsx':
        return pd.read_excel(filename)
    elif file_type == 'parquet':
        return pd.read_parquet(filename)

def ensure_output_folder_exists():
    """Checks if the 'output/' folder exists, creates it if not."""

    folder_name = "output/"

    if not os.path.exists(folder_name):
        # Create the folder if it doesn't exist
        os.makedirs(folder_name)
        print(f"Created the 'output/' folder.")
    else:
        print(f"The 'output/' folder already exists.")

def custom_regex_load(in_file:List[str], file_type:str = "Allow list"):
    '''
    When file is loaded, update the column dropdown choices and write to relevant data states.
    '''

    custom_regex = pd.DataFrame()

    if in_file:
        file_list = [string.name for string in in_file]

        regex_file_names = [string for string in file_list if "csv" in string.lower()]
        if regex_file_names:
            regex_file_name = regex_file_names[0]
            custom_regex = pd.read_csv(regex_file_name, low_memory=False, header=None)
            #regex_file_name_no_ext = get_file_path_end(regex_file_name)

            output_text = file_type + " file loaded."

            print(output_text)
    else:
        output_text = "No file provided."
        print(output_text)
        return output_text, custom_regex
       
    return output_text, custom_regex

def put_columns_in_df(in_file):
    new_choices = []
    concat_choices = []
    all_sheet_names = []
    number_of_excel_files = 0
    
    for file in in_file:
        file_name = file.name
        file_type = detect_file_type(file_name)
        print("File type is:", file_type)

        if file_type == 'xlsx':
            number_of_excel_files += 1
            new_choices = []
            print("Running through all xlsx sheets")
            anon_xlsx = pd.ExcelFile(file_name)
            new_sheet_names = anon_xlsx.sheet_names
            # Iterate through the sheet names
            for sheet_name in new_sheet_names:
                # Read each sheet into a DataFrame
                df = pd.read_excel(file_name, sheet_name=sheet_name)

                # Process the DataFrame (e.g., print its contents)
                print(f"Sheet Name: {sheet_name}")
                print(df.head())  # Print the first few rows

                new_choices.extend(list(df.columns))

            all_sheet_names.extend(new_sheet_names)

        else:
            df = read_file(file_name)
            new_choices = list(df.columns)

        concat_choices.extend(new_choices)
        
    # Drop duplicate columns
    concat_choices = list(set(concat_choices))

    if number_of_excel_files > 0:      
        return gr.Dropdown(choices=concat_choices, value=concat_choices), gr.Dropdown(choices=all_sheet_names, value=all_sheet_names, visible=True)
    else:
        return gr.Dropdown(choices=concat_choices, value=concat_choices), gr.Dropdown(visible=False)

# Following function is only relevant for locally-created executable files based on this app (when using pyinstaller it creates a _internal folder that contains tesseract and poppler. These need to be added to the system path to enable the app to run)
def add_folder_to_path(folder_path: str):
    '''
    Check if a folder exists on your system. If so, get the absolute path and then add it to the system Path variable if it doesn't already exist.
    '''

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(folder_path, "folder exists.")

        # Resolve relative path to absolute path
        absolute_path = os.path.abspath(folder_path)

        current_path = os.environ['PATH']
        if absolute_path not in current_path.split(os.pathsep):
            full_path_extension = absolute_path + os.pathsep + current_path
            os.environ['PATH'] = full_path_extension
            #print(f"Updated PATH with: ", full_path_extension)
        else:
            print(f"Directory {folder_path} already exists in PATH.")
    else:
        print(f"Folder not found at {folder_path} - not added to PATH")

# Upon running a process, the feedback buttons are revealed
def reveal_feedback_buttons():
    return gr.Radio(visible=True, label="Please give some feedback about the results of the redaction. A reminder that the app is only expected to identify about 60% of personally identifiable information in a given (typed) document."), gr.Textbox(visible=True), gr.Button(visible=True), gr.Markdown(visible=True)

def wipe_logs(feedback_logs_loc, usage_logs_loc):
    try:
        os.remove(feedback_logs_loc)
    except Exception as e:
        print("Could not remove feedback logs file", e)
    try:
        os.remove(usage_logs_loc)
    except Exception as e:
        print("Could not remove usage logs file", e)

# Retrieving or setting CUSTOM_HEADER
CUSTOM_HEADER = get_or_create_env_var('CUSTOM_HEADER', '')
print(f'CUSTOM_HEADER found')

# Retrieving or setting CUSTOM_HEADER_VALUE
CUSTOM_HEADER_VALUE = get_or_create_env_var('CUSTOM_HEADER_VALUE', '')
print(f'CUSTOM_HEADER_VALUE found')

async def get_connection_params(request: gr.Request):
    base_folder = ""

    #print("request user:", request.username)

    #request_data = await request.json()  # Parse JSON body
    #print("All request data:", request_data)
    #context_value = request_data.get('context') 
    #if 'context' in request_data:
    #     print("Request context dictionary:", request_data['context'])

    print("Request headers dictionary:", request.headers)
    print("All host elements", request.client)           
    print("IP address:", request.client.host)
    print("Query parameters:", dict(request.query_params))
    # To get the underlying FastAPI items you would need to use await and some fancy @ stuff for a live query: https://fastapi.tiangolo.com/vi/reference/request/
    #print("Request dictionary to object:", request.request.body())
    print("Session hash:", request.session_hash)

    if CUSTOM_HEADER and CUSTOM_HEADER_VALUE:
            if CUSTOM_HEADER in request.headers:
                supplied_custom_header_value = request.headers[CUSTOM_HEADER]
                if supplied_custom_header_value == CUSTOM_HEADER_VALUE:
                    print("Custom header supplied and matches CUSTOM_HEADER_VALUE")
                else:
                    print("Custom header value does not match expected value.")
                    raise ValueError("Custom header value does not match expected value.")
            else:
                print("Custom header value not found.")
                raise ValueError("Custom header value not found.")   

    # Get output save folder from 1 - username passed in from direct Cognito login, 2 - Cognito ID header passed through a Lambda authenticator, 3 - the session hash.

    if request.username:
        out_session_hash = request.username
        base_folder = "user-files/"
        print("Request username found:", out_session_hash)

    elif 'x-cognito-id' in request.headers:
        out_session_hash = request.headers['x-cognito-id']
        base_folder = "user-files/"
        print("Cognito ID found:", out_session_hash)

    elif 'x-amzn-oidc-identity' in request.headers:
        out_session_hash = request.headers['x-amzn-oidc-identity']
        base_folder = "user-files/"
        print("Cognito ID found:", out_session_hash)

    else:
        out_session_hash = request.session_hash
        base_folder = "temp-files/"
        # print("Cognito ID not found. Using session hash as save folder:", out_session_hash)

    output_folder = base_folder + out_session_hash + "/"
    #if bucket_name:
    #    print("S3 output folder is: " + "s3://" + bucket_name + "/" + output_folder)

    return out_session_hash, output_folder, out_session_hash
    

def clean_unicode_text(text):
    # Step 1: Normalize unicode characters to decompose any special forms
    normalized_text = unicodedata.normalize('NFKC', text)

    # Step 2: Replace smart quotes and special punctuation with standard ASCII equivalents
    replacements = {
        '‘': "'", '’': "'", '“': '"', '”': '"', 
        '–': '-', '—': '-', '…': '...', '•': '*',
    }

    # Perform replacements
    for old_char, new_char in replacements.items():
        normalized_text = normalized_text.replace(old_char, new_char)

    # Step 3: Optionally remove non-ASCII characters if needed
    # This regex removes any remaining non-ASCII characters, if desired.
    # Comment this line if you want to keep all Unicode characters.
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', normalized_text)

    return cleaned_text