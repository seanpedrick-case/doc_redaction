import boto3
from typing import List
import io
import os
import json
import pikepdf
import time
import pandas as pd
from tools.custom_image_analyser_engine import OCRResult, CustomImageRecognizerResult
from tools.config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION

def extract_textract_metadata(response:object):
    """Extracts metadata from an AWS Textract response."""

    request_id = response['ResponseMetadata']['RequestId']
    pages = response['DocumentMetadata']['Pages']

    return str({
        'RequestId': request_id,
        'Pages': pages
    })

def analyse_page_with_textract(pdf_page_bytes:object, page_no:int, client:str="", handwrite_signature_checkbox:List[str]=["Extract handwriting", "Redact all identified signatures"]):
    '''
    Analyse page with AWS Textract
    '''
    if client == "":
        try:               
            if AWS_ACCESS_KEY and AWS_SECRET_KEY:
                client = boto3.client('textract', 
                aws_access_key_id=AWS_ACCESS_KEY, 
                aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)
            else:
                client = boto3.client('textract', region_name=AWS_REGION)
        except:
            out_message = "Cannot connect to AWS Textract"
            print(out_message)
            raise Exception(out_message)
            return [], ""  # Return an empty list and an empty string
    
    # Redact signatures if specified
    if "Redact all identified signatures" in handwrite_signature_checkbox:
        #print("Analysing document with signature detection")
        try:
            response = client.analyze_document(Document={'Bytes': pdf_page_bytes}, FeatureTypes=["SIGNATURES"])
        except Exception as e:
            print("Textract call failed due to:", e, "trying again in 3 seconds.")
            time.sleep(3)
            response = client.analyze_document(Document={'Bytes': pdf_page_bytes}, FeatureTypes=["SIGNATURES"])
    else:
        # Call detect_document_text to extract plain text
        try:
            response = client.detect_document_text(Document={'Bytes': pdf_page_bytes})
        except Exception as e:
            print("Textract call failed due to:", e, "trying again in 5 seconds.")
            time.sleep(5)
            response = client.detect_document_text(Document={'Bytes': pdf_page_bytes})

     # Add the 'Page' attribute to each block
    if "Blocks" in response:
        for block in response["Blocks"]:
            block["Page"] = page_no  # Inject the page number into each block

    # Wrap the response with the page number in the desired format
    wrapped_response = {
        'page_no': page_no,
        'data': response
    }

    request_metadata = extract_textract_metadata(response)  # Metadata comes out as a string

    # Return a list containing the wrapped response and the metadata
    return wrapped_response, request_metadata  # Return as a list to match the desired structure

def convert_pike_pdf_page_to_bytes(pdf:object, page_num:int):
    # Create a new empty PDF
    new_pdf = pikepdf.Pdf.new()

    # Specify the page number you want to extract (0-based index)
    page_num = 0  # Example: first page

    # Extract the specific page and add it to the new PDF
    new_pdf.pages.append(pdf.pages[page_num])

    # Save the new PDF to a bytes buffer
    buffer = io.BytesIO()
    new_pdf.save(buffer)

    # Get the PDF bytes
    pdf_bytes = buffer.getvalue()

    # Now you can use the `pdf_bytes` to convert it to an image or further process
    buffer.close()

    return pdf_bytes

def json_to_ocrresult(json_data:dict, page_width:float, page_height:float, page_no:int):
    '''
    Convert the json response from textract to the OCRResult format used elsewhere in the code. Looks for lines, words, and signatures. Handwriting and signatures are set aside especially for later in case the user wants to override the default behaviour and redact all handwriting/signatures.
    '''
    all_ocr_results = []
    signature_or_handwriting_recogniser_results = []
    signature_recogniser_results = []
    handwriting_recogniser_results = []
    signatures = []
    handwriting = []
    ocr_results_with_words = {}
    text_block={}

    text_line_number = 1

    # Assuming json_data is structured as a dictionary with a "pages" key

    # Find the specific page data
    page_json_data = json_data #next((page for page in json_data["pages"] if page["page_no"] == page_no), None)

    if "Blocks" in page_json_data:
        # Access the data for the specific page
        text_blocks = page_json_data["Blocks"]  # Access the Blocks within the page data
    # This is a new page
    elif "page_no" in page_json_data:
        text_blocks = page_json_data["data"]["Blocks"]
    else: text_blocks = []

    is_signature = False
    is_handwriting = False

    for text_block in text_blocks:    
    
        if (text_block['BlockType'] == 'LINE') | (text_block['BlockType'] == 'SIGNATURE'): # (text_block['BlockType'] == 'WORD') |

            # Extract text and bounding box for the line
            line_bbox = text_block["Geometry"]["BoundingBox"]
            line_left = int(line_bbox["Left"] * page_width)
            line_top = int(line_bbox["Top"] * page_height)
            line_right = int((line_bbox["Left"] + line_bbox["Width"]) * page_width)
            line_bottom = int((line_bbox["Top"] + line_bbox["Height"]) * page_height)

            width_abs = int(line_bbox["Width"] * page_width)
            height_abs = int(line_bbox["Height"] * page_height)

            if text_block['BlockType'] == 'LINE':
                
                # Extract text and bounding box for the line
                line_text = text_block.get('Text', '')
                words = []
                current_line_handwriting_results = []  # Track handwriting results for this line

                if 'Relationships' in text_block:
                    for relationship in text_block['Relationships']:
                        if relationship['Type'] == 'CHILD':
                            for child_id in relationship['Ids']:
                                child_block = next((block for block in text_blocks if block['Id'] == child_id), None)
                                if child_block and child_block['BlockType'] == 'WORD':
                                    word_text = child_block.get('Text', '')
                                    word_bbox = child_block["Geometry"]["BoundingBox"]
                                    confidence = child_block.get('Confidence','')
                                    word_left = int(word_bbox["Left"] * page_width)
                                    word_top = int(word_bbox["Top"] * page_height)
                                    word_right = int((word_bbox["Left"] + word_bbox["Width"]) * page_width)
                                    word_bottom = int((word_bbox["Top"] + word_bbox["Height"]) * page_height)

                                    # Extract BoundingBox details
                                    word_width = word_bbox["Width"]
                                    word_height = word_bbox["Height"]

                                    # Convert proportional coordinates to absolute coordinates
                                    word_width_abs = int(word_width * page_width)
                                    word_height_abs = int(word_height * page_height)
                                    
                                    words.append({
                                        'text': word_text,
                                        'bounding_box': (word_left, word_top, word_right, word_bottom)
                                    })
                                    # Check for handwriting
                                    text_type = child_block.get("TextType", '')

                                    if text_type == "HANDWRITING":
                                        is_handwriting = True
                                        entity_name = "HANDWRITING"
                                        word_end = len(word_text)

                                        recogniser_result = CustomImageRecognizerResult(
                                            entity_type=entity_name,
                                            text=word_text,
                                            score=confidence,
                                            start=0,
                                            end=word_end,
                                            left=word_left,
                                            top=word_top,
                                            width=word_width_abs,
                                            height=word_height_abs
                                        )

                                        # Add to handwriting collections immediately
                                        handwriting.append(recogniser_result)
                                        handwriting_recogniser_results.append(recogniser_result)
                                        signature_or_handwriting_recogniser_results.append(recogniser_result)
                                        current_line_handwriting_results.append(recogniser_result)

            # If handwriting or signature, add to bounding box               

            elif (text_block['BlockType'] == 'SIGNATURE'):
                line_text = "SIGNATURE"
                is_signature = True
                entity_name = "SIGNATURE"
                confidence = text_block.get('Confidence', 0)
                word_end = len(line_text)

                recogniser_result = CustomImageRecognizerResult(
                    entity_type=entity_name,
                    text=line_text,
                    score=confidence,
                    start=0,
                    end=word_end,
                    left=line_left,
                    top=line_top,
                    width=width_abs,
                    height=height_abs
                )

                # Add to signature collections immediately
                signatures.append(recogniser_result)
                signature_recogniser_results.append(recogniser_result)
                signature_or_handwriting_recogniser_results.append(recogniser_result)

                words = [{
                    'text': line_text,
                    'bounding_box': (line_left, line_top, line_right, line_bottom)
                }]
        else:
            line_text = ""
            words=[]
            line_left = 0
            line_top = 0
            line_right = 0
            line_bottom = 0
            width_abs = 0
            height_abs = 0

        if line_text:

            ocr_results_with_words["text_line_" + str(text_line_number)] = {
                "line": text_line_number,
                'text': line_text,
                'bounding_box': (line_left, line_top, line_right, line_bottom),
                'words': words,
                'page': page_no
            }

            # Create OCRResult with absolute coordinates
            ocr_result = OCRResult(line_text, line_left, line_top, width_abs, height_abs, conf=confidence, line=text_line_number)
            all_ocr_results.append(ocr_result)

            # Increase line number
            text_line_number += 1

        is_signature_or_handwriting = is_signature | is_handwriting

        # If it is signature or handwriting, will overwrite the default behaviour of the PII analyser
        if is_signature_or_handwriting:
            if recogniser_result not in signature_or_handwriting_recogniser_results: 
                signature_or_handwriting_recogniser_results.append(recogniser_result)

            if is_signature:
                if recogniser_result not in signature_recogniser_results:          
                    signature_recogniser_results.append(recogniser_result)

            if is_handwriting: 
                if recogniser_result not in handwriting_recogniser_results: 
                    handwriting_recogniser_results.append(recogniser_result)

        

    # Add page key to the line level results
    all_ocr_results_with_page = {"page": page_no, "results": all_ocr_results}
    ocr_results_with_words_with_page = {"page": page_no, "results": ocr_results_with_words}

    return all_ocr_results_with_page, signature_or_handwriting_recogniser_results, signature_recogniser_results, handwriting_recogniser_results, ocr_results_with_words_with_page

def load_and_convert_textract_json(textract_json_file_path:str, log_files_output_paths:str, page_sizes_df:pd.DataFrame):
    """
    Loads Textract JSON from a file, detects if conversion is needed, and converts if necessary.
    """
    
    if not os.path.exists(textract_json_file_path):
        print("No existing Textract results file found.")
        return {}, True, log_files_output_paths  # Return empty dict and flag indicating missing file
    
    no_textract_file = False
    print("Found existing Textract json results file.")

    # Track log files
    if textract_json_file_path not in log_files_output_paths:
        log_files_output_paths.append(textract_json_file_path)

    try:
        with open(textract_json_file_path, 'r', encoding='utf-8') as json_file:
            textract_data = json.load(json_file)
    except json.JSONDecodeError:
        print("Error: Failed to parse Textract JSON file. Returning empty data.")
        return {}, True, log_files_output_paths  # Indicate failure

    # Check if conversion is needed
    if "pages" in textract_data:
        print("JSON already in the correct format for app. No changes needed.")
        return textract_data, False, log_files_output_paths  # No conversion required

    if "Blocks" in textract_data:
        print("Need to convert Textract JSON to app format.")
        try:
            
            textract_data = restructure_textract_output(textract_data, page_sizes_df)
            return textract_data, False, log_files_output_paths  # Successfully converted
        
        except Exception as e:
            print("Failed to convert JSON data to app format due to:", e)
            return {}, True, log_files_output_paths  # Conversion failed
    else:
        print("Invalid Textract JSON format: 'Blocks' missing.")
        #print("textract data:", textract_data)
        return {}, True, log_files_output_paths  # Return empty data if JSON is not recognized

def restructure_textract_output(textract_output: dict, page_sizes_df:pd.DataFrame):
    """
    Reorganise Textract output from the bulk Textract analysis option on AWS 
    into a format that works in this redaction app, reducing size.
    """
    pages_dict = {}

    # Extract total pages from DocumentMetadata
    document_metadata = textract_output.get("DocumentMetadata", {})

    # For efficient lookup, set 'page' as index if it's not already
    if 'page' in page_sizes_df.columns:
        page_sizes_df = page_sizes_df.set_index('page')

    for block in textract_output.get("Blocks", []):
        page_no = block.get("Page", 1)  # Default to 1 if missing

        # --- Geometry Conversion Logic ---
        try:
            page_info = page_sizes_df.loc[page_no]
            cb_width = page_info['cropbox_width']
            cb_height = page_info['cropbox_height']
            mb_width = page_info['mediabox_width']
            mb_height = page_info['mediabox_height']
            cb_x_offset = page_info['cropbox_x_offset']
            cb_y_offset_top = page_info['cropbox_y_offset_from_top']

            # Check if conversion is needed (and avoid division by zero)
            needs_conversion = (
                abs(cb_width - mb_width) > 1e-6 or \
                abs(cb_height - mb_height) > 1e-6
            ) and mb_width > 1e-6 and mb_height > 1e-6 # Avoid division by zero

            if needs_conversion and 'Geometry' in block:
                geometry = block['Geometry'] # Work directly on the block's geometry

                # --- Convert BoundingBox ---
                if 'BoundingBox' in geometry:
                    bbox = geometry['BoundingBox']
                    old_left = bbox['Left']
                    old_top = bbox['Top']
                    old_width = bbox['Width']
                    old_height = bbox['Height']

                    # Calculate absolute coordinates within CropBox
                    abs_cb_x = old_left * cb_width
                    abs_cb_y = old_top * cb_height
                    abs_cb_width = old_width * cb_width
                    abs_cb_height = old_height * cb_height

                    # Calculate absolute coordinates relative to MediaBox top-left
                    abs_mb_x = cb_x_offset + abs_cb_x
                    abs_mb_y = cb_y_offset_top + abs_cb_y

                    # Convert back to normalized coordinates relative to MediaBox
                    bbox['Left'] = abs_mb_x / mb_width
                    bbox['Top'] = abs_mb_y / mb_height
                    bbox['Width'] = abs_cb_width / mb_width
                    bbox['Height'] = abs_cb_height / mb_height
        except KeyError:
            print(f"Warning: Page number {page_no} not found in page_sizes_df. Skipping coordinate conversion for this block.")
            # Decide how to handle missing page info: skip conversion, raise error, etc.
        except ZeroDivisionError:
             print(f"Warning: MediaBox width or height is zero for page {page_no}. Skipping coordinate conversion for this block.")

        # Initialise page structure if not already present
        if page_no not in pages_dict:
            pages_dict[page_no] = {"page_no": str(page_no), "data": {"Blocks": []}}

        # Keep only essential fields to reduce size
        filtered_block = {
            key: block[key] for key in ["BlockType", "Confidence", "Text", "Geometry", "Page", "Id", "Relationships"]
            if key in block
        }
        
        pages_dict[page_no]["data"]["Blocks"].append(filtered_block)

    # Convert pages dictionary to a sorted list
    structured_output = {
        "DocumentMetadata": document_metadata,  # Store metadata separately
        "pages": [pages_dict[page] for page in sorted(pages_dict.keys())]
    }

    return structured_output
