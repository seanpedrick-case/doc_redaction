import time
import re
import json
import io
import os
import boto3

from tqdm import tqdm
from PIL import Image, ImageChops, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True

from typing import List, Dict, Tuple
import pandas as pd

#from presidio_image_redactor.entities import ImageRecognizerResult
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine, LTTextLineHorizontal, LTAnno
from pikepdf import Pdf, Dictionary, Name
import pymupdf
from pymupdf import Rect
from fitz import Document, Page

import gradio as gr
from gradio import Progress
from collections import defaultdict  # For efficient grouping

from presidio_analyzer import RecognizerResult

from tools.custom_image_analyser_engine import CustomImageAnalyzerEngine, OCRResult, combine_ocr_results, CustomImageRecognizerResult
from tools.file_conversion import process_file
from tools.load_spacy_model_custom_recognisers import nlp_analyser, score_threshold
from tools.helper_functions import get_file_path_end, output_folder, clean_unicode_text, get_or_create_env_var
from tools.file_conversion import process_file, is_pdf, is_pdf_or_image
# from tools.data_anonymise import generate_decision_process_output
from tools.aws_textract import analyse_page_with_textract, json_to_ocrresult
from tools.aws_functions import comprehend_client
from tools.presidio_analyzer_custom import recognizer_result_from_dict

# Number of pages to loop through before breaking. Currently set very high, as functions are breaking on time metrics (e.g. every 105 seconds), rather than on number of pages redacted.

page_break_value = get_or_create_env_var('page_break_value', '500')
print(f'The value of page_break_value is {page_break_value}')

max_time_value = get_or_create_env_var('max_time_value', '105')
print(f'The value of max_time_value is {max_time_value}')


def sum_numbers_before_seconds(string:str):
    """Extracts numbers that precede the word 'seconds' from a string and adds them up.

    Args:
        string: The input string.

    Returns:
        The sum of all numbers before 'seconds' in the string.
    """

    # Extract numbers before 'seconds' using regular expression
    numbers = re.findall(r'(\d+\.\d+)?\s*seconds', string)

    # Extract the numbers from the matches
    numbers = [float(num.split()[0]) for num in numbers]

    # Sum up the extracted numbers
    sum_of_numbers = round(sum(numbers),1)

    return sum_of_numbers

def choose_and_run_redactor(file_paths:List[str],
 prepared_pdf_file_paths:List[str],
 prepared_pdf_image_paths:List[str],
 language:str,
 chosen_redact_entities:List[str],
 chosen_redact_comprehend_entities:List[str],
 in_redact_method:str,
 in_allow_list:List[List[str]]=None,
 latest_file_completed:int=0,
 out_message:list=[],
 out_file_paths:list=[],
 log_files_output_paths:list=[],
 first_loop_state:bool=False,
 page_min:int=0,
 page_max:int=999,
 estimated_time_taken_state:float=0.0,
 handwrite_signature_checkbox:List[str]=["Redact all identified handwriting", "Redact all identified signatures"],
 all_request_metadata_str:str = "",
 annotations_all_pages:dict={},
 all_line_level_ocr_results_df=[],
 all_decision_process_table=[],
 pymupdf_doc=[],
 current_loop_page:int=0,
 page_break_return:bool=False,
 pii_identification_method:str="Local",
 progress=gr.Progress(track_tqdm=True)):
    '''
    This function orchestrates the redaction process based on the specified method and parameters. It takes the following inputs:

    - file_paths (List[str]): A list of paths to the files to be redacted.
    - prepared_pdf_file_paths (List[str]): A list of paths to the PDF files prepared for redaction.
    - prepared_pdf_image_paths (List[str]): A list of paths to the PDF files converted to images for redaction.
    - language (str): The language of the text in the files.
    - chosen_redact_entities (List[str]): A list of entity types to redact from the files using the local model (spacy) with Microsoft Presidio.
    - chosen_redact_comprehend_entities (List[str]): A list of entity types to redact from files, chosen from the official list from AWS Comprehend service
    - in_redact_method (str): The method to use for redaction.
    - in_allow_list (List[List[str]], optional): A list of allowed terms for redaction. Defaults to None.
    - latest_file_completed (int, optional): The index of the last completed file. Defaults to 0.
    - out_message (list, optional): A list to store output messages. Defaults to an empty list.
    - out_file_paths (list, optional): A list to store paths to the output files. Defaults to an empty list.
    - log_files_output_paths (list, optional): A list to store paths to the log files. Defaults to an empty list.
    - first_loop_state (bool, optional): A flag indicating if this is the first iteration. Defaults to False.
    - page_min (int, optional): The minimum page number to start redaction from. Defaults to 0.
    - page_max (int, optional): The maximum page number to end redaction at. Defaults to 999.
    - estimated_time_taken_state (float, optional): The estimated time taken for the redaction process. Defaults to 0.0.
    - handwrite_signature_checkbox (List[str], optional): A list of options for redacting handwriting and signatures. Defaults to ["Redact all identified handwriting", "Redact all identified signatures"].
    - all_request_metadata_str (str, optional): A string containing all request metadata. Defaults to an empty string.
    - annotations_all_pages (dict, optional): A dictionary containing all image annotations. Defaults to an empty dictionary.
    - all_line_level_ocr_results_df (optional): A DataFrame containing all line-level OCR results. Defaults to an empty DataFrame.
    - all_decision_process_table (optional): A DataFrame containing all decision process tables. Defaults to an empty DataFrame.
    - pymupdf_doc (optional): A list containing the PDF document object. Defaults to an empty list.
    - current_loop_page (int, optional): The current page being processed in the loop. Defaults to 0.
    - page_break_return (bool, optional): A flag indicating if the function should return after a page break. Defaults to False.
    - pii_identification_method (str, optional): The method to redact personal information. Either 'Local' (spacy model), or 'AWS Comprehend' (AWS Comprehend API).
    - progress (gr.Progress, optional): A progress tracker for the redaction process. Defaults to a Progress object with track_tqdm set to True.

    The function returns a redacted document along with processing logs.
    '''
    combined_out_message = ""
    tic = time.perf_counter()
    all_request_metadata = all_request_metadata_str.split('\n') if all_request_metadata_str else []


    # If this is the first time around, set variables to 0/blank
    if first_loop_state==True:
        print("First_loop_state is True")
        latest_file_completed = 0
        current_loop_page = 0
        out_file_paths = []
        estimate_total_processing_time = 0
        estimated_time_taken_state = 0

    # If not the first time around, and the current page loop has been set to a huge number (been through all pages), reset current page to 0
    elif (first_loop_state == False) & (current_loop_page == 999):
        current_loop_page = 0


    if not out_file_paths:
        out_file_paths = []

    latest_file_completed = int(latest_file_completed)

    number_of_pages = len(prepared_pdf_image_paths)

    if isinstance(file_paths,str):
        number_of_files = 1
    else:
        number_of_files = len(file_paths)

    # If we have already redacted the last file, return the input out_message and file list to the relevant components
    if latest_file_completed >= number_of_files:

        print("latest_file_completed is equal to or greater than the number of files")
        # Set to a very high number so as not to mix up with subsequent file processing by the user
        latest_file_completed = 99
        current_loop_page = 0

        if isinstance(out_message, list):
            combined_out_message = '\n'.join(out_message)
        else:
            combined_out_message = out_message
        
        estimate_total_processing_time = sum_numbers_before_seconds(combined_out_message)
        print("Estimated total processing time:", str(estimate_total_processing_time))

        return combined_out_message, out_file_paths, out_file_paths, gr.Number(value=latest_file_completed, label="Number of documents redacted", interactive=False, visible=False), log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages, gr.Number(value=current_loop_page,precision=0, interactive=False, label = "Last redacted page in document", visible=False), gr.Checkbox(value = True, label="Page break reached", visible=False), all_line_level_ocr_results_df, all_decision_process_table
    
    # If we have reached the last page, return message
    if current_loop_page >= number_of_pages:
        print("current_loop_page:", current_loop_page, "is equal to or greater than number of pages in document:", number_of_pages)

        # Set to a very high number so as not to mix up with subsequent file processing by the user
        current_loop_page = 999
        combined_out_message = out_message

        return combined_out_message, out_file_paths, out_file_paths, gr.Number(value=latest_file_completed, label="Number of documents redacted", interactive=False, visible=False), log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages, gr.Number(value=current_loop_page,precision=0, interactive=False, label = "Last redacted page in document", visible=False), gr.Checkbox(value = False, label="Page break reached", visible=False), all_line_level_ocr_results_df, all_decision_process_table

    # Create allow list
    if not in_allow_list.empty:
        in_allow_list_flat = in_allow_list[0].tolist()
        print("In allow list:", in_allow_list_flat)
    else:
        in_allow_list_flat = []

    progress(0.5, desc="Redacting file")

    
    if isinstance(file_paths, str):
        file_paths_list = [file_paths]
        file_paths_loop = file_paths_list
    else:
        file_paths_list = file_paths
        file_paths_loop = [file_paths_list[int(latest_file_completed)]]    


    for file in file_paths_loop:
        if isinstance(file, str):
            file_path = file
        else:
            file_path = file.name    

        if file_path:
            file_path_without_ext = get_file_path_end(file_path)
            print("Redacting file:", file_path_without_ext)

            is_a_pdf = is_pdf(file_path) == True
            if is_a_pdf == False:
                # If user has not submitted a pdf, assume it's an image
                print("File is not a pdf, assuming that image analysis needs to be used.")
                in_redact_method = "Quick image analysis - typed text"
        else:
            out_message = "No file selected"
            print(out_message)

            return combined_out_message, out_file_paths, out_file_paths, gr.Number(value=latest_file_completed, label="Number of documents redacted", interactive=False, visible=False), log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages, gr.Number(value=current_loop_page,precision=0, interactive=False, label = "Last redacted page in document", visible=False), gr.Checkbox(value = True, label="Page break reached", visible=False), all_line_level_ocr_results_df, all_decision_process_table

        if in_redact_method == "Quick image analysis - typed text" or in_redact_method == "Complex image analysis - docs with handwriting/signatures (AWS Textract)":

            if in_redact_method == "Complex image analysis - docs with handwriting/signatures (AWS Textract)":
                # Try accessing Textract through boto3
                try:
                    boto3.client('textract')
                except:
                    out_message = "Cannot connect to AWS Textract. Please choose another redaction method."
                    print(out_message)
                    return out_message, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages

            #Analyse and redact image-based pdf or image
            if is_pdf_or_image(file_path) == False:
                out_message = "Please upload a PDF file or image file (JPG, PNG) for image analysis."
                return out_message, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages

            print("Redacting file " + file_path_without_ext + " as an image-based file")

            pymupdf_doc,all_decision_process_table,logging_file_paths,new_request_metadata, annotations_all_pages, current_loop_page, page_break_return, all_line_level_ocr_results_df = redact_image_pdf(file_path,
             prepared_pdf_image_paths,
             language,
             chosen_redact_entities,
             chosen_redact_comprehend_entities,
             in_allow_list_flat,
             is_a_pdf,
             page_min,
             page_max,
             in_redact_method,
             handwrite_signature_checkbox,
             "",
             current_loop_page,
             page_break_return,
             prepared_pdf_image_paths,
             annotations_all_pages,
             all_line_level_ocr_results_df,
             all_decision_process_table,
             pymupdf_doc,
             pii_identification_method)

            # Save Textract request metadata (if exists)
            if new_request_metadata:
                print("Request metadata:", new_request_metadata)
                all_request_metadata.append(new_request_metadata)              

        elif in_redact_method == "Simple text analysis - PDFs with selectable text":

            logging_file_paths = ""
            
            if is_pdf(file_path) == False:
                out_message = "Please upload a PDF file for text analysis. If you have an image, select 'Image analysis'."
                return out_message, out_file_paths, out_file_paths, gr.Number(value=latest_file_completed, label="Number of documents redacted", interactive=False, visible=False), log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages, gr.Number(value=current_loop_page,precision=0, interactive=False, label = "Last redacted page in document", visible=False), gr.Checkbox(value = True, label="Page break reached", visible=False), all_line_level_ocr_results_df, all_decision_process_table
            
            # Analyse text-based pdf
            print('Redacting file as text-based PDF')
            
            pymupdf_doc, all_decision_process_table, all_line_level_ocr_results_df, annotations_all_pages, current_loop_page, page_break_return = redact_text_pdf(file_path,
            prepared_pdf_image_paths,language,
            chosen_redact_entities,
            chosen_redact_comprehend_entities,
            in_allow_list_flat,
            page_min,
            page_max,
            "Simple text analysis - PDFs with selectable text",
            current_loop_page,
            page_break_return,
            annotations_all_pages,
            all_line_level_ocr_results_df,
            all_decision_process_table,
            pymupdf_doc,
            pii_identification_method)

        else:
            out_message = "No redaction method selected"
            print(out_message)
            return out_message, out_file_paths, out_file_paths, gr.Number(value=latest_file_completed, label="Number of documents redacted", interactive=False, visible=False), log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages, gr.Number(value=current_loop_page,precision=0, interactive=False, label = "Last redacted page in document", visible=False), gr.Checkbox(value = True, label="Page break reached", visible=False), all_line_level_ocr_results_df, all_decision_process_table
        
        # If at last page, save to file
        if current_loop_page >= number_of_pages:

            print("Current page loop:", current_loop_page, "is greater or equal to number of pages:", number_of_pages)
            latest_file_completed += 1
            current_loop_page = 999

            if latest_file_completed != len(file_paths):
                print("Completed file number:", str(latest_file_completed), "there are more files to do")                    

            # Save file
            if is_pdf(file_path) == False:
                out_image_file_path = output_folder + file_path_without_ext + "_redacted_as_img.pdf"
                pymupdf_doc[0].save(out_image_file_path, "PDF" ,resolution=100.0, save_all=True, append_images=pymupdf_doc[1:])
            
            else:
                out_image_file_path = output_folder + file_path_without_ext + "_redacted.pdf"
                pymupdf_doc.save(out_image_file_path)

            out_file_paths.append(out_image_file_path)

            if logging_file_paths:
                log_files_output_paths.extend(logging_file_paths)

            logs_output_file_name = out_image_file_path + "_decision_process_output.csv"
            all_decision_process_table.to_csv(logs_output_file_name, index = None, encoding="utf-8")
            #log_files_output_paths.append(logs_output_file_name)
            out_file_paths.append(logs_output_file_name)

            all_text_output_file_name = out_image_file_path + "_ocr_output.csv"
            all_line_level_ocr_results_df.to_csv(all_text_output_file_name, index = None, encoding="utf-8")
            #log_files_output_paths.append(all_text_output_file_name)
            out_file_paths.append(all_text_output_file_name)

            # Make a combined message for the file                
            if isinstance(out_message, list):
                combined_out_message = '\n'.join(out_message)  # Ensure out_message is a list of strings
            else: combined_out_message = out_message

            toc = time.perf_counter()
            time_taken = toc - tic
            estimated_time_taken_state = estimated_time_taken_state + time_taken

            out_time_message = f" Redacted in {estimated_time_taken_state:0.1f} seconds."
            combined_out_message = combined_out_message + " " + out_time_message  # Ensure this is a single string

            estimate_total_processing_time = sum_numbers_before_seconds(combined_out_message)
            print("Estimated total processing time:", str(estimate_total_processing_time))

            #out_time_message = f" Redacted in {estimated_time_taken_state:0.1f} seconds."
            #combined_out_message = combined_out_message + " " + out_time_message  # Ensure this is a single string
        
            # Increase latest file completed count unless we are at the last file
            # if latest_file_completed != len(file_paths):
            #     print("Completed file number:", str(latest_file_completed), "more files to do") 

            # if current_loop_page >= number_of_pages:

            #     print("Current page loop", current_loop_page, "is greater than or equal to number of pages:", number_of_pages)
            #     latest_file_completed += 1

            #     # Set to 999 to be a big number not to interrupt processing of large files by user
            #     current_loop_page = 999

            #     out_text_file_path = output_folder + file_path_without_ext + "_text_redacted.pdf"
            #     pymupdf_doc.save(out_text_file_path)
            #     out_file_paths.append(out_text_file_path)   
            
            #     # Write logs to file
            #     decision_logs_output_file_name = out_text_file_path + "_decision_process_output.csv"
            #     all_decision_process_table.to_csv(decision_logs_output_file_name)
            #     log_files_output_paths.append(decision_logs_output_file_name)

            #     all_text_output_file_name = out_text_file_path + "_all_text_output.csv"
            #     all_line_level_ocr_results_df.to_csv(all_text_output_file_name)
            #     log_files_output_paths.append(all_text_output_file_name)

            #     out_message_new = "File '" + file_path_without_ext + "' successfully redacted"

            #     if isinstance(out_message, list):
            #         out_message.append(out_message_new)  # Ensure out_message is a list of strings

            if latest_file_completed != len(file_paths):
                print("Completed file number:", str(latest_file_completed), " there are more files to do")
                

            # Make a combined message for the file
            if isinstance(out_message, list):
                combined_out_message = '\n'.join(out_message)  # Ensure out_message is a list of strings
            else: combined_out_message = out_message


   # If textract requests made, write to logging file
    if all_request_metadata:
        all_request_metadata_str = '\n'.join(all_request_metadata)

        all_request_metadata_file_path = output_folder + file_path_without_ext + "_textract_request_metadata.txt"   

        with open(all_request_metadata_file_path, "w") as f:
            f.write(all_request_metadata_str)

        # Add the request metadata to the log outputs if not there already
        if all_request_metadata_file_path not in log_files_output_paths:
            log_files_output_paths.append(all_request_metadata_file_path)

    if combined_out_message: out_message = combined_out_message
    
    print("\nout_message at choose_and_run_redactor end is:", out_message)

    # Ensure no duplicated output files
    log_files_output_paths = list(set(log_files_output_paths))
    out_file_paths = list(set(out_file_paths))


    return out_message, out_file_paths, out_file_paths, gr.Number(value=latest_file_completed, label="Number of documents redacted", interactive=False, visible=False), log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages, gr.Number(value=current_loop_page, precision=0, interactive=False, label = "Last redacted page in document", visible=False), gr.Checkbox(value = True, label="Page break reached", visible=False), all_line_level_ocr_results_df, all_decision_process_table

def convert_pikepdf_coords_to_pymudf(pymupdf_page, annot):
    '''
    Convert annotations from pikepdf to pymupdf format
    '''

    mediabox_height = pymupdf_page.mediabox[3] - pymupdf_page.mediabox[1]
    mediabox_width = pymupdf_page.mediabox[2] - pymupdf_page.mediabox[0]
    rect_height = pymupdf_page.rect.height
    rect_width = pymupdf_page.rect.width  

    # Adjust coordinates based on scaling factors
    page_x_adjust = (rect_width - mediabox_width) / 2  # Center adjustment
    page_y_adjust = (rect_height - mediabox_height) / 2  # Center adjustment

    #print("In the pikepdf conversion function")
    # Extract the /Rect field
    rect_field = annot["/Rect"]

    # Convert the extracted /Rect field to a list of floats (since pikepdf uses Decimal objects)
    rect_coordinates = [float(coord) for coord in rect_field]

    # Convert the Y-coordinates (flip using the page height)
    x1, y1, x2, y2 = rect_coordinates
    x1 = x1 + page_x_adjust
    new_y1 = (rect_height - y2) - page_y_adjust
    x2 = x2 + page_x_adjust
    new_y2 = (rect_height - y1) - page_y_adjust

    return x1, new_y1, x2, new_y2

def convert_pikepdf_to_image_coords(pymupdf_page, annot, image:Image):
    '''
    Convert annotations from pikepdf coordinates to image coordinates.
    '''

    # Get the dimensions of the page in points with pymupdf
    rect_height = pymupdf_page.rect.height
    rect_width = pymupdf_page.rect.width 

    # Get the dimensions of the image
    image_page_width, image_page_height = image.size

    # Calculate scaling factors between pymupdf and PIL image
    scale_width = image_page_width / rect_width
    scale_height = image_page_height / rect_height

    # Extract the /Rect field
    rect_field = annot["/Rect"]

    # Convert the extracted /Rect field to a list of floats
    rect_coordinates = [float(coord) for coord in rect_field]

    # Convert the Y-coordinates (flip using the image height)
    x1, y1, x2, y2 = rect_coordinates
    x1_image = x1 * scale_width
    new_y1_image = image_page_height - (y2 * scale_height)  # Flip Y0 (since it starts from bottom)
    x2_image = x2 * scale_width
    new_y2_image = image_page_height - (y1 * scale_height)  # Flip Y1

    return x1_image, new_y1_image, x2_image, new_y2_image

def convert_image_coords_to_pymupdf(pymupdf_page, annot:CustomImageRecognizerResult, image:Image):
    '''
    Converts an image with redaction coordinates from a CustomImageRecognizerResult to pymupdf coordinates.
    '''

    rect_height = pymupdf_page.rect.height
    rect_width = pymupdf_page.rect.width 

    image_page_width, image_page_height = image.size

    # Calculate scaling factors between PIL image and pymupdf
    scale_width = rect_width / image_page_width
    scale_height = rect_height / image_page_height

    # Calculate scaled coordinates
    x1 = (annot.left * scale_width)# + page_x_adjust
    new_y1 = (annot.top * scale_height)# - page_y_adjust  # Flip Y0 (since it starts from bottom)
    x2 = ((annot.left + annot.width) * scale_width)# + page_x_adjust  # Calculate x1
    new_y2 = ((annot.top + annot.height) * scale_height)# - page_y_adjust  # Calculate y1 correctly

    return x1, new_y1, x2, new_y2

def convert_gradio_annotation_coords_to_pymupdf(pymupdf_page:Page, annot:dict, image:Image):
    '''
    Converts an image with redaction coordinates from a gradio annotation component to pymupdf coordinates.
    '''

    rect_height = pymupdf_page.rect.height
    rect_width = pymupdf_page.rect.width 

    image_page_width, image_page_height = image.size

    # Calculate scaling factors between PIL image and pymupdf
    scale_width = rect_width / image_page_width
    scale_height = rect_height / image_page_height

    # Calculate scaled coordinates
    x1 = (annot["xmin"] * scale_width)# + page_x_adjust
    new_y1 = (annot["ymin"] * scale_height)# - page_y_adjust  # Flip Y0 (since it starts from bottom)
    x2 = ((annot["xmax"]) * scale_width)# + page_x_adjust  # Calculate x1
    new_y2 = ((annot["ymax"]) * scale_height)# - page_y_adjust  # Calculate y1 correctly

    return x1, new_y1, x2, new_y2

def move_page_info(file_path: str) -> str:
    # Split the string at '.png'
    base, extension = file_path.rsplit('.pdf', 1)
    
    # Extract the page info
    page_info = base.split('page ')[1].split(' of')[0]  # Get the page number
    new_base = base.replace(f'page {page_info} of ', '')  # Remove the page info from the original position
    
    # Construct the new file path
    new_file_path = f"{new_base}_page_{page_info}.png"
    
    return new_file_path

def redact_page_with_pymupdf(page:Page, annotations_on_page, image = None):

    mediabox_height = page.mediabox[3] - page.mediabox[1]
    mediabox_width = page.mediabox[2] - page.mediabox[0]
    rect_height = page.rect.height
    rect_width = page.rect.width    

    out_annotation_boxes = {}
    all_image_annotation_boxes = []
    image_path = ""

    if isinstance(image, Image.Image):
        image_path = move_page_info(str(page))
        image.save(image_path)
    elif isinstance(image, str):
        image_path = image
        image = Image.open(image_path)

    # Check if this is an object used in the Gradio Annotation component
    if isinstance (annotations_on_page, dict):
        annotations_on_page = annotations_on_page["boxes"]

    for annot in annotations_on_page:
        # Check if an Image recogniser result, or a Gradio annotation object
        if (isinstance(annot, CustomImageRecognizerResult)) | isinstance(annot, dict):

            img_annotation_box = {}

            # Should already be in correct format if img_annotator_box is an input
            if isinstance(annot, dict):
                img_annotation_box = annot
                try:
                    img_annotation_box["label"] = annot.entity_type
                except:
                    img_annotation_box["label"] = "Redaction"

                x1, pymupdf_y1, x2, pymupdf_y2 = convert_gradio_annotation_coords_to_pymupdf(page, annot, image)

            # Else should be CustomImageRecognizerResult
            else:
                x1, pymupdf_y1, x2, pymupdf_y2 = convert_image_coords_to_pymupdf(page, annot, image)

                img_annotation_box["xmin"] = annot.left
                img_annotation_box["ymin"] = annot.top 
                img_annotation_box["xmax"] = annot.left + annot.width
                img_annotation_box["ymax"] = annot.top + annot.height
                img_annotation_box["color"] = (0,0,0)
                try:
                    img_annotation_box["label"] = annot.entity_type
                except:
                    img_annotation_box["label"] = "Redaction"

            rect = Rect(x1, pymupdf_y1, x2, pymupdf_y2)  # Create the PyMuPDF Rect

        # Else it should be a pikepdf annotation object
        else:           
            x1, pymupdf_y1, x2, pymupdf_y2 = convert_pikepdf_coords_to_pymudf(page, annot)

            rect = Rect(x1, pymupdf_y1, x2, pymupdf_y2)

            img_annotation_box = {}

            if image:
                image_x1, image_y1, image_x2, image_y2 = convert_pikepdf_to_image_coords(page, annot, image)
                
                img_annotation_box["xmin"] = image_x1
                img_annotation_box["ymin"] = image_y1
                img_annotation_box["xmax"] = image_x2
                img_annotation_box["ymax"] = image_y2
                img_annotation_box["color"] = (0,0,0)

                if isinstance(annot, Dictionary):
                    #print("Trying to get label out of annotation", annot["/T"])
                    img_annotation_box["label"] = str(annot["/T"])
                    #print("Label is:", img_annotation_box["label"])
                else:
                    img_annotation_box["label"] = "REDACTION"

        # Convert to a PyMuPDF Rect object
        #rect = Rect(rect_coordinates)

        all_image_annotation_boxes.append(img_annotation_box)

        # Calculate the middle y value and set height to 1 pixel
        middle_y = (pymupdf_y1 + pymupdf_y2) / 2
        rect_single_pixel_height = Rect(x1, middle_y - 2, x2, middle_y + 2)  # Small height in middle of word to remove text

        # Add the annotation to the middle of the character line, so that it doesn't delete text from adjacent lines
        page.add_redact_annot(rect_single_pixel_height)

        # Set up drawing a black box over the whole rect
        shape = page.new_shape()
        shape.draw_rect(rect)
        shape.finish(color=(0, 0, 0), fill=(0, 0, 0))  # Black fill for the rectangle
        shape.commit()

    out_annotation_boxes = {
        "image": image_path, #Image.open(image_path), #image_path,
        "boxes": all_image_annotation_boxes
    }

    page.apply_redactions(images=0, graphics=0)
    page.clean_contents()

    return page, out_annotation_boxes

def bounding_boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    return (box1[0] < box2[2] and box2[0] < box1[2] and
            box1[1] < box2[3] and box2[1] < box1[3])

def merge_img_bboxes(bboxes, combined_results: Dict, signature_recogniser_results=[], handwriting_recogniser_results=[], handwrite_signature_checkbox: List[str]=["Redact all identified handwriting", "Redact all identified signatures"], horizontal_threshold:int=50, vertical_threshold:int=12):
    merged_bboxes = []
    grouped_bboxes = defaultdict(list)

    # Reconstruct bounding boxes for substrings of interest
    reconstructed_bboxes = []
    for bbox in bboxes:
        #print("bbox:", bbox)
        bbox_box = (bbox.left, bbox.top, bbox.left + bbox.width, bbox.top + bbox.height)
        for line_text, line_info in combined_results.items():
            line_box = line_info['bounding_box']
            if bounding_boxes_overlap(bbox_box, line_box): 
                if bbox.text in line_text:
                    start_char = line_text.index(bbox.text)
                    end_char = start_char + len(bbox.text)
                    
                    relevant_words = []
                    current_char = 0
                    for word in line_info['words']:
                        word_end = current_char + len(word['text'])
                        if current_char <= start_char < word_end or current_char < end_char <= word_end or (start_char <= current_char and word_end <= end_char):
                            relevant_words.append(word)
                        if word_end >= end_char:
                            break
                        current_char = word_end
                        if not word['text'].endswith(' '):
                            current_char += 1  # +1 for space if the word doesn't already end with a space

                    if relevant_words:
                        #print("Relevant words:", relevant_words)
                        left = min(word['bounding_box'][0] for word in relevant_words)
                        top = min(word['bounding_box'][1] for word in relevant_words)
                        right = max(word['bounding_box'][2] for word in relevant_words)
                        bottom = max(word['bounding_box'][3] for word in relevant_words)
                        
                        # Combine the text of all relevant words
                        combined_text = " ".join(word['text'] for word in relevant_words)

                        # Calculate new dimensions for the merged box
                        reconstructed_bbox = CustomImageRecognizerResult(
                            bbox.entity_type,
                            bbox.start,
                            bbox.end,
                            bbox.score,
                            left,
                            top,
                            right - left,  # width
                            bottom - top,  # height
                            combined_text
                        )
                        reconstructed_bboxes.append(reconstructed_bbox)
                        break
        else:
            # If the bbox text is not found in any line in combined_results, keep the original bbox
            reconstructed_bboxes.append(bbox)

    # Group reconstructed bboxes by approximate vertical proximity
    for box in reconstructed_bboxes:
        grouped_bboxes[round(box.top / vertical_threshold)].append(box)

    # Merge within each group
    for _, group in grouped_bboxes.items():
        group.sort(key=lambda box: box.left)

        merged_box = group[0]
        for next_box in group[1:]:
            if next_box.left - (merged_box.left + merged_box.width) <= horizontal_threshold:
                # Calculate new dimensions for the merged box
                if merged_box.text == next_box.text:
                    new_text = merged_box.text
                else:
                    new_text = merged_box.text + " " + next_box.text

                if merged_box.text == next_box.text:
                    new_text = merged_box.text
                    new_entity_type = merged_box.entity_type  # Keep the original entity type
                else:
                    new_text = merged_box.text + " " + next_box.text
                    new_entity_type = merged_box.entity_type + " - " + next_box.entity_type  # Concatenate entity types

                new_left = min(merged_box.left, next_box.left)
                new_top = min(merged_box.top, next_box.top)
                new_width = max(merged_box.left + merged_box.width, next_box.left + next_box.width) - new_left
                new_height = max(merged_box.top + merged_box.height, next_box.top + next_box.height) - new_top
                merged_box = CustomImageRecognizerResult(
                    new_entity_type, merged_box.start, merged_box.end, merged_box.score, new_left, new_top, new_width, new_height, new_text
                )
            else:
                merged_bboxes.append(merged_box)
                merged_box = next_box  

        merged_bboxes.append(merged_box)

        # Process signature and handwriting results
    if signature_recogniser_results or handwriting_recogniser_results:
        if "Redact all identified handwriting" in handwrite_signature_checkbox:
            #print("Handwriting boxes exist at merge:", handwriting_recogniser_results)
            merged_bboxes.extend(handwriting_recogniser_results)

        if "Redact all identified signatures" in handwrite_signature_checkbox:
            #print("Signature boxes exist at merge:", signature_recogniser_results)
            merged_bboxes.extend(signature_recogniser_results)

    #print("bboxes:", bboxes)

    return merged_bboxes

def redact_image_pdf(file_path:str,
                     prepared_pdf_file_paths:List[str],
                     language:str,
                     chosen_redact_entities:List[str],
                     chosen_redact_comprehend_entities:List[str],
                     allow_list:List[str]=None,
                     is_a_pdf:bool=True,
                     page_min:int=0,
                     page_max:int=999,
                     analysis_type:str="Quick image analysis - typed text",
                     handwrite_signature_checkbox:List[str]=["Redact all identified handwriting", "Redact all identified signatures"],
                     request_metadata:str="", current_loop_page:int=0,
                     page_break_return:bool=False,
                     images=[],
                     annotations_all_pages:List=[],
                     all_line_level_ocr_results_df = pd.DataFrame(),
                     all_decision_process_table = pd.DataFrame(),
                     pymupdf_doc = [],
                     pii_identification_method:str="Local",
                     page_break_val:int=int(page_break_value),
                     logging_file_paths:List=[],
                     max_time:int=int(max_time_value),                     
                     progress=Progress(track_tqdm=True)):

    '''
    This function redacts sensitive information from a PDF document. It takes the following parameters:

    - file_path (str): The path to the PDF file to be redacted.
    - prepared_pdf_file_paths (List[str]): A list of paths to the PDF file pages converted to images.
    - language (str): The language of the text in the PDF.
    - chosen_redact_entities (List[str]): A list of entity types to redact from the PDF.
    - chosen_redact_comprehend_entities (List[str]): A list of entity types to redact from the list allowed by the AWS Comprehend service.
    - allow_list (List[str], optional): A list of entity types to allow in the PDF. Defaults to None.
    - is_a_pdf (bool, optional): Indicates if the input file is a PDF. Defaults to True.
    - page_min (int, optional): The minimum page number to start redaction from. Defaults to 0.
    - page_max (int, optional): The maximum page number to end redaction at. Defaults to 999.
    - analysis_type (str, optional): The type of analysis to perform on the PDF. Defaults to "Quick image analysis - typed text".
    - handwrite_signature_checkbox (List[str], optional): A list of options for redacting handwriting and signatures. Defaults to ["Redact all identified handwriting", "Redact all identified signatures"].
    - request_metadata (str, optional): Metadata related to the redaction request. Defaults to an empty string.
    - page_break_return (bool, optional): Indicates if the function should return after a page break. Defaults to False.
    - images (list, optional): List of image objects for each PDF page.
    - annotations_all_pages (List, optional): List of annotations on all pages that is used by the gradio_image_annotation object.
    - all_line_level_ocr_results_df (pd.DataFrame(), optional): All line level OCR results for the document as a Pandas dataframe,
    - all_decision_process_table (pd.DataFrame(), optional): All redaction decisions for document as a Pandas dataframe.
    - pymupdf_doc (List, optional): The document as a PyMupdf object.
    - pii_identification_method (str, optional): The method to redact personal information. Either 'Local' (spacy model), or 'AWS Comprehend' (AWS Comprehend API).
    - page_break_val (int, optional): The value at which to trigger a page break. Defaults to 3.
    - logging_file_paths (List, optional): List of file paths used for saving redaction process logging results.
    - max_time (int, optional): The maximum amount of time (s) that the function should be running before it breaks. To avoid timeout errors with some APIs.    
    - progress (Progress, optional): A progress tracker for the redaction process. Defaults to a Progress object with track_tqdm set to True.

    The function returns a fully or partially-redacted PDF document.
    '''
    file_name = get_file_path_end(file_path)
    fill = (0, 0, 0)   # Fill colour
    image_analyser = CustomImageAnalyzerEngine(nlp_analyser)

    #print("pymupdf_doc at start of redact_image_pdf function:", pymupdf_doc)

    tic = time.perf_counter()

    if not prepared_pdf_file_paths:
        out_message = "PDF does not exist as images. Converting pages to image"
        print(out_message)

        prepared_pdf_file_paths = process_file(file_path)

    number_of_pages = len(prepared_pdf_file_paths)
    print("Number of pages:", str(number_of_pages))

    # Check that page_min and page_max are within expected ranges
    if page_max > number_of_pages or page_max == 0:
        page_max = number_of_pages

    if page_min <= 0: page_min = 0
    else: page_min = page_min - 1

    print("Page range:", str(page_min + 1), "to", str(page_max))
    print("Current_loop_page:", current_loop_page)
    
    if analysis_type == "Quick image analysis - typed text": ocr_results_file_path = output_folder + "ocr_results_" + file_name + "_pages_" + str(page_min + 1) + "_" + str(page_max) + ".csv"
    elif analysis_type == "Complex image analysis - docs with handwriting/signatures (AWS Textract)": ocr_results_file_path = output_folder + "ocr_results_" + file_name + "_pages_" + str(page_min + 1) + "_" + str(page_max) + "_textract.csv"    
    
    if current_loop_page == 0: page_loop_start = 0
    else: page_loop_start = current_loop_page

    #progress_bar = progress.tqdm(range(page_loop_start, number_of_pages), unit="pages", desc="Redacting pages")
    progress_bar = tqdm(range(page_loop_start, number_of_pages), unit="pages remaining", desc="Redacting pages")

    for page_no in progress_bar:

        handwriting_or_signature_boxes = []
        signature_recogniser_results = []
        handwriting_recogniser_results = []
        page_break_return = False

        reported_page_number = str(page_no + 1)
        print("Redacting page:", reported_page_number)
        
        # Assuming prepared_pdf_file_paths[page_no] is a PIL image object
        try:
            image = prepared_pdf_file_paths[page_no]#.copy()
            #print("image:", image)
        except Exception as e:
            print("Could not redact page:", reported_page_number, "due to:")
            print(e)            
            continue

        image_annotations = {"image": image, "boxes": []}        
        pymupdf_page = pymupdf_doc.load_page(page_no)
 
        if page_no >= page_min and page_no < page_max:    

            #print("Image is in range of pages to redact")            
            if isinstance(image, str):
                #print("image is a file path")
                image = Image.open(image)

            # Need image size to convert textract OCR outputs to the correct sizes
            page_width, page_height = image.size

            # Possibility to use different languages
            if language == 'en':
                ocr_lang = 'eng'
            else: ocr_lang = language

            # Step 1: Perform OCR. Either with Tesseract, or with AWS Textract
            if analysis_type == "Quick image analysis - typed text":
                
                word_level_ocr_results = image_analyser.perform_ocr(image)

                # Combine OCR results
                line_level_ocr_results, line_level_ocr_results_with_children = combine_ocr_results(word_level_ocr_results)

    
            # Import results from json and convert
            if analysis_type == "Complex image analysis - docs with handwriting/signatures (AWS Textract)":
                
                # Convert the image to bytes using an in-memory buffer
                image_buffer = io.BytesIO()
                image.save(image_buffer, format='PNG')  # Save as PNG, or adjust format if needed
                pdf_page_as_bytes = image_buffer.getvalue()
                
                #json_file_path = output_folder + file_name + "_page_" + reported_page_number + "_textract.json"
                json_file_path = output_folder + file_name + "_textract.json"
                
                if not os.path.exists(json_file_path):
                    text_blocks, new_request_metadata = analyse_page_with_textract(pdf_page_as_bytes, reported_page_number)  # Analyse page with Textract
                    logging_file_paths.append(json_file_path)
                    request_metadata = request_metadata + "\n" + new_request_metadata

                    wrapped_text_blocks = {"pages":[text_blocks]}

                    # Write the updated existing_data back to the JSON file
                    with open(json_file_path, 'w') as json_file:
                        json.dump(wrapped_text_blocks, json_file, indent=4)  # indent=4 makes the JSON file pretty-printed
                else:
                    # Open the file and load the JSON data
                    print("Found existing Textract json results file.")
                    with open(json_file_path, 'r') as json_file:
                        existing_data = json.load(json_file)

                        # Check if the current reported_page_number exists in the loaded JSON
                        page_exists = any(page['page_no'] == reported_page_number for page in existing_data.get("pages", []))

                        if not page_exists:  # If the page does not exist, analyze again
                            print(f"Page number {reported_page_number} not found in existing data. Analyzing again.")
                            text_blocks, new_request_metadata = analyse_page_with_textract(pdf_page_as_bytes, reported_page_number)  # Analyse page with Textract

                            # Check if "pages" key exists, if not, initialize it as an empty list
                            if "pages" not in existing_data:
                                existing_data["pages"] = []

                            # Append the new page data
                            existing_data["pages"].append(text_blocks)

                            # Write the updated existing_data back to the JSON file
                            with open(json_file_path, 'w') as json_file:
                                json.dump(existing_data, json_file, indent=4)  # indent=4 makes the JSON file pretty-printed

                            logging_file_paths.append(json_file_path)
                            request_metadata = request_metadata + "\n" + new_request_metadata
                        else:
                            # If the page exists, retrieve the data
                            text_blocks = next(page['data'] for page in existing_data["pages"] if page['page_no'] == reported_page_number)

                line_level_ocr_results, handwriting_or_signature_boxes, signature_recogniser_results, handwriting_recogniser_results, line_level_ocr_results_with_children = json_to_ocrresult(text_blocks, page_width, page_height, reported_page_number)

            # Step 2: Analyze text and identify PII
            if chosen_redact_entities:

                pii_identification_method= "AWS Comprehend" #"Local"

                redaction_bboxes = image_analyser.analyze_text(
                    line_level_ocr_results,
                    line_level_ocr_results_with_children,
                    chosen_redact_comprehend_entities = chosen_redact_comprehend_entities,
                    pii_identification_method = pii_identification_method,                    
                    language=language,
                    entities=chosen_redact_entities,
                    allow_list=allow_list,
                    score_threshold=score_threshold
                )                

                # redaction_bboxes = choose_redaction_method_and_analyse_pii(line_level_ocr_results,
                #     line_level_ocr_results_with_children,
                #     language,
                #     chosen_redact_entities,
                #     allow_list,
                #     score_threshold,
                #     pii_identification_method)
                
            else:
                redaction_bboxes = []
                

            if analysis_type == "Quick image analysis - typed text": interim_results_file_path = output_folder + "interim_analyser_bboxes_" + file_name + "_pages_" + str(page_min + 1) + "_" + str(page_max) + ".txt"
            elif analysis_type == "Complex image analysis - docs with handwriting/signatures (AWS Textract)": interim_results_file_path = output_folder + "interim_analyser_bboxes_" + file_name + "_pages_" + str(page_min + 1) + "_" + str(page_max) + "_textract.txt" 

            # Save decision making process
            bboxes_str = str(redaction_bboxes)
            with open(interim_results_file_path, "w") as f:
                f.write(bboxes_str)

            # Merge close bounding boxes
            merged_redaction_bboxes = merge_img_bboxes(redaction_bboxes, line_level_ocr_results_with_children, signature_recogniser_results, handwriting_recogniser_results, handwrite_signature_checkbox)
            
            # 3. Draw the merged boxes
            if is_pdf(file_path) == False:
                draw = ImageDraw.Draw(image)

                all_image_annotations_boxes = []

                for box in merged_redaction_bboxes:
                    print("box:", box)

                    x0 = box.left
                    y0 = box.top
                    x1 = x0 + box.width
                    y1 = y0 + box.height

                    try:
                        label = box.entity_type
                    except:
                        label = "Redaction"

                    # Directly append the dictionary with the required keys
                    all_image_annotations_boxes.append({
                        "xmin": x0,
                        "ymin": y0,
                        "xmax": x1,
                        "ymax": y1,
                        "label": label,
                        "color": (0, 0, 0)
                    })

                    draw.rectangle([x0, y0, x1, y1], fill=fill)  # Adjusted to use a list for rectangle

                image_annotations = {"image": file_path, "boxes": all_image_annotations_boxes}

            ## Apply annotations with pymupdf
            else:
                pymupdf_page, image_annotations = redact_page_with_pymupdf(pymupdf_page, merged_redaction_bboxes, image)

            # Convert decision process to table
            decision_process_table = pd.DataFrame([{
                'page': reported_page_number,
                'entity_type': result.entity_type,
                'start': result.start,
                'end': result.end,
                'score': result.score,
                'left': result.left,
                'top': result.top,
                'width': result.width,
                'height': result.height,
                'text': result.text
            } for result in merged_redaction_bboxes])

            all_decision_process_table = pd.concat([all_decision_process_table, decision_process_table])

            # Convert to DataFrame and add to ongoing logging table
            line_level_ocr_results_df = pd.DataFrame([{
                'page': reported_page_number,
                'text': result.text,
                'left': result.left,
                'top': result.top,
                'width': result.width,
                'height': result.height
            } for result in line_level_ocr_results])

            all_line_level_ocr_results_df = pd.concat([all_line_level_ocr_results_df, line_level_ocr_results_df])

            toc = time.perf_counter()

            time_taken = toc - tic

            #print("toc - tic:", time_taken)

            # Break if time taken is greater than max_time seconds
            if time_taken > max_time:
                print("Processing for", max_time, "seconds, breaking loop.")
                page_break_return = True
                progress.close(_tqdm=progress_bar)
                tqdm._instances.clear()

                if is_pdf(file_path) == False:
                    images.append(image)
                    pymupdf_doc = images

                annotations_all_pages.append(image_annotations)

                current_loop_page += 1

                return pymupdf_doc, all_decision_process_table, logging_file_paths, request_metadata, annotations_all_pages, current_loop_page, page_break_return, all_line_level_ocr_results_df

        if is_pdf(file_path) == False:
            images.append(image)
            pymupdf_doc = images

        annotations_all_pages.append(image_annotations)

        current_loop_page += 1

        # Break if new page is a multiple of chosen page_break_val
        if current_loop_page % page_break_val == 0:
            page_break_return = True
            progress.close(_tqdm=progress_bar)
            tqdm._instances.clear()

            return pymupdf_doc, all_decision_process_table, logging_file_paths, request_metadata, annotations_all_pages, current_loop_page, page_break_return, all_line_level_ocr_results_df

    return pymupdf_doc, all_decision_process_table, logging_file_paths, request_metadata, annotations_all_pages, current_loop_page, page_break_return, all_line_level_ocr_results_df


###
# PIKEPDF TEXT PDF REDACTION
###

def get_text_container_characters(text_container:LTTextContainer):

    if isinstance(text_container, LTTextContainer):
        characters = [char
                    for line in text_container
                    if isinstance(line, LTTextLine) or isinstance(line, LTTextLineHorizontal)
                    for char in line]
    
        return characters
    return []

def create_text_bounding_boxes_from_characters(char_objects:List[LTChar]) -> Tuple[List[OCRResult], List[LTChar]]:
    '''
    Create an OCRResult object based on a list of pdfminer LTChar objects.
    '''

    line_level_results_out = []
    line_level_characters_out = []
    #all_line_level_characters_out = []
    character_objects_out = []  # New list to store character objects

    # Initialize variables
    full_text = ""
    added_text = ""
    overall_bbox = [float('inf'), float('inf'), float('-inf'), float('-inf')]  # [x0, y0, x1, y1]
    word_bboxes = []

    # Iterate through the character objects
    current_word = ""
    current_word_bbox = [float('inf'), float('inf'), float('-inf'), float('-inf')]  # [x0, y0, x1, y1]

    for char in char_objects:
        character_objects_out.append(char)  # Collect character objects

        if isinstance(char, LTAnno):
            # Handle space separately by finalizing the word
            full_text += char.get_text()  # Adds space or newline

            if current_word:  # Only finalize if there is a current word
                word_bboxes.append((current_word, current_word_bbox))
                current_word = ""
                current_word_bbox = [float('inf'), float('inf'), float('-inf'), float('-inf')]  # Reset for next word

            # Check for line break (assuming a new line is indicated by a specific character)
            if '\n' in char.get_text():
                #print("char_anno:", char)
                # Finalize the current line
                if current_word:
                    word_bboxes.append((current_word, current_word_bbox))
                # Create an OCRResult for the current line
                line_level_results_out.append(OCRResult(full_text, round(overall_bbox[0], 2), round(overall_bbox[1], 2), round(overall_bbox[2] - overall_bbox[0], 2), round(overall_bbox[3] - overall_bbox[1], 2)))
                line_level_characters_out.append(character_objects_out)
                # Reset for the next line
                character_objects_out = []
                full_text = ""
                overall_bbox = [float('inf'), float('inf'), float('-inf'), float('-inf')]
                current_word = ""
                current_word_bbox = [float('inf'), float('inf'), float('-inf'), float('-inf')]

            continue

        # Concatenate text for LTChar


        #full_text += char.get_text()
        #added_text = re.sub(r'[^\x00-\x7F]+', ' ', char.get_text())
        added_text = char.get_text()
        if re.search(r'[^\x00-\x7F]', added_text):  # Matches any non-ASCII character
            #added_text.encode('latin1', errors='replace').decode('utf-8')
            added_text = clean_unicode_text(added_text)
        full_text += added_text  # Adds space or newline, removing 

        

        # Update overall bounding box
        x0, y0, x1, y1 = char.bbox
        overall_bbox[0] = min(overall_bbox[0], x0)  # x0
        overall_bbox[1] = min(overall_bbox[1], y0)  # y0
        overall_bbox[2] = max(overall_bbox[2], x1)  # x1
        overall_bbox[3] = max(overall_bbox[3], y1)  # y1
        
        # Update current word
        #current_word += char.get_text()
        current_word += added_text
        
        # Update current word bounding box
        current_word_bbox[0] = min(current_word_bbox[0], x0)  # x0
        current_word_bbox[1] = min(current_word_bbox[1], y0)  # y0
        current_word_bbox[2] = max(current_word_bbox[2], x1)  # x1
        current_word_bbox[3] = max(current_word_bbox[3], y1)  # y1

    # Finalize the last word if any
    if current_word:
        word_bboxes.append((current_word, current_word_bbox))

    if full_text:
        #print("full_text before:", full_text)
        if re.search(r'[^\x00-\x7F]', full_text):  # Matches any non-ASCII character
            # Convert special characters to a human-readable format
            #full_text = full_text.encode('latin1', errors='replace').decode('utf-8')
            full_text = clean_unicode_text(full_text)
        #print("full_text:", full_text)

        line_level_results_out.append(OCRResult(full_text, round(overall_bbox[0],2), round(overall_bbox[1], 2), round(overall_bbox[2]-overall_bbox[0],2), round(overall_bbox[3]-overall_bbox[1],2)))

    #line_level_characters_out = character_objects_out        

    return line_level_results_out, line_level_characters_out  # Return both results and character objects

def merge_text_bounding_boxes(analyser_results:CustomImageRecognizerResult, characters:List[LTChar], combine_pixel_dist:int=20, vertical_padding:int=0):
    '''
    Merge identified bounding boxes containing PII that are very close to one another
    '''
    analysed_bounding_boxes = []
    if len(analyser_results) > 0 and len(characters) > 0:
        # Extract bounding box coordinates for sorting
        bounding_boxes = []
        text_out = []
        for result in analyser_results:
            char_boxes = [char.bbox for char in characters[result.start:result.end] if isinstance(char, LTChar)]
            char_text = [char._text for char in characters[result.start:result.end] if isinstance(char, LTChar)]
            if char_boxes:
                # Calculate the bounding box that encompasses all characters
                left = min(box[0] for box in char_boxes)
                bottom = min(box[1] for box in char_boxes)
                right = max(box[2] for box in char_boxes)
                top = max(box[3] for box in char_boxes) + vertical_padding
                bounding_boxes.append((bottom, left, result, [left, bottom, right, top], char_text))  # (y, x, result, bbox, text)

        char_text = "".join(char_text)

        # Sort the results by y-coordinate and then by x-coordinate
        bounding_boxes.sort()

        merged_bounding_boxes = []
        current_box = None
        current_y = None
        current_result = None
        current_text = []

        for y, x, result, char_box, text in bounding_boxes:
            #print(f"Considering result: {result}")
            #print(f"Character box: {char_box}")

            if current_y is None or current_box is None:
                current_box = char_box
                current_y = char_box[1]
                current_result = result
                current_text = list(text)
                #print(f"Starting new box: {current_box}")
            else:
                vertical_diff_bboxes = abs(char_box[1] - current_y)
                horizontal_diff_bboxes = abs(char_box[0] - current_box[2])

                #print(f"Comparing boxes: current_box={current_box}, char_box={char_box}, current_text={current_text}, char_text={text}")
                #print(f"Vertical diff: {vertical_diff_bboxes}, Horizontal diff: {horizontal_diff_bboxes}")

                if (
                    vertical_diff_bboxes <= 5 and horizontal_diff_bboxes <= combine_pixel_dist
                ):
                    #print("box is being extended")
                    current_box[2] = char_box[2]  # Extend the current box horizontally
                    current_box[3] = max(current_box[3], char_box[3])  # Ensure the top is the highest
                    current_result.end = max(current_result.end, result.end)  # Extend the text range
                    try:
                        current_result.entity_type = current_result.entity_type + " - " + result.entity_type
                    except Exception as e:
                        print("Unable to combine result entity types:")
                        print(e)
                    # Add a space if current_text is not empty
                    if current_text:
                        current_text.append(" ")  # Add space between texts
                    current_text.extend(text)

                    #print(f"Latest merged box: {current_box[-1]}")
                else:
                    merged_bounding_boxes.append(
                        {"text":"".join(current_text),"boundingBox": current_box, "result": current_result})
                    #print(f"Appending merged box: {current_box}")
                    #print(f"Latest merged box: {merged_bounding_boxes[-1]}")

                    # Reset current_box and current_y after appending
                    current_box = char_box
                    current_y = char_box[1]
                    current_result = result
                    current_text = list(text)
                    #print(f"Starting new box: {current_box}")

        # After finishing with the current result, add the last box for this result
        if current_box:
            merged_bounding_boxes.append({"text":"".join(current_text), "boundingBox": current_box, "result": current_result})
            #print(f"Appending final box for result: {current_box}")

        if not merged_bounding_boxes:
            analysed_bounding_boxes.extend(
                {"text":text, "boundingBox": char.bbox, "result": result} 
                for result in analyser_results 
                for char in characters[result.start:result.end] 
                if isinstance(char, LTChar)
            )
        else:
            analysed_bounding_boxes.extend(merged_bounding_boxes)

        #print("Analyzed bounding boxes:\n\n", analysed_bounding_boxes)
    
    return analysed_bounding_boxes

def identify_pii_in_text_container(text_container:OCRResult, language:str, chosen_redact_entities:List[str], chosen_redact_comprehend_entities:List[str], score_threshold:float, allow_list:List[str], pii_identification_method:str="Local") -> List[RecognizerResult]:
    '''
    Take text and bounding boxes in OCRResult format and analyze it for PII using spacy and the Microsoft Presidio package, or the AWS Comprehend service.
    '''

    analyser_results = []

    #text_to_analyse = initial_clean(text_container.text).strip()

    text_to_analyse = text_container.text

    if chosen_redact_entities:
        if pii_identification_method == "Local":
            analyser_results = nlp_analyser.analyze(text=text_to_analyse,
                                                    language=language, 
                                                    entities=chosen_redact_entities,
                                                    score_threshold=score_threshold,
                                                    return_decision_process=True,
                                                    allow_list=allow_list)
        
        elif pii_identification_method == "AWS Comprehend":

            # Call the detect_pii_entities method
            response = comprehend_client.detect_pii_entities(
                Text=text_to_analyse,
                LanguageCode=language  # Specify the language of the text
            )

            for result in response["Entities"]:

                result_text = text_to_analyse[result["BeginOffset"]:result["EndOffset"]+1]

                if result_text not in allow_list:
                    if result.get("Type") in chosen_redact_comprehend_entities:

                        recogniser_entity = recognizer_result_from_dict(result)

                        analyser_results.append(recogniser_entity)

        else:
            analyser_results = []
    else:
        analyser_results = []
    
         
    return analyser_results

def create_text_redaction_process_results(analyser_results, analysed_bounding_boxes, page_num):
    decision_process_table = pd.DataFrame()

    if len(analyser_results) > 0:
        # Create summary df of annotations to be made
        analysed_bounding_boxes_df_new = pd.DataFrame(analysed_bounding_boxes)
        analysed_bounding_boxes_df_text = analysed_bounding_boxes_df_new['result'].astype(str).str.split(",",expand=True).replace(".*: ", "", regex=True)
        analysed_bounding_boxes_df_text.columns = ["type", "start", "end", "score"]
        analysed_bounding_boxes_df_new = pd.concat([analysed_bounding_boxes_df_new, analysed_bounding_boxes_df_text], axis = 1)
        analysed_bounding_boxes_df_new['page'] = page_num + 1
        decision_process_table = pd.concat([decision_process_table, analysed_bounding_boxes_df_new], axis = 0).drop('result', axis=1)

        #print('\n\ndecision_process_table:\n\n', decision_process_table)
    
    return decision_process_table

def create_annotations_for_bounding_boxes(analysed_bounding_boxes):
    annotations_on_page = []
    for analysed_bounding_box in analysed_bounding_boxes:
        bounding_box = analysed_bounding_box["boundingBox"]
        annotation = Dictionary(
            Type=Name.Annot,
            Subtype=Name.Square, #Name.Highlight,
            QuadPoints=[bounding_box[0], bounding_box[3], bounding_box[2], bounding_box[3],
                        bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[1]],
            Rect=[bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]],
            C=[0, 0, 0],
            IC=[0, 0, 0],
            CA=1, # Transparency
            T=analysed_bounding_box["result"].entity_type,
            BS=Dictionary(
                W=0,                     # Border width: 1 point
                S=Name.S                # Border style: solid
            )
        )
        annotations_on_page.append(annotation)
    return annotations_on_page

def redact_text_pdf(
    filename: str,  # Path to the PDF file to be redacted
    prepared_pdf_image_path: str,  # Path to the prepared PDF image for redaction
    language: str,  # Language of the PDF content
    chosen_redact_entities: List[str],  # List of entities to be redacted
    chosen_redact_comprehend_entities: List[str],
    allow_list: List[str] = None,  # Optional list of allowed entities
    page_min: int = 0,  # Minimum page number to start redaction
    page_max: int = 999,  # Maximum page number to end redaction
    analysis_type: str = "Simple text analysis - PDFs with selectable text",  # Type of analysis to perform
    current_loop_page: int = 0,  # Current page being processed in the loop
    page_break_return: bool = False,  # Flag to indicate if a page break should be returned
    annotations_all_pages: List = [],  # List of annotations across all pages
    all_line_level_ocr_results_df: pd.DataFrame = pd.DataFrame(),  # DataFrame for OCR results
    all_decision_process_table: pd.DataFrame = pd.DataFrame(),  # DataFrame for decision process table
    pymupdf_doc: List = [],  # List of PyMuPDF documents
    pii_identification_method: str = "Local",
    page_break_val: int = int(page_break_value),  # Value for page break
    max_time: int = int(max_time_value),    
    progress: Progress = Progress(track_tqdm=True)  # Progress tracking object
):
    
    '''
    Redact chosen entities from a PDF that is made up of multiple pages that are not images.
    
    Input Variables:
    - filename: Path to the PDF file to be redacted
    - prepared_pdf_image_path: Path to the prepared PDF image for redaction
    - language: Language of the PDF content
    - chosen_redact_entities: List of entities to be redacted
    - chosen_redact_comprehend_entities: List of entities to be redacted for AWS Comprehend
    - allow_list: Optional list of allowed entities
    - page_min: Minimum page number to start redaction
    - page_max: Maximum page number to end redaction
    - analysis_type: Type of analysis to perform
    - current_loop_page: Current page being processed in the loop
    - page_break_return: Flag to indicate if a page break should be returned
    - annotations_all_pages: List of annotations across all pages
    - all_line_level_ocr_results_df: DataFrame for OCR results
    - all_decision_process_table: DataFrame for decision process table
    - pymupdf_doc: List of PyMuPDF documents
    - pii_identification_method (str, optional): The method to redact personal information. Either 'Local' (spacy model), or 'AWS Comprehend' (AWS Comprehend API).
    - page_break_val: Value for page break
    - max_time (int, optional): The maximum amount of time (s) that the function should be running before it breaks. To avoid timeout errors with some APIs.    
    - progress: Progress tracking object
    '''

    tic = time.perf_counter()

    # Open with Pikepdf to get text lines
    pikepdf_pdf = Pdf.open(filename)
    number_of_pages = len(pikepdf_pdf.pages)
    
    # Check that page_min and page_max are within expected ranges
    if page_max > number_of_pages or page_max == 0:
        page_max = number_of_pages

    if page_min <= 0: page_min = 0
    else: page_min = page_min - 1

    print("Page range is",str(page_min + 1), "to", str(page_max))
    print("Current_loop_page:", current_loop_page)

    if current_loop_page == 0: page_loop_start = 0
    else: page_loop_start = current_loop_page

    progress_bar = tqdm(range(current_loop_page, number_of_pages), unit="pages remaining", desc="Redacting pages")

    #for page_no in range(0, number_of_pages):     
    for page_no in progress_bar:

        reported_page_number = str(page_no + 1)
        print("Redacting page:", reported_page_number)

        # Assuming prepared_pdf_file_paths[page_no] is a PIL image object
        try:
            image = prepared_pdf_image_path[page_no]#.copy()
            #print("image:", image)
        except Exception as e:
            print("Could not redact page:", reported_page_number, "due to:")
            print(e)
            continue

        image_annotations = {"image": image, "boxes": []} 
        pymupdf_page = pymupdf_doc.load_page(page_no)

        if page_min <= page_no < page_max:

            for page_layout in extract_pages(filename, page_numbers = [page_no], maxpages=1):
                
                page_analyser_results = []
                page_analysed_bounding_boxes = []            
                
                characters = []
                annotations_on_page = []
                decision_process_table_on_page = pd.DataFrame()    
                page_text_outputs = pd.DataFrame()  

                if analysis_type == "Simple text analysis - PDFs with selectable text":
                    for n, text_container in enumerate(page_layout):

                        text_container_analyser_results = []
                        text_container_analysed_bounding_boxes = []
                        characters = []

                        if isinstance(text_container, LTTextContainer) or isinstance(text_container, LTAnno):
                            characters = get_text_container_characters(text_container)

                        # Create dataframe for all the text on the page
                        line_level_text_results_list, line_characters = create_text_bounding_boxes_from_characters(characters)

                        # Create page_text_outputs (OCR format outputs)
                        if line_level_text_results_list:
                            # Convert to DataFrame and add to ongoing logging table
                            line_level_text_results_df = pd.DataFrame([{
                                'page': page_no + 1,
                                'text': result.text,
                                'left': result.left,
                                'top': result.top,
                                'width': result.width,
                                'height': result.height
                            } for result in line_level_text_results_list])

                            page_text_outputs = pd.concat([page_text_outputs, line_level_text_results_df])

                        # Analyse each line of text in turn for PII and add to list
                        for i, text_line in enumerate(line_level_text_results_list):

                            text_line_analyser_result = []
                            text_line_bounding_boxes = []

                            # text_line_analyser_result = identify_pii_in_text_container(text_line, language, chosen_redact_entities, score_threshold, allow_list)

                            #pii_identification_method="AWS Comprehend"#"Local"

                            if chosen_redact_entities:

                                text_line_analyser_result = identify_pii_in_text_container(text_line, language, chosen_redact_entities, chosen_redact_comprehend_entities, score_threshold, allow_list, pii_identification_method)
                                
                            else:
                                text_line_analyser_result = []

                            # Merge bounding boxes for the line if multiple found close together                    
                            if text_line_analyser_result:

                                #print("Analysed text container, now merging bounding boxes")

                                # Merge bounding boxes if very close together
                                text_line_bounding_boxes = merge_text_bounding_boxes(text_line_analyser_result, line_characters[i])

                                #print("merged bounding boxes")

                                text_container_analyser_results.extend(text_line_analyser_result)
                                text_container_analysed_bounding_boxes.extend(text_line_bounding_boxes)
                                                    
                        page_analyser_results.extend(text_container_analyser_results)
                        page_analysed_bounding_boxes.extend(text_container_analysed_bounding_boxes)

                # Annotate redactions on page
                annotations_on_page = create_annotations_for_bounding_boxes(page_analysed_bounding_boxes)

                # Make pymupdf page redactions
                pymupdf_page, image_annotations = redact_page_with_pymupdf(pymupdf_page, annotations_on_page, image)

                #print("Did redact_page_with_pymupdf function")

                print("For page number:", page_no, "there are", len(image_annotations["boxes"]), "annotations")

                # Write logs
                # Create decision process table
                decision_process_table_on_page = create_text_redaction_process_results(page_analyser_results, page_analysed_bounding_boxes, current_loop_page)     

                if not decision_process_table_on_page.empty:
                    all_decision_process_table = pd.concat([all_decision_process_table, decision_process_table_on_page])

                if not page_text_outputs.empty:
                    page_text_outputs = page_text_outputs.sort_values(["top", "left"], ascending=[False, False]).reset_index(drop=True)
                    all_line_level_ocr_results_df = pd.concat([all_line_level_ocr_results_df, page_text_outputs])

                toc = time.perf_counter()

                time_taken = toc - tic

                #print("toc - tic:", time_taken)

                # Break if time taken is greater than max_time seconds
                if time_taken > max_time:
                    print("Processing for", max_time, "seconds, breaking.")
                    page_break_return = True
                    progress.close(_tqdm=progress_bar)
                    tqdm._instances.clear()

                    annotations_all_pages.append(image_annotations)

                    current_loop_page += 1

                    return pymupdf_doc, all_decision_process_table, all_line_level_ocr_results_df, annotations_all_pages, current_loop_page, page_break_return
                

        annotations_all_pages.append(image_annotations)

        current_loop_page += 1

        # Break if new page is a multiple of 10
        if current_loop_page % page_break_val == 0:
            page_break_return = True
            progress.close(_tqdm=progress_bar)

            return pymupdf_doc, all_decision_process_table, all_line_level_ocr_results_df, annotations_all_pages, current_loop_page, page_break_return
        
            
    return pymupdf_doc, all_decision_process_table, all_line_level_ocr_results_df, annotations_all_pages, current_loop_page, page_break_return