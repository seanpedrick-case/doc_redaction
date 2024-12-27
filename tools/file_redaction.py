import time
import re
import json
import io
import os
import boto3
import copy

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
from fitz import Page
import gradio as gr
from gradio import Progress
from collections import defaultdict  # For efficient grouping

from presidio_analyzer import RecognizerResult
from tools.aws_functions import RUN_AWS_FUNCTIONS
from tools.custom_image_analyser_engine import CustomImageAnalyzerEngine, OCRResult, combine_ocr_results, CustomImageRecognizerResult
from tools.file_conversion import process_file, image_dpi, convert_review_json_to_pandas_df, redact_whole_pymupdf_page, redact_single_box, convert_pymupdf_to_image_coords
from tools.load_spacy_model_custom_recognisers import nlp_analyser, score_threshold, custom_entities, custom_recogniser, custom_word_list_recogniser
from tools.helper_functions import get_file_path_end, output_folder, clean_unicode_text, get_or_create_env_var, tesseract_ocr_option, text_ocr_option, textract_option, local_pii_detector, aws_pii_detector
from tools.file_conversion import process_file, is_pdf, is_pdf_or_image
from tools.aws_textract import analyse_page_with_textract, json_to_ocrresult
from tools.presidio_analyzer_custom import recognizer_result_from_dict

# Number of pages to loop through before breaking. Currently set very high, as functions are breaking on time metrics (e.g. every 105 seconds), rather than on number of pages redacted.
page_break_value = get_or_create_env_var('page_break_value', '50000')
print(f'The value of page_break_value is {page_break_value}')

max_time_value = get_or_create_env_var('max_time_value', '999999')
print(f'The value of max_time_value is {max_time_value}')

def bounding_boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    return (box1[0] < box2[2] and box2[0] < box1[2] and
            box1[1] < box2[3] and box2[1] < box1[3])

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
 custom_recogniser_word_list:List[str]=None, 
 redact_whole_page_list:List[str]=None,
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
 comprehend_query_number:int=0,
 output_folder:str=output_folder,
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
    - custom_recogniser_word_list (List[List[str]], optional): A list of allowed terms for redaction. Defaults to None.
    - redact_whole_page_list (List[List[str]], optional): A list of allowed terms for redaction. Defaults to None.
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
    - comprehend_query_number (int, optional): A counter tracking the number of queries to AWS Comprehend.
    - output_folder (str, optional): Output folder for results.
    - progress (gr.Progress, optional): A progress tracker for the redaction process. Defaults to a Progress object with track_tqdm set to True.

    The function returns a redacted document along with processing logs.
    '''
    combined_out_message = ""
    tic = time.perf_counter()
    all_request_metadata = all_request_metadata_str.split('\n') if all_request_metadata_str else []

    if isinstance(custom_recogniser_word_list, pd.DataFrame):
        custom_recogniser_word_list = custom_recogniser_word_list.iloc[:,0].tolist()

        # Sort the strings in order from the longest string to the shortest
        custom_recogniser_word_list = sorted(custom_recogniser_word_list, key=len, reverse=True)

    if isinstance(redact_whole_page_list, pd.DataFrame):
        redact_whole_page_list = redact_whole_page_list.iloc[:,0].tolist()


    # If this is the first time around, set variables to 0/blank
    if first_loop_state==True:
        #print("First_loop_state is True")
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

        print("Completed last file")
        # Set to a very high number so as not to mix up with subsequent file processing by the user
        # latest_file_completed = 99
        current_loop_page = 0

        if isinstance(out_message, list):
            combined_out_message = '\n'.join(out_message)
        else:
            combined_out_message = out_message
        
        estimate_total_processing_time = sum_numbers_before_seconds(combined_out_message)
        print("Estimated total processing time:", str(estimate_total_processing_time))

        return combined_out_message, out_file_paths, out_file_paths, gr.Number(value=latest_file_completed, label="Number of documents redacted", interactive=False, visible=False), log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages, gr.Number(value=current_loop_page,precision=0, interactive=False, label = "Last redacted page in document", visible=False), gr.Checkbox(value = True, label="Page break reached", visible=False), all_line_level_ocr_results_df, all_decision_process_table, comprehend_query_number
    
    # If we have reached the last page, return message
    if current_loop_page >= number_of_pages:
        print("current_loop_page:", current_loop_page, "is equal to or greater than number of pages in document:", number_of_pages)

        # Set to a very high number so as not to mix up with subsequent file processing by the user
        current_loop_page = 999
        combined_out_message = out_message

        return combined_out_message, out_file_paths, out_file_paths, gr.Number(value=latest_file_completed, label="Number of documents redacted", interactive=False, visible=False), log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages, gr.Number(value=current_loop_page,precision=0, interactive=False, label = "Last redacted page in document", visible=False), gr.Checkbox(value = False, label="Page break reached", visible=False), all_line_level_ocr_results_df, all_decision_process_table, comprehend_query_number

    # Create allow list
    # If string, assume file path
    if isinstance(in_allow_list, str):
        in_allow_list = pd.read_csv(in_allow_list)

    if not in_allow_list.empty:
        in_allow_list_flat = in_allow_list.iloc[:,0].tolist()
        #print("In allow list:", in_allow_list_flat)
    else:
        in_allow_list_flat = []


    # Try to connect to AWS services only if RUN_AWS_FUNCTIONS environmental variable is 1
    if pii_identification_method == "AWS Comprehend":
        print("Trying to connect to AWS Comprehend service")
        if RUN_AWS_FUNCTIONS == "1":
            comprehend_client = boto3.client('comprehend')
        else:
            comprehend_client = ""
            out_message = "Cannot connect to AWS Comprehend service. Please choose another PII identification method."
            print(out_message)
            return out_message, out_file_paths, out_file_paths, gr.Number(value=latest_file_completed, label="Number of documents redacted", interactive=False, visible=False), log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages, gr.Number(value=current_loop_page, precision=0, interactive=False, label = "Last redacted page in document", visible=False), gr.Checkbox(value = True, label="Page break reached", visible=False), all_line_level_ocr_results_df, all_decision_process_table, comprehend_query_number
    else:
        comprehend_client = ""
        
    if in_redact_method == textract_option:
        print("Trying to connect to AWS Comprehend service")
        if RUN_AWS_FUNCTIONS == "1":
            textract_client = boto3.client('textract')
        else:
            textract_client = ""
            out_message = "Cannot connect to AWS Textract. Please choose another text extraction method."
            print(out_message)
            return out_message, out_file_paths, out_file_paths, gr.Number(value=latest_file_completed, label="Number of documents redacted", interactive=False, visible=False), log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages, gr.Number(value=current_loop_page, precision=0, interactive=False, label = "Last redacted page in document", visible=False), gr.Checkbox(value = True, label="Page break reached", visible=False), all_line_level_ocr_results_df, all_decision_process_table, comprehend_query_number
    else:
        textract_client = ""

    # Check if output_folder exists, create it if it doesn't
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    progress(0.5, desc="Redacting file")
    
    if isinstance(file_paths, str):
        file_paths_list = [os.path.abspath(file_paths)]
        file_paths_loop = file_paths_list
    elif isinstance(file_paths, dict):
        file_paths = file_paths["name"]
        file_paths_list = [os.path.abspath(file_paths)]
        file_paths_loop = file_paths_list
    else:
        file_paths_list = file_paths
        file_paths_loop = [file_paths_list[int(latest_file_completed)]]    

    # print("file_paths_list in choose_redactor function:", file_paths_list)


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
                in_redact_method = tesseract_ocr_option
        else:
            out_message = "No file selected"
            print(out_message)

            return combined_out_message, out_file_paths, out_file_paths, gr.Number(value=latest_file_completed, label="Number of documents redacted", interactive=False, visible=False), log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages, gr.Number(value=current_loop_page,precision=0, interactive=False, label = "Last redacted page in document", visible=False), gr.Checkbox(value = True, label="Page break reached", visible=False), all_line_level_ocr_results_df, all_decision_process_table, comprehend_query_number

        if in_redact_method == tesseract_ocr_option or in_redact_method == textract_option:

            #Analyse and redact image-based pdf or image
            if is_pdf_or_image(file_path) == False:
                out_message = "Please upload a PDF file or image file (JPG, PNG) for image analysis."
                return out_message, out_file_paths, out_file_paths, gr.Number(value=latest_file_completed, label="Number of documents redacted", interactive=False, visible=False), log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages, gr.Number(value=current_loop_page, precision=0, interactive=False, label = "Last redacted page in document", visible=False), gr.Checkbox(value = True, label="Page break reached", visible=False), all_line_level_ocr_results_df, all_decision_process_table, comprehend_query_number

            print("Redacting file " + file_path_without_ext + " as an image-based file")

            pymupdf_doc,all_decision_process_table,log_files_output_paths,new_request_metadata, annotations_all_pages, current_loop_page, page_break_return, all_line_level_ocr_results_df, comprehend_query_number = redact_image_pdf(file_path,
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
             pii_identification_method,
             comprehend_query_number,
             comprehend_client,
             textract_client,
             custom_recogniser_word_list,
             redact_whole_page_list)

            # Save Textract request metadata (if exists)
            if new_request_metadata:
                print("Request metadata:", new_request_metadata)
                all_request_metadata.append(new_request_metadata)              

        elif in_redact_method == text_ocr_option:

            #log_files_output_paths = []
            
            if is_pdf(file_path) == False:
                out_message = "Please upload a PDF file for text analysis. If you have an image, select 'Image analysis'."
                return out_message, out_file_paths, out_file_paths, gr.Number(value=latest_file_completed, label="Number of documents redacted", interactive=False, visible=False), log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages, gr.Number(value=current_loop_page,precision=0, interactive=False, label = "Last redacted page in document", visible=False), gr.Checkbox(value = True, label="Page break reached", visible=False), all_line_level_ocr_results_df, all_decision_process_table, comprehend_query_number
            
            # Analyse text-based pdf
            print('Redacting file as text-based PDF')
            
            pymupdf_doc, all_decision_process_table, all_line_level_ocr_results_df, annotations_all_pages, current_loop_page, page_break_return, comprehend_query_number = redact_text_pdf(file_path,
            prepared_pdf_image_paths,language,
            chosen_redact_entities,
            chosen_redact_comprehend_entities,
            in_allow_list_flat,
            page_min,
            page_max,
            text_ocr_option,
            current_loop_page,
            page_break_return,
            annotations_all_pages,
            all_line_level_ocr_results_df,
            all_decision_process_table,
            pymupdf_doc,
            pii_identification_method,
            comprehend_query_number,
            comprehend_client,
            custom_recogniser_word_list,
            redact_whole_page_list)

        else:
            out_message = "No redaction method selected"
            print(out_message)
            return out_message, out_file_paths, out_file_paths, gr.Number(value=latest_file_completed, label="Number of documents redacted", interactive=False, visible=False), log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages, gr.Number(value=current_loop_page,precision=0, interactive=False, label = "Last redacted page in document", visible=False), gr.Checkbox(value = True, label="Page break reached", visible=False), all_line_level_ocr_results_df, all_decision_process_table, comprehend_query_number
        
        # If at last page, save to file
        if current_loop_page >= number_of_pages:

            print("Current page loop:", current_loop_page, "is greater or equal to number of pages:", number_of_pages)
            latest_file_completed += 1
            current_loop_page = 999

            if latest_file_completed != len(file_paths_list):
                print("Completed file number:", str(latest_file_completed), "there are more files to do")                    

            # Save file
            if is_pdf(file_path) == False:
                out_image_file_path = output_folder + file_path_without_ext + "_redacted_as_pdf.pdf"
                pymupdf_doc[0].save(out_image_file_path, "PDF" ,resolution=image_dpi, save_all=False)#, append_images=pymupdf_doc[:1])
            
            else:
                out_image_file_path = output_folder + file_path_without_ext + "_redacted.pdf"
                pymupdf_doc.save(out_image_file_path)

            out_file_paths.append(out_image_file_path)

            #if log_files_output_paths:
            #    log_files_output_paths.extend(log_files_output_paths)

            logs_output_file_name = out_image_file_path + "_decision_process_output.csv"
            all_decision_process_table.to_csv(logs_output_file_name, index = None, encoding="utf-8")
            log_files_output_paths.append(logs_output_file_name)

            all_text_output_file_name = out_image_file_path + "_ocr_output.csv"
            all_line_level_ocr_results_df.to_csv(all_text_output_file_name, index = None, encoding="utf-8")
            out_file_paths.append(all_text_output_file_name)

            # Save the gradio_annotation_boxes to a JSON file
            try:
                print("Saving annotations to JSON")

                out_annotation_file_path = out_image_file_path + '_review_file.json'
                with open(out_annotation_file_path, 'w') as f:
                    json.dump(annotations_all_pages, f)
                log_files_output_paths.append(out_annotation_file_path)

                #print("Saving annotations to CSV")

                # Convert json to csv and also save this
                #print("annotations_all_pages:", annotations_all_pages)

                review_df = convert_review_json_to_pandas_df(annotations_all_pages, all_decision_process_table)

                out_review_file_file_path = out_image_file_path + '_review_file.csv'
                review_df.to_csv(out_review_file_file_path, index=None)
                out_file_paths.append(out_review_file_file_path)

                print("Saved review file to csv")

            except Exception as e:
                print("Could not save annotations to json file:", e)

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

        else:
            toc = time.perf_counter()
            time_taken = toc - tic
            estimated_time_taken_state = estimated_time_taken_state + time_taken


   # If textract requests made, write to logging file
    if all_request_metadata:
        all_request_metadata_str = '\n'.join(all_request_metadata).strip()

        all_request_metadata_file_path = output_folder + file_path_without_ext + "_textract_request_metadata.txt"   

        with open(all_request_metadata_file_path, "w") as f:
            f.write(all_request_metadata_str)

        # Add the request metadata to the log outputs if not there already
        if all_request_metadata_file_path not in log_files_output_paths:
            log_files_output_paths.append(all_request_metadata_file_path)

    if combined_out_message: out_message = combined_out_message
    
    #print("\nout_message at choose_and_run_redactor end is:", out_message)

    # Ensure no duplicated output files
    log_files_output_paths = list(set(log_files_output_paths))
    out_file_paths = list(set(out_file_paths))


    return out_message, out_file_paths, out_file_paths, gr.Number(value=latest_file_completed, label="Number of documents redacted", interactive=False, visible=False), log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str, pymupdf_doc, annotations_all_pages, gr.Number(value=current_loop_page, precision=0, interactive=False, label = "Last redacted page in document", visible=False), gr.Checkbox(value = True, label="Page break reached", visible=False), all_line_level_ocr_results_df, all_decision_process_table, comprehend_query_number

def convert_pikepdf_coords_to_pymupdf(pymupdf_page, pikepdf_bbox, type="pikepdf_annot"):
    '''
    Convert annotations from pikepdf to pymupdf format, handling the mediabox larger than rect.
    '''
    # Use cropbox if available, otherwise use mediabox
    reference_box = pymupdf_page.rect
    mediabox = pymupdf_page.mediabox

    reference_box_height = reference_box.height
    reference_box_width = reference_box.width
    
    # Convert PyMuPDF coordinates back to PDF coordinates (bottom-left origin)
    media_height = mediabox.height
    media_width = mediabox.width

    media_reference_y_diff = media_height - reference_box_height
    media_reference_x_diff = media_width - reference_box_width

    y_diff_ratio = media_reference_y_diff / reference_box_height
    x_diff_ratio = media_reference_x_diff / reference_box_width
    
    # Extract the annotation rectangle field
    if type=="pikepdf_annot":
        rect_field = pikepdf_bbox["/Rect"]
    else:
        rect_field = pikepdf_bbox
    rect_coordinates = [float(coord) for coord in rect_field]  # Convert to floats

    # Unpack coordinates
    x1, y1, x2, y2 = rect_coordinates
    
    new_x1 = x1 - (media_reference_x_diff * x_diff_ratio)
    new_y1 = media_height - y2 - (media_reference_y_diff * y_diff_ratio)
    new_x2 = x2 - (media_reference_x_diff * x_diff_ratio)
    new_y2 = media_height - y1 - (media_reference_y_diff * y_diff_ratio)
    
    return new_x1, new_y1, new_x2, new_y2

def convert_pikepdf_to_image_coords(pymupdf_page, annot, image:Image, type="pikepdf_annot"):
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
    if type=="pikepdf_annot":
        rect_field = annot["/Rect"]
    else:
        rect_field = annot

    # Convert the extracted /Rect field to a list of floats
    rect_coordinates = [float(coord) for coord in rect_field]

    # Convert the Y-coordinates (flip using the image height)
    x1, y1, x2, y2 = rect_coordinates
    x1_image = x1 * scale_width
    new_y1_image = image_page_height - (y2 * scale_height)  # Flip Y0 (since it starts from bottom)
    x2_image = x2 * scale_width
    new_y2_image = image_page_height - (y1 * scale_height)  # Flip Y1

    return x1_image, new_y1_image, x2_image, new_y2_image

def convert_pikepdf_decision_output_to_image_coords(pymupdf_page, pikepdf_decision_ouput_data:List, image):
    if isinstance(image, str):
        image_path = image
        image = Image.open(image_path)

    # Loop through each item in the data
    for item in pikepdf_decision_ouput_data:
        # Extract the bounding box
        bounding_box = item['boundingBox']
        
        # Create a pikepdf_bbox dictionary to match the expected input
        pikepdf_bbox = {"/Rect": bounding_box}
        
        # Call the conversion function
        new_x1, new_y1, new_x2, new_y2 = convert_pikepdf_to_image_coords(pymupdf_page, pikepdf_bbox, image, type="pikepdf_annot")
        
        # Update the original object with the new bounding box values
        item['boundingBox'] = [new_x1, new_y1, new_x2, new_y2]

    return pikepdf_decision_ouput_data

def convert_image_coords_to_pymupdf(pymupdf_page, annot, image:Image, type="image_recognizer"):
    '''
    Converts an image with redaction coordinates from a CustomImageRecognizerResult or pikepdf object with image coordinates to pymupdf coordinates.
    '''

    rect_height = pymupdf_page.rect.height
    rect_width = pymupdf_page.rect.width 

    image_page_width, image_page_height = image.size

    # Calculate scaling factors between PIL image and pymupdf
    scale_width = rect_width / image_page_width
    scale_height = rect_height / image_page_height

    # Calculate scaled coordinates
    if type == "image_recognizer":
        x1 = (annot.left * scale_width)# + page_x_adjust
        new_y1 = (annot.top * scale_height)# - page_y_adjust  # Flip Y0 (since it starts from bottom)
        x2 = ((annot.left + annot.width) * scale_width)# + page_x_adjust  # Calculate x1
        new_y2 = ((annot.top + annot.height) * scale_height)# - page_y_adjust  # Calculate y1 correctly
    # Else assume it is a pikepdf derived object
    else:
        rect_field = annot["/Rect"]
        rect_coordinates = [float(coord) for coord in rect_field]  # Convert to floats

        # Unpack coordinates
        x1, y1, x2, y2 = rect_coordinates

        #print("scale_width:", scale_width)
        #print("scale_height:", scale_height)

        x1 = (x1* scale_width)# + page_x_adjust
        new_y1 = ((y2 + (y1 - y2))* scale_height)# - page_y_adjust  # Calculate y1 correctly        
        x2 = ((x1 + (x2 - x1)) * scale_width)# + page_x_adjust  # Calculate x1
        new_y2 = (y2 * scale_height)# - page_y_adjust  # Flip Y0 (since it starts from bottom)
        

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

def redact_page_with_pymupdf(page:Page, page_annotations:dict, image=None, custom_colours:bool=False, redact_whole_page:bool=False, convert_coords:bool=True):

    mediabox_height = page.mediabox[3] - page.mediabox[1]
    mediabox_width = page.mediabox[2] - page.mediabox[0]
    rect_height = page.rect.height
    rect_width = page.rect.width    

    pymupdf_x1 = None
    pymupdf_x2 = None

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
    if isinstance (page_annotations, dict):
        page_annotations = page_annotations["boxes"]

    for annot in page_annotations:
        # Check if an Image recogniser result, or a Gradio annotation object
        if (isinstance(annot, CustomImageRecognizerResult)) | isinstance(annot, dict):

            img_annotation_box = {}

            # Should already be in correct format if img_annotator_box is an input
            if isinstance(annot, dict):
                img_annotation_box = annot
                pymupdf_x1, pymupdf_y1, pymupdf_x2, pymupdf_y2 = convert_gradio_annotation_coords_to_pymupdf(page, annot, image)

                x1 = pymupdf_x1
                x2 = pymupdf_x2

                # if hasattr(annot, 'text') and annot.text:
                #     img_annotation_box["text"] = annot.text
                # else:
                #     img_annotation_box["text"] = ""

            # Else should be CustomImageRecognizerResult
            else:
                pymupdf_x1, pymupdf_y1, pymupdf_x2, pymupdf_y2 = convert_image_coords_to_pymupdf(page, annot, image)

                x1 = pymupdf_x1
                x2 = pymupdf_x2

                img_annotation_box["xmin"] = annot.left
                img_annotation_box["ymin"] = annot.top 
                img_annotation_box["xmax"] = annot.left + annot.width
                img_annotation_box["ymax"] = annot.top + annot.height
                img_annotation_box["color"] = (0,0,0)
                try:
                    img_annotation_box["label"] = annot.entity_type
                except:
                    img_annotation_box["label"] = "Redaction"
                # if hasattr(annot, 'text') and annot.text:
                #     img_annotation_box["text"] = annot.text
                # else:
                #     img_annotation_box["text"] = ""

            rect = Rect(x1, pymupdf_y1, x2, pymupdf_y2)  # Create the PyMuPDF Rect

        # Else it should be a pikepdf annotation object
        else:
            if convert_coords == True:    
                pymupdf_x1, pymupdf_y1, pymupdf_x2, pymupdf_y2 = convert_pikepdf_coords_to_pymupdf(page, annot)
            else:
                pymupdf_x1, pymupdf_y1, pymupdf_x2, pymupdf_y2 = convert_image_coords_to_pymupdf(page, annot, image, type="pikepdf_image_coords")

            x1 = pymupdf_x1
            x2 = pymupdf_x2

            rect = Rect(x1, pymupdf_y1, x2, pymupdf_y2)

            img_annotation_box = {}

            if image:
                img_width, img_height = image.size

                print("annot:", annot)

                x1, image_y1, x2, image_y2 = convert_pymupdf_to_image_coords(page, x1, pymupdf_y1, x2, pymupdf_y2, image)

                img_annotation_box["xmin"] = x1  #* (img_width / rect_width) # Use adjusted x1
                img_annotation_box["ymin"] = image_y1  #* (img_width / rect_width) # Use adjusted y1
                img_annotation_box["xmax"] = x2# * (img_height / rect_height) # Use adjusted x2
                img_annotation_box["ymax"] = image_y2 #* (img_height / rect_height) # Use adjusted y2
                img_annotation_box["color"] = (0, 0, 0)

                if isinstance(annot, Dictionary):
                    img_annotation_box["label"] = str(annot["/T"])
                else:
                    img_annotation_box["label"] = "REDACTION"
                # if hasattr(annot, 'text') and annot.text:
                #     img_annotation_box["text"] = annot.text
                # else:
                #     img_annotation_box["text"] = ""

        # Convert to a PyMuPDF Rect object
        #rect = Rect(rect_coordinates)

        all_image_annotation_boxes.append(img_annotation_box)

        redact_single_box(page, rect, img_annotation_box, custom_colours)

    # If whole page is to be redacted, do that here
    if redact_whole_page == True:

        whole_page_img_annotation_box = redact_whole_pymupdf_page(rect_height, rect_width, image, page, custom_colours, border = 5)
        all_image_annotation_boxes.append(whole_page_img_annotation_box)

    out_annotation_boxes = {
        "image": image_path, #Image.open(image_path), #image_path,
        "boxes": all_image_annotation_boxes
    }

    


    page.apply_redactions(images=0, graphics=0)
    page.clean_contents()

    return page, out_annotation_boxes



def merge_img_bboxes(bboxes, combined_results: Dict, signature_recogniser_results=[], handwriting_recogniser_results=[], handwrite_signature_checkbox: List[str]=["Redact all identified handwriting", "Redact all identified signatures"], horizontal_threshold:int=50, vertical_threshold:int=12):

    all_bboxes = []
    merged_bboxes = []
    grouped_bboxes = defaultdict(list)

    # Deep copy original bounding boxes to retain them
    original_bboxes = copy.deepcopy(bboxes)

    # Process signature and handwriting results
    if signature_recogniser_results or handwriting_recogniser_results:
        if "Redact all identified handwriting" in handwrite_signature_checkbox:
            merged_bboxes.extend(copy.deepcopy(handwriting_recogniser_results))

        if "Redact all identified signatures" in handwrite_signature_checkbox:
            merged_bboxes.extend(copy.deepcopy(signature_recogniser_results))

    # Reconstruct bounding boxes for substrings of interest
    reconstructed_bboxes = []
    for bbox in bboxes:
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
                        left = min(word['bounding_box'][0] for word in relevant_words)
                        top = min(word['bounding_box'][1] for word in relevant_words)
                        right = max(word['bounding_box'][2] for word in relevant_words)
                        bottom = max(word['bounding_box'][3] for word in relevant_words)

                        combined_text = " ".join(word['text'] for word in relevant_words)

                        reconstructed_bbox = CustomImageRecognizerResult(
                            bbox.entity_type,
                            bbox.start,
                            bbox.end,
                            bbox.score,
                            left,
                            top,
                            right - left,  # width
                            bottom - top,  # height,
                            combined_text
                        )
                        #reconstructed_bboxes.append(bbox)  # Add original bbox
                        reconstructed_bboxes.append(reconstructed_bbox)  # Add merged bbox
                        break
        else:
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
                new_text = merged_box.text + " " + next_box.text
                new_entity_type = merged_box.entity_type + " - " + next_box.entity_type

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

    all_bboxes.extend(original_bboxes)
    #all_bboxes.extend(reconstructed_bboxes)
    all_bboxes.extend(merged_bboxes)

    # Return the unique original and merged bounding boxes
    unique_bboxes = list({(bbox.left, bbox.top, bbox.width, bbox.height): bbox for bbox in all_bboxes}.values())
    return unique_bboxes

def redact_image_pdf(file_path:str,
                     prepared_pdf_file_paths:List[str],
                     language:str,
                     chosen_redact_entities:List[str],
                     chosen_redact_comprehend_entities:List[str],
                     allow_list:List[str]=None,
                     is_a_pdf:bool=True,
                     page_min:int=0,
                     page_max:int=999,
                     analysis_type:str=tesseract_ocr_option,
                     handwrite_signature_checkbox:List[str]=["Redact all identified handwriting", "Redact all identified signatures"],
                     request_metadata:str="", current_loop_page:int=0,
                     page_break_return:bool=False,
                     images=[],
                     annotations_all_pages:List=[],
                     all_line_level_ocr_results_df = pd.DataFrame(),
                     all_decision_process_table = pd.DataFrame(),
                     pymupdf_doc = [],
                     pii_identification_method:str="Local",
                     comprehend_query_number:int=0,
                     comprehend_client:str="",
                     textract_client:str="",
                     custom_recogniser_word_list:List[str]=[],
                     redact_whole_page_list:List[str]=[],
                     page_break_val:int=int(page_break_value),
                     log_files_output_paths:List=[],
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
    - analysis_type (str, optional): The type of analysis to perform on the PDF. Defaults to tesseract_ocr_option.
    - handwrite_signature_checkbox (List[str], optional): A list of options for redacting handwriting and signatures. Defaults to ["Redact all identified handwriting", "Redact all identified signatures"].
    - request_metadata (str, optional): Metadata related to the redaction request. Defaults to an empty string.
    - page_break_return (bool, optional): Indicates if the function should return after a page break. Defaults to False.
    - images (list, optional): List of image objects for each PDF page.
    - annotations_all_pages (List, optional): List of annotations on all pages that is used by the gradio_image_annotation object.
    - all_line_level_ocr_results_df (pd.DataFrame(), optional): All line level OCR results for the document as a Pandas dataframe,
    - all_decision_process_table (pd.DataFrame(), optional): All redaction decisions for document as a Pandas dataframe.
    - pymupdf_doc (List, optional): The document as a PyMupdf object.
    - pii_identification_method (str, optional): The method to redact personal information. Either 'Local' (spacy model), or 'AWS Comprehend' (AWS Comprehend API).
    - comprehend_query_number (int, optional): A counter tracking the number of queries to AWS Comprehend.
    - comprehend_client (optional): A connection to the AWS Comprehend service via the boto3 package.
    - textract_client (optional): A connection to the AWS Textract service via the boto3 package.
    - custom_recogniser_word_list (optional): A list of custom words that the user has chosen specifically to redact.
    - redact_whole_page_list (optional, List[str]): A list of pages to fully redact.
    - page_break_val (int, optional): The value at which to trigger a page break. Defaults to 3.
    - log_files_output_paths (List, optional): List of file paths used for saving redaction process logging results.
    - max_time (int, optional): The maximum amount of time (s) that the function should be running before it breaks. To avoid timeout errors with some APIs.      
    - progress (Progress, optional): A progress tracker for the redaction process. Defaults to a Progress object with track_tqdm set to True.

    The function returns a fully or partially-redacted PDF document.
    '''
    file_name = get_file_path_end(file_path)
    fill = (0, 0, 0)   # Fill colour for redactions
    comprehend_query_number_new = 0

    # Update custom word list analyser object with any new words that have been added to the custom deny list
    #print("custom_recogniser_word_list:", custom_recogniser_word_list)
    if custom_recogniser_word_list:        
        nlp_analyser.registry.remove_recognizer("CUSTOM")
        new_custom_recogniser = custom_word_list_recogniser(custom_recogniser_word_list)
        #print("new_custom_recogniser:", new_custom_recogniser)
        nlp_analyser.registry.add_recognizer(new_custom_recogniser)

        # List all elements currently in the nlp_analyser registry
        #print("Current recognizers in nlp_analyser registry:")
        for recognizer_name in nlp_analyser.registry.recognizers:
            print(recognizer_name)

        


    image_analyser = CustomImageAnalyzerEngine(nlp_analyser)
    

    if pii_identification_method == "AWS Comprehend" and comprehend_client == "":
        print("Connection to AWS Comprehend service unsuccessful.")

        return pymupdf_doc, all_decision_process_table, log_files_output_paths, request_metadata, annotations_all_pages, current_loop_page, page_break_return, all_line_level_ocr_results_df, comprehend_query_number
    
    if analysis_type == textract_option and textract_client == "":
        print("Connection to AWS Textract service unsuccessful.")

        return pymupdf_doc, all_decision_process_table, log_files_output_paths, request_metadata, annotations_all_pages, current_loop_page, page_break_return, all_line_level_ocr_results_df, comprehend_query_number

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
    #print("Current_loop_page:", current_loop_page)
    
    if analysis_type == tesseract_ocr_option: ocr_results_file_path = output_folder + "ocr_results_" + file_name + "_pages_" + str(page_min + 1) + "_" + str(page_max) + ".csv"
    elif analysis_type == textract_option: ocr_results_file_path = output_folder + "ocr_results_" + file_name + "_pages_" + str(page_min + 1) + "_" + str(page_max) + "_textract.csv"    
    
    # If running Textract, check if file already exists. If it does, load in existing data
    # Import results from json and convert
    if analysis_type == textract_option:
                
        json_file_path = output_folder + file_name + "_textract.json"
        log_files_output_paths.append(json_file_path)
        
        if not os.path.exists(json_file_path):
            no_textract_file = True
            print("No existing Textract results file found.")
            existing_data = {}
            #text_blocks, new_request_metadata = analyse_page_with_textract(pdf_page_as_bytes, reported_page_number, textract_client, handwrite_signature_checkbox)  # Analyse page with Textract
            #log_files_output_paths.append(json_file_path)
            #request_metadata = request_metadata + "\n" + new_request_metadata
            #wrapped_text_blocks = {"pages":[text_blocks]}
        else:
            # Open the file and load the JSON data
            no_textract_file = False
            print("Found existing Textract json results file.")
            with open(json_file_path, 'r') as json_file:
                existing_data = json.load(json_file)

    ###

    if current_loop_page == 0: page_loop_start = 0
    else: page_loop_start = current_loop_page

    progress_bar = tqdm(range(page_loop_start, number_of_pages), unit="pages remaining", desc="Redacting pages")

    for page_no in progress_bar:

        handwriting_or_signature_boxes = []
        signature_recogniser_results = []
        handwriting_recogniser_results = []
        page_break_return = False

        reported_page_number = str(page_no + 1)
        #print("Redacting page:", reported_page_number)
        
        # Assuming prepared_pdf_file_paths[page_no] is a PIL image object
        try:
            image = prepared_pdf_file_paths[page_no]#.copy()
            #print("image:", image)
        except Exception as e:
            print("Could not redact page:", reported_page_number, "due to:", e)    
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
            if language == 'en': ocr_lang = 'eng'
            else: ocr_lang = language

            # Step 1: Perform OCR. Either with Tesseract, or with AWS Textract
            if analysis_type == tesseract_ocr_option:
                
                word_level_ocr_results = image_analyser.perform_ocr(image)

                # Combine OCR results
                line_level_ocr_results, line_level_ocr_results_with_children = combine_ocr_results(word_level_ocr_results)

    
            # Import results from json and convert
            if analysis_type == textract_option:
                
                # Convert the image to bytes using an in-memory buffer
                image_buffer = io.BytesIO()
                image.save(image_buffer, format='PNG')  # Save as PNG, or adjust format if needed
                pdf_page_as_bytes = image_buffer.getvalue()

                if not existing_data:
                    text_blocks, new_request_metadata = analyse_page_with_textract(pdf_page_as_bytes, reported_page_number, textract_client, handwrite_signature_checkbox)  # Analyse page with Textract
                    log_files_output_paths.append(json_file_path)
                    request_metadata = request_metadata + "\n" + new_request_metadata

                    existing_data = {"pages":[text_blocks]}

                else:
                    # Check if the current reported_page_number exists in the loaded JSON
                    page_exists = any(page['page_no'] == reported_page_number for page in existing_data.get("pages", []))

                    if not page_exists:  # If the page does not exist, analyze again
                        print(f"Page number {reported_page_number} not found in existing Textract data. Analysing.")
                        text_blocks, new_request_metadata = analyse_page_with_textract(pdf_page_as_bytes, reported_page_number, textract_client, handwrite_signature_checkbox)  # Analyse page with Textract

                        # Check if "pages" key exists, if not, initialize it as an empty list
                        if "pages" not in existing_data:
                            existing_data["pages"] = []

                        # Append the new page data
                        existing_data["pages"].append(text_blocks)
                        
                        request_metadata = request_metadata + "\n" + new_request_metadata
                    else:
                        # If the page exists, retrieve the data
                        text_blocks = next(page['data'] for page in existing_data["pages"] if page['page_no'] == reported_page_number)
                
                
                # if not os.path.exists(json_file_path):
                #     text_blocks, new_request_metadata = analyse_page_with_textract(pdf_page_as_bytes, reported_page_number, textract_client, handwrite_signature_checkbox)  # Analyse page with Textract
                #     log_files_output_paths.append(json_file_path)
                #     request_metadata = request_metadata + "\n" + new_request_metadata

                #     existing_data = {"pages":[text_blocks]}


                # else:
                #     # Open the file and load the JSON data
                #     print("Found existing Textract json results file.")
                #     with open(json_file_path, 'r') as json_file:
                #         existing_data = json.load(json_file)

                #         # Check if the current reported_page_number exists in the loaded JSON
                #         page_exists = any(page['page_no'] == reported_page_number for page in existing_data.get("pages", []))

                #         if not page_exists:  # If the page does not exist, analyze again
                #             print(f"Page number {reported_page_number} not found in existing data. Analyzing again.")
                #             text_blocks, new_request_metadata = analyse_page_with_textract(pdf_page_as_bytes, reported_page_number, textract_client, handwrite_signature_checkbox)  # Analyse page with Textract

                #             # Check if "pages" key exists, if not, initialize it as an empty list
                #             if "pages" not in existing_data:
                #                 existing_data["pages"] = []

                #             # Append the new page data
                #             existing_data["pages"].append(text_blocks)

                #             # Write the updated existing_data back to the JSON file
                #             with open(json_file_path, 'w') as json_file:
                #                 json.dump(existing_data, json_file, indent=4)  # indent=4 makes the JSON file pretty-printed

                #             log_files_output_paths.append(json_file_path)
                #             request_metadata = request_metadata + "\n" + new_request_metadata
                #         else:
                #             # If the page exists, retrieve the data
                #             text_blocks = next(page['data'] for page in existing_data["pages"] if page['page_no'] == reported_page_number)

                line_level_ocr_results, handwriting_or_signature_boxes, signature_recogniser_results, handwriting_recogniser_results, line_level_ocr_results_with_children = json_to_ocrresult(text_blocks, page_width, page_height, reported_page_number)

            # Step 2: Analyze text and identify PII
            if chosen_redact_entities:

                redaction_bboxes, comprehend_query_number_new = image_analyser.analyze_text(
                    line_level_ocr_results,
                    line_level_ocr_results_with_children,
                    chosen_redact_comprehend_entities = chosen_redact_comprehend_entities,
                    pii_identification_method = pii_identification_method,
                    comprehend_client=comprehend_client,                 
                    language=language,
                    entities=chosen_redact_entities,
                    allow_list=allow_list,
                    score_threshold=score_threshold
                )                

                comprehend_query_number = comprehend_query_number + comprehend_query_number_new
                
            else:
                redaction_bboxes = []
                

            if analysis_type == tesseract_ocr_option: interim_results_file_path = output_folder + "interim_analyser_bboxes_" + file_name + "_pages_" + str(page_min + 1) + "_" + str(page_max) + ".txt"
            elif analysis_type == textract_option: interim_results_file_path = output_folder + "interim_analyser_bboxes_" + file_name + "_pages_" + str(page_min + 1) + "_" + str(page_max) + "_textract.txt" 

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
                #print("redact_whole_page_list:", redact_whole_page_list)
                if redact_whole_page_list:
                    if current_loop_page in redact_whole_page_list: redact_whole_page = True
                    else: redact_whole_page = False
                else: redact_whole_page = False

                pymupdf_page, image_annotations = redact_page_with_pymupdf(pymupdf_page, merged_redaction_bboxes, image, redact_whole_page=redact_whole_page)

            # Convert decision process to table
            decision_process_table = pd.DataFrame([{
                'text': result.text,
                'xmin': result.left,
                'ymin': result.top,
                'xmax': result.left + result.width,
                'ymax': result.top + result.height, 
                'label': result.entity_type,
                'start': result.start,
                'end': result.end,
                'score': result.score,
                'page': reported_page_number
                
            } for result in merged_redaction_bboxes]) #'left': result.left,
                #'top': result.top,
                #'width': result.width,
                #'height': result.height,

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

                # Check if the image already exists in annotations_all_pages
                #print("annotations_all_pages:", annotations_all_pages)
                existing_index = next((index for index, ann in enumerate(annotations_all_pages) if ann["image"] == image_annotations["image"]), None)
                if existing_index is not None:
                    # Replace the existing annotation
                    annotations_all_pages[existing_index] = image_annotations
                else:
                    # Append new annotation if it doesn't exist
                    annotations_all_pages.append(image_annotations)

                if analysis_type == textract_option:
                    # Write the updated existing textract data back to the JSON file
                    with open(json_file_path, 'w') as json_file:
                        json.dump(existing_data, json_file, indent=4)  # indent=4 makes the JSON file pretty-printed

                current_loop_page += 1

                return pymupdf_doc, all_decision_process_table, log_files_output_paths, request_metadata, annotations_all_pages, current_loop_page, page_break_return, all_line_level_ocr_results_df, comprehend_query_number

        if is_pdf(file_path) == False:
            images.append(image)
            pymupdf_doc = images

        # Check if the image already exists in annotations_all_pages
        #print("annotations_all_pages:", annotations_all_pages)
        existing_index = next((index for index, ann in enumerate(annotations_all_pages) if ann["image"] == image_annotations["image"]), None)
        if existing_index is not None:
            # Replace the existing annotation
            annotations_all_pages[existing_index] = image_annotations
        else:
            # Append new annotation if it doesn't exist
            annotations_all_pages.append(image_annotations)

        current_loop_page += 1

        # Break if new page is a multiple of chosen page_break_val
        if current_loop_page % page_break_val == 0:
            page_break_return = True
            progress.close(_tqdm=progress_bar)
            tqdm._instances.clear()

            if analysis_type == textract_option:
                # Write the updated existing textract data back to the JSON file
                with open(json_file_path, 'w') as json_file:
                    json.dump(existing_data, json_file, indent=4)  # indent=4 makes the JSON file pretty-printed

            return pymupdf_doc, all_decision_process_table, log_files_output_paths, request_metadata, annotations_all_pages, current_loop_page, page_break_return, all_line_level_ocr_results_df, comprehend_query_number
        
    if analysis_type == textract_option:
        # Write the updated existing textract data back to the JSON file
        
        with open(json_file_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)  # indent=4 makes the JSON file pretty-printed

    return pymupdf_doc, all_decision_process_table, log_files_output_paths, request_metadata, annotations_all_pages, current_loop_page, page_break_return, all_line_level_ocr_results_df, comprehend_query_number


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

def merge_text_bounding_boxes(analyser_results, characters: List[LTChar], combine_pixel_dist: int = 20, vertical_padding: int = 0):
    '''
    Merge identified bounding boxes containing PII that are very close to one another
    '''
    analysed_bounding_boxes = []
    original_bounding_boxes = []  # List to hold original bounding boxes

    if len(analyser_results) > 0 and len(characters) > 0:
        # Extract bounding box coordinates for sorting
        bounding_boxes = []
        for result in analyser_results:
            #print("Result:", result)
            char_boxes = [char.bbox for char in characters[result.start:result.end] if isinstance(char, LTChar)]
            char_text = [char._text for char in characters[result.start:result.end] if isinstance(char, LTChar)]
            if char_boxes:
                # Calculate the bounding box that encompasses all characters
                left = min(box[0] for box in char_boxes)
                bottom = min(box[1] for box in char_boxes)
                right = max(box[2] for box in char_boxes)
                top = max(box[3] for box in char_boxes) + vertical_padding
                bbox = [left, bottom, right, top]
                bounding_boxes.append((bottom, left, result, bbox, char_text))  # (y, x, result, bbox, text)

                # Store original bounding boxes
                original_bounding_boxes.append({"text": "".join(char_text), "boundingBox": bbox, "result": copy.deepcopy(result)})
                #print("Original bounding boxes:", original_bounding_boxes)

        # Sort the results by y-coordinate and then by x-coordinate
        bounding_boxes.sort()

        merged_bounding_boxes = []
        current_box = None
        current_y = None
        current_result = None
        current_text = []

        for y, x, result, next_box, text in bounding_boxes:
            if current_y is None or current_box is None:
                # Initialize the first bounding box
                current_box = next_box
                current_y = next_box[1]
                current_result = result
                current_text = list(text)
            else:
                vertical_diff_bboxes = abs(next_box[1] - current_y)
                horizontal_diff_bboxes = abs(next_box[0] - current_box[2])

                if vertical_diff_bboxes <= 5 and horizontal_diff_bboxes <= combine_pixel_dist:
                    # Merge bounding boxes
                    #print("Merging boxes")
                    merged_box = current_box.copy()
                    merged_result = current_result
                    merged_text = current_text.copy()

                    #print("current_box_max_x:", current_box[2])
                    #print("char_max_x:", next_box[2])

                    merged_box[2] = next_box[2]  # Extend horizontally
                    merged_box[3] = max(current_box[3], next_box[3])  # Adjust the top
                    merged_result.end = max(current_result.end, result.end)  # Extend text range
                    try:
                        merged_result.entity_type = current_result.entity_type + " - " + result.entity_type
                    except Exception as e:
                        print("Unable to combine result entity types:", e)
                    if current_text:
                        merged_text.append(" ")  # Add space between texts
                    merged_text.extend(text)

                    merged_bounding_boxes.append({
                        "text": "".join(merged_text),
                        "boundingBox": merged_box,
                        "result": merged_result
                    })

                else:
                    # Save the current merged box before starting a new one
                    # merged_bounding_boxes.append({
                    #     "text": "".join(current_text),
                    #     "boundingBox": current_box,
                    #     "result": current_result
                    # })
                    # Start a new bounding box
                    current_box = next_box
                    current_y = next_box[1]
                    current_result = result
                    current_text = list(text)

        # Handle the last box
        # if current_box is not None:
        #     merged_bounding_boxes.append({
        #         "text": "".join(current_text),
        #         "boundingBox": current_box,
        #         "result": current_result
        #     })

        # Combine original and merged bounding boxes
        analysed_bounding_boxes.extend(original_bounding_boxes)
        analysed_bounding_boxes.extend(merged_bounding_boxes)

        #print("Analysed bounding boxes:", analysed_bounding_boxes)

    return analysed_bounding_boxes

def create_text_redaction_process_results(analyser_results, analysed_bounding_boxes, page_num):
    decision_process_table = pd.DataFrame()

    if len(analyser_results) > 0:
        # Create summary df of annotations to be made
        analysed_bounding_boxes_df_new = pd.DataFrame(analysed_bounding_boxes)

        # Remove brackets and split the string into four separate columns
        #print("analysed_bounding_boxes_df_new:", analysed_bounding_boxes_df_new['boundingBox'])
        # analysed_bounding_boxes_df_new[['xmin', 'ymin', 'xmax', 'ymax']] = analysed_bounding_boxes_df_new['boundingBox'].str.strip('[]').str.split(',', expand=True)

        # Split the boundingBox list into four separate columns
        analysed_bounding_boxes_df_new[['xmin', 'ymin', 'xmax', 'ymax']] = analysed_bounding_boxes_df_new['boundingBox'].apply(pd.Series)

        # Convert the new columns to integers (if needed)
        analysed_bounding_boxes_df_new[['xmin', 'ymin', 'xmax', 'ymax']] = analysed_bounding_boxes_df_new[['xmin', 'ymin', 'xmax', 'ymax']].astype(float)

        analysed_bounding_boxes_df_text = analysed_bounding_boxes_df_new['result'].astype(str).str.split(",",expand=True).replace(".*: ", "", regex=True)
        analysed_bounding_boxes_df_text.columns = ["label", "start", "end", "score"]
        analysed_bounding_boxes_df_new = pd.concat([analysed_bounding_boxes_df_new, analysed_bounding_boxes_df_text], axis = 1)
        analysed_bounding_boxes_df_new['page'] = page_num + 1
        decision_process_table = pd.concat([decision_process_table, analysed_bounding_boxes_df_new], axis = 0).drop('result', axis=1)

        #print('\n\ndecision_process_table:\n\n', decision_process_table)
    
    return decision_process_table

def create_pikepdf_annotations_for_bounding_boxes(analysed_bounding_boxes):
    pikepdf_annotations_on_page = []
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
        pikepdf_annotations_on_page.append(annotation)
    return pikepdf_annotations_on_page

def redact_text_pdf(
    filename: str,  # Path to the PDF file to be redacted
    prepared_pdf_image_path: str,  # Path to the prepared PDF image for redaction
    language: str,  # Language of the PDF content
    chosen_redact_entities: List[str],  # List of entities to be redacted
    chosen_redact_comprehend_entities: List[str],
    allow_list: List[str] = None,  # Optional list of allowed entities
    page_min: int = 0,  # Minimum page number to start redaction
    page_max: int = 999,  # Maximum page number to end redaction
    analysis_type: str = text_ocr_option,  # Type of analysis to perform
    current_loop_page: int = 0,  # Current page being processed in the loop
    page_break_return: bool = False,  # Flag to indicate if a page break should be returned
    annotations_all_pages: List = [],  # List of annotations across all pages
    all_line_level_ocr_results_df: pd.DataFrame = pd.DataFrame(),  # DataFrame for OCR results
    all_decision_process_table: pd.DataFrame = pd.DataFrame(),  # DataFrame for decision process table
    pymupdf_doc: List = [],  # List of PyMuPDF documents
    pii_identification_method: str = "Local",
    comprehend_query_number:int = 0,
    comprehend_client="",
    custom_recogniser_word_list:List[str]=[],
    redact_whole_page_list:List[str]=[],
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
    - comprehend_query_number (int, optional): A counter tracking the number of queries to AWS Comprehend.
    - comprehend_client (optional): A connection to the AWS Comprehend service via the boto3 package.
    - custom_recogniser_word_list (optional, List[str]): A list of custom words that the user has chosen specifically to redact.
    - redact_whole_page_list (optional, List[str]): A list of pages to fully redact.
    - page_break_val: Value for page break
    - max_time (int, optional): The maximum amount of time (s) that the function should be running before it breaks. To avoid timeout errors with some APIs.     
    - progress: Progress tracking object
    '''

    if pii_identification_method == "AWS Comprehend" and comprehend_client == "":
        print("Connection to AWS Comprehend service not found.")

        return pymupdf_doc, all_decision_process_table, all_line_level_ocr_results_df, annotations_all_pages, current_loop_page, page_break_return, comprehend_query_number
    
    # Update custom word list analyser object with any new words that have been added to the custom deny list
    #print("custom_recogniser_word_list:", custom_recogniser_word_list)
    if custom_recogniser_word_list:        
        nlp_analyser.registry.remove_recognizer("CUSTOM")
        new_custom_recogniser = custom_word_list_recogniser(custom_recogniser_word_list)
        #print("new_custom_recogniser:", new_custom_recogniser)
        nlp_analyser.registry.add_recognizer(new_custom_recogniser)

        # List all elements currently in the nlp_analyser registry
        #print("Current recognizers in nlp_analyser registry:")
        #for recognizer_name in nlp_analyser.registry.recognizers:
        #    print(recognizer_name)

        #print("Custom recogniser:", nlp_analyser.registry.)

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

            if isinstance(image, str):
                image_path = image
                image = Image.open(image_path)

            for page_layout in extract_pages(filename, page_numbers = [page_no], maxpages=1):
                
                page_analyser_results = []
                page_analysed_bounding_boxes = []            
                
                characters = []
                pikepdf_annotations_on_page = []
                decision_process_table_on_page = pd.DataFrame()    
                page_text_outputs = pd.DataFrame()  

                if analysis_type == text_ocr_option:
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

                        # Initialize batching variables
                        current_batch = ""
                        current_batch_mapping = []  # List of (start_pos, line_index, OCRResult) tuples
                        all_text_line_results = []  # Store results for all lines

                        # First pass: collect all lines into batches
                        for i, text_line in enumerate(line_level_text_results_list):
                            if chosen_redact_entities:
                                if pii_identification_method == "Local":

                                    #print("chosen_redact_entities:", chosen_redact_entities)

                                    # Process immediately for local analysis
                                    text_line_analyser_result = nlp_analyser.analyze(
                                        text=text_line.text,
                                        language=language,
                                        entities=chosen_redact_entities,
                                        score_threshold=score_threshold,
                                        return_decision_process=True,
                                        allow_list=allow_list
                                    )
                                    all_text_line_results.append((i, text_line_analyser_result))

                                                                    
                                elif pii_identification_method == "AWS Comprehend":

                                    # First use the local Spacy model to pick up custom entities that AWS Comprehend can't search for.
                                    custom_redact_entities = [entity for entity in chosen_redact_comprehend_entities if entity in custom_entities]


                                    text_line_analyser_result = nlp_analyser.analyze(
                                        text=text_line.text,
                                        language=language,
                                        entities=custom_redact_entities,
                                        score_threshold=score_threshold,
                                        return_decision_process=True,
                                        allow_list=allow_list
                                    )
                                    all_text_line_results.append((i, text_line_analyser_result))


                                    if len(text_line.text) >= 3:
                                        # Add separator between lines
                                        if current_batch:
                                            current_batch += " | "
                                        
                                        start_pos = len(current_batch)
                                        current_batch += text_line.text
                                        current_batch_mapping.append((start_pos, i, text_line))

                                        # Process batch if approaching 300 characters or last line
                                        if len(current_batch) >= 200 or i == len(line_level_text_results_list) - 1:
                                            print("length of text for Comprehend:", len(current_batch))
                                            
                                            try:
                                                response = comprehend_client.detect_pii_entities(
                                                    Text=current_batch,
                                                    LanguageCode=language
                                                )
                                            except Exception as e:
                                                print(e)
                                                time.sleep(3)
                                                response = comprehend_client.detect_pii_entities(
                                                    Text=current_batch,
                                                    LanguageCode=language
                                                )

                                            comprehend_query_number += 1

                                            # Process response and map back to original lines
                                            if response and "Entities" in response:
                                                for entity in response["Entities"]:
                                                    entity_start = entity["BeginOffset"]
                                                    entity_end = entity["EndOffset"]

                                                    # Find which line this entity belongs to
                                                    for batch_start, line_idx, original_line in current_batch_mapping:
                                                        batch_end = batch_start + len(original_line.text)

                                                        # Check if entity belongs to this line
                                                        if batch_start <= entity_start < batch_end:
                                                            # Adjust offsets relative to original line
                                                            relative_start = entity_start - batch_start
                                                            relative_end = min(entity_end - batch_start, len(original_line.text))
                                                            
                                                            result_text = original_line.text[relative_start:relative_end]

                                                            if result_text not in allow_list:
                                                                if entity.get("Type") in chosen_redact_comprehend_entities:
                                                                    # Create adjusted entity
                                                                    adjusted_entity = entity.copy()
                                                                    adjusted_entity["BeginOffset"] = relative_start
                                                                    adjusted_entity["EndOffset"] = relative_end

                                                                    recogniser_entity = recognizer_result_from_dict(adjusted_entity)
                                                                    
                                                                    # Add to results for this line
                                                                    existing_results = next((results for idx, results in all_text_line_results if idx == line_idx), [])
                                                                    if not existing_results:
                                                                        all_text_line_results.append((line_idx, [recogniser_entity]))
                                                                    else:
                                                                        existing_results.append(recogniser_entity)

                                            # Reset batch
                                            current_batch = ""
                                            current_batch_mapping = []

                        # Second pass: process results for each line
                        for i, text_line in enumerate(line_level_text_results_list):
                            text_line_analyser_result = []
                            text_line_bounding_boxes = []

                            # Get results for this line
                            line_results = next((results for idx, results in all_text_line_results if idx == i), [])
                            
                            if line_results:
                                text_line_analyser_result = line_results

                                #print("Analysed text container, now merging bounding boxes")

                                # Merge bounding boxes if very close together
                                text_line_bounding_boxes = merge_text_bounding_boxes(text_line_analyser_result, line_characters[i])

                                #print("merged bounding boxes")

                                text_container_analyser_results.extend(text_line_analyser_result)
                                text_container_analysed_bounding_boxes.extend(text_line_bounding_boxes)

                                #print("text_container_analyser_results:", text_container_analyser_results)

                                page_analyser_results.extend(text_container_analyser_results)  # Add this line
                                page_analysed_bounding_boxes.extend(text_line_bounding_boxes)  # Add this line


                #print("page_analyser_results:", page_analyser_results)
                #print("page_analysed_bounding_boxes:", page_analysed_bounding_boxes)
                #print("image:", image)

                page_analysed_bounding_boxes = convert_pikepdf_decision_output_to_image_coords(pymupdf_page, page_analysed_bounding_boxes, image)

                #print("page_analysed_bounding_boxes_out_converted:", page_analysed_bounding_boxes)

                # Annotate redactions on page
                pikepdf_annotations_on_page = create_pikepdf_annotations_for_bounding_boxes(page_analysed_bounding_boxes)

                #print("pikepdf_annotations_on_page:", pikepdf_annotations_on_page)

                # Make pymupdf page redactions
                #print("redact_whole_page_list:", redact_whole_page_list)
                if redact_whole_page_list:                    
                    if current_loop_page in redact_whole_page_list: redact_whole_page = True
                    else: redact_whole_page = False
                else: redact_whole_page = False

                pymupdf_page, image_annotations = redact_page_with_pymupdf(pymupdf_page, pikepdf_annotations_on_page, image, redact_whole_page=redact_whole_page, convert_coords=False)

                #print("image_annotations:", image_annotations)

                #print("Did redact_page_with_pymupdf function")
                reported_page_no = page_no + 1
                print("For page number:", reported_page_no, "there are", len(image_annotations["boxes"]), "annotations")

                # Write logs
                # Create decision process table
                decision_process_table_on_page = create_text_redaction_process_results(page_analyser_results, page_analysed_bounding_boxes, current_loop_page)     

                if not decision_process_table_on_page.empty:
                    all_decision_process_table = pd.concat([all_decision_process_table, decision_process_table_on_page])
                    #print("all_decision_process_table:", all_decision_process_table)

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

                    # Check if the image already exists in annotations_all_pages
                    existing_index = next((index for index, ann in enumerate(annotations_all_pages) if ann["image"] == image_annotations["image"]), None)
                    if existing_index is not None:
                        # Replace the existing annotation
                        annotations_all_pages[existing_index] = image_annotations
                    else:
                        # Append new annotation if it doesn't exist
                        annotations_all_pages.append(image_annotations)

                    current_loop_page += 1

                    return pymupdf_doc, all_decision_process_table, all_line_level_ocr_results_df, annotations_all_pages, current_loop_page, page_break_return, comprehend_query_number
                

        # Check if the image already exists in annotations_all_pages
        existing_index = next((index for index, ann in enumerate(annotations_all_pages) if ann["image"] == image_annotations["image"]), None)
        if existing_index is not None:
            # Replace the existing annotation
            annotations_all_pages[existing_index] = image_annotations
        else:
            # Append new annotation if it doesn't exist
            annotations_all_pages.append(image_annotations)

        current_loop_page += 1

        # Break if new page is a multiple of 10
        if current_loop_page % page_break_val == 0:
            page_break_return = True
            progress.close(_tqdm=progress_bar)

            return pymupdf_doc, all_decision_process_table, all_line_level_ocr_results_df, annotations_all_pages, current_loop_page, page_break_return, comprehend_query_number
        
            
    return pymupdf_doc, all_decision_process_table, all_line_level_ocr_results_df, annotations_all_pages, current_loop_page, page_break_return, comprehend_query_number