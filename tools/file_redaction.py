import time
import re
import json
import io
import os
from PIL import Image, ImageChops, ImageDraw
from typing import List, Dict
import pandas as pd

#from presidio_image_redactor.entities import ImageRecognizerResult
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine #, LTAnno
from pikepdf import Pdf, Dictionary, Name
import gradio as gr
from gradio import Progress

from collections import defaultdict  # For efficient grouping

from tools.custom_image_analyser_engine import CustomImageAnalyzerEngine, OCRResult, combine_ocr_results, CustomImageRecognizerResult
from tools.file_conversion import process_file
from tools.load_spacy_model_custom_recognisers import nlp_analyser, score_threshold
from tools.helper_functions import get_file_path_end, output_folder
from tools.file_conversion import process_file, is_pdf, convert_text_pdf_to_img_pdf, is_pdf_or_image
from tools.data_anonymise import generate_decision_process_output
from tools.aws_textract import analyse_page_with_textract, convert_pike_pdf_page_to_bytes, json_to_ocrresult

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

def choose_and_run_redactor(file_paths:List[str], image_paths:List[str], language:str, chosen_redact_entities:List[str], in_redact_method:str, in_allow_list:List[List[str]]=None, latest_file_completed:int=0, out_message:list=[], out_file_paths:list=[], log_files_output_paths:list=[], first_loop_state:bool=False, page_min:int=0, page_max:int=999, estimated_time_taken_state:float=0.0, handwrite_signature_checkbox:List[str]=["Redact all identified handwriting", "Redact all identified signatures"], all_request_metadata_str:str = "", progress=gr.Progress(track_tqdm=True)):
    '''
    Based on the type of redaction selected, pass the document file content onto the relevant function and return a redacted document plus processing logs.
    '''

    tic = time.perf_counter()
    all_request_metadata = all_request_metadata_str.split('\n') if all_request_metadata_str else []

    # If this is the first time around, set variables to 0/blank
    if first_loop_state==True:
        latest_file_completed = 0
        #out_message = []
        out_file_paths = []

    # If out message is string or out_file_paths are blank, change to a list so it can be appended to
    if isinstance(out_message, str):
        out_message = [out_message]

    if not out_file_paths:
        out_file_paths = []

    latest_file_completed = int(latest_file_completed)

    # If we have already redacted the last file, return the input out_message and file list to the relevant components
    if latest_file_completed >= len(file_paths):
        print("Last file reached")
        # Set to a very high number so as not to mix up with subsequent file processing by the user
        latest_file_completed = 99
        final_out_message = '\n'.join(out_message)
        #final_out_message = final_out_message + "\n\nGo to to the Redaction settings tab to see redaction logs. Please give feedback on the results below to help improve this app."
        
        estimate_total_processing_time = sum_numbers_before_seconds(final_out_message)
        print("Estimated total processing time:", str(estimate_total_processing_time))

        return final_out_message, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths, estimate_total_processing_time, all_request_metadata_str
    
    file_paths_loop = [file_paths[int(latest_file_completed)]]

    if not in_allow_list.empty:
        in_allow_list_flat = in_allow_list[0].tolist()
        print("In allow list:", in_allow_list_flat)
    else:
        in_allow_list_flat = []
    
    for file in progress.tqdm(file_paths_loop, desc="Redacting files", unit = "files"):
        file_path = file.name

        if file_path:
            file_path_without_ext = get_file_path_end(file_path)
            is_a_pdf = is_pdf(file_path) == True
            if is_a_pdf == False:
                # If user has not submitted a pdf, assume it's an image
                print("File is not a pdf, assuming that image analysis needs to be used.")
                in_redact_method = "Quick image analysis - typed text"
        else:
            out_message = "No file selected"
            print(out_message)
            return out_message, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str

        if in_redact_method == "Quick image analysis - typed text" or in_redact_method == "Complex image analysis - AWS Textract, handwriting/signatures":
            #Analyse and redact image-based pdf or image
            if is_pdf_or_image(file_path) == False:
                out_message = "Please upload a PDF file or image file (JPG, PNG) for image analysis."
                return out_message, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str

            print("Redacting file " + file_path_without_ext + " as an image-based file")
            pdf_images, output_logs, logging_file_paths, new_request_metadata = redact_image_pdf(file_path, image_paths, language, chosen_redact_entities, in_allow_list_flat, is_a_pdf, page_min, page_max, in_redact_method, handwrite_signature_checkbox)

            # Save file
            out_image_file_path = output_folder + file_path_without_ext + "_redacted_as_img.pdf"
            pdf_images[0].save(out_image_file_path, "PDF" ,resolution=100.0, save_all=True, append_images=pdf_images[1:])

            out_file_paths.append(out_image_file_path)
            if logging_file_paths:
                log_files_output_paths.extend(logging_file_paths)

            out_message.append("File '" + file_path_without_ext + "' successfully redacted")

            # Save decision making process
            output_logs_str = str(output_logs)
            logs_output_file_name = out_image_file_path + "_decision_process_output.txt"
            with open(logs_output_file_name, "w") as f:
                f.write(output_logs_str)
            log_files_output_paths.append(logs_output_file_name)

           # Save Textract request metadata (if exists)
            if new_request_metadata:
                print("Request metadata:", new_request_metadata)
                all_request_metadata.append(new_request_metadata)

            # Increase latest file completed count unless we are at the last file
            if latest_file_completed != len(file_paths):
                print("Completed file number:", str(latest_file_completed))
                latest_file_completed += 1                

        elif in_redact_method == "Simple text analysis - PDFs with selectable text":
            
            if is_pdf(file_path) == False:
                return "Please upload a PDF file for text analysis. If you have an image, select 'Image analysis'.", None, None
            
            # Analyse text-based pdf
            print('Redacting file as text-based PDF')
import time
import re
import json
import io
import os
from PIL import Image, ImageChops, ImageDraw
from typing import List, Dict
import pandas as pd

#from presidio_image_redactor.entities import ImageRecognizerResult
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine #, LTAnno
from pikepdf import Pdf, Dictionary, Name
import gradio as gr
from gradio import Progress

from collections import defaultdict  # For efficient grouping

from tools.custom_image_analyser_engine import CustomImageAnalyzerEngine, OCRResult, combine_ocr_results, CustomImageRecognizerResult
from tools.file_conversion import process_file
from tools.load_spacy_model_custom_recognisers import nlp_analyser, score_threshold
from tools.helper_functions import get_file_path_end, output_folder
from tools.file_conversion import process_file, is_pdf, convert_text_pdf_to_img_pdf, is_pdf_or_image
from tools.data_anonymise import generate_decision_process_output
from tools.aws_textract import analyse_page_with_textract, convert_pike_pdf_page_to_bytes, json_to_ocrresult

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

def choose_and_run_redactor(file_paths:List[str], image_paths:List[str], language:str, chosen_redact_entities:List[str], in_redact_method:str, in_allow_list:List[List[str]]=None, latest_file_completed:int=0, out_message:list=[], out_file_paths:list=[], log_files_output_paths:list=[], first_loop_state:bool=False, page_min:int=0, page_max:int=999, estimated_time_taken_state:float=0.0, handwrite_signature_checkbox:List[str]=["Redact all identified handwriting", "Redact all identified signatures"], all_request_metadata_str:str = "", progress=gr.Progress(track_tqdm=True)):
    '''
    Based on the type of redaction selected, pass the document file content onto the relevant function and return a redacted document plus processing logs.
    '''

    tic = time.perf_counter()
    all_request_metadata = all_request_metadata_str.split('\n') if all_request_metadata_str else []

    # If this is the first time around, set variables to 0/blank
    if first_loop_state==True:
        latest_file_completed = 0
        #out_message = []
        out_file_paths = []

    # If out message is string or out_file_paths are blank, change to a list so it can be appended to
    if isinstance(out_message, str):
        out_message = [out_message]

    if not out_file_paths:
        out_file_paths = []

    latest_file_completed = int(latest_file_completed)

    # If we have already redacted the last file, return the input out_message and file list to the relevant components
    if latest_file_completed >= len(file_paths):
        print("Last file reached")
        # Set to a very high number so as not to mix up with subsequent file processing by the user
        latest_file_completed = 99
        final_out_message = '\n'.join(out_message)
        #final_out_message = final_out_message + "\n\nGo to to the Redaction settings tab to see redaction logs. Please give feedback on the results below to help improve this app."
        
        estimate_total_processing_time = sum_numbers_before_seconds(final_out_message)
        print("Estimated total processing time:", str(estimate_total_processing_time))

        return final_out_message, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths, estimate_total_processing_time, all_request_metadata_str
    
    file_paths_loop = [file_paths[int(latest_file_completed)]]

    if not in_allow_list.empty:
        in_allow_list_flat = in_allow_list[0].tolist()
        print("In allow list:", in_allow_list_flat)
    else:
        in_allow_list_flat = []
    
    for file in progress.tqdm(file_paths_loop, desc="Redacting files", unit = "files"):
        file_path = file.name

        if file_path:
            file_path_without_ext = get_file_path_end(file_path)
            is_a_pdf = is_pdf(file_path) == True
            if is_a_pdf == False:
                # If user has not submitted a pdf, assume it's an image
                print("File is not a pdf, assuming that image analysis needs to be used.")
                in_redact_method = "Quick image analysis - typed text"
        else:
            out_message = "No file selected"
            print(out_message)
            return out_message, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str

        if in_redact_method == "Quick image analysis - typed text" or in_redact_method == "Complex image analysis - AWS Textract, handwriting/signatures":
            #Analyse and redact image-based pdf or image
            if is_pdf_or_image(file_path) == False:
                out_message = "Please upload a PDF file or image file (JPG, PNG) for image analysis."
                return out_message, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str

            print("Redacting file " + file_path_without_ext + " as an image-based file")
            pdf_images, output_logs, logging_file_paths, new_request_metadata = redact_image_pdf(file_path, image_paths, language, chosen_redact_entities, in_allow_list_flat, is_a_pdf, page_min, page_max, in_redact_method, handwrite_signature_checkbox)

            # Save file
            out_image_file_path = output_folder + file_path_without_ext + "_redacted_as_img.pdf"
            pdf_images[0].save(out_image_file_path, "PDF" ,resolution=100.0, save_all=True, append_images=pdf_images[1:])

            out_file_paths.append(out_image_file_path)
            if logging_file_paths:
                log_files_output_paths.extend(logging_file_paths)

            out_message.append("File '" + file_path_without_ext + "' successfully redacted")

            # Save decision making process
            output_logs_str = str(output_logs)
            logs_output_file_name = out_image_file_path + "_decision_process_output.txt"
            with open(logs_output_file_name, "w") as f:
                f.write(output_logs_str)
            log_files_output_paths.append(logs_output_file_name)

           # Save Textract request metadata (if exists)
            if new_request_metadata:
                print("Request metadata:", new_request_metadata)
                all_request_metadata.append(new_request_metadata)

            # Increase latest file completed count unless we are at the last file
            if latest_file_completed != len(file_paths):
                print("Completed file number:", str(latest_file_completed))
                latest_file_completed += 1                

        elif in_redact_method == "Simple text analysis - PDFs with selectable text":
            
            if is_pdf(file_path) == False:
                return "Please upload a PDF file for text analysis. If you have an image, select 'Image analysis'.", None, None
            
            # Analyse text-based pdf
            print('Redacting file as text-based PDF')
            pdf_text, output_logs = redact_text_pdf(file_path, language, chosen_redact_entities, in_allow_list_flat, page_min, page_max, "Simple text analysis - PDFs with selectable text")
            out_text_file_path = output_folder + file_path_without_ext + "_text_redacted.pdf"
            pdf_text.save(out_text_file_path)            

            # Convert message
            convert_message="Converting PDF to image-based PDF to embed redactions."
            print(convert_message)

            # Convert document to image-based document to 'embed' redactions
            img_output_summary, img_output_file_path = convert_text_pdf_to_img_pdf(file_path, [out_text_file_path])
            out_file_paths.extend(img_output_file_path)

            output_logs_str = str(output_logs)
            logs_output_file_name = img_output_file_path[0] + "_decision_process_output.txt"
            with open(logs_output_file_name, "w") as f:
                f.write(output_logs_str)
            log_files_output_paths.append(logs_output_file_name)

            out_message_new = "File '" + file_path_without_ext + "' successfully redacted"
            out_message.append(out_message_new)

            if latest_file_completed != len(file_paths):
                print("Completed file number:", str(latest_file_completed), "more files to do")
                latest_file_completed += 1
                            
        else:
            out_message = "No redaction method selected"
            print(out_message)
            return out_message, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str
    
    toc = time.perf_counter()
    out_time = f"in {toc - tic:0.1f} seconds."
    print(out_time)

    out_message_out = '\n'.join(out_message)
    out_message_out = out_message_out + " " + out_time

   # If textract requests made, write to logging file
    if all_request_metadata:
        all_request_metadata_str = '\n'.join(all_request_metadata)

        all_request_metadata_file_path = output_folder + file_path_without_ext + "_textract_request_metadata.txt"   

        with open(all_request_metadata_file_path, "w") as f:
            f.write(all_request_metadata_str)

        # Add the request metadata to the log outputs if not there already
        if all_request_metadata_file_path not in log_files_output_paths:
            log_files_output_paths.append(all_request_metadata_file_path)


    return out_message_out, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str



def bounding_boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    return (box1[0] < box2[2] and box2[0] < box1[2] and
            box1[1] < box2[3] and box2[1] < box1[3])

def merge_img_bboxes(bboxes, combined_results: Dict, signature_recogniser_results=[], handwriting_recogniser_results=[], handwrite_signature_checkbox: List[str]=["Redact all identified handwriting", "Redact all identified signatures"], horizontal_threshold=150, vertical_threshold=25):
    merged_bboxes = []
    grouped_bboxes = defaultdict(list)

    # Process signature and handwriting results
    if signature_recogniser_results or handwriting_recogniser_results:
        if "Redact all identified handwriting" in handwrite_signature_checkbox:
            print("Handwriting boxes exist at merge:", handwriting_recogniser_results)
            bboxes.extend(handwriting_recogniser_results)

        if "Redact all identified signatures" in handwrite_signature_checkbox:
            print("Signature boxes exist at merge:", signature_recogniser_results)
            bboxes.extend(signature_recogniser_results)

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
                        if current_char <= start_char < word_end or current_char < end_char <= word_end:
                            relevant_words.append(word)
                        if word_end >= end_char:
                            break
                        current_char = word_end  # +1 for space
                        if not word['text'].endswith(' '):
                            current_char += 1  # +1 for space if the word doesn't already end with a space

                    if relevant_words:
                        print("Relevant words:", relevant_words)
                        left = min(word['bounding_box'][0] for word in relevant_words)
                        top = min(word['bounding_box'][1] for word in relevant_words)
                        right = max(word['bounding_box'][2] for word in relevant_words)
                        bottom = max(word['bounding_box'][3] for word in relevant_words)
                        
                        # Combine the text of the relevant words
                        combined_text = " ".join(word['text'] for word in relevant_words)
                        
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

                new_left = min(merged_box.left, next_box.left)
                new_top = min(merged_box.top, next_box.top)
                new_width = max(merged_box.left + merged_box.width, next_box.left + next_box.width) - new_left
                new_height = max(merged_box.top + merged_box.height, next_box.top + next_box.height) - new_top
                merged_box = CustomImageRecognizerResult(
                    merged_box.entity_type, merged_box.start, merged_box.end, merged_box.score, new_left, new_top, new_width, new_height, new_text
                )
            else:
                merged_bboxes.append(merged_box)
                merged_box = next_box  

        merged_bboxes.append(merged_box) 

    return merged_bboxes

def redact_image_pdf(file_path:str, image_paths:List[str], language:str, chosen_redact_entities:List[str], allow_list:List[str]=None, is_a_pdf:bool=True, page_min:int=0, page_max:int=999, analysis_type:str="Quick image analysis - typed text", handwrite_signature_checkbox:List[str]=["Redact all identified handwriting", "Redact all identified signatures"], request_metadata:str="", progress=Progress(track_tqdm=True)):
    '''
    Take an path for an image of a document, then run this image through the Presidio ImageAnalyzer and PIL to get a redacted page back. Adapted from Presidio ImageRedactorEngine.
    '''
    # json_file_path is for AWS Textract outputs
    logging_file_paths = []
    file_name = get_file_path_end(file_path)
    fill = (0, 0, 0)   # Fill colour
    decision_process_output_str = ""
    images = []
    #request_metadata = {}
    image_analyser = CustomImageAnalyzerEngine(nlp_analyser)

    if not image_paths:
        out_message = "PDF does not exist as images. Converting pages to image"
        print(out_message)

        image_paths = process_file(file_path)

    if not isinstance(image_paths, list):
        print("Converting image_paths to list")
        image_paths = [image_paths]

    #print("Image paths:", image_paths)
    number_of_pages = len(image_paths[0])

    print("Number of pages:", str(number_of_pages))

    out_message = "Redacting pages"
    print(out_message)
    #progress(0.1, desc=out_message)

    # Check that page_min and page_max are within expected ranges
    if page_max > number_of_pages or page_max == 0:
        page_max = number_of_pages

    if page_min <= 0:
        page_min = 0
    else:
        page_min = page_min - 1

    print("Page range:", str(page_min + 1), "to", str(page_max))

    #for i in progress.tqdm(range(0,number_of_pages), total=number_of_pages, unit="pages", desc="Redacting pages"):

    all_ocr_results = []
    all_decision_process = []

    if analysis_type == "Quick image analysis - typed text": ocr_results_file_path = output_folder + "ocr_results_" + file_name + "_pages_" + str(page_min + 1) + "_" + str(page_max) + ".txt"
    elif analysis_type == "Complex image analysis - AWS Textract, handwriting/signatures": ocr_results_file_path = output_folder + "ocr_results_" + file_name + "_pages_" + str(page_min + 1) + "_" + str(page_max) + "_textract.txt"    
    
    for n in range(0, number_of_pages):
        handwriting_or_signature_boxes = []
        signature_recogniser_results = []
        handwriting_recogniser_results = []

        try:
            image = image_paths[0][n]#.copy()
            print("Skipping page", str(n))
            #print("image:", image)
        except Exception as e:
            print("Could not redact page:", str(n), "due to:")
            print(e)
            continue

        if n >= page_min and n < page_max:

            i = n

            reported_page_number = str(i + 1)

            print("Redacting page", reported_page_number)

            
            # Assuming image_paths[i] is your PIL image object
            try:
                image = image_paths[0][i]#.copy()
                #print("image:", image)
            except Exception as e:
                print("Could not redact page:", reported_page_number, "due to:")
                print(e)
                continue

            # Need image size to convert textract OCR outputs to the correct sizes
            page_width, page_height = image.size

            # Possibility to use different languages
            if language == 'en':
                ocr_lang = 'eng'
            else: ocr_lang = language

            # Step 1: Perform OCR. Either with Tesseract, or with AWS Textract
            if analysis_type == "Quick image analysis - typed text":
                
                ocr_results = image_analyser.perform_ocr(image)

                # Combine OCR results
                ocr_results, ocr_results_with_children = combine_ocr_results(ocr_results)

                # Save decision making process
                ocr_results_with_children_str = str(ocr_results_with_children)
                logs_output_file_name = output_folder + "ocr_with_children.txt"
                with open(logs_output_file_name, "w") as f:
                    f.write(ocr_results_with_children_str)
    
            # Import results from json and convert
            if analysis_type == "Complex image analysis - AWS Textract, handwriting/signatures":
                
                # Convert the image to bytes using an in-memory buffer
                image_buffer = io.BytesIO()
                image.save(image_buffer, format='PNG')  # Save as PNG, or adjust format if needed
                pdf_page_as_bytes = image_buffer.getvalue()
                
                json_file_path = output_folder + file_name + "_page_" + reported_page_number + "_textract.json"
                
                if not os.path.exists(json_file_path):
                    text_blocks, new_request_metadata = analyse_page_with_textract(pdf_page_as_bytes, json_file_path) # Analyse page with Textract
                    logging_file_paths.append(json_file_path)
                    request_metadata = request_metadata + "\n" + new_request_metadata
                else:
                    # Open the file and load the JSON data
                    print("Found existing Textract json results file for this page.")
                    with open(json_file_path, 'r') as json_file:
                        text_blocks = json.load(json_file)
                        text_blocks = text_blocks['Blocks']

                ocr_results, handwriting_or_signature_boxes, signature_recogniser_results, handwriting_recogniser_results, ocr_results_with_children = json_to_ocrresult(text_blocks, page_width, page_height)

                # Save decision making process
                ocr_results_with_children_str = str(ocr_results_with_children)
                logs_output_file_name = output_folder + "ocr_with_children_textract.txt"
                with open(logs_output_file_name, "w") as f:
                    f.write(ocr_results_with_children_str)

            # Step 2: Analyze text and identify PII
            bboxes = image_analyser.analyze_text(
                ocr_results,
                language=language,
                entities=chosen_redact_entities,
                allow_list=allow_list,
                score_threshold=score_threshold,
            )

            if analysis_type == "Quick image analysis - typed text": interim_results_file_path = output_folder + "interim_analyser_bboxes_" + file_name + "_pages_" + str(page_min + 1) + "_" + str(page_max) + ".txt"
            elif analysis_type == "Complex image analysis - AWS Textract, handwriting/signatures": interim_results_file_path = output_folder + "interim_analyser_bboxes_" + file_name + "_pages_" + str(page_min + 1) + "_" + str(page_max) + "_textract.txt" 

            # Save decision making process
            bboxes_str = str(bboxes)
            with open(interim_results_file_path, "w") as f:
                f.write(bboxes_str)

            # Merge close bounding boxes
            merged_bboxes = merge_img_bboxes(bboxes, ocr_results_with_children, signature_recogniser_results, handwriting_recogniser_results, handwrite_signature_checkbox)

            # Export the decision making process
            if merged_bboxes:
                for bbox in merged_bboxes:
                    print(f"Entity: {bbox.entity_type}, Text: {bbox.text}, Bbox: ({bbox.left}, {bbox.top}, {bbox.width}, {bbox.height})")

                
                decision_process_output_str = "Page " + reported_page_number + ":\n" + str(merged_bboxes)
                all_decision_process.append(decision_process_output_str)

            # 3. Draw the merged boxes
            draw = ImageDraw.Draw(image)

            for box in merged_bboxes:
                x0 = box.left
                y0 = box.top
                x1 = x0 + box.width
                y1 = y0 + box.height
                draw.rectangle([x0, y0, x1, y1], fill=fill)

            ocr_results_str = "Page:" + reported_page_number + "\n" + str(ocr_results)
            all_ocr_results.append(ocr_results_str) 

        images.append(image)

    # Write OCR results as a log file    
    ocr_results_out = "\n".join(all_ocr_results)
    with open(ocr_results_file_path, "w") as f:
        f.write(ocr_results_out)
    logging_file_paths.append(ocr_results_file_path)

    all_decision_process_str = "\n".join(all_decision_process)

    return images, all_decision_process_str, logging_file_paths, request_metadata

def analyze_text_container(text_container, language, chosen_redact_entities, score_threshold, allow_list):
    if isinstance(text_container, LTTextContainer):
        text_to_analyze = text_container.get_text()

        analyzer_results = nlp_analyser.analyze(text=text_to_analyze,
                                                language=language, 
                                                entities=chosen_redact_entities,
                                                score_threshold=score_threshold,
                                                return_decision_process=True,
                                                allow_list=allow_list)
        characters = [char
                for line in text_container
                if isinstance(line, LTTextLine)
                for char in line]
        
        return analyzer_results, characters
    return [], []

# Inside the loop where you process analyzer_results, merge bounding boxes that are right next to each other:
def merge_bounding_boxes(analyzer_results, characters, combine_pixel_dist, vertical_padding=2):
    '''
    Merge identified bounding boxes containing PII that are very close to one another
    '''
    analyzed_bounding_boxes = []
    if len(analyzer_results) > 0 and len(characters) > 0:
        merged_bounding_boxes = []
        current_box = None
        current_y = None

        for i, result in enumerate(analyzer_results):
            print("Considering result", str(i))
            for char in characters[result.start : result.end]:
                if isinstance(char, LTChar):
                    char_box = list(char.bbox)
                    # Add vertical padding to the top of the box
                    char_box[3] += vertical_padding

                    if current_y is None or current_box is None:
                        current_box = char_box
                        current_y = char_box[1]
                    else:
                        vertical_diff_bboxes = abs(char_box[1] - current_y)
                        horizontal_diff_bboxes = abs(char_box[0] - current_box[2])

                        if (
                            vertical_diff_bboxes <= 5
                            and horizontal_diff_bboxes <= combine_pixel_dist
                        ):
                            current_box[2] = char_box[2]  # Extend the current box horizontally
                            current_box[3] = max(current_box[3], char_box[3])  # Ensure the top is the highest
                        else:
                            merged_bounding_boxes.append(
                                {"boundingBox": current_box, "result": result})
                            
                            # Reset current_box and current_y after appending
                            current_box = char_box
                            current_y = char_box[1]
            
            # After finishing with the current result, add the last box for this result
            if current_box:
                merged_bounding_boxes.append({"boundingBox": current_box, "result": result})
                current_box = None
                current_y = None  # Reset for the next result

        if not merged_bounding_boxes:
            analyzed_bounding_boxes.extend(
                {"boundingBox": char.bbox, "result": result} 
                for result in analyzer_results 
                for char in characters[result.start:result.end] 
                if isinstance(char, LTChar)
            )
        else:
            analyzed_bounding_boxes.extend(merged_bounding_boxes)

        print("analysed_bounding_boxes:\n\n", analyzed_bounding_boxes)
    
    return analyzed_bounding_boxes

# def merge_bounding_boxes(analyzer_results, characters, combine_pixel_dist, vertical_padding=2, signature_bounding_boxes=None):
#     '''
#     Merge identified bounding boxes containing PII or signatures that are very close to one another.
#     '''
#     analyzed_bounding_boxes = []
#     merged_bounding_boxes = []
#     current_box = None
#     current_y = None

#     # Handle PII and text bounding boxes first
#     if len(analyzer_results) > 0 and len(characters) > 0:
#         for i, result in enumerate(analyzer_results):
#             #print("Considering result", str(i))
#             #print("Result:", result)
#             #print("Characters:", characters)

#             for char in characters[result.start: result.end]:
#                 if isinstance(char, LTChar):
#                     char_box = list(char.bbox)
#                     # Add vertical padding to the top of the box
#                     char_box[3] += vertical_padding

#                     if current_y is None or current_box is None:
#                         current_box = char_box
#                         current_y = char_box[1]
#                     else:
#                         vertical_diff_bboxes = abs(char_box[1] - current_y)
#                         horizontal_diff_bboxes = abs(char_box[0] - current_box[2])

#                         if (
#                             vertical_diff_bboxes <= 5
#                             and horizontal_diff_bboxes <= combine_pixel_dist
#                         ):
#                             current_box[2] = char_box[2]  # Extend the current box horizontally
#                             current_box[3] = max(current_box[3], char_box[3])  # Ensure the top is the highest
#                         else:
#                             merged_bounding_boxes.append(
#                                 {"boundingBox": current_box, "result": result})
                            
#                             # Reset current_box and current_y after appending
#                             current_box = char_box
#                             current_y = char_box[1]

#             # After finishing with the current result, add the last box for this result
#             if current_box:
#                 merged_bounding_boxes.append({"boundingBox": current_box, "result": result})
#                 current_box = None
#                 current_y = None  # Reset for the next result

#     # Handle signature bounding boxes (without specific characters)
#     if signature_bounding_boxes is not None:
#         for sig_box in signature_bounding_boxes:
#             sig_box = list(sig_box)  # Ensure it's a list to modify the values
#             if current_y is None or current_box is None:
#                 current_box = sig_box
#                 current_y = sig_box[1]
#             else:
#                 vertical_diff_bboxes = abs(sig_box[1] - current_y)
#                 horizontal_diff_bboxes = abs(sig_box[0] - current_box[2])

#                 if (
#                     vertical_diff_bboxes <= 5
#                     and horizontal_diff_bboxes <= combine_pixel_dist
#                 ):
#                     current_box[2] = sig_box[2]  # Extend the current box horizontally
#                     current_box[3] = max(current_box[3], sig_box[3])  # Ensure the top is the highest
#                 else:
#                     merged_bounding_boxes.append({"boundingBox": current_box, "type": "signature"})
                    
#                     # Reset current_box and current_y after appending
#                     current_box = sig_box
#                     current_y = sig_box[1]

#             # Add the last bounding box for the signature
#             if current_box:
#                 merged_bounding_boxes.append({"boundingBox": current_box, "type": "signature"})
#                 current_box = None
#                 current_y = None

#     # If no bounding boxes were merged, add individual character bounding boxes
#     if not merged_bounding_boxes:
#         analyzed_bounding_boxes.extend(
#             {"boundingBox": char.bbox, "result": result}
#             for result in analyzer_results
#             for char in characters[result.start:result.end]
#             if isinstance(char, LTChar)
#         )
#     else:
#         analyzed_bounding_boxes.extend(merged_bounding_boxes)

#     #print("analysed_bounding_boxes:\n\n", analyzed_bounding_boxes)
    
#     return analyzed_bounding_boxes

def create_text_redaction_process_results(analyzer_results, analyzed_bounding_boxes, page_num):
    decision_process_table = pd.DataFrame()

    if len(analyzer_results) > 0:
        # Create summary df of annotations to be made
        analyzed_bounding_boxes_df_new = pd.DataFrame(analyzed_bounding_boxes)
        analyzed_bounding_boxes_df_text = analyzed_bounding_boxes_df_new['result'].astype(str).str.split(",",expand=True).replace(".*: ", "", regex=True)
        analyzed_bounding_boxes_df_text.columns = ["type", "start", "end", "score"]
        analyzed_bounding_boxes_df_new = pd.concat([analyzed_bounding_boxes_df_new, analyzed_bounding_boxes_df_text], axis = 1)
        analyzed_bounding_boxes_df_new['page'] = page_num + 1
        decision_process_table = pd.concat([decision_process_table, analyzed_bounding_boxes_df_new], axis = 0).drop('result', axis=1)

        print('\n\ndecision_process_table:\n\n', decision_process_table)
    
    return decision_process_table

def create_annotations_for_bounding_boxes(analyzed_bounding_boxes):
    annotations_on_page = []
    for analyzed_bounding_box in analyzed_bounding_boxes:
        bounding_box = analyzed_bounding_box["boundingBox"]
        annotation = Dictionary(
            Type=Name.Annot,
            Subtype=Name.Square, #Name.Highlight,
            QuadPoints=[bounding_box[0], bounding_box[3], bounding_box[2], bounding_box[3],
                        bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[1]],
            Rect=[bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]],
            C=[0, 0, 0],
            IC=[0, 0, 0],
            CA=1, # Transparency
            T=analyzed_bounding_box["result"].entity_type,
            BS=Dictionary(
                W=0,                     # Border width: 1 point
                S=Name.S                # Border style: solid
            )
        )
        annotations_on_page.append(annotation)
    return annotations_on_page

def redact_text_pdf(filename:str, language:str, chosen_redact_entities:List[str], allow_list:List[str]=None, page_min:int=0, page_max:int=999, analysis_type:str = "Simple text analysis - PDFs with selectable text", progress=Progress(track_tqdm=True)):
    '''
    Redact chosen entities from a pdf that is made up of multiple pages that are not images.
    '''
    annotations_all_pages = []
    decision_process_table_all_pages = []
    
    combine_pixel_dist = 200 # Horizontal distance between PII bounding boxes under/equal they are combined into one

    pdf = Pdf.open(filename)
    page_num = 0

    number_of_pages = len(pdf.pages)

    # Check that page_min and page_max are within expected ranges
    if page_max > number_of_pages or page_max == 0:
        page_max = number_of_pages
    #else:
    #    page_max = page_max - 1

    if page_min <= 0:
        page_min = 0
    else:
        page_min = page_min - 1

    print("Page range is",str(page_min), "to", str(page_max))
    
    for page_no in range(page_min, page_max):
        page = pdf.pages[page_no]

        print("Page number is:", page_no)

        # The /MediaBox in a PDF specifies the size of the page [left, bottom, right, top]
        media_box = page.MediaBox
        page_width = media_box[2] - media_box[0]
        page_height = media_box[3] - media_box[1]
        

        annotations_on_page = []
        decision_process_table_on_page = []       

        for page_layout in extract_pages(filename, page_numbers = [page_no], maxpages=1):
            
            page_analyzer_results = []
            page_analyzed_bounding_boxes = []
            text_container_analyzer_results = []
            text_container_analyzed_bounding_boxes = []
            characters = []

            if analysis_type == "Simple text analysis - PDFs with selectable text":
                for i, text_container in enumerate(page_layout):

                    text_container_analyzer_results, characters = analyze_text_container(text_container, language, chosen_redact_entities, score_threshold, allow_list)
                                 
                    # Merge bounding boxes if very close together
                    text_container_analyzed_bounding_boxes = merge_bounding_boxes(text_container_analyzer_results, characters, combine_pixel_dist, vertical_padding = 2)


                    page_analyzed_bounding_boxes.extend(text_container_analyzed_bounding_boxes)
                    page_analyzer_results.extend(text_container_analyzer_results)


            decision_process_table_on_page = create_text_redaction_process_results(page_analyzer_results, page_analyzed_bounding_boxes, page_num)           

            annotations_on_page = create_annotations_for_bounding_boxes(page_analyzed_bounding_boxes)
            #print('\n\nannotations_on_page:', annotations_on_page)    
          
            # Make page annotations
            page.Annots = pdf.make_indirect(annotations_on_page)

            annotations_all_pages.extend([annotations_on_page])
            decision_process_table_all_pages.extend([decision_process_table_on_page])
            
            print("For page number:", page_no, "there are", len(annotations_all_pages[page_num]), "annotations")
            
            #page_num += 1

    return pdf, decision_process_table_all_pages
