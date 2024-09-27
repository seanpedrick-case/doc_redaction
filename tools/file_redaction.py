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
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine, LTTextLineHorizontal, LTAnno
from pikepdf import Pdf, Dictionary, Name
import pymupdf
from pymupdf import Rect   

import gradio as gr
from gradio import Progress

from typing import Tuple

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

    progress(0.5, desc="Redacting file")
    
    for file in file_paths_loop:
    #for file in progress.tqdm(file_paths_loop, desc="Redacting files", unit = "files"):
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

        if in_redact_method == "Quick image analysis - typed text" or in_redact_method == "Complex image analysis - docs with handwriting/signatures (AWS Textract)":
            #Analyse and redact image-based pdf or image
            if is_pdf_or_image(file_path) == False:
                out_message = "Please upload a PDF file or image file (JPG, PNG) for image analysis."
                return out_message, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths, estimated_time_taken_state, all_request_metadata_str

            print("Redacting file " + file_path_without_ext + " as an image-based file")

            pdf_images, redaction_logs, logging_file_paths, new_request_metadata = redact_image_pdf(file_path, image_paths, language, chosen_redact_entities, in_allow_list_flat, is_a_pdf, page_min, page_max, in_redact_method, handwrite_signature_checkbox)

            # Save file
            if is_pdf(file_path) == False:
                out_image_file_path = output_folder + file_path_without_ext + "_redacted_as_img.pdf"
                pdf_images[0].save(out_image_file_path, "PDF" ,resolution=100.0, save_all=True, append_images=pdf_images[1:])
            
            else:
                out_image_file_path = output_folder + file_path_without_ext + "_redacted.pdf"
                pdf_images.save(out_image_file_path)

            out_file_paths.append(out_image_file_path)
            if logging_file_paths:
                log_files_output_paths.extend(logging_file_paths)

            out_message.append("File '" + file_path_without_ext + "' successfully redacted")

            # Save decision making process
            # output_logs_str = str(output_logs)
            # logs_output_file_name = out_image_file_path + "_decision_process_output.txt"
            # with open(logs_output_file_name, "w") as f:
            #     f.write(output_logs_str)
            # log_files_output_paths.append(logs_output_file_name)

            logs_output_file_name = out_image_file_path + "_decision_process_output.csv"
            redaction_logs.to_csv(logs_output_file_name)
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

            print("file_path:", file_path)
            
            if is_pdf(file_path) == False:
                return "Please upload a PDF file for text analysis. If you have an image, select 'Image analysis'.", None, None
            
            # Analyse text-based pdf
            print('Redacting file as text-based PDF')
            pdf_text, decision_process_logs, page_text_outputs = redact_text_pdf(file_path, language, chosen_redact_entities, in_allow_list_flat, page_min, page_max, "Simple text analysis - PDFs with selectable text")
            
            out_text_file_path = output_folder + file_path_without_ext + "_text_redacted.pdf"
            pdf_text.save(out_text_file_path)
            out_file_paths.append(out_text_file_path)   

            # Convert message
            #convert_message="Converting PDF to image-based PDF to embed redactions."
            #print(convert_message)

            # Convert document to image-based document to 'embed' redactions
            #img_output_summary, img_output_file_path = convert_text_pdf_to_img_pdf(file_path, [out_text_file_path])
            #out_file_paths.extend(img_output_file_path)
            
            # Write logs to file
            decision_logs_output_file_name = out_text_file_path + "_decision_process_output.csv"
            decision_process_logs.to_csv(decision_logs_output_file_name)
            log_files_output_paths.append(decision_logs_output_file_name)

            all_text_output_file_name = out_text_file_path + "_all_text_output.csv"
            page_text_outputs.to_csv(all_text_output_file_name)
            log_files_output_paths.append(all_text_output_file_name)

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

def redact_page_with_pymupdf(doc, annotations_on_page, page_no, scale=(1,1)): 

    page = doc.load_page(page_no)
    page_height = max(page.rect.height, page.mediabox[3] - page.mediabox[1])

    #print("page_rect_height:", page.rect.height)
    #print("page mediabox size:", page.mediabox[3] - page.mediabox[1])

    for annot in annotations_on_page:
        if isinstance(annot, CustomImageRecognizerResult):
            scale_width = scale[0]
            scale_height = scale[1]

            print("scale:", scale)

            # Calculate scaled coordinates
            x1 = annot.left * scale_width
            new_y1 = (annot.top * scale_height)  # Flip Y0 (since it starts from bottom)
            x2 = (annot.left + annot.width) * scale_width  # Calculate x1
            new_y2 = ((annot.top + annot.height) * scale_height)  # Calculate y1 correctly

            rect = Rect(x1, new_y1, x2, new_y2)  # Create the PyMuPDF Rect (y1, y0 are flipped)

        else:
            #print("In the pikepdf conversion function")
            # Extract the /Rect field
            rect_field = annot["/Rect"]

            # Convert the extracted /Rect field to a list of floats (since pikepdf uses Decimal objects)
            rect_coordinates = [float(coord) for coord in rect_field]

            # Convert the Y-coordinates (flip using the page height)
            x1, y1, x2, y2 = rect_coordinates
            new_y1 = page_height - y2
            new_y2 = page_height - y1

            rect = Rect(x1, new_y1, x2, new_y2)

        # Convert to a PyMuPDF Rect object
        #rect = Rect(rect_coordinates)

                    # Calculate the middle y value and set height to 1 pixel
        middle_y = (new_y1 + new_y2) / 2
        rect_single_pixel_height = Rect(x1, middle_y, x2, middle_y + 1)  # Height of 1 pixel
        
        print("rect:", rect)
        # Add a redaction annotation
        #page.add_redact_annot(rect)

        # Add the annotation to the middle of the character line, so that it doesn't delete text from adjacent lines
        page.add_redact_annot(rect_single_pixel_height)

        # Set up drawing a black box over the whole rect
        shape = page.new_shape()
        shape.draw_rect(rect)
        shape.finish(color=(0, 0, 0), fill=(0, 0, 0))  # Black fill for the rectangle
        shape.commit()

    page.apply_redactions(images=0, graphics=0)
    page.clean_contents()

    return doc

def bounding_boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    return (box1[0] < box2[2] and box2[0] < box1[2] and
            box1[1] < box2[3] and box2[1] < box1[3])

def merge_img_bboxes(bboxes, combined_results: Dict, signature_recogniser_results=[], handwriting_recogniser_results=[], handwrite_signature_checkbox: List[str]=["Redact all identified handwriting", "Redact all identified signatures"], horizontal_threshold:int=50, vertical_threshold:int=12):
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
                        if current_char <= start_char < word_end or current_char < end_char <= word_end or (start_char <= current_char and word_end <= end_char):
                            relevant_words.append(word)
                        if word_end >= end_char:
                            break
                        current_char = word_end
                        if not word['text'].endswith(' '):
                            current_char += 1  # +1 for space if the word doesn't already end with a space

                    if relevant_words:
                        print("Relevant words:", relevant_words)
                        left = min(word['bounding_box'][0] for word in relevant_words)
                        top = min(word['bounding_box'][1] for word in relevant_words)
                        right = max(word['bounding_box'][2] for word in relevant_words)
                        bottom = max(word['bounding_box'][3] for word in relevant_words)
                        
                        # Combine the text of all relevant words
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

    # Also open as pymupdf pdf to apply annotations later on
    doc = pymupdf.open(file_path)

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
    all_line_level_ocr_results_df = pd.DataFrame()
    all_decision_process_table = pd.DataFrame()

    if analysis_type == "Quick image analysis - typed text": ocr_results_file_path = output_folder + "ocr_results_" + file_name + "_pages_" + str(page_min + 1) + "_" + str(page_max) + ".csv"
    elif analysis_type == "Complex image analysis - docs with handwriting/signatures (AWS Textract)": ocr_results_file_path = output_folder + "ocr_results_" + file_name + "_pages_" + str(page_min + 1) + "_" + str(page_max) + "_textract.csv"    
    
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


            # Get the dimensions of the page in points with pymupdf to get relative scale
            page = doc.load_page(i)
            mu_page_rect = page.rect
            #mu_page_width = mu_page_rect.width
            mu_page_height = max(mu_page_rect.height, page.mediabox[3] - page.mediabox[1])
            mu_page_width = max(mu_page_rect.width, page.mediabox[2] - page.mediabox[0])
            #mu_page_height = mu_page_rect.height

            # Calculate scaling factors between PIL image and pymupdf
            scale_width = mu_page_width / page_width
            scale_height = mu_page_height / page_height

            scale = (scale_width, scale_height)


            # Possibility to use different languages
            if language == 'en':
                ocr_lang = 'eng'
            else: ocr_lang = language

            # Step 1: Perform OCR. Either with Tesseract, or with AWS Textract
            if analysis_type == "Quick image analysis - typed text":
                
                word_level_ocr_results = image_analyser.perform_ocr(image)

                # Combine OCR results
                line_level_ocr_results, line_level_ocr_results_with_children = combine_ocr_results(word_level_ocr_results)

                #print("ocr_results after:", ocr_results)

                # Save ocr_with_children_outputs
                ocr_results_with_children_str = str(line_level_ocr_results_with_children)
                logs_output_file_name = output_folder + "ocr_with_children.txt"
                with open(logs_output_file_name, "w") as f:
                    f.write(ocr_results_with_children_str)
    
            # Import results from json and convert
            if analysis_type == "Complex image analysis - docs with handwriting/signatures (AWS Textract)":
                
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

                line_level_ocr_results, handwriting_or_signature_boxes, signature_recogniser_results, handwriting_recogniser_results, line_level_ocr_results_with_children = json_to_ocrresult(text_blocks, page_width, page_height)

                # Save ocr_with_children_output
                # ocr_results_with_children_str = str(line_level_ocr_results_with_children)
                # logs_output_file_name = output_folder + "ocr_with_children_textract.txt"
                # with open(logs_output_file_name, "w") as f:
                #     f.write(ocr_results_with_children_str)

            # Step 2: Analyze text and identify PII
            redaction_bboxes = image_analyser.analyze_text(
                line_level_ocr_results,
                line_level_ocr_results_with_children,
                language=language,
                entities=chosen_redact_entities,
                allow_list=allow_list,
                score_threshold=score_threshold,
            )

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

                for box in merged_redaction_bboxes:
                    x0 = box.left
                    y0 = box.top
                    x1 = x0 + box.width
                    y1 = y0 + box.height
                    draw.rectangle([x0, y0, x1, y1], fill=fill)


            ## Apply annotations with pymupdf
            else:
                doc = redact_page_with_pymupdf(doc, merged_redaction_bboxes, i, scale)

            #doc.save("image_redact.pdf")

            # Log OCR results

            #line_level_ocr_results_str = "Page:" + reported_page_number + "\n" + str(line_level_ocr_results)
            #all_ocr_results.append(line_level_ocr_results_str) 

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

            # Convert decision process to table
            # Export the decision making process
            if merged_redaction_bboxes:
                # for bbox in merged_redaction_bboxes:
                #     print(f"Entity: {bbox.entity_type}, Text: {bbox.text}, Bbox: ({bbox.left}, {bbox.top}, {bbox.width}, {bbox.height})")
                
                #decision_process_output_str = "Page " + reported_page_number + ":\n" + str(merged_redaction_bboxes)
                #all_decision_process.append(decision_process_output_str)

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

        if is_pdf(file_path) == False:
            images.append(image)
            doc = images

    # Write OCR results as a log file    
    # line_level_ocr_results_out = "\n".join(all_ocr_results)
    # with open(ocr_results_file_path, "w") as f:
    #     f.write(line_level_ocr_results_out)

    all_line_level_ocr_results_df.to_csv(ocr_results_file_path)
    logging_file_paths.append(ocr_results_file_path)

    return doc, all_decision_process_table, logging_file_paths, request_metadata

def get_text_container_characters(text_container:LTTextContainer):

    if isinstance(text_container, LTTextContainer):
        characters = [char
                    for line in text_container
                    if isinstance(line, LTTextLine) or isinstance(line, LTTextLineHorizontal)
                    for char in line]
    
        return characters
    return []
    

def analyze_text_container(text_container:OCRResult, language:str, chosen_redact_entities:List[str], score_threshold:float, allow_list:List[str]):
    '''
    Take text and bounding boxes in OCRResult format and analyze it for PII using spacy and the Microsoft Presidio package.
    '''

    text_to_analyze = text_container.text
    #print("text_to_analyze:", text_to_analyze)

    analyzer_results = nlp_analyser.analyze(text=text_to_analyze,
                                            language=language, 
                                            entities=chosen_redact_entities,
                                            score_threshold=score_threshold,
                                            return_decision_process=True,
                                            allow_list=allow_list)        
    return analyzer_results


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
        full_text += char.get_text()

        # Update overall bounding box
        x0, y0, x1, y1 = char.bbox
        overall_bbox[0] = min(overall_bbox[0], x0)  # x0
        overall_bbox[1] = min(overall_bbox[1], y0)  # y0
        overall_bbox[2] = max(overall_bbox[2], x1)  # x1
        overall_bbox[3] = max(overall_bbox[3], y1)  # y1
        
        # Update current word
        current_word += char.get_text()
        
        # Update current word bounding box
        current_word_bbox[0] = min(current_word_bbox[0], x0)  # x0
        current_word_bbox[1] = min(current_word_bbox[1], y0)  # y0
        current_word_bbox[2] = max(current_word_bbox[2], x1)  # x1
        current_word_bbox[3] = max(current_word_bbox[3], y1)  # y1


    # Finalize the last word if any
    if current_word:
        word_bboxes.append((current_word, current_word_bbox))

    if full_text:
        line_level_results_out.append(OCRResult(full_text, round(overall_bbox[0],2), round(overall_bbox[1], 2), round(overall_bbox[2]-overall_bbox[0],2), round(overall_bbox[3]-overall_bbox[1],2)))
        

    return line_level_results_out, line_level_characters_out  # Return both results and character objects

def merge_text_bounding_boxes(analyzer_results:CustomImageRecognizerResult, characters:List[LTChar], combine_pixel_dist:int, vertical_padding:int=0):
    '''
    Merge identified bounding boxes containing PII that are very close to one another
    '''
    analyzed_bounding_boxes = []
    if len(analyzer_results) > 0 and len(characters) > 0:
        # Extract bounding box coordinates for sorting
        bounding_boxes = []
        text_out = []
        for result in analyzer_results:
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
                    # Add a space if current_text is not empty
                    if current_text:
                        current_text.append(" ")  # Add space between texts
                    current_text.extend(text)
                else:
                    merged_bounding_boxes.append(
                        {"text":"".join(current_text),"boundingBox": current_box, "result": current_result})
                    #print(f"Appending merged box: {current_box}")

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
            analyzed_bounding_boxes.extend(
                {"text":text, "boundingBox": char.bbox, "result": result} 
                for result in analyzer_results 
                for char in characters[result.start:result.end] 
                if isinstance(char, LTChar)
            )
        else:
            analyzed_bounding_boxes.extend(merged_bounding_boxes)

        #print("Analyzed bounding boxes:\n\n", analyzed_bounding_boxes)
    
    return analyzed_bounding_boxes

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

        #print('\n\ndecision_process_table:\n\n', decision_process_table)
    
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
    page_text_outputs_all_pages = pd.DataFrame()
    decision_process_table_all_pages = pd.DataFrame()
    
    combine_pixel_dist = 20 # Horizontal distance between PII bounding boxes under/equal they are combined into one

    # Open with Pikepdf to get text lines
    pdf = Pdf.open(filename)
    # Also open pdf with pymupdf to be able to annotate later while retaining text
    doc = pymupdf.open(filename) 
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
        #media_box = page.MediaBox
        #page_width = media_box[2] - media_box[0]
        #page_height = media_box[3] - media_box[1]
        
        for page_layout in extract_pages(filename, page_numbers = [page_no], maxpages=1):
            
            page_analyzer_results = []
            page_analyzed_bounding_boxes = []            
            
            characters = []
            annotations_on_page = []
            decision_process_table_on_page = pd.DataFrame()    
            page_text_outputs = pd.DataFrame()  

            if analysis_type == "Simple text analysis - PDFs with selectable text":
                for text_container in page_layout:

                    text_container_analyzer_results = []
                    text_container_analyzed_bounding_boxes = []

                    characters = get_text_container_characters(text_container)

                    # Create dataframe for all the text on the page
                    line_level_text_results_list, line_characters = create_text_bounding_boxes_from_characters(characters)

                    print("line_characters:", line_characters)

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
                        text_line_analyzer_result = []
                        text_line_bounding_boxes = []

                        #print("text_line:", text_line.text)

                        text_line_analyzer_result = analyze_text_container(text_line, language, chosen_redact_entities, score_threshold, allow_list)

                        # Merge bounding boxes for the line if multiple found close together                    
                        if text_line_analyzer_result:
                            # Merge bounding boxes if very close together
                            print("text_line_bounding_boxes:", text_line_bounding_boxes)
                            print("line_characters:")
                            #print(line_characters[i])
                            print("".join(char._text for char in line_characters[i]))
                            text_line_bounding_boxes = merge_text_bounding_boxes(text_line_analyzer_result, line_characters[i], combine_pixel_dist, vertical_padding = 0)

                            text_container_analyzer_results.extend(text_line_analyzer_result)
                            text_container_analyzed_bounding_boxes.extend(text_line_bounding_boxes)
                        
                        print("\n FINAL text_container_analyzer_results:", text_container_analyzer_results)

                    
                    page_analyzer_results.extend(text_container_analyzer_results)
                    page_analyzed_bounding_boxes.extend(text_container_analyzed_bounding_boxes)

      

            # Annotate redactions on page
            annotations_on_page = create_annotations_for_bounding_boxes(page_analyzed_bounding_boxes)
 
            # Make pymupdf redactions
            doc = redact_page_with_pymupdf(doc, annotations_on_page, page_no)
          
            # Make page annotations
            #page.Annots = pdf.make_indirect(annotations_on_page)
            if annotations_on_page:
                annotations_all_pages.extend([annotations_on_page])

            print("For page number:", page_no, "there are", len(annotations_all_pages[page_num]), "annotations")

            # Write logs
            # Create decision process table
            decision_process_table_on_page = create_text_redaction_process_results(page_analyzer_results, page_analyzed_bounding_boxes, page_num)     

            if not decision_process_table_on_page.empty:
                decision_process_table_all_pages = pd.concat([decision_process_table_all_pages, decision_process_table_on_page])

            if not page_text_outputs.empty:
                page_text_outputs = page_text_outputs.sort_values(["top", "left"], ascending=[False, False]).reset_index(drop=True)
                #page_text_outputs.to_csv("text_page_text_outputs.csv")
                page_text_outputs_all_pages = pd.concat([page_text_outputs_all_pages, page_text_outputs])
            
    return doc, decision_process_table_all_pages, page_text_outputs_all_pages
