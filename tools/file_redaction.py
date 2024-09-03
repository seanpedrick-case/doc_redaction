from PIL import Image, ImageChops, ImageDraw
from typing import List
import pandas as pd
from presidio_image_redactor import ImageRedactorEngine, ImageAnalyzerEngine
from presidio_image_redactor.entities import ImageRecognizerResult
from pdfminer.high_level import extract_pages
from tools.file_conversion import process_file
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine #, LTAnno
from pikepdf import Pdf, Dictionary, Name
from gradio import Progress
import time
from collections import defaultdict  # For efficient grouping

from tools.load_spacy_model_custom_recognisers import nlp_analyser, score_threshold
from tools.helper_functions import get_file_path_end, output_folder
from tools.file_conversion import process_file, is_pdf, convert_text_pdf_to_img_pdf
from tools.data_anonymise import generate_decision_process_output
import gradio as gr


def choose_and_run_redactor(file_paths:List[str], image_paths:List[str], language:str, chosen_redact_entities:List[str], in_redact_method:str, in_allow_list:List[List[str]]=None, latest_file_completed:int=0, out_message:list=[], out_file_paths:list=[], log_files_output_paths:list=[], first_loop_state:bool=False, page_min:int=0, page_max:int=999, progress=gr.Progress(track_tqdm=True)):

    tic = time.perf_counter()


    # If this is the first time around, set variables to 0/blank
    if first_loop_state==True:
        latest_file_completed = 0
        out_message = []
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
        # Set to a very high number so as not to mess with subsequent file processing by the user
        latest_file_completed = 99
        final_out_message = '\n'.join(out_message)
        return final_out_message, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths
    
    file_paths_loop = [file_paths[int(latest_file_completed)]]

    if in_allow_list:
        in_allow_list_flat = [item for sublist in in_allow_list for item in sublist]
    

    for file in progress.tqdm(file_paths_loop, desc="Redacting files", unit = "files"):
        file_path = file.name

        if file_path:
            file_path_without_ext = get_file_path_end(file_path)
            is_a_pdf = is_pdf(file_path) == True
            if is_a_pdf == False:
                # If user has not submitted a pdf, assume it's an image
                print("File is not a pdf, assuming that image analysis needs to be used.")
                in_redact_method = "Image analysis"
        else:
            out_message = "No file selected"
            print(out_message)
            return out_message, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths

        if in_redact_method == "Image analysis":
            # Analyse and redact image-based pdf or image
            # if is_pdf_or_image(file_path) == False:
            #     return "Please upload a PDF file or image file (JPG, PNG) for image analysis.", None

            print("Redacting file as image-based file")
            pdf_images, output_logs = redact_image_pdf(file_path, image_paths, language, chosen_redact_entities, in_allow_list_flat, is_a_pdf, page_min, page_max)
            out_image_file_path = output_folder + file_path_without_ext + "_redacted_as_img.pdf"
            pdf_images[0].save(out_image_file_path, "PDF" ,resolution=100.0, save_all=True, append_images=pdf_images[1:])

            out_file_paths.append(out_image_file_path)
            out_message.append("File '" + file_path_without_ext + "' successfully redacted and saved to file")

            output_logs_str = str(output_logs)
            logs_output_file_name = out_image_file_path + "_decision_process_output.txt"
            with open(logs_output_file_name, "w") as f:
                f.write(output_logs_str)
            log_files_output_paths.append(logs_output_file_name)

            # Increase latest file completed count unless we are at the last file
            if latest_file_completed != len(file_paths):
                print("Completed file number:", str(latest_file_completed))
                latest_file_completed += 1                

        elif in_redact_method == "Text analysis":
            if is_pdf(file_path) == False:
                return "Please upload a PDF file for text analysis. If you have an image, select 'Image analysis'.", None, None

            # Analyse text-based pdf
            print('Redacting file as text-based PDF')
            pdf_text, output_logs = redact_text_pdf(file_path, language, chosen_redact_entities, in_allow_list_flat, page_min, page_max)
            out_text_file_path = output_folder + file_path_without_ext + "_text_redacted.pdf"
            pdf_text.save(out_text_file_path)

            #out_file_paths.append(out_text_file_path)
            out_message_new = "File " + file_path_without_ext + " successfully redacted"
            out_message.append(out_message_new)

            # Convert message
            convert_message="Converting PDF to image-based PDF to embed redactions."
            #progress(0.8, desc=convert_message)
            print(convert_message)

            # Convert document to image-based document to 'embed' redactions
            img_output_summary, img_output_file_path = convert_text_pdf_to_img_pdf(file_path, [out_text_file_path])
            out_file_paths.extend(img_output_file_path)

            output_logs_str = str(output_logs)
            logs_output_file_name = img_output_file_path[0] + "_decision_process_output.txt"
            with open(logs_output_file_name, "w") as f:
                f.write(output_logs_str)
            log_files_output_paths.append(logs_output_file_name)

            # Add confirmation for converting to image if you want
            # out_message.append(img_output_summary)

            if latest_file_completed != len(file_paths):
                print("Completed file number:", str(latest_file_completed), "more files to do")
                latest_file_completed += 1                
            
        else:
            out_message = "No redaction method selected"
            print(out_message)
            return out_message, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths
        
    
    toc = time.perf_counter()
    out_time = f"in {toc - tic:0.1f} seconds."
    print(out_time)

    out_message_out = '\n'.join(out_message)
    out_message_out = out_message_out + " " + out_time

    out_message_out = out_message_out + "\n\nGo to to the Redaction settings tab to see redaction logs. Please give feedback on the results below to help improve this app."

    return out_message_out, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths

def merge_img_bboxes(bboxes, horizontal_threshold=150, vertical_threshold=25):
            merged_bboxes = []
            grouped_bboxes = defaultdict(list)

            # 1. Group by approximate vertical proximity
            for box in bboxes:
                grouped_bboxes[round(box.top / vertical_threshold)].append(box)

            # 2. Merge within each group
            for _, group in grouped_bboxes.items():
                group.sort(key=lambda box: box.left)

                merged_box = group[0]
                for next_box in group[1:]:
                    if next_box.left - (merged_box.left + merged_box.width) <= horizontal_threshold:
                        #print("Merging a box")
                        # Calculate new dimensions for the merged box
                        new_left = min(merged_box.left, next_box.left)
                        new_top = min(merged_box.top, next_box.top)
                        new_width = max(merged_box.left + merged_box.width, next_box.left + next_box.width) - new_left
                        new_height = max(merged_box.top + merged_box.height, next_box.top + next_box.height) - new_top
                        merged_box = ImageRecognizerResult(
                            merged_box.entity_type, merged_box.start, merged_box.end, merged_box.score, new_left, new_top, new_width, new_height
                        )
                    else:
                        merged_bboxes.append(merged_box)
                        merged_box = next_box  

                merged_bboxes.append(merged_box) 
            return merged_bboxes

def redact_image_pdf(file_path:str, image_paths:List[str], language:str, chosen_redact_entities:List[str], allow_list:List[str]=None, is_a_pdf:bool=True, page_min:int=0, page_max:int=999, progress=Progress(track_tqdm=True)):
    '''
    Take an path for an image of a document, then run this image through the Presidio ImageAnalyzer and PIL to get a redacted page back. Adapted from Presidio ImageRedactorEngine.
    '''

    fill = (0, 0, 0)
    decision_process_output_str = ""

    if not image_paths:
        out_message = "PDF does not exist as images. Converting pages to image"
        print(out_message)
        #progress(0, desc=out_message)

        image_paths = process_file(file_path)

    images = []

    #print("Image paths:", image_paths)
    number_of_pages = len(image_paths[0])

    print("Number of pages:", str(number_of_pages))

    out_message = "Redacting pages"
    print(out_message)
    #progress(0.1, desc=out_message)

    # Check that page_min and page_max are within expected ranges
    if page_max > number_of_pages or page_max == 0:
        page_max = number_of_pages
    #else:
    #    page_max = page_max - 1

    if page_min <= 0:
        page_min = 0
    else:
        page_min = page_min - 1

    print("Page range:", str(page_min), "to", str(page_max))

    #for i in progress.tqdm(range(0,number_of_pages), total=number_of_pages, unit="pages", desc="Redacting pages"):
    
    for n in range(0, number_of_pages):

        try:
            image = image_paths[0][n]#.copy()
            print("Skipping page", str(n))
            #print("image:", image)
        except Exception as e:
            print("Could not redact page:", str(i), "due to:")
            print(e)
            continue

        if n >= page_min and n <= page_max:
        #for i in range(page_min, page_max):

            i = n

            print("Redacting page", str(i))

            # Get the image to redact using PIL lib (pillow)
            #print("image_paths:", image_paths)

            #image = ImageChops.duplicate(image_paths[i])
            #print("Image paths i:", image_paths[0])

            # Assuming image_paths[i] is your PIL image object
            try:
                image = image_paths[0][i]#.copy()
                #print("image:", image)
            except Exception as e:
                print("Could not redact page:", str(i), "due to:")
                print(e)
                continue

            # %%
            image_analyser = ImageAnalyzerEngine(nlp_analyser)
            engine = ImageRedactorEngine(image_analyser)

            if language == 'en':
                ocr_lang = 'eng'
            else: ocr_lang = language

            bboxes = image_analyser.analyze(image,ocr_kwargs={"lang": ocr_lang},
                    **{
                    "allow_list": allow_list,
                    "language": language,
                    "entities": chosen_redact_entities,
                    "score_threshold": score_threshold,
                    "return_decision_process":True,
                })
            
            # Text placeholder in this processing step, as the analyze method does not return the OCR text
            if bboxes:
                decision_process_output_str = str(bboxes)
                print("Decision process:", decision_process_output_str)
            
            #print("For page: ", str(i), "Bounding boxes: ", bboxes)

            draw = ImageDraw.Draw(image)
                
            merged_bboxes = merge_img_bboxes(bboxes)

            #print("For page:", str(i), "Merged bounding boxes:", merged_bboxes)

            # 3. Draw the merged boxes (unchanged)
            for box in merged_bboxes:
                x0 = box.left
                y0 = box.top
                x1 = x0 + box.width
                y1 = y0 + box.height
                draw.rectangle([x0, y0, x1, y1], fill=fill)

        images.append(image)

    return images, decision_process_output_str

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
def merge_bounding_boxes(analyzer_results, characters, combine_pixel_dist):
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

def redact_text_pdf(filename:str, language:str, chosen_redact_entities:List[str], allow_list:List[str]=None, page_min:int=0, page_max:int=999, progress=Progress(track_tqdm=True)):
    '''
    Redact chosen entities from a pdf that is made up of multiple pages that are not images.
    '''
    annotations_all_pages = []
    decision_process_table_all_pages = []
    
    combine_pixel_dist = 100 # Horizontal distance between PII bounding boxes under/equal they are combined into one

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

        annotations_on_page = []
        decision_process_table_on_page = []       

        for page_layout in extract_pages(filename, page_numbers = [page_no], maxpages=1):
            
            page_analyzer_results = []
            page_analyzed_bounding_boxes = []
            text_container_analyzer_results = []
            text_container_analyzed_bounding_boxes = []
            characters = []

            for text_container in page_layout:
                text_container_analyzer_results, characters = analyze_text_container(text_container, language, chosen_redact_entities, score_threshold, allow_list)
                # Merge bounding boxes if very close together
                text_container_analyzed_bounding_boxes = merge_bounding_boxes(text_container_analyzer_results, characters, combine_pixel_dist)

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
