from pdf2image import convert_from_path, pdfinfo_from_path
from tools.helper_functions import get_file_path_end, output_folder, tesseract_ocr_option, text_ocr_option, textract_option, local_pii_detector, aws_pii_detector
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import re
import gradio as gr
import time
import json
import pymupdf
from gradio import Progress
from typing import List, Optional

image_dpi = 300.0

def is_pdf_or_image(filename):
    """
    Check if a file name is a PDF or an image file.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if the file name ends with ".pdf", ".jpg", or ".png", False otherwise.
    """
    if filename.lower().endswith(".pdf") or filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png"):
        output = True
    else:
        output = False
    return output

def is_pdf(filename):
    """
    Check if a file name is a PDF.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if the file name ends with ".pdf", False otherwise.
    """
    return filename.lower().endswith(".pdf")

# %%
## Convert pdf to image if necessary

def convert_pdf_to_images(pdf_path:str, page_min:int = 0, image_dpi:float = image_dpi, progress=Progress(track_tqdm=True)):

    # Get the number of pages in the PDF
    page_count = pdfinfo_from_path(pdf_path)['Pages']
    print("Number of pages in PDF: ", str(page_count))

    images = []

    # Open the PDF file
    #for page_num in progress.tqdm(range(0,page_count), total=page_count, unit="pages", desc="Converting pages"): range(page_min,page_count): #
    for page_num in progress.tqdm(range(page_min,page_count), total=page_count, unit="pages", desc="Preparing pages"):
        
        print("Converting page: ", str(page_num + 1))

        # Convert one page to image
        out_path  = pdf_path + "_" + str(page_num) + ".png"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Check if the image already exists
        if os.path.exists(out_path):
            #print(f"Loading existing image from {out_path}.")
            image = Image.open(out_path)  # Load the existing image



        else:
            image_l = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=image_dpi, use_cropbox=True, use_pdftocairo=False)

            image = image_l[0]

            # Convert to greyscale
            image = image.convert("L")

            image.save(out_path, format="PNG")  # Save the new image

        # If no images are returned, break the loop
        if not image:
            print("Conversion of page", str(page_num), "to file failed.")
            break

        # print("Conversion of page", str(page_num), "to file succeeded.")
        # print("image:", image)

        images.append(out_path)

    print("PDF has been converted to images.")
    # print("Images:", images)

    return images

# Function to take in a file path, decide if it is an image or pdf, then process appropriately.
def process_file(file_path):
    # Get the file extension
    file_extension = os.path.splitext(file_path)[1].lower()

    # Check if the file is an image type
    if file_extension in ['.jpg', '.jpeg', '.png']:
        print(f"{file_path} is an image file.")
        # Perform image processing here
        img_object = [Image.open(file_path)]
        # Load images from the file paths

    # Check if the file is a PDF
    elif file_extension == '.pdf':
        print(f"{file_path} is a PDF file. Converting to image set")
        # Run your function for processing PDF files here
        img_object = convert_pdf_to_images(file_path)

    else:
        print(f"{file_path} is not an image or PDF file.")
        img_object = ['']

    return img_object

def get_input_file_names(file_input):
    '''
    Get list of input files to report to logs.
    '''

    all_relevant_files = []
    file_name_with_extension = ""
    full_file_name = ""

    #print("file_input:", file_input)

    if isinstance(file_input, str):
        file_input_list = [file_input]
    else:
        file_input_list = file_input

    for file in file_input_list:
        if isinstance(file, str):
            file_path = file
        else:
            file_path = file.name

        file_path_without_ext = get_file_path_end(file_path)

        file_extension = os.path.splitext(file_path)[1].lower()

        # Check if the file is an image type
        if file_extension in ['.jpg', '.jpeg', '.png', '.pdf', '.xlsx', '.csv', '.parquet']:
            all_relevant_files.append(file_path_without_ext)
            file_name_with_extension = file_path_without_ext + file_extension
            full_file_name = file_path
    
    all_relevant_files_str = ", ".join(all_relevant_files)

    print("all_relevant_files_str:", all_relevant_files_str)

    return all_relevant_files_str, file_name_with_extension, full_file_name

def prepare_image_or_pdf(
    file_paths: List[str],
    in_redact_method: str,
    in_allow_list: Optional[List[List[str]]] = None,
    latest_file_completed: int = 0,
    out_message: List[str] = [],
    first_loop_state: bool = False,
    number_of_pages:int = 1,
    current_loop_page_number:int=0,
    all_annotations_object:List = [],
    prepare_for_review:bool = False,
    progress: Progress = Progress(track_tqdm=True)
) -> tuple[List[str], List[str]]:
    """
    Prepare and process image or text PDF files for redaction.

    This function takes a list of file paths, processes each file based on the specified redaction method,
    and returns the output messages and processed file paths.

    Args:
        file_paths (List[str]): List of file paths to process.
        in_redact_method (str): The redaction method to use.
        in_allow_list (Optional[List[List[str]]]): List of allowed terms for redaction.
        latest_file_completed (int): Index of the last completed file.
        out_message (List[str]): List to store output messages.
        first_loop_state (bool): Flag indicating if this is the first iteration.
        number_of_pages (int): integer indicating the number of pages in the document
        all_annotations_object(List of annotation objects): All annotations for current document
        prepare_for_review(bool): Is this preparation step preparing pdfs and json files to review current redactions?
        progress (Progress): Progress tracker for the operation.
        

    Returns:
        tuple[List[str], List[str]]: A tuple containing the output messages and processed file paths.
    """

    tic = time.perf_counter()

    # If this is the first time around, set variables to 0/blank
    if first_loop_state==True:
        print("first_loop_state is True")
        latest_file_completed = 0
        out_message = []
        all_annotations_object = []
    else:
        print("Now attempting file:", str(latest_file_completed))
    
    # This is only run when a new page is loaded, so can reset page loop values. If end of last file (99), current loop number set to 999
    # if latest_file_completed == 99:
    #     current_loop_page_number = 999
    #     page_break_return = False
    # else:
    #     current_loop_page_number = 0
    #     page_break_return = False

    # If out message or converted_file_paths are blank, change to a list so it can be appended to
    if isinstance(out_message, str):
        out_message = [out_message]  

    converted_file_paths = []
    image_file_paths = []
    pymupdf_doc = []

    if not file_paths:
        file_paths = []

    if isinstance(file_paths, str):
        file_path_number = 1
    else:
        file_path_number = len(file_paths)

    #print("Current_loop_page_number at start of prepare_image_or_pdf function is:", current_loop_page_number)
    print("Number of file paths:", file_path_number)
    print("Latest_file_completed:", latest_file_completed)
    
    latest_file_completed = int(latest_file_completed)

    # If we have already redacted the last file, return the input out_message and file list to the relevant components
    if latest_file_completed >= file_path_number:
        print("Last file reached, returning files:", str(latest_file_completed))
        if isinstance(out_message, list):
            final_out_message = '\n'.join(out_message)
        else:
            final_out_message = out_message
        return final_out_message, converted_file_paths, image_file_paths, number_of_pages, number_of_pages, pymupdf_doc, all_annotations_object

    #in_allow_list_flat = [item for sublist in in_allow_list for item in sublist]

    progress(0.1, desc='Preparing file')

    if isinstance(file_paths, str):
        file_paths_list = [file_paths]
        file_paths_loop = file_paths_list
    else:
        if prepare_for_review == False:
            file_paths_list = file_paths
            file_paths_loop = [file_paths_list[int(latest_file_completed)]]
        else:
            file_paths_list = file_paths
            file_paths_loop = file_paths
             # Sort files to prioritise PDF files first, then JSON files. This means that the pdf can be loaded in, and pdf page path locations can be added to the json
            file_paths_loop = sorted(file_paths_loop, key=lambda x: (os.path.splitext(x)[1] != '.pdf', os.path.splitext(x)[1] != '.json'))      

    # Loop through files to load in
    for file in file_paths_loop:
        if isinstance(file, str):
            file_path = file
        else:
            file_path = file.name
        file_path_without_ext = get_file_path_end(file_path)

        if not file_path:
            out_message = "Please select a file."
            print(out_message)
            return out_message, converted_file_paths, image_file_paths, number_of_pages, number_of_pages, pymupdf_doc, all_annotations_object

        file_extension = os.path.splitext(file_path)[1].lower()

        # Check if the file is an image type
        if file_extension in ['.jpg', '.jpeg', '.png']:
            in_redact_method = tesseract_ocr_option


        # If the file name ends with redactions.json, assume it is an annoations object, overwrite the current variable
        if file_path.endswith(".json"):

            if prepare_for_review == True:
                if isinstance(file_path, str):
                    with open(file_path, 'r') as json_file:
                        all_annotations_object = json.load(json_file)
                else:
                    # Assuming file_path is a NamedString or similar
                    all_annotations_object = json.loads(file_path)  # Use loads for string content

                # Get list of page numbers
                image_file_paths_pages = [
                int(re.search(r'_(\d+)\.png$', os.path.basename(s)).group(1)) 
                for s in image_file_paths 
                if re.search(r'_(\d+)\.png$', os.path.basename(s))
                ]
                image_file_paths_pages = [int(i) for i in image_file_paths_pages]
                

                # If PDF pages have been converted to image files, replace the current image paths in the json to this
                if image_file_paths:
                    for i, annotation in enumerate(all_annotations_object):
                        annotation_page_number = int(re.search(r'_(\d+)\.png$', annotation["image"]).group(1))

                        # Check if the annotation page number exists in the image file paths pages
                        if annotation_page_number in image_file_paths_pages:

                            # Set the correct image page directly since we know it's in the list
                            correct_image_page = annotation_page_number
                            annotation["image"] = image_file_paths[correct_image_page]
                        else:
                            print("Page not found.")

                    #print("all_annotations_object:", all_annotations_object)

                # Write the response to a JSON file in output folder
                out_folder = output_folder + file_path_without_ext + file_extension
                with open(out_folder, 'w') as json_file:
                    json.dump(all_annotations_object, json_file, indent=4)  # indent=4 makes the JSON file pretty-printed
                continue

            else:
                # If the file loaded has end textract.json, assume this is a textract response object. Save this to the output folder so it can be found later during redaction and go to the next file.
                json_contents = json.load(file_path)
                # Write the response to a JSON file in output folder
                out_folder = output_folder + file_path_without_ext + file_extension
                with open(out_folder, 'w') as json_file:
                    json.dump(json_contents, json_file, indent=4)  # indent=4 makes the JSON file pretty-printed
                continue

        # Convert pdf/image file to correct format for redaction
        if in_redact_method == tesseract_ocr_option or in_redact_method == textract_option:
            if is_pdf_or_image(file_path) == False:
                out_message = "Please upload a PDF file or image file (JPG, PNG) for image analysis."
                print(out_message)
                return out_message, converted_file_paths, image_file_paths, number_of_pages, number_of_pages, pymupdf_doc, all_annotations_object
            
            converted_file_path = process_file(file_path)
            image_file_path = converted_file_path

        elif in_redact_method == text_ocr_option:
            if is_pdf(file_path) == False:
                out_message = "Please upload a PDF file for text analysis."
                print(out_message)
                return out_message, converted_file_paths, image_file_paths, number_of_pages, number_of_pages, pymupdf_doc, all_annotations_object
            
            converted_file_path = file_path # Pikepdf works with the basic unconverted pdf file
            image_file_path = process_file(file_path)

        converted_file_paths.append(converted_file_path)
        image_file_paths.extend(image_file_path)

        # If a pdf, load as a pymupdf document
        if is_pdf(file_path):
            pymupdf_doc = pymupdf.open(file_path)

        elif is_pdf_or_image(file_path):  # Alternatively, if it's an image
            # Convert image to a pymupdf document
            pymupdf_doc = pymupdf.open()  # Create a new empty document
            img = Image.open(file_path)  # Open the image file
            rect = pymupdf.Rect(0, 0, img.width, img.height)  # Create a rectangle for the image
            page = pymupdf_doc.new_page(width=img.width, height=img.height)  # Add a new page
            page.insert_image(rect, filename=file_path)  # Insert the image into the page


        toc = time.perf_counter()
        out_time = f"File '{file_path_without_ext}' prepared in {toc - tic:0.1f} seconds."

        print(out_time)

        out_message.append(out_time)
        out_message_out = '\n'.join(out_message)

        number_of_pages = len(image_file_paths)

        
    return out_message_out, converted_file_paths, image_file_paths, number_of_pages, number_of_pages, pymupdf_doc, all_annotations_object

def convert_text_pdf_to_img_pdf(in_file_path:str, out_text_file_path:List[str], image_dpi:float=image_dpi):
    file_path_without_ext = get_file_path_end(in_file_path)

    out_file_paths = out_text_file_path

    # Convert annotated text pdf back to image to give genuine redactions
    print("Creating image version of redacted PDF to embed redactions.")
    
    pdf_text_image_paths = process_file(out_text_file_path[0])
    out_text_image_file_path = output_folder + file_path_without_ext + "_text_redacted_as_img.pdf"
    pdf_text_image_paths[0].save(out_text_image_file_path, "PDF" ,resolution=image_dpi, save_all=True, append_images=pdf_text_image_paths[1:])

    # out_file_paths.append(out_text_image_file_path)

    out_file_paths = [out_text_image_file_path]

    out_message = "PDF " + file_path_without_ext + " converted to image-based file."
    print(out_message)

    #print("Out file paths:", out_file_paths)

    return out_message, out_file_paths
