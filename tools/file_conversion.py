from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image, ImageFile
import os
import re
import time
import json
import numpy as np
import pymupdf
from pymupdf import Document, Page, Rect
import pandas as pd
import shutil
import zipfile
from collections import defaultdict
from tqdm import tqdm
from gradio import Progress
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdf2image import convert_from_path
from PIL import Image
from scipy.spatial import cKDTree
import random
import string
import warnings # To warn about potential type changes

from tools.config import OUTPUT_FOLDER, INPUT_FOLDER, IMAGES_DPI, LOAD_TRUNCATED_IMAGES, MAX_IMAGE_PIXELS, CUSTOM_BOX_COLOUR, COMPRESS_REDACTED_PDF, TESSERACT_TEXT_EXTRACT_OPTION, SELECTABLE_TEXT_EXTRACT_OPTION, TEXTRACT_TEXT_EXTRACT_OPTION
from tools.helper_functions import get_file_name_without_type, read_file
# from tools.aws_textract import load_and_convert_textract_json

IMAGE_NUM_REGEX = re.compile(r'_(\d+)\.png$')

pd.set_option('future.no_silent_downcasting', True)

image_dpi = float(IMAGES_DPI)
if not MAX_IMAGE_PIXELS: Image.MAX_IMAGE_PIXELS = None
else: Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = LOAD_TRUNCATED_IMAGES.lower() == "true"
COMPRESS_REDACTED_PDF = COMPRESS_REDACTED_PDF.lower() == "true"

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

## Convert pdf to image if necessary

def check_image_size_and_reduce(out_path:str, image:Image):
    '''
    Check if a given image size is above around 4.5mb, and reduce size if necessary. 5mb is the maximum possible to submit to AWS Textract.
    '''

    all_img_details = []
    page_num = 0

    # Check file size and resize if necessary
    max_size = 4.5 * 1024 * 1024  # 5 MB in bytes # 5
    file_size = os.path.getsize(out_path)        

    width = image.width
    height = image.height

    # Resize images if they are too big
    if file_size > max_size:
        # Start with the original image size          

        print(f"Image size before {width}x{height}, original file_size: {file_size}")

        while file_size > max_size:
            # Reduce the size by a factor (e.g., 50% of the current size)
            new_width = int(width * 0.5)
            new_height = int(height * 0.5)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save the resized image
            image.save(out_path, format="PNG", optimize=True)
            
            # Update the file size
            file_size = os.path.getsize(out_path)
            print(f"Resized to {new_width}x{new_height}, new file_size: {file_size}")
    else:
        new_width = width
        new_height = height
    
    
    all_img_details.append((page_num, image, new_width, new_height))

    return image, new_width, new_height, all_img_details, out_path

def process_single_page_for_image_conversion(pdf_path:str, page_num:int, image_dpi:float=image_dpi, create_images:bool = True, input_folder: str = INPUT_FOLDER) -> tuple[int, str, float, float]:

    out_path_placeholder = "placeholder_image_" + str(page_num) + ".png"

    if create_images == True:
        try:
            # Construct the full output directory path
            image_output_dir = os.path.join(os.getcwd(), input_folder)
            out_path = os.path.join(image_output_dir, f"{os.path.basename(pdf_path)}_{page_num}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            if os.path.exists(out_path):
                # Load existing image
                image = Image.open(out_path)
            elif pdf_path.lower().endswith(".pdf"):
                # Convert PDF page to image
                image_l = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, 
                                            dpi=image_dpi, use_cropbox=False, use_pdftocairo=False)
                image = image_l[0]
                image = image.convert("L")

                image.save(out_path, format="PNG")
            elif pdf_path.lower().endswith(".jpg") or pdf_path.lower().endswith(".png") or pdf_path.lower().endswith(".jpeg"):
                image = Image.open(pdf_path)
                image.save(out_path, format="PNG")

            width, height = image.size

            # Check if image size too large and reduce if necessary
            #print("Checking size of image and reducing if necessary.")
            image, width, height, all_img_details, img_path = check_image_size_and_reduce(out_path, image)                

            return page_num, out_path, width, height

        except Exception as e:
            print(f"Error processing page {page_num + 1}: {e}")
            return page_num,  out_path_placeholder, pd.NA, pd.NA
    else:
        # print("Not creating image for page", page_num)
        return page_num,  out_path_placeholder, pd.NA, pd.NA

def convert_pdf_to_images(pdf_path: str, prepare_for_review:bool=False, page_min: int = 0, page_max:int = 0, create_images:bool=True, image_dpi: float = image_dpi, num_threads: int = 8, input_folder: str = INPUT_FOLDER):

    # If preparing for review, just load the first page (not currently used)
    if prepare_for_review == True:
        page_count = pdfinfo_from_path(pdf_path)['Pages'] #1
        page_min = 0
        page_max = page_count
    else:
        page_count = pdfinfo_from_path(pdf_path)['Pages']

    print(f"Number of pages in PDF: {page_count}")

    # Set page max to length of pdf if not specified
    if page_max == 0: page_max = page_count

    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for page_num in range(page_min, page_max):
            futures.append(executor.submit(process_single_page_for_image_conversion, pdf_path, page_num, image_dpi, create_images=create_images, input_folder=input_folder))
        
        for future in tqdm(as_completed(futures), total=len(futures), unit="pages", desc="Converting pages to image"):
            page_num, img_path, width, height = future.result()
            if img_path:
                results.append((page_num, img_path, width, height))
            else:
                print(f"Page {page_num + 1} failed to process.")
                results.append((page_num, "placeholder_image_" + str(page_num) + ".png", pd.NA, pd.NA))
    
    # Sort results by page number
    results.sort(key=lambda x: x[0])
    images = [result[1] for result in results]
    widths = [result[2] for result in results]
    heights = [result[3] for result in results]

    #print("PDF has been converted to images.")
    return images, widths, heights, results

# Function to take in a file path, decide if it is an image or pdf, then process appropriately.
def process_file_for_image_creation(file_path:str, prepare_for_review:bool=False, input_folder:str=INPUT_FOLDER, create_images:bool=True):
    # Get the file extension
    file_extension = os.path.splitext(file_path)[1].lower()
 
    # Check if the file is an image type
    if file_extension in ['.jpg', '.jpeg', '.png']:
        print(f"{file_path} is an image file.")
        # Perform image processing here
        img_object = [file_path] #[Image.open(file_path)]

        # Load images from the file paths. Test to see if it is bigger than 4.5 mb and reduct if needed (Textract limit is 5mb)
        image = Image.open(file_path)
        img_object, image_sizes_width, image_sizes_height, all_img_details, img_path = check_image_size_and_reduce(file_path, image)

        if not isinstance(image_sizes_width, list):
            img_path = [img_path]
            image_sizes_width = [image_sizes_width]
            image_sizes_height = [image_sizes_height]
            all_img_details = [all_img_details]
            

    # Check if the file is a PDF
    elif file_extension == '.pdf':
        # print(f"{file_path} is a PDF file. Converting to image set")

        # Run your function for processing PDF files here
        img_path, image_sizes_width, image_sizes_height, all_img_details = convert_pdf_to_images(file_path, prepare_for_review, input_folder=input_folder, create_images=create_images)

    else:
        print(f"{file_path} is not an image or PDF file.")
        img_path = []
        image_sizes_width = []
        image_sizes_height = []
        all_img_details = []

    return img_path, image_sizes_width, image_sizes_height, all_img_details

def get_input_file_names(file_input:List[str]):
    '''
    Get list of input files to report to logs.
    '''

    all_relevant_files = []
    file_name_with_extension = ""
    full_file_name = ""
    total_pdf_page_count = 0

    if isinstance(file_input, dict):
        file_input = os.path.abspath(file_input["name"])

    if isinstance(file_input, str):
        file_input_list = [file_input]
    else:
        file_input_list = file_input

    for file in file_input_list:
        if isinstance(file, str):
            file_path = file
        else:
            file_path = file.name

        file_path_without_ext = get_file_name_without_type(file_path)

        file_extension = os.path.splitext(file_path)[1].lower()

        # Check if the file is in acceptable types
        if (file_extension in ['.jpg', '.jpeg', '.png', '.pdf', '.xlsx', '.csv', '.parquet']) & ("review_file" not in file_path_without_ext) & ("ocr_output" not in file_path_without_ext):
            all_relevant_files.append(file_path_without_ext)
            file_name_with_extension = file_path_without_ext + file_extension
            full_file_name = file_path

        # If PDF, get number of pages
        if (file_extension in ['.pdf']):
            # Open the PDF file
            pdf_document = pymupdf.open(file_path)
            # Get the number of pages
            page_count = pdf_document.page_count
            
            # Close the document
            pdf_document.close()
        else:
            page_count = 1

        total_pdf_page_count += page_count
    
    all_relevant_files_str = ", ".join(all_relevant_files)

    return all_relevant_files_str, file_name_with_extension, full_file_name, all_relevant_files, total_pdf_page_count

def convert_color_to_range_0_1(color):
    return tuple(component / 255 for component in color)

def redact_single_box(pymupdf_page:Page, pymupdf_rect:Rect, img_annotation_box:dict, custom_colours:bool=False):
    '''
    Commit redaction boxes to a PyMuPDF page.
    '''

    pymupdf_x1 = pymupdf_rect[0]
    pymupdf_y1 = pymupdf_rect[1]
    pymupdf_x2 = pymupdf_rect[2]
    pymupdf_y2 = pymupdf_rect[3]

    # Calculate area to actually remove text from the pdf (different from black box size)     
    redact_bottom_y = pymupdf_y1 + 2
    redact_top_y = pymupdf_y2 - 2

    # Calculate the middle y value and set a small height if default values are too close together
    if (redact_top_y - redact_bottom_y) < 1:        
        middle_y = (pymupdf_y1 + pymupdf_y2) / 2
        redact_bottom_y = middle_y - 1
        redact_top_y = middle_y + 1


    rect_small_pixel_height = Rect(pymupdf_x1, redact_bottom_y, pymupdf_x2, redact_top_y)  # Slightly smaller than outside box

    # Add the annotation to the middle of the character line, so that it doesn't delete text from adjacent lines
    #page.add_redact_annot(rect)#rect_small_pixel_height)
    pymupdf_page.add_redact_annot(rect_small_pixel_height)

    # Set up drawing a black box over the whole rect
    shape = pymupdf_page.new_shape()
    shape.draw_rect(pymupdf_rect)

    if custom_colours == True:
        if img_annotation_box["color"][0] > 1:
            out_colour = convert_color_to_range_0_1(img_annotation_box["color"])
        else:
            out_colour = img_annotation_box["color"]
    else:
        if CUSTOM_BOX_COLOUR == "grey":
            out_colour = (0.5, 0.5, 0.5)        
        else:
            out_colour = (0,0,0)

    shape.finish(color=out_colour, fill=out_colour)  # Black fill for the rectangle
    #shape.finish(color=(0, 0, 0))  # Black fill for the rectangle
    shape.commit()

def convert_pymupdf_to_image_coords(pymupdf_page:Page, x1:float, y1:float, x2:float, y2:float, image: Image=None, image_dimensions:dict={}):
    '''
    Converts coordinates from pymupdf format to image coordinates,
    accounting for mediabox dimensions and offset.
    '''
    # Get rect dimensions
    rect = pymupdf_page.rect
    rect_width = rect.width
    rect_height = rect.height
    
    # Get mediabox dimensions and position
    mediabox = pymupdf_page.mediabox
    mediabox_width = mediabox.width
    mediabox_height = mediabox.height
    
    # Get target image dimensions
    if image:
        image_page_width, image_page_height = image.size
    elif image_dimensions:
        image_page_width, image_page_height = image_dimensions['image_width'], image_dimensions['image_height']
    else:
        image_page_width, image_page_height = mediabox_width, mediabox_height

    # Calculate scaling factors
    image_to_mediabox_x_scale = image_page_width / mediabox_width
    image_to_mediabox_y_scale = image_page_height / mediabox_height

    # Adjust coordinates:
    # Apply scaling to match image dimensions
    x1_image = x1 * image_to_mediabox_x_scale    
    x2_image = x2 * image_to_mediabox_x_scale
    y1_image = y1 * image_to_mediabox_y_scale
    y2_image = y2 * image_to_mediabox_y_scale

    # Correct for difference in rect and mediabox size
    if mediabox_width != rect_width:
        
        mediabox_to_rect_x_scale = mediabox_width / rect_width
        mediabox_to_rect_y_scale = mediabox_height / rect_height

        rect_to_mediabox_x_scale = rect_width / mediabox_width
        #rect_to_mediabox_y_scale = rect_height / mediabox_height

        mediabox_rect_x_diff = (mediabox_width - rect_width) * (image_to_mediabox_x_scale / 2)
        mediabox_rect_y_diff = (mediabox_height - rect_height) * (image_to_mediabox_y_scale / 2)

        x1_image -= mediabox_rect_x_diff
        x2_image -= mediabox_rect_x_diff
        y1_image += mediabox_rect_y_diff
        y2_image += mediabox_rect_y_diff

        #
        x1_image *= mediabox_to_rect_x_scale
        x2_image *= mediabox_to_rect_x_scale
        y1_image *= mediabox_to_rect_y_scale
        y2_image *= mediabox_to_rect_y_scale

    return x1_image, y1_image, x2_image, y2_image

def redact_whole_pymupdf_page(rect_height:float, rect_width:float, image:Image, page:Page, custom_colours, border:float = 5, image_dimensions:dict={}):
    # Small border to page that remains white
    border = 5
    # Define the coordinates for the Rect
    whole_page_x1, whole_page_y1 = 0 + border, 0 + border  # Bottom-left corner
    whole_page_x2, whole_page_y2 = rect_width - border, rect_height - border  # Top-right corner

    # whole_page_image_x1, whole_page_image_y1, whole_page_image_x2, whole_page_image_y2 = convert_pymupdf_to_image_coords(page, whole_page_x1, whole_page_y1, whole_page_x2, whole_page_y2, image, image_dimensions=image_dimensions)

    # Create new image annotation element based on whole page coordinates
    whole_page_rect = Rect(whole_page_x1, whole_page_y1, whole_page_x2, whole_page_y2)

    # Write whole page annotation to annotation boxes
    whole_page_img_annotation_box = {}
    whole_page_img_annotation_box["xmin"] = whole_page_x1 #whole_page_image_x1
    whole_page_img_annotation_box["ymin"] = whole_page_y1 #whole_page_image_y1
    whole_page_img_annotation_box["xmax"] = whole_page_x2 #whole_page_image_x2
    whole_page_img_annotation_box["ymax"] =  whole_page_y2 #whole_page_image_y2
    whole_page_img_annotation_box["color"] = (0,0,0)
    whole_page_img_annotation_box["label"] = "Whole page"

    redact_single_box(page, whole_page_rect, whole_page_img_annotation_box, custom_colours)

    return whole_page_img_annotation_box

def create_page_size_objects(pymupdf_doc:Document, image_sizes_width:List[float], image_sizes_height:List[float], image_file_paths:List[str]):
    page_sizes = []
    original_cropboxes = []

    for page_no, page in enumerate(pymupdf_doc):
        reported_page_no = page_no + 1
        
        pymupdf_page = pymupdf_doc.load_page(page_no)
        original_cropboxes.append(pymupdf_page.cropbox)  # Save original CropBox

        # Create a page_sizes_object. If images have been created, then image width an height come from this value. Otherwise, they are set to the cropbox size        
        out_page_image_sizes = {
            "page":reported_page_no,                                            
            "mediabox_width":pymupdf_page.mediabox.width,
            "mediabox_height": pymupdf_page.mediabox.height,
            "cropbox_width":pymupdf_page.cropbox.width,
            "cropbox_height":pymupdf_page.cropbox.height,
            "original_cropbox":original_cropboxes[-1],
            "image_path":image_file_paths[page_no]}
        
        # cropbox_x_offset: Distance from MediaBox left edge to CropBox left edge
        # This is simply the difference in their x0 coordinates.
        out_page_image_sizes['cropbox_x_offset'] = pymupdf_page.cropbox.x0 - pymupdf_page.mediabox.x0

        # cropbox_y_offset_from_top: Distance from MediaBox top edge to CropBox top edge
        # MediaBox top y = mediabox.y1
        # CropBox top y = cropbox.y1
        # The difference is mediabox.y1 - cropbox.y1
        out_page_image_sizes['cropbox_y_offset_from_top'] = pymupdf_page.mediabox.y1 - pymupdf_page.cropbox.y1
        
        if image_sizes_width and image_sizes_height:
            out_page_image_sizes["image_width"] = image_sizes_width[page_no]
            out_page_image_sizes["image_height"] = image_sizes_height[page_no]        
        
        page_sizes.append(out_page_image_sizes)

    return page_sizes, original_cropboxes

def prepare_image_or_pdf(
    file_paths: List[str],
    in_redact_method: str,
    latest_file_completed: int = 0,
    out_message: List[str] = [],
    first_loop_state: bool = False,
    number_of_pages:int = 0,
    all_annotations_object:List = [],
    prepare_for_review:bool = False,
    in_fully_redacted_list:List[int]=[],
    output_folder:str=OUTPUT_FOLDER,
    input_folder:str=INPUT_FOLDER,
    prepare_images:bool=True,
    page_sizes:list[dict]=[],
    textract_output_found:bool = False,
    local_ocr_output_found:bool = False,   
    progress: Progress = Progress(track_tqdm=True)
) -> tuple[List[str], List[str]]:
    """
    Prepare and process image or text PDF files for redaction.

    This function takes a list of file paths, processes each file based on the specified redaction method,
    and returns the output messages and processed file paths.

    Args:
        file_paths (List[str]): List of file paths to process.
        in_redact_method (str): The redaction method to use.
        latest_file_completed (optional, int): Index of the last completed file.
        out_message (optional, List[str]): List to store output messages.
        first_loop_state (optional, bool): Flag indicating if this is the first iteration.
        number_of_pages (optional, int): integer indicating the number of pages in the document
        all_annotations_object(optional, List of annotation objects): All annotations for current document
        prepare_for_review(optional, bool): Is this preparation step preparing pdfs and json files to review current redactions?
        in_fully_redacted_list(optional, List of int): A list of pages to fully redact
        output_folder (optional, str): The output folder for file save
        prepare_images (optional, bool): A boolean indicating whether to create images for each PDF page. Defaults to True.
        page_sizes(optional, List[dict]): A list of dicts containing information about page sizes in various formats.
        textract_output_found (optional, bool): A boolean indicating whether Textract analysis output has already been found. Defaults to False.
        local_ocr_output_found (optional, bool): A boolean indicating whether local OCR analysis output has already been found. Defaults to False.
        progress (optional, Progress): Progress tracker for the operation
        

    Returns:
        tuple[List[str], List[str]]: A tuple containing the output messages and processed file paths.
    """

    tic = time.perf_counter()
    json_from_csv = False
    original_cropboxes = []  # Store original CropBox values
    converted_file_paths = []
    image_file_paths = []
    pymupdf_doc = []
    all_img_details = []    
    review_file_csv = pd.DataFrame()
    all_line_level_ocr_results_df = pd.DataFrame()
    out_textract_path = ""
    combined_out_message = ""
    final_out_message = ""

    if isinstance(in_fully_redacted_list, pd.DataFrame):
        if not in_fully_redacted_list.empty:
            in_fully_redacted_list = in_fully_redacted_list.iloc[:,0].tolist()

    # If this is the first time around, set variables to 0/blank
    if first_loop_state==True:
        latest_file_completed = 0
        out_message = []
        all_annotations_object = []
    else:
        print("Now redacting file", str(latest_file_completed))
  
    # If combined out message or converted_file_paths are blank, change to a list so it can be appended to
    if isinstance(out_message, str): out_message = [out_message]

    if not file_paths: file_paths = []

    if isinstance(file_paths, dict): file_paths = os.path.abspath(file_paths["name"])

    if isinstance(file_paths, str): file_path_number = 1
    else: file_path_number = len(file_paths)
    
    latest_file_completed = int(latest_file_completed)

    # If we have already redacted the last file, return the input out_message and file list to the relevant components
    if latest_file_completed >= file_path_number:
        print("Last file reached, returning files:", str(latest_file_completed))
        if isinstance(out_message, list):
            final_out_message = '\n'.join(out_message)
        else:
            final_out_message = out_message
        return final_out_message, converted_file_paths, image_file_paths, number_of_pages, number_of_pages, pymupdf_doc, all_annotations_object, review_file_csv, original_cropboxes, page_sizes, textract_output_found, all_img_details, all_line_level_ocr_results_df, local_ocr_output_found

    progress(0.1, desc='Preparing file')

    if isinstance(file_paths, str):
        file_paths_list = [file_paths]
        file_paths_loop = file_paths_list
    else:
        file_paths_list = file_paths
        file_paths_loop = sorted(file_paths_list, key=lambda x: (os.path.splitext(x)[1] != '.pdf', os.path.splitext(x)[1] != '.json')) 
        
    # Loop through files to load in
    for file in file_paths_loop:
        converted_file_path = []
        image_file_path = []

        if isinstance(file, str):
            file_path = file
        else:
            file_path = file.name
        file_path_without_ext = get_file_name_without_type(file_path)
        file_name_with_ext = os.path.basename(file_path)

        if not file_path:
            out_message = "Please select a file."
            print(out_message)
            raise Exception(out_message)
            
        file_extension = os.path.splitext(file_path)[1].lower()

        # If a pdf, load as a pymupdf document
        if is_pdf(file_path):
            pymupdf_doc = pymupdf.open(file_path)
            pymupdf_pages = pymupdf_doc.page_count

            converted_file_path = file_path

            if prepare_images==True:
                image_file_paths, image_sizes_width, image_sizes_height, all_img_details = process_file_for_image_creation(file_path, prepare_for_review, input_folder, create_images=True)
            else:
                image_file_paths, image_sizes_width, image_sizes_height, all_img_details = process_file_for_image_creation(file_path, prepare_for_review, input_folder, create_images=False)
            
            page_sizes, original_cropboxes = create_page_size_objects(pymupdf_doc, image_sizes_width, image_sizes_height, image_file_paths)

            #Create base version of the annotation object that doesn't have any annotations in it
            if (not all_annotations_object) & (prepare_for_review == True):
                all_annotations_object = []

                for image_path in image_file_paths:
                    annotation = {}
                    annotation["image"] = image_path
                    annotation["boxes"] = []

                    all_annotations_object.append(annotation)
            
        elif is_pdf_or_image(file_path):  # Alternatively, if it's an image
            # Check if the file is an image type and the user selected text ocr option
            if file_extension in ['.jpg', '.jpeg', '.png'] and in_redact_method == SELECTABLE_TEXT_EXTRACT_OPTION:
                in_redact_method = TESSERACT_TEXT_EXTRACT_OPTION

            # Convert image to a pymupdf document
            pymupdf_doc = pymupdf.open()  # Create a new empty document

            img = Image.open(file_path)  # Open the image file
            rect = pymupdf.Rect(0, 0, img.width, img.height)  # Create a rectangle for the image
            pymupdf_page = pymupdf_doc.new_page(width=img.width, height=img.height)  # Add a new page
            pymupdf_page.insert_image(rect, filename=file_path)  # Insert the image into the page
            pymupdf_page = pymupdf_doc.load_page(0)

            file_path_str = str(file_path)

            image_file_paths, image_sizes_width, image_sizes_height, all_img_details = process_file_for_image_creation(file_path_str, prepare_for_review, input_folder, create_images=True)

            # Create a page_sizes_object
            page_sizes, original_cropboxes = create_page_size_objects(pymupdf_doc, image_sizes_width, image_sizes_height, image_file_paths)

            converted_file_path = output_folder + file_name_with_ext

            pymupdf_doc.save(converted_file_path, garbage=4, deflate=True, clean=True)

        elif file_extension in ['.csv']:
            if '_review_file' in file_path_without_ext:
                review_file_csv = read_file(file_path)                
                all_annotations_object = convert_review_df_to_annotation_json(review_file_csv, image_file_paths, page_sizes)
                json_from_csv = True
                #print("Converted CSV review file to image annotation object")
            elif '_ocr_output' in file_path_without_ext:
                all_line_level_ocr_results_df = read_file(file_path)                
                json_from_csv = False

        # NEW IF STATEMENT
        # If the file name ends with .json, check if we are loading for review. If yes, assume it is an annoations object, overwrite the current annotations object. If false, assume this is a Textract object, load in to Textract

        if (file_extension in ['.json']) | (json_from_csv == True):

            if (file_extension in ['.json']) &  (prepare_for_review == True):
                if isinstance(file_path, str):
                    with open(file_path, 'r') as json_file:
                        all_annotations_object = json.load(json_file)
                else:
                    # Assuming file_path is a NamedString or similar
                    all_annotations_object = json.loads(file_path)  # Use loads for string content

            # Save Textract file to folder
            elif (file_extension in ['.json']) and '_textract' in file_path_without_ext: #(prepare_for_review != True):
                print("Saving Textract output")
                # Copy it to the output folder so it can be used later.
                output_textract_json_file_name = file_path_without_ext
                if not file_path.endswith("_textract.json"): output_textract_json_file_name = file_path_without_ext + "_textract.json"
                else: output_textract_json_file_name = file_path_without_ext + ".json"

                out_textract_path = os.path.join(output_folder, output_textract_json_file_name)

                # Use shutil to copy the file directly
                shutil.copy2(file_path, out_textract_path)  # Preserves metadata
                textract_output_found = True                
                continue

            elif (file_extension in ['.json']) and '_ocr_results_with_words' in file_path_without_ext: #(prepare_for_review != True):
                print("Saving local OCR output")
                # Copy it to the output folder so it can be used later.
                output_ocr_results_with_words_json_file_name = file_path_without_ext
                if not file_path.endswith("_ocr_results_with_words.json"): output_ocr_results_with_words_json_file_name = file_path_without_ext + "_ocr_results_with_words.json"
                else: output_ocr_results_with_words_json_file_name = file_path_without_ext + ".json"

                out_ocr_results_with_words_path = os.path.join(output_folder, output_ocr_results_with_words_json_file_name)

                # Use shutil to copy the file directly
                shutil.copy2(file_path, out_ocr_results_with_words_path)  # Preserves metadata
                local_ocr_output_found = True                
                continue

            # NEW IF STATEMENT
            # If you have an annotations object from the above code
            if all_annotations_object:

                # Get list of page numbers
                image_file_paths_pages = [
                int(re.search(r'_(\d+)\.png$', os.path.basename(s)).group(1)) 
                for s in image_file_paths 
                if re.search(r'_(\d+)\.png$', os.path.basename(s))
                ]
                image_file_paths_pages = [int(i) for i in image_file_paths_pages]
                
                # If PDF pages have been converted to image files, replace the current image paths in the json to this. 
                if image_file_paths:
                    for i, image_file_path in enumerate(image_file_paths):

                        if i < len(all_annotations_object): 
                            annotation = all_annotations_object[i]
                        else: 
                            annotation = {}
                            all_annotations_object.append(annotation)

                        try:
                            if not annotation:
                                annotation = {"image":"", "boxes": []}
                                annotation_page_number = int(re.search(r'_(\d+)\.png$', image_file_path).group(1))
                            else:
                                annotation_page_number = int(re.search(r'_(\d+)\.png$', annotation["image"]).group(1))
                        except Exception as e:
                            print("Extracting page number from image failed due to:", e)
                            annotation_page_number = 0

                        # Check if the annotation page number exists in the image file paths pages
                        if annotation_page_number in image_file_paths_pages:

                            # Set the correct image page directly since we know it's in the list
                            correct_image_page = annotation_page_number
                            annotation["image"] = image_file_paths[correct_image_page]
                        else:
                            print("Page", annotation_page_number, "image file not found.")

                        all_annotations_object[i] = annotation
                
                if isinstance(in_fully_redacted_list, list):
                    in_fully_redacted_list = pd.DataFrame(data={"fully_redacted_pages_list":in_fully_redacted_list})

                # Get list of pages that are to be fully redacted and redact them
                if not in_fully_redacted_list.empty:
                    print("Redacting whole pages")

                    for i, image in enumerate(image_file_paths):
                        page = pymupdf_doc.load_page(i)
                        rect_height = page.rect.height
                        rect_width = page.rect.width 
                        whole_page_img_annotation_box = redact_whole_pymupdf_page(rect_height, rect_width, image, page, custom_colours = False, border = 5, image_dimensions={"image_width":image_sizes_width[i], "image_height":image_sizes_height[i]})

                        all_annotations_object.append(whole_page_img_annotation_box)

                # Write the response to a JSON file in output folder
                out_folder = output_folder + file_path_without_ext + ".json"
                # with open(out_folder, 'w') as json_file:
                #     json.dump(all_annotations_object, json_file, separators=(",", ":"))
                continue

        # If it's a zip, it could be extract from a Textract bulk API call. Check it's this, and load in json if found
        elif file_extension in ['.zip']:

            # Assume it's a Textract response object. Copy it to the output folder so it can be used later.
            out_folder = os.path.join(output_folder, file_path_without_ext + "_textract.json")

            # Use shutil to copy the file directly
            # Open the ZIP file to check its contents
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                json_files = [f for f in zip_ref.namelist() if f.lower().endswith('.json')]

                if len(json_files) == 1:  # Ensure only one JSON file exists
                    json_filename = json_files[0]

                    # Extract the JSON file to the same directory as the ZIP file
                    extracted_path = os.path.join(os.path.dirname(file_path), json_filename)
                    zip_ref.extract(json_filename, os.path.dirname(file_path))

                    # Move the extracted JSON to the intended output location
                    shutil.move(extracted_path, out_folder)

                    textract_output_found = True
                else:
                    print(f"Skipping {file_path}: Expected 1 JSON file, found {len(json_files)}")

        elif file_extension in ['.csv'] and "ocr_output" in file_path:
            continue

        # Must be something else, return with error message
        else:
            if in_redact_method == TESSERACT_TEXT_EXTRACT_OPTION or in_redact_method == TEXTRACT_TEXT_EXTRACT_OPTION:
                if is_pdf_or_image(file_path) == False:
                    out_message = "Please upload a PDF file or image file (JPG, PNG) for image analysis."
                    print(out_message)
                    raise Exception(out_message)

            elif in_redact_method == SELECTABLE_TEXT_EXTRACT_OPTION:
                if is_pdf(file_path) == False:
                    out_message = "Please upload a PDF file for text analysis."
                    print(out_message)
                    raise Exception(out_message)

        converted_file_paths.append(converted_file_path)
        image_file_paths.extend(image_file_path)        

        toc = time.perf_counter()
        out_time = f"File '{file_path_without_ext}' prepared in {toc - tic:0.1f} seconds."

        print(out_time)

        out_message.append(out_time)
        combined_out_message = '\n'.join(out_message)

    number_of_pages = len(page_sizes)#len(image_file_paths)
        
    return combined_out_message, converted_file_paths, image_file_paths, number_of_pages, number_of_pages, pymupdf_doc, all_annotations_object, review_file_csv, original_cropboxes, page_sizes, textract_output_found, all_img_details, all_line_level_ocr_results_df, local_ocr_output_found

def load_and_convert_ocr_results_with_words_json(ocr_results_with_words_json_file_path:str, log_files_output_paths:str, page_sizes_df:pd.DataFrame):
    """
    Loads Textract JSON from a file, detects if conversion is needed, and converts if necessary.
    """
    
    if not os.path.exists(ocr_results_with_words_json_file_path):
        print("No existing OCR results file found.")
        return [], True, log_files_output_paths  # Return empty dict and flag indicating missing file
    
    no_ocr_results_with_words_file = False
    print("Found existing OCR results json results file.")

    # Track log files
    if ocr_results_with_words_json_file_path not in log_files_output_paths:
        log_files_output_paths.append(ocr_results_with_words_json_file_path)

    try:
        with open(ocr_results_with_words_json_file_path, 'r', encoding='utf-8') as json_file:
            ocr_results_with_words_data = json.load(json_file)
    except json.JSONDecodeError:
        print("Error: Failed to parse OCR results JSON file. Returning empty data.")
        return [], True, log_files_output_paths  # Indicate failure

    # Check if conversion is needed
    if "page" and "results" in ocr_results_with_words_data[0]:
        print("JSON already in the correct format for app. No changes needed.")
        return ocr_results_with_words_data, False, log_files_output_paths  # No conversion required

    else:
        print("Invalid OCR result JSON format: 'page' or 'results' key missing.")
        #print("OCR results with words data:", ocr_results_with_words_data)
        return [], True, log_files_output_paths  # Return empty data if JSON is not recognized

def convert_text_pdf_to_img_pdf(in_file_path:str, out_text_file_path:List[str], image_dpi:float=image_dpi, output_folder:str=OUTPUT_FOLDER, input_folder:str=INPUT_FOLDER):
    file_path_without_ext = get_file_name_without_type(in_file_path)

    out_file_paths = out_text_file_path

    # Convert annotated text pdf back to image to give genuine redactions   
    pdf_text_image_paths, image_sizes_width, image_sizes_height, all_img_details = process_file_for_image_creation(out_file_paths[0], input_folder=input_folder)
    out_text_image_file_path = output_folder + file_path_without_ext + "_text_redacted_as_img.pdf"
    pdf_text_image_paths[0].save(out_text_image_file_path, "PDF" ,resolution=image_dpi, save_all=True, append_images=pdf_text_image_paths[1:])

    out_file_paths = [out_text_image_file_path]

    out_message = "PDF " + file_path_without_ext + " converted to image-based file."
    print(out_message)

    return out_message, out_file_paths

def save_pdf_with_or_without_compression(pymupdf_doc:object, out_redacted_pdf_file_path, COMPRESS_REDACTED_PDF:bool=COMPRESS_REDACTED_PDF):
    '''
    Save a pymupdf document with basic cleaning or with full compression options. Can be useful for low memory systems to do minimal cleaning to avoid crashing with large PDFs.
    '''
    if COMPRESS_REDACTED_PDF == True:
        pymupdf_doc.save(out_redacted_pdf_file_path, garbage=4, deflate=True, clean=True)
    else:
        pymupdf_doc.save(out_redacted_pdf_file_path, garbage=1, clean=True)

def join_values_within_threshold(df1:pd.DataFrame, df2:pd.DataFrame):
    # Threshold for matching
    threshold = 5

    # Perform a cross join
    df1['key'] = 1
    df2['key'] = 1
    merged = pd.merge(df1, df2, on='key').drop(columns=['key'])

    # Apply conditions for all columns
    conditions = (
        (abs(merged['xmin_x'] - merged['xmin_y']) <= threshold) &
        (abs(merged['xmax_x'] - merged['xmax_y']) <= threshold) &
        (abs(merged['ymin_x'] - merged['ymin_y']) <= threshold) &
        (abs(merged['ymax_x'] - merged['ymax_y']) <= threshold)
    )

    # Filter rows that satisfy all conditions
    filtered = merged[conditions]

    # Drop duplicates if needed (e.g., keep only the first match for each row in df1)
    result = filtered.drop_duplicates(subset=['xmin_x', 'xmax_x', 'ymin_x', 'ymax_x'])

    # Merge back into the original DataFrame (if necessary)
    final_df = pd.merge(df1, result, left_on=['xmin', 'xmax', 'ymin', 'ymax'], right_on=['xmin_x', 'xmax_x', 'ymin_x', 'ymax_x'], how='left')

    # Clean up extra columns
    final_df = final_df.drop(columns=['key'])

def remove_duplicate_images_with_blank_boxes(data: List[dict]) -> List[dict]:
    '''
    Remove items from the annotator object where the same page exists twice.
    '''
    # Group items by 'image'
    image_groups = defaultdict(list)
    for item in data:
        image_groups[item['image']].append(item)

    # Process each group to prioritize items with non-empty boxes
    result = []
    for image, items in image_groups.items():
        # Filter items with non-empty boxes
        non_empty_boxes = [item for item in items if item.get('boxes')]

         # Remove 'text' elements from boxes (deprecated)
        #for item in non_empty_boxes:
        #    if 'boxes' in item:
        #        item['boxes'] = [{k: v for k, v in box.items() if k != 'text'} for box in item['boxes']]

        if non_empty_boxes:
            # Keep the first entry with non-empty boxes
            result.append(non_empty_boxes[0])
        else:
            # If all items have empty or missing boxes, keep the first item
            result.append(items[0])

    return result

def divide_coordinates_by_page_sizes(
    review_file_df: pd.DataFrame,
    page_sizes_df: pd.DataFrame,
    xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"
) -> pd.DataFrame:
    """
    Optimized function to convert absolute image coordinates (>1) to relative coordinates (<=1).

    Identifies rows with absolute coordinates, merges page size information,
    divides coordinates by dimensions, and combines with already-relative rows.

    Args:
        review_file_df: Input DataFrame with potentially mixed coordinate systems.
        page_sizes_df: DataFrame with page dimensions ('page', 'image_width',
                       'image_height', 'mediabox_width', 'mediabox_height').
        xmin, xmax, ymin, ymax: Names of the coordinate columns.

    Returns:
        DataFrame with coordinates converted to relative system, sorted.
    """
    if review_file_df.empty or xmin not in review_file_df.columns:
        return review_file_df # Return early if empty or key column missing

    # --- Initial Type Conversion ---
    coord_cols = [xmin, xmax, ymin, ymax]
    cols_to_convert = coord_cols + ["page"]
    temp_df = review_file_df.copy() # Work on a copy initially

    for col in cols_to_convert:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce")
        else:
            # If essential 'page' or coord column missing, cannot proceed meaningfully
            if col == 'page' or col in coord_cols:
                 print(f"Warning: Required column '{col}' not found in review_file_df. Returning original DataFrame.")
                 return review_file_df

    # --- Identify Absolute Coordinates ---
    # Create mask for rows where *all* coordinates are potentially absolute (> 1)
    # Handle potential NaNs introduced by to_numeric - treat NaN as not absolute.
    is_absolute_mask = (
        (temp_df[xmin] > 1) & (temp_df[xmin].notna()) &
        (temp_df[xmax] > 1) & (temp_df[xmax].notna()) &
        (temp_df[ymin] > 1) & (temp_df[ymin].notna()) &
        (temp_df[ymax] > 1) & (temp_df[ymax].notna())
    )

    # --- Separate DataFrames ---
    df_rel = temp_df[~is_absolute_mask] # Rows already relative or with NaN/mixed coords
    df_abs = temp_df[is_absolute_mask].copy() # Absolute rows - COPY here to allow modifications

    # --- Process Absolute Coordinates ---
    if not df_abs.empty:
        # Merge page sizes if necessary
        if "image_width" not in df_abs.columns and not page_sizes_df.empty:
            ps_df_copy = page_sizes_df.copy() # Work on a copy of page sizes

            # Ensure page is numeric for merge key matching
            ps_df_copy['page'] = pd.to_numeric(ps_df_copy['page'], errors='coerce')

            # Columns to merge from page_sizes
            merge_cols = ['page', 'image_width', 'image_height', 'mediabox_width', 'mediabox_height']
            available_merge_cols = [col for col in merge_cols if col in ps_df_copy.columns]

            # Prepare dimension columns in the copy
            for col in ['image_width', 'image_height', 'mediabox_width', 'mediabox_height']:
                 if col in ps_df_copy.columns:
                     # Replace "<NA>" string if present
                     if ps_df_copy[col].dtype == 'object':
                          ps_df_copy[col] = ps_df_copy[col].replace("<NA>", pd.NA)
                     # Convert to numeric
                     ps_df_copy[col] = pd.to_numeric(ps_df_copy[col], errors='coerce')

            # Perform the merge
            if 'page' in available_merge_cols: # Check if page exists for merging
                df_abs = df_abs.merge(
                    ps_df_copy[available_merge_cols],
                    on="page",
                    how="left"
                )
            else:
                 print("Warning: 'page' column not found in page_sizes_df. Cannot merge dimensions.")


        # Fallback to mediabox dimensions if image dimensions are missing
        if "image_width" in df_abs.columns and "mediabox_width" in df_abs.columns:
             # Check if image_width mostly missing - use .isna().all() or check percentage
             if df_abs["image_width"].isna().all():
                 #print("Falling back to mediabox dimensions as image_width is entirely missing.")
                 df_abs["image_width"] = df_abs["image_width"].fillna(df_abs["mediabox_width"])
                 df_abs["image_height"] = df_abs["image_height"].fillna(df_abs["mediabox_height"])
             else:
                  # Optional: Fill only missing image dims if some exist?
                  # df_abs["image_width"].fillna(df_abs["mediabox_width"], inplace=True)
                  # df_abs["image_height"].fillna(df_abs["mediabox_height"], inplace=True)
                  pass # Current logic only falls back if ALL image_width are NaN

        # Ensure divisor columns are numeric before division
        divisors_numeric = True
        for col in ["image_width", "image_height"]:
            if col in df_abs.columns:
                 df_abs[col] = pd.to_numeric(df_abs[col], errors='coerce')
            else:
                 print(f"Warning: Dimension column '{col}' missing. Cannot perform division.")
                 divisors_numeric = False


        # Perform division if dimensions are available and numeric
        if divisors_numeric and "image_width" in df_abs.columns and "image_height" in df_abs.columns:
             # Use np.errstate to suppress warnings about division by zero or NaN if desired
             with np.errstate(divide='ignore', invalid='ignore'):
                df_abs[xmin] = df_abs[xmin] / df_abs["image_width"]
                df_abs[xmax] = df_abs[xmax] / df_abs["image_width"]
                df_abs[ymin] = df_abs[ymin] / df_abs["image_height"]
                df_abs[ymax] = df_abs[ymax] / df_abs["image_height"]
                # Replace potential infinities with NaN (optional, depending on desired outcome)
                df_abs.replace([np.inf, -np.inf], np.nan, inplace=True)
        else:
             print("Skipping coordinate division due to missing or non-numeric dimension columns.")


    # --- Combine Relative and Processed Absolute DataFrames ---
    dfs_to_concat = [df for df in [df_rel, df_abs] if not df.empty]

    if dfs_to_concat:
        final_df = pd.concat(dfs_to_concat, ignore_index=True)
    else:
        # If both splits were empty, return an empty DF with original columns
        print("Warning: Both relative and absolute splits resulted in empty DataFrames.")
        final_df = pd.DataFrame(columns=review_file_df.columns)


    # --- Final Sort ---
    required_sort_columns = {"page", xmin, ymin}
    if not final_df.empty and required_sort_columns.issubset(final_df.columns):
        # Ensure sort columns are numeric before sorting
        final_df['page'] = pd.to_numeric(final_df['page'], errors='coerce')
        final_df[ymin] = pd.to_numeric(final_df[ymin], errors='coerce')
        final_df[xmin] = pd.to_numeric(final_df[xmin], errors='coerce')
        # Sort by page, ymin, xmin (note order compared to multiply function)
        final_df.sort_values(["page", ymin, xmin], inplace=True, na_position='last')


    # --- Clean Up Columns ---
    # Correctly drop columns and reassign the result
    cols_to_drop = ["image_width", "image_height", "mediabox_width", "mediabox_height"]
    final_df = final_df.drop(columns=cols_to_drop, errors="ignore")

    return final_df

def multiply_coordinates_by_page_sizes(
    review_file_df: pd.DataFrame,
    page_sizes_df: pd.DataFrame,
    xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"
):
    """
    Optimized function to convert relative coordinates to absolute based on page sizes.

    Separates relative (<=1) and absolute (>1) coordinates, merges page sizes
    for relative coordinates, calculates absolute pixel values, and recombines.
    """
    if review_file_df.empty or xmin not in review_file_df.columns:
        return review_file_df # Return early if empty or key column missing

    coord_cols = [xmin, xmax, ymin, ymax]
    # Initial type conversion for coordinates and page
    for col in coord_cols + ["page"]:
        if col in review_file_df.columns:
             # Use astype for potentially faster conversion if confident,
             # but to_numeric is safer for mixed types/errors
            review_file_df[col] = pd.to_numeric(review_file_df[col], errors="coerce")

    # --- Identify relative coordinates ---
    # Create mask for rows where *all* coordinates are potentially relative (<= 1)
    # Handle potential NaNs introduced by to_numeric - treat NaN as not relative here.
    is_relative_mask = (
        (review_file_df[xmin].le(1) & review_file_df[xmin].notna()) &
        (review_file_df[xmax].le(1) & review_file_df[xmax].notna()) &
        (review_file_df[ymin].le(1) & review_file_df[ymin].notna()) &
        (review_file_df[ymax].le(1) & review_file_df[ymax].notna())
    )

    # Separate DataFrames (minimal copies)
    df_abs = review_file_df[~is_relative_mask].copy() # Keep absolute rows separately
    df_rel = review_file_df[is_relative_mask].copy()  # Work only with relative rows

    if df_rel.empty:
        # If no relative coordinates, just sort and return absolute ones (if any)
        if not df_abs.empty and {"page", xmin, ymin}.issubset(df_abs.columns):
             df_abs.sort_values(["page", xmin, ymin], inplace=True, na_position='last')
        return df_abs

    # --- Process relative coordinates ---
    if "image_width" not in df_rel.columns and not page_sizes_df.empty:
        # Prepare page_sizes_df for merge
        page_sizes_df = page_sizes_df.copy() # Avoid modifying original page_sizes_df
        page_sizes_df['page'] = pd.to_numeric(page_sizes_df['page'], errors='coerce')
        # Ensure proper NA handling for image dimensions
        page_sizes_df[['image_width', 'image_height']] = page_sizes_df[['image_width','image_height']].replace("<NA>", pd.NA)
        page_sizes_df['image_width'] = pd.to_numeric(page_sizes_df['image_width'], errors='coerce')
        page_sizes_df['image_height'] = pd.to_numeric(page_sizes_df['image_height'], errors='coerce')

        # Merge page sizes
        df_rel = df_rel.merge(
            page_sizes_df[['page', 'image_width', 'image_height']],
            on="page",
            how="left"
        )

    # Multiply coordinates where image dimensions are available
    if "image_width" in df_rel.columns:
        # Create mask for rows in df_rel that have valid image dimensions
        has_size_mask = df_rel["image_width"].notna() & df_rel["image_height"].notna()

        # Apply multiplication using .loc and the mask (vectorized and efficient)
        # Ensure columns are numeric before multiplication (might be redundant if types are good)
        # df_rel.loc[has_size_mask, coord_cols + ['image_width', 'image_height']] = df_rel.loc[has_size_mask, coord_cols + ['image_width', 'image_height']].apply(pd.to_numeric, errors='coerce')

        df_rel.loc[has_size_mask, xmin] *= df_rel.loc[has_size_mask, "image_width"]
        df_rel.loc[has_size_mask, xmax] *= df_rel.loc[has_size_mask, "image_width"]
        df_rel.loc[has_size_mask, ymin] *= df_rel.loc[has_size_mask, "image_height"]
        df_rel.loc[has_size_mask, ymax] *= df_rel.loc[has_size_mask, "image_height"]


    # --- Combine absolute and processed relative DataFrames ---
    # Use list comprehension to handle potentially empty DataFrames
    dfs_to_concat = [df for df in [df_abs, df_rel] if not df.empty]

    if not dfs_to_concat:
        return pd.DataFrame() # Return empty if both are empty

    final_df = pd.concat(dfs_to_concat, ignore_index=True) # ignore_index is good practice after filtering/concat

    # --- Final Sort ---
    required_sort_columns = {"page", xmin, ymin}
    if not final_df.empty and required_sort_columns.issubset(final_df.columns):
        # Handle potential NaNs in sort columns gracefully
        final_df.sort_values(["page", xmin, ymin], inplace=True, na_position='last')

    return final_df

def do_proximity_match_by_page_for_text(df1:pd.DataFrame, df2:pd.DataFrame):
    '''
    Match text from one dataframe to another based on proximity matching of coordinates page by page.
    '''

    if not 'text' in df2.columns: df2['text'] = ''
    if not 'text' in df1.columns: df1['text'] = ''

    # Create a unique key based on coordinates and label for exact merge
    merge_keys = ['xmin', 'ymin', 'xmax', 'ymax', 'label', 'page']
    df1['key'] = df1[merge_keys].astype(str).agg('_'.join, axis=1)
    df2['key'] = df2[merge_keys].astype(str).agg('_'.join, axis=1)

    # Attempt exact merge first
    merged_df = df1.merge(df2[['key', 'text']], on='key', how='left', suffixes=('', '_duplicate'))

    # If a match is found, keep that text; otherwise, keep the original df1 text
    merged_df['text'] = np.where(
        merged_df['text'].isna() | (merged_df['text'] == ''),
        merged_df.pop('text_duplicate'),
        merged_df['text']
    )

    # Define tolerance for proximity matching
    tolerance = 0.02

    # Precompute KDTree for each page in df2
    page_trees = {}
    for page in df2['page'].unique():
        df2_page = df2[df2['page'] == page]
        coords = df2_page[['xmin', 'ymin', 'xmax', 'ymax']].values
        if np.all(np.isfinite(coords)) and len(coords) > 0:
            page_trees[page] = (cKDTree(coords), df2_page)

    # Perform proximity matching
    for i, row in df1.iterrows():
        page_number = row['page']

        if page_number in page_trees:
            tree, df2_page = page_trees[page_number]

            # Query KDTree for nearest neighbor
            dist, idx = tree.query([row[['xmin', 'ymin', 'xmax', 'ymax']].values], distance_upper_bound=tolerance)

            if dist[0] < tolerance and idx[0] < len(df2_page):
                merged_df.at[i, 'text'] = df2_page.iloc[idx[0]]['text']

    # Drop the temporary key column
    merged_df.drop(columns=['key'], inplace=True)

    return merged_df

def do_proximity_match_all_pages_for_text(df1:pd.DataFrame, df2:pd.DataFrame, threshold:float=0.03):
    '''
    Match text from one dataframe to another based on proximity matching of coordinates across all pages.
    '''

    if not 'text' in df2.columns: df2['text'] = ''
    if not 'text' in df1.columns: df1['text'] = ''

    for col in ['xmin', 'ymin', 'xmax', 'ymax']:
        df1[col] = pd.to_numeric(df1[col], errors='coerce')

    for col in ['xmin', 'ymin', 'xmax', 'ymax']:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')

    # Create a unique key based on coordinates and label for exact merge
    merge_keys = ['xmin', 'ymin', 'xmax', 'ymax', 'label', 'page']
    df1['key'] = df1[merge_keys].astype(str).agg('_'.join, axis=1)
    df2['key'] = df2[merge_keys].astype(str).agg('_'.join, axis=1)

    # Attempt exact merge first, renaming df2['text'] to avoid suffixes
    merged_df = df1.merge(df2[['key', 'text']], on='key', how='left', suffixes=('', '_duplicate'))

    # If a match is found, keep that text; otherwise, keep the original df1 text
    merged_df['text'] = np.where(
        merged_df['text'].isna() | (merged_df['text'] == ''),
        merged_df.pop('text_duplicate'),
        merged_df['text']
    )

    # Handle missing matches using a proximity-based approach
    # Convert coordinates to numpy arrays for KDTree lookup
    

    query_coords = np.array(df1[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype=float)

    # Check for NaN or infinite values in query_coords and filter them out
    finite_mask = np.isfinite(query_coords).all(axis=1)
    if not finite_mask.all():
        #print("Warning: query_coords contains non-finite values. Filtering out non-finite entries.")
        query_coords = query_coords[finite_mask]  # Filter out rows with NaN or infinite values
    else:
        pass
    
    # Proceed only if query_coords is not empty
    if query_coords.size > 0:
        # Ensure df2 is filtered for finite values before creating the KDTree
        finite_mask_df2 = np.isfinite(df2[['xmin', 'ymin', 'xmax', 'ymax']].values).all(axis=1)
        df2_finite = df2[finite_mask_df2]

        # Create the KDTree with the filtered data
        tree = cKDTree(df2_finite[['xmin', 'ymin', 'xmax', 'ymax']].values)

        # Find nearest neighbors within a reasonable tolerance (e.g., 1% of page)
        tolerance = threshold
        distances, indices = tree.query(query_coords, distance_upper_bound=tolerance)

        # Assign text values where matches are found
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist < tolerance and idx < len(df2_finite):
                merged_df.at[i, 'text'] = df2_finite.iloc[idx]['text']

    # Drop the temporary key column
    merged_df.drop(columns=['key'], inplace=True)

    return merged_df

def _extract_page_number(image_path: Any) -> int:
    """Helper function to safely extract page number."""
    if not isinstance(image_path, str):
        return 1
    match = IMAGE_NUM_REGEX.search(image_path)
    if match:
        try:
            return int(match.group(1)) + 1
        except (ValueError, TypeError):
            return 1
    return 1

def convert_annotation_data_to_dataframe(all_annotations: List[Dict[str, Any]]):
    '''
    Convert annotation list to DataFrame using Pandas explode and json_normalize.
    '''
    if not all_annotations:
        # Return an empty DataFrame with the expected schema if input is empty
        return pd.DataFrame(columns=["image", "page", "xmin", "xmax", "ymin", "ymax", "text", "id"])

    # 1. Create initial DataFrame from the list of annotations
    # Use list comprehensions with .get() for robustness
    df = pd.DataFrame({
        "image": [anno.get("image") for anno in all_annotations],
        # Ensure 'boxes' defaults to an empty list if missing or None
        "boxes": [anno.get("boxes") if isinstance(anno.get("boxes"), list) else [] for anno in all_annotations]
    })

    # 2. Calculate the page number using the helper function
    df['page'] = df['image'].apply(_extract_page_number)

    # 3. Handle empty 'boxes' lists *before* exploding.
    # Explode removes rows where the list is empty. We want to keep them
    # as rows with NA values. Replace empty lists with a list containing
    # a single placeholder dictionary.
    placeholder_box = {"xmin": pd.NA, "xmax": pd.NA, "ymin": pd.NA, "ymax": pd.NA, "text": pd.NA, "id": pd.NA}
    df['boxes'] = df['boxes'].apply(lambda x: x if x else [placeholder_box])

    # 4. Explode the 'boxes' column. Each item in the list becomes a new row.
    df_exploded = df.explode('boxes', ignore_index=True)

    # 5. Normalize the 'boxes' column (which now contains dictionaries or the placeholder)
    # This turns the dictionaries into separate columns.
    # Check for NaNs or non-dict items just in case, though placeholder handles most cases.
    mask = df_exploded['boxes'].notna() & df_exploded['boxes'].apply(isinstance, args=(dict,))
    normalized_boxes = pd.json_normalize(df_exploded.loc[mask, 'boxes'])

    # 6. Combine the base data (image, page) with the normalized box data
    # Use the index of the exploded frame (where mask is True) to ensure correct alignment
    final_df = df_exploded.loc[mask, ['image', 'page']].reset_index(drop=True).join(normalized_boxes)

    # --- Optional: Handle rows that might have had non-dict items in 'boxes' ---
    # If there were rows filtered out by 'mask', you might want to add them back
    # with NA values for box columns. However, the placeholder strategy usually
    # prevents this from being necessary.

    # 7. Ensure essential columns exist and set column order
    essential_box_cols = ["xmin", "xmax", "ymin", "ymax", "text", "id", "label"]
    for col in essential_box_cols:
        if col not in final_df.columns:
            final_df[col] = pd.NA # Add column with NA if it wasn't present in any box
        final_df[col] = final_df[col].replace({None: pd.NA})

    base_cols = ["image"]
    extra_box_cols = [col for col in final_df.columns if col not in base_cols and col not in essential_box_cols]
    final_col_order = base_cols + essential_box_cols + sorted(extra_box_cols)

    # Reindex to ensure consistent column order and presence of essential columns
    # Using fill_value=pd.NA isn't strictly needed here as we added missing columns above,
    # but it's good practice if columns could be missing for other reasons.
    final_df = final_df.reindex(columns=final_col_order, fill_value=pd.NA)
    final_df = final_df.dropna(subset=["xmin", "xmax", "ymin", "ymax", "text", "id", "label"], how="all")
    final_df.replace({None: pd.NA})

    return final_df

def create_annotation_dicts_from_annotation_df(
    all_image_annotations_df: pd.DataFrame,
    page_sizes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    '''
    Convert annotation DataFrame back to list of dicts using dictionary lookup.
    Ensures all images from page_sizes are present without duplicates.
    '''
    # 1. Create a dictionary keyed by image path for efficient lookup & update
    # Initialize with all images from page_sizes. Use .get for safety.
    image_dict: Dict[str, Dict[str, Any]] = {}
    for item in page_sizes:
        image_path = item.get("image_path")
        if image_path:  # Only process if image_path exists and is not None/empty
            image_dict[image_path] = {"image": image_path, "boxes": []}

    # Check if the DataFrame is empty or lacks necessary columns
    if all_image_annotations_df.empty or 'image' not in all_image_annotations_df.columns:
        #print("Warning: Annotation DataFrame is empty or missing 'image' column.")
        return list(image_dict.values()) # Return based on page_sizes only

    # 2. Define columns to extract for boxes and check availability
    # Make sure these columns actually exist in the DataFrame
    box_cols = ['xmin', 'ymin', 'xmax', 'ymax', 'color', 'label', 'text', 'id']
    available_cols = [col for col in box_cols if col in all_image_annotations_df.columns]

    if 'text' in all_image_annotations_df.columns:
        all_image_annotations_df['text'] = all_image_annotations_df['text'].fillna('')
        #all_image_annotations_df.loc[all_image_annotations_df['text'].isnull(), 'text'] = ''

    if not available_cols:
        print(f"Warning: None of the expected box columns ({box_cols}) found in DataFrame.")
        return list(image_dict.values()) # Return based on page_sizes only

    # 3. Group the DataFrame by image and update the dictionary
    # Drop rows where essential coordinates might be NA (adjust if NA is meaningful)
    coord_cols = ['xmin', 'ymin', 'xmax', 'ymax']
    valid_box_df = all_image_annotations_df.dropna(
        subset=[col for col in coord_cols if col in available_cols]
    ).copy() # Use .copy() to avoid SettingWithCopyWarning if modifying later


    # Check if any valid boxes remain after dropping NAs
    if valid_box_df.empty:
         print("Warning: No valid annotation rows found in DataFrame after dropping NA coordinates.")
         return list(image_dict.values())

    # Process groups
    try:
        for image_path, group in valid_box_df.groupby('image', observed=True, sort=False):
            # Check if this image path exists in our target dictionary (from page_sizes)
            if image_path in image_dict:
                # Convert the relevant columns of the group to a list of dicts
                # Using only columns that are actually available
                boxes = group[available_cols].to_dict(orient='records')
                # Update the 'boxes' list in the dictionary
                image_dict[image_path]['boxes'] = boxes
            # Else: Image found in DataFrame but not required by page_sizes; ignore it.
    except KeyError:
        # This shouldn't happen due to the 'image' column check above, but handle defensively
        print("Error: Issue grouping DataFrame by 'image'.")
        return list(image_dict.values())


    # 4. Convert the dictionary values back into the final list format
    result = list(image_dict.values())

    return result

def convert_annotation_json_to_review_df(
    all_annotations: List[dict],
    redaction_decision_output: pd.DataFrame = pd.DataFrame(),
    page_sizes: List[dict] = [],
    do_proximity_match: bool = True
) -> pd.DataFrame:
    '''
    Convert the annotation json data to a dataframe format.
    Add on any text from the initial review_file dataframe by joining based on 'id' if available
    in both sources, otherwise falling back to joining on pages/co-ordinates (if option selected).

    Refactored for improved efficiency, prioritizing ID-based join and conditionally applying
    coordinate division and proximity matching.
    '''

    # 1. Convert annotations to DataFrame
    review_file_df = convert_annotation_data_to_dataframe(all_annotations)

    # Only keep rows in review_df where there are coordinates (assuming xmin is representative)
    # Use .notna() for robustness with potential None or NaN values
    review_file_df.dropna(subset=['xmin', 'ymin', 'xmax', 'ymax'], how='any', inplace=True)

    # Exit early if the initial conversion results in an empty DataFrame
    if review_file_df.empty:
        # Define standard columns for an empty return DataFrame
        # Ensure 'id' is included if it was potentially expected based on input structure
        # We don't know the columns from convert_annotation_data_to_dataframe without seeing it,
        # but let's assume a standard set and add 'id' if it appeared.
        standard_cols = ["image", "page", "label", "color", "xmin", "ymin", "xmax", "ymax", "text"]
        if 'id' in review_file_df.columns:
             standard_cols.append('id')
        return pd.DataFrame(columns=standard_cols)

    # Ensure 'id' column exists for logic flow, even if empty
    if 'id' not in review_file_df.columns:
        review_file_df['id'] = ''
    # Do the same for redaction_decision_output if it's not empty
    if not redaction_decision_output.empty and 'id' not in redaction_decision_output.columns:
         redaction_decision_output['id'] = ''


    # 2. Process page sizes if provided - needed potentially for coordinate division later
    # Process this once upfront if the data is available
    page_sizes_df = pd.DataFrame() # Initialize as empty
    if page_sizes:
        page_sizes_df = pd.DataFrame(page_sizes)
        if not page_sizes_df.empty:
            # Safely convert page column to numeric and then int
            page_sizes_df["page"] = pd.to_numeric(page_sizes_df["page"], errors="coerce")
            page_sizes_df.dropna(subset=["page"], inplace=True)
            if not page_sizes_df.empty: # Check again after dropping NaNs
                page_sizes_df["page"] = page_sizes_df["page"].astype(int)
            else:
                 print("Warning: Page sizes DataFrame became empty after processing, coordinate division will be skipped.")


    # 3. Join additional data from redaction_decision_output if provided
    text_added_successfully = False # Flag to track if text was added by any method

    if not redaction_decision_output.empty:
        # --- Attempt to join data based on 'id' column first ---

        # Check if 'id' columns are present and have non-null values in *both* dataframes
        id_col_exists_in_review = 'id' in review_file_df.columns and not review_file_df['id'].isnull().all() and not (review_file_df['id'] == '').all()
        id_col_exists_in_redaction = 'id' in redaction_decision_output.columns and not redaction_decision_output['id'].isnull().all() and not (redaction_decision_output['id'] == '').all()

        if id_col_exists_in_review and id_col_exists_in_redaction:
            #print("Attempting to join data based on 'id' column.")
            try:
                # Ensure 'id' columns are of string type for robust merging
                review_file_df['id'] = review_file_df['id'].astype(str)
                # Make a copy if needed, but try to avoid if redaction_decision_output isn't modified later
                # Let's use a copy for safety as in the original code
                redaction_copy = redaction_decision_output.copy()
                redaction_copy['id'] = redaction_copy['id'].astype(str)

                # Select columns to merge from redaction output. Prioritize 'text'.
                cols_to_merge = ['id']
                if 'text' in redaction_copy.columns:
                    cols_to_merge.append('text')
                else:
                    print("Warning: 'text' column not found in redaction_decision_output. Cannot merge text using 'id'.")

                # Perform a left merge to keep all annotations and add matching text
                # Use a suffix for the text column from the right DataFrame
                original_text_col_exists = 'text' in review_file_df.columns
                merge_suffix = '_redaction' if original_text_col_exists else ''

                merged_df = pd.merge(
                    review_file_df,
                    redaction_copy[cols_to_merge],
                    on='id',
                    how='left',
                    suffixes=('', merge_suffix)
                )

                # Update the 'text' column if a new one was brought in
                if 'text' + merge_suffix in merged_df.columns:
                    redaction_text_col = 'text' + merge_suffix
                    if original_text_col_exists:
                         # Combine: Use text from redaction where available, otherwise keep original
                         merged_df['text'] = merged_df[redaction_text_col].combine_first(merged_df['text'])
                         # Drop the temporary column
                         merged_df = merged_df.drop(columns=[redaction_text_col])
                    else:
                         # Redaction output had text, but review_file_df didn't. Rename the new column.
                         merged_df = merged_df.rename(columns={redaction_text_col: 'text'})

                    text_added_successfully = True # Indicate text was potentially added

                review_file_df = merged_df # Update the main DataFrame

                #print("Successfully attempted to join data using 'id'.") # Note: Text might not have been in redaction data

            except Exception as e:
                print(f"Error during 'id'-based merge: {e}. Checking for proximity match fallback.")
                # Fall through to proximity match logic below

        # --- Fallback to proximity match if ID join wasn't possible/successful and enabled ---
        # Note: If id_col_exists_in_review or id_col_exists_in_redaction was False,
        # the block above was skipped, and we naturally fall here.
        # If an error occurred in the try block, joined_by_id would implicitly be False
        # because text_added_successfully wasn't set to True.

        # Only attempt proximity match if text wasn't added by ID join and proximity is requested
        if not text_added_successfully and do_proximity_match:
             #print("Attempting proximity match to add text data.")

             # Ensure 'page' columns are numeric before coordinate division and proximity match
             # (Assuming divide_coordinates_by_page_sizes and do_proximity_match_all_pages_for_text need this)
             if 'page' in review_file_df.columns:
                 review_file_df['page'] = pd.to_numeric(review_file_df['page'], errors='coerce').fillna(-1).astype(int) # Use -1 for NaN pages
                 review_file_df = review_file_df[review_file_df['page'] != -1] # Drop rows where page conversion failed
             if not redaction_decision_output.empty and 'page' in redaction_decision_output.columns:
                  redaction_decision_output['page'] = pd.to_numeric(redaction_decision_output['page'], errors='coerce').fillna(-1).astype(int)
                  redaction_decision_output = redaction_decision_output[redaction_decision_output['page'] != -1]

             # Perform coordinate division IF page_sizes were processed and DataFrame is not empty
             if not page_sizes_df.empty:
                  # Apply coordinate division *before* proximity match
                  review_file_df = divide_coordinates_by_page_sizes(review_file_df, page_sizes_df)
                  if not redaction_decision_output.empty:
                       redaction_decision_output = divide_coordinates_by_page_sizes(redaction_decision_output, page_sizes_df)

             # Now perform the proximity match
             # Note: Potential DataFrame copies happen inside do_proximity_match based on its implementation
             if not redaction_decision_output.empty:
                try:
                    review_file_df = do_proximity_match_all_pages_for_text(
                        df1=review_file_df, # Pass directly, avoid caller copy if possible by modifying function signature
                        df2=redaction_decision_output # Pass directly
                    )
                    # Assuming do_proximity_match_all_pages_for_text adds/updates the 'text' column
                    if 'text' in review_file_df.columns:
                         text_added_successfully = True
                    #print("Proximity match completed.")
                except Exception as e:
                    print(f"Error during proximity match: {e}. Text data may not be added.")

        elif not text_added_successfully and not do_proximity_match:
             print("Skipping joining text data (ID join not possible/failed, proximity match disabled).")

    # 4. Ensure required columns exist and are ordered
    # Define base required columns. 'id' and 'text' are conditionally added.
    required_columns_base = ["image", "page", "label", "color", "xmin", "ymin", "xmax", "ymax"]
    final_columns = required_columns_base[:] # Start with base columns

    # Add 'id' and 'text' if they exist in the DataFrame at this point
    if 'id' in review_file_df.columns:
        final_columns.append('id')
    if 'text' in review_file_df.columns:
        final_columns.append('text') # Add text column if it was created/merged

    # Add any missing required columns with a default value (e.g., blank string)
    for col in final_columns:
        if col not in review_file_df.columns:
            # Use appropriate default based on expected type, '' for text/id, np.nan for coords?
            # Sticking to '' as in original for simplicity, but consider data types.
            review_file_df[col] = '' # Or np.nan for numerical, but coords already checked by dropna

    # Select and order the final set of columns
    # Ensure all selected columns actually exist after adding defaults
    review_file_df = review_file_df[[col for col in final_columns if col in review_file_df.columns]]


    # 5. Final processing and sorting
    # Convert colours from list to tuple if necessary - apply is okay here unless lists are vast
    if 'color' in review_file_df.columns:
         # Check if the column actually contains lists before applying lambda
         if review_file_df['color'].apply(lambda x: isinstance(x, list)).any():
            review_file_df["color"] = review_file_df["color"].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    # Sort the results
    # Ensure sort columns exist before sorting
    sort_columns = ['page', 'ymin', 'xmin', 'label']
    valid_sort_columns = [col for col in sort_columns if col in review_file_df.columns]
    if valid_sort_columns and not review_file_df.empty: # Only sort non-empty df
         # Convert potential numeric sort columns to appropriate types if necessary
         # (e.g., 'page', 'ymin', 'xmin') to ensure correct sorting.
         # dropna(subset=[...], inplace=True) earlier should handle NaNs in coords.
         # page conversion already done before proximity match.
         try:
             review_file_df = review_file_df.sort_values(valid_sort_columns)
         except TypeError as e:
              print(f"Warning: Could not sort DataFrame due to type error in sort columns: {e}")
              # Proceed without sorting

    base_cols = ["xmin", "xmax", "ymin", "ymax", "text", "id", "label"]

    for col in base_cols:
        if col not in review_file_df.columns:
            review_file_df[col] = pd.NA

    review_file_df = review_file_df.dropna(subset=base_cols, how="all")

    return review_file_df

def fill_missing_box_ids(data_input: dict) -> dict:
    """
    Generates unique alphanumeric IDs for bounding boxes in an input dictionary
    where the 'id' is missing, blank, or not a 12-character string.

    Args:
        data_input (dict): The input dictionary containing 'image' and 'boxes' keys.
                           'boxes' should be a list of dictionaries, each potentially
                           with an 'id' key.

    Returns:
        dict: The input dictionary with missing/invalid box IDs filled.
              Note: The function modifies the input dictionary in place.
    """

    # --- Input Validation ---
    if not isinstance(data_input, dict):
        raise TypeError("Input 'data_input' must be a dictionary.")
    #if 'boxes' not in data_input or not isinstance(data_input.get('boxes'), list):
    #    raise ValueError("Input dictionary must contain a 'boxes' key with a list value.")

    boxes = data_input#['boxes']
    id_length = 12
    character_set = string.ascii_letters + string.digits # a-z, A-Z, 0-9

    # --- Get Existing IDs to Ensure Uniqueness ---
    # Collect all valid existing IDs first
    existing_ids = set()
    #for box in boxes:
    # Check if 'id' exists, is a string, and is the correct length
    box_id = boxes.get('id')
    if isinstance(box_id, str) and len(box_id) == id_length:
        existing_ids.add(box_id)

    # --- Identify and Fill Rows Needing IDs ---
    generated_ids_set = set() # Keep track of IDs generated *in this run*
    num_filled = 0

    #for box in boxes:
    box_id = boxes.get('id')

    # Check if ID needs to be generated
    # Needs ID if: key is missing, value is None, value is not a string,
    # value is an empty string after stripping whitespace, or value is a string
    # but not of the correct length.
    needs_new_id = (
        box_id is None or
        not isinstance(box_id, str) or
        box_id.strip() == "" or
        len(box_id) != id_length
    )

    if needs_new_id:
        # Generate a unique ID
        attempts = 0
        while True:
            candidate_id = ''.join(random.choices(character_set, k=id_length))
            # Check against *all* existing valid IDs and *newly* generated ones in this run
            if candidate_id not in existing_ids and candidate_id not in generated_ids_set:
                generated_ids_set.add(candidate_id)
                boxes['id'] = candidate_id # Assign the new ID directly to the box dict
                num_filled += 1
                break # Found a unique ID
            attempts += 1
            # Safety break for unlikely infinite loop (though highly improbable with 12 chars)
            if attempts > len(boxes) * 100 + 1000:
                    raise RuntimeError(f"Failed to generate a unique ID after {attempts} attempts. Check ID length or existing IDs.")

    if num_filled > 0:
        pass
        #print(f"Successfully filled {num_filled} missing or invalid box IDs.")
    else:
        pass
        #print("No missing or invalid box IDs found.")


    # The input dictionary 'data_input' has been modified in place
    return data_input

def fill_missing_ids(df: pd.DataFrame, column_name: str = 'id', length: int = 12) -> pd.DataFrame:
    """
    Optimized: Generates unique alphanumeric IDs for rows in a DataFrame column
    where the value is missing (NaN, None) or an empty/whitespace string.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        column_name (str): The name of the column to check and fill (defaults to 'id').
                           This column will be added if it doesn't exist.
        length (int): The desired length of the generated IDs (defaults to 12).

    Returns:
        pd.DataFrame: The DataFrame with missing/empty IDs filled in the specified column.
                      Note: The function modifies the DataFrame directly (in-place).
    """

    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a Pandas DataFrame.")
    if not isinstance(column_name, str) or not column_name:
        raise ValueError("'column_name' must be a non-empty string.")
    if not isinstance(length, int) or length <= 0:
        raise ValueError("'length' must be a positive integer.")

    # --- Ensure Column Exists ---
    original_dtype = None
    if column_name not in df.columns:
        #print(f"Column '{column_name}' not found. Adding it to the DataFrame.")
        # Initialize with None (which Pandas often treats as NaN but allows object dtype)
        df[column_name] = None
        # Set original_dtype to object so it likely becomes string later
        original_dtype = object
    else:
        original_dtype = df[column_name].dtype

    # --- Identify Rows Needing IDs ---
    # 1. Check for actual null values (NaN, None, NaT)
    is_null = df[column_name].isna()

    # 2. Check for empty or whitespace-only strings AFTER converting potential values to string
    #    Only apply string checks on rows that are *not* null to avoid errors/warnings
    #    Fill NaN temporarily for string operations, then check length or equality
    is_empty_str = pd.Series(False, index=df.index) # Default to False
    if not is_null.all(): # Only check strings if there are non-null values
         temp_str_col = df.loc[~is_null, column_name].astype(str).str.strip()
         is_empty_str.loc[~is_null] = (temp_str_col == '')

    # Combine the conditions
    is_missing_or_empty = is_null | is_empty_str

    rows_to_fill_index = df.index[is_missing_or_empty]
    num_needed = len(rows_to_fill_index)

    if num_needed == 0:
        # Ensure final column type is consistent if nothing was done
        if pd.api.types.is_object_dtype(original_dtype) or pd.api.types.is_string_dtype(original_dtype):
             pass # Likely already object or string
        else:
             # If original was numeric/etc., but might contain strings now? Unlikely here.
             pass # Or convert to object: df[column_name] = df[column_name].astype(object)
        # print(f"No missing or empty values found requiring IDs in column '{column_name}'.")
        return df

    #print(f"Found {num_needed} rows requiring a unique ID in column '{column_name}'.")

    # --- Get Existing IDs to Ensure Uniqueness ---
    # Consider only rows that are *not* missing/empty
    valid_rows = df.loc[~is_missing_or_empty, column_name]
    # Drop any remaining nulls (shouldn't be any based on mask, but belts and braces)
    valid_rows = valid_rows.dropna()
    # Convert to string *only* if not already string/object, then filter out empty strings again
    if not pd.api.types.is_object_dtype(valid_rows.dtype) and not pd.api.types.is_string_dtype(valid_rows.dtype):
         existing_ids = set(valid_rows.astype(str).str.strip())
    else: # Already string or object, just strip and convert to set
         existing_ids = set(valid_rows.astype(str).str.strip()) # astype(str) handles mixed types in object column

    # Remove empty string from existing IDs if it's there after stripping
    existing_ids.discard('')


    # --- Generate Unique IDs ---
    character_set = string.ascii_letters + string.digits # a-z, A-Z, 0-9
    generated_ids_set = set() # Keep track of IDs generated *in this run*
    new_ids_list = []      # Store the generated IDs in order

    max_possible_ids = len(character_set) ** length
    if num_needed > max_possible_ids:
        raise ValueError(f"Cannot generate {num_needed} unique IDs with length {length}. Maximum possible is {max_possible_ids}.")

    # Pre-calculate safety break limit
    max_attempts_per_id = max(1000, num_needed * 10) # Adjust multiplier as needed

    #print(f"Generating {num_needed} unique IDs of length {length}...")
    for i in range(num_needed):
        attempts = 0
        while True:
            candidate_id = ''.join(random.choices(character_set, k=length))
            # Check against *all* known existing IDs and *newly* generated ones
            if candidate_id not in existing_ids and candidate_id not in generated_ids_set:
                generated_ids_set.add(candidate_id)
                new_ids_list.append(candidate_id)
                break # Found a unique ID
            attempts += 1
            if attempts > max_attempts_per_id : # Safety break
                raise RuntimeError(f"Failed to generate a unique ID after {attempts} attempts. Check length, character set, or density of existing IDs.")

        # Optional progress update
        # if (i + 1) % 1000 == 0:
        #    print(f"Generated {i+1}/{num_needed} IDs...")


    # --- Assign New IDs ---
    # Use the previously identified index to assign the new IDs correctly
    # Assigning string IDs might change the column's dtype to 'object'
    if not pd.api.types.is_object_dtype(original_dtype) and not pd.api.types.is_string_dtype(original_dtype):
         df['id'] = df['id'].astype(str, errors="ignore")
         #warnings.warn(f"Column '{column_name}' dtype might change from '{original_dtype}' to 'object' due to string ID assignment.", UserWarning)

    df.loc[rows_to_fill_index, column_name] = new_ids_list
    print(f"Successfully assigned {len(new_ids_list)} new unique IDs to column '{column_name}'.")

    # Optional: Convert the entire column to string type at the end for consistency
    # df[column_name] = df[column_name].astype(str)

    return df

def convert_review_df_to_annotation_json(
    review_file_df: pd.DataFrame,
    image_paths: List[str], # List of image file paths
    page_sizes: List[Dict], # List of dicts like [{'page': 1, 'image_path': '...', 'image_width': W, 'image_height': H}, ...]
    xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax" # Coordinate column names
) -> List[Dict]:
    """
    Optimized function to convert review DataFrame to Gradio Annotation JSON format.

    Ensures absolute coordinates, handles missing IDs, deduplicates based on key fields,
    selects final columns, and structures data per image/page based on page_sizes.

    Args:
        review_file_df: Input DataFrame with annotation data.
        image_paths: List of image file paths (Note: currently unused if page_sizes provides paths).
        page_sizes: REQUIRED list of dictionaries, each containing 'page',
                    'image_path', 'image_width', and 'image_height'. Defines
                    output structure and dimensions for coordinate conversion.
        xmin, xmax, ymin, ymax: Names of the coordinate columns.

    Returns:
        List of dictionaries suitable for Gradio Annotation output, one dict per image/page.
    """
    base_cols = ["xmin", "xmax", "ymin", "ymax", "text", "id", "label"]

    for col in base_cols:
        if col not in review_file_df.columns:
            review_file_df[col] = pd.NA

    review_file_df = review_file_df.dropna(subset=["xmin", "xmax", "ymin", "ymax", "text", "id", "label"], how='all')

    if not page_sizes:
        raise ValueError("page_sizes argument is required and cannot be empty.")

    # --- Prepare Page Sizes DataFrame ---
    try:
        page_sizes_df = pd.DataFrame(page_sizes)
        required_ps_cols = {'page', 'image_path', 'image_width', 'image_height'}
        if not required_ps_cols.issubset(page_sizes_df.columns):
            missing = required_ps_cols - set(page_sizes_df.columns)
            raise ValueError(f"page_sizes is missing required keys: {missing}")
        # Convert page sizes columns to appropriate numeric types early
        page_sizes_df['page'] = pd.to_numeric(page_sizes_df['page'], errors='coerce')
        page_sizes_df['image_width'] = pd.to_numeric(page_sizes_df['image_width'], errors='coerce')
        page_sizes_df['image_height'] = pd.to_numeric(page_sizes_df['image_height'], errors='coerce')
        # Use nullable Int64 for page number consistency
        page_sizes_df['page'] = page_sizes_df['page'].astype('Int64')

    except Exception as e:
        raise ValueError(f"Error processing page_sizes: {e}") from e




    # Handle empty input DataFrame gracefully
    if review_file_df.empty:
        print("Input review_file_df is empty. Proceeding to generate JSON structure with empty boxes.")
        # Ensure essential columns exist even if empty for later steps
        for col in [xmin, xmax, ymin, ymax, "page", "label", "color", "id", "text"]:
             if col not in review_file_df.columns:
                 review_file_df[col] = pd.NA
    else:
        # --- Coordinate Conversion (if needed) ---
        coord_cols_to_check = [c for c in [xmin, xmax, ymin, ymax] if c in review_file_df.columns]
        needs_multiplication = False
        if coord_cols_to_check:
            temp_df_numeric = review_file_df[coord_cols_to_check].apply(pd.to_numeric, errors='coerce')
            if temp_df_numeric.le(1).any().any(): # Check if any numeric coord <= 1 exists
                 needs_multiplication = True

        if needs_multiplication:
            #print("Relative coordinates detected or suspected, running multiplication...")
            review_file_df = multiply_coordinates_by_page_sizes(
                review_file_df.copy(), # Pass a copy to avoid modifying original outside function
                page_sizes_df,
                xmin, xmax, ymin, ymax
            )
        else:
            #print("No relative coordinates detected or required columns missing, skipping multiplication.")
            # Still ensure essential coordinate/page columns are numeric if they exist
            cols_to_convert = [c for c in [xmin, xmax, ymin, ymax, "page"] if c in review_file_df.columns]
            for col in cols_to_convert:
                review_file_df[col] = pd.to_numeric(review_file_df[col], errors='coerce')

        # Handle potential case where multiplication returns an empty DF
        if review_file_df.empty:
            print("DataFrame became empty after coordinate processing.")
            # Re-add essential columns if they were lost
            for col in [xmin, xmax, ymin, ymax, "page", "label", "color", "id", "text"]:
                if col not in review_file_df.columns:
                    review_file_df[col] = pd.NA

        # --- Fill Missing IDs ---
        review_file_df = fill_missing_ids(review_file_df.copy()) # Pass a copy

        # --- Deduplicate Based on Key Fields ---
        base_dedupe_cols = ["page", xmin, ymin, xmax, ymax, "label", "id"]
        # Identify which deduplication columns actually exist in the DataFrame
        cols_for_dedupe = [col for col in base_dedupe_cols if col in review_file_df.columns]
        # Add 'image' column for deduplication IF it exists (matches original logic intent)
        if "image" in review_file_df.columns:
            cols_for_dedupe.append("image")

        # Ensure placeholder columns exist if they are needed for deduplication
        # (e.g., 'label', 'id' should be present after fill_missing_ids)
        for col in ['label', 'id']:
            if col in cols_for_dedupe and col not in review_file_df.columns:
                 # This might indicate an issue in fill_missing_ids or prior steps
                 print(f"Warning: Column '{col}' needed for dedupe but not found. Adding NA.")
                 review_file_df[col] = "" # Add default empty string

        if cols_for_dedupe: # Only attempt dedupe if we have columns to check
            #print(f"Deduplicating based on columns: {cols_for_dedupe}")
            # Convert relevant columns to string before dedupe to avoid type issues with mixed data (optional, depends on data)
            # for col in cols_for_dedupe:
            #    review_file_df[col] = review_file_df[col].astype(str)
            review_file_df = review_file_df.drop_duplicates(subset=cols_for_dedupe)
        else:
            print("Skipping deduplication: No valid columns found to deduplicate by.")


    # --- Select and Prepare Final Output Columns ---
    required_final_cols = ["page", "label", "color", xmin, ymin, xmax, ymax, "id", "text"]
    # Identify which of the desired final columns exist in the (now potentially deduplicated) DataFrame
    available_final_cols = [col for col in required_final_cols if col in review_file_df.columns]

    # Ensure essential output columns exist, adding defaults if missing AFTER deduplication
    for col in required_final_cols:
         if col not in review_file_df.columns:
             print(f"Adding missing final column '{col}' with default value.")
             if col in ['label', 'id', 'text']:
                 review_file_df[col] = "" # Default empty string
             elif col == 'color':
                 review_file_df[col] = None # Default None or a default color tuple
             else: # page, coordinates
                 review_file_df[col] = pd.NA # Default NA for numeric/page
             available_final_cols.append(col) # Add to list of available columns

    # Select only the final desired columns in the correct order
    review_file_df = review_file_df[available_final_cols]

    # --- Final Formatting ---
    if not review_file_df.empty:
        # Convert list colors to tuples (important for some downstream uses)
        if 'color' in review_file_df.columns:
            review_file_df['color'] = review_file_df['color'].apply(
                lambda x: tuple(x) if isinstance(x, list) else x
            )
        # Ensure page column is nullable integer type for reliable grouping
        if 'page' in review_file_df.columns:
             review_file_df['page'] = review_file_df['page'].astype('Int64')

    # --- Group Annotations by Page ---
    if 'page' in review_file_df.columns:
        grouped_annotations = review_file_df.groupby('page')
        group_keys = set(grouped_annotations.groups.keys()) # Use set for faster lookups
    else:
        # Cannot group if page column is missing
        print("Error: 'page' column missing, cannot group annotations.")
        grouped_annotations = None
        group_keys = set()


    # --- Build JSON Structure ---
    json_data = []
    output_cols_for_boxes = [col for col in ["label", "color", xmin, ymin, xmax, ymax, "id", "text"] if col in review_file_df.columns]

    # Iterate through page_sizes_df to define the structure (one entry per image path)
    for _, row in page_sizes_df.iterrows():
        page_num = row['page'] # Already Int64
        pdf_image_path = row['image_path']
        annotation_boxes = [] # Default to empty list

        # Check if the page exists in the grouped annotations (using the faster set lookup)
        # Check pd.notna because page_num could be <NA> if conversion failed
        if pd.notna(page_num) and page_num in group_keys and grouped_annotations:
            try:
                page_group_df = grouped_annotations.get_group(page_num)
                # Convert the group to list of dicts, selecting only needed box properties
                # Handle potential NaN coordinates before conversion to JSON
                annotation_boxes = page_group_df[output_cols_for_boxes].replace({np.nan: None}).to_dict(orient='records')

                # Optional: Round coordinates here if needed AFTER potential multiplication
                # for box in annotation_boxes:
                #     for coord in [xmin, ymin, xmax, ymax]:
                #         if coord in box and box[coord] is not None:
                #             box[coord] = round(float(box[coord]), 2) # Example: round to 2 decimals

            except KeyError:
                 print(f"Warning: Group key {page_num} not found despite being in group_keys (should not happen).")
                 annotation_boxes = [] # Keep empty

        # Append the structured data for this image/page
        json_data.append({
            "image": pdf_image_path,
            "boxes": annotation_boxes
        })

    return json_data
