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
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdf2image import convert_from_path
from PIL import Image
from scipy.spatial import cKDTree

IMAGE_NUM_REGEX = re.compile(r'_(\d+)\.png$')

pd.set_option('future.no_silent_downcasting', True)

from tools.config import OUTPUT_FOLDER, INPUT_FOLDER, IMAGES_DPI, LOAD_TRUNCATED_IMAGES, MAX_IMAGE_PIXELS, CUSTOM_BOX_COLOUR
from tools.helper_functions import get_file_name_without_type, tesseract_ocr_option, text_ocr_option, textract_option, read_file
# from tools.aws_textract import load_and_convert_textract_json

image_dpi = float(IMAGES_DPI)
if not MAX_IMAGE_PIXELS: Image.MAX_IMAGE_PIXELS = None
else: Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = LOAD_TRUNCATED_IMAGES.lower() == "true"

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

    print("PDF has been converted to images.")
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
        print(f"{file_path} is a PDF file. Converting to image set")

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

        # Create a page_sizes_object.
        # If images have been created, then image width an height come from this value. Otherwise, they are set to the cropbox size
        if image_sizes_width and image_sizes_height:
            out_page_image_sizes = {"page":reported_page_no, "image_path":image_file_paths[page_no], "image_width":image_sizes_width[page_no], "image_height":image_sizes_height[page_no], "mediabox_width":pymupdf_page.mediabox.width, "mediabox_height": pymupdf_page.mediabox.height, "cropbox_width":pymupdf_page.cropbox.width, "cropbox_height":pymupdf_page.cropbox.height, "original_cropbox":original_cropboxes[-1]}
        else:
            out_page_image_sizes = {"page":reported_page_no, "image_path":image_file_paths[page_no], "image_width":pd.NA, "image_height":pd.NA, "mediabox_width":pymupdf_page.mediabox.width, "mediabox_height": pymupdf_page.mediabox.height, "cropbox_width":pymupdf_page.cropbox.width, "cropbox_height":pymupdf_page.cropbox.height, "original_cropbox":original_cropboxes[-1]}
        
        page_sizes.append(out_page_image_sizes)

    return page_sizes, original_cropboxes

def prepare_image_or_pdf(
    file_paths: List[str],
    in_redact_method: str,
    latest_file_completed: int = 0,
    out_message: List[str] = [],
    first_loop_state: bool = False,
    number_of_pages:int = 1,
    all_annotations_object:List = [],
    prepare_for_review:bool = False,
    in_fully_redacted_list:List[int]=[],
    output_folder:str=OUTPUT_FOLDER,
    input_folder:str=INPUT_FOLDER,
    prepare_images:bool=True,
    page_sizes:list[dict]=[],
    textract_output_found:bool = False,    
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
        textract_output_found (optional, bool): A boolean indicating whether textract output has already been found . Defaults to False.
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
  
    # If out message or converted_file_paths are blank, change to a list so it can be appended to
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
        return final_out_message, converted_file_paths, image_file_paths, number_of_pages, number_of_pages, pymupdf_doc, all_annotations_object, review_file_csv, original_cropboxes, page_sizes, textract_output_found, all_img_details, all_line_level_ocr_results_df

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
            if file_extension in ['.jpg', '.jpeg', '.png'] and in_redact_method == text_ocr_option:
                in_redact_method = tesseract_ocr_option

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
                #print("file_path:", file_path)
                review_file_csv = read_file(file_path)
                all_annotations_object = convert_review_df_to_annotation_json(review_file_csv, image_file_paths, page_sizes)
                json_from_csv = True
                print("Converted CSV review file to image annotation object")
            elif '_ocr_output' in file_path_without_ext:
                all_line_level_ocr_results_df = read_file(file_path)
                json_from_csv = False

        # NEW IF STATEMENT
        # If the file name ends with redactions.json, assume it is an annoations object, overwrite the current variable
        if (file_extension in ['.json']) | (json_from_csv == True):

            if (file_extension in ['.json']) &  (prepare_for_review == True):
                if isinstance(file_path, str):
                    with open(file_path, 'r') as json_file:
                        all_annotations_object = json.load(json_file)
                else:
                    # Assuming file_path is a NamedString or similar
                    all_annotations_object = json.loads(file_path)  # Use loads for string content

            # Assume it's a textract json
            elif (file_extension == '.json') and (prepare_for_review is not True):
                # Copy it to the output folder so it can be used later.
                out_textract_path = os.path.join(output_folder, file_path_without_ext + "_textract.json")

                # Use shutil to copy the file directly
                shutil.copy2(file_path, out_textract_path)  # Preserves metadata
                textract_output_found = True                
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
            if in_redact_method == tesseract_ocr_option or in_redact_method == textract_option:
                if is_pdf_or_image(file_path) == False:
                    out_message = "Please upload a PDF file or image file (JPG, PNG) for image analysis."
                    print(out_message)
                    raise Exception(out_message)

            elif in_redact_method == text_ocr_option:
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
        out_message_out = '\n'.join(out_message)

    number_of_pages = len(image_file_paths)
        
    return out_message_out, converted_file_paths, image_file_paths, number_of_pages, number_of_pages, pymupdf_doc, all_annotations_object, review_file_csv, original_cropboxes, page_sizes, textract_output_found, all_img_details, all_line_level_ocr_results_df

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

         # Remove 'text' elements from boxes
        for item in non_empty_boxes:
            if 'boxes' in item:
                item['boxes'] = [{k: v for k, v in box.items() if k != 'text'} for box in item['boxes']]

        if non_empty_boxes:
            # Keep the first entry with non-empty boxes
            result.append(non_empty_boxes[0])
        else:
            # If all items have empty or missing boxes, keep the first item
            result.append(items[0])

    return result

def divide_coordinates_by_page_sizes(review_file_df:pd.DataFrame, page_sizes_df:pd.DataFrame, xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"):

    '''Convert data to same coordinate system. If all coordinates all greater than one, this is a absolute image coordinates - change back to relative coordinates.'''

    review_file_df_out = review_file_df

    if xmin in review_file_df.columns and not review_file_df.empty:
        review_file_df_orig = review_file_df.copy().loc[(review_file_df[xmin] <= 1) & (review_file_df[xmax] <= 1) & (review_file_df[ymin] <= 1) & (review_file_df[ymax] <= 1),:]
        
        review_file_df = review_file_df.loc[(review_file_df[xmin] > 1) & (review_file_df[xmax] > 1) & (review_file_df[ymin] > 1) & (review_file_df[ymax] > 1),:]

        review_file_df.loc[:, "page"] = pd.to_numeric(review_file_df["page"], errors="coerce")

        review_file_df_div = review_file_df

        if "image_width" not in review_file_df_div.columns and not page_sizes_df.empty:  

            page_sizes_df["image_width"] = page_sizes_df["image_width"].replace("<NA>", pd.NA)
            page_sizes_df["image_height"] = page_sizes_df["image_height"].replace("<NA>", pd.NA)
            review_file_df_div = review_file_df_div.merge(page_sizes_df[["page", "image_width", "image_height", "mediabox_width", "mediabox_height"]], on="page", how="left")

        if "image_width" in review_file_df_div.columns:
            if review_file_df_div["image_width"].isna().all():  # Check if all are NaN values. If so, assume we only have mediabox coordinates available
                review_file_df_div["image_width"] = review_file_df_div["image_width"].fillna(review_file_df_div["mediabox_width"]).infer_objects()
                review_file_df_div["image_height"] = review_file_df_div["image_height"].fillna(review_file_df_div["mediabox_height"]).infer_objects()

            convert_type_cols = ["image_width", "image_height", xmin, xmax, ymin, ymax]
            review_file_df_div[convert_type_cols] = review_file_df_div[convert_type_cols].apply(pd.to_numeric, errors="coerce")  

            review_file_df_div[xmin] = review_file_df_div[xmin] / review_file_df_div["image_width"]
            review_file_df_div[xmax] = review_file_df_div[xmax] / review_file_df_div["image_width"]
            review_file_df_div[ymin] = review_file_df_div[ymin] / review_file_df_div["image_height"]
            review_file_df_div[ymax] = review_file_df_div[ymax] / review_file_df_div["image_height"]

        # Concatenate the original and modified DataFrames
        dfs_to_concat = [df for df in [review_file_df_orig, review_file_df_div] if not df.empty]
        if dfs_to_concat:  # Ensure there's at least one non-empty DataFrame
            review_file_df_out = pd.concat(dfs_to_concat)
        else:
            review_file_df_out = review_file_df  # Return an original DataFrame instead of raising an error

        # Only sort if the DataFrame is not empty and contains the required columns
        required_sort_columns = {"page", xmin, ymin}
        if not review_file_df_out.empty and required_sort_columns.issubset(review_file_df_out.columns):
            review_file_df_out.sort_values(["page", ymin, xmin], inplace=True)

    review_file_df_out.drop(["image_width", "image_height", "mediabox_width", "mediabox_height"], axis=1, errors="ignore")

    return review_file_df_out

def multiply_coordinates_by_page_sizes(review_file_df: pd.DataFrame, page_sizes_df: pd.DataFrame, xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"):


    if xmin in review_file_df.columns and not review_file_df.empty:
        # Separate absolute vs relative coordinates
        review_file_df_orig = review_file_df.loc[
            (review_file_df[xmin] > 1) & (review_file_df[xmax] > 1) & 
            (review_file_df[ymin] > 1) & (review_file_df[ymax] > 1), :].copy()

        review_file_df = review_file_df.loc[
            (review_file_df[xmin] <= 1) & (review_file_df[xmax] <= 1) & 
            (review_file_df[ymin] <= 1) & (review_file_df[ymax] <= 1), :].copy()

        if review_file_df.empty:
            return review_file_df_orig  # If nothing is left, return the original absolute-coordinates DataFrame

        review_file_df.loc[:, "page"] = pd.to_numeric(review_file_df["page"], errors="coerce")

        if "image_width" not in review_file_df.columns and not page_sizes_df.empty:
            page_sizes_df[['image_width', 'image_height']] = page_sizes_df[['image_width','image_height']].replace("<NA>", pd.NA)  # Ensure proper NA handling
            review_file_df = review_file_df.merge(page_sizes_df, on="page", how="left")

        if "image_width" in review_file_df.columns:
            # Split into rows with/without image size info
            review_file_df_not_na = review_file_df.loc[review_file_df["image_width"].notna()].copy()
            review_file_df_na = review_file_df.loc[review_file_df["image_width"].isna()].copy()

            if not review_file_df_not_na.empty:
                convert_type_cols = ["image_width", "image_height", xmin, xmax, ymin, ymax]
                review_file_df_not_na[convert_type_cols] = review_file_df_not_na[convert_type_cols].apply(pd.to_numeric, errors="coerce")

                # Multiply coordinates by image sizes
                review_file_df_not_na[xmin] *= review_file_df_not_na["image_width"]
                review_file_df_not_na[xmax] *= review_file_df_not_na["image_width"]
                review_file_df_not_na[ymin] *= review_file_df_not_na["image_height"]
                review_file_df_not_na[ymax] *= review_file_df_not_na["image_height"]

            # Concatenate the modified and unmodified data
            review_file_df = pd.concat([df for df in [review_file_df_not_na, review_file_df_na] if not df.empty])

        # Merge with the original absolute-coordinates DataFrame
        dfs_to_concat = [df for df in [review_file_df_orig, review_file_df] if not df.empty]
        if dfs_to_concat:  # Ensure there's at least one non-empty DataFrame
            review_file_df = pd.concat(dfs_to_concat)
        else:
            review_file_df = pd.DataFrame()  # Return an empty DataFrame instead of raising an error

        # Only sort if the DataFrame is not empty and contains the required columns
        required_sort_columns = {"page", "xmin", "ymin"}
        if not review_file_df.empty and required_sort_columns.issubset(review_file_df.columns):
            review_file_df.sort_values(["page", "xmin", "ymin"], inplace=True)

    return review_file_df


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
        print("Warning: query_coords contains non-finite values. Filtering out non-finite entries.")
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
        return pd.DataFrame(columns=["image", "page", "xmin", "xmax", "ymin", "ymax", "text"])

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
    placeholder_box = {"xmin": pd.NA, "xmax": pd.NA, "ymin": pd.NA, "ymax": pd.NA, "text": pd.NA}
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
    essential_box_cols = ["xmin", "xmax", "ymin", "ymax", "text"]
    for col in essential_box_cols:
        if col not in final_df.columns:
            final_df[col] = pd.NA # Add column with NA if it wasn't present in any box

    base_cols = ["image", "page"]
    extra_box_cols = [col for col in final_df.columns if col not in base_cols and col not in essential_box_cols]
    final_col_order = base_cols + essential_box_cols + sorted(extra_box_cols)

    # Reindex to ensure consistent column order and presence of essential columns
    # Using fill_value=pd.NA isn't strictly needed here as we added missing columns above,
    # but it's good practice if columns could be missing for other reasons.
    final_df = final_df.reindex(columns=final_col_order, fill_value=pd.NA)

    return final_df


# def convert_annotation_data_to_dataframe(all_annotations:List[dict]):
#         '''
#         Convert an annotation list of dictionaries to a dataframe with all boxes on a separate row
#         '''
#         # Flatten the data
#         flattened_annotation_data = []        

#         for annotation in all_annotations:
#             image_path = annotation["image"]

#             if image_path:
#                 match = re.search(r'_(\d+)\.png$', image_path)
#                 if match:
#                     number = match.group(1)
#                     reported_number = int(number) + 1
#                 else:
#                     reported_number = 1
#             else:
#                 reported_number = 1

#             # Check if 'boxes' is in the annotation, if not, add an empty list
#             if 'boxes' not in annotation:
#                 annotation['boxes'] = []

#             # If boxes are empty, create a row with blank values for xmin, xmax, ymin, ymax
#             if not annotation["boxes"]:
#                 data_to_add = {"image": image_path, "page": reported_number, "xmin": pd.NA, "xmax": pd.NA, "ymin": pd.NA, "ymax": pd.NA}
#                 flattened_annotation_data.append(data_to_add)
#             else:
#                 for box in annotation["boxes"]:
#                     if 'xmin' not in box:
#                         data_to_add = {"image": image_path, "page": reported_number, "xmin": pd.NA, 'xmax': pd.NA, 'ymin': pd.NA, 'ymax': pd.NA}
#                     elif 'text' not in box:
#                         data_to_add = {"image": image_path, "page": reported_number, **box}
#                     else:
#                         data_to_add = {"image": image_path, "page": reported_number, "text": box['text'], **box}
#                     flattened_annotation_data.append(data_to_add)

#         # Convert to a DataFrame
#         review_file_df = pd.DataFrame(flattened_annotation_data)

#         return review_file_df

# def create_annotation_dicts_from_annotation_df(all_image_annotations_df:pd.DataFrame, page_sizes:List[dict]):
#         '''
#         From an annotation object as a dataframe, convert back to a list of dictionaries that can be used in the Gradio Image Annotator component
#         '''
#         result = []

#         # Ensure that every page has an entry in the resulting list of dicts
#         for image_path in page_sizes:
#             annotation = {}
#             annotation["image"] = image_path["image_path"]
#             annotation["boxes"] = []

#             result.append(annotation)

#         # Then add in all the filled in data
#         for image, group in all_image_annotations_df.groupby('image'):
#             boxes = group[['xmin', 'ymin', 'xmax', 'ymax', 'color', 'label']].to_dict(orient='records')
#             result.append({'image': image, 'boxes': boxes})
        
#         return result

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
    box_cols = ['xmin', 'ymin', 'xmax', 'ymax', 'color', 'label']
    available_cols = [col for col in box_cols if col in all_image_annotations_df.columns]

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

# import pandas as pd
# from typing import List, Dict, Any

# def create_annotation_dicts_from_annotation_df(
#     all_image_annotations_df: pd.DataFrame,
#     page_sizes: List[Dict[str, Any]]
# ) -> List[Dict[str, Any]]:
#     '''
#     Convert annotation DataFrame back to list of dicts using Pandas merge.
#     Ensures all images from page_sizes are present without duplicates.
#     '''
#     # 1. Create a DataFrame containing all required image paths from page_sizes
#     if not page_sizes:
#         return []
#     all_image_paths = [item.get("image_path") for item in page_sizes if item.get("image_path")]
#     if not all_image_paths:
#         return []
#     # Use unique paths
#     pages_df = pd.DataFrame({'image': list(set(all_image_paths))})

#     # Check if the DataFrame is empty or lacks necessary columns
#     if all_image_annotations_df.empty or 'image' not in all_image_annotations_df.columns:
#         print("Warning: Annotation DataFrame is empty or missing 'image' column.")
#         # Add empty boxes column and return
#         pages_df['boxes'] = [[] for _ in range(len(pages_df))]
#         return pages_df.to_dict(orient='records')

#     # 2. Define columns to extract and check availability
#     box_cols = ['xmin', 'ymin', 'xmax', 'ymax', 'color', 'label']
#     available_cols = [col for col in box_cols if col in all_image_annotations_df.columns]

#     if not available_cols:
#         print(f"Warning: None of the expected box columns ({box_cols}) found in DataFrame.")
#         pages_df['boxes'] = [[] for _ in range(len(pages_df))]
#         return pages_df.to_dict(orient='records')

#     # 3. Prepare the annotation data: drop invalid rows and aggregate boxes
#     coord_cols = ['xmin', 'ymin', 'xmax', 'ymax']
#     valid_box_df = all_image_annotations_df.dropna(
#         subset=[col for col in coord_cols if col in available_cols]
#     ).copy() # Use .copy()

#     if valid_box_df.empty:
#          print("Warning: No valid annotation rows found after dropping NA coordinates.")
#          pages_df['boxes'] = [[] for _ in range(len(pages_df))]
#          return pages_df.to_dict(orient='records')


#     # Aggregate boxes into lists of dictionaries per image
#     def aggregate_boxes(group):
#         return group[available_cols].to_dict(orient='records')

#     # Group by image and apply the aggregation
#     grouped_boxes = valid_box_df.groupby('image', observed=True, sort=False).apply(aggregate_boxes).reset_index(name='boxes')

#     # 4. Perform a left merge: keep all images from pages_df, add boxes where they exist
#     merged_df = pd.merge(pages_df, grouped_boxes, on='image', how='left')

#     # 5. Fill NaN in 'boxes' column (for images with no annotations) with empty lists
#     # Ensure the column exists before trying to fillna
#     if 'boxes' in merged_df.columns:
#          # Use apply with a lambda for robust filling of NAs or potential None values
#          merged_df['boxes'] = merged_df['boxes'].apply(lambda x: [] if pd.isna(x) else x)
#     else:
#          # Should not happen with left merge, but handle defensively
#          merged_df['boxes'] = [[] for _ in range(len(merged_df))]


#     # 6. Convert the final DataFrame to the list of dictionaries format
#     result = merged_df.to_dict(orient='records')

#     return result

def convert_annotation_json_to_review_df(all_annotations:List[dict],
                                         redaction_decision_output:pd.DataFrame=pd.DataFrame(),
                                         page_sizes:pd.DataFrame=pd.DataFrame(),
                                         do_proximity_match:bool=True) -> pd.DataFrame:
    '''
    Convert the annotation json data to a dataframe format. Add on any text from the initial review_file dataframe by joining on pages/co-ordinates (if option selected).
    '''
    
    review_file_df = convert_annotation_data_to_dataframe(all_annotations)
    
    if page_sizes:
        page_sizes_df = pd.DataFrame(page_sizes)
        page_sizes_df[["page"]] = page_sizes_df[["page"]].apply(pd.to_numeric, errors="coerce")        
    
        review_file_df = divide_coordinates_by_page_sizes(review_file_df, page_sizes_df)

        redaction_decision_output = divide_coordinates_by_page_sizes(redaction_decision_output, page_sizes_df)

    # Join on additional text data from decision output results if included, if text not already there
    if not redaction_decision_output.empty and not review_file_df.empty and do_proximity_match == True:       

        # Match text to review file to match on text
        review_file_df = do_proximity_match_all_pages_for_text(df1 = review_file_df.copy(), df2 = redaction_decision_output.copy())

    # Ensure required columns exist, filling with blank if they don't
    check_columns = ["image", "page", "label", "color", "xmin", "ymin", "xmax", "ymax", "text"]

    for col in check_columns:
        if col not in review_file_df.columns:
            review_file_df[col] = ''

    if not review_file_df.empty:
        review_file_df = review_file_df[check_columns]
    else:
        review_file_df = pd.DataFrame(columns=check_columns)

    # If colours are saved as list, convert to tuple
    review_file_df.loc[:,"color"] = review_file_df.loc[:,"color"].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    review_file_df = review_file_df.sort_values(['page', 'ymin', 'xmin', 'label'])

    return review_file_df

def convert_review_df_to_annotation_json(review_file_df:pd.DataFrame,
                                         image_paths:List[Image.Image],
                                         page_sizes:List[dict]=[]) -> List[dict]:
    '''
    Convert a review csv to a json file for use by the Gradio Annotation object.
    '''
    # Make sure all relevant cols are float
    float_cols = ["page", "xmin", "xmax", "ymin", "ymax"]
    for col in float_cols:
        review_file_df.loc[:, col] = pd.to_numeric(review_file_df.loc[:, col], errors='coerce')
    
    # Convert relative co-ordinates into image coordinates for the image annotation output object
    if page_sizes:        
        page_sizes_df = pd.DataFrame(page_sizes)
        page_sizes_df[["page"]] = page_sizes_df[["page"]].apply(pd.to_numeric, errors="coerce")

        review_file_df = multiply_coordinates_by_page_sizes(review_file_df, page_sizes_df)
           
    # Keep only necessary columns
    review_file_df = review_file_df[["image", "page", "label", "color", "xmin", "ymin", "xmax", "ymax"]].drop_duplicates(subset=["image", "page", "xmin", "ymin", "xmax", "ymax", "label"])

    # If colours are saved as list, convert to tuple
    review_file_df.loc[:, "color"] = review_file_df.loc[:,"color"].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    # Group the DataFrame by the 'image' column
    grouped_csv_pages = review_file_df.groupby('page')

    # Create a list to hold the JSON data
    json_data = []

    for page_no, pdf_image_path in enumerate(page_sizes_df["image_path"]):
        
        reported_page_number = int(page_no + 1)

        if reported_page_number in review_file_df["page"].values:

            # Convert each relevant group to a list of box dictionaries
            selected_csv_pages = grouped_csv_pages.get_group(reported_page_number)
            annotation_boxes = selected_csv_pages.drop(columns=['image', 'page']).to_dict(orient='records')
            
            annotation = {
                "image": pdf_image_path,
                "boxes": annotation_boxes
            }

        else:
            annotation = {}
            annotation["image"] = pdf_image_path
            annotation["boxes"] = []

        # Append the structured data to the json_data list
        json_data.append(annotation)

    return json_data