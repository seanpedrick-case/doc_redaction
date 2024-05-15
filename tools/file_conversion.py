from pdf2image import convert_from_path, pdfinfo_from_path
from tools.helper_functions import get_file_path_end
from PIL import Image
import os
from gradio import Progress
from typing import List

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

def convert_pdf_to_images(pdf_path:str, progress=Progress(track_tqdm=True)):

    # Get the number of pages in the PDF
    page_count = pdfinfo_from_path(pdf_path)['Pages']
    print("Number of pages in PDF: ", str(page_count))

    images = []

    # Open the PDF file
    for page_num in progress.tqdm(range(0,page_count), total=page_count, unit="pages", desc="Converting pages"):
        
        print("Current page: ", str(page_num))

        # Convert one page to image
        image = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
        
        # If no images are returned, break the loop
        if not image:
            break

        images.extend(image)

    print("PDF has been converted to images.")

    return images


# %% Function to take in a file path, decide if it is an image or pdf, then process appropriately.
def process_file(file_path):
    # Get the file extension
    file_extension = os.path.splitext(file_path)[1].lower()

    # Check if the file is an image type
    if file_extension in ['.jpg', '.jpeg', '.png']:
        print(f"{file_path} is an image file.")
        # Perform image processing here
        out_path = [Image.open(file_path)]

    # Check if the file is a PDF
    elif file_extension == '.pdf':
        print(f"{file_path} is a PDF file. Converting to image set")
        # Run your function for processing PDF files here
        out_path = convert_pdf_to_images(file_path)

    else:
        print(f"{file_path} is not an image or PDF file.")
        out_path = ['']

    return out_path

def prepare_image_or_text_pdf(file_path:str, in_redact_method:str, in_allow_list:List[List[str]]=None):

    out_message = ''
    out_file_paths = []

    in_allow_list_flat = [item for sublist in in_allow_list for item in sublist]

    if file_path:
        file_path_without_ext = get_file_path_end(file_path)
    else:
        out_message = "No file selected"
        print(out_message)
        return out_message, out_file_paths

    if in_redact_method == "Image analysis":
        # Analyse and redact image-based pdf or image
        if is_pdf_or_image(file_path) == False:
            return "Please upload a PDF file or image file (JPG, PNG) for image analysis.", None
        
        out_file_path = process_file(file_path)

    elif in_redact_method == "Text analysis":
        if is_pdf(file_path) == False:
            return "Please upload a PDF file for text analysis.", None
        
        out_file_path = file_path
    
    return out_message, out_file_path


def convert_text_pdf_to_img_pdf(in_file_path:str, out_text_file_path:List[str]):
    file_path_without_ext = get_file_path_end(in_file_path)

    out_file_paths = out_text_file_path

    # Convert annotated text pdf back to image to give genuine redactions
    print("Creating image version of results")
    pdf_text_image_paths = process_file(out_text_file_path[0])
    out_text_image_file_path = "output/" + file_path_without_ext + "_result_as_text_back_to_img.pdf"
    pdf_text_image_paths[0].save(out_text_image_file_path, "PDF" ,resolution=100.0, save_all=True, append_images=pdf_text_image_paths[1:])

    out_file_paths.append(out_text_image_file_path)

    out_message = "Image-based PDF successfully redacted and saved to text-based annotated file, and image-based file."

    return out_message, out_file_paths

        

        
    

