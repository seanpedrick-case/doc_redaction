from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
import os
from gradio import Progress

def is_pdf_or_image(filename):
    """
    Check if a file name is a PDF or an image file.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if the file name ends with ".pdf", ".jpg", or ".png", False otherwise.
    """
    if filename.lower().endswith(".pdf") or filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
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

def convert_pdf_to_images(pdf_path, progress=Progress(track_tqdm=True)):

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

        # # Convert PDF to a list of images
        # images = convert_from_path(pdf_path)

        # images = []

        images.extend(image)

    # Save each image as a separate file - deprecated
    #image_paths = []
    # for i, image in enumerate(images):
    #     page_path = f"processing/page_{i+1}.png"
    #     image.save(page_path, "PNG")
    #     image_paths.append(page_path)

    print("PDF has been converted to images.")

    return images

# %%
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

