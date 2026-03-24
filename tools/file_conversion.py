import json
import os
import random
import re
import shutil
import string
import threading
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import polars as pl
import pymupdf
from gradio import Progress
from PIL import Image, ImageFile
from pymupdf import Document, Page
from scipy.spatial import cKDTree
from tqdm import tqdm

from tools.config import (
    COMPRESS_REDACTED_PDF,
    IMAGES_DPI,
    INPUT_FOLDER,
    LOAD_REDACTION_ANNOTATIONS_FROM_PDF,
    LOAD_TRUNCATED_IMAGES,
    MAX_IMAGE_PIXELS,
    MAX_SIMULTANEOUS_FILES,
    MAX_WORKERS,
    OUTPUT_FOLDER,
    SELECTABLE_TEXT_EXTRACT_OPTION,
    TESSERACT_TEXT_EXTRACT_OPTION,
    TEXTRACT_TEXT_EXTRACT_OPTION,
)
from tools.helper_functions import get_file_name_without_type, read_file
from tools.secure_path_utils import secure_file_read, secure_join
from tools.secure_regex_utils import safe_extract_page_number_from_path

IMAGE_NUM_REGEX = re.compile(r"_(\d+)\.png$")

pd.set_option("future.no_silent_downcasting", True)

image_dpi = float(IMAGES_DPI)
if not MAX_IMAGE_PIXELS:
    Image.MAX_IMAGE_PIXELS = None
else:
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

ImageFile.LOAD_TRUNCATED_IMAGES = LOAD_TRUNCATED_IMAGES


_PDF_DOC_CACHE = threading.local()


def _get_threadlocal_pymupdf_doc(pdf_path: str) -> Document:
    """
    Cache a PyMuPDF Document per thread to avoid reopening the same PDF for each page.

    ThreadPoolExecutor threads are long-lived for the duration of the pool, so this
    cuts overhead significantly when processing many pages.
    """
    cache = getattr(_PDF_DOC_CACHE, "docs", None)
    if cache is None:
        cache = {}
        _PDF_DOC_CACHE.docs = cache
    doc = cache.get(pdf_path)
    if doc is None:
        doc = pymupdf.open(pdf_path)
        cache[pdf_path] = doc
    return doc


def _render_pdf_page_to_png_pymupdf_mediabox(
    pdf_path: str,
    page_num: int,
    out_path: str,
    dpi: float,
) -> Image.Image:
    """
    Render a single PDF page to a grayscale PNG using PyMuPDF, ensuring MediaBox render.

    PyMuPDF's Page.get_pixmap() respects the CropBox; to render the MediaBox we
    temporarily set CropBox=MediaBox and restore it afterwards.
    """
    doc = _get_threadlocal_pymupdf_doc(pdf_path)
    page = doc.load_page(page_num)

    old_crop = page.cropbox
    old_rot = page.rotation
    try:
        page.set_cropbox(page.mediabox)
        # Preserve the PDF's intrinsic rotation (e.g. 180deg pages).
        # Downstream coordinate logic assumes the rendered image matches PyMuPDF's display space.
        page.set_rotation(old_rot)
        pix = page.get_pixmap(
            dpi=int(dpi) if dpi is not None else None,
            colorspace=pymupdf.csGRAY,
            alpha=False,
            annots=False,
        )
        # Fast path: write PNG via MuPDF, then load with PIL for downstream resizing.
        pix.save(out_path)
        # Embed DPI in PNG (MuPDF write does not set pHYs); matches IMAGES_DPI render scale.
        _dpi = max(1, int(round(float(dpi))))
        _pil = Image.open(out_path)
        _pil.save(out_path, format="PNG", dpi=(_dpi, _dpi))
        return _pil
    finally:
        page.set_cropbox(old_crop)
        if old_rot != 0:
            page.set_rotation(old_rot)


def is_pdf_or_image(filename):
    """
    Check if a file name is a PDF or an image file.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if the file name ends with ".pdf", ".jpg", or ".png", False otherwise.
    """
    if (
        filename.lower().endswith(".pdf")
        or filename.lower().endswith(".jpg")
        or filename.lower().endswith(".jpeg")
        or filename.lower().endswith(".png")
    ):
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


def check_image_size_and_reduce(out_path: str, image: Image):
    """
    Check if a given image size is above around 4.5mb, and reduce size if necessary.
    5mb is the maximum possible to submit to AWS Textract.

    Args:
        out_path (str): The file path where the image is currently saved and will be saved after resizing.
        image (Image): The PIL Image object to be checked and potentially resized.
    """

    all_img_details = list()
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
            _dp = max(1, int(round(float(image_dpi))))
            image.save(out_path, format="PNG", optimize=True, dpi=(_dp, _dp))

            # Update the file size
            file_size = os.path.getsize(out_path)
            print(f"Resized to {new_width}x{new_height}, new file_size: {file_size}")
    else:
        new_width = width
        new_height = height

    all_img_details.append((page_num, image, new_width, new_height))

    return image, new_width, new_height, all_img_details, out_path


def process_single_page_for_image_conversion(
    pdf_path: str,
    page_num: int,
    image_dpi: float = image_dpi,
    create_images: bool = True,
    input_folder: str = INPUT_FOLDER,
) -> tuple[int, str, float, float]:
    """
    Processes a single page of a PDF or image file for image conversion,
    saving it as a PNG and optionally resizing it if too large.

    Args:
        pdf_path (str): The path to the input PDF or image file.
        page_num (int): The 0-indexed page number to process.
        image_dpi (float, optional): The DPI to use for PDF to image conversion. Defaults to image_dpi from config.
        create_images (bool, optional): Whether to create and save the image. Defaults to True.
        input_folder (str, optional): The folder where the converted images will be saved. Defaults to INPUT_FOLDER from config.

    Returns:
        tuple[int, str, float, float]: A tuple containing:
            - The processed page number.
            - The path to the saved output image.
            - The width of the processed image.
            - The height of the processed image.
    """

    out_path_placeholder = "placeholder_image_" + str(page_num) + ".png"

    if create_images is True:
        try:
            # Construct the full output directory path
            # Normalize input_folder to ensure it's used as-is without sanitization
            if os.path.isabs(input_folder):
                image_output_dir = Path(input_folder).resolve()
            else:
                # Join with cwd, but ensure input_folder is used as-is
                base_dir = Path(os.getcwd()).resolve()
                # Use Path.joinpath which doesn't sanitize folder names
                image_output_dir = base_dir / input_folder
                image_output_dir = image_output_dir.resolve()

            # Ensure the directory exists
            image_output_dir.mkdir(parents=True, exist_ok=True)

            # Construct the output file path using secure_path_join for the filename only
            from tools.secure_path_utils import secure_path_join

            out_path = secure_path_join(
                image_output_dir, f"{os.path.basename(pdf_path)}_{page_num}.png"
            )
            # Convert Path object to string immediately to avoid downstream type issues
            out_path = str(out_path)

            if os.path.exists(out_path):
                # Load existing image
                image = Image.open(out_path)
            elif pdf_path.lower().endswith(".pdf"):
                # Convert PDF page to image (MediaBox) using PyMuPDF for speed.
                # We render directly as grayscale and save as PNG.
                image = _render_pdf_page_to_png_pymupdf_mediabox(
                    pdf_path=pdf_path,
                    page_num=page_num,
                    out_path=out_path,
                    dpi=image_dpi,
                )
            elif (
                pdf_path.lower().endswith(".jpg")
                or pdf_path.lower().endswith(".png")
                or pdf_path.lower().endswith(".jpeg")
            ):
                image = Image.open(pdf_path)
                _dp = max(1, int(round(float(image_dpi))))
                image.save(out_path, format="PNG", dpi=(_dp, _dp))
            else:
                raise Warning("Could not create image.")

            width, height = image.size

            # Check if image size too large and reduce if necessary
            # print("Checking size of image and reducing if necessary.")
            image, width, height, all_img_details, img_path = (
                check_image_size_and_reduce(out_path, image)
            )

            return page_num, out_path, width, height

        except Exception as e:

            print(f"Error processing page {page_num + 1}: {e}")
            return page_num, out_path_placeholder, pd.NA, pd.NA
    else:
        # print("Not creating image for page", page_num)
        return page_num, out_path_placeholder, pd.NA, pd.NA


def convert_pdf_to_images(
    pdf_path: str,
    prepare_for_review: bool = False,
    page_min: int = 0,
    page_max: int = 0,
    create_images: bool = True,
    image_dpi: float = image_dpi,
    num_threads: Optional[int] = None,
    input_folder: str = INPUT_FOLDER,
    progress: Progress = Progress(track_tqdm=True),
    page_numbers: Optional[List[int]] = None,
):
    """
    Converts a PDF document into a series of images, processing each page concurrently.

    Args:
        pdf_path (str): The path to the PDF file to convert.
        prepare_for_review (bool, optional): If True, only the first page is processed (feature not currently used). Defaults to False.
        page_min (int, optional): The starting page number (0-indexed) for conversion. If 0, uses the first page. Defaults to 0.
        page_max (int, optional): The ending page number (exclusive, 0-indexed) for conversion. If 0, uses the last page of the document. Defaults to 0.
        create_images (bool, optional): If True, images are created and saved to disk. Defaults to True.
        image_dpi (float, optional): The DPI (dots per inch) to use for converting PDF pages to images. Defaults to the global `image_dpi`.
        num_threads (int, optional): The number of threads to use for concurrent page processing. Defaults to MAX_WORKERS from config/env.
        input_folder (str, optional): The base input folder, used for determining output paths. Defaults to `INPUT_FOLDER`.
        page_numbers (list, optional): If provided, only these 0-indexed page numbers are converted; page_min/page_max are ignored.

    Returns:
        list: A list of tuples, where each tuple contains (page_num, image_path, width, height) for successfully processed pages.
              For failed pages, it returns (page_num, placeholder_path, pd.NA, pd.NA).
    """
    if num_threads is None:
        num_threads = MAX_WORKERS

    # Page count via PyMuPDF (faster + avoids Poppler dependency for this step)
    try:
        _count_doc = pymupdf.open(pdf_path)
        page_count = len(_count_doc)
    finally:
        try:
            _count_doc.close()
        except Exception:
            pass

    # If preparing for review, just load the first page (not currently used)
    if prepare_for_review is True:
        page_min = 0
        page_max = page_count
        page_numbers = None

    if page_numbers is not None:
        pages_to_convert = sorted(
            set(int(p) for p in page_numbers if 0 <= p < page_count)
        )
        total_pages = len(pages_to_convert)
        if total_pages == 0:
            return [], [], [], []
        print(f"Creating images for {total_pages} page(s) (EFFICIENT_OCR).")
    else:
        print(f"Creating images. Number of pages in PDF: {page_count}")
        # Handle special cases for page range
        if page_min == 0:
            page_min = 0
        else:
            page_min = page_min - 1
        if page_max == 0:
            page_max = page_count
        pages_to_convert = list(range(page_min, page_max))
        total_pages = len(pages_to_convert)

    progress(0.1, desc="Creating images")

    results = list()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = list()
        for page_num in pages_to_convert:
            futures.append(
                executor.submit(
                    process_single_page_for_image_conversion,
                    pdf_path,
                    page_num,
                    image_dpi,
                    create_images=create_images,
                    input_folder=input_folder,
                )
            )

        completed = 0
        # Throttle Gradio updates to ~every 2% or every 10 pages so UI stays responsive
        update_interval = max(1, min(total_pages // 50, 10))
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            unit="pages",
            desc="Converting pages to image",
        ):
            page_num, img_path, width, height = future.result()
            completed += 1
            # Report progress to Gradio so the upload component shows page-by-page progress
            if completed % update_interval == 0 or completed == total_pages:
                progress(
                    0.1 + 0.8 * (completed / total_pages),
                    desc="Converting pages to image",
                )
            if img_path:
                results.append((page_num, img_path, width, height))
            else:
                print(f"Page {page_num + 1} failed to process.")
                results.append(
                    (
                        page_num,
                        "placeholder_image_" + str(page_num) + ".png",
                        pd.NA,
                        pd.NA,
                    )
                )

    # Sort results by page number
    progress(0.95, desc="Loading images")
    results.sort(key=lambda x: x[0])
    images = [result[1] for result in results]
    widths = [result[2] for result in results]
    heights = [result[3] for result in results]

    print("PDF has been converted to images.")
    return images, widths, heights, results


# Function to take in a file path, decide if it is an image or pdf, then process appropriately.
def process_file_for_image_creation(
    file_path: str,
    prepare_for_review: bool = False,
    input_folder: str = INPUT_FOLDER,
    create_images: bool = True,
    page_min: int = 0,
    page_max: int = 0,
    progress: Progress = Progress(track_tqdm=True),
):
    """
    Processes a given file path, determining if it's an image or a PDF,
    and then converts it into a list of image paths, along with their dimensions.

    Args:
        file_path (str): The path to the file (image or PDF) to be processed.
        prepare_for_review (bool, optional): If True, prepares the PDF for review
                                             (e.g., by converting pages to images). Defaults to False.
        input_folder (str, optional): The folder where input files are located. Defaults to INPUT_FOLDER.
        create_images (bool, optional): If True, images will be created from PDF pages.
                                        If False, only metadata will be extracted. Defaults to True.
        page_min (int, optional): The minimum page number to process (0-indexed). If 0, uses the first page. Defaults to 0.
        page_max (int, optional): The maximum page number to process (0-indexed). If 0, uses the last page of the document. Defaults to 0.
        progress (Progress, optional): The progress object to update. Defaults to a Progress object with track_tqdm=True.
    """
    # Get the file extension
    file_extension = os.path.splitext(file_path)[1].lower()

    # Check if the file is an image type
    if file_extension in [".jpg", ".jpeg", ".png"]:
        print(f"{file_path} is an image file.")
        progress(0.1, desc="Processing image file")
        # Perform image processing here
        img_object = [file_path]  # [Image.open(file_path)]

        # Load images from the file paths. Test to see if it is bigger than 4.5 mb and reduct if needed (Textract limit is 5mb)
        image = Image.open(file_path)
        img_object, image_sizes_width, image_sizes_height, all_img_details, img_path = (
            check_image_size_and_reduce(file_path, image)
        )

        if not isinstance(image_sizes_width, list):
            img_path = [img_path]
            image_sizes_width = [image_sizes_width]
            image_sizes_height = [image_sizes_height]
            all_img_details = [all_img_details]

    # Check if the file is a PDF
    elif file_extension == ".pdf":

        # Run your function for processing PDF files here
        img_path, image_sizes_width, image_sizes_height, all_img_details = (
            convert_pdf_to_images(
                file_path,
                prepare_for_review,
                page_min=page_min,
                page_max=page_max,
                input_folder=input_folder,
                create_images=create_images,
                progress=progress,
            )
        )

    else:
        print(f"{file_path} is not an image or PDF file.")
        img_path = list()
        image_sizes_width = list()
        image_sizes_height = list()
        all_img_details = list()

    return img_path, image_sizes_width, image_sizes_height, all_img_details


def _process_one_input_file(
    file: Any,
    source_document_only: bool,
    source_document_extensions: tuple,
) -> Tuple[str, str, str, bool, bool, int]:
    """
    Process a single file for get_input_file_names; safe to run in a thread.
    Returns (file_path_without_ext, file_extension, file_path, acceptable, is_source, page_count).
    """
    file_path = file if isinstance(file, str) else file.name
    file_path_without_ext = get_file_name_without_type(file_path)
    file_path_without_ext_lower = (file_path_without_ext or "").lower()
    file_extension = os.path.splitext(file_path)[1].lower()
    is_excluded_name = (
        "review_file" in file_path_without_ext_lower
        or "ocr_output" in file_path_without_ext_lower
        or "ocr_results_with_words" in file_path_without_ext_lower
    )
    acceptable = (
        file_extension
        in (".jpg", ".jpeg", ".png", ".pdf", ".xlsx", ".csv", ".parquet", ".docx")
        and not is_excluded_name
    )
    if file_extension == ".pdf":
        try:
            pdf_document = pymupdf.open(file_path)
            page_count = pdf_document.page_count
            pdf_document.close()
        except Exception:
            page_count = 1
    else:
        page_count = 1
    is_source = not source_document_only or file_extension in source_document_extensions
    return (
        file_path_without_ext,
        file_extension,
        file_path,
        acceptable,
        is_source,
        page_count,
    )


def get_input_file_names(
    file_input: List[str],
    source_document_only: bool = False,
):
    """
    Get list of input files to report to logs.

    When source_document_only is True (e.g. for document redaction / review tab),
    full_file_name is only set for source documents (PDF, image, or Word), never
    for outputs like review CSVs or OCR CSVs. This keeps doc_full_file_name_textbox
    referring to the document being redacted (PDF or image).
    """
    all_relevant_files = list()
    file_name_with_extension = ""
    full_file_name = ""
    total_pdf_page_count = 0
    source_document_extensions = (".pdf", ".jpg", ".jpeg", ".png", ".docx")

    if isinstance(file_input, dict):
        file_input = os.path.abspath(file_input["name"])

    if isinstance(file_input, str):
        file_input_list = [file_input]
    else:
        file_input_list = file_input

    if not file_input_list:
        return (
            ", ".join(all_relevant_files),
            file_name_with_extension,
            full_file_name,
            all_relevant_files,
            total_pdf_page_count,
        )

    max_workers = min(MAX_WORKERS, len(file_input_list))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                lambda f: _process_one_input_file(
                    f, source_document_only, source_document_extensions
                ),
                file_input_list,
            )
        )

    for (
        file_path_without_ext,
        file_extension,
        file_path,
        acceptable,
        is_source,
        page_count,
    ) in results:
        total_pdf_page_count += page_count
        if acceptable:
            all_relevant_files.append(file_path_without_ext)
            file_name_with_extension = file_path_without_ext + file_extension
            if is_source:
                full_file_name = file_path

    all_relevant_files_str = ", ".join(all_relevant_files)
    print("file_name_with_extension on document upload:", file_name_with_extension)
    return (
        all_relevant_files_str,
        file_name_with_extension,
        full_file_name,
        all_relevant_files,
        total_pdf_page_count,
    )


def get_document_file_names(file_input: List[str]):
    """
    Same as get_input_file_names but with source_document_only=True, so the
    returned full_file_name is only ever a PDF, image, or Word doc (the document
    being redacted), never a review CSV or OCR output. Use this for flows that
    update doc_full_file_name_textbox.
    """
    return get_input_file_names(file_input, source_document_only=True)


def convert_pymupdf_to_image_coords(
    pymupdf_page: Page,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image: Image = None,
    image_dimensions: dict = dict(),
):
    """
    Converts bounding box coordinates from PyMuPDF page format to image coordinates.

    PyMuPDF uses coordinates relative to ``page.rect`` (the visible area). The
    rectangle is often **normalised** so its top-left is (0, 0) even when the CropBox
    is inset in the MediaBox, so the inset must be taken from ``page.cropbox`` vs
    ``page.mediabox`` (same as ``off_x`` / ``off_y`` in
    ``process_page_to_structured_ocr_pymupdf``). Review/annotation images are rendered
    from the full MediaBox (see ``_render_pdf_page_to_png_pymupdf_mediabox``), so we
    shift rect-local points to MediaBox-local, then scale by image_size /
    mediabox_size. This replaces the old symmetric (mediabox−rect)/2 heuristic, which
    was wrong for asymmetric crops.

    Args:
        pymupdf_page (Page): The PyMuPDF page object from which the coordinates originate.
        x1 (float): The x-coordinate of the top-left corner in PyMuPDF page units.
        y1 (float): The y-coordinate of the top-left corner in PyMuPDF page units.
        x2 (float): The x-coordinate of the bottom-right corner in PyMuPDF page units.
        y2 (float): The y-coordinate of the bottom-right corner in PyMuPDF page units.
        image (Image, optional): A PIL Image object. If provided, its dimensions
                                 are used as the target image dimensions. Defaults to None.
        image_dimensions (dict, optional): A dictionary containing 'image_width' and
                                           'image_height'. Used if 'image' is not provided
                                           and 'image' is None. Defaults to an empty dictionary.
    """
    mediabox = pymupdf_page.mediabox
    cropbox = pymupdf_page.cropbox
    mediabox_width = mediabox.width
    mediabox_height = mediabox.height

    if mediabox_width <= 0 or mediabox_height <= 0:
        return x1, y1, x2, y2

    if image:
        image_page_width, image_page_height = image.size
    elif image_dimensions:
        image_page_width, image_page_height = (
            image_dimensions["image_width"],
            image_dimensions["image_height"],
        )
    else:
        image_page_width, image_page_height = mediabox_width, mediabox_height

    sx = image_page_width / mediabox_width
    sy = image_page_height / mediabox_height

    # Rect-local → MediaBox-local: use cropbox vs mediabox (rect may be normalised to
    # origin 0,0 while cropbox keeps PDF placement).
    dx = cropbox.x0 - mediabox.x0
    dy = cropbox.y0 - mediabox.y0

    x1_image = (x1 + dx) * sx
    x2_image = (x2 + dx) * sx
    y1_image = (y1 + dy) * sy
    y2_image = (y2 + dy) * sy

    return x1_image, y1_image, x2_image, y2_image


def create_page_size_objects(
    pymupdf_doc: Document,
    image_sizes_width: List[float],
    image_sizes_height: List[float],
    image_file_paths: List[str],
    page_min: int = 0,
    page_max: int = 0,
):
    """
    Creates page size objects for a PyMuPDF document.

    Creates entries for ALL pages in the document. Pages that were processed for image creation
    will have actual image paths and dimensions. Pages that were not processed will have
    placeholder image paths and no image dimensions.

    Args:
        pymupdf_doc (Document): The PyMuPDF document object.
        image_sizes_width (List[float]): List of image widths for processed pages.
        image_sizes_height (List[float]): List of image heights for processed pages.
        image_file_paths (List[str]): List of image file paths for processed pages.
        page_min (int, optional): The minimum page number that was processed (0-indexed). If 0, uses the first page. Defaults to 0.
        page_max (int, optional): The maximum page number that was processed (0-indexed). If 0, uses the last page of the document. Defaults to 0.
    """
    page_sizes = list()
    original_cropboxes = list()

    # Handle special cases for page range
    # If page_min is 0, use the first page (0-indexed)
    if page_min == 0:
        page_min = 0  # First page is 0-indexed
    else:
        page_min = page_min - 1

    # If page_max is 0, use the last page of the document
    if page_max == 0:
        page_max = len(pymupdf_doc)

    # Process ALL pages in the document, not just the ones with images
    for page_no in range(len(pymupdf_doc)):
        reported_page_no = page_no + 1
        pymupdf_page = pymupdf_doc.load_page(page_no)
        original_cropboxes.append(pymupdf_page.cropbox)  # Save original CropBox

        # Check if this page was processed for image creation
        is_page_in_range = page_min <= page_no < page_max
        image_index = page_no - page_min if is_page_in_range else None

        # Create a page_sizes_object for every page
        out_page_image_sizes = {
            "page": reported_page_no,
            "mediabox_width": pymupdf_page.mediabox.width,
            "mediabox_height": pymupdf_page.mediabox.height,
            "cropbox_width": pymupdf_page.cropbox.width,
            "cropbox_height": pymupdf_page.cropbox.height,
            "original_cropbox": original_cropboxes[-1],
        }

        # cropbox_x_offset: Distance from MediaBox left edge to CropBox left edge
        # This is simply the difference in their x0 coordinates.
        out_page_image_sizes["cropbox_x_offset"] = (
            pymupdf_page.cropbox.x0 - pymupdf_page.mediabox.x0
        )

        # cropbox_y_offset_from_top: Distance from MediaBox top edge to CropBox top edge
        out_page_image_sizes["cropbox_y_offset_from_top"] = (
            pymupdf_page.mediabox.y1 - pymupdf_page.cropbox.y1
        )

        # Set image path and dimensions based on whether this page was processed
        if (
            is_page_in_range
            and image_index is not None
            and image_index < len(image_file_paths)
        ):
            # This page was processed for image creation
            out_page_image_sizes["image_path"] = image_file_paths[image_index]

            # Add image dimensions if available
            if (
                image_sizes_width
                and image_sizes_height
                and image_index < len(image_sizes_width)
                and image_index < len(image_sizes_height)
            ):
                out_page_image_sizes["image_width"] = image_sizes_width[image_index]
                out_page_image_sizes["image_height"] = image_sizes_height[image_index]
        else:
            # This page was not processed for image creation - use placeholder
            out_page_image_sizes["image_path"] = f"image_placeholder_{page_no}.png"
            # No image dimensions for placeholder pages

        page_sizes.append(out_page_image_sizes)

    return page_sizes, original_cropboxes


def prepare_images_for_pages(
    file_path: str,
    pages_1based: List[int],
    input_folder: str,
    pymupdf_doc: Document,
    page_sizes: List[dict],
    progress: Progress = Progress(track_tqdm=True),
) -> Tuple[List[str], List[dict]]:
    """
    Create images only for the given pages (e.g. EFFICIENT_OCR pages that need OCR).
    Updates page_sizes in place and returns a full-length pdf_image_file_paths list
    (real paths only for the requested pages, empty string for others).

    Args:
        file_path: Path to the PDF.
        pages_1based: 1-based page numbers to convert to images.
        input_folder: Folder used for image output paths.
        pymupdf_doc: Open PyMuPDF document (used for page count).
        page_sizes: List of page size dicts (one per page), updated in place.
        progress: Progress callback.

    Returns:
        (pdf_image_file_paths, page_sizes) where pdf_image_file_paths has length
        len(pymupdf_doc) with real path at index (p-1) for each p in pages_1based.
    """
    if not pages_1based:
        return [""] * len(pymupdf_doc), page_sizes

    page_numbers_0based = [p - 1 for p in pages_1based if 1 <= p <= len(pymupdf_doc)]
    if not page_numbers_0based:
        return [""] * len(pymupdf_doc), page_sizes

    _, _, _, results = convert_pdf_to_images(
        file_path,
        prepare_for_review=False,
        create_images=True,
        input_folder=input_folder,
        progress=progress,
        page_numbers=page_numbers_0based,
    )

    num_pages = len(pymupdf_doc)
    pdf_image_file_paths = [""] * num_pages
    for page_num, img_path, width, height in results:
        if (
            0 <= page_num < num_pages
            and img_path
            and "placeholder" not in str(img_path)
        ):
            pdf_image_file_paths[page_num] = img_path
            page_sizes[page_num]["image_path"] = img_path
            if pd.notna(width) and pd.notna(height):
                page_sizes[page_num]["image_width"] = width
                page_sizes[page_num]["image_height"] = height

    return pdf_image_file_paths, page_sizes


def _get_bbox(d: dict) -> list:
    """Get bounding box list from dict; support both 'bounding_box' and 'boundingBox'."""
    return d.get("bounding_box") or d.get("boundingBox") or [0, 0, 0, 0]


def word_level_ocr_output_to_dataframe(ocr_results: dict) -> pd.DataFrame:
    """
    Convert a json of ocr results to a dataframe

    Args:
        ocr_results (dict): A dictionary containing OCR results.

    Returns:
        pd.DataFrame: A dataframe containing the OCR results.
    """
    rows = list()
    ocr_results[0]

    for ocr_result in ocr_results:

        page_number = int(ocr_result["page"])

        for line_key, line_data in ocr_result["results"].items():

            line_number = int(line_data["line"])
            # Support both "confidence" (Textract/json_to_ocrresult) and "conf" (other OCR)
            line_conf = line_data.get("confidence", line_data.get("conf", 100.0))
            line_bbox = _get_bbox(line_data)
            for word in line_data["words"]:
                word_conf = word.get("confidence", word.get("conf", 100.0))
                word_bbox = _get_bbox(word)
                rows.append(
                    {
                        "page": page_number,
                        "line": line_number,
                        "word_text": word["text"],
                        "word_x0": word_bbox[0],
                        "word_y0": word_bbox[1],
                        "word_x1": word_bbox[2],
                        "word_y1": word_bbox[3],
                        "word_conf": word_conf,
                        "line_text": "",  # line_data['text'], # This data is too large to include
                        "line_x0": line_bbox[0],
                        "line_y0": line_bbox[1],
                        "line_x1": line_bbox[2],
                        "line_y1": line_bbox[3],
                        "line_conf": line_conf,
                    }
                )

    return pd.DataFrame(rows)


def word_level_ocr_df_to_line_level_ocr_df(
    word_level_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert word-level OCR results dataframe to line-level OCR results dataframe.

    Word-level format has one row per word (page, line, word_text, word_x0, word_y0,
    word_x1, word_y1, word_conf, ...). Line-level format has one row per line with
    aggregated text and bounding box (page, text, left, top, width, height, line, conf).

    Args:
        word_level_df: DataFrame with columns including page, line, word_text,
            word_x0, word_y0, word_x1, word_y1, and word_conf (or line_conf).

    Returns:
        DataFrame with columns page, text, left, top, width, height, line, conf.
    """
    required = ["page", "line", "word_text", "word_x0", "word_y0", "word_x1", "word_y1"]
    for col in required:
        if col not in word_level_df.columns:
            raise ValueError(
                f"word_level_df must contain column '{col}'. "
                f"Found: {list(word_level_df.columns)}"
            )

    def agg_line(group: pd.DataFrame) -> pd.Series:
        text = " ".join(group["word_text"].astype(str).dropna())
        x0 = group["word_x0"].min()
        y0 = group["word_y0"].min()
        x1 = group["word_x1"].max()
        y1 = group["word_y1"].max()
        if "line_conf" in group.columns and group["line_conf"].notna().any():
            conf = group["line_conf"].dropna().iloc[0]
        else:
            conf = group["word_conf"].mean() if "word_conf" in group.columns else 100.0
        return pd.Series(
            {
                "text": text,
                "left": x0,
                "top": y0,
                "width": x1 - x0,
                "height": y1 - y0,
                "conf": conf,
            }
        )

    line_level = (
        word_level_df.groupby(["page", "line"], sort=False)
        .apply(agg_line)
        .reset_index()
    )
    # Match expected column order: page, text, left, top, width, height, line, conf
    return line_level[
        ["page", "text", "left", "top", "width", "height", "line", "conf"]
    ]


def extract_redactions(
    doc: Document, page_sizes: List[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], Document]:
    """
    Extracts all redaction annotations from a PDF document and converts them
    to Gradio Annotation JSON format.

    Note: This function identifies the *markings* for redaction. It does not
    tell you if the redaction has been *applied* (i.e., the underlying
    content is permanently removed).

    Args:
        doc: The PyMuPDF document object.
        page_sizes: List of dictionaries containing page information with keys:
                   'page', 'image_path', 'image_width', 'image_height'.
                   If None, will create placeholder structure.

    Returns:
        List of dictionaries suitable for Gradio Annotation output, one dict per image/page.
        PyMuPDF document object.
        Each dict has structure: {"image": image_path, "boxes": [list of annotation boxes]}
    """

    # Helper function to generate unique IDs
    def _generate_unique_ids(num_ids: int, existing_ids: set = None) -> List[str]:
        if existing_ids is None:
            existing_ids = set()

        id_length = 12
        character_set = string.ascii_letters + string.digits
        unique_ids = list()

        for _ in range(num_ids):
            while True:
                candidate_id = "".join(random.choices(character_set, k=id_length))
                if candidate_id not in existing_ids:
                    existing_ids.add(candidate_id)
                    unique_ids.append(candidate_id)
                    break

        return unique_ids

    # Extract redaction annotations from the document
    redactions_by_page = dict()
    existing_ids = set()

    for page_num, page in enumerate(doc):
        page_redactions = list()

        # The page.annots() method is a generator for all annotations on the page
        for annot in page.annots():
            # The type of a redaction annotation is 12
            if annot.type[0] == pymupdf.PDF_ANNOT_REDACT:

                # Get annotation info with fallbacks
                annot_info = annot.info or {}
                annot_colors = annot.colors or {}

                # Extract coordinates from the annotation rectangle (PDF space, same units as mediabox)
                rect = annot.rect
                x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1

                # Convert PDF coordinates to image pixel coordinates (always scale by image size)
                page_size_info = None
                if page_sizes:
                    for ps in page_sizes:
                        if ps.get("page") == page_num + 1:
                            page_size_info = ps
                            break

                if not page_size_info:
                    raise ValueError(
                        f"extract_redactions: no page_sizes entry for page {page_num + 1}. "
                        "Ensure page_sizes is built and images exist before extracting redactions."
                    )

                mediabox_width = page_size_info.get("mediabox_width", 1)
                mediabox_height = page_size_info.get("mediabox_height", 1)
                image_width = page_size_info.get("image_width")
                image_height = page_size_info.get("image_height")

                try:
                    w = float(image_width) if image_width is not None else 0
                    h = float(image_height) if image_height is not None else 0
                    has_valid_image_dims = w > 0 and h > 0
                except (TypeError, ValueError):
                    has_valid_image_dims = False

                if not has_valid_image_dims:
                    scale_x = 1
                    scale_y = 1
                    rel_x0 = x0
                    rel_y0 = y0
                    rel_x1 = x1
                    rel_y1 = y1
                    # raise ValueError(
                    #     f"extract_redactions: page {page_num + 1} has no valid image dimensions "
                    #     "(image_width/image_height). Create images for all pages before loading redactions."
                    # )
                else:
                    scale_x = w / mediabox_width
                    scale_y = h / mediabox_height
                    rel_x0 = x0 * scale_x
                    rel_y0 = y0 * scale_y
                    rel_x1 = x1 * scale_x
                    rel_y1 = y1 * scale_y

                # Get color and convert from 0-1 range to 0-255 range
                fill_color = annot_colors.get(
                    "fill", (0, 0, 0)
                )  # Default to black if no color
                if isinstance(fill_color, (tuple, list)) and len(fill_color) >= 3:
                    # Convert from 0-1 range to 0-255 range
                    color_255 = tuple(
                        int(component * 255) if component <= 1 else int(component)
                        for component in fill_color[:3]
                    )
                else:
                    color_255 = (0, 0, 0)  # Default to black

                # Create annotation box in the required format
                redaction_box = {
                    "label": annot_info.get(
                        "title", f"Redaction {len(page_redactions) + 1}"
                    ),
                    "color": str(color_255),
                    "xmin": rel_x0,
                    "ymin": rel_y0,
                    "xmax": rel_x1,
                    "ymax": rel_y1,
                    "text": annot_info.get("content", ""),
                    "id": None,  # Will be filled after generating IDs
                }

                page_redactions.append(redaction_box)

                # Remove the redaction annotation from the pymupdf document
                page.delete_annot(annot)

        # if page.annots:
        #     page.annots = [
        #         annot
        #         for annot in page.annots()
        #         if annot.type[0] != pymupdf.PDF_ANNOT_REDACT
        #     ]

        if page_redactions:
            redactions_by_page[page_num + 1] = page_redactions

    # Generate unique IDs for all redaction boxes
    all_boxes = list()
    for page_redactions in redactions_by_page.values():
        all_boxes.extend(page_redactions)

    if all_boxes:
        unique_ids = _generate_unique_ids(len(all_boxes), existing_ids)

        # Assign IDs to boxes
        box_idx = 0
        for page_num, page_redactions in redactions_by_page.items():
            for box in page_redactions:
                box["id"] = unique_ids[box_idx]
                box_idx += 1

    # Build JSON structure based on page_sizes or create placeholder structure
    json_data = list()

    if page_sizes:
        # Use provided page_sizes to build structure
        for page_info in page_sizes:
            page_num = page_info.get("page", 1)
            image_path = page_info.get(
                "image_path", f"placeholder_image_{page_num}.png"
            )

            # Get redactions for this page
            annotation_boxes = redactions_by_page.get(page_num, [])

            json_data.append({"image": image_path, "boxes": annotation_boxes})
    else:
        # Create placeholder structure based on document pages
        for page_num in range(1, doc.page_count + 1):
            image_path = f"placeholder_image_{page_num}.png"
            annotation_boxes = redactions_by_page.get(page_num, [])

            json_data.append({"image": image_path, "boxes": annotation_boxes})

    total_redactions = sum(len(boxes) for boxes in redactions_by_page.values())
    print(f"Found {total_redactions} redactions in the document")

    # Convert the gradio annotation boxes to relative coordinates
    page_sizes_df = pd.DataFrame(page_sizes)
    page_sizes_df.loc[:, "page"] = pd.to_numeric(page_sizes_df["page"], errors="coerce")

    all_image_annotations_df = convert_annotation_data_to_dataframe(json_data)
    all_image_annotations_df = divide_coordinates_by_page_sizes(
        all_image_annotations_df,
        page_sizes_df,
        xmin="xmin",
        xmax="xmax",
        ymin="ymin",
        ymax="ymax",
    )
    annotations_all_pages_divide = create_annotation_dicts_from_annotation_df(
        all_image_annotations_df, page_sizes
    )

    return annotations_all_pages_divide, doc


def _rects_match(rect_a, rect_b, tolerance: float = 0.5) -> bool:
    """Return True if two PyMuPDF rects are the same within tolerance (in points)."""
    return (
        abs(rect_a.x0 - rect_b.x0) <= tolerance
        and abs(rect_a.y0 - rect_b.y0) <= tolerance
        and abs(rect_a.x1 - rect_b.x1) <= tolerance
        and abs(rect_a.y1 - rect_b.y1) <= tolerance
    )


def _dst_page_has_duplicate_redact(dst_page, rect, title: str, content: str) -> bool:
    """Return True if dst_page already has a redaction annot with same rect, title, and content."""
    title = (title or "").strip()
    content = (content or "").strip()
    for existing in dst_page.annots():
        if existing.type[0] != pymupdf.PDF_ANNOT_REDACT:
            continue
        if not _rects_match(rect, existing.rect):
            continue
        info = existing.info or {}
        existing_title = (info.get("title") or "").strip()
        existing_content = (info.get("content") or "").strip()
        if existing_title == title and existing_content == content:
            return True
    return False


def _get_base_name_from_review_pdf_path(file_path: str) -> str:
    """
    Extract the base file name from a '_redactions_for_review...' path.
    E.g. 'mydoc_redactions_for_review.pdf' -> 'mydoc',
         'mydoc_redactions_for_review_pages_1-2.pdf' -> 'mydoc'.
    """
    basename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(basename)[0]
    suffix = "_redactions_for_review"
    if suffix in name_without_ext:
        return name_without_ext.split(suffix)[0]
    return name_without_ext


def _parse_review_pdf_page_suffix(
    file_path: str,
) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    If the review PDF path ends with a page-range suffix _N_M (e.g. _2_4), return
    (True, N, M). Otherwise return (False, None, None).
    E.g. 'mydoc_redactions_for_review_2_4.pdf' -> (True, 2, 4)
         'mydoc_redactions_for_review.pdf' -> (False, None, None)
    """
    basename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(basename)[0]
    match = re.search(r"_(\d+)_(\d+)$", name_without_ext)
    if match:
        return True, int(match.group(1)), int(match.group(2))
    return False, None, None


def _get_review_pdf_combined_output_base(file_path: str) -> str:
    """
    From a review PDF path, get the base for the combined output filename:
    everything up to and including '_redactions_for_review', excluding any
    text after that (e.g. " (1)", " (2)", "_2_4").
    E.g. 'file_redactions_for_review (1).pdf' -> 'file_redactions_for_review'
         'file_FINAL_redactions_for_review.pdf' -> 'file_FINAL_redactions_for_review'
    """
    basename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(basename)[0]
    suffix = "_redactions_for_review"
    if suffix in name_without_ext:
        idx = name_without_ext.index(suffix) + len(suffix)
        return name_without_ext[:idx]
    return name_without_ext


def combine_review_pdf_files(file_list, output_folder: str = OUTPUT_FOLDER):
    """
    Combine redaction comments from multiple '_redactions_for_review' PDFs into one PDF.

    Only validates that all files have the same number of pages. File names may
    differ (e.g. file_redactions_for_review (1).pdf, file_redactions_for_review (2).pdf,
    or file_FINAL_redactions_for_review.pdf). The output filename is derived from
    the first input file: the name up to and including 'redactions_for_review' is
    taken (anything after that is dropped), then '_combined' is appended, e.g.
    file_redactions_for_review_combined.pdf.

    Args:
        file_list: List of file paths or Gradio FileData-like objects with .name.
        output_folder: Folder to write the combined PDF.

    Returns:
        List containing the path to the combined PDF for use as gr.File output.
        On validation error, raises ValueError (e.g. page count mismatch).
    """
    if not file_list:
        return []

    # Normalise to paths (Gradio may pass FileData with .name or dict with "name")
    paths = []
    for f in file_list:
        p = (
            getattr(f, "name", None)
            or (f.get("name") if isinstance(f, dict) else None)
            or f
        )
        if isinstance(p, str):
            paths.append(p)
    if not paths:
        return []

    output_base = _get_review_pdf_combined_output_base(paths[0])
    first_doc = pymupdf.open(paths[0])
    page_count = len(first_doc)

    for p in paths[1:]:
        other_doc = pymupdf.open(p)
        if len(other_doc) != page_count:
            other_doc.close()
            first_doc.close()
            raise ValueError(
                f"All files must have the same number of pages. "
                f"'{os.path.basename(paths[0])}' has {page_count} pages but "
                f"'{os.path.basename(p)}' has {len(other_doc)} pages."
            )
        # Copy redaction annotations from each page of other_doc into first_doc
        for page_num in range(page_count):
            src_page = other_doc[page_num]
            dst_page = first_doc[page_num]
            # Collect annots so we don't modify while iterating
            annots = list(src_page.annots())
            for annot in annots:
                if annot.type[0] != pymupdf.PDF_ANNOT_REDACT:
                    continue
                rect = annot.rect
                annot_colors = annot.colors or {}
                annot_info = annot.info or {}
                title = annot_info.get("title", "Redaction")
                content = annot_info.get("content", "")
                # Skip duplicate: same position and same label/text content
                if _dst_page_has_duplicate_redact(dst_page, rect, title, content):
                    continue
                stroke = annot_colors.get("stroke", (0, 0, 0))
                fill = annot_colors.get("fill", (0, 0, 0))
                new_annot = dst_page.add_redact_annot(rect)
                new_annot.set_colors(stroke=stroke, fill=fill, colors=fill)
                new_annot.set_name(title)
                new_annot.set_info(
                    info=title,
                    title=title,
                    subject=annot_info.get("subject", "Redaction"),
                    content=content,
                    creationDate=annot_info.get("creationDate", ""),
                )
                new_annot.update(opacity=0.5, cross_out=False)
        other_doc.close()

    out_path = os.path.join(output_folder, output_base + "_combined.pdf")
    first_doc.save(out_path, clean=True)
    first_doc.close()
    return [out_path]


def prepare_image_or_pdf_with_efficient_ocr(
    file_paths,
    text_extract_method,
    all_page_line_level_ocr_results_df_base,
    all_page_line_level_ocr_results_with_words_df_base,
    latest_file_completed_num,
    out_message,
    first_loop_state,
    number_of_pages,
    all_annotations_object,
    prepare_for_review,
    in_fully_redacted_list,
    output_folder,
    input_folder,
    efficient_ocr,
    prepare_images_bool_false,
    page_sizes,
    pymupdf_doc,
    page_min,
    page_max,
):
    """When EFFICIENT_OCR is enabled, skip loading all images; they are created later only for pages that need OCR."""
    prepare_images = (
        False
        if efficient_ocr
        else (
            prepare_images_bool_false if prepare_images_bool_false is not None else True
        )
    )
    return prepare_image_or_pdf(
        file_paths,
        text_extract_method,
        all_page_line_level_ocr_results_df_base,
        all_page_line_level_ocr_results_with_words_df_base,
        latest_file_completed_num,
        out_message,
        first_loop_state,
        number_of_pages,
        all_annotations_object,
        prepare_for_review,
        in_fully_redacted_list,
        output_folder,
        input_folder,
        prepare_images,
        page_sizes,
        pymupdf_doc,
        page_min,
        page_max,
    )


def prepare_image_or_pdf(
    file_paths: List[str],
    text_extract_method: str,
    all_line_level_ocr_results_df: pd.DataFrame = None,
    all_page_line_level_ocr_results_with_words_df: pd.DataFrame = None,
    latest_file_completed: int = 0,
    out_message: List[str] = list(),
    first_loop_state: bool = False,
    number_of_pages: int = 0,
    all_annotations_object: List = list(),
    prepare_for_review: bool = False,
    in_fully_redacted_list: List[int] = list(),
    output_folder: str = OUTPUT_FOLDER,
    input_folder: str = INPUT_FOLDER,
    prepare_images: bool = True,
    page_sizes: list[dict] = list(),
    pymupdf_doc: Document = list(),
    textract_output_found: bool = False,
    relevant_ocr_output_with_words_found: bool = False,
    page_min: int = 0,
    page_max: int = 0,
    progress: Progress = Progress(track_tqdm=True),
) -> tuple[List[str], List[str]]:
    """
    Prepare and process image or text PDF files for redaction.

    This function takes a list of file paths, processes each file based on the specified redaction method,
    and returns the output messages and processed file paths.

    Args:
        file_paths (List[str]): List of file paths to process.
        text_extract_method (str): The redaction method to use.
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
        pymupdf_doc(optional, Document): A pymupdf document object that indicates the existing PDF document object.
        textract_output_found (optional, bool): A boolean indicating whether Textract analysis output has already been found. Defaults to False.
        relevant_ocr_output_with_words_found (optional, bool): A boolean indicating whether local OCR analysis output has already been found. Defaults to False.
        page_min (optional, int): The minimum page number to process (0-indexed). If 0, uses the first page. Defaults to 0.
        page_max (optional, int): The maximum page number to process (0-indexed). If 0, uses the last page of the document. Defaults to 0.
        progress (optional, Progress): Progress tracker for the operation


    Returns:
        tuple[List[str], List[str]]: A tuple containing the output messages and processed file paths.
    """

    tic = time.perf_counter()
    json_from_csv = False
    original_cropboxes = list()  # Store original CropBox values
    converted_file_paths = list()
    image_file_paths = list()
    all_img_details = list()
    review_file_csv = pd.DataFrame()
    out_textract_path = ""
    combined_out_message = ""
    final_out_message = ""
    log_files_output_paths = list()

    if isinstance(in_fully_redacted_list, pd.DataFrame):
        if not in_fully_redacted_list.empty:
            in_fully_redacted_list = in_fully_redacted_list.iloc[:, 0].tolist()

    # If this is the first time around, set variables to 0/blank
    if first_loop_state is True:
        latest_file_completed = 0
        out_message = list()
        all_annotations_object = list()
    else:
        print("Now redacting file", str(latest_file_completed))

    # If combined out message or converted_file_paths are blank, change to a list so it can be appended to
    if isinstance(out_message, str):
        out_message = [out_message]

    if not file_paths:
        file_paths = list()

    if isinstance(file_paths, dict):
        file_paths = os.path.abspath(file_paths["name"])

    if isinstance(file_paths, str):
        file_path_number = 1
    else:
        file_path_number = len(file_paths)

    if file_path_number > MAX_SIMULTANEOUS_FILES:
        out_message = f"Number of files loaded is greater than {MAX_SIMULTANEOUS_FILES}. Please submit a smaller number of files."
        print(out_message)
        raise Exception(out_message)

    latest_file_completed = int(latest_file_completed)

    # If we have already redacted the last file, return the input out_message and file list to the relevant components
    if latest_file_completed >= file_path_number:
        print("Last file reached, returning files:", str(latest_file_completed))
        if isinstance(out_message, list):
            final_out_message = "\n".join(out_message)
        else:
            final_out_message = out_message

        return (
            final_out_message,
            converted_file_paths,
            image_file_paths,
            number_of_pages,
            number_of_pages,
            pymupdf_doc,
            all_annotations_object,
            review_file_csv,
            original_cropboxes,
            page_sizes,
            textract_output_found,
            all_img_details,
            all_line_level_ocr_results_df,
            relevant_ocr_output_with_words_found,
            all_page_line_level_ocr_results_with_words_df,
        )

    progress(0.1, desc="Preparing file")

    def _file_item_to_path(item):
        """Normalize Gradio file input (str, dict with 'name'/'path', or object with .name/.path) to path string."""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return item.get("name") or item.get("path") or ""
        return getattr(item, "name", None) or getattr(item, "path", None) or ""

    if isinstance(file_paths, str):
        file_paths_list = [file_paths]
        file_paths_loop = file_paths_list
    else:
        file_paths_list = [_file_item_to_path(f) for f in file_paths if f is not None]
        file_paths_list = [p for p in file_paths_list if p and str(p).strip()]
        file_paths_loop = sorted(
            file_paths_list,
            key=lambda x: (
                os.path.splitext(x)[1] != ".pdf",
                os.path.splitext(x)[1] != ".json",
            ),
        )

    # Loop through files to load in
    for file in file_paths_loop:
        converted_file_path = list()
        image_file_path = list()

        file_path = file if isinstance(file, str) else _file_item_to_path(file)
        file_path_without_ext = get_file_name_without_type(file_path)
        file_name_with_ext = os.path.basename(file_path)

        print("Loading file:", file_name_with_ext)

        if not file_path:
            out_message = "Please select at least one file."
            print(out_message)
            raise Warning(out_message)

        file_extension = os.path.splitext(file_path)[1].lower()

        progress(0.2, desc="Preparing file")

        # If a pdf, load as a pymupdf document
        if is_pdf(file_path):
            print(f"File {file_name_with_ext} is a PDF")
            pymupdf_doc = pymupdf.open(file_path)

            converted_file_path = file_path

            if prepare_images is True:
                (
                    image_file_paths,
                    image_sizes_width,
                    image_sizes_height,
                    all_img_details,
                ) = process_file_for_image_creation(
                    file_path,
                    prepare_for_review,
                    input_folder,
                    create_images=True,
                    page_min=page_min,
                    page_max=page_max,
                    progress=progress,
                )
            else:
                (
                    image_file_paths,
                    image_sizes_width,
                    image_sizes_height,
                    all_img_details,
                ) = process_file_for_image_creation(
                    file_path,
                    prepare_for_review,
                    input_folder,
                    create_images=False,
                    page_min=page_min,
                    page_max=page_max,
                    progress=progress,
                )

            page_sizes, original_cropboxes = create_page_size_objects(
                pymupdf_doc,
                image_sizes_width,
                image_sizes_height,
                image_file_paths,
                page_min,
                page_max,
            )

            # Create base version of the annotation object that doesn't have any annotations in it
            if (not all_annotations_object) & (prepare_for_review is True):
                all_annotations_object = list()

                for image_path in image_file_paths:
                    annotation = dict()
                    annotation["image"] = image_path
                    annotation["boxes"] = list()

                    all_annotations_object.append(annotation)

            # If we are loading redactions from the pdf, extract the redactions
            if LOAD_REDACTION_ANNOTATIONS_FROM_PDF and prepare_for_review is True:

                redactions_list, pymupdf_doc = extract_redactions(
                    pymupdf_doc, page_sizes
                )
                all_annotations_object = redactions_list

                review_file_csv = convert_annotation_json_to_review_df(
                    all_annotations_object
                )

        elif is_pdf_or_image(file_path):  # Alternatively, if it's an image
            print(f"File {file_name_with_ext} is an image")
            # Check if the file is an image type and the user selected text ocr option
            if (
                file_extension in [".jpg", ".jpeg", ".png"]
                and text_extract_method == SELECTABLE_TEXT_EXTRACT_OPTION
            ):
                text_extract_method = TESSERACT_TEXT_EXTRACT_OPTION

            # Convert image to a pymupdf document
            pymupdf_doc = pymupdf.open()  # Create a new empty document

            img = Image.open(file_path)  # Open the image file
            rect = pymupdf.Rect(
                0, 0, img.width, img.height
            )  # Create a rectangle for the image
            pymupdf_page = pymupdf_doc.new_page(
                width=img.width, height=img.height
            )  # Add a new page
            pymupdf_page.insert_image(
                rect, filename=file_path
            )  # Insert the image into the page
            pymupdf_page = pymupdf_doc.load_page(0)

            file_path_str = str(file_path)

            image_file_paths, image_sizes_width, image_sizes_height, all_img_details = (
                process_file_for_image_creation(
                    file_path_str,
                    prepare_for_review,
                    input_folder,
                    create_images=True,
                    progress=progress,
                )
            )

            # Create a page_sizes_object
            page_sizes, original_cropboxes = create_page_size_objects(
                pymupdf_doc, image_sizes_width, image_sizes_height, image_file_paths
            )

            # Create base version of the annotation object for review (same as PDF branch)
            if (not all_annotations_object) and (prepare_for_review is True):
                all_annotations_object = list()
                for image_path in image_file_paths:
                    annotation = dict()
                    annotation["image"] = image_path
                    annotation["boxes"] = list()
                    all_annotations_object.append(annotation)

            converted_file_path = output_folder + file_name_with_ext

            pymupdf_doc.save(converted_file_path, garbage=4, deflate=True, clean=True)

        # Loading in review files, ocr_outputs, or ocr_outputs_with_words
        elif file_extension in [".csv"]:
            if "_review_file" in file_path_without_ext:
                review_file_csv = read_file(file_path)
                all_annotations_object = convert_review_df_to_annotation_json(
                    review_file_csv, image_file_paths, page_sizes
                )
                json_from_csv = True
            elif "_ocr_output" in file_path_without_ext:
                all_line_level_ocr_results_df = read_file(file_path)

                if "line" not in all_line_level_ocr_results_df.columns:
                    all_line_level_ocr_results_df["line"] = ""

                json_from_csv = False
            elif "_ocr_results_with_words" in file_path_without_ext:
                all_page_line_level_ocr_results_with_words_df = read_file(file_path)
                json_from_csv = False

                # Convert word-level OCR results to line-level if line-level is empty
                if all_line_level_ocr_results_df is None or (
                    isinstance(all_line_level_ocr_results_df, pd.DataFrame)
                    and all_line_level_ocr_results_df.empty
                ):
                    all_line_level_ocr_results_df = (
                        word_level_ocr_df_to_line_level_ocr_df(
                            all_page_line_level_ocr_results_with_words_df
                        )
                    )
                    if "line" not in all_line_level_ocr_results_df.columns:
                        all_line_level_ocr_results_df["line"] = ""

        # If the file name ends with .json, check if we are loading for review. If yes, assume it is an annotations object, overwrite the current annotations object. If false, assume this is a Textract object, load in to Textract

        if (file_extension in [".json"]) | (json_from_csv is True):

            if (file_extension in [".json"]) & (prepare_for_review is True):
                if isinstance(file_path, str):
                    # Split the path into base directory and filename for security
                    file_path_obj = Path(file_path)
                    base_dir = file_path_obj.parent
                    filename = file_path_obj.name

                    json_content = secure_file_read(base_dir, filename)
                    all_annotations_object = json.loads(json_content)
                else:
                    # Assuming file_path is a NamedString or similar
                    all_annotations_object = json.loads(
                        file_path
                    )  # Use loads for string content

            # Save Textract file to folder
            elif (
                file_extension in [".json"]
            ) and "_textract" in file_path_without_ext:  # (prepare_for_review != True):
                print("Saving Textract output")
                # Copy it to the output folder so it can be used later.
                # If the path already ends with _textract.json (e.g. _sig_textract.json), preserve the basename;
                # otherwise append _textract.json. Use endswith instead of regex to avoid ReDoS (CodeQL py/polynomial-redos).
                if file_path.endswith("_textract.json"):
                    # File already has a textract suffix, preserve it
                    output_textract_json_file_name = file_path_without_ext + ".json"
                else:
                    # No textract suffix found, add default one
                    output_textract_json_file_name = (
                        file_path_without_ext + "_textract.json"
                    )

                out_textract_path = secure_join(
                    output_folder, output_textract_json_file_name
                )

                # Use shutil to copy the file directly
                shutil.copy2(file_path, out_textract_path)  # Preserves metadata
                textract_output_found = True
                continue

            elif (
                file_extension in [".json"]
            ) and "_ocr_results_with_words" in file_path_without_ext:  # (prepare_for_review != True):
                print("Saving local OCR output with words")
                # Copy it to the output folder so it can be used later.
                output_ocr_results_with_words_json_file_name = (
                    file_path_without_ext + ".json"
                )

                out_ocr_results_with_words_path = secure_join(
                    output_folder, output_ocr_results_with_words_json_file_name
                )

                # Use shutil to copy the file directly
                shutil.copy2(
                    file_path, out_ocr_results_with_words_path
                )  # Preserves metadata

                if prepare_for_review is True:
                    print("Converting local OCR output with words to csv")
                    page_sizes_df = pd.DataFrame(page_sizes)
                    (
                        all_page_line_level_ocr_results_with_words,
                        is_missing,
                        log_files_output_paths,
                    ) = load_and_convert_ocr_results_with_words_json(
                        out_ocr_results_with_words_path,
                        log_files_output_paths,
                        page_sizes_df,
                    )
                    all_page_line_level_ocr_results_with_words_df = (
                        word_level_ocr_output_to_dataframe(
                            all_page_line_level_ocr_results_with_words
                        )
                    )

                    # Use mediabox for division when loading text-extraction output (PDF-point coords)
                    coords_in_pdf_points = file_path.endswith(
                        "_ocr_results_with_words_local_text.json"
                    )
                    all_page_line_level_ocr_results_with_words_df = (
                        divide_coordinates_by_page_sizes(
                            all_page_line_level_ocr_results_with_words_df,
                            page_sizes_df,
                            xmin="word_x0",
                            xmax="word_x1",
                            ymin="word_y0",
                            ymax="word_y1",
                            coordinates_in_pdf_points=coords_in_pdf_points,
                        )
                    )
                    all_page_line_level_ocr_results_with_words_df = (
                        divide_coordinates_by_page_sizes(
                            all_page_line_level_ocr_results_with_words_df,
                            page_sizes_df,
                            xmin="line_x0",
                            xmax="line_x1",
                            ymin="line_y0",
                            ymax="line_y1",
                            coordinates_in_pdf_points=coords_in_pdf_points,
                        )
                    )

                if (
                    text_extract_method == SELECTABLE_TEXT_EXTRACT_OPTION
                    and file_path.endswith("_ocr_results_with_words_local_text.json")
                ):
                    relevant_ocr_output_with_words_found = True
                if (
                    text_extract_method == TESSERACT_TEXT_EXTRACT_OPTION
                    and file_path.endswith("_ocr_results_with_words_local_ocr.json")
                ):
                    relevant_ocr_output_with_words_found = True
                if (
                    text_extract_method == TEXTRACT_TEXT_EXTRACT_OPTION
                    and file_path.endswith("_ocr_results_with_words_textract.json")
                ):
                    relevant_ocr_output_with_words_found = True
                continue

            # If you have an annotations object from the above code
            if all_annotations_object:

                image_file_paths_pages = [
                    safe_extract_page_number_from_path(s)
                    for s in image_file_paths
                    if safe_extract_page_number_from_path(s) is not None
                ]
                image_file_paths_pages = [int(i) for i in image_file_paths_pages]

                # If PDF pages have been converted to image files, replace the current image paths in the json to this.
                if image_file_paths:
                    for i, image_file_path in enumerate(image_file_paths):

                        if i < len(all_annotations_object):
                            annotation = all_annotations_object[i]
                        else:
                            annotation = dict()
                            all_annotations_object.append(annotation)

                        try:
                            if not annotation:
                                annotation = {"image": "", "boxes": []}
                                annotation_page_number = (
                                    safe_extract_page_number_from_path(image_file_path)
                                )
                                if annotation_page_number is None:
                                    continue
                            else:
                                annotation_page_number = (
                                    safe_extract_page_number_from_path(
                                        annotation["image"]
                                    )
                                )
                                if annotation_page_number is None:
                                    continue
                        except Exception as e:
                            print("Extracting page number from image failed due to:", e)
                            annotation_page_number = 0

                        # Check if the annotation page number exists in the image file paths pages
                        if annotation_page_number in image_file_paths_pages:

                            # Set the correct image page directly since we know it's in the list
                            correct_image_page = annotation_page_number
                            annotation["image"] = image_file_paths[correct_image_page]
                        else:
                            print(
                                "Page", annotation_page_number, "image file not found."
                            )

                        all_annotations_object[i] = annotation

                # Write the response to a JSON file in output folder
                out_folder = output_folder + file_path_without_ext + ".json"
                # with open(out_folder, 'w') as json_file:
                #     json.dump(all_annotations_object, json_file, separators=(",", ":"))
                continue

        # If it's a zip, it could be extract from a Textract bulk API call. Check it's this, and load in json if found
        if file_extension in [".zip"]:

            # Assume it's a Textract response object. Copy it to the output folder so it can be used later.
            out_folder = secure_join(
                output_folder, file_path_without_ext + "_textract.json"
            )

            # Use shutil to copy the file directly
            # Open the ZIP file to check its contents
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                json_files = [
                    f for f in zip_ref.namelist() if f.lower().endswith(".json")
                ]

                if len(json_files) == 1:  # Ensure only one JSON file exists
                    json_filename = json_files[0]

                    # Extract the JSON file to the same directory as the ZIP file
                    extracted_path = secure_join(
                        os.path.dirname(file_path), json_filename
                    )
                    zip_ref.extract(json_filename, os.path.dirname(file_path))

                    # Move the extracted JSON to the intended output location
                    shutil.move(extracted_path, out_folder)

                    textract_output_found = True
                else:
                    print(
                        f"Skipping {file_path}: Expected 1 JSON file, found {len(json_files)}"
                    )

        converted_file_paths.append(converted_file_path)
        image_file_paths.extend(image_file_path)

        toc = time.perf_counter()
        out_time = f"File '{file_name_with_ext}' prepared in {toc - tic:0.1f} seconds."

        print(out_time)

        if not out_message:
            out_message = list()

        out_message.append(out_time)
        combined_out_message = "\n".join(out_message).strip() if out_message else ""

    if not page_sizes:
        number_of_pages = 1
    else:
        number_of_pages = len(page_sizes)

    if first_loop_state is True:
        print(f"Finished loading in {file_path_number} file(s)")
        gr.Info(f"Finished loading in {file_path_number} file(s)")

    return (
        combined_out_message,
        converted_file_paths,
        image_file_paths,
        number_of_pages,
        number_of_pages,
        pymupdf_doc,
        all_annotations_object,
        review_file_csv,
        original_cropboxes,
        page_sizes,
        textract_output_found,
        all_img_details,
        all_line_level_ocr_results_df,
        relevant_ocr_output_with_words_found,
        all_page_line_level_ocr_results_with_words_df,
    )


def load_and_convert_ocr_results_with_words_json(
    ocr_results_with_words_json_file_path: str,
    log_files_output_paths: str,
    page_sizes_df: pd.DataFrame,
):
    """
    Loads Textract JSON from a file, detects if conversion is needed, and converts if necessary.
    """

    if not os.path.exists(ocr_results_with_words_json_file_path):
        print("No existing OCR results file found.")
        return (
            [],
            True,
            log_files_output_paths,
        )  # Return empty dict and flag indicating missing file

    print("Found existing OCR results json results file.")

    # Track log files
    if ocr_results_with_words_json_file_path not in log_files_output_paths:
        log_files_output_paths.append(ocr_results_with_words_json_file_path)

    try:
        with open(
            ocr_results_with_words_json_file_path, "r", encoding="utf-8"
        ) as json_file:
            ocr_results_with_words_data = json.load(json_file)
    except json.JSONDecodeError:
        print("Error: Failed to parse OCR results JSON file. Returning empty data.")
        return [], True, log_files_output_paths  # Indicate failure

    # Check if conversion is needed
    if "page" and "results" in ocr_results_with_words_data[0]:
        print("JSON already in the correct format for app. No changes needed.")
        return (
            ocr_results_with_words_data,
            False,
            log_files_output_paths,
        )  # No conversion required

    else:
        print("Invalid OCR result JSON format: 'page' or 'results' key missing.")

        return (
            [],
            True,
            log_files_output_paths,
        )  # Return empty data if JSON is not recognized


def convert_text_pdf_to_img_pdf(
    in_file_path: str,
    out_text_file_path: List[str],
    image_dpi: float = image_dpi,
    output_folder: str = OUTPUT_FOLDER,
    input_folder: str = INPUT_FOLDER,
):
    file_path_without_ext = get_file_name_without_type(in_file_path)

    out_file_paths = out_text_file_path

    # Convert annotated text pdf back to image to give genuine redactions
    pdf_text_image_paths, image_sizes_width, image_sizes_height, all_img_details = (
        process_file_for_image_creation(out_file_paths[0], input_folder=input_folder)
    )
    out_text_image_file_path = (
        output_folder + file_path_without_ext + "_text_redacted_as_img.pdf"
    )
    pdf_text_image_paths[0].save(
        out_text_image_file_path,
        "PDF",
        resolution=image_dpi,
        save_all=True,
        append_images=pdf_text_image_paths[1:],
    )

    out_file_paths = [out_text_image_file_path]

    out_message = "PDF " + file_path_without_ext + " converted to image-based file."
    print(out_message)

    return out_message, out_file_paths


def save_pdf_with_or_without_compression(
    pymupdf_doc: object,
    out_redacted_pdf_file_path,
    COMPRESS_REDACTED_PDF: bool = COMPRESS_REDACTED_PDF,
):
    """
    Save a pymupdf document with basic cleaning or with full compression options. Can be useful for low memory systems to do minimal cleaning to avoid crashing with large PDFs.
    """
    if COMPRESS_REDACTED_PDF is True:
        try:
            pymupdf_doc.save(
                out_redacted_pdf_file_path, garbage=4, deflate=True, clean=True
            )
        except Exception as e:
            print(
                f"Error saving PDF with compression: {e}, trying again without compression"
            )
            pymupdf_doc.save(out_redacted_pdf_file_path, clean=True)
    else:
        try:
            pymupdf_doc.save(out_redacted_pdf_file_path, garbage=1, clean=True)
        except Exception as e:
            print(f"Error saving PDF without compression: {e}, trying again")
            pymupdf_doc.save(out_redacted_pdf_file_path, clean=True)


def join_values_within_threshold(df1: pd.DataFrame, df2: pd.DataFrame):
    # Threshold for matching
    threshold = 5

    # Perform a cross join
    df1["key"] = 1
    df2["key"] = 1
    merged = pd.merge(df1, df2, on="key").drop(columns=["key"])

    # Apply conditions for all columns
    conditions = (
        (abs(merged["xmin_x"] - merged["xmin_y"]) <= threshold)
        & (abs(merged["xmax_x"] - merged["xmax_y"]) <= threshold)
        & (abs(merged["ymin_x"] - merged["ymin_y"]) <= threshold)
        & (abs(merged["ymax_x"] - merged["ymax_y"]) <= threshold)
    )

    # Filter rows that satisfy all conditions
    filtered = merged[conditions]

    # Drop duplicates if needed (e.g., keep only the first match for each row in df1)
    result = filtered.drop_duplicates(subset=["xmin_x", "xmax_x", "ymin_x", "ymax_x"])

    # Merge back into the original DataFrame (if necessary)
    final_df = pd.merge(
        df1,
        result,
        left_on=["xmin", "xmax", "ymin", "ymax"],
        right_on=["xmin_x", "xmax_x", "ymin_x", "ymax_x"],
        how="left",
    )

    # Clean up extra columns
    final_df = final_df.drop(columns=["key"])


def _pick_one_item_per_image(image: str, items: List[dict]) -> dict:
    """Choose one item per image (prefer non-empty boxes); safe to run in a thread."""
    non_empty_boxes = [item for item in items if item.get("boxes")]
    return non_empty_boxes[0] if non_empty_boxes else items[0]


def remove_duplicate_images_with_blank_boxes(data: List[dict]) -> List[dict]:
    """
    Remove items from the annotator object where the same page exists twice.
    """
    image_groups = defaultdict(list)
    for item in data:
        image_groups[item["image"]].append(item)

    if not image_groups:
        return []

    groups_list = list(image_groups.items())
    max_workers = min(MAX_WORKERS, len(groups_list))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        result = list(
            executor.map(
                lambda pair: _pick_one_item_per_image(pair[0], pair[1]),
                groups_list,
            )
        )
    return result


def divide_coordinates_by_page_sizes_pl(
    df: pl.DataFrame,
    page_sizes_df: pd.DataFrame,
    xmin: str = "xmin",
    xmax: str = "xmax",
    ymin: str = "ymin",
    ymax: str = "ymax",
    coordinates_in_pdf_points: bool = False,
    pages_in_pdf_points: Optional[Set[int]] = None,
) -> pl.DataFrame:
    """
    Polars-only coordinate division: absolute coords (>1) to relative (<=1).
    Expects df to have numeric coord columns. Returns pl.DataFrame.
    """
    coord_cols = [xmin, xmax, ymin, ymax]
    for col in coord_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

    # Clip to 0 and round
    df = df.with_columns(
        [
            pl.col(c).clip(0, float("inf")).round(6)
            for c in coord_cols
            if c in df.columns
        ]
    )

    # Identify absolute coordinates (any coord > 1 and not null)
    is_absolute = (
        (pl.col(xmin) > 1) & pl.col(xmin).is_not_nan()
        | (pl.col(xmax) > 1) & pl.col(xmax).is_not_nan()
        | (pl.col(ymin) > 1) & pl.col(ymin).is_not_nan()
        | (pl.col(ymax) > 1) & pl.col(ymax).is_not_nan()
    )
    df_rel = df.filter(~is_absolute)
    df_abs = df.filter(is_absolute)

    if not df_abs.is_empty() and not page_sizes_df.empty:
        merge_cols = [
            "page",
            "image_width",
            "image_height",
            "mediabox_width",
            "mediabox_height",
        ]
        available = [c for c in merge_cols if c in page_sizes_df.columns]
        if "page" in available:
            ps = pl.from_pandas(page_sizes_df[available].copy())
            for c in [
                "page",
                "image_width",
                "image_height",
                "mediabox_width",
                "mediabox_height",
            ]:
                if c in ps.columns:
                    # Cast page to Int64 so join key matches df_abs; cast sizes to Float64
                    dtype = pl.Int64 if c == "page" else pl.Float64
                    ps = ps.with_columns(pl.col(c).cast(dtype, strict=False))
            df_abs = df_abs.join(ps, on="page", how="left")

        if "mediabox_width" in df_abs.columns and "mediabox_height" in df_abs.columns:
            if coordinates_in_pdf_points:
                df_abs = df_abs.with_columns(
                    pl.col("mediabox_width").alias("image_width"),
                    pl.col("mediabox_height").alias("image_height"),
                )
            elif pages_in_pdf_points is not None:
                # Normalize to int set so 1-based page matches (e.g. 1.0 -> 1)
                _pdf_pts = {int(p) for p in pages_in_pdf_points}
                use_mediabox = (
                    pl.col("page").cast(pl.Int64, strict=False).is_in(list(_pdf_pts))
                )
                img_w = pl.col("mediabox_width")
                img_h = pl.col("mediabox_height")
                if "image_width" in df_abs.columns:
                    img_w = pl.col("image_width").fill_null(pl.col("mediabox_width"))
                if "image_height" in df_abs.columns:
                    img_h = pl.col("image_height").fill_null(pl.col("mediabox_height"))
                # For pages_in_pdf_points always use mediabox so text-path coords (PDF points) divide correctly
                df_abs = df_abs.with_columns(
                    pl.when(use_mediabox)
                    .then(pl.col("mediabox_width"))
                    .otherwise(img_w)
                    .alias("image_width"),
                    pl.when(use_mediabox)
                    .then(pl.col("mediabox_height"))
                    .otherwise(img_h)
                    .alias("image_height"),
                )
                # If join missed (nulls), fall back to mediabox for those pages so we don't divide by image pixels
                df_abs = df_abs.with_columns(
                    pl.when(use_mediabox & pl.col("image_width").is_null())
                    .then(pl.col("mediabox_width"))
                    .otherwise(pl.col("image_width"))
                    .alias("image_width"),
                    pl.when(use_mediabox & pl.col("image_height").is_null())
                    .then(pl.col("mediabox_height"))
                    .otherwise(pl.col("image_height"))
                    .alias("image_height"),
                )
            elif "image_width" not in df_abs.columns:
                df_abs = df_abs.with_columns(
                    pl.col("mediabox_width").alias("image_width"),
                    pl.col("mediabox_height").alias("image_height"),
                )
            else:
                df_abs = df_abs.with_columns(
                    pl.col("image_width")
                    .fill_null(pl.col("mediabox_width"))
                    .alias("image_width"),
                    pl.col("image_height")
                    .fill_null(pl.col("mediabox_height"))
                    .alias("image_height"),
                )

        if "image_width" in df_abs.columns and "image_height" in df_abs.columns:
            df_abs = df_abs.with_columns(
                (pl.col(xmin) / pl.col("image_width")).round(6).alias(xmin),
                (pl.col(xmax) / pl.col("image_width")).round(6).alias(xmax),
                (pl.col(ymin) / pl.col("image_height")).round(6).alias(ymin),
                (pl.col(ymax) / pl.col("image_height")).round(6).alias(ymax),
            )
            df_abs = df_abs.with_columns(
                [
                    pl.when(pl.col(c).is_in([float("inf"), float("-inf")]))
                    .then(pl.lit(None).cast(pl.Float64))
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in coord_cols
                ]
            )
        else:
            print(
                "Skipping coordinate division due to missing or non-numeric dimension columns."
            )

    if df_rel.is_empty() and df_abs.is_empty():
        print(
            "Warning: Both relative and absolute splits resulted in empty DataFrames."
        )
        return df_rel
    # Drop dimension columns from df_abs so concat matches df_rel schema (Polars requires same width)
    for c in ["image_width", "image_height", "mediabox_width", "mediabox_height"]:
        if c in df_abs.columns:
            df_abs = df_abs.drop(c)
    out = pl.concat([df_rel, df_abs])

    if not out.is_empty():
        out = out.sort(["page", ymin, xmin], nulls_last=True)

        # Clamp to [0,1] while preserving box dimensions.
        # Cap ymax at 1 - 1e-6 so no box spans the full bottom (avoids single-char words with ymax=1).
        _ymax_cap = 1.0 - 1e-6
        out = out.with_columns(
            pl.col(ymin).alias("_ymin_orig"),
            pl.col(ymax).alias("_ymax_orig"),
            pl.col(xmin).alias("_xmin_orig"),
            pl.col(xmax).alias("_xmax_orig"),
        )
        out = out.with_columns(
            pl.col(ymin).clip(0, float("inf")).alias(ymin),
            pl.col(xmin).clip(0, float("inf")).alias(xmin),
            pl.col(xmax).clip(float("-inf"), 1).alias(xmax),
            pl.col(ymax).clip(float("-inf"), _ymax_cap).alias(ymax),
        )
        # Preserve height/width when clamping
        out = out.with_columns(
            pl.when(pl.col("_ymax_orig") > 1)
            .then(
                (pl.col(ymin) + (pl.col("_ymax_orig") - pl.col("_ymin_orig"))).clip(
                    float("-inf"), _ymax_cap
                )
            )
            .otherwise(pl.col(ymax))
            .alias(ymax),
            pl.when(pl.col("_xmax_orig") > 1)
            .then(
                (pl.col(xmin) + (pl.col("_xmax_orig") - pl.col("_xmin_orig"))).clip(
                    float("-inf"), 1
                )
            )
            .otherwise(pl.col(xmax))
            .alias(xmax),
        )
        out = out.with_columns(
            pl.when(pl.col("_ymin_orig") < 0)
            .then(
                (pl.col(ymax) - (pl.col("_ymax_orig") - pl.col("_ymin_orig"))).clip(
                    0, float("inf")
                )
            )
            .otherwise(pl.col(ymin))
            .alias(ymin),
            pl.when(pl.col("_xmin_orig") < 0)
            .then(
                (pl.col(xmax) - (pl.col("_xmax_orig") - pl.col("_xmin_orig"))).clip(
                    0, float("inf")
                )
            )
            .otherwise(pl.col(xmin))
            .alias(xmin),
        )
        out = out.drop(["_ymin_orig", "_ymax_orig", "_xmin_orig", "_xmax_orig"])
        out = out.with_columns(
            [pl.col(c).round(6) for c in coord_cols if c in out.columns]
        )

    return out


def divide_coordinates_by_page_sizes(
    review_file_df: pd.DataFrame,
    page_sizes_df: pd.DataFrame,
    xmin="xmin",
    xmax="xmax",
    ymin="ymin",
    ymax="ymax",
    coordinates_in_pdf_points: bool = False,
    pages_in_pdf_points: Optional[Set[int]] = None,
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
        coordinates_in_pdf_points: If True, coordinates are in PDF space (points);
            use mediabox_width/mediabox_height for division regardless of
            image_width/image_height (e.g. when called from redact_text_pdf).
        pages_in_pdf_points: If set, page numbers (1-based) whose coordinates are in PDF
            points; all other pages use image dimensions. Used when EFFICIENT_OCR mixes
            text-extracted pages (PDF points) and OCR pages (image pixels). Ignored if
            coordinates_in_pdf_points is True for the whole dataframe.

    Returns:
        DataFrame with coordinates converted to relative system, sorted.
    """
    if review_file_df.empty or xmin not in review_file_df.columns:
        return review_file_df

    coord_cols = [xmin, xmax, ymin, ymax]
    cols_to_convert = coord_cols + ["page"]
    for col in cols_to_convert:
        if col not in review_file_df.columns:
            if col == "page" or col in coord_cols:
                print(
                    f"Warning: Required column '{col}' not found in review_file_df. Returning original DataFrame."
                )
                return review_file_df

    temp_pd = review_file_df.copy()
    for col in cols_to_convert:
        if col in temp_pd.columns:
            temp_pd[col] = pd.to_numeric(temp_pd[col], errors="coerce")
    for col in temp_pd.columns:
        if col not in cols_to_convert and temp_pd[col].dtype == object:
            temp_pd[col] = temp_pd[col].astype(str)
    df = pl.from_pandas(temp_pd)
    out = divide_coordinates_by_page_sizes_pl(
        df,
        page_sizes_df,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        coordinates_in_pdf_points=coordinates_in_pdf_points,
        pages_in_pdf_points=pages_in_pdf_points,
    )
    result = out.to_pandas()
    if "page" in result.columns and not result.empty:
        result["page"] = pd.to_numeric(result["page"], errors="coerce")
        result["page"] = result["page"].astype("Int64")
    for c in coord_cols:
        if c in result.columns:
            result[c] = result[c].astype(float)
    return result


def multiply_coordinates_by_page_sizes(
    review_file_df: pd.DataFrame,
    page_sizes_df: pd.DataFrame,
    xmin="xmin",
    xmax="xmax",
    ymin="ymin",
    ymax="ymax",
):
    """
    Optimized function to convert relative coordinates to absolute based on page sizes.

    Separates relative (<=1) and absolute (>1) coordinates, merges page sizes
    for relative coordinates, calculates absolute pixel values, and recombines.
    Implemented with Polars for performance; returns pandas DataFrame.
    """
    if review_file_df.empty or xmin not in review_file_df.columns:
        return review_file_df  # Return early if empty or key column missing

    coord_cols = [xmin, xmax, ymin, ymax]
    df = pl.from_pandas(review_file_df)

    # Cast coordinates and page to numeric (single with_columns for less overhead)
    cast_cols = [c for c in coord_cols + ["page"] if c in df.columns]
    if cast_cols:
        df = df.with_columns(
            [pl.col(c).cast(pl.Float64, strict=False) for c in cast_cols]
        )

    # Identify relative coordinates (all <= 1 and not null)
    is_relative = (
        pl.col(xmin).le(1)
        & pl.col(xmin).is_not_nan()
        & pl.col(xmax).le(1)
        & pl.col(xmax).is_not_nan()
        & pl.col(ymin).le(1)
        & pl.col(ymin).is_not_nan()
        & pl.col(ymax).le(1)
        & pl.col(ymax).is_not_nan()
    )
    df_abs = df.filter(~is_relative)
    df_rel = df.filter(is_relative)

    if df_rel.is_empty():
        if not df_abs.is_empty() and {"page", xmin, ymin}.issubset(df_abs.columns):
            df_abs = df_abs.sort(["page", xmin, ymin], nulls_last=True)
        result_early = df_abs.to_pandas()
        for c in coord_cols:
            if c in result_early.columns:
                result_early[c] = result_early[c].astype(float)
        return result_early

    # Join page sizes for relative rows
    if (
        not page_sizes_df.empty
        and "image_width" in page_sizes_df.columns
        and "image_height" in page_sizes_df.columns
    ):
        ps = pl.from_pandas(
            page_sizes_df[["page", "image_width", "image_height"]].copy()
        )
        ps = ps.with_columns(
            pl.col("page").cast(pl.Float64, strict=False),
            pl.col("image_width").cast(pl.Float64, strict=False),
            pl.col("image_height").cast(pl.Float64, strict=False),
        )
        df_rel = df_rel.join(ps, on="page", how="left")

    # Multiply coordinates where dimensions exist
    has_size = pl.col("image_width").is_not_nan() & pl.col("image_height").is_not_nan()
    df_rel = df_rel.with_columns(
        [
            pl.when(has_size)
            .then((pl.col(xmin) * pl.col("image_width")).round(6))
            .otherwise(pl.col(xmin))
            .alias(xmin),
            pl.when(has_size)
            .then((pl.col(xmax) * pl.col("image_width")).round(6))
            .otherwise(pl.col(xmax))
            .alias(xmax),
            pl.when(has_size)
            .then((pl.col(ymin) * pl.col("image_height")).round(6))
            .otherwise(pl.col(ymin))
            .alias(ymin),
            pl.when(has_size)
            .then((pl.col(ymax) * pl.col("image_height")).round(6))
            .otherwise(pl.col(ymax))
            .alias(ymax),
        ]
    )
    drop_cols = [c for c in ["image_width", "image_height"] if c in df_rel.columns]
    if drop_cols:
        df_rel = df_rel.drop(drop_cols)

    out = pl.concat([df_abs, df_rel])
    out = out.sort(["page", xmin, ymin], nulls_last=True)
    out = out.with_columns(
        [
            pl.col(c).clip(0, float("inf")).round(6)
            for c in coord_cols
            if c in out.columns
        ]
    )
    result = out.to_pandas()
    for c in coord_cols:
        if c in result.columns:
            result[c] = result[c].astype(float)
    return result


def do_proximity_match_by_page_for_text(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Match text from one dataframe to another based on proximity matching of coordinates page by page.
    """

    if "text" not in df2.columns:
        df2["text"] = ""
    if "text" not in df1.columns:
        df1["text"] = ""

    # Create a unique key based on coordinates and label for exact merge
    merge_keys = ["xmin", "ymin", "xmax", "ymax", "label", "page"]
    df1["key"] = df1[merge_keys].astype(str).agg("_".join, axis=1)
    df2["key"] = df2[merge_keys].astype(str).agg("_".join, axis=1)

    # Attempt exact merge first
    merged_df = df1.merge(
        df2[["key", "text"]], on="key", how="left", suffixes=("", "_duplicate")
    )

    # If a match is found, keep that text; otherwise, keep the original df1 text
    merged_df["text"] = np.where(
        merged_df["text"].isna() | (merged_df["text"] == ""),
        merged_df.pop("text_duplicate"),
        merged_df["text"],
    )

    # Define tolerance for proximity matching
    tolerance = 0.02

    # Precompute KDTree for each page in df2
    page_trees = dict()
    for page in df2["page"].unique():
        df2_page = df2[df2["page"] == page]
        coords = df2_page[["xmin", "ymin", "xmax", "ymax"]].values
        if np.all(np.isfinite(coords)) and len(coords) > 0:
            page_trees[page] = (cKDTree(coords), df2_page)

    # Perform proximity matching
    for i, row in df1.iterrows():
        page_number = row["page"]

        if page_number in page_trees:
            tree, df2_page = page_trees[page_number]

            # Query KDTree for nearest neighbor
            dist, idx = tree.query(
                [row[["xmin", "ymin", "xmax", "ymax"]].values],
                distance_upper_bound=tolerance,
            )

            if dist[0] < tolerance and idx[0] < len(df2_page):
                merged_df.at[i, "text"] = df2_page.iloc[idx[0]]["text"]

    # Drop the temporary key column
    merged_df.drop(columns=["key"], inplace=True)

    return merged_df


def do_proximity_match_all_pages_for_text(
    df1: pd.DataFrame, df2: pd.DataFrame, threshold: float = 0.03
):
    """
    Match text from one dataframe to another based on proximity matching of coordinates across all pages.
    """

    if "text" not in df2.columns:
        df2["text"] = ""
    if "text" not in df1.columns:
        df1["text"] = ""

    for col in ["xmin", "ymin", "xmax", "ymax"]:
        df1[col] = pd.to_numeric(df1[col], errors="coerce")

    for col in ["xmin", "ymin", "xmax", "ymax"]:
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

    # Create a unique key based on coordinates and label for exact merge
    merge_keys = ["xmin", "ymin", "xmax", "ymax", "label", "page"]
    df1["key"] = df1[merge_keys].astype(str).agg("_".join, axis=1)
    df2["key"] = df2[merge_keys].astype(str).agg("_".join, axis=1)

    # Attempt exact merge first, renaming df2['text'] to avoid suffixes
    merged_df = df1.merge(
        df2[["key", "text"]], on="key", how="left", suffixes=("", "_duplicate")
    )

    # If a match is found, keep that text; otherwise, keep the original df1 text
    merged_df["text"] = np.where(
        merged_df["text"].isna() | (merged_df["text"] == ""),
        merged_df.pop("text_duplicate"),
        merged_df["text"],
    )

    # Handle missing matches using a proximity-based approach
    # Convert coordinates to numpy arrays for KDTree lookup

    query_coords = np.array(df1[["xmin", "ymin", "xmax", "ymax"]].values, dtype=float)

    # Check for NaN or infinite values in query_coords and filter them out
    finite_mask = np.isfinite(query_coords).all(axis=1)
    if not finite_mask.all():
        # print("Warning: query_coords contains non-finite values. Filtering out non-finite entries.")
        query_coords = query_coords[
            finite_mask
        ]  # Filter out rows with NaN or infinite values
    else:
        pass

    # Proceed only if query_coords is not empty
    if query_coords.size > 0:
        # Ensure df2 is filtered for finite values before creating the KDTree
        finite_mask_df2 = np.isfinite(df2[["xmin", "ymin", "xmax", "ymax"]].values).all(
            axis=1
        )
        df2_finite = df2[finite_mask_df2]

        # Create the KDTree with the filtered data
        tree = cKDTree(df2_finite[["xmin", "ymin", "xmax", "ymax"]].values)

        # Find nearest neighbors within a reasonable tolerance (e.g., 1% of page)
        tolerance = threshold
        distances, indices = tree.query(query_coords, distance_upper_bound=tolerance)

        # Assign text values where matches are found
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist < tolerance and idx < len(df2_finite):
                merged_df.at[i, "text"] = df2_finite.iloc[idx]["text"]

    # Drop the temporary key column
    merged_df.drop(columns=["key"], inplace=True)

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
    """
    Convert annotation list to DataFrame using Polars for performance.
    Returns a pandas DataFrame with columns image, page, label, color, xmin, xmax, ymin, ymax, text, id.
    """
    if not all_annotations:
        print("No annotations found, returning empty dataframe")
        return pd.DataFrame(
            columns=[
                "image",
                "page",
                "label",
                "color",
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "text",
                "id",
            ]
        )

    records = []
    for anno in all_annotations:
        image = anno.get("image")
        page_from_image = _extract_page_number(image)
        boxes = anno.get("boxes")
        if not isinstance(boxes, list):
            boxes = [boxes] if isinstance(boxes, dict) else []
        # Do not add a placeholder box when boxes is empty; that created blank annotations
        # in review_file_state when changing page or saving.
        for box in boxes:
            if isinstance(box, dict):
                # Skip blank/zero-area boxes (e.g. from image_annotator with 0,0,0,0 or None).
                def _num(v):
                    if v is None:
                        return None
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None

                xmin, ymin, xmax, ymax = (
                    _num(box.get("xmin")),
                    _num(box.get("ymin")),
                    _num(box.get("xmax")),
                    _num(box.get("ymax")),
                )
                if xmin is None and ymin is None and xmax is None and ymax is None:
                    continue
                if (xmin or 0) == (xmax or 0) and (ymin or 0) == (ymax or 0):
                    continue
                if (xmin or 0) >= (xmax or 0) or (ymin or 0) >= (ymax or 0):
                    continue

                # Use per-box page when present (e.g. text-path with empty image so all don't become page 1).
                # Reject 0 or negative (UI/state use 1-based pages); fall back to page_from_image.
                box_page = box.get("page")
                if box_page is not None:
                    try:
                        p = int(float(box_page))
                        page = p if p >= 1 else page_from_image
                    except (TypeError, ValueError):
                        page = page_from_image
                else:
                    page = page_from_image
                row = {"image": image, "page": page}
                for k, v in box.items():
                    if k != "page" and k != "image":
                        # Normalise colour to list so Polars gets a consistent schema
                        # (some boxes have color as list [r,g,b], tuple, or string)
                        if k == "color" and v is not None:
                            if isinstance(v, (list, tuple)) and len(v) >= 3:
                                v = [int(float(x)) for x in v[:3]]
                            elif isinstance(v, str):
                                s = v.strip("()").replace(" ", "")
                                # e.g. "(128,128,128)" or "128,128,128"
                                parts = s.split(",")
                                if len(parts) >= 3:
                                    v = [int(float(p)) for p in parts[:3]]
                                elif s.startswith("#") and len(s) in (4, 7):
                                    # Hex #rgb or #rrggbb (from gradio_image_annotation label_colors)
                                    hex_s = s[1:]
                                    if len(hex_s) == 3:
                                        v = [
                                            int(hex_s[i : i + 1] * 2, 16)
                                            for i in (0, 1, 2)
                                        ]
                                    else:
                                        v = [
                                            int(hex_s[i : i + 2], 16) for i in (0, 2, 4)
                                        ]
                                else:
                                    v = [0, 0, 0]
                            else:
                                v = [0, 0, 0]
                        elif k == "color" and v is None:
                            v = [0, 0, 0]
                        if k == "color":
                            # Store as string "(r, g, b)" so column survives Polars/pandas
                            # round-trip (list columns can be lost or corrupted)
                            v = (
                                f"({int(v[0])}, {int(v[1])}, {int(v[2])})"
                                if isinstance(v, (list, tuple)) and len(v) >= 3
                                else "(0, 0, 0)"
                            )
                        row[k] = v
                if "color" not in row:
                    row["color"] = "(0, 0, 0)"
                records.append(row)

    if not records:
        return pd.DataFrame(
            columns=[
                "image",
                "page",
                "label",
                "color",
                "xmin",
                "xmax",
                "ymin",
                "ymax",
                "text",
                "id",
            ]
        )

    df = pl.from_dicts(records)
    essential_box_cols = ["xmin", "xmax", "ymin", "ymax", "text", "id", "label"]
    for col in essential_box_cols:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))
    # Drop rows where all of the essential box fields are null (matches pandas dropna(..., how="all"))
    null_mask = pl.all_horizontal(
        [pl.col(c).is_null() for c in essential_box_cols if c in df.columns]
    )
    df = df.filter(~null_mask)
    base_cols = ["image"]
    extra_box_cols = sorted(
        [c for c in df.columns if c not in base_cols and c not in essential_box_cols]
    )
    final_col_order = base_cols + essential_box_cols + extra_box_cols
    final_col_order = [c for c in final_col_order if c in df.columns]
    df = df.select(final_col_order)
    return df.to_pandas()


def create_annotation_dicts_from_annotation_df(
    all_image_annotations_df: pd.DataFrame, page_sizes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convert annotation DataFrame back to list of dicts using dictionary lookup.
    Ensures all images from page_sizes are present without duplicates.
    """
    # 1. Create a dictionary keyed by image path for efficient lookup & update
    # Initialize with all images from page_sizes. Use .get for safety.
    image_dict: Dict[str, Dict[str, Any]] = dict()
    for item in page_sizes:
        image_path = item.get("image_path")
        if image_path:  # Only process if image_path exists and is not None/empty
            image_dict[image_path] = {"image": image_path, "boxes": []}

    # Check if the DataFrame is empty or lacks necessary columns
    if (
        all_image_annotations_df.empty
        or "image" not in all_image_annotations_df.columns
    ):
        # print("Warning: Annotation DataFrame is empty or missing 'image' column.")
        return list(image_dict.values())  # Return based on page_sizes only

    # 2. Define columns to extract for boxes and check availability
    # Make sure these columns actually exist in the DataFrame
    box_cols = ["xmin", "ymin", "xmax", "ymax", "color", "label", "text", "id"]
    available_cols = [
        col for col in box_cols if col in all_image_annotations_df.columns
    ]

    if "text" in all_image_annotations_df.columns:
        all_image_annotations_df["text"] = all_image_annotations_df["text"].fillna("")
        # all_image_annotations_df.loc[all_image_annotations_df['text'].isnull(), 'text'] = ''

    if not available_cols:
        print(
            f"Warning: None of the expected box columns ({box_cols}) found in DataFrame."
        )
        return list(image_dict.values())  # Return based on page_sizes only

    # 3. Group the DataFrame by image and update the dictionary
    coord_cols = ["xmin", "ymin", "xmax", "ymax"]
    valid_box_df = all_image_annotations_df.dropna(
        subset=[col for col in coord_cols if col in available_cols]
    ).copy()

    if valid_box_df.empty:
        print(
            "Warning: No valid annotation rows found in DataFrame after dropping NA coordinates."
        )
        return list(image_dict.values())

    # Ensure every image path in the dataframe has an entry (e.g. EFFICIENT_OCR text-path
    # pages may use a different path in annotations than in page_sizes, so boxes would be dropped).
    for image_path in valid_box_df["image"].unique():
        if image_path and image_path not in image_dict:
            image_dict[image_path] = {"image": image_path, "boxes": []}

    # Build list of (image_path, group) for all images in the dataframe
    group_items = [
        (image_path, group)
        for image_path, group in valid_box_df.groupby(
            "image", observed=True, sort=False
        )
    ]

    if group_items:
        max_workers = min(MAX_WORKERS, len(group_items))

        def _boxes_for_group(item):
            _image_path, _group = item
            boxes = _group[available_cols].to_dict(orient="records")
            return (_image_path, boxes)

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for image_path, boxes in executor.map(_boxes_for_group, group_items):
                    image_dict[image_path]["boxes"] = boxes
        except KeyError:
            print("Error: Issue grouping DataFrame by 'image'.")
            return list(image_dict.values())

    return list(image_dict.values())


def convert_annotation_json_to_review_df(
    all_annotations: List[dict],
    redaction_decision_output: pd.DataFrame = pd.DataFrame(),
    page_sizes: List[dict] = list(),
    do_proximity_match: bool = True,
    prebuilt_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Convert the annotation json data to a dataframe format.
    Add on any text from the initial review_file dataframe by joining based on 'id' if available
    in both sources, otherwise falling back to joining on pages/co-ordinates (if option selected).

    Refactored for improved efficiency, prioritizing ID-based join and conditionally applying
    coordinate division and proximity matching.

    When prebuilt_df is provided (e.g. from chunked parallel build), it is used as the initial
    DataFrame and the annotation-to-DataFrame conversion is skipped.
    """

    # 1. Convert annotations to DataFrame (or use prebuilt from chunked parallel build)
    if prebuilt_df is not None:
        review_file_df = prebuilt_df.copy()
    else:
        review_file_df = convert_annotation_data_to_dataframe(
            all_annotations if all_annotations else []
        )

    # Only keep rows in review_df where there are coordinates (assuming xmin is representative)
    # Use .notna() for robustness with potential None or NaN values
    review_file_df.dropna(
        subset=["xmin", "ymin", "xmax", "ymax"], how="any", inplace=True
    )

    # Drop blank/zero-area annotations (e.g. image_annotator sometimes sends 0,0,0,0 boxes)
    if not review_file_df.empty and all(
        c in review_file_df.columns for c in ["xmin", "ymin", "xmax", "ymax"]
    ):
        xmin, ymin, xmax, ymax = (
            pd.to_numeric(review_file_df["xmin"], errors="coerce"),
            pd.to_numeric(review_file_df["ymin"], errors="coerce"),
            pd.to_numeric(review_file_df["xmax"], errors="coerce"),
            pd.to_numeric(review_file_df["ymax"], errors="coerce"),
        )
        zero_area = (xmin >= xmax) | (ymin >= ymax)
        review_file_df = review_file_df.loc[~zero_area]

    # Exit early if the initial conversion results in an empty DataFrame
    if review_file_df.empty:
        # Define standard columns for an empty return DataFrame
        # Ensure 'id' is included if it was potentially expected based on input structure
        # We don't know the columns from convert_annotation_data_to_dataframe without seeing it,
        # but let's assume a standard set and add 'id' if it appeared.
        standard_cols = [
            "image",
            "page",
            "label",
            "color",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "text",
        ]
        if "id" in review_file_df.columns:
            standard_cols.append("id")
        return pd.DataFrame(columns=standard_cols)

    # Ensure 'id' column exists for logic flow, even if empty
    if "id" not in review_file_df.columns:
        review_file_df["id"] = ""
    # Do the same for redaction_decision_output if it's not empty
    if (
        not redaction_decision_output.empty
        and "id" not in redaction_decision_output.columns
    ):
        redaction_decision_output["id"] = ""

    # 2. Process page sizes if provided - needed potentially for coordinate division later
    # Process this once upfront if the data is available
    page_sizes_df = pd.DataFrame()  # Initialize as empty
    if page_sizes:
        page_sizes_df = pd.DataFrame(page_sizes)
        if not page_sizes_df.empty:
            # Safely convert page column to numeric and then int
            page_sizes_df["page"] = pd.to_numeric(
                page_sizes_df["page"], errors="coerce"
            )
            page_sizes_df.dropna(subset=["page"], inplace=True)
            if not page_sizes_df.empty:  # Check again after dropping NaNs
                page_sizes_df["page"] = page_sizes_df["page"].astype(int)
            else:
                print(
                    "Warning: Page sizes DataFrame became empty after processing, coordinate division will be skipped."
                )

    # 3. Join additional data from redaction_decision_output if provided
    text_added_successfully = False  # Flag to track if text was added by any method

    if not redaction_decision_output.empty:
        # --- Attempt to join data based on 'id' column first ---

        # Check if 'id' columns are present and have non-null values in *both* dataframes
        id_col_exists_in_review = (
            "id" in review_file_df.columns
            and not review_file_df["id"].isnull().all()
            and not (review_file_df["id"] == "").all()
        )
        id_col_exists_in_redaction = (
            "id" in redaction_decision_output.columns
            and not redaction_decision_output["id"].isnull().all()
            and not (redaction_decision_output["id"] == "").all()
        )

        if id_col_exists_in_review and id_col_exists_in_redaction:
            # print("Attempting to join data based on 'id' column.")
            try:
                # Ensure 'id' columns are of string type for robust merging
                review_file_df["id"] = review_file_df["id"].astype(str)
                # Make a copy if needed, but try to avoid if redaction_decision_output isn't modified later
                # Let's use a copy for safety as in the original code
                redaction_copy = redaction_decision_output.copy()
                redaction_copy["id"] = redaction_copy["id"].astype(str)

                # Select columns to merge from redaction output. Prioritize 'text'.
                cols_to_merge = ["id"]
                if "text" in redaction_copy.columns:
                    cols_to_merge.append("text")
                else:
                    print(
                        "Warning: 'text' column not found in redaction_decision_output. Cannot merge text using 'id'."
                    )

                # Perform a left merge to keep all annotations and add matching text
                # Use a suffix for the text column from the right DataFrame
                original_text_col_exists = "text" in review_file_df.columns
                merge_suffix = "_redaction" if original_text_col_exists else ""

                merged_df = pd.merge(
                    review_file_df,
                    redaction_copy[cols_to_merge],
                    on="id",
                    how="left",
                    suffixes=("", merge_suffix),
                )

                # Update the 'text' column if a new one was brought in
                if "text" + merge_suffix in merged_df.columns:
                    redaction_text_col = "text" + merge_suffix
                    if original_text_col_exists:
                        # Combine: Use text from redaction where available, otherwise keep original
                        merged_df["text"] = merged_df[redaction_text_col].combine_first(
                            merged_df["text"]
                        )
                        # Drop the temporary column
                        merged_df = merged_df.drop(columns=[redaction_text_col])
                    else:
                        # Redaction output had text, but review_file_df didn't. Rename the new column.
                        merged_df = merged_df.rename(
                            columns={redaction_text_col: "text"}
                        )

                    text_added_successfully = (
                        True  # Indicate text was potentially added
                    )

                review_file_df = merged_df  # Update the main DataFrame

                # print("Successfully attempted to join data using 'id'.") # Note: Text might not have been in redaction data

            except Exception as e:
                print(
                    f"Error during 'id'-based merge: {e}. Checking for proximity match fallback."
                )
                # Fall through to proximity match logic below

        # --- Fallback to proximity match if ID join wasn't possible/successful and enabled ---
        # Note: If id_col_exists_in_review or id_col_exists_in_redaction was False,
        # the block above was skipped, and we naturally fall here.
        # If an error occurred in the try block, joined_by_id would implicitly be False
        # because text_added_successfully wasn't set to True.

        # Only attempt proximity match if text wasn't added by ID join and proximity is requested
        if not text_added_successfully and do_proximity_match:
            # print("Attempting proximity match to add text data.")

            # Ensure 'page' columns are numeric before coordinate division and proximity match
            # (Assuming divide_coordinates_by_page_sizes and do_proximity_match_all_pages_for_text need this)
            if "page" in review_file_df.columns:
                review_file_df["page"] = (
                    pd.to_numeric(review_file_df["page"], errors="coerce")
                    .fillna(-1)
                    .astype(int)
                )  # Use -1 for NaN pages
                review_file_df = review_file_df[
                    review_file_df["page"] != -1
                ]  # Drop rows where page conversion failed
            if (
                not redaction_decision_output.empty
                and "page" in redaction_decision_output.columns
            ):
                redaction_decision_output["page"] = (
                    pd.to_numeric(redaction_decision_output["page"], errors="coerce")
                    .fillna(-1)
                    .astype(int)
                )
                redaction_decision_output = redaction_decision_output[
                    redaction_decision_output["page"] != -1
                ]

            # Perform coordinate division IF page_sizes were processed and DataFrame is not empty
            if not page_sizes_df.empty:
                # Apply coordinate division *before* proximity match
                review_file_df = divide_coordinates_by_page_sizes(
                    review_file_df, page_sizes_df
                )
                if not redaction_decision_output.empty:
                    redaction_decision_output = divide_coordinates_by_page_sizes(
                        redaction_decision_output, page_sizes_df
                    )

            # Now perform the proximity match
            # Note: Potential DataFrame copies happen inside do_proximity_match based on its implementation
            if not redaction_decision_output.empty:
                try:
                    review_file_df = do_proximity_match_all_pages_for_text(
                        df1=review_file_df,  # Pass directly, avoid caller copy if possible by modifying function signature
                        df2=redaction_decision_output,  # Pass directly
                    )
                    # Assuming do_proximity_match_all_pages_for_text adds/updates the 'text' column
                    if "text" in review_file_df.columns:
                        text_added_successfully = True
                    # print("Proximity match completed.")
                except Exception as e:
                    print(
                        f"Error during proximity match: {e}. Text data may not be added."
                    )

        elif not text_added_successfully and not do_proximity_match:
            print(
                "Skipping joining text data (ID join not possible/failed, proximity match disabled)."
            )

    # 4. Ensure required columns exist and are ordered
    # Define base required columns. 'id' and 'text' are conditionally added.
    required_columns_base = [
        "image",
        "page",
        "label",
        "color",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    final_columns = required_columns_base[:]  # Start with base columns

    # Add 'id' and 'text' if they exist in the DataFrame at this point
    if "id" in review_file_df.columns:
        final_columns.append("id")
    if "text" in review_file_df.columns:
        final_columns.append("text")  # Add text column if it was created/merged

    # Add any missing required columns with a default value (e.g., blank string)
    for col in final_columns:
        if col not in review_file_df.columns:
            # Use appropriate default based on expected type, '' for text/id, np.nan for coords?
            # Sticking to '' as in original for simplicity, but consider data types.
            review_file_df[col] = (
                ""  # Or np.nan for numerical, but coords already checked by dropna
            )

    # Select and order the final set of columns
    # Ensure all selected columns actually exist after adding defaults
    review_file_df = review_file_df[
        [col for col in final_columns if col in review_file_df.columns]
    ]

    # 5. Final processing and sorting
    # Convert colours from list to tuple if necessary - apply is okay here unless lists are vast
    if "color" in review_file_df.columns:
        # Check if the column actually contains lists before applying lambda
        if review_file_df["color"].apply(lambda x: isinstance(x, list)).any():
            review_file_df.loc[:, "color"] = review_file_df.loc[:, "color"].apply(
                lambda x: tuple(x) if isinstance(x, list) else x
            )

    # Sort the results
    # Ensure sort columns exist before sorting
    sort_columns = ["page", "ymin", "xmin", "label"]
    valid_sort_columns = [col for col in sort_columns if col in review_file_df.columns]
    if valid_sort_columns and not review_file_df.empty:  # Only sort non-empty df
        # Convert potential numeric sort columns to appropriate types if necessary
        # (e.g., 'page', 'ymin', 'xmin') to ensure correct sorting.
        # dropna(subset=[...], inplace=True) earlier should handle NaNs in coords.
        # page conversion already done before proximity match.
        try:
            review_file_df = review_file_df.sort_values(valid_sort_columns)
        except TypeError as e:
            print(
                f"Warning: Could not sort DataFrame due to type error in sort columns: {e}"
            )
            # Proceed without sorting

    base_cols = ["xmin", "xmax", "ymin", "ymax", "text", "id", "label"]

    for col in base_cols:
        if col not in review_file_df.columns:
            review_file_df[col] = pd.NA

    review_file_df = review_file_df.dropna(subset=base_cols, how="all")

    return review_file_df


def fill_missing_ids_in_list(data_list: list) -> list:
    """
    Generates unique alphanumeric IDs for dictionaries in a list where the 'id' is
    missing, blank, or not a 12-character string.

    Args:
        data_list (list): A list of dictionaries, each potentially with an 'id' key.

    Returns:
        list: The input list with missing/invalid IDs filled.
              Note: The function modifies the input list in place.
    """

    # --- Input Validation ---
    if not isinstance(data_list, list):
        raise TypeError("Input 'data_list' must be a list.")

    if not data_list:
        return data_list  # Return empty list as-is

    id_length = 12
    character_set = string.ascii_letters + string.digits  # a-z, A-Z, 0-9

    # --- Get Existing IDs to Ensure Uniqueness ---
    # Collect all valid existing IDs first
    existing_ids = set()
    for item in data_list:
        if not isinstance(item, dict):
            continue  # Skip non-dictionary items
        item_id = item.get("id")
        if isinstance(item_id, str) and len(item_id) == id_length:
            existing_ids.add(item_id)

    # --- Identify and Fill Items Needing IDs ---
    generated_ids_set = set()  # Keep track of IDs generated *in this run*
    num_filled = 0

    for item in data_list:
        if not isinstance(item, dict):
            continue  # Skip non-dictionary items

        item_id = item.get("id")

        # Check if ID needs to be generated
        # Needs ID if: key is missing, value is None, value is not a string,
        # value is an empty string after stripping whitespace, or value is a string
        # but not of the correct length.
        needs_new_id = (
            item_id is None
            or not isinstance(item_id, str)
            or item_id.strip() == ""
            or len(item_id) != id_length
        )

        if needs_new_id:
            # Generate a unique ID
            attempts = 0
            while True:
                candidate_id = "".join(random.choices(character_set, k=id_length))
                # Check against *all* existing valid IDs and *newly* generated ones in this run
                if (
                    candidate_id not in existing_ids
                    and candidate_id not in generated_ids_set
                ):
                    generated_ids_set.add(candidate_id)
                    item["id"] = (
                        candidate_id  # Assign the new ID directly to the item dict
                    )
                    num_filled += 1
                    break  # Found a unique ID
                attempts += 1
                # Safety break for unlikely infinite loop (though highly improbable with 12 chars)
                if attempts > len(data_list) * 100 + 1000:
                    raise RuntimeError(
                        f"Failed to generate a unique ID after {attempts} attempts. Check ID length or existing IDs."
                    )

    if num_filled > 0:
        pass
        # print(f"Successfully filled {num_filled} missing or invalid IDs.")
    else:
        pass
        # print("No missing or invalid IDs found.")

    # The input list 'data_list' has been modified in place
    return data_list


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
    # if 'boxes' not in data_input or not isinstance(data_input.get('boxes'), list):
    #    raise ValueError("Input dictionary must contain a 'boxes' key with a list value.")

    boxes = data_input  # ['boxes']
    id_length = 12
    character_set = string.ascii_letters + string.digits  # a-z, A-Z, 0-9

    # --- Get Existing IDs to Ensure Uniqueness ---
    # Collect all valid existing IDs first
    existing_ids = set()
    # for box in boxes:
    # Check if 'id' exists, is a string, and is the correct length
    box_id = boxes.get("id")
    if isinstance(box_id, str) and len(box_id) == id_length:
        existing_ids.add(box_id)

    # --- Identify and Fill Rows Needing IDs ---
    generated_ids_set = set()  # Keep track of IDs generated *in this run*
    num_filled = 0

    # for box in boxes:
    box_id = boxes.get("id")

    # Check if ID needs to be generated
    # Needs ID if: key is missing, value is None, value is not a string,
    # value is an empty string after stripping whitespace, or value is a string
    # but not of the correct length.
    needs_new_id = (
        box_id is None
        or not isinstance(box_id, str)
        or box_id.strip() == ""
        or len(box_id) != id_length
    )

    if needs_new_id:
        # Generate a unique ID
        attempts = 0
        while True:
            candidate_id = "".join(random.choices(character_set, k=id_length))
            # Check against *all* existing valid IDs and *newly* generated ones in this run
            if (
                candidate_id not in existing_ids
                and candidate_id not in generated_ids_set
            ):
                generated_ids_set.add(candidate_id)
                boxes["id"] = candidate_id  # Assign the new ID directly to the box dict
                num_filled += 1
                break  # Found a unique ID
            attempts += 1
            # Safety break for unlikely infinite loop (though highly improbable with 12 chars)
            if attempts > len(boxes) * 100 + 1000:
                raise RuntimeError(
                    f"Failed to generate a unique ID after {attempts} attempts. Check ID length or existing IDs."
                )

    if num_filled > 0:
        pass
        # print(f"Successfully filled {num_filled} missing or invalid box IDs.")
    else:
        pass
        # print("No missing or invalid box IDs found.")

    # The input dictionary 'data_input' has been modified in place
    return data_input


def fill_missing_box_ids_each_box(data_input: Dict) -> Dict:
    """
    Generates unique alphanumeric IDs for bounding boxes in a list
    where the 'id' is missing, blank, or not a 12-character string.

    Args:
        data_input (Dict): The input dictionary containing 'image' and 'boxes' keys.
                           'boxes' should be a list of dictionaries, each potentially
                           with an 'id' key.

    Returns:
        Dict: The input dictionary with missing/invalid box IDs filled.
              Note: The function modifies the input dictionary in place.
    """
    # --- Input Validation ---
    if not isinstance(data_input, dict):
        raise TypeError("Input 'data_input' must be a dictionary.")
    if "boxes" not in data_input or not isinstance(data_input.get("boxes"), list):
        # If there are no boxes, there's nothing to do.
        return data_input

    boxes_list = data_input["boxes"]
    id_length = 12
    character_set = string.ascii_letters + string.digits

    # --- 1. Get ALL Existing IDs to Ensure Uniqueness ---
    # Collect all valid existing IDs from the entire list first.
    existing_ids = set()
    for box in boxes_list:
        if isinstance(box, dict):
            box_id = box.get("id")
            if isinstance(box_id, str) and len(box_id) == id_length:
                existing_ids.add(box_id)

    # --- 2. Iterate and Fill IDs for each box ---
    generated_ids_this_run = set()  # Keep track of IDs generated in this run
    num_filled = 0

    for box in boxes_list:
        if not isinstance(box, dict):
            continue  # Skip items in the list that are not dictionaries

        box_id = box.get("id")

        # Check if this specific box needs a new ID
        needs_new_id = (
            box_id is None
            or not isinstance(box_id, str)
            or box_id.strip() == ""
            or len(box_id) != id_length
        )

        if needs_new_id:
            # Generate a truly unique ID
            while True:
                candidate_id = "".join(random.choices(character_set, k=id_length))
                # Check against original IDs and newly generated IDs
                if (
                    candidate_id not in existing_ids
                    and candidate_id not in generated_ids_this_run
                ):
                    generated_ids_this_run.add(candidate_id)
                    box["id"] = candidate_id  # Assign the ID to the individual box
                    num_filled += 1
                    break  # Move to the next box

    if num_filled > 0:
        pass
        # print(f"Successfully filled {num_filled} missing or invalid box IDs.")

    # The input dictionary 'data_input' has been modified in place
    return data_input


def fill_missing_ids(
    df: pd.DataFrame, column_name: str = "id", length: int = 12
) -> pd.DataFrame:
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
        # print(f"Column '{column_name}' not found. Adding it to the DataFrame.")
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
    is_empty_str = pd.Series(False, index=df.index)  # Default to False
    if not is_null.all():  # Only check strings if there are non-null values
        temp_str_col = df.loc[~is_null, column_name].astype(str).str.strip()
        is_empty_str.loc[~is_null] = temp_str_col == ""

    # Combine the conditions
    is_missing_or_empty = is_null | is_empty_str

    rows_to_fill_index = df.index[is_missing_or_empty]
    num_needed = len(rows_to_fill_index)

    if num_needed == 0:
        # Ensure final column type is consistent if nothing was done
        if pd.api.types.is_object_dtype(original_dtype) or pd.api.types.is_string_dtype(
            original_dtype
        ):
            pass  # Likely already object or string
        else:
            # If original was numeric/etc., but might contain strings now? Unlikely here.
            pass  # Or convert to object: df[column_name] = df[column_name].astype(object)
        # print(f"No missing or empty values found requiring IDs in column '{column_name}'.")
        return df

    # print(f"Found {num_needed} rows requiring a unique ID in column '{column_name}'.")

    # --- Get Existing IDs to Ensure Uniqueness ---
    # Consider only rows that are *not* missing/empty
    valid_rows = df.loc[~is_missing_or_empty, column_name]
    # Drop any remaining nulls (shouldn't be any based on mask, but belts and braces)
    valid_rows = valid_rows.dropna()
    # Convert to string *only* if not already string/object, then filter out empty strings again
    if not pd.api.types.is_object_dtype(
        valid_rows.dtype
    ) and not pd.api.types.is_string_dtype(valid_rows.dtype):
        existing_ids = set(valid_rows.astype(str).str.strip())
    else:  # Already string or object, just strip and convert to set
        existing_ids = set(
            valid_rows.astype(str).str.strip()
        )  # astype(str) handles mixed types in object column

    # Remove empty string from existing IDs if it's there after stripping
    existing_ids.discard("")

    # --- Generate Unique IDs ---
    character_set = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    generated_ids_set = set()  # Keep track of IDs generated *in this run*
    new_ids_list = list()  # Store the generated IDs in order

    max_possible_ids = len(character_set) ** length
    if num_needed > max_possible_ids:
        raise ValueError(
            f"Cannot generate {num_needed} unique IDs with length {length}. Maximum possible is {max_possible_ids}."
        )

    # Pre-calculate safety break limit
    max_attempts_per_id = max(1000, num_needed * 10)  # Adjust multiplier as needed

    # print(f"Generating {num_needed} unique IDs of length {length}...")
    for i in range(num_needed):
        attempts = 0
        while True:
            candidate_id = "".join(random.choices(character_set, k=length))
            # Check against *all* known existing IDs and *newly* generated ones
            if (
                candidate_id not in existing_ids
                and candidate_id not in generated_ids_set
            ):
                generated_ids_set.add(candidate_id)
                new_ids_list.append(candidate_id)
                break  # Found a unique ID
            attempts += 1
            if attempts > max_attempts_per_id:  # Safety break
                raise RuntimeError(
                    f"Failed to generate a unique ID after {attempts} attempts. Check length, character set, or density of existing IDs."
                )

        # Optional progress update
        # if (i + 1) % 1000 == 0:
        #    print(f"Generated {i+1}/{num_needed} IDs...")

    # --- Assign New IDs ---
    # Use the previously identified index to assign the new IDs correctly
    # Assigning string IDs might change the column's dtype to 'object'
    if not pd.api.types.is_object_dtype(
        original_dtype
    ) and not pd.api.types.is_string_dtype(original_dtype):
        df["id"] = df["id"].astype(str, errors="ignore")
        # warnings.warn(f"Column '{column_name}' dtype might change from '{original_dtype}' to 'object' due to string ID assignment.", UserWarning)

    df.loc[rows_to_fill_index, column_name] = new_ids_list
    # print(
    #     f"Successfully assigned {len(new_ids_list)} new unique IDs to column '{column_name}'."
    # )

    return df


def convert_review_df_to_annotation_json(
    review_file_df: pd.DataFrame,
    image_paths: List[str],  # List of image file paths
    page_sizes: List[
        Dict
    ],  # List of dicts like [{'page': 1, 'image_path': '...', 'image_width': W, 'image_height': H}, ...]
    xmin="xmin",
    xmax="xmax",
    ymin="ymin",
    ymax="ymax",  # Coordinate column names
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

    review_file_df = review_file_df.dropna(
        subset=["xmin", "xmax", "ymin", "ymax", "text", "id", "label"], how="all"
    )

    if not page_sizes:
        raise ValueError("page_sizes argument is required and cannot be empty.")

    # --- Prepare Page Sizes DataFrame ---
    try:
        page_sizes_df = pd.DataFrame(page_sizes)
        required_ps_cols = {"page", "image_path", "image_width", "image_height"}
        if not required_ps_cols.issubset(page_sizes_df.columns):
            missing = required_ps_cols - set(page_sizes_df.columns)
            raise ValueError(f"page_sizes is missing required keys: {missing}")
        # Convert page sizes columns to appropriate numeric types early
        page_sizes_df["page"] = pd.to_numeric(page_sizes_df["page"], errors="coerce")
        page_sizes_df["image_width"] = pd.to_numeric(
            page_sizes_df["image_width"], errors="coerce"
        )
        page_sizes_df["image_height"] = pd.to_numeric(
            page_sizes_df["image_height"], errors="coerce"
        )
        # Use nullable Int64 for page number consistency
        page_sizes_df["page"] = page_sizes_df["page"].astype("Int64")

    except Exception as e:
        raise ValueError(f"Error processing page_sizes: {e}") from e

    # Handle empty input DataFrame gracefully
    if review_file_df.empty:
        print(
            "Input review_file_df is empty. Proceeding to generate JSON structure with empty boxes."
        )
        # Ensure essential columns exist even if empty for later steps
        for col in [xmin, xmax, ymin, ymax, "page", "label", "color", "id", "text"]:
            if col not in review_file_df.columns:
                review_file_df[col] = pd.NA
    else:
        # --- Coordinate Conversion (if needed) ---
        coord_cols_to_check = [
            c for c in [xmin, xmax, ymin, ymax] if c in review_file_df.columns
        ]
        needs_multiplication = False
        if coord_cols_to_check:
            temp_df_numeric = review_file_df[coord_cols_to_check].apply(
                pd.to_numeric, errors="coerce"
            )
            if (
                temp_df_numeric.le(1).any().any()
            ):  # Check if any numeric coord <= 1 exists
                needs_multiplication = True

        if needs_multiplication:
            # print("Relative coordinates detected or suspected, running multiplication...")
            review_file_df = multiply_coordinates_by_page_sizes(
                review_file_df.copy(),  # Pass a copy to avoid modifying original outside function
                page_sizes_df,
                xmin,
                xmax,
                ymin,
                ymax,
            )
        else:
            # print("No relative coordinates detected or required columns missing, skipping multiplication.")
            # Still ensure essential coordinate/page columns are numeric if they exist
            cols_to_convert = [
                c
                for c in [xmin, xmax, ymin, ymax, "page"]
                if c in review_file_df.columns
            ]
            for col in cols_to_convert:
                review_file_df[col] = pd.to_numeric(
                    review_file_df[col], errors="coerce"
                )

        # Handle potential case where multiplication returns an empty DF
        if review_file_df.empty:
            print("DataFrame became empty after coordinate processing.")
            # Re-add essential columns if they were lost
            for col in [xmin, xmax, ymin, ymax, "page", "label", "color", "id", "text"]:
                if col not in review_file_df.columns:
                    review_file_df[col] = pd.NA

        # --- Fill Missing IDs ---
        review_file_df = fill_missing_ids(review_file_df.copy())  # Pass a copy

        # --- Deduplicate Based on Key Fields ---
        base_dedupe_cols = ["page", xmin, ymin, xmax, ymax, "label", "id"]
        # Identify which deduplication columns actually exist in the DataFrame
        cols_for_dedupe = [
            col for col in base_dedupe_cols if col in review_file_df.columns
        ]
        # Add 'image' column for deduplication IF it exists (matches original logic intent)
        if "image" in review_file_df.columns:
            cols_for_dedupe.append("image")

        # Ensure placeholder columns exist if they are needed for deduplication
        # (e.g., 'label', 'id' should be present after fill_missing_ids)
        for col in ["label", "id"]:
            if col in cols_for_dedupe and col not in review_file_df.columns:
                # This might indicate an issue in fill_missing_ids or prior steps
                print(
                    f"Warning: Column '{col}' needed for dedupe but not found. Adding NA."
                )
                review_file_df[col] = ""  # Add default empty string

        if cols_for_dedupe:  # Only attempt dedupe if we have columns to check
            # print(f"Deduplicating based on columns: {cols_for_dedupe}")
            # Convert relevant columns to string before dedupe to avoid type issues with mixed data (optional, depends on data)
            # for col in cols_for_dedupe:
            #    review_file_df[col] = review_file_df[col].astype(str)
            review_file_df = review_file_df.drop_duplicates(subset=cols_for_dedupe)
        else:
            print("Skipping deduplication: No valid columns found to deduplicate by.")

    # --- Select and Prepare Final Output Columns ---
    required_final_cols = [
        "page",
        "label",
        "color",
        xmin,
        ymin,
        xmax,
        ymax,
        "id",
        "text",
    ]
    # Identify which of the desired final columns exist in the (now potentially deduplicated) DataFrame
    available_final_cols = [
        col for col in required_final_cols if col in review_file_df.columns
    ]

    # Ensure essential output columns exist, adding defaults if missing AFTER deduplication
    for col in required_final_cols:
        if col not in review_file_df.columns:
            print(f"Adding missing final column '{col}' with default value.")
            if col in ["label", "id", "text"]:
                review_file_df[col] = ""  # Default empty string
            elif col == "color":
                review_file_df[col] = None  # Default None or a default color tuple
            else:  # page, coordinates
                review_file_df[col] = pd.NA  # Default NA for numeric/page
            available_final_cols.append(col)  # Add to list of available columns

    # Select only the final desired columns in the correct order
    review_file_df = review_file_df[available_final_cols]

    # --- Final Formatting ---
    if not review_file_df.empty:
        # Convert list colors to tuples (important for some downstream uses)
        if "color" in review_file_df.columns:
            is_list = review_file_df["color"].apply(lambda x: isinstance(x, list))
            if is_list.any():
                review_file_df.loc[is_list, "color"] = review_file_df.loc[
                    is_list, "color"
                ].apply(tuple)
        # Ensure page column is nullable integer type for reliable grouping
        if "page" in review_file_df.columns:
            review_file_df["page"] = review_file_df["page"].astype("Int64")

    # --- Group Annotations by Page ---
    output_cols_for_boxes = [
        col
        for col in ["label", "color", xmin, ymin, xmax, ymax, "id", "text"]
        if col in review_file_df.columns
    ]

    # Ensure coordinate columns are native Python floats (not np.float64) for JSON/dict
    for c in [xmin, xmax, ymin, ymax]:
        if c in review_file_df.columns:
            review_file_df[c] = review_file_df[c].apply(
                lambda x: float(x) if pd.notna(x) else x
            )

    if "page" in review_file_df.columns:
        # Build page -> list of box dicts once (avoids iterrows + get_group per page)
        page_to_boxes = {}
        for page_num, group in review_file_df.groupby("page"):
            if pd.notna(page_num):
                page_to_boxes[page_num] = (
                    group[output_cols_for_boxes]
                    .replace({np.nan: None})
                    .to_dict(orient="records")
                )
    else:
        print("Error: 'page' column missing, cannot group annotations.")
        page_to_boxes = {}

    # --- Build JSON Structure ---
    # Iterate page_sizes by column (no iterrows); lookup boxes by page
    json_data = [
        {
            "image": pdf_image_path,
            "boxes": page_to_boxes.get(page_num, []) if pd.notna(page_num) else [],
        }
        for page_num, pdf_image_path in zip(
            page_sizes_df["page"], page_sizes_df["image_path"]
        )
    ]

    return json_data
