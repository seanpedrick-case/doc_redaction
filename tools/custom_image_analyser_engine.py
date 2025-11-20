import ast
import base64
import copy
import io
import json
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import botocore
import cv2
import gradio as gr
import numpy as np
import pandas as pd
import pytesseract
import requests
import spaces
from pdfminer.layout import LTChar
from PIL import Image, ImageDraw, ImageFont
from presidio_analyzer import AnalyzerEngine, RecognizerResult

from tools.config import (
    AWS_PII_OPTION,
    CONVERT_LINE_TO_WORD_LEVEL,
    DEFAULT_LANGUAGE,
    HYBRID_OCR_CONFIDENCE_THRESHOLD,
    HYBRID_OCR_MAX_NEW_TOKENS,
    HYBRID_OCR_PADDING,
    INFERENCE_SERVER_API_URL,
    INFERENCE_SERVER_MODEL_NAME,
    INFERENCE_SERVER_TIMEOUT,
    LOAD_PADDLE_AT_STARTUP,
    LOCAL_OCR_MODEL_OPTIONS,
    LOCAL_PII_OPTION,
    MAX_SPACES_GPU_RUN_TIME,
    OUTPUT_FOLDER,
    PADDLE_DET_DB_UNCLIP_RATIO,
    PADDLE_FONT_PATH,
    PADDLE_MODEL_PATH,
    PADDLE_USE_TEXTLINE_ORIENTATION,
    PREPROCESS_LOCAL_OCR_IMAGES,
    REPORT_VLM_OUTPUTS_TO_GUI,
    SAVE_EXAMPLE_HYBRID_IMAGES,
    SAVE_PAGE_OCR_VISUALISATIONS,
    SAVE_PREPROCESS_IMAGES,
    SAVE_VLM_INPUT_IMAGES,
    SELECTED_MODEL,
    TESSERACT_SEGMENTATION_LEVEL,
    TESSERACT_WORD_LEVEL_OCR,
    VLM_MAX_DPI,
    VLM_MAX_IMAGE_SIZE,
)
from tools.helper_functions import clean_unicode_text, get_system_font_path
from tools.load_spacy_model_custom_recognisers import custom_entities
from tools.presidio_analyzer_custom import recognizer_result_from_dict
from tools.run_vlm import (
    extract_text_from_image_vlm,
    full_page_ocr_vlm_prompt,
    model_default_max_new_tokens,
    model_default_prompt,
    model_default_repetition_penalty,
    model_default_seed,
    model_default_temperature,
    model_default_top_k,
    model_default_top_p,
)
from tools.secure_path_utils import validate_folder_containment
from tools.secure_regex_utils import safe_sanitize_text
from tools.word_segmenter import AdaptiveSegmenter

if LOAD_PADDLE_AT_STARTUP:
    # Set PaddleOCR font path BEFORE importing to prevent font downloads during import
    if (
        PADDLE_FONT_PATH
        and PADDLE_FONT_PATH.strip()
        and os.path.exists(PADDLE_FONT_PATH)
    ):
        os.environ["PADDLE_PDX_LOCAL_FONT_FILE_PATH"] = PADDLE_FONT_PATH
    else:
        system_font_path = get_system_font_path()
        if system_font_path:
            os.environ["PADDLE_PDX_LOCAL_FONT_FILE_PATH"] = system_font_path

    try:
        from paddleocr import PaddleOCR

        print("PaddleOCR imported successfully")
    except Exception as e:
        print(f"Error importing PaddleOCR: {e}")
        PaddleOCR = None
else:
    PaddleOCR = None


# --- Language utilities ---
def _normalize_lang(language: str) -> str:
    return language.strip().lower().replace("-", "_") if language else "en"


def _tesseract_lang_code(language: str) -> str:
    """Map a user language input to a Tesseract traineddata code."""
    lang = _normalize_lang(language)

    mapping = {
        # Common
        "en": "eng",
        "eng": "eng",
        "fr": "fra",
        "fre": "fra",
        "fra": "fra",
        "de": "deu",
        "ger": "deu",
        "deu": "deu",
        "es": "spa",
        "spa": "spa",
        "it": "ita",
        "ita": "ita",
        "nl": "nld",
        "dut": "nld",
        "nld": "nld",
        "pt": "por",
        "por": "por",
        "ru": "rus",
        "rus": "rus",
        "ar": "ara",
        "ara": "ara",
        # Nordics
        "sv": "swe",
        "swe": "swe",
        "no": "nor",
        "nb": "nor",
        "nn": "nor",
        "nor": "nor",
        "fi": "fin",
        "fin": "fin",
        "da": "dan",
        "dan": "dan",
        # Eastern/Central
        "pl": "pol",
        "pol": "pol",
        "cs": "ces",
        "cz": "ces",
        "ces": "ces",
        "hu": "hun",
        "hun": "hun",
        "ro": "ron",
        "rum": "ron",
        "ron": "ron",
        "bg": "bul",
        "bul": "bul",
        "el": "ell",
        "gre": "ell",
        "ell": "ell",
        # Asian
        "ja": "jpn",
        "jp": "jpn",
        "jpn": "jpn",
        "zh": "chi_sim",
        "zh_cn": "chi_sim",
        "zh_hans": "chi_sim",
        "chi_sim": "chi_sim",
        "zh_tw": "chi_tra",
        "zh_hk": "chi_tra",
        "zh_tr": "chi_tra",
        "chi_tra": "chi_tra",
        "hi": "hin",
        "hin": "hin",
        "bn": "ben",
        "ben": "ben",
        "ur": "urd",
        "urd": "urd",
        "fa": "fas",
        "per": "fas",
        "fas": "fas",
    }

    return mapping.get(lang, "eng")


def _paddle_lang_code(language: str) -> str:
    """Map a user language input to a PaddleOCR language code.

    PaddleOCR supports codes like: 'en', 'ch', 'chinese_cht', 'korean', 'japan', 'german', 'fr', 'it', 'es',
    as well as script packs like 'arabic', 'cyrillic', 'latin'.
    """
    lang = _normalize_lang(language)

    mapping = {
        "en": "en",
        "fr": "fr",
        "de": "german",
        "es": "es",
        "it": "it",
        "pt": "pt",
        "nl": "nl",
        "ru": "cyrillic",  # Russian is covered by cyrillic models
        "uk": "cyrillic",
        "bg": "cyrillic",
        "sr": "cyrillic",
        "ar": "arabic",
        "tr": "tr",
        "fa": "arabic",  # fallback to arabic script pack
        "zh": "ch",
        "zh_cn": "ch",
        "zh_tw": "chinese_cht",
        "zh_hk": "chinese_cht",
        "ja": "japan",
        "jp": "japan",
        "ko": "korean",
        "hi": "latin",  # fallback; dedicated Hindi not always available
    }

    return mapping.get(lang, "en")


@dataclass
class OCRResult:
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float = None
    line: int = None
    model: str = (
        None  # Track which OCR model was used (e.g., "Tesseract", "Paddle", "VLM")
    )


@dataclass
class CustomImageRecognizerResult:
    entity_type: str
    start: int
    end: int
    score: float
    left: int
    top: int
    width: int
    height: int
    text: str
    color: tuple = (0, 0, 0)


class ImagePreprocessor:
    """ImagePreprocessor class. Parent class for image preprocessing objects."""

    def __init__(self, use_greyscale: bool = True) -> None:
        self.use_greyscale = use_greyscale

    def preprocess_image(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        return image, {}

    def convert_image_to_array(self, image: Image.Image) -> np.ndarray:
        if isinstance(image, np.ndarray):
            img = image
        else:
            if self.use_greyscale:
                image = image.convert("L")
            img = np.asarray(image)
        return img

    @staticmethod
    def _get_bg_color(
        image: np.ndarray, is_greyscale: bool, invert: bool = False
    ) -> Union[int, Tuple[int, int, int]]:
        # Note: Modified to expect numpy array for bincount
        if invert:
            image = 255 - image  # Simple inversion for greyscale numpy array

        if is_greyscale:
            bg_color = int(np.bincount(image.flatten()).argmax())
        else:
            # This part would need more complex logic for color numpy arrays
            # For this pipeline, we only use greyscale, so it's fine.
            # A simple alternative:
            from scipy import stats

            bg_color = tuple(stats.mode(image.reshape(-1, 3), axis=0)[0][0])
        return bg_color

    @staticmethod
    def _get_image_contrast(image: np.ndarray) -> Tuple[float, float]:
        contrast = np.std(image)
        mean_intensity = np.mean(image)
        return contrast, mean_intensity


class BilateralFilter(ImagePreprocessor):
    """Applies bilateral filtering."""

    def __init__(
        self, diameter: int = 9, sigma_color: int = 75, sigma_space: int = 75
    ) -> None:
        super().__init__(use_greyscale=True)
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        # Modified to accept and return numpy array for consistency in the pipeline
        filtered_image = cv2.bilateralFilter(
            image, self.diameter, self.sigma_color, self.sigma_space
        )
        metadata = {
            "diameter": self.diameter,
            "sigma_color": self.sigma_color,
            "sigma_space": self.sigma_space,
        }
        return filtered_image, metadata


class SegmentedAdaptiveThreshold(ImagePreprocessor):
    """Applies adaptive thresholding."""

    def __init__(
        self,
        block_size: int = 21,
        contrast_threshold: int = 40,
        c_low_contrast: int = 5,
        c_high_contrast: int = 10,
        bg_threshold: int = 127,
    ) -> None:
        super().__init__(use_greyscale=True)
        self.block_size = (
            block_size if block_size % 2 == 1 else block_size + 1
        )  # Ensure odd
        self.c_low_contrast = c_low_contrast
        self.c_high_contrast = c_high_contrast
        self.bg_threshold = bg_threshold
        self.contrast_threshold = contrast_threshold

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        # Modified to accept and return numpy array
        background_color = self._get_bg_color(image, True)
        contrast, _ = self._get_image_contrast(image)
        c = (
            self.c_low_contrast
            if contrast <= self.contrast_threshold
            else self.c_high_contrast
        )

        if background_color < self.bg_threshold:  # Dark background, light text
            adaptive_threshold_image = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.block_size,
                -c,
            )
        else:  # Light background, dark text
            adaptive_threshold_image = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.block_size,
                c,
            )
        metadata = {"C": c, "background_color": background_color, "contrast": contrast}
        return adaptive_threshold_image, metadata


class ImageRescaling(ImagePreprocessor):
    """Rescales images based on their size."""

    def __init__(self, target_dpi: int = 300, assumed_input_dpi: int = 96) -> None:
        super().__init__(use_greyscale=True)
        self.target_dpi = target_dpi
        self.assumed_input_dpi = assumed_input_dpi

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        # Modified to accept and return numpy array
        scale_factor = self.target_dpi / self.assumed_input_dpi
        metadata = {"scale_factor": 1.0}

        if scale_factor != 1.0:
            width = int(image.shape[1] * scale_factor)
            height = int(image.shape[0] * scale_factor)
            dimensions = (width, height)

            # Use better interpolation for upscaling vs downscaling
            interpolation = cv2.INTER_CUBIC if scale_factor > 1.0 else cv2.INTER_AREA
            rescaled_image = cv2.resize(image, dimensions, interpolation=interpolation)
            metadata["scale_factor"] = scale_factor
            return rescaled_image, metadata

        return image, metadata


class ContrastSegmentedImageEnhancer(ImagePreprocessor):
    """Class containing all logic to perform contrastive segmentation."""

    def __init__(
        self,
        bilateral_filter: Optional[BilateralFilter] = None,
        adaptive_threshold: Optional[SegmentedAdaptiveThreshold] = None,
        image_rescaling: Optional[ImageRescaling] = None,
        low_contrast_threshold: int = 40,
    ) -> None:
        super().__init__(use_greyscale=True)
        self.bilateral_filter = bilateral_filter or BilateralFilter()
        self.adaptive_threshold = adaptive_threshold or SegmentedAdaptiveThreshold()
        self.image_rescaling = image_rescaling or ImageRescaling()
        self.low_contrast_threshold = low_contrast_threshold

    def _improve_contrast(self, image: np.ndarray) -> Tuple[np.ndarray, str, str]:
        contrast, mean_intensity = self._get_image_contrast(image)
        if contrast <= self.low_contrast_threshold:
            # Using CLAHE as a generally more robust alternative
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            adjusted_image = clahe.apply(image)
            adjusted_contrast, _ = self._get_image_contrast(adjusted_image)
        else:
            adjusted_image = image
            adjusted_contrast = contrast
        return adjusted_image, contrast, adjusted_contrast

    def _deskew(self, image_np: np.ndarray) -> np.ndarray:
        """
        Corrects the skew of an image.
        This method works best on a grayscaled image.
        """
        # We'll work with a copy for angle detection
        gray = (
            cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            if len(image_np.shape) == 3
            else image_np.copy()
        )

        # Invert the image for contour finding
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # Adjust the angle for rotation
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Don't rotate if the angle is negligible
        if abs(angle) < 0.1:
            return image_np

        (h, w) = image_np.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Use the original numpy image for the rotation to preserve quality
        rotated = cv2.warpAffine(
            image_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    def preprocess_image(
        self,
        image: Image.Image,
        perform_deskew: bool = False,
        perform_binarization: bool = False,
    ) -> Tuple[Image.Image, dict]:
        """
        A pipeline for OCR preprocessing.
        Order: Deskew -> Greyscale -> Rescale -> Denoise -> Enhance Contrast -> Binarize
        """
        # 1. Convert PIL image to NumPy array for OpenCV processing
        # Assuming the original image is RGB
        image_np = np.array(image.convert("RGB"))
        # OpenCV uses BGR, so we convert RGB to BGR
        image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # --- REVISED PIPELINE ---

        # 2. Deskew the image (critical new step)
        # This is best done early on the full-quality image.
        if perform_deskew:
            deskewed_image_np = self._deskew(image_np_bgr)
        else:
            deskewed_image_np = image_np_bgr

        # 3. Convert to greyscale
        # Your convert_image_to_array probably does this, but for clarity:
        gray_image_np = cv2.cvtColor(deskewed_image_np, cv2.COLOR_BGR2GRAY)

        # 4. Rescale image to optimal DPI
        # Assuming your image_rescaling object can handle a greyscale numpy array
        rescaled_image_np, scale_metadata = self.image_rescaling.preprocess_image(
            gray_image_np
        )

        # 5. Apply filtering for noise reduction
        # Suggestion: A Median filter is often very effective for scanned docs
        # filtered_image_np = cv2.medianBlur(rescaled_image_np, 3)
        # Or using your existing bilateral filter:
        filtered_image_np, _ = self.bilateral_filter.preprocess_image(rescaled_image_np)

        # 6. Improve contrast
        adjusted_image_np, _, _ = self._improve_contrast(filtered_image_np)

        # 7. Adaptive Thresholding (Binarization) - Final optional step
        if perform_binarization:
            final_image_np, threshold_metadata = (
                self.adaptive_threshold.preprocess_image(adjusted_image_np)
            )
        else:
            final_image_np = adjusted_image_np
            threshold_metadata = {}

        # Combine metadata
        final_metadata = {**scale_metadata, **threshold_metadata}

        # Convert final numpy array back to PIL Image for return
        # The final image is greyscale, so it's safe to use 'L' mode
        return Image.fromarray(final_image_np).convert("L"), final_metadata


def rescale_ocr_data(ocr_data, scale_factor: float):

    # We loop from 0 to the number of detected words.
    num_boxes = len(ocr_data["text"])
    for i in range(num_boxes):
        # We only want to process actual words, not empty boxes Tesseract might find
        if int(ocr_data["conf"][i]) > -1:  # -1 confidence is for structural elements
            # Get coordinates from the processed image using the index 'i'
            x_proc = ocr_data["left"][i]
            y_proc = ocr_data["top"][i]
            w_proc = ocr_data["width"][i]
            h_proc = ocr_data["height"][i]

            # Apply the inverse transformation (division)
            x_orig = int(x_proc / scale_factor)
            y_orig = int(y_proc / scale_factor)
            w_orig = int(w_proc / scale_factor)
            h_orig = int(h_proc / scale_factor)

            # --- THE MAPPING STEP ---
            # Update the dictionary values in-place using the same index 'i'
            ocr_data["left"][i] = x_orig
            ocr_data["top"][i] = y_orig
            ocr_data["width"][i] = w_orig
            ocr_data["height"][i] = h_orig

    return ocr_data


def filter_entities_for_language(
    entities: List[str], valid_language_entities: List[str], language: str
) -> List[str]:

    if not valid_language_entities:
        print(f"No valid entities supported for language: {language}")
        # raise Warning(f"No valid entities supported for language: {language}")
    if not entities:
        print(f"No entities provided for language: {language}")
        # raise Warning(f"No entities provided for language: {language}")

    filtered_entities = [
        entity for entity in entities if entity in valid_language_entities
    ]

    if not filtered_entities:
        print(f"No relevant entities supported for language: {language}")
        # raise Warning(f"No relevant entities supported for language: {language}")

    if language != "en":
        gr.Info(
            f"Using {str(filtered_entities)} entities for local model analysis for language: {language}"
        )

    return filtered_entities


def _get_tesseract_psm(segmentation_level: str) -> int:
    """
    Get the appropriate Tesseract PSM (Page Segmentation Mode) value based on segmentation level.

    Args:
        segmentation_level: "word" or "line"

    Returns:
        PSM value for Tesseract configuration
    """
    if segmentation_level.lower() == "line":
        return 6  # Uniform block of text
    elif segmentation_level.lower() == "word":
        return 11  # Sparse text (word-level)
    else:
        print(
            f"Warning: Unknown segmentation level '{segmentation_level}', defaulting to word-level (PSM 11)"
        )
        return 11


def _prepare_image_for_vlm(image: Image.Image) -> Image.Image:
    """
    Prepare image for VLM by ensuring it doesn't exceed maximum size and DPI limits.

    Args:
        image: PIL Image to prepare

    Returns:
        PIL Image that has been resized if necessary to meet size and DPI constraints
    """
    if image is None:
        return image

    width, height = image.size

    # Get DPI information (if available)
    dpi = image.info.get("dpi", (72, 72))  # Default to 72 DPI if not specified
    if isinstance(dpi, tuple):
        dpi_x, dpi_y = dpi
        # Use the maximum DPI value
        current_dpi = max(dpi_x, dpi_y)
    else:
        current_dpi = float(dpi) if dpi else 72.0

    # Calculate scale factors needed
    size_scale = 1.0
    dpi_scale = 1.0

    # Check if total pixels exceed maximum
    total_pixels = width * height
    if total_pixels > VLM_MAX_IMAGE_SIZE:
        # Calculate scale factor to reduce total pixels to maximum
        # Since area scales with scale^2, we need sqrt of the ratio
        size_scale = (VLM_MAX_IMAGE_SIZE / total_pixels) ** 0.5
        print(
            f"VLM image size check: Image has {total_pixels:,} pixels ({width}x{height}), exceeds maximum {VLM_MAX_IMAGE_SIZE:,} pixels. Will resize by factor {size_scale:.3f}"
        )

    # Check if DPI exceeds maximum
    if current_dpi > VLM_MAX_DPI:
        dpi_scale = VLM_MAX_DPI / current_dpi
        # print(
        #     f"VLM DPI check: Image DPI {current_dpi:.1f} exceeds maximum {VLM_MAX_DPI:.1f} DPI. Will resize by factor {dpi_scale:.3f}"
        # )

    # Use the smaller scale factor to ensure both constraints are met
    final_scale = min(size_scale, dpi_scale)

    # Resize if necessary
    if final_scale < 1.0:
        new_width = int(width * final_scale)
        new_height = int(height * final_scale)
        # print(
        #     f"VLM image preparation: Resizing image from {width}x{height} to {new_width}x{new_height} (scale: {final_scale:.3f})"
        # )

        # Use high-quality resampling for downscaling
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Update DPI info if it was set
        if "dpi" in image.info:
            new_dpi = (current_dpi * final_scale, current_dpi * final_scale)
            # Create a copy with updated DPI info
            image_info = image.info.copy()
            image_info["dpi"] = new_dpi
            # Note: PIL doesn't allow direct modification of info dict, so we'll just note it
            # print(
            #     f"VLM image preparation: Effective DPI after resize: {new_dpi[0]:.1f}"
            # )
    else:
        total_pixels = width * height
        # print(
        #     f"VLM image preparation: Image size {width}x{height} ({total_pixels:,} pixels) and DPI {current_dpi:.1f} are within limits (max pixels: {VLM_MAX_IMAGE_SIZE:,}, max DPI: {VLM_MAX_DPI})"
        # )

    return image


def _call_inference_server_vlm_api(
    image: Image.Image,
    prompt: str,
    api_url: str = None,
    model_name: str = None,
    max_new_tokens: int = None,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
    repetition_penalty: float = None,
    timeout: int = None,
    stream: bool = True,
    seed: int = None,
) -> str:
    """
    Calls a inference-server API endpoint with an image and text prompt.

    This function converts a PIL Image to base64 and sends it to the inference-server
    API endpoint using the OpenAI-compatible chat completions format.

    Args:
        image: PIL Image to process
        prompt: Text prompt for the VLM
        api_url: Base URL of the inference-server API (defaults to INFERENCE_SERVER_API_URL from config)
        model_name: Optional model name to use (defaults to INFERENCE_SERVER_MODEL_NAME from config)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for token repetition
        timeout: Request timeout in seconds (defaults to INFERENCE_SERVER_TIMEOUT from config)
        stream: Whether to stream the response
        seed: Random seed for generation

    Returns:
        str: The generated text response from the model

    Raises:
        ConnectionError: If the API request fails
        ValueError: If the response format is invalid
    """
    if api_url is None:
        api_url = INFERENCE_SERVER_API_URL
    if model_name is None:
        model_name = (
            INFERENCE_SERVER_MODEL_NAME if INFERENCE_SERVER_MODEL_NAME else None
        )
    if timeout is None:
        timeout = INFERENCE_SERVER_TIMEOUT

    # Convert PIL Image to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Prepare the request payload in OpenAI-compatible format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    payload = {
        "messages": messages,
        "stream": stream,
    }

    # Add optional parameters if provided
    if model_name:
        payload["model"] = model_name
    if max_new_tokens is not None:
        payload["max_tokens"] = max_new_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if repetition_penalty is not None:
        payload["repeat_penalty"] = repetition_penalty
    if seed is not None:
        payload["seed"] = seed
    endpoint = f"{api_url}/v1/chat/completions"

    try:
        if stream:
            # Handle streaming response
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=timeout,
            )
            response.raise_for_status()

            final_tokens = []

            for line in response.iter_lines():
                if not line:  # Skip empty lines
                    continue

                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                print(token, end="", flush=True)
                                final_tokens.append(token)
                                # output_tokens += 1
                    except json.JSONDecodeError:
                        continue

            print()  # newline after stream finishes

            text = "".join(final_tokens)

            # Estimate input tokens (rough approximation)
            # input_tokens = len(prompt.split())

            # return {
            #     "choices": [
            #         {
            #             "index": 0,
            #             "finish_reason": "stop",
            #             "message": {"role": "assistant", "content": text},
            #         }
            #     ],
            #     "usage": {
            #         "prompt_tokens": input_tokens,
            #         "completion_tokens": output_tokens,
            #         "total_tokens": input_tokens + output_tokens,
            #     },
            # }
            return text

        else:
            # Handle non-streaming response
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            response.raise_for_status()

            result = response.json()

            # Ensure the response has the expected format
            if "choices" not in result or len(result["choices"]) == 0:
                raise ValueError(
                    "Invalid response format from inference-server: no choices found"
                )

            message = result["choices"][0].get("message", {})
            content = message.get("content", "")

            if not content:
                raise ValueError(
                    "Invalid response format from inference-server: no content in message"
                )

            return content

    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"Failed to connect to inference-server at {api_url}: {str(e)}"
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from inference-server: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error calling inference-server API: {str(e)}")


def _vlm_ocr_predict(
    image: Image.Image,
    prompt: str = model_default_prompt,
) -> Dict[str, Any]:
    """
    VLM OCR prediction function that mimics PaddleOCR's interface.

    Args:
        image: PIL Image to process
        prompt: Text prompt for the VLM

    Returns:
        Dictionary in PaddleOCR format with 'rec_texts' and 'rec_scores'
    """
    try:
        # Validate image exists and is not None
        if image is None:
            print("VLM OCR error: Image is None")
            return {"rec_texts": [], "rec_scores": []}

        # Validate image has valid size (at least 10x10 pixels)
        try:
            width, height = image.size
            if width < 10 or height < 10:
                print(
                    f"VLM OCR error: Image is too small ({width}x{height} pixels). Minimum size is 10x10."
                )
                return {"rec_texts": [], "rec_scores": []}
        except Exception as size_error:
            print(f"VLM OCR error: Could not get image size: {size_error}")
            return {"rec_texts": [], "rec_scores": []}

        # Ensure image is in RGB mode (convert if needed)
        try:
            if image.mode != "RGB":
                # print(f"VLM OCR: Converting image from {image.mode} to RGB mode")
                image = image.convert("RGB")
                # Update width/height after conversion (should be same, but ensure consistency)
                width, height = image.size
        except Exception as convert_error:
            print(f"VLM OCR error: Could not convert image to RGB: {convert_error}")
            return {"rec_texts": [], "rec_scores": []}

        # Check and resize image if it exceeds maximum size or DPI limits
        try:
            image = _prepare_image_for_vlm(image)
            width, height = image.size
        except Exception as prep_error:
            print(f"VLM OCR error: Could not prepare image for VLM: {prep_error}")
            return {"rec_texts": [], "rec_scores": []}

        # Use the VLM to extract text
        # Pass None for parameters to prioritize model-specific defaults from run_vlm.py
        # If model defaults are not available, general defaults will be used (matching current values)
        # print(f"Calling extract_text_from_image_vlm with image size: {width}x{height}")
        extracted_text = extract_text_from_image_vlm(
            text=prompt,
            image=image,
            max_new_tokens=HYBRID_OCR_MAX_NEW_TOKENS,  # Use model default if available, otherwise MAX_NEW_TOKENS from config
            temperature=None,  # Use model default if available, otherwise 0.7
            top_p=None,  # Use model default if available, otherwise 0.9
            top_k=None,  # Use model default if available, otherwise 50
            repetition_penalty=None,  # Use model default if available, otherwise 1.3
            presence_penalty=None,  # Use model default if available, otherwise None (only supported by Qwen3-VL models)
        )

        # Check if extracted_text is None or empty
        if extracted_text is None:
            # print("VLM OCR warning: extract_text_from_image_vlm returned None")
            return {"rec_texts": [], "rec_scores": []}

        if not isinstance(extracted_text, str):
            # print(f"VLM OCR warning: extract_text_from_image_vlm returned unexpected type: {type(extracted_text)}")
            return {"rec_texts": [], "rec_scores": []}

        if extracted_text.strip():

            # Clean the text

            cleaned_text = re.sub(r"[\r\n]+", " ", extracted_text)
            cleaned_text = cleaned_text.strip()

            # Split into words for compatibility with PaddleOCR format
            words = cleaned_text.split()

            # If text has more than 30 words, assume something went wrong and skip it
            if len(words) > 30:
                print(
                    f"VLM OCR warning: Extracted text has {len(words)} words, which exceeds the 30 word limit. Skipping."
                )
                return {"rec_texts": [], "rec_scores": []}

            # Create PaddleOCR-compatible result
            result = {
                "rec_texts": words,
                "rec_scores": [1.0] * len(words),  # High confidence for VLM results
            }

            return result
        else:
            # print("VLM OCR warning: Extracted text is empty after stripping")
            return {"rec_texts": [], "rec_scores": []}

    except Exception:
        # print(f"VLM OCR error: {e}")
        # print(f"VLM OCR error traceback: {traceback.format_exc()}")
        return {"rec_texts": [], "rec_scores": []}


def _inference_server_ocr_predict(
    image: Image.Image,
    prompt: str = model_default_prompt,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """
    Inference-server OCR prediction function that mimics PaddleOCR's interface.
    Calls an external inference-server API instead of a local model.

    Args:
        image: PIL Image to process
        prompt: Text prompt for the VLM
        max_retries: Maximum number of retry attempts for API calls (default: 5)

    Returns:
        Dictionary in PaddleOCR format with 'rec_texts' and 'rec_scores'

    Raises:
        Exception: If all retry attempts fail after max_retries attempts
    """
    try:
        # Validate image exists and is not None
        if image is None:
            print("Inference-server OCR error: Image is None")
            return {"rec_texts": [], "rec_scores": []}

        # Validate image has valid size (at least 10x10 pixels)
        try:
            width, height = image.size
            if width < 10 or height < 10:
                print(
                    f"Inference-server OCR error: Image is too small ({width}x{height} pixels). Minimum size is 10x10."
                )
                return {"rec_texts": [], "rec_scores": []}
        except Exception as size_error:
            print(f"Inference-server OCR error: Could not get image size: {size_error}")
            return {"rec_texts": [], "rec_scores": []}

        # Ensure image is in RGB mode (convert if needed)
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
                width, height = image.size
        except Exception as convert_error:
            print(
                f"Inference-server OCR error: Could not convert image to RGB: {convert_error}"
            )
            return {"rec_texts": [], "rec_scores": []}

        # Check and resize image if it exceeds maximum size or DPI limits
        try:
            image = _prepare_image_for_vlm(image)
            width, height = image.size
        except Exception as prep_error:
            print(
                f"Inference-server OCR error: Could not prepare image for VLM: {prep_error}"
            )
            return {"rec_texts": [], "rec_scores": []}

        # Use the inference-server API to extract text with retry logic
        extracted_text = None

        for attempt in range(1, max_retries + 1):
            try:
                extracted_text = _call_inference_server_vlm_api(
                    image=image,
                    prompt=prompt,
                    max_new_tokens=HYBRID_OCR_MAX_NEW_TOKENS,
                    temperature=model_default_temperature,
                    top_p=model_default_top_p,
                    top_k=model_default_top_k,
                    repetition_penalty=model_default_repetition_penalty,
                    seed=int(model_default_seed),
                )
                # If we get here, the API call succeeded
                break
            except Exception as api_error:
                print(
                    f"Inference-server OCR retry attempt {attempt}/{max_retries} failed: {api_error}"
                )
                if attempt == max_retries:
                    # All retries exhausted, raise the exception
                    raise Exception(
                        f"Inference-server OCR failed after {max_retries} attempts. Last error: {str(api_error)}"
                    ) from api_error
                # Continue to next retry attempt

        # Check if extracted_text is None or empty
        if extracted_text is None:
            return {"rec_texts": [], "rec_scores": []}

        if not isinstance(extracted_text, str):
            return {"rec_texts": [], "rec_scores": []}

        if extracted_text.strip():
            # Clean the text
            cleaned_text = re.sub(r"[\r\n]+", " ", extracted_text)
            cleaned_text = cleaned_text.strip()

            # Split into words for compatibility with PaddleOCR format
            words = cleaned_text.split()

            # If text has more than 30 words, assume something went wrong and skip it
            if len(words) > 30:
                print(
                    f"Inference-server OCR warning: Extracted text has {len(words)} words, which exceeds the 30 word limit. Skipping."
                )
                return {"rec_texts": [], "rec_scores": []}

            # Create PaddleOCR-compatible result
            result = {
                "rec_texts": words,
                "rec_scores": [1.0]
                * len(words),  # High confidence for inference-server results
            }

            return result
        else:
            return {"rec_texts": [], "rec_scores": []}

    except Exception as e:
        # Re-raise if it's the retry exhaustion exception
        if "failed after" in str(e) and "attempts" in str(e):
            raise
        # Otherwise, handle other exceptions as before
        print(f"Inference-server OCR error: {e}")
        import traceback

        print(f"Inference-server OCR error traceback: {traceback.format_exc()}")
        return {"rec_texts": [], "rec_scores": []}


def plot_text_bounding_boxes(
    image: Image.Image,
    bounding_boxes: List[Dict],
    image_name: str = "initial_vlm_output_bounding_boxes.png",
    image_folder: str = "inference_server_visualisations",
    output_folder: str = OUTPUT_FOLDER,
):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalised coordinates, and different colors.

    Args:
        image: The PIL Image object.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
        image_name: The name of the image for debugging.
        image_folder: The folder name (relative to output_folder) where the image will be saved.
        output_folder: The folder where the image will be saved.
    """

    # Load the image
    img = image
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    font = ImageFont.load_default()

    # Iterate over the bounding boxes
    for i, bbox_dict in enumerate(ast.literal_eval(bounding_boxes)):
        color = "green"

        # Extract the bounding box coordinates (preserve the original dict for text extraction)
        if "bb" in bbox_dict:
            bbox_coords = bbox_dict["bb"]
        elif "bbox" in bbox_dict:
            bbox_coords = bbox_dict["bbox"]
        elif "bbox_2d" in bbox_dict:
            bbox_coords = bbox_dict["bbox_2d"]
        else:
            # Skip if no valid bbox found
            continue

        # Ensure bbox_coords is a list with 4 elements
        if not isinstance(bbox_coords, list) or len(bbox_coords) != 4:
            # Try to fix malformed bbox
            fixed_bbox = _fix_malformed_bbox(bbox_coords)
            if fixed_bbox is not None:
                bbox_coords = fixed_bbox
            else:
                continue

        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bbox_coords[1] / 999 * height)
        abs_x1 = int(bbox_coords[0] / 999 * width)
        abs_y2 = int(bbox_coords[3] / 999 * height)
        abs_x2 = int(bbox_coords[2] / 999 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=1)

        # Draw the text - extract from the original dictionary, not the coordinates
        text_to_draw = "No text"
        if "text" in bbox_dict:
            text_to_draw = bbox_dict["text"]
        elif "text_content" in bbox_dict:
            text_to_draw = bbox_dict["text_content"]

        draw.text((abs_x1, abs_y2), text_to_draw, fill=color, font=font)

    try:
        debug_dir = os.path.join(
            output_folder,
            image_folder,
        )
        # Security: Validate that the constructed path is safe
        normalized_debug_dir = os.path.normpath(os.path.abspath(debug_dir))
        if not validate_folder_containment(normalized_debug_dir, OUTPUT_FOLDER):
            raise ValueError(
                f"Unsafe image folder path: {debug_dir}. Must be contained within {OUTPUT_FOLDER}"
            )
        os.makedirs(normalized_debug_dir, exist_ok=True)
        # Increment the number at the end of image_name before .png
        # This converts zero-indexed input to one-indexed output
        incremented_image_name = image_name
        if image_name.endswith(".png"):
            # Find the number pattern at the end before .png
            # Matches patterns like: _0.png, _00.png, 0.png, 00.png, etc.
            pattern = r"(\d+)(\.png)$"
            match = re.search(pattern, image_name)
            if match:
                number_str = match.group(1)
                number = int(number_str)
                incremented_number = number + 1
                # Preserve the same number of digits (padding with zeros if needed)
                incremented_str = str(incremented_number).zfill(len(number_str))
                incremented_image_name = re.sub(
                    pattern, lambda m: incremented_str + m.group(2), image_name
                )

        image_name_safe = safe_sanitize_text(incremented_image_name)
        image_name_shortened = image_name_safe[:50]
        filename = f"{image_name_shortened}_initial_bounding_box_output.png"
        filepath = os.path.join(normalized_debug_dir, filename)
        img.save(filepath)
    except Exception as e:
        print(f"Error saving image with bounding boxes: {e}")

    # Display the image
    # img.show()


def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(
                lines[i + 1 :]
            )  # Remove everything before "```json"
            json_output = json_output.split("```")[
                0
            ]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


def _fix_malformed_bbox_in_json_string(json_string):
    """
    Fixes malformed bounding box values in a JSON string before parsing.

    Handles cases like:
    - "bb": "779, 767, 874, 789], "text" (missing opening bracket, missing closing quote)
    - "bb": "[779, 767, 874, 789]" (stringified array)
    - "bb": "779, 767, 874, 789" (no brackets)

    Args:
        json_string: The raw JSON string that may contain malformed bbox values

    Returns:
        str: The JSON string with malformed bbox values fixed
    """
    import re

    # Pattern 1: Match malformed bbox like: "bb": "779, 767, 874, 789], "text"
    # The issue: missing opening bracket, missing closing quote after the bracket
    # Matches: "bb": " followed by numbers, ], then , "
    pattern1 = (
        r'("(?:bb|bbox|bbox_2d)"\s*:\s*)"(\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+)\]\s*,\s*"'
    )

    def fix_bbox_match1(match):
        key_part = match.group(1)  # "bb": "
        bbox_str = match.group(2)  # "779, 767, 874, 789"

        # Format as proper JSON array (no quotes around it)
        fixed_bbox = "[" + bbox_str.strip() + "]"

        # Return the fixed version: "bb": [779, 767, 874, 789], "
        return key_part + fixed_bbox + ', "'

    # Pattern 2: Match malformed bbox like: "bb": "779, 767, 874, 789]"
    # Missing opening bracket, but has closing quote
    pattern2 = r'("(?:bb|bbox|bbox_2d)"\s*:\s*)"(\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+)\]"'

    def fix_bbox_match2(match):
        key_part = match.group(1)
        bbox_str = match.group(2)
        fixed_bbox = "[" + bbox_str.strip() + "]"
        return key_part + fixed_bbox + '"'

    # Pattern 3: Match malformed bbox like: "bb": "779, 767, 874, 789] (end of object, no quote)
    pattern3 = (
        r'("(?:bb|bbox|bbox_2d)"\s*:\s*)"(\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+)\]\s*\}'
    )

    def fix_bbox_match3(match):
        key_part = match.group(1)
        bbox_str = match.group(2)
        fixed_bbox = "[" + bbox_str.strip() + "]"
        return key_part + fixed_bbox + "}"

    # Apply the fixes in order
    fixed_json = re.sub(pattern1, fix_bbox_match1, json_string)
    fixed_json = re.sub(pattern2, fix_bbox_match2, fixed_json)
    fixed_json = re.sub(pattern3, fix_bbox_match3, fixed_json)

    return fixed_json


def _fix_malformed_bbox(bbox):
    """
    Attempts to fix malformed bounding box values.

    Handles cases where bbox is:
    - A string like "779, 767, 874, 789]" (missing opening bracket)
    - A string like "[779, 767, 874, 789]" (should be parsed)
    - A string like "779, 767, 874, 789" (no brackets at all)
    - Already a valid list (returns as-is)

    Args:
        bbox: The bounding box value (could be list, string, or other)

    Returns:
        list: A list of 4 numbers [x1, y1, x2, y2], or None if parsing fails
    """
    # If it's already a valid list, return it
    if isinstance(bbox, list) and len(bbox) == 4:
        return bbox

    # If it's not a string, we can't fix it
    if not isinstance(bbox, str):
        return None

    try:
        # Remove any leading/trailing whitespace
        bbox_str = bbox.strip()

        # Remove quotes if present
        if bbox_str.startswith('"') and bbox_str.endswith('"'):
            bbox_str = bbox_str[1:-1]
        elif bbox_str.startswith("'") and bbox_str.endswith("'"):
            bbox_str = bbox_str[1:-1]

        # Try to extract numbers from various formats
        # Pattern 1: "779, 767, 874, 789]" (missing opening bracket)
        # Pattern 2: "[779, 767, 874, 789]" (has brackets)
        # Pattern 3: "779, 767, 874, 789" (no brackets)

        # Remove brackets if present
        if bbox_str.startswith("["):
            bbox_str = bbox_str[1:]
        if bbox_str.endswith("]"):
            bbox_str = bbox_str[:-1]

        # Split by comma and extract numbers
        parts = [part.strip() for part in bbox_str.split(",")]

        if len(parts) != 4:
            return None

        # Convert each part to float
        coords = []
        for part in parts:
            try:
                coords.append(float(part))
            except (ValueError, TypeError):
                return None

        return coords

    except Exception:
        return None


def _vlm_page_ocr_predict(
    image: Image.Image,
    image_name: str = "vlm_page_ocr_input_image.png",
    normalised_coords_range: Optional[int] = 999,
    output_folder: str = OUTPUT_FOLDER,
) -> Dict[str, List]:
    """
    VLM page-level OCR prediction that returns structured line-level results with bounding boxes.

    Args:
        image: PIL Image to process (full page)
        image_name: Name of the image for debugging
        normalised_coords_range: If set, bounding boxes are assumed to be in normalized coordinates
            from 0 to this value (e.g., 999, default for Qwen3-VL). Coordinates will be rescaled to match the processed image size. If None, coordinates are assumed to be in absolute pixel coordinates.
        output_folder: The folder where output images will be saved
    Returns:
        Dictionary with 'text', 'left', 'top', 'width', 'height', 'conf', 'model' keys
        matching the format expected by perform_ocr
    """
    try:
        # Validate image exists and is not None
        if image is None:
            print("VLM page OCR error: Image is None")
            return {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": [],
                "model": [],
            }

        # Validate image has valid size (at least 10x10 pixels)
        try:
            width, height = image.size
            if width < 10 or height < 10:
                print(
                    f"VLM page OCR error: Image is too small ({width}x{height} pixels). Minimum size is 10x10."
                )
                return {
                    "text": [],
                    "left": [],
                    "top": [],
                    "width": [],
                    "height": [],
                    "conf": [],
                    "model": [],
                }
        except Exception as size_error:
            print(f"VLM page OCR error: Could not get image size: {size_error}")
            return {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": [],
                "model": [],
            }

        # Ensure image is in RGB mode (convert if needed)
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
                width, height = image.size
        except Exception as convert_error:
            print(
                f"VLM page OCR error: Could not convert image to RGB: {convert_error}"
            )
            return {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": [],
                "model": [],
            }

        # Check and resize image if it exceeds maximum size or DPI limits
        scale_x = 1.0
        scale_y = 1.0
        try:
            original_width, original_height = image.size
            processed_image = _prepare_image_for_vlm(image)
            processed_width, processed_height = processed_image.size

            # Use float division to avoid rounding errors
            scale_x = (
                float(original_width) / float(processed_width)
                if processed_width > 0
                else 1.0
            )
            scale_y = (
                float(original_height) / float(processed_height)
                if processed_height > 0
                else 1.0
            )

            # Debug: print scale factors to verify
            if scale_x != 1.0 or scale_y != 1.0:
                print(f"Scale factors: x={scale_x:.6f}, y={scale_y:.6f}")
                print(
                    f"Original: {original_width}x{original_height}, Processed: {processed_width}x{processed_height}"
                )
        except Exception as prep_error:
            print(f"VLM page OCR error: Could not prepare image for VLM: {prep_error}")
            return {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": [],
                "model": [],
            }

        # Save input image for debugging if environment variable is set
        if SAVE_VLM_INPUT_IMAGES:
            try:
                vlm_debug_dir = os.path.join(
                    output_folder,
                    "vlm_visualisations/vlm_input_images",
                )
                os.makedirs(vlm_debug_dir, exist_ok=True)
                # Increment the number at the end of image_name before .png
                # This converts zero-indexed input to one-indexed output
                incremented_image_name = image_name
                if image_name.endswith(".png"):
                    # Find the number pattern at the end before .png
                    # Matches patterns like: _0.png, _00.png, 0.png, 00.png, etc.
                    pattern = r"(\d+)(\.png)$"
                    match = re.search(pattern, image_name)
                    if match:
                        number_str = match.group(1)
                        number = int(number_str)
                        incremented_number = number + 1
                        # Preserve the same number of digits (padding with zeros if needed)
                        incremented_str = str(incremented_number).zfill(len(number_str))
                        incremented_image_name = re.sub(
                            pattern, lambda m: incremented_str + m.group(2), image_name
                        )
                image_name_safe = safe_sanitize_text(incremented_image_name)
                image_name_shortened = image_name_safe[:50]
                filename = f"{image_name_shortened}_vlm_page_input_image.png"
                filepath = os.path.join(vlm_debug_dir, filename)
                processed_image.save(filepath)
                # print(f"Saved VLM input image to: {filepath}")
            except Exception as save_error:
                print(f"Warning: Could not save VLM input image: {save_error}")

        # Create prompt that requests structured JSON output with bounding boxes
        prompt = full_page_ocr_vlm_prompt

        # Use the VLM to extract structured text
        extracted_text = extract_text_from_image_vlm(
            text=prompt,
            image=processed_image,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            presence_penalty=None,
        )

        # Check if extracted_text is None or empty
        if extracted_text is None or not isinstance(extracted_text, str):
            print(
                "VLM page OCR warning: extract_text_from_image_vlm returned None or invalid type"
            )
            return {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": [],
                "model": [],
            }

        # Try to parse JSON from the response
        # The VLM might return JSON wrapped in markdown code blocks or with extra text
        extracted_text = extracted_text.strip()

        # Fix malformed bounding box values in the JSON string before parsing
        # This handles cases like: "bb": "779, 767, 874, 789],
        extracted_text = _fix_malformed_bbox_in_json_string(extracted_text)

        lines_data = None

        # First, try to parse the entire response as JSON
        try:
            lines_data = json.loads(extracted_text)
        except json.JSONDecodeError:
            pass

        # If that fails, try to extract JSON from markdown code blocks
        if lines_data is None:
            json_match = re.search(
                r"```(?:json)?\s*(\[.*?\])", extracted_text, re.DOTALL
            )
            if json_match:
                try:
                    lines_data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

        # If that fails, try to find JSON array in the text (more lenient)
        if lines_data is None:
            # Try to find array starting with [ and ending with ]
            # This is a simple approach - look for balanced brackets
            start_idx = extracted_text.find("[")
            if start_idx >= 0:
                bracket_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(extracted_text)):
                    if extracted_text[i] == "[":
                        bracket_count += 1
                    elif extracted_text[i] == "]":
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i
                            break
                if end_idx > start_idx:
                    try:
                        lines_data = json.loads(extracted_text[start_idx : end_idx + 1])
                    except json.JSONDecodeError:
                        pass

        # If that fails, try parsing multiple JSON arrays (may span multiple lines)
        # This handles cases where the response has multiple JSON arrays separated by newlines
        # Each array might be on a single line or span multiple lines
        if lines_data is None:
            try:
                combined_data = []
                # Find all JSON arrays in the text (they may span multiple lines)
                # This approach handles both single-line and multi-line arrays
                text = extracted_text
                while True:
                    start_idx = text.find("[")
                    if start_idx < 0:
                        break

                    # Find the matching closing bracket
                    bracket_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(text)):
                        if text[i] == "[":
                            bracket_count += 1
                        elif text[i] == "]":
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_idx = i
                                break

                    if end_idx > start_idx:
                        try:
                            array_str = text[start_idx : end_idx + 1]
                            array_data = json.loads(array_str)
                            if isinstance(array_data, list):
                                combined_data.extend(array_data)
                        except json.JSONDecodeError:
                            pass

                    # Move past this array to find the next one
                    text = text[end_idx + 1 :]

                if combined_data:
                    lines_data = combined_data
            except Exception:
                pass

        # Final attempt: try to parse as-is
        if lines_data is None:
            try:
                lines_data = json.loads(extracted_text)
            except json.JSONDecodeError:
                pass

        # If we still couldn't parse JSON, return empty results
        if lines_data is None:
            print("VLM page OCR error: Could not parse JSON response")
            print(
                f"Response text: {extracted_text[:500]}"
            )  # Print first 500 chars for debugging
            return {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": [],
                "model": [],
            }

        # Validate that lines_data is a list
        if not isinstance(lines_data, list):
            print(f"VLM page OCR error: Expected list, got {type(lines_data)}")
            return {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": [],
                "model": [],
            }

        if SAVE_VLM_INPUT_IMAGES:
            plot_text_bounding_boxes(
                processed_image,
                extracted_text,
                image_name=image_name,
                image_folder="vlm_visualisations",
                output_folder=output_folder,
            )

        # Store a copy of the processed image for debug visualization (before rescaling)
        # IMPORTANT: This must be the EXACT same image that was sent to the API
        processed_image_for_debug = (
            processed_image.copy() if SAVE_VLM_INPUT_IMAGES else None
        )

        # Collect all valid bounding boxes before rescaling for debug visualization
        pre_scaled_boxes = []

        # Convert VLM results to expected format
        result = {
            "text": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
            "conf": [],
            "model": [],
        }

        for line_item in lines_data:
            if not isinstance(line_item, dict):
                continue

            # Check for text_content (matching ocr.ipynb) or text field
            text = line_item.get("text_content") or line_item.get("text", "").strip()
            if not text:
                continue

            # Check for bbox_2d format (matching ocr.ipynb) or bbox format
            bbox = (
                line_item.get("bbox_2d")
                or line_item.get("bbox", [])
                or line_item.get("bb", [])
            )
            confidence = line_item.get(
                "confidence", 100
            )  # Default to 100 if not provided

            # Attempt to fix malformed bounding boxes (e.g., string instead of array)
            fixed_bbox = _fix_malformed_bbox(bbox)
            if fixed_bbox is not None:
                if not isinstance(bbox, list) or len(bbox) != 4:
                    print(
                        f"VLM page OCR: Fixed malformed bbox for line '{text[:50]}': {bbox} -> {fixed_bbox}"
                    )
                bbox = fixed_bbox
            elif not isinstance(bbox, list) or len(bbox) != 4:
                print(
                    f"VLM page OCR warning: Invalid bbox format for line '{text[:50]}': {bbox}"
                )
                continue

            # Handle bbox_2d format [x1, y1, x2, y2] (matching ocr.ipynb) or bbox format [x1, y1, x2, y2]
            # ocr.ipynb uses bbox_2d with format [x1, y1, x2, y2] - same as standard bbox format
            # Both formats use [x1, y1, x2, y2] order
            x1, y1, x2, y2 = bbox

            # Ensure coordinates are valid numbers
            try:
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
            except (ValueError, TypeError):
                print(
                    f"VLM page OCR warning: Invalid bbox coordinates for line '{text[:50]}': {bbox}"
                )
                continue

            # Ensure x2 > x1 and y2 > y1
            if x2 <= x1 or y2 <= y1:
                print(
                    f"VLM page OCR warning: Invalid bbox dimensions for line '{text[:50]}': {bbox}"
                )
                continue

            # If coordinates are normalized (0 to normalised_coords_range), rescale directly to processed image dimensions
            # This matches the ocr.ipynb approach: direct normalization to image size using /999 * dimension
            # ocr.ipynb uses: abs_x1 = int(bounding_box["bbox_2d"][0]/999 * width)
            #                  abs_y1 = int(bounding_box["bbox_2d"][1]/999 * height)
            if normalised_coords_range is not None and normalised_coords_range > 0:
                # Direct normalization: match ocr.ipynb approach exactly
                # Formula: (coord / normalised_coords_range) * image_dimension
                # Note: ocr.ipynb uses 999, but we allow configurable range
                x1 = (x1 / float(normalised_coords_range)) * processed_width
                y1 = (y1 / float(normalised_coords_range)) * processed_height
                x2 = (x2 / float(normalised_coords_range)) * processed_width
                y2 = (y2 / float(normalised_coords_range)) * processed_height

            # Store bounding box after normalization (if applied) but before rescaling to original image space
            if processed_image_for_debug is not None:
                pre_scaled_boxes.append({"bbox": (x1, y1, x2, y2), "text": text})

            # Step 3: Scale coordinates back to original image space if image was resized
            if scale_x != 1.0 or scale_y != 1.0:
                x1 = x1 * scale_x
                y1 = y1 * scale_y
                x2 = x2 * scale_x
                y2 = y2 * scale_y

            # Convert from (x1, y1, x2, y2) to (left, top, width, height)
            left = int(round(x1))
            top = int(round(y1))
            width = int(round(x2 - x1))
            height = int(round(y2 - y1))

            # Ensure confidence is in valid range (0-100)
            try:
                confidence = float(confidence)
                confidence = max(0, min(100, confidence))  # Clamp to 0-100
            except (ValueError, TypeError):
                confidence = 100  # Default if invalid

            result["text"].append(clean_unicode_text(text))
            result["left"].append(left)
            result["top"].append(top)
            result["width"].append(width)
            result["height"].append(height)
            result["conf"].append(int(round(confidence)))
            result["model"].append("VLM")

        return result

    except Exception as e:
        print(f"VLM page OCR error: {e}")
        import traceback

        print(f"VLM page OCR error traceback: {traceback.format_exc()}")
        return {
            "text": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
            "conf": [],
            "model": [],
        }


def _inference_server_page_ocr_predict(
    image: Image.Image,
    image_name: str = "inference_server_page_ocr_input_image.png",
    normalised_coords_range: Optional[int] = 999,
    output_folder: str = OUTPUT_FOLDER,
) -> Dict[str, List]:
    """
    Inference-server page-level OCR prediction that returns structured line-level results with bounding boxes.
    Calls an external inference-server API instead of a local model.

    Args:
        image: PIL Image to process (full page)
        image_name: Name of the image for debugging
        normalised_coords_range: If set, bounding boxes are assumed to be in normalized coordinates
            from 0 to this value (e.g., 999, default for Qwen3-VL). Coordinates will be rescaled to match the processed image size. If None, coordinates are assumed to be in absolute pixel coordinates.
        output_folder: The folder where output images will be saved
    Returns:
        Dictionary with 'text', 'left', 'top', 'width', 'height', 'conf', 'model' keys
        matching the format expected by perform_ocr
    """
    try:
        # Validate image exists and is not None
        if image is None:
            print("Inference-server page OCR error: Image is None")
            return {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": [],
                "model": [],
            }

        # Validate image has valid size (at least 10x10 pixels)
        try:
            width, height = image.size
            if width < 10 or height < 10:
                print(
                    f"Inference-server page OCR error: Image is too small ({width}x{height} pixels). Minimum size is 10x10."
                )
                return {
                    "text": [],
                    "left": [],
                    "top": [],
                    "width": [],
                    "height": [],
                    "conf": [],
                    "model": [],
                }
        except Exception as size_error:
            print(
                f"Inference-server page OCR error: Could not get image size: {size_error}"
            )
            return {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": [],
                "model": [],
            }

        # Ensure image is in RGB mode (convert if needed)
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
                width, height = image.size
        except Exception as convert_error:
            print(
                f"Inference-server page OCR error: Could not convert image to RGB: {convert_error}"
            )
            return {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": [],
                "model": [],
            }

        # Check and resize image if it exceeds maximum size or DPI limits
        scale_x = 1.0
        scale_y = 1.0
        # In _inference_server_page_ocr_predict, around line 1465-1471:
        try:
            original_width, original_height = image.size
            processed_image = _prepare_image_for_vlm(image)
            processed_width, processed_height = processed_image.size

            # Use float division to avoid rounding errors
            scale_x = (
                float(original_width) / float(processed_width)
                if processed_width > 0
                else 1.0
            )
            scale_y = (
                float(original_height) / float(processed_height)
                if processed_height > 0
                else 1.0
            )

            # Debug: print scale factors to verify
            if scale_x != 1.0 or scale_y != 1.0:
                print(f"Scale factors: x={scale_x:.6f}, y={scale_y:.6f}")
                print(
                    f"Original: {original_width}x{original_height}, Processed: {processed_width}x{processed_height}"
                )
        except Exception as prep_error:
            print(
                f"Inference-server page OCR error: Could not prepare image for VLM: {prep_error}"
            )
            return {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": [],
                "model": [],
            }

        # Save input image for debugging if environment variable is set
        if SAVE_VLM_INPUT_IMAGES:
            try:
                vlm_debug_dir = os.path.join(
                    output_folder,
                    "inference_server_visualisations/vlm_input_images",
                )
                os.makedirs(vlm_debug_dir, exist_ok=True)
                # Increment the number at the end of image_name before .png
                # This converts zero-indexed input to one-indexed output
                incremented_image_name = image_name
                if image_name.endswith(".png"):
                    # Find the number pattern at the end before .png
                    # Matches patterns like: _0.png, _00.png, 0.png, 00.png, etc.
                    pattern = r"(\d+)(\.png)$"
                    match = re.search(pattern, image_name)
                    if match:
                        number_str = match.group(1)
                        number = int(number_str)
                        incremented_number = number + 1
                        # Preserve the same number of digits (padding with zeros if needed)
                        incremented_str = str(incremented_number).zfill(len(number_str))
                        incremented_image_name = re.sub(
                            pattern, lambda m: incremented_str + m.group(2), image_name
                        )
                image_name_safe = safe_sanitize_text(incremented_image_name)
                image_name_shortened = image_name_safe[:50]
                filename = (
                    f"{image_name_shortened}_inference_server_page_input_image.png"
                )
                filepath = os.path.join(vlm_debug_dir, filename)
                print(f"Saving inference-server input image to: {filepath}")
                processed_image.save(filepath)
                # print(f"Saved VLM input image to: {filepath}")
            except Exception as save_error:
                print(f"Warning: Could not save VLM input image: {save_error}")

        # Create prompt that requests structured JSON output with bounding boxes
        prompt = full_page_ocr_vlm_prompt

        # Get processed image dimensions for normalization
        # PIL Image.size returns (width, height), not (height, width)
        processed_width, processed_height = processed_image.size

        # Use the inference-server API to extract structured text
        extracted_text = _call_inference_server_vlm_api(
            image=processed_image,
            prompt=prompt,
            max_new_tokens=model_default_max_new_tokens,
            temperature=model_default_temperature,
            top_p=model_default_top_p,
            top_k=model_default_top_k,
            repetition_penalty=model_default_repetition_penalty,
            seed=model_default_seed,
        )

        # Check if extracted_text is None or empty
        if extracted_text is None or not isinstance(extracted_text, str):
            print(
                "Inference-server page OCR warning: API returned None or invalid type"
            )
            return {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": [],
                "model": [],
            }

        # Try to parse JSON from the response
        # The API might return JSON wrapped in markdown code blocks or with extra text
        extracted_text = extracted_text.strip()

        # Fix malformed bounding box values in the JSON string before parsing
        # This handles cases like: "bb": "779, 767, 874, 789],
        extracted_text = _fix_malformed_bbox_in_json_string(extracted_text)

        lines_data = None

        # First, try to parse the entire response as JSON
        try:
            lines_data = json.loads(extracted_text)
        except json.JSONDecodeError:
            pass

        # If that fails, try to extract JSON from markdown code blocks
        if lines_data is None:
            json_match = re.search(
                r"```(?:json)?\s*(\[.*?\])", extracted_text, re.DOTALL
            )
            if json_match:
                try:
                    lines_data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

        # If that fails, try to find JSON array in the text (more lenient)
        if lines_data is None:
            # Try to find array starting with [ and ending with ]
            start_idx = extracted_text.find("[")
            if start_idx >= 0:
                bracket_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(extracted_text)):
                    if extracted_text[i] == "[":
                        bracket_count += 1
                    elif extracted_text[i] == "]":
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i
                            break
                if end_idx > start_idx:
                    try:
                        lines_data = json.loads(extracted_text[start_idx : end_idx + 1])
                    except json.JSONDecodeError:
                        pass

        # If that fails, try parsing multiple JSON arrays (may span multiple lines)
        # This handles cases where the response has multiple JSON arrays separated by newlines
        # Each array might be on a single line or span multiple lines
        if lines_data is None:
            try:
                combined_data = []
                # Find all JSON arrays in the text (they may span multiple lines)
                # This approach handles both single-line and multi-line arrays
                text = extracted_text
                while True:
                    start_idx = text.find("[")
                    if start_idx < 0:
                        break

                    # Find the matching closing bracket
                    bracket_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(text)):
                        if text[i] == "[":
                            bracket_count += 1
                        elif text[i] == "]":
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_idx = i
                                break

                    if end_idx > start_idx:
                        try:
                            array_str = text[start_idx : end_idx + 1]
                            array_data = json.loads(array_str)
                            if isinstance(array_data, list):
                                combined_data.extend(array_data)
                        except json.JSONDecodeError:
                            pass

                    # Move past this array to find the next one
                    text = text[end_idx + 1 :]

                if combined_data:
                    lines_data = combined_data
            except Exception:
                pass

        # Final attempt: try to parse as-is
        if lines_data is None:
            try:
                lines_data = json.loads(extracted_text)
            except json.JSONDecodeError:
                pass

        # If we still couldn't parse JSON, return empty results
        if lines_data is None:
            print("Inference-server page OCR error: Could not parse JSON response")
            print(
                f"Response text: {extracted_text[:500]}"
            )  # Print first 500 chars for debugging
            return {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": [],
                "model": [],
            }

        # Validate that lines_data is a list
        if not isinstance(lines_data, list):
            print(
                f"Inference-server page OCR error: Expected list, got {type(lines_data)}"
            )
            return {
                "text": [],
                "left": [],
                "top": [],
                "width": [],
                "height": [],
                "conf": [],
                "model": [],
            }

        if SAVE_VLM_INPUT_IMAGES:
            plot_text_bounding_boxes(
                processed_image,
                extracted_text,
                image_name=image_name,
                image_folder="inference_server_visualisations",
                output_folder=output_folder,
            )

        # Store a copy of the processed image for debug visualization (before rescaling)
        # IMPORTANT: This must be the EXACT same image that was sent to the API
        processed_image_for_debug = (
            processed_image.copy() if SAVE_VLM_INPUT_IMAGES else None
        )

        # Collect all valid bounding boxes before rescaling for debug visualization
        pre_scaled_boxes = []

        # Convert API results to expected format
        result = {
            "text": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
            "conf": [],
            "model": [],
        }

        for line_item in lines_data:
            if not isinstance(line_item, dict):
                continue

            # Check for text_content (matching ocr.ipynb) or text field
            text = line_item.get("text_content") or line_item.get("text", "").strip()
            if not text:
                continue

            # Check for bbox_2d format (matching ocr.ipynb) or bbox format
            bbox = (
                line_item.get("bbox_2d")
                or line_item.get("bbox", [])
                or line_item.get("bb", [])
            )
            confidence = line_item.get(
                "confidence", 100
            )  # Default to 100 if not provided

            # Attempt to fix malformed bounding boxes (e.g., string instead of array)
            fixed_bbox = _fix_malformed_bbox(bbox)
            if fixed_bbox is not None:
                if not isinstance(bbox, list) or len(bbox) != 4:
                    print(
                        f"Inference-server page OCR: Fixed malformed bbox for line '{text[:50]}': {bbox} -> {fixed_bbox}"
                    )
                bbox = fixed_bbox
            elif not isinstance(bbox, list) or len(bbox) != 4:
                print(
                    f"Inference-server page OCR warning: Invalid bbox format for line '{text[:50]}': {bbox}"
                )
                continue

            # Handle bbox_2d format [x1, y1, x2, y2] (matching ocr.ipynb) or bbox format [x1, y1, x2, y2]
            # ocr.ipynb uses bbox_2d with format [x1, y1, x2, y2] - same as standard bbox format
            # Both formats use [x1, y1, x2, y2] order
            x1, y1, x2, y2 = bbox

            # Ensure coordinates are valid numbers
            try:
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
            except (ValueError, TypeError):
                print(
                    f"Inference-server page OCR warning: Invalid bbox coordinates for line '{text[:50]}': {bbox}"
                )
                continue

            # Ensure x2 > x1 and y2 > y1
            if x2 <= x1 or y2 <= y1:
                print(
                    f"Inference-server page OCR warning: Invalid bbox dimensions for line '{text[:50]}': {bbox}"
                )
                continue

            # If coordinates are normalized (0 to normalised_coords_range), rescale directly to processed image dimensions
            # This matches the Qwen 3-VL approach: direct normalization to image size using /999 * dimension
            if normalised_coords_range is not None and normalised_coords_range > 0:
                # Direct normalization: match ocr.ipynb approach exactly
                # Formula: (coord / normalised_coords_range) * image_dimension
                # Note: Qwen 3-VL uses 999, but we allow configurable range
                x1 = (x1 / float(normalised_coords_range)) * processed_width
                y1 = (y1 / float(normalised_coords_range)) * processed_height
                x2 = (x2 / float(normalised_coords_range)) * processed_width
                y2 = (y2 / float(normalised_coords_range)) * processed_height

            # Store bounding box after normalization (if applied) but before rescaling to original image space
            if processed_image_for_debug is not None:
                pre_scaled_boxes.append({"bbox": (x1, y1, x2, y2), "text": text})

            # Step 3: Scale coordinates back to original image space if image was resized
            if scale_x != 1.0 or scale_y != 1.0:
                x1 = x1 * scale_x
                y1 = y1 * scale_y
                x2 = x2 * scale_x
                y2 = y2 * scale_y

            # Convert from (x1, y1, x2, y2) to (left, top, width, height)
            left = int(round(x1))
            top = int(round(y1))
            width = int(round(x2 - x1))
            height = int(round(y2 - y1))

            # Ensure confidence is in valid range (0-100)
            try:
                confidence = float(confidence)
                confidence = max(0, min(100, confidence))  # Clamp to 0-100
            except (ValueError, TypeError):
                confidence = 50  # Default if invalid

            result["text"].append(clean_unicode_text(text))
            result["left"].append(left)
            result["top"].append(top)
            result["width"].append(width)
            result["height"].append(height)
            result["conf"].append(int(round(confidence)))
            result["model"].append("Inference server")

        return result

    except Exception as e:
        print(f"Inference-server page OCR error: {e}")
        import traceback

        print(f"Inference-server page OCR error traceback: {traceback.format_exc()}")
        return {
            "text": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
            "conf": [],
            "model": [],
        }


class CustomImageAnalyzerEngine:
    def __init__(
        self,
        analyzer_engine: Optional[AnalyzerEngine] = None,
        ocr_engine: str = "tesseract",
        tesseract_config: Optional[str] = None,
        paddle_kwargs: Optional[Dict[str, Any]] = None,
        image_preprocessor: Optional[ImagePreprocessor] = None,
        language: Optional[str] = DEFAULT_LANGUAGE,
        output_folder: str = OUTPUT_FOLDER,
    ):
        """
        Initializes the CustomImageAnalyzerEngine.

        :param ocr_engine: The OCR engine to use ("tesseract", "paddle", "vlm", "hybrid-paddle", "hybrid-vlm", "hybrid-paddle-vlm", "hybrid-paddle-inference-server", or "inference-server").
        :param analyzer_engine: The Presidio AnalyzerEngine instance.
        :param tesseract_config: Configuration string for Tesseract. If None, uses TESSERACT_SEGMENTATION_LEVEL config.
        :param paddle_kwargs: Dictionary of keyword arguments for PaddleOCR constructor.
        :param image_preprocessor: Optional image preprocessor.
        :param language: Preferred OCR language (e.g., "en", "fr", "de"). Defaults to DEFAULT_LANGUAGE.
        :param output_folder: The folder to save the output images to.
        """
        if ocr_engine not in LOCAL_OCR_MODEL_OPTIONS:
            raise ValueError(
                f"ocr_engine must be one of the following: {LOCAL_OCR_MODEL_OPTIONS}"
            )

        self.ocr_engine = ocr_engine

        # Language setup
        self.language = language or DEFAULT_LANGUAGE or "en"
        self.tesseract_lang = _tesseract_lang_code(self.language)
        self.paddle_lang = _paddle_lang_code(self.language)

        # Security: Validate and normalize output_folder at construction time
        # This ensures the object is always in a secure state and prevents
        # any future code from accidentally using an untrusted directory
        normalized_output_folder = os.path.normpath(os.path.abspath(output_folder))
        if not validate_folder_containment(normalized_output_folder, OUTPUT_FOLDER):
            raise ValueError(
                f"Unsafe output folder path: {output_folder}. Must be contained within {OUTPUT_FOLDER}"
            )
        self.output_folder = normalized_output_folder

        if (
            self.ocr_engine == "paddle"
            or self.ocr_engine == "hybrid-paddle"
            or self.ocr_engine == "hybrid-paddle-vlm"
            or self.ocr_engine == "hybrid-paddle-inference-server"
        ):
            # Set PaddleOCR environment variables BEFORE importing PaddleOCR
            # This ensures fonts are configured before the package loads

            # Set PaddleOCR model directory environment variable (only if specified).
            if PADDLE_MODEL_PATH and PADDLE_MODEL_PATH.strip():
                os.environ["PADDLEOCR_MODEL_DIR"] = PADDLE_MODEL_PATH
                print(f"Setting PaddleOCR model path to: {PADDLE_MODEL_PATH}")
            else:
                print("Using default PaddleOCR model storage location")

            # Set PaddleOCR font path to use system fonts instead of downloading simfang.ttf/PingFang-SC-Regular.ttf
            # This MUST be set before importing PaddleOCR to prevent font downloads
            if (
                PADDLE_FONT_PATH
                and PADDLE_FONT_PATH.strip()
                and os.path.exists(PADDLE_FONT_PATH)
            ):
                os.environ["PADDLE_PDX_LOCAL_FONT_FILE_PATH"] = PADDLE_FONT_PATH
                print(
                    f"Setting PaddleOCR font path to configured font: {PADDLE_FONT_PATH}"
                )
            else:
                system_font_path = get_system_font_path()
                if system_font_path:
                    os.environ["PADDLE_PDX_LOCAL_FONT_FILE_PATH"] = system_font_path
                    print(
                        f"Setting PaddleOCR font path to system font: {system_font_path}"
                    )
                else:
                    print(
                        "Warning: No suitable system font found. PaddleOCR may download default fonts."
                    )

            try:
                from paddleocr import PaddleOCR
            except Exception as e:
                raise ImportError(
                    f"Error importing PaddleOCR: {e}. Please install it using 'pip install paddleocr paddlepaddle' in your python environment and retry."
                )

            # Default paddle configuration if none provided
            if paddle_kwargs is None:
                paddle_kwargs = {
                    "det_db_unclip_ratio": PADDLE_DET_DB_UNCLIP_RATIO,
                    "use_textline_orientation": PADDLE_USE_TEXTLINE_ORIENTATION,
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "lang": self.paddle_lang,
                }
            else:
                # Enforce language if not explicitly provided
                paddle_kwargs.setdefault("lang", self.paddle_lang)

            try:
                self.paddle_ocr = PaddleOCR(**paddle_kwargs)
            except Exception as e:
                # Handle DLL loading errors (common on Windows with GPU version)
                if (
                    "WinError 127" in str(e)
                    or "could not be found" in str(e).lower()
                    or "dll" in str(e).lower()
                ):
                    print(
                        f"Warning: GPU initialization failed (likely missing CUDA/cuDNN dependencies): {e}"
                    )
                    print("PaddleOCR will not be available. To fix GPU issues:")
                    print("1. Install Visual C++ Redistributables (latest version)")
                    print("2. Ensure CUDA runtime libraries are in your PATH")
                    print(
                        "3. Or reinstall paddlepaddle CPU version: pip install paddlepaddle"
                    )
                    raise ImportError(
                        f"Error initializing PaddleOCR: {e}. Please install it using 'pip install paddleocr paddlepaddle' in your python environment and retry."
                    )
                else:
                    raise e

        elif self.ocr_engine == "hybrid-vlm":
            # VLM-based hybrid OCR - no additional initialization needed
            # The VLM model is loaded when run_vlm.py is imported
            print(f"Initializing hybrid VLM OCR with model: {SELECTED_MODEL}")
            self.paddle_ocr = None  # Not using PaddleOCR

        elif self.ocr_engine == "vlm":
            # VLM page-level OCR - no additional initialization needed
            # The VLM model is loaded when run_vlm.py is imported
            print(f"Initializing VLM OCR with model: {SELECTED_MODEL}")
            self.paddle_ocr = None  # Not using PaddleOCR

        if self.ocr_engine == "hybrid-paddle-vlm":
            # Hybrid PaddleOCR + VLM - requires both PaddleOCR and VLM
            # The VLM model is loaded when run_vlm.py is imported
            print(
                f"Initializing hybrid PaddleOCR + VLM OCR with model: {SELECTED_MODEL}"
            )

        if self.ocr_engine == "hybrid-paddle-inference-server":
            # Hybrid PaddleOCR + Inference-server - requires both PaddleOCR and inference-server API
            print("Initializing hybrid PaddleOCR + Inference-server OCR")

        if not analyzer_engine:
            analyzer_engine = AnalyzerEngine()
        self.analyzer_engine = analyzer_engine

        # Set Tesseract configuration based on segmentation level
        if tesseract_config:
            self.tesseract_config = tesseract_config
        else:
            # Following function does not actually work correctly, so always use PSM 11
            psm_value = TESSERACT_SEGMENTATION_LEVEL  # _get_tesseract_psm(TESSERACT_SEGMENTATION_LEVEL)
            self.tesseract_config = f"--oem 3 --psm {psm_value}"
            # print(
            #     f"Tesseract configured for {TESSERACT_SEGMENTATION_LEVEL}-level segmentation (PSM {psm_value})"
            # )

        if not image_preprocessor:
            image_preprocessor = ContrastSegmentedImageEnhancer()
        self.image_preprocessor = image_preprocessor

    def _sanitize_filename(
        self, text: str, max_length: int = 20, fallback_prefix: str = "unknown_text"
    ) -> str:
        """
        Sanitizes text for use in filenames by removing invalid characters and limiting length.

        :param text: The text to sanitize
        :param max_length: Maximum length of the sanitized text
        :param fallback_prefix: Prefix to use if sanitization fails
        :return: Sanitized text safe for filenames
        """

        # Remove or replace invalid filename characters
        # Windows: < > : " | ? * \ /
        # Unix: / (forward slash)

        sanitized = safe_sanitize_text(text)

        # Remove leading/trailing underscores and spaces
        sanitized = sanitized.strip("_ ")

        # If empty after sanitization, use a default value
        if not sanitized:
            sanitized = fallback_prefix

        # Limit to max_length characters
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            # Ensure we don't end with an underscore if we cut in the middle
            sanitized = sanitized.rstrip("_")

        # Final check: if still empty or too short, use fallback
        if not sanitized or len(sanitized) < 3:
            sanitized = fallback_prefix

        return sanitized

    def _create_safe_filename_with_confidence(
        self,
        original_text: str,
        new_text: str,
        conf: int,
        new_conf: int,
        ocr_type: str = "OCR",
    ) -> str:
        """
        Creates a safe filename using confidence values when text sanitization fails.

        Args:
            original_text: Original text from Tesseract
            new_text: New text from VLM/PaddleOCR
            conf: Original confidence score
            new_conf: New confidence score
            ocr_type: Type of OCR used (VLM, Paddle, etc.)

        Returns:
            Safe filename string
        """
        # Try to sanitize both texts
        safe_original = self._sanitize_filename(
            original_text, max_length=15, fallback_prefix=f"orig_conf_{conf}"
        )
        safe_new = self._sanitize_filename(
            new_text, max_length=15, fallback_prefix=f"new_conf_{new_conf}"
        )

        # If both sanitizations resulted in fallback names, create a confidence-based name
        if safe_original.startswith("unknown_text") and safe_new.startswith(
            "unknown_text"
        ):
            return f"{ocr_type}_conf_{conf}_to_conf_{new_conf}"

        return f"{safe_original}_conf_{conf}_to_{safe_new}_conf_{new_conf}"

    def _is_line_level_data(self, ocr_data: Dict[str, List]) -> bool:
        """
        Determines if OCR data contains line-level results (multiple words per bounding box).

        Args:
            ocr_data: Dictionary with OCR data

        Returns:
            True if data appears to be line-level, False otherwise
        """
        if not ocr_data or not ocr_data.get("text"):
            return False

        # Check if any text entries contain multiple words
        for text in ocr_data["text"]:
            if text.strip() and len(text.split()) > 1:
                return True

        return False

    def _convert_paddle_to_tesseract_format(
        self,
        paddle_results: List[Any],
        input_image_width: int = None,
        input_image_height: int = None,
        image_name: str = None,
        image: Image.Image = None,
    ) -> Dict[str, List]:
        """Converts PaddleOCR result format to Tesseract's dictionary format using relative coordinates.

        This function uses a safer approach: converts PaddleOCR coordinates to relative (0-1) coordinates
        based on whatever coordinate space PaddleOCR uses, then scales them to the input image dimensions.
        This avoids issues with PaddleOCR's internal image resizing.

        Args:
            paddle_results: List of PaddleOCR result dictionaries
            input_image_width: Width of the input image passed to PaddleOCR (target dimensions for scaling)
            input_image_height: Height of the input image passed to PaddleOCR (target dimensions for scaling)
            image_name: Name of the image
            image: Image object
        """

        output = {
            "text": list(),
            "left": list(),
            "top": list(),
            "width": list(),
            "height": list(),
            "conf": list(),
            "model": list(),
        }

        # paddle_results is now a list of dictionaries with detailed information
        if not paddle_results:
            return output

        # Validate that we have target dimensions
        if input_image_width is None or input_image_height is None:
            print(
                "Warning: Input image dimensions not provided. PaddleOCR coordinates may be incorrectly scaled."
            )
            # Fallback: we'll try to detect from coordinates, but this is less reliable
            use_relative_coords = False
        else:
            use_relative_coords = True

        for page_result in paddle_results:
            # Extract text recognition results from the new format
            rec_texts = page_result.get("rec_texts", list())
            rec_scores = page_result.get("rec_scores", list())
            rec_polys = page_result.get("rec_polys", list())
            rec_models = page_result.get("rec_models", list())

            # PaddleOCR may return image dimensions in the result - check for them
            # Some versions of PaddleOCR include this information
            result_image_width = page_result.get("image_width")
            result_image_height = page_result.get("image_height")

            # PaddleOCR typically returns coordinates in the input image space
            # However, it may internally resize images, so we need to check if coordinates
            # are in a different space by comparing with explicit metadata or detecting from coordinates

            # First pass: determine PaddleOCR's coordinate space by finding max coordinates
            # This tells us what coordinate space PaddleOCR is actually using
            max_x_coord = 0
            max_y_coord = 0

            for bounding_box in rec_polys:
                if hasattr(bounding_box, "tolist"):
                    box = bounding_box.tolist()
                else:
                    box = bounding_box

                if box and len(box) > 0:
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]
                    max_x_coord = max(max_x_coord, max(x_coords) if x_coords else 0)
                    max_y_coord = max(max_y_coord, max(y_coords) if y_coords else 0)

            # Determine PaddleOCR's coordinate space dimensions
            # Priority: explicit result metadata > input dimensions (standard PaddleOCR behavior)
            # Note: PaddleOCR typically returns coordinates in the input image space.
            # We only use a different coordinate space if PaddleOCR provides explicit metadata.
            # Using max coordinates to detect coordinate space is unreliable because:
            # 1. Text might not extend to image edges
            # 2. There might be padding
            # 3. Max coordinates don't necessarily equal image dimensions
            if result_image_width is not None and result_image_height is not None:
                # Use explicit metadata from PaddleOCR if available (most reliable)
                paddle_coord_width = result_image_width
                paddle_coord_height = result_image_height
                # Only use relative conversion if coordinate space differs from input
                if (
                    paddle_coord_width != input_image_width
                    or paddle_coord_height != input_image_height
                ):
                    print(
                        f"PaddleOCR metadata indicates coordinate space ({paddle_coord_width}x{paddle_coord_height}) "
                        f"differs from input ({input_image_width}x{input_image_height}). "
                        f"Using metadata for coordinate conversion."
                    )
            elif input_image_width is not None and input_image_height is not None:
                # Default: assume coordinates are in input image space (standard PaddleOCR behavior)
                # This is the most common case and avoids incorrect scaling
                paddle_coord_width = input_image_width
                paddle_coord_height = input_image_height
            else:
                # Fallback: use max coordinates if we have no other information
                paddle_coord_width = max_x_coord if max_x_coord > 0 else 1
                paddle_coord_height = max_y_coord if max_y_coord > 0 else 1
                use_relative_coords = False
                print(
                    f"Warning: No input dimensions provided. Using detected coordinate space ({paddle_coord_width}x{paddle_coord_height}) from max coordinates."
                )

            # Validate coordinate space dimensions
            if paddle_coord_width is None or paddle_coord_height is None:
                paddle_coord_width = input_image_width or 1
                paddle_coord_height = input_image_height or 1
                use_relative_coords = False

            if paddle_coord_width <= 0 or paddle_coord_height <= 0:
                print(
                    f"Warning: Invalid PaddleOCR coordinate space dimensions ({paddle_coord_width}x{paddle_coord_height}). Using input dimensions."
                )
                paddle_coord_width = input_image_width or 1
                paddle_coord_height = input_image_height or 1
                use_relative_coords = False

            # If coordinate space matches input dimensions, coordinates are already in the correct space
            # Only use relative coordinate conversion if coordinate space differs from input
            if (
                paddle_coord_width == input_image_width
                and paddle_coord_height == input_image_height
                and input_image_width is not None
                and input_image_height is not None
            ):
                # Coordinates are already in input space, no conversion needed
                use_relative_coords = False
                print(
                    f"PaddleOCR coordinates are in input image space ({input_image_width}x{input_image_height}). "
                    f"Using coordinates directly without conversion."
                )

            # Second pass: convert coordinates using relative coordinate approach
            # Use default "Paddle" if rec_models is not available or doesn't match length
            if len(rec_models) != len(rec_texts):
                print(
                    f"Warning: rec_models length ({len(rec_models)}) doesn't match rec_texts length ({len(rec_texts)}). Using default 'Paddle' for all."
                )
                rec_models = ["Paddle"] * len(rec_texts)
                # Update page_result to keep it consistent
                page_result["rec_models"] = rec_models
            else:
                # Ensure we're using the rec_models from page_result (which may have been modified)
                rec_models = page_result.get("rec_models", rec_models)

            # Debug: Print model distribution
            vlm_count = sum(1 for m in rec_models if m == "VLM")
            if vlm_count > 0:
                print(
                    f"Found {vlm_count} VLM-labeled lines out of {len(rec_models)} total lines in page_result"
                )

            for line_text, line_confidence, bounding_box, line_model in zip(
                rec_texts, rec_scores, rec_polys, rec_models
            ):
                # bounding_box is now a numpy array with shape (4, 2)
                # Convert to list of coordinates if it's a numpy array
                if hasattr(bounding_box, "tolist"):
                    box = bounding_box.tolist()
                else:
                    box = bounding_box

                if not box or len(box) == 0:
                    continue

                # box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]

                # Extract bounding box coordinates in PaddleOCR's coordinate space
                line_left_paddle = float(min(x_coords))
                line_top_paddle = float(min(y_coords))
                line_right_paddle = float(max(x_coords))
                line_bottom_paddle = float(max(y_coords))
                line_width_paddle = line_right_paddle - line_left_paddle
                line_height_paddle = line_bottom_paddle - line_top_paddle

                # Convert to relative coordinates (0-1) based on PaddleOCR's coordinate space
                # Then scale to input image dimensions
                if (
                    use_relative_coords
                    and paddle_coord_width > 0
                    and paddle_coord_height > 0
                ):
                    # Normalize to relative coordinates [0-1]
                    rel_left = line_left_paddle / paddle_coord_width
                    rel_top = line_top_paddle / paddle_coord_height
                    rel_width = line_width_paddle / paddle_coord_width
                    rel_height = line_height_paddle / paddle_coord_height

                    # Scale to input image dimensions
                    line_left = rel_left * input_image_width
                    line_top = rel_top * input_image_height
                    line_width = rel_width * input_image_width
                    line_height = rel_height * input_image_height
                else:
                    # Fallback: use coordinates directly (may cause issues if coordinate spaces don't match)
                    line_left = line_left_paddle
                    line_top = line_top_paddle
                    line_width = line_width_paddle
                    line_height = line_height_paddle
                    # if input_image_width and input_image_height:
                    #     print(f"Warning: Using PaddleOCR coordinates directly. This may cause scaling issues.")

                # Ensure coordinates are within valid bounds
                if input_image_width and input_image_height:
                    line_left = max(0, min(line_left, input_image_width))
                    line_top = max(0, min(line_top, input_image_height))
                    line_width = max(0, min(line_width, input_image_width - line_left))
                    line_height = max(
                        0, min(line_height, input_image_height - line_top)
                    )

                # Add line-level data
                output["text"].append(line_text)
                output["left"].append(round(line_left, 2))
                output["top"].append(round(line_top, 2))
                output["width"].append(round(line_width, 2))
                output["height"].append(round(line_height, 2))
                output["conf"].append(int(line_confidence * 100))
                output["model"].append(line_model if line_model else "Paddle")

        return output

    def _convert_line_to_word_level(
        self,
        line_data: Dict[str, List],
        image_width: int,
        image_height: int,
        image: Image.Image,
        image_name: str = None,
    ) -> Dict[str, List]:
        """
        Converts line-level OCR results to word-level using AdaptiveSegmenter.segment().
        This method processes each line individually using the adaptive segmentation algorithm.

        Args:
            line_data: Dictionary with keys "text", "left", "top", "width", "height", "conf" (all lists)
            image_width: Width of the full image
            image_height: Height of the full image
            image: PIL Image object of the full image
            image_name: Name of the image
        Returns:
            Dictionary with same keys as input, containing word-level bounding boxes
        """
        output = {
            "text": list(),
            "left": list(),
            "top": list(),
            "width": list(),
            "height": list(),
            "conf": list(),
            "model": list(),
        }

        if not line_data or not line_data.get("text"):
            return output

        # Convert PIL Image to numpy array (BGR format for OpenCV)
        if hasattr(image, "size"):  # PIL Image
            image_np = np.array(image)
            if len(image_np.shape) == 3:
                # Convert RGB to BGR for OpenCV
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            elif len(image_np.shape) == 2:
                # Grayscale - convert to BGR
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        else:
            # Already numpy array
            image_np = image.copy()
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

        # Validate that image_np dimensions match the expected image_width and image_height
        # PIL Image.size returns (width, height), but numpy array shape is (height, width, channels)
        actual_height, actual_width = image_np.shape[:2]
        if actual_width != image_width or actual_height != image_height:
            print(
                f"Warning: Image dimension mismatch! Expected {image_width}x{image_height}, but got {actual_width}x{actual_height}"
            )
            # print(f"Using actual dimensions: {actual_width}x{actual_height}")
            # Update to use actual dimensions
            image_width = actual_width
            image_height = actual_height

        print("segmenting line-level OCR results to word-level...")

        segmenter = AdaptiveSegmenter(output_folder=self.output_folder)

        # Process each line
        for i in range(len(line_data["text"])):
            line_text = line_data["text"][i]
            line_conf = line_data["conf"][i]
            # Extract model, defaulting to "Paddle" if not available
            if "model" in line_data and len(line_data["model"]) > i:
                line_model = line_data["model"][i]
            else:
                line_model = "Paddle"

            # Get the float values
            f_left = float(line_data["left"][i])
            f_top = float(line_data["top"][i])
            f_width = float(line_data["width"][i])
            f_height = float(line_data["height"][i])

            # A simple heuristic to check if coords are normalized
            # If any value is > 1.0, assume they are already pixels
            is_normalized = (
                f_left <= 1.0 and f_top <= 1.0 and f_width <= 1.0 and f_height <= 1.0
            )

            if is_normalized:
                # Convert from normalized (0.0-1.0) to absolute pixels
                line_left = float(round(f_left * image_width))
                line_top = float(round(f_top * image_height))
                line_width = float(round(f_width * image_width))
                line_height = float(round(f_height * image_height))
            else:
                # They are already pixels, just convert to int
                line_left = float(round(f_left))
                line_top = float(round(f_top))
                line_width = float(round(f_width))
                line_height = float(round(f_height))

            if not line_text.strip():
                continue

            # Clamp bounding box to image boundaries
            line_left = int(max(0, min(line_left, image_width - 1)))
            line_top = int(max(0, min(line_top, image_height - 1)))
            line_width = int(max(1, min(line_width, image_width - line_left)))
            line_height = int(max(1, min(line_height, image_height - line_top)))

            # Validate crop coordinates are within bounds
            if line_left >= image_width or line_top >= image_height:
                # print(f"Warning: Line coordinates out of bounds. Skipping line '{line_text[:50]}...'")
                continue

            if line_left + line_width > image_width:
                line_width = image_width - line_left
                # print(f"Warning: Adjusted line_width to {line_width} to fit within image")

            if line_top + line_height > image_height:
                line_height = image_height - line_top
                # print(f"Warning: Adjusted line_height to {line_height} to fit within image")

            # Ensure we have valid dimensions
            if line_width <= 0 or line_height <= 0:
                # print(f"Warning: Invalid line dimensions ({line_width}x{line_height}). Skipping line '{line_text[:50]}...'")
                continue

            # Crop the line image from the full image
            try:
                line_image = image_np[
                    line_top : line_top + line_height,
                    line_left : line_left + line_width,
                ]
            except IndexError:
                # print(f"Error cropping line image: {e}")
                # print(f"Attempted to crop: [{line_top}:{line_top + line_height}, {line_left}:{line_left + line_width}]")
                # print(f"Image_np shape: {image_np.shape}")
                continue

            if line_image is None or line_image.size == 0:
                # print(f"Warning: Cropped line_image is None or empty. Skipping line '{line_text[:50]}...'")
                continue

            # Validate line_image has valid shape
            if len(line_image.shape) < 2:
                # print(f"Warning: line_image has invalid shape {line_image.shape}. Skipping line '{line_text[:50]}...'")
                continue

            # Create single-line data structure for segment method
            single_line_data = {
                "text": [line_text],
                "left": [0],  # Relative to cropped image
                "top": [0],
                "width": [line_width],
                "height": [line_height],
                "conf": [line_conf],
            }

            # Validate line_image before passing to segmenter
            if line_image is None:
                # print(f"Error: line_image is None for line '{line_text[:50]}...'")
                continue

            # Use AdaptiveSegmenter.segment() to segment this line
            try:
                word_output, _ = segmenter.segment(
                    single_line_data, line_image, image_name=image_name
                )
            except Exception:
                # print(f"Error in segmenter.segment for line '{line_text[:50]}...': {e}")
                # print(f"line_image shape: {line_image.shape if line_image is not None else 'None'}")
                raise

            if not word_output or not word_output.get("text"):
                # If segmentation failed, fall back to proportional estimation
                words = line_text.split()
                if words:
                    num_chars = len("".join(words))
                    num_spaces = len(words) - 1
                    if num_chars > 0:
                        char_space_ratio = 2.0
                        estimated_space_width = (
                            line_width / (num_chars * char_space_ratio + num_spaces)
                            if (num_chars * char_space_ratio + num_spaces) > 0
                            else line_width / num_chars
                        )
                        avg_char_width = estimated_space_width * char_space_ratio
                        current_left = 0
                        for word in words:
                            word_width = len(word) * avg_char_width
                            clamped_left = max(0, min(current_left, line_width))
                            clamped_width = max(
                                0, min(word_width, line_width - clamped_left)
                            )
                            output["text"].append(word)
                            output["left"].append(
                                line_left + clamped_left
                            )  # Add line offset
                            output["top"].append(line_top)
                            output["width"].append(clamped_width)
                            output["height"].append(line_height)
                            output["conf"].append(line_conf)
                            output["model"].append(line_model)
                            current_left += word_width + estimated_space_width
                continue

            # Adjust coordinates back to full image coordinates
            for j in range(len(word_output["text"])):
                output["text"].append(word_output["text"][j])
                output["left"].append(line_left + word_output["left"][j])
                output["top"].append(line_top + word_output["top"][j])
                output["width"].append(word_output["width"][j])
                output["height"].append(word_output["height"][j])
                output["conf"].append(word_output["conf"][j])
                # Preserve the model from the line-level data
                output["model"].append(line_model)

        return output

    def _visualize_tesseract_bounding_boxes(
        self,
        image: Image.Image,
        ocr_data: Dict[str, List],
        image_name: str = None,
        visualisation_folder: str = "tesseract_visualisations",
    ) -> None:
        """
        Visualizes Tesseract OCR bounding boxes with confidence-based colors and a legend.

        Args:
            image: The PIL Image object
            ocr_data: Tesseract OCR data dictionary
            image_name: Optional name for the saved image file
        """
        if not ocr_data or not ocr_data.get("text"):
            return

        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Get image dimensions
        height, width = image_cv.shape[:2]

        # Define confidence ranges and colors
        confidence_ranges = [
            (80, 100, (0, 255, 0), "High (80-100%)"),  # Green
            (50, 79, (0, 165, 255), "Medium (50-79%)"),  # Orange
            (0, 49, (0, 0, 255), "Low (0-49%)"),  # Red
        ]

        # Process each detected text element
        for i in range(len(ocr_data["text"])):
            text = ocr_data["text"][i]
            conf = int(ocr_data["conf"][i])

            # Skip empty text or invalid confidence
            if not text.strip() or conf == -1:
                continue

            left = ocr_data["left"][i]
            top = ocr_data["top"][i]
            width_box = ocr_data["width"][i]
            height_box = ocr_data["height"][i]

            # Calculate bounding box coordinates
            x1 = int(left)
            y1 = int(top)
            x2 = int(left + width_box)
            y2 = int(top + height_box)

            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # Skip if bounding box is invalid
            if x2 <= x1 or y2 <= y1:
                continue

            # Determine color based on confidence score
            color = (0, 0, 255)  # Default to red
            for min_conf, max_conf, conf_color, _ in confidence_ranges:
                if min_conf <= conf <= max_conf:
                    color = conf_color
                    break

            # Draw bounding box
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 1)

        # Add legend
        self._add_confidence_legend(image_cv, confidence_ranges)

        # Save the visualization
        tesseract_viz_folder = os.path.join(self.output_folder, visualisation_folder)

        # Double-check the constructed path is safe
        if not validate_folder_containment(tesseract_viz_folder, OUTPUT_FOLDER):
            raise ValueError(
                f"Unsafe tesseract visualisations folder path: {tesseract_viz_folder}"
            )

        os.makedirs(tesseract_viz_folder, exist_ok=True)

        # Generate filename
        if image_name:
            # Remove file extension if present
            base_name = os.path.splitext(image_name)[0]
            filename = f"{base_name}_{visualisation_folder}.jpg"
        else:
            timestamp = int(time.time())
            filename = f"{visualisation_folder}_{timestamp}.jpg"

        output_path = os.path.join(tesseract_viz_folder, filename)

        # Save the image
        cv2.imwrite(output_path, image_cv)
        print(f"Tesseract visualization saved to: {output_path}")

    def _add_confidence_legend(
        self, image_cv: np.ndarray, confidence_ranges: List[Tuple]
    ) -> None:
        """
        Adds a confidence legend to the visualization image.

        Args:
            image_cv: OpenCV image array
            confidence_ranges: List of tuples containing (min_conf, max_conf, color, label)
        """
        height, width = image_cv.shape[:2]

        # Legend parameters
        legend_width = 200
        legend_height = 100
        legend_x = width - legend_width - 20
        legend_y = 20

        # Draw legend background
        cv2.rectangle(
            image_cv,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            (255, 255, 255),  # White background
            -1,
        )
        cv2.rectangle(
            image_cv,
            (legend_x, legend_y),
            (legend_x + legend_width, legend_y + legend_height),
            (0, 0, 0),  # Black border
            2,
        )

        # Add title
        title_text = "Confidence Levels"
        font_scale = 0.6
        font_thickness = 2
        (title_width, title_height), _ = cv2.getTextSize(
            title_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        title_x = legend_x + (legend_width - title_width) // 2
        title_y = legend_y + title_height + 10
        cv2.putText(
            image_cv,
            title_text,
            (title_x, title_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text
            font_thickness,
        )

        # Add confidence range items
        item_spacing = 25
        start_y = title_y + 25

        for i, (min_conf, max_conf, color, label) in enumerate(confidence_ranges):
            item_y = start_y + i * item_spacing

            # Draw color box
            box_size = 15
            box_x = legend_x + 10
            box_y = item_y - box_size
            cv2.rectangle(
                image_cv,
                (box_x, box_y),
                (box_x + box_size, box_y + box_size),
                color,
                -1,
            )
            cv2.rectangle(
                image_cv,
                (box_x, box_y),
                (box_x + box_size, box_y + box_size),
                (0, 0, 0),  # Black border
                1,
            )

            # Add label text
            label_x = box_x + box_size + 10
            label_y = item_y - 5
            cv2.putText(
                image_cv,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                1,
            )

    # Calculate line-level bounding boxes and average confidence
    def _calculate_line_bbox(self, group):
        # Get the leftmost and rightmost positions
        left = group["left"].min()
        top = group["top"].min()
        right = (group["left"] + group["width"]).max()
        bottom = (group["top"] + group["height"]).max()

        # Calculate width and height
        width = right - left
        height = bottom - top

        # Calculate average confidence
        avg_conf = round(group["conf"].mean(), 0)

        return pd.Series(
            {
                "text": " ".join(group["text"].astype(str).tolist()),
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "conf": avg_conf,
            }
        )

    def _perform_hybrid_ocr(
        self,
        image: Image.Image,
        confidence_threshold: int = HYBRID_OCR_CONFIDENCE_THRESHOLD,
        padding: int = HYBRID_OCR_PADDING,
        ocr: Optional[Any] = None,
        image_name: str = "unknown_image_name",
    ) -> Dict[str, list]:
        """
        Performs hybrid OCR on an image using Tesseract for initial OCR and PaddleOCR/VLM to enhance
        results for low-confidence or uncertain words.

        Args:
            image (Image.Image): The input image (PIL format) to be processed.
            confidence_threshold (int, optional): Tesseract confidence threshold below which words are
                re-analyzed with secondary OCR (PaddleOCR/VLM). Defaults to HYBRID_OCR_CONFIDENCE_THRESHOLD.
            padding (int, optional): Pixel padding (in all directions) to add around each word box when
                cropping for secondary OCR. Defaults to HYBRID_OCR_PADDING.
            ocr (Optional[Any], optional): An instance of the PaddleOCR or VLM engine. If None, will use the
                instance's `paddle_ocr` attribute if available. Only necessary for PaddleOCR-based pipelines.
            image_name (str, optional): Optional name of the image, useful for debugging and visualization.

        Returns:
            Dict[str, list]: OCR results in the dictionary format of pytesseract.image_to_data (keys:
                'text', 'left', 'top', 'width', 'height', 'conf', 'model', ...).
        """
        # Determine if we're using VLM or PaddleOCR
        use_vlm = self.ocr_engine == "hybrid-vlm"

        if not use_vlm:
            if ocr is None:
                if hasattr(self, "paddle_ocr") and self.paddle_ocr is not None:
                    ocr = self.paddle_ocr
                else:
                    raise ValueError(
                        "No OCR object provided and 'paddle_ocr' is not initialized."
                    )

        # print("Starting hybrid OCR process...")

        # 1. Get initial word-level results from Tesseract
        tesseract_data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config=self.tesseract_config,
            lang=self.tesseract_lang,
        )

        if TESSERACT_WORD_LEVEL_OCR is False:
            ocr_df = pd.DataFrame(tesseract_data)

            # Filter out invalid entries (confidence == -1)
            ocr_df = ocr_df[ocr_df.conf != -1]

            # Group by line and aggregate text
            line_groups = ocr_df.groupby(["block_num", "par_num", "line_num"])

            ocr_data = line_groups.apply(self._calculate_line_bbox).reset_index()

            # Overwrite tesseract_data with the aggregated data
            tesseract_data = {
                "text": ocr_data["text"].tolist(),
                "left": ocr_data["left"].astype(int).tolist(),
                "top": ocr_data["top"].astype(int).tolist(),
                "width": ocr_data["width"].astype(int).tolist(),
                "height": ocr_data["height"].astype(int).tolist(),
                "conf": ocr_data["conf"].tolist(),
                "model": ["Tesseract"] * len(ocr_data),  # Add model field
            }

        final_data = {
            "text": list(),
            "left": list(),
            "top": list(),
            "width": list(),
            "height": list(),
            "conf": list(),
            "model": list(),  # Track which model was used for each word
        }

        num_words = len(tesseract_data["text"])

        # This handles the "no text on page" case. If num_words is 0, the loop is skipped
        # and an empty dictionary with empty lists is returned, which is the correct behavior.
        for i in range(num_words):
            text = tesseract_data["text"][i]
            conf = int(tesseract_data["conf"][i])

            # Skip empty text boxes or non-word elements (like page/block markers)
            if not text.strip() or conf == -1:
                continue

            left = tesseract_data["left"][i]
            top = tesseract_data["top"][i]
            width = tesseract_data["width"][i]
            height = tesseract_data["height"][i]
            # line_number = tesseract_data['abs_line_id'][i]

            # Initialize model as Tesseract (default)
            model_used = "Tesseract"

            # If confidence is low, use PaddleOCR for a second opinion
            if conf <= confidence_threshold:
                img_width, img_height = image.size
                crop_left = max(0, left - padding)
                crop_top = max(0, top - padding)
                crop_right = min(img_width, left + width + padding)
                crop_bottom = min(img_height, top + height + padding)

                # Ensure crop dimensions are valid
                if crop_right <= crop_left or crop_bottom <= crop_top:
                    continue  # Skip invalid crops

                cropped_image = image.crop(
                    (crop_left, crop_top, crop_right, crop_bottom)
                )
                if use_vlm:
                    # Use VLM for OCR
                    vlm_result = _vlm_ocr_predict(cropped_image)
                    rec_texts = vlm_result.get("rec_texts", [])
                    rec_scores = vlm_result.get("rec_scores", [])
                else:
                    # Use PaddleOCR
                    cropped_image_np = np.array(cropped_image)

                    if len(cropped_image_np.shape) == 2:
                        cropped_image_np = np.stack([cropped_image_np] * 3, axis=-1)

                    paddle_results = ocr.predict(cropped_image_np)

                    if paddle_results and paddle_results[0]:
                        rec_texts = paddle_results[0].get("rec_texts", [])
                        rec_scores = paddle_results[0].get("rec_scores", [])
                    else:
                        rec_texts = []
                        rec_scores = []

                if rec_texts and rec_scores:
                    new_text = " ".join(rec_texts)
                    new_conf = int(round(np.median(rec_scores) * 100, 0))

                    # Only replace if Paddle's/VLM's confidence is better
                    if new_conf >= conf:
                        ocr_type = "VLM" if use_vlm else "Paddle"
                        print(
                            f"  Re-OCR'd word: '{text}' (conf: {conf}) -> '{new_text}' (conf: {new_conf:.0f}) [{ocr_type}]"
                        )

                        # For exporting example image comparisons, not used here
                        safe_filename = self._create_safe_filename_with_confidence(
                            text, new_text, conf, new_conf, ocr_type
                        )

                        if SAVE_EXAMPLE_HYBRID_IMAGES:
                            # Normalize and validate image_name to prevent path traversal attacks
                            normalized_image_name = os.path.normpath(
                                image_name + "_" + ocr_type
                            )
                            # Ensure the image name doesn't contain path traversal characters
                            if (
                                ".." in normalized_image_name
                                or "/" in normalized_image_name
                                or "\\" in normalized_image_name
                            ):
                                normalized_image_name = (
                                    "safe_image"  # Fallback to safe default
                                )

                            hybrid_ocr_examples_folder = (
                                self.output_folder
                                + f"/hybrid_ocr_examples/{normalized_image_name}"
                            )
                            # Validate the constructed path is safe before creating directories
                            if not validate_folder_containment(
                                hybrid_ocr_examples_folder, OUTPUT_FOLDER
                            ):
                                raise ValueError(
                                    f"Unsafe hybrid_ocr_examples folder path: {hybrid_ocr_examples_folder}"
                                )

                            if not os.path.exists(hybrid_ocr_examples_folder):
                                os.makedirs(hybrid_ocr_examples_folder)
                            output_image_path = (
                                hybrid_ocr_examples_folder + f"/{safe_filename}.png"
                            )
                            print(f"Saving example image to {output_image_path}")
                            cropped_image.save(output_image_path)

                        text = new_text
                        conf = new_conf
                        model_used = ocr_type  # Update model to VLM or Paddle

                    else:
                        ocr_type = "VLM" if use_vlm else "Paddle"
                        print(
                            f"  '{text}' (conf: {conf}) -> {ocr_type} result '{new_text}' (conf: {new_conf:.0f}) was not better. Keeping original."
                        )
                else:
                    # OCR ran but found nothing, discard original word
                    ocr_type = "VLM" if use_vlm else "Paddle"
                    print(
                        f"  '{text}' (conf: {conf}) -> No text found by {ocr_type}. Discarding."
                    )
                    text = ""

            # Append the final result (either original, replaced, or skipped if empty)
            if text.strip():
                final_data["text"].append(clean_unicode_text(text))
                final_data["left"].append(left)
                final_data["top"].append(top)
                final_data["width"].append(width)
                final_data["height"].append(height)
                final_data["conf"].append(int(conf))
                final_data["model"].append(model_used)
                # final_data['line_number'].append(int(line_number))

        return final_data

    def _perform_hybrid_paddle_vlm_ocr(
        self,
        image: Image.Image,
        ocr: Optional[Any] = None,
        paddle_results: List[Any] = None,
        confidence_threshold: int = HYBRID_OCR_CONFIDENCE_THRESHOLD,
        padding: int = HYBRID_OCR_PADDING,
        image_name: str = "unknown_image_name",
        input_image_width: int = None,
        input_image_height: int = None,
    ) -> List[Any]:
        """
        Performs OCR using PaddleOCR at line level, then VLM for low-confidence lines.
        Returns modified paddle_results in the same format as PaddleOCR output.

        Args:
            image: PIL Image to process
            ocr: PaddleOCR instance (optional, uses self.paddle_ocr if not provided)
            paddle_results: PaddleOCR results in original format (List of dicts with rec_texts, rec_scores, rec_polys)
            confidence_threshold: Confidence threshold below which VLM is used
            padding: Padding to add around line crops
            image_name: Name of the image for logging/debugging
            input_image_width: Original image width (before preprocessing)
            input_image_height: Original image height (before preprocessing)

        Returns:
            Modified paddle_results with VLM replacements for low-confidence lines
        """
        if ocr is None:
            if hasattr(self, "paddle_ocr") and self.paddle_ocr is not None:
                ocr = self.paddle_ocr
            else:
                raise ValueError(
                    "No OCR object provided and 'paddle_ocr' is not initialized."
                )

        if paddle_results is None or not paddle_results:
            return paddle_results

        print("Starting hybrid PaddleOCR + VLM OCR process...")

        # Get image dimensions
        img_width, img_height = image.size

        # Use original dimensions if provided, otherwise use current image dimensions
        if input_image_width is None:
            input_image_width = img_width
        if input_image_height is None:
            input_image_height = img_height

        # Create a deep copy of paddle_results to modify
        copied_paddle_results = copy.deepcopy(paddle_results)

        def _normalize_paddle_result_lists(rec_texts, rec_scores, rec_polys):
            """
            Normalizes PaddleOCR result lists to ensure they all have the same length.
            Pads missing entries with appropriate defaults:
            - rec_texts: empty string ""
            - rec_scores: 0.0 (low confidence)
            - rec_polys: empty list []

            Args:
                rec_texts: List of recognized text strings
                rec_scores: List of confidence scores
                rec_polys: List of bounding box polygons

            Returns:
                Tuple of (normalized_rec_texts, normalized_rec_scores, normalized_rec_polys, max_length)
            """
            len_texts = len(rec_texts)
            len_scores = len(rec_scores)
            len_polys = len(rec_polys)
            max_length = max(len_texts, len_scores, len_polys)

            # Only normalize if there's a mismatch
            if max_length > 0 and (
                len_texts != max_length
                or len_scores != max_length
                or len_polys != max_length
            ):
                print(
                    f"Warning: List length mismatch detected - rec_texts: {len_texts}, "
                    f"rec_scores: {len_scores}, rec_polys: {len_polys}. "
                    f"Padding to length {max_length}."
                )

                # Pad rec_texts
                if len_texts < max_length:
                    rec_texts = list(rec_texts) + [""] * (max_length - len_texts)

                # Pad rec_scores
                if len_scores < max_length:
                    rec_scores = list(rec_scores) + [0.0] * (max_length - len_scores)

                # Pad rec_polys
                if len_polys < max_length:
                    rec_polys = list(rec_polys) + [[]] * (max_length - len_polys)

            return rec_texts, rec_scores, rec_polys, max_length

        @spaces.GPU(duration=MAX_SPACES_GPU_RUN_TIME)
        def _process_page_result_with_hybrid_vlm_ocr(
            page_results: list,
            image: Image.Image,
            img_width: int,
            img_height: int,
            input_image_width: int,
            input_image_height: int,
            confidence_threshold: float,
            image_name: str,
            output_folder: str,
            padding: int = 0,
        ):
            """
            Processes OCR page results using a hybrid system that combines PaddleOCR for initial recognition
            and VLM for low-confidence lines. When PaddleOCR's recognition confidence for a detected line is
            below the specified threshold, the line is re-processed using a higher-quality (but slower) VLM
            model and the result is used to replace the low-confidence recognition. Results are kept in
            PaddleOCR's standard output format for downstream compatibility.

            Args:
                page_results (list): The list of page result dicts from PaddleOCR to process. Each dict should
                    contain keys like 'rec_texts', 'rec_scores', 'rec_polys', and optionally 'image_width',
                    'image_height', and 'rec_models'.
                image (PIL.Image.Image): The PIL Image object of the full page to allow line cropping.
                img_width (int): The width of the (possibly preprocessed) image in pixels.
                img_height (int): The height of the (possibly preprocessed) image in pixels.
                input_image_width (int): The original image width (before any resizing/preprocessing).
                input_image_height (int): The original image height (before any resizing/preprocessing).
                confidence_threshold (float): Lines recognized by PaddleOCR with confidence lower than this
                    threshold will be replaced using the VLM.
                image_name (str): The name of the source image, used for logging/debugging.
                output_folder (str): The output folder path for saving example images.
                padding (int): Padding to add around line crops.

            Returns:
                Modified page_results with VLM replacements for low-confidence lines.
            """

            # Helper function to create safe filename (inlined to avoid needing instance_self)
            def _create_safe_filename_with_confidence(
                original_text: str,
                new_text: str,
                conf: int,
                new_conf: int,
                ocr_type: str = "OCR",
            ) -> str:
                """Creates a safe filename using confidence values when text sanitization fails."""

                # Helper to sanitize text similar to _sanitize_filename
                def _sanitize_text_for_filename(
                    text: str,
                    max_length: int = 20,
                    fallback_prefix: str = "unknown_text",
                ) -> str:
                    """Sanitizes text for use in filenames."""
                    sanitized = safe_sanitize_text(text)
                    # Remove leading/trailing underscores and spaces
                    sanitized = sanitized.strip("_ ")
                    # If empty after sanitization, use a default value
                    if not sanitized:
                        sanitized = fallback_prefix
                    # Limit to max_length characters
                    if len(sanitized) > max_length:
                        sanitized = sanitized[:max_length]
                        sanitized = sanitized.rstrip("_")
                    # Final check: if still empty or too short, use fallback
                    if not sanitized or len(sanitized) < 3:
                        sanitized = fallback_prefix
                    return sanitized

                # Try to sanitize both texts
                safe_original = _sanitize_text_for_filename(
                    original_text, max_length=15, fallback_prefix=f"orig_conf_{conf}"
                )
                safe_new = _sanitize_text_for_filename(
                    new_text, max_length=15, fallback_prefix=f"new_conf_{new_conf}"
                )

                # If both sanitizations resulted in fallback names, create a confidence-based name
                if safe_original.startswith("orig_conf") and safe_new.startswith(
                    "new_conf"
                ):
                    return f"{ocr_type}_conf_{conf}_to_conf_{new_conf}"

                return f"{safe_original}_conf_{conf}_to_{safe_new}_conf_{new_conf}"

            # Process each page result in paddle_results
            for page_result in page_results:
                # Extract text recognition results from the paddle format
                rec_texts = page_result.get("rec_texts", list())
                rec_scores = page_result.get("rec_scores", list())
                rec_polys = page_result.get("rec_polys", list())

                # Normalize lists to ensure they all have the same length
                rec_texts, rec_scores, rec_polys, num_lines = (
                    _normalize_paddle_result_lists(rec_texts, rec_scores, rec_polys)
                )

                # Update page_result with normalized lists
                page_result["rec_texts"] = rec_texts
                page_result["rec_scores"] = rec_scores
                page_result["rec_polys"] = rec_polys

                # Initialize rec_models list with "Paddle" as default for all lines
                if (
                    "rec_models" not in page_result
                    or len(page_result.get("rec_models", [])) != num_lines
                ):
                    rec_models = ["Paddle"] * num_lines
                    page_result["rec_models"] = rec_models
                else:
                    rec_models = page_result["rec_models"]

                # Since we're using the exact image PaddleOCR processed, coordinates are directly in image space
                # No coordinate conversion needed - coordinates match the image dimensions exactly

                # Process each line
                # print(f"Processing {num_lines} lines from PaddleOCR results...")

                for i in range(num_lines):
                    line_text = rec_texts[i]
                    line_conf = float(rec_scores[i]) * 100  # Convert to percentage
                    bounding_box = rec_polys[i]

                    # Skip if bounding box is empty (from padding)
                    # Handle numpy arrays, lists, and None values safely
                    if bounding_box is None:
                        continue

                    # Convert to list first to handle numpy arrays safely
                    if hasattr(bounding_box, "tolist"):
                        box = bounding_box.tolist()
                    else:
                        box = bounding_box

                    # Check if box is empty (handles both list and numpy array cases)
                    if not box or (isinstance(box, list) and len(box) == 0):
                        continue

                    # Skip empty lines
                    if not line_text.strip():
                        continue

                    # Convert polygon to bounding box
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]
                    line_left_paddle = float(min(x_coords))
                    line_top_paddle = float(min(y_coords))
                    line_right_paddle = float(max(x_coords))
                    line_bottom_paddle = float(max(y_coords))
                    line_width_paddle = line_right_paddle - line_left_paddle
                    line_height_paddle = line_bottom_paddle - line_top_paddle

                    # Since we're using the exact image PaddleOCR processed, coordinates are already in image space
                    # No conversion needed - use coordinates directly
                    line_left = line_left_paddle
                    line_top = line_top_paddle
                    line_width = line_width_paddle
                    line_height = line_height_paddle

                    # Initialize model as PaddleOCR (default)

                    # Count words in PaddleOCR output
                    paddle_words = line_text.split()
                    paddle_word_count = len(paddle_words)

                    # If confidence is low, use VLM for a second opinion
                    if line_conf <= confidence_threshold:

                        # Ensure minimum line height for VLM processing
                        # If line_height is too small, use a minimum height based on typical text line height
                        min_line_height = max(
                            line_height, 20
                        )  # Minimum 20 pixels for text line

                        # Calculate crop coordinates with padding
                        # Convert floats to integers and apply padding, clamping to image bounds
                        crop_left = max(0, int(round(line_left - padding)))
                        crop_top = max(0, int(round(line_top - padding)))
                        crop_right = min(
                            img_width, int(round(line_left + line_width + padding))
                        )
                        crop_bottom = min(
                            img_height, int(round(line_top + min_line_height + padding))
                        )

                        # Ensure crop dimensions are valid
                        if crop_right <= crop_left or crop_bottom <= crop_top:
                            # Invalid crop, keep original PaddleOCR result
                            continue

                        # Crop the line image
                        cropped_image = image.crop(
                            (crop_left, crop_top, crop_right, crop_bottom)
                        )

                        # Check if cropped image is too small for VLM processing
                        crop_width = crop_right - crop_left
                        crop_height = crop_bottom - crop_top
                        if crop_width < 10 or crop_height < 10:
                            continue

                        # Ensure cropped image is in RGB mode before passing to VLM
                        if cropped_image.mode != "RGB":
                            cropped_image = cropped_image.convert("RGB")

                        # Save input image for debugging if environment variable is set
                        if SAVE_VLM_INPUT_IMAGES:
                            try:
                                vlm_debug_dir = os.path.join(
                                    output_folder,
                                    "hybrid_paddle_vlm_visualisations/hybrid_analysis_input_images",
                                )
                                os.makedirs(vlm_debug_dir, exist_ok=True)
                                line_text_safe = safe_sanitize_text(line_text)
                                line_text_shortened = line_text_safe[:20]
                                image_name_safe = safe_sanitize_text(image_name)
                                image_name_shortened = image_name_safe[:20]
                                filename = f"{image_name_shortened}_{line_text_shortened}_hybrid_analysis_input_image.png"
                                filepath = os.path.join(vlm_debug_dir, filename)
                                cropped_image.save(filepath)
                                # print(f"Saved VLM input image to: {filepath}")
                            except Exception as save_error:
                                print(
                                    f"Warning: Could not save VLM input image: {save_error}"
                                )

                        # Use VLM for OCR on this line with error handling
                        vlm_result = None
                        vlm_rec_texts = []
                        vlm_rec_scores = []

                        try:
                            vlm_result = _vlm_ocr_predict(cropped_image)
                            vlm_rec_texts = (
                                vlm_result.get("rec_texts", []) if vlm_result else []
                            )
                            vlm_rec_scores = (
                                vlm_result.get("rec_scores", []) if vlm_result else []
                            )
                        except Exception:
                            # Ensure we keep original PaddleOCR result on error
                            vlm_rec_texts = []
                            vlm_rec_scores = []

                        if vlm_rec_texts and vlm_rec_scores:
                            # Combine VLM words into a single text string
                            vlm_text = " ".join(vlm_rec_texts)
                            vlm_word_count = len(vlm_rec_texts)
                            vlm_conf = float(
                                np.median(vlm_rec_scores)
                            )  # Keep as 0-1 range for paddle format

                            # Only replace if word counts match
                            word_count_allowed_difference = 4
                            if (
                                vlm_word_count - paddle_word_count
                                <= word_count_allowed_difference
                                and vlm_word_count - paddle_word_count
                                >= -word_count_allowed_difference
                            ):
                                text_output = f"  Re-OCR'd line: '{line_text}' (conf: {line_conf:.1f}, words: {paddle_word_count}) "
                                text_output += f"-> '{vlm_text}' (conf: {vlm_conf*100:.1f}, words: {vlm_word_count}) [VLM]"
                                print(text_output)

                                if REPORT_VLM_OUTPUTS_TO_GUI:
                                    gr.Info(text_output, duration=2)

                                # For exporting example image comparisons
                                safe_filename = _create_safe_filename_with_confidence(
                                    line_text,
                                    vlm_text,
                                    int(line_conf),
                                    int(vlm_conf * 100),
                                    "VLM",
                                )

                                if SAVE_EXAMPLE_HYBRID_IMAGES:
                                    # Normalize and validate image_name to prevent path traversal attacks
                                    normalized_image_name = os.path.normpath(
                                        image_name + "_hybrid_paddle_vlm"
                                    )
                                    if (
                                        ".." in normalized_image_name
                                        or "/" in normalized_image_name
                                        or "\\" in normalized_image_name
                                    ):
                                        normalized_image_name = "safe_image"

                                    hybrid_ocr_examples_folder = (
                                        output_folder
                                        + f"/hybrid_ocr_examples/{normalized_image_name}"
                                    )
                                    # Validate the constructed path is safe
                                    if not validate_folder_containment(
                                        hybrid_ocr_examples_folder, OUTPUT_FOLDER
                                    ):
                                        raise ValueError(
                                            f"Unsafe hybrid_ocr_examples folder path: {hybrid_ocr_examples_folder}"
                                        )

                                    if not os.path.exists(hybrid_ocr_examples_folder):
                                        os.makedirs(hybrid_ocr_examples_folder)
                                    output_image_path = (
                                        hybrid_ocr_examples_folder
                                        + f"/{safe_filename}.png"
                                    )
                                    # print(f"Saving example image to {output_image_path}")
                                    cropped_image.save(output_image_path)

                                # Replace with VLM result in paddle_results format
                                # Update rec_texts, rec_scores, and rec_models for this line
                                rec_texts[i] = vlm_text
                                rec_scores[i] = vlm_conf
                                rec_models[i] = "VLM"
                                # Ensure page_result is updated with the modified rec_models list
                                page_result["rec_models"] = rec_models
                            else:
                                print(
                                    f"  Line: '{line_text}' (conf: {line_conf:.1f}, words: {paddle_word_count}) -> "
                                    f"VLM result '{vlm_text}' (conf: {vlm_conf*100:.1f}, words: {vlm_word_count}) "
                                    f"word count mismatch. Keeping PaddleOCR result."
                                )
                        else:
                            # VLM returned empty or no results - keep original PaddleOCR result
                            if line_conf <= confidence_threshold:
                                pass

            # Debug: Print summary of model labels before returning
            for page_idx, page_result in enumerate(page_results):
                rec_models = page_result.get("rec_models", [])
                sum(1 for m in rec_models if m == "VLM")
                sum(1 for m in rec_models if m == "Paddle")

            return page_results

        modified_paddle_results = _process_page_result_with_hybrid_vlm_ocr(
            copied_paddle_results,
            image,
            img_width,
            img_height,
            input_image_width,
            input_image_height,
            confidence_threshold,
            image_name,
            self.output_folder,
            padding,
        )

        return modified_paddle_results

    def _perform_hybrid_paddle_inference_server_ocr(
        self,
        image: Image.Image,
        ocr: Optional[Any] = None,
        paddle_results: List[Any] = None,
        confidence_threshold: int = HYBRID_OCR_CONFIDENCE_THRESHOLD,
        padding: int = HYBRID_OCR_PADDING,
        image_name: str = "unknown_image_name",
        input_image_width: int = None,
        input_image_height: int = None,
    ) -> List[Any]:
        """
        Performs OCR using PaddleOCR at line level, then inference-server API for low-confidence lines.
        Returns modified paddle_results in the same format as PaddleOCR output.

        Args:
            image: PIL Image to process
            ocr: PaddleOCR instance (optional, uses self.paddle_ocr if not provided)
            paddle_results: PaddleOCR results in original format (List of dicts with rec_texts, rec_scores, rec_polys)
            confidence_threshold: Confidence threshold below which inference-server is used
            padding: Padding to add around line crops
            image_name: Name of the image for logging/debugging
            input_image_width: Original image width (before preprocessing)
            input_image_height: Original image height (before preprocessing)

        Returns:
            Modified paddle_results with inference-server replacements for low-confidence lines
        """
        if ocr is None:
            if hasattr(self, "paddle_ocr") and self.paddle_ocr is not None:
                ocr = self.paddle_ocr
            else:
                raise ValueError(
                    "No OCR object provided and 'paddle_ocr' is not initialized."
                )

        if paddle_results is None or not paddle_results:
            return paddle_results

        print("Starting hybrid PaddleOCR + Inference-server OCR process...")

        # Get image dimensions
        img_width, img_height = image.size

        # Use original dimensions if provided, otherwise use current image dimensions
        if input_image_width is None:
            input_image_width = img_width
        if input_image_height is None:
            input_image_height = img_height

        # Create a deep copy of paddle_results to modify
        copied_paddle_results = copy.deepcopy(paddle_results)

        def _normalize_paddle_result_lists(rec_texts, rec_scores, rec_polys):
            """
            Normalizes PaddleOCR result lists to ensure they all have the same length.
            Pads missing entries with appropriate defaults:
            - rec_texts: empty string ""
            - rec_scores: 0.0 (low confidence)
            - rec_polys: empty list []

            Args:
                rec_texts: List of recognized text strings
                rec_scores: List of confidence scores
                rec_polys: List of bounding box polygons

            Returns:
                Tuple of (normalized_rec_texts, normalized_rec_scores, normalized_rec_polys, max_length)
            """
            len_texts = len(rec_texts)
            len_scores = len(rec_scores)
            len_polys = len(rec_polys)
            max_length = max(len_texts, len_scores, len_polys)

            # Only normalize if there's a mismatch
            if max_length > 0 and (
                len_texts != max_length
                or len_scores != max_length
                or len_polys != max_length
            ):
                print(
                    f"Warning: List length mismatch detected - rec_texts: {len_texts}, "
                    f"rec_scores: {len_scores}, rec_polys: {len_polys}. "
                    f"Padding to length {max_length}."
                )

                # Pad rec_texts
                if len_texts < max_length:
                    rec_texts = list(rec_texts) + [""] * (max_length - len_texts)

                # Pad rec_scores
                if len_scores < max_length:
                    rec_scores = list(rec_scores) + [0.0] * (max_length - len_scores)

                # Pad rec_polys
                if len_polys < max_length:
                    rec_polys = list(rec_polys) + [[]] * (max_length - len_polys)

            return rec_texts, rec_scores, rec_polys, max_length

        def _process_page_result_with_hybrid_inference_server_ocr(
            page_results: list,
            image: Image.Image,
            img_width: int,
            img_height: int,
            input_image_width: int,
            input_image_height: int,
            confidence_threshold: float,
            image_name: str,
            instance_self: object,
            padding: int = 0,
        ):
            """
            Processes OCR page results using a hybrid system that combines PaddleOCR for initial recognition
            and an inference server for low-confidence lines. When PaddleOCR's recognition confidence for a
            detected line is below the specified threshold, the line is re-processed using a higher-quality
            (but slower) server model and the result is used to replace the low-confidence recognition.
            Results are kept in PaddleOCR's standard output format for downstream compatibility.

            Args:
                page_results (list): The list of page result dicts from PaddleOCR to process. Each dict should
                    contain keys like 'rec_texts', 'rec_scores', 'rec_polys', and optionally 'image_width',
                    'image_height', and 'rec_models'.
                image (PIL.Image.Image): The PIL Image object of the full page to allow line cropping.
                img_width (int): The width of the (possibly preprocessed) image in pixels.
                img_height (int): The height of the (possibly preprocessed) image in pixels.
                input_image_width (int): The original image width (before any resizing/preprocessing).
                input_image_height (int): The original image height (before any resizing/preprocessing).
                confidence_threshold (float): Lines recognized by PaddleOCR with confidence lower than this
                    threshold will be replaced using the inference server.
                image_name (str): The name of the source image, used for logging/debugging.
                instance_self (object): The enclosing class instance to access inference invocation.

            Returns:
                None. Modifies page_results in place with higher-confidence text replacements when possible.
            """

            # Process each page result in paddle_results
            for page_result in page_results:
                # Extract text recognition results from the paddle format
                rec_texts = page_result.get("rec_texts", list())
                rec_scores = page_result.get("rec_scores", list())
                rec_polys = page_result.get("rec_polys", list())

                # Normalize lists to ensure they all have the same length
                rec_texts, rec_scores, rec_polys, num_lines = (
                    _normalize_paddle_result_lists(rec_texts, rec_scores, rec_polys)
                )

                # Update page_result with normalized lists
                page_result["rec_texts"] = rec_texts
                page_result["rec_scores"] = rec_scores
                page_result["rec_polys"] = rec_polys

                # Initialize rec_models list with "Paddle" as default for all lines
                if (
                    "rec_models" not in page_result
                    or len(page_result.get("rec_models", [])) != num_lines
                ):
                    rec_models = ["Paddle"] * num_lines
                    page_result["rec_models"] = rec_models
                else:
                    rec_models = page_result["rec_models"]

                # Since we're using the exact image PaddleOCR processed, coordinates are directly in image space
                # No coordinate conversion needed - coordinates match the image dimensions exactly

                # Process each line
                for i in range(num_lines):
                    line_text = rec_texts[i]

                    line_conf = float(rec_scores[i]) * 100  # Convert to percentage
                    bounding_box = rec_polys[i]

                    # Skip if bounding box is empty (from padding)
                    # Handle numpy arrays, lists, and None values safely
                    if bounding_box is None:
                        print(
                            f"Current line {i + 1} of {num_lines}: Bounding box is None"
                        )
                        continue

                    # Convert to list first to handle numpy arrays safely
                    if hasattr(bounding_box, "tolist"):
                        box = bounding_box.tolist()
                    else:
                        box = bounding_box

                    # Check if box is empty (handles both list and numpy array cases)
                    if not box or (isinstance(box, list) and len(box) == 0):
                        print(f"Current line {i + 1} of {num_lines}: Box is empty")
                        continue

                    # Skip empty lines
                    if not line_text.strip():
                        print(
                            f"Current line {i + 1} of {num_lines}: Line text is empty"
                        )
                        continue

                    # Convert polygon to bounding box
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]

                    line_left_paddle = float(min(x_coords))
                    line_top_paddle = float(min(y_coords))
                    line_right_paddle = float(max(x_coords))
                    line_bottom_paddle = float(max(y_coords))
                    line_width_paddle = line_right_paddle - line_left_paddle
                    line_height_paddle = line_bottom_paddle - line_top_paddle

                    # Since we're using the exact image PaddleOCR processed, coordinates are already in image space
                    line_left = line_left_paddle
                    line_top = line_top_paddle
                    line_width = line_width_paddle
                    line_height = line_height_paddle

                    # Count words in PaddleOCR output
                    paddle_words = line_text.split()
                    paddle_word_count = len(paddle_words)

                    # If confidence is low, use inference-server for a second opinion
                    if line_conf <= confidence_threshold:

                        # Ensure minimum line height for inference-server processing
                        min_line_height = max(
                            line_height, 20
                        )  # Minimum 20 pixels for text line

                        # Calculate crop coordinates with padding
                        # Convert floats to integers and apply padding, clamping to image bounds
                        crop_left = max(0, int(round(line_left - padding)))
                        crop_top = max(0, int(round(line_top - padding)))
                        crop_right = min(
                            img_width, int(round(line_left + line_width + padding))
                        )
                        crop_bottom = min(
                            img_height, int(round(line_top + min_line_height + padding))
                        )

                        # Ensure crop dimensions are valid
                        if crop_right <= crop_left or crop_bottom <= crop_top:
                            # Invalid crop, keep original PaddleOCR result
                            print(
                                f"Current line {i + 1} of {num_lines}: Invalid crop, keeping original PaddleOCR result"
                            )
                            continue

                        # Crop the line image
                        cropped_image = image.crop(
                            (crop_left, crop_top, crop_right, crop_bottom)
                        )

                        # Check if cropped image is too small for inference-server processing
                        crop_width = crop_right - crop_left
                        crop_height = crop_bottom - crop_top
                        if crop_width < 10 or crop_height < 10:
                            # Keep original PaddleOCR result for this line
                            print(
                                f"Current line {i + 1} of {num_lines}: Cropped image is too small, keeping original PaddleOCR result"
                            )
                            continue

                        # Ensure cropped image is in RGB mode before passing to inference-server
                        if cropped_image.mode != "RGB":
                            cropped_image = cropped_image.convert("RGB")

                        # Save input image for debugging if environment variable is set
                        if SAVE_VLM_INPUT_IMAGES:
                            try:
                                inference_server_debug_dir = os.path.join(
                                    self.output_folder,
                                    "hybrid_paddle_inference_server_visualisations/hybrid_analysis_input_images",
                                )
                                os.makedirs(inference_server_debug_dir, exist_ok=True)
                                line_text_safe = safe_sanitize_text(line_text)
                                line_text_shortened = line_text_safe[:20]
                                image_name_safe = safe_sanitize_text(image_name)
                                image_name_shortened = image_name_safe[:20]
                                filename = f"{image_name_shortened}_{line_text_shortened}_hybrid_analysis_input_image.png"
                                filepath = os.path.join(
                                    inference_server_debug_dir, filename
                                )
                                cropped_image.save(filepath)
                            except Exception as save_error:
                                print(
                                    f"Warning: Could not save inference-server input image: {save_error}"
                                )

                        # Use inference-server for OCR on this line with error handling
                        inference_server_result = None
                        inference_server_rec_texts = []
                        inference_server_rec_scores = []

                        try:
                            inference_server_result = _inference_server_ocr_predict(
                                cropped_image
                            )
                            inference_server_rec_texts = (
                                inference_server_result.get("rec_texts", [])
                                if inference_server_result
                                else []
                            )

                            inference_server_rec_scores = (
                                inference_server_result.get("rec_scores", [])
                                if inference_server_result
                                else []
                            )
                        except Exception as e:
                            print(
                                f"Current line {i + 1} of {num_lines}: Error in inference-server OCR: {e}"
                            )
                            # Ensure we keep original PaddleOCR result on error
                            inference_server_rec_texts = []
                            inference_server_rec_scores = []

                        if inference_server_rec_texts and inference_server_rec_scores:
                            # Combine inference-server words into a single text string
                            inference_server_text = " ".join(inference_server_rec_texts)
                            inference_server_word_count = len(
                                inference_server_rec_texts
                            )
                            inference_server_conf = float(
                                np.median(inference_server_rec_scores)
                            )  # Keep as 0-1 range for paddle format

                            # Only replace if word counts match
                            word_count_allowed_difference = 4
                            if (
                                inference_server_word_count - paddle_word_count
                                <= word_count_allowed_difference
                                and inference_server_word_count - paddle_word_count
                                >= -word_count_allowed_difference
                            ):
                                print(
                                    f"  Re-OCR'd line: '{line_text}' (conf: {line_conf:.1f}, words: {paddle_word_count}) "
                                    f"-> '{inference_server_text}' (conf: {inference_server_conf*100:.1f}, words: {inference_server_word_count}) [Inference Server]"
                                )

                                # For exporting example image comparisons
                                safe_filename = (
                                    instance_self._create_safe_filename_with_confidence(
                                        line_text,
                                        inference_server_text,
                                        int(line_conf),
                                        int(inference_server_conf * 100),
                                        "Inference Server",
                                    )
                                )

                                if SAVE_EXAMPLE_HYBRID_IMAGES:
                                    # Normalize and validate image_name to prevent path traversal attacks
                                    normalized_image_name = os.path.normpath(
                                        image_name + "_hybrid_paddle_inference_server"
                                    )
                                    if (
                                        ".." in normalized_image_name
                                        or "/" in normalized_image_name
                                        or "\\" in normalized_image_name
                                    ):
                                        normalized_image_name = "safe_image"

                                    hybrid_ocr_examples_folder = (
                                        instance_self.output_folder
                                        + f"/hybrid_ocr_examples/{normalized_image_name}"
                                    )
                                    # Validate the constructed path is safe
                                    if not validate_folder_containment(
                                        hybrid_ocr_examples_folder, OUTPUT_FOLDER
                                    ):
                                        raise ValueError(
                                            f"Unsafe hybrid_ocr_examples folder path: {hybrid_ocr_examples_folder}"
                                        )

                                    if not os.path.exists(hybrid_ocr_examples_folder):
                                        os.makedirs(hybrid_ocr_examples_folder)
                                    output_image_path = (
                                        hybrid_ocr_examples_folder
                                        + f"/{safe_filename}.png"
                                    )
                                    cropped_image.save(output_image_path)

                                # Replace with inference-server result in paddle_results format
                                # Update rec_texts, rec_scores, and rec_models for this line
                                rec_texts[i] = inference_server_text
                                rec_scores[i] = inference_server_conf
                                rec_models[i] = "Inference Server"
                                # Ensure page_result is updated with the modified rec_models list
                                page_result["rec_models"] = rec_models
                            else:
                                print(
                                    f"  Line: '{line_text}' (conf: {line_conf:.1f}, words: {paddle_word_count}) -> "
                                    f"Inference-server result '{inference_server_text}' (conf: {inference_server_conf*100:.1f}, words: {inference_server_word_count}) "
                                    f"word count mismatch. Keeping PaddleOCR result."
                                )
                        else:
                            # Inference-server returned empty or no results - keep original PaddleOCR result
                            if line_conf <= confidence_threshold:
                                pass

            return page_results

        modified_paddle_results = _process_page_result_with_hybrid_inference_server_ocr(
            copied_paddle_results,
            image,
            img_width,
            img_height,
            input_image_width,
            input_image_height,
            confidence_threshold,
            image_name,
            self,
            padding,
        )

        return modified_paddle_results

    def perform_ocr(
        self, image: Union[str, Image.Image, np.ndarray], ocr: Optional[Any] = None
    ) -> List[OCRResult]:
        """
        Performs OCR on the given image using the configured engine.
        """
        if isinstance(image, str):
            image_path = image
            image_name = os.path.basename(image)
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image_path = ""
            image_name = "unknown_image_name"

        # Pre-process image
        # Store original dimensions BEFORE preprocessing (needed for coordinate conversion)
        original_image_width = None
        original_image_height = None
        original_image_for_visualization = (
            None  # Store original image for visualization
        )

        if PREPROCESS_LOCAL_OCR_IMAGES:
            # print("Pre-processing image...")
            # Get original dimensions before preprocessing
            original_image_width, original_image_height = image.size
            # Store original image for visualization (coordinates are in original space)
            original_image_for_visualization = image.copy()
            image, preprocessing_metadata = self.image_preprocessor.preprocess_image(
                image
            )
            if SAVE_PREPROCESS_IMAGES:
                # print("Saving pre-processed image...")
                image_basename = os.path.basename(image_name)
                output_path = os.path.join(
                    self.output_folder,
                    "preprocessed_images",
                    image_basename + "_preprocessed_image.png",
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path)
                # print(f"Pre-processed image saved to {output_path}")
        else:
            preprocessing_metadata = dict()
            original_image_width, original_image_height = image.size
            # When preprocessing is disabled, the current image is the original
            original_image_for_visualization = image.copy()

        image_width, image_height = image.size

        # Store original image for line-to-word conversion when PaddleOCR processes original image
        original_image_for_cropping = None
        paddle_processed_original = False

        # Note: In testing I haven't seen that this necessarily improves results
        if self.ocr_engine == "hybrid-paddle":
            try:
                pass
            except Exception as e:
                raise ImportError(
                    f"Error importing PaddleOCR: {e}. Please install it using 'pip install paddleocr paddlepaddle' in your python environment and retry."
                )

            # Try hybrid with original image for cropping:
            ocr_data = self._perform_hybrid_ocr(image, image_name=image_name)

        elif self.ocr_engine == "hybrid-vlm":
            # Try hybrid VLM with original image for cropping:
            ocr_data = self._perform_hybrid_ocr(image, image_name=image_name)

        elif self.ocr_engine == "vlm":
            # VLM page-level OCR - sends whole page to VLM and gets structured line-level results
            # Use original image (before preprocessing) for VLM since coordinates should be in original space
            vlm_image = (
                original_image_for_visualization
                if original_image_for_visualization is not None
                else image
            )
            ocr_data = _vlm_page_ocr_predict(
                vlm_image, image_name=image_name, output_folder=self.output_folder
            )
            # VLM returns data already in the expected format, so no conversion needed

        elif self.ocr_engine == "inference-server":
            # Inference-server page-level OCR - sends whole page to inference-server API and gets structured line-level results
            # Use original image (before preprocessing) for inference-server since coordinates should be in original space
            inference_server_image = (
                original_image_for_visualization
                if original_image_for_visualization is not None
                else image
            )
            ocr_data = _inference_server_page_ocr_predict(
                inference_server_image,
                image_name=image_name,
                normalised_coords_range=999,
                output_folder=self.output_folder,
            )
            # Inference-server returns data already in the expected format, so no conversion needed

        elif self.ocr_engine == "tesseract":

            ocr_data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config=self.tesseract_config,
                lang=self.tesseract_lang,  # Ensure the Tesseract language data (e.g., fra.traineddata) is installed on your system.
            )

            if TESSERACT_WORD_LEVEL_OCR is False:
                ocr_df = pd.DataFrame(ocr_data)

                # Filter out invalid entries (confidence == -1)
                ocr_df = ocr_df[ocr_df.conf != -1]

                # Group by line and aggregate text
                line_groups = ocr_df.groupby(["block_num", "par_num", "line_num"])

                ocr_data = line_groups.apply(self._calculate_line_bbox).reset_index()

                # Convert DataFrame to dictionary of lists format expected by downstream code
                ocr_data = {
                    "text": ocr_data["text"].tolist(),
                    "left": ocr_data["left"].astype(int).tolist(),
                    "top": ocr_data["top"].astype(int).tolist(),
                    "width": ocr_data["width"].astype(int).tolist(),
                    "height": ocr_data["height"].astype(int).tolist(),
                    "conf": ocr_data["conf"].tolist(),
                    "model": ["Tesseract"] * len(ocr_data),  # Add model field
                }

        elif (
            self.ocr_engine == "paddle"
            or self.ocr_engine == "hybrid-paddle-vlm"
            or self.ocr_engine == "hybrid-paddle-inference-server"
        ):

            if ocr is None:
                if hasattr(self, "paddle_ocr") and self.paddle_ocr is not None:
                    ocr = self.paddle_ocr
                else:
                    raise ValueError(
                        "No OCR object provided and 'paddle_ocr' is not initialised."
                    )

            try:
                pass
            except Exception as e:
                raise ImportError(
                    f"Error importing PaddleOCR: {e}. Please install it using 'pip install paddleocr paddlepaddle' in your python environment and retry."
                )

            if not image_path:
                image_np = np.array(image)  # image_processed

                # Check that sizes are the same
                image_np_height, image_np_width = image_np.shape[:2]
                if image_np_width != image_width or image_np_height != image_height:
                    raise ValueError(
                        f"Image size mismatch: {image_np_width}x{image_np_height} != {image_width}x{image_height}"
                    )

                # PaddleOCR may need an RGB image. Ensure it has 3 channels.
                if len(image_np.shape) == 2:
                    image_np = np.stack([image_np] * 3, axis=-1)
                else:
                    image_np = np.array(image)

                paddle_results = ocr.predict(image_np)
                # PaddleOCR processed the preprocessed image
                paddle_processed_original = False

                # Store the exact image that PaddleOCR processed (convert numpy array back to PIL Image)
                # This ensures we crop from the exact same image PaddleOCR analyzed
                if len(image_np.shape) == 3:
                    paddle_processed_image = Image.fromarray(image_np.astype(np.uint8))
                else:
                    paddle_processed_image = Image.fromarray(image_np.astype(np.uint8))
            else:
                # When using image path, load image to get dimensions
                temp_image = Image.open(image_path)

                # For file path, use the original dimensions (before preprocessing)
                # original_image_width and original_image_height are already set above
                paddle_results = ocr.predict(image_path)
                # PaddleOCR processed the original image from file path
                paddle_processed_original = True
                # Store the original image for cropping
                original_image_for_cropping = temp_image.copy()
                # Store the exact image that PaddleOCR processed (from file path)
                paddle_processed_image = temp_image.copy()

            # Save PaddleOCR visualization with bounding boxes
            if paddle_results and SAVE_PAGE_OCR_VISUALISATIONS is True:

                for res in paddle_results:
                    # self.output_folder is already validated and normalized at construction time
                    paddle_viz_folder = os.path.join(
                        self.output_folder, "paddle_visualisations"
                    )
                    # Double-check the constructed path is safe
                    if not validate_folder_containment(
                        paddle_viz_folder, OUTPUT_FOLDER
                    ):
                        raise ValueError(
                            f"Unsafe paddle visualisations folder path: {paddle_viz_folder}"
                        )

                    os.makedirs(paddle_viz_folder, exist_ok=True)
                    res.save_to_img(paddle_viz_folder)

            if self.ocr_engine == "hybrid-paddle-vlm":

                modified_paddle_results = self._perform_hybrid_paddle_vlm_ocr(
                    paddle_processed_image,  # Use the exact image PaddleOCR processed
                    ocr=ocr,
                    paddle_results=copy.deepcopy(paddle_results),
                    image_name=image_name,
                    input_image_width=original_image_width,
                    input_image_height=original_image_height,
                )

            elif self.ocr_engine == "hybrid-paddle-inference-server":

                modified_paddle_results = self._perform_hybrid_paddle_inference_server_ocr(
                    paddle_processed_image,  # Use the exact image PaddleOCR processed
                    ocr=ocr,
                    paddle_results=copy.deepcopy(paddle_results),
                    image_name=image_name,
                    input_image_width=original_image_width,
                    input_image_height=original_image_height,
                )
            else:
                modified_paddle_results = copy.deepcopy(paddle_results)

            ocr_data = self._convert_paddle_to_tesseract_format(
                modified_paddle_results,
                input_image_width=original_image_width,
                input_image_height=original_image_height,
            )

            if SAVE_PAGE_OCR_VISUALISATIONS is True:
                # Save output to image with identified bounding boxes
                # Use original image since coordinates are in original image space
                # Prefer original_image_for_cropping (when PaddleOCR processed from file path),
                # otherwise use original_image_for_visualization (stored before preprocessing)
                viz_image = (
                    original_image_for_cropping
                    if original_image_for_cropping is not None
                    else (
                        original_image_for_visualization
                        if original_image_for_visualization is not None
                        else image
                    )
                )
                if isinstance(viz_image, Image.Image):
                    # Convert PIL Image to numpy array in BGR format for OpenCV
                    image_cv = cv2.cvtColor(np.array(viz_image), cv2.COLOR_RGB2BGR)
                else:
                    image_cv = np.array(viz_image)
                    if len(image_cv.shape) == 2:
                        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2BGR)
                    elif len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
                        # Assume RGB, convert to BGR
                        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

                # Draw all bounding boxes on the image
                for i in range(len(ocr_data["text"])):
                    left = int(ocr_data["left"][i])
                    top = int(ocr_data["top"][i])
                    width = int(ocr_data["width"][i])
                    height = int(ocr_data["height"][i])
                    # Ensure coordinates are within image bounds
                    left = max(0, min(left, image_cv.shape[1] - 1))
                    top = max(0, min(top, image_cv.shape[0] - 1))
                    right = max(left + 1, min(left + width, image_cv.shape[1]))
                    bottom = max(top + 1, min(top + height, image_cv.shape[0]))
                    cv2.rectangle(
                        image_cv, (left, top), (right, bottom), (0, 255, 0), 2
                    )

                # Save the visualization once with all boxes drawn
                paddle_viz_folder = os.path.join(
                    self.output_folder, "paddle_visualisations"
                )
                # Double-check the constructed path is safe
                if not validate_folder_containment(paddle_viz_folder, OUTPUT_FOLDER):
                    raise ValueError(
                        f"Unsafe paddle visualisations folder path: {paddle_viz_folder}"
                    )

                os.makedirs(paddle_viz_folder, exist_ok=True)

                # Generate safe filename
                if image_name:
                    base_name = os.path.splitext(os.path.basename(image_name))[0]
                    # Increment the number at the end of base_name
                    # This converts zero-indexed input to one-indexed output
                    incremented_base_name = base_name
                    # Find the number pattern at the end
                    # Matches patterns like: _0, _00, 0, 00, etc.
                    pattern = r"(\d+)$"
                    match = re.search(pattern, base_name)
                    if match:
                        number_str = match.group(1)
                        number = int(number_str)
                        incremented_number = number + 1
                        # Preserve the same number of digits (padding with zeros if needed)
                        incremented_str = str(incremented_number).zfill(len(number_str))
                        incremented_base_name = re.sub(
                            pattern, incremented_str, base_name
                        )
                    # Sanitize filename to avoid issues with special characters
                    incremented_base_name = safe_sanitize_text(
                        incremented_base_name, max_length=50
                    )
                    filename = f"{incremented_base_name}_initial_bounding_boxes.jpg"
                else:
                    timestamp = int(time.time())
                    filename = f"initial_bounding_boxes_{timestamp}.jpg"

                output_path = os.path.join(paddle_viz_folder, filename)
                cv2.imwrite(output_path, image_cv)

        else:
            raise RuntimeError(f"Unsupported OCR engine: {self.ocr_engine}")

        # Always check for scale_factor, even if preprocessing_metadata is empty
        # This ensures rescaling happens correctly when preprocessing was applied
        scale_factor = (
            preprocessing_metadata.get("scale_factor", 1.0)
            if preprocessing_metadata
            else 1.0
        )
        if scale_factor != 1.0:
            # Skip rescaling for PaddleOCR since _convert_paddle_to_tesseract_format
            # already scales coordinates directly to original image dimensions
            # hybrid-paddle-vlm also uses PaddleOCR and converts to original space
            # Skip rescaling for VLM since it returns coordinates in original image space
            if (
                self.ocr_engine == "paddle"
                or self.ocr_engine == "hybrid-paddle-vlm"
                or self.ocr_engine == "hybrid-paddle-inference-server"
                or self.ocr_engine == "vlm"
                or self.ocr_engine == "inference-server"
            ):
                pass
                # print(f"Skipping rescale_ocr_data for PaddleOCR/VLM (already scaled to original dimensions)")
            else:
                # print("rescaling ocr_data with scale_factor: ", scale_factor)
                ocr_data = rescale_ocr_data(ocr_data, scale_factor)

        # Convert line-level results to word-level if configured and needed
        if CONVERT_LINE_TO_WORD_LEVEL and self._is_line_level_data(ocr_data):
            # print("Converting line-level OCR results to word-level...")

            # Check if coordinates need to be scaled to match the image we're cropping from
            # For PaddleOCR: _convert_paddle_to_tesseract_format converts coordinates to original image space
            #   - If PaddleOCR processed the original image (image_path provided), crop from original image (no scaling)
            #   - If PaddleOCR processed the preprocessed image (no image_path), scale coordinates to preprocessed space and crop from preprocessed image
            # For Tesseract: OCR runs on preprocessed image
            #   - If scale_factor != 1.0, rescale_ocr_data converted coordinates to original space, so crop from original image
            #   - If scale_factor == 1.0, coordinates are still in preprocessed space, so crop from preprocessed image

            needs_scaling = False
            crop_image = image  # Default to preprocessed image
            crop_image_width = image_width
            crop_image_height = image_height

            if (
                PREPROCESS_LOCAL_OCR_IMAGES
                and original_image_width
                and original_image_height
            ):
                if (
                    self.ocr_engine == "paddle"
                    or self.ocr_engine == "hybrid-paddle-vlm"
                    or self.ocr_engine == "hybrid-paddle-inference-server"
                ):
                    # PaddleOCR coordinates are converted to original space by _convert_paddle_to_tesseract_format
                    # hybrid-paddle-vlm also uses PaddleOCR and converts to original space
                    if paddle_processed_original:
                        # PaddleOCR processed the original image, so crop from original image
                        # No scaling needed - coordinates are already in original space
                        crop_image = original_image_for_cropping
                        crop_image_width = original_image_width
                        crop_image_height = original_image_height
                        needs_scaling = False
                    else:
                        # PaddleOCR processed the preprocessed image, so scale coordinates to preprocessed space
                        needs_scaling = True
                elif self.ocr_engine == "vlm" or self.ocr_engine == "inference-server":
                    # VLM returns coordinates in original image space (since we pass original image to VLM)
                    # So we need to crop from the original image, not the preprocessed image
                    if original_image_for_visualization is not None:
                        # Coordinates are in original space, so crop from original image
                        crop_image = original_image_for_visualization
                        crop_image_width = original_image_width
                        crop_image_height = original_image_height
                        needs_scaling = False
                    else:
                        # Fallback to preprocessed image if original not available
                        needs_scaling = False
                elif self.ocr_engine == "tesseract":
                    # For Tesseract: if scale_factor != 1.0, rescale_ocr_data converted coordinates to original space
                    # So we need to crop from the original image, not the preprocessed image
                    if (
                        scale_factor != 1.0
                        and original_image_for_visualization is not None
                    ):
                        # Coordinates are in original space, so crop from original image
                        crop_image = original_image_for_visualization
                        crop_image_width = original_image_width
                        crop_image_height = original_image_height
                        needs_scaling = False
                    else:
                        # scale_factor == 1.0, so coordinates are still in preprocessed space
                        # Crop from preprocessed image - no scaling needed
                        needs_scaling = False

            if needs_scaling:
                # Calculate scale factors from original to preprocessed
                scale_x = image_width / original_image_width
                scale_y = image_height / original_image_height
                # Scale coordinates to preprocessed image space for cropping
                scaled_ocr_data = {
                    "text": ocr_data["text"],
                    "left": [x * scale_x for x in ocr_data["left"]],
                    "top": [y * scale_y for y in ocr_data["top"]],
                    "width": [w * scale_x for w in ocr_data["width"]],
                    "height": [h * scale_y for h in ocr_data["height"]],
                    "conf": ocr_data["conf"],
                    "model": ocr_data["model"],
                }
                ocr_data = self._convert_line_to_word_level(
                    scaled_ocr_data,
                    crop_image_width,
                    crop_image_height,
                    crop_image,
                    image_name=image_name,
                )
                # Scale word-level results back to original image space
                scale_factor_x = original_image_width / image_width
                scale_factor_y = original_image_height / image_height
                for i in range(len(ocr_data["left"])):
                    ocr_data["left"][i] = ocr_data["left"][i] * scale_factor_x
                    ocr_data["top"][i] = ocr_data["top"][i] * scale_factor_y
                    ocr_data["width"][i] = ocr_data["width"][i] * scale_factor_x
                    ocr_data["height"][i] = ocr_data["height"][i] * scale_factor_y
            else:
                # No scaling needed - coordinates match the crop image space
                ocr_data = self._convert_line_to_word_level(
                    ocr_data,
                    crop_image_width,
                    crop_image_height,
                    crop_image,
                    image_name=image_name,
                )

        # The rest of your processing pipeline now works for both engines
        ocr_result = ocr_data

        # Filter out empty strings and low confidence results
        valid_indices = [
            i
            for i, text in enumerate(ocr_result["text"])
            if text.strip() and int(ocr_result["conf"][i]) > 0
        ]

        # Determine default model based on OCR engine if model field is not present
        if "model" in ocr_result:
            # Model field exists and has correct length - use it
            def get_model(idx):
                return ocr_result["model"][idx]

        else:
            # Model field not present or incorrect length - use default based on engine
            default_model = (
                "Tesseract"
                if self.ocr_engine == "tesseract"
                else (
                    "Paddle"
                    if self.ocr_engine == "paddle"
                    else (
                        "Tesseract"
                        if self.ocr_engine == "hybrid-paddle"
                        else (
                            "Tesseract"
                            if self.ocr_engine == "hybrid-vlm"
                            else (
                                "Paddle"
                                if self.ocr_engine == "hybrid-paddle-vlm"
                                else (
                                    "Paddle"
                                    if self.ocr_engine
                                    == "hybrid-paddle-inference-server"
                                    else (
                                        "VLM"
                                        if self.ocr_engine == "vlm"
                                        else (
                                            "Inference Server"
                                            if self.ocr_engine == "inference-server"
                                            else None
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )

            def get_model(idx):
                return default_model

        output = [
            OCRResult(
                text=clean_unicode_text(ocr_result["text"][i]),
                left=ocr_result["left"][i],
                top=ocr_result["top"][i],
                width=ocr_result["width"][i],
                height=ocr_result["height"][i],
                conf=round(float(ocr_result["conf"][i]), 0),
                model=get_model(i),
            )
            for i in valid_indices
        ]

        return output

    def analyze_text(
        self,
        line_level_ocr_results: List[OCRResult],
        ocr_results_with_words: Dict[str, Dict],
        chosen_redact_comprehend_entities: List[str],
        pii_identification_method: str = LOCAL_PII_OPTION,
        comprehend_client="",
        custom_entities: List[str] = custom_entities,
        language: Optional[str] = DEFAULT_LANGUAGE,
        nlp_analyser: AnalyzerEngine = None,
        **text_analyzer_kwargs,
    ) -> List[CustomImageRecognizerResult]:

        page_text = ""
        page_text_mapping = list()
        all_text_line_results = list()
        comprehend_query_number = 0

        if not nlp_analyser:
            nlp_analyser = self.analyzer_engine

        # Collect all text and create mapping
        for i, line_level_ocr_result in enumerate(line_level_ocr_results):
            if page_text:
                page_text += " "
            start_pos = len(page_text)
            page_text += line_level_ocr_result.text
            # Note: We're not passing line_characters here since it's not needed for this use case
            page_text_mapping.append((start_pos, i, line_level_ocr_result, None))

        # Determine language for downstream services
        aws_language = language or getattr(self, "language", None) or "en"

        valid_language_entities = nlp_analyser.registry.get_supported_entities(
            languages=[language]
        )
        if "CUSTOM" not in valid_language_entities:
            valid_language_entities.append("CUSTOM")
        if "CUSTOM_FUZZY" not in valid_language_entities:
            valid_language_entities.append("CUSTOM_FUZZY")

        # Process using either Local or AWS Comprehend
        if pii_identification_method == LOCAL_PII_OPTION:

            language_supported_entities = filter_entities_for_language(
                custom_entities, valid_language_entities, language
            )

            if language_supported_entities:
                text_analyzer_kwargs["entities"] = language_supported_entities

            else:
                out_message = f"No relevant entities supported for language: {language}"
                print(out_message)
                raise Warning(out_message)

            analyzer_result = nlp_analyser.analyze(
                text=page_text, language=language, **text_analyzer_kwargs
            )
            all_text_line_results = map_back_entity_results(
                analyzer_result, page_text_mapping, all_text_line_results
            )

        elif pii_identification_method == AWS_PII_OPTION:

            # Handle custom entities first
            if custom_entities:
                custom_redact_entities = [
                    entity
                    for entity in chosen_redact_comprehend_entities
                    if entity in custom_entities
                ]

                if custom_redact_entities:
                    # Filter entities to only include those supported by the language
                    language_supported_entities = filter_entities_for_language(
                        custom_redact_entities, valid_language_entities, language
                    )

                    if language_supported_entities:
                        text_analyzer_kwargs["entities"] = language_supported_entities

                    page_analyser_result = nlp_analyser.analyze(
                        text=page_text, language=language, **text_analyzer_kwargs
                    )
                    all_text_line_results = map_back_entity_results(
                        page_analyser_result, page_text_mapping, all_text_line_results
                    )

            # Process text in batches for AWS Comprehend
            current_batch = ""
            current_batch_mapping = list()
            batch_char_count = 0
            batch_word_count = 0

            for i, text_line in enumerate(
                line_level_ocr_results
            ):  # Changed from line_level_text_results_list
                words = text_line.text.split()
                word_start_positions = list()
                current_pos = 0

                for word in words:
                    word_start_positions.append(current_pos)
                    current_pos += len(word) + 1

                for word_idx, word in enumerate(words):
                    new_batch_char_count = len(current_batch) + len(word) + 1

                    if batch_word_count >= 50 or new_batch_char_count >= 200:
                        # Process current batch
                        all_text_line_results = do_aws_comprehend_call(
                            current_batch,
                            current_batch_mapping,
                            comprehend_client,
                            aws_language,
                            text_analyzer_kwargs.get("allow_list", []),
                            chosen_redact_comprehend_entities,
                            all_text_line_results,
                        )
                        comprehend_query_number += 1

                        # Reset batch
                        current_batch = word
                        batch_word_count = 1
                        batch_char_count = len(word)
                        current_batch_mapping = [
                            (0, i, text_line, None, word_start_positions[word_idx])
                        ]
                    else:
                        if current_batch:
                            current_batch += " "
                            batch_char_count += 1
                        current_batch += word
                        batch_char_count += len(word)
                        batch_word_count += 1

                        if (
                            not current_batch_mapping
                            or current_batch_mapping[-1][1] != i
                        ):
                            current_batch_mapping.append(
                                (
                                    batch_char_count - len(word),
                                    i,
                                    text_line,
                                    None,
                                    word_start_positions[word_idx],
                                )
                            )

            # Process final batch if any
            if current_batch:
                all_text_line_results = do_aws_comprehend_call(
                    current_batch,
                    current_batch_mapping,
                    comprehend_client,
                    aws_language,
                    text_analyzer_kwargs.get("allow_list", []),
                    chosen_redact_comprehend_entities,
                    all_text_line_results,
                )
                comprehend_query_number += 1

        # Process results and create bounding boxes
        combined_results = list()
        for i, text_line in enumerate(line_level_ocr_results):
            line_results = next(
                (results for idx, results in all_text_line_results if idx == i), []
            )
            if line_results and i < len(ocr_results_with_words):
                child_level_key = list(ocr_results_with_words.keys())[i]
                ocr_results_with_words_line_level = ocr_results_with_words[
                    child_level_key
                ]

                for result in line_results:
                    bbox_results = self.map_analyzer_results_to_bounding_boxes(
                        [result],
                        [
                            OCRResult(
                                text=text_line.text[result.start : result.end],
                                left=text_line.left,
                                top=text_line.top,
                                width=text_line.width,
                                height=text_line.height,
                                conf=text_line.conf,
                            )
                        ],
                        text_line.text,
                        text_analyzer_kwargs.get("allow_list", []),
                        ocr_results_with_words_line_level,
                    )
                    combined_results.extend(bbox_results)

        return combined_results, comprehend_query_number

    @staticmethod
    def map_analyzer_results_to_bounding_boxes(
        text_analyzer_results: List[RecognizerResult],
        redaction_relevant_ocr_results: List[OCRResult],
        full_text: str,
        allow_list: List[str],
        ocr_results_with_words_child_info: Dict[str, Dict],
    ) -> List[CustomImageRecognizerResult]:
        redaction_bboxes = list()

        for redaction_relevant_ocr_result in redaction_relevant_ocr_results:

            line_text = ocr_results_with_words_child_info["text"]
            line_length = len(line_text)
            redaction_text = redaction_relevant_ocr_result.text

            for redaction_result in text_analyzer_results:
                # Check if the redaction text is not in the allow list

                if redaction_text not in allow_list:

                    # Adjust start and end to be within line bounds
                    start_in_line = max(0, redaction_result.start)
                    end_in_line = min(line_length, redaction_result.end)

                    # Get the matched text from this line
                    matched_text = line_text[start_in_line:end_in_line]
                    matched_text.split()

                    # Find the corresponding words in the OCR results
                    matching_word_boxes = list()

                    current_position = 0

                    for word_info in ocr_results_with_words_child_info.get("words", []):
                        word_text = word_info["text"]
                        word_length = len(word_text)

                        word_start = current_position
                        word_end = current_position + word_length

                        # Update current position for the next word
                        current_position += (
                            word_length + 1
                        )  # +1 for the space after the word

                        # Check if the word's bounding box is within the start and end bounds
                        if word_start >= start_in_line and word_end <= (
                            end_in_line + 1
                        ):
                            matching_word_boxes.append(word_info["bounding_box"])

                    if matching_word_boxes:
                        # Calculate the combined bounding box for all matching words
                        left = min(box[0] for box in matching_word_boxes)
                        top = min(box[1] for box in matching_word_boxes)
                        right = max(box[2] for box in matching_word_boxes)
                        bottom = max(box[3] for box in matching_word_boxes)

                        redaction_bboxes.append(
                            CustomImageRecognizerResult(
                                entity_type=redaction_result.entity_type,
                                start=start_in_line,
                                end=end_in_line,
                                score=round(redaction_result.score, 2),
                                left=left,
                                top=top,
                                width=right - left,
                                height=bottom - top,
                                text=matched_text,
                            )
                        )

        return redaction_bboxes

    @staticmethod
    def remove_space_boxes(ocr_result: dict) -> dict:
        """Remove OCR bboxes that are for spaces.
        :param ocr_result: OCR results (raw or thresholded).
        :return: OCR results with empty words removed.
        """
        # Get indices of items with no text
        idx = list()
        for i, text in enumerate(ocr_result["text"]):
            is_not_space = text.isspace() is False
            if text != "" and is_not_space:
                idx.append(i)

        # Only retain items with text
        filtered_ocr_result = {}
        for key in list(ocr_result.keys()):
            filtered_ocr_result[key] = [ocr_result[key][i] for i in idx]

        return filtered_ocr_result

    @staticmethod
    def _scale_bbox_results(
        ocr_result: Dict[str, List[Union[int, str]]], scale_factor: float
    ) -> Dict[str, float]:
        """Scale down the bounding box results based on a scale percentage.
        :param ocr_result: OCR results (raw).
        :param scale_percent: Scale percentage for resizing the bounding box.
        :return: OCR results (scaled).
        """
        scaled_results = deepcopy(ocr_result)
        coordinate_keys = ["left", "top"]
        dimension_keys = ["width", "height"]

        for coord_key in coordinate_keys:
            scaled_results[coord_key] = [
                int(np.ceil((x) / (scale_factor))) for x in scaled_results[coord_key]
            ]

        for dim_key in dimension_keys:
            scaled_results[dim_key] = [
                max(1, int(np.ceil(x / (scale_factor))))
                for x in scaled_results[dim_key]
            ]
        return scaled_results

    @staticmethod
    def estimate_x_offset(full_text: str, start: int) -> int:
        # Estimate the x-offset based on character position
        # This is a simple estimation and might need refinement for variable-width fonts
        return int(start / len(full_text) * len(full_text))

    def estimate_width(self, ocr_result: OCRResult, start: int, end: int) -> int:
        # Extract the relevant text portion
        relevant_text = ocr_result.text[start:end]

        # If the relevant text is the same as the full text, return the full width
        if relevant_text == ocr_result.text:
            return ocr_result.width

        # Estimate width based on the proportion of the relevant text length to the total text length
        total_text_length = len(ocr_result.text)
        relevant_text_length = len(relevant_text)

        if total_text_length == 0:
            return 0  # Avoid division by zero

        # Proportion of the relevant text to the total text
        proportion = relevant_text_length / total_text_length

        # Estimate the width based on the proportion
        estimated_width = int(proportion * ocr_result.width)

        return estimated_width


def bounding_boxes_overlap(box1: List, box2: List):
    """Check if two bounding boxes overlap."""
    return (
        box1[0] < box2[2]
        and box2[0] < box1[2]
        and box1[1] < box2[3]
        and box2[1] < box1[3]
    )


def map_back_entity_results(
    page_analyser_result: dict,
    page_text_mapping: dict,
    all_text_line_results: List[Tuple],
):
    for entity in page_analyser_result:
        entity_start = entity.start
        entity_end = entity.end

        # Track if the entity has been added to any line
        added_to_line = False

        for batch_start, line_idx, original_line, chars in page_text_mapping:
            batch_end = batch_start + len(original_line.text)

            # Check if the entity overlaps with the current line
            if (
                batch_start < entity_end and batch_end > entity_start
            ):  # Overlap condition
                relative_start = max(
                    0, entity_start - batch_start
                )  # Adjust start relative to the line
                relative_end = min(
                    entity_end - batch_start, len(original_line.text)
                )  # Adjust end relative to the line

                # Create a new adjusted entity
                adjusted_entity = copy.deepcopy(entity)
                adjusted_entity.start = relative_start
                adjusted_entity.end = relative_end

                # Check if this line already has an entry
                existing_entry = next(
                    (entry for idx, entry in all_text_line_results if idx == line_idx),
                    None,
                )

                if existing_entry is None:
                    all_text_line_results.append((line_idx, [adjusted_entity]))
                else:
                    existing_entry.append(
                        adjusted_entity
                    )  # Append to the existing list of entities

                added_to_line = True

        # If the entity spans multiple lines, you may want to handle that here
        if not added_to_line:
            # Handle cases where the entity does not fit in any line (optional)
            print(f"Entity '{entity}' does not fit in any line.")

    return all_text_line_results


def map_back_comprehend_entity_results(
    response: object,
    current_batch_mapping: List[Tuple],
    allow_list: List[str],
    chosen_redact_comprehend_entities: List[str],
    all_text_line_results: List[Tuple],
):
    if not response or "Entities" not in response:
        return all_text_line_results

    for entity in response["Entities"]:
        if entity.get("Type") not in chosen_redact_comprehend_entities:
            continue

        entity_start = entity["BeginOffset"]
        entity_end = entity["EndOffset"]

        # Track if the entity has been added to any line
        added_to_line = False

        # Find the correct line and offset within that line
        for (
            batch_start,
            line_idx,
            original_line,
            chars,
            line_offset,
        ) in current_batch_mapping:
            batch_end = batch_start + len(original_line.text[line_offset:])

            # Check if the entity overlaps with the current line
            if (
                batch_start < entity_end and batch_end > entity_start
            ):  # Overlap condition
                # Calculate the absolute position within the line
                relative_start = max(0, entity_start - batch_start + line_offset)
                relative_end = min(
                    entity_end - batch_start + line_offset, len(original_line.text)
                )

                result_text = original_line.text[relative_start:relative_end]

                if result_text not in allow_list:
                    adjusted_entity = entity.copy()
                    adjusted_entity["BeginOffset"] = (
                        relative_start  # Now relative to the full line
                    )
                    adjusted_entity["EndOffset"] = relative_end

                    recogniser_entity = recognizer_result_from_dict(adjusted_entity)

                    existing_entry = next(
                        (
                            entry
                            for idx, entry in all_text_line_results
                            if idx == line_idx
                        ),
                        None,
                    )
                    if existing_entry is None:
                        all_text_line_results.append((line_idx, [recogniser_entity]))
                    else:
                        existing_entry.append(
                            recogniser_entity
                        )  # Append to the existing list of entities

                added_to_line = True

        # Optional: Handle cases where the entity does not fit in any line
        if not added_to_line:
            print(f"Entity '{entity}' does not fit in any line.")

    return all_text_line_results


def do_aws_comprehend_call(
    current_batch: str,
    current_batch_mapping: List[Tuple],
    comprehend_client: botocore.client.BaseClient,
    language: str,
    allow_list: List[str],
    chosen_redact_comprehend_entities: List[str],
    all_text_line_results: List[Tuple],
):
    if not current_batch:
        return all_text_line_results

    max_retries = 3
    retry_delay = 3

    for attempt in range(max_retries):
        try:
            response = comprehend_client.detect_pii_entities(
                Text=current_batch.strip(), LanguageCode=language
            )

            all_text_line_results = map_back_comprehend_entity_results(
                response,
                current_batch_mapping,
                allow_list,
                chosen_redact_comprehend_entities,
                all_text_line_results,
            )

            return all_text_line_results

        except Exception as e:
            if attempt == max_retries - 1:
                print("AWS Comprehend calls failed due to", e)
                raise
            time.sleep(retry_delay)


def run_page_text_redaction(
    language: str,
    chosen_redact_entities: List[str],
    chosen_redact_comprehend_entities: List[str],
    line_level_text_results_list: List[str],
    line_characters: List,
    page_analyser_results: List = list(),
    page_analysed_bounding_boxes: List = list(),
    comprehend_client=None,
    allow_list: List[str] = None,
    pii_identification_method: str = LOCAL_PII_OPTION,
    nlp_analyser: AnalyzerEngine = None,
    score_threshold: float = 0.0,
    custom_entities: List[str] = None,
    comprehend_query_number: int = 0,
):
    """
    This function performs text redaction on a page based on the specified language and chosen entities.

    Args:
        language (str): The language code for the text being processed.
        chosen_redact_entities (List[str]): A list of entities to be redacted from the text.
        chosen_redact_comprehend_entities (List[str]): A list of entities identified by AWS Comprehend for redaction.
        line_level_text_results_list (List[str]): A list of text lines extracted from the page.
        line_characters (List): A list of character-level information for each line of text.
        page_analyser_results (List, optional): Results from previous page analysis. Defaults to an empty list.
        page_analysed_bounding_boxes (List, optional): Bounding boxes for the analysed page. Defaults to an empty list.
        comprehend_client: The AWS Comprehend client for making API calls. Defaults to None.
        allow_list (List[str], optional): A list of allowed entities that should not be redacted. Defaults to None.
        pii_identification_method (str, optional): The method used for PII identification. Defaults to LOCAL_PII_OPTION.
        nlp_analyser (AnalyzerEngine, optional): The NLP analyzer engine used for local analysis. Defaults to None.
        score_threshold (float, optional): The threshold score for entity detection. Defaults to 0.0.
        custom_entities (List[str], optional): A list of custom entities for redaction. Defaults to None.
        comprehend_query_number (int, optional): A counter for the number of Comprehend queries made. Defaults to 0.
    """

    page_text = ""
    page_text_mapping = list()
    all_text_line_results = list()
    comprehend_query_number = 0

    # Collect all text from the page
    for i, text_line in enumerate(line_level_text_results_list):
        if chosen_redact_entities:
            if page_text:
                page_text += " "

            start_pos = len(page_text)
            page_text += text_line.text
            page_text_mapping.append((start_pos, i, text_line, line_characters[i]))

    valid_language_entities = nlp_analyser.registry.get_supported_entities(
        languages=[language]
    )
    if "CUSTOM" not in valid_language_entities:
        valid_language_entities.append("CUSTOM")
    if "CUSTOM_FUZZY" not in valid_language_entities:
        valid_language_entities.append("CUSTOM_FUZZY")

    # Process based on identification method
    if pii_identification_method == LOCAL_PII_OPTION:
        if not nlp_analyser:
            raise ValueError("nlp_analyser is required for Local identification method")

        language_supported_entities = filter_entities_for_language(
            chosen_redact_entities, valid_language_entities, language
        )

        page_analyser_result = nlp_analyser.analyze(
            text=page_text,
            language=language,
            entities=language_supported_entities,
            score_threshold=score_threshold,
            return_decision_process=True,
            allow_list=allow_list,
        )

        all_text_line_results = map_back_entity_results(
            page_analyser_result, page_text_mapping, all_text_line_results
        )

    elif pii_identification_method == AWS_PII_OPTION:

        # Process custom entities if any
        if custom_entities:
            custom_redact_entities = [
                entity
                for entity in chosen_redact_comprehend_entities
                if entity in custom_entities
            ]

            language_supported_entities = filter_entities_for_language(
                custom_redact_entities, valid_language_entities, language
            )

            if language_supported_entities:
                page_analyser_result = nlp_analyser.analyze(
                    text=page_text,
                    language=language,
                    entities=language_supported_entities,
                    score_threshold=score_threshold,
                    return_decision_process=True,
                    allow_list=allow_list,
                )

                all_text_line_results = map_back_entity_results(
                    page_analyser_result, page_text_mapping, all_text_line_results
                )

        current_batch = ""
        current_batch_mapping = list()
        batch_char_count = 0
        batch_word_count = 0

        for i, text_line in enumerate(line_level_text_results_list):
            words = text_line.text.split()
            word_start_positions = list()

            # Calculate word start positions within the line
            current_pos = 0
            for word in words:
                word_start_positions.append(current_pos)
                current_pos += len(word) + 1  # +1 for space

            for word_idx, word in enumerate(words):
                new_batch_char_count = len(current_batch) + len(word) + 1

                if batch_word_count >= 50 or new_batch_char_count >= 200:
                    # Process current batch
                    all_text_line_results = do_aws_comprehend_call(
                        current_batch,
                        current_batch_mapping,
                        comprehend_client,
                        language,
                        allow_list,
                        chosen_redact_comprehend_entities,
                        all_text_line_results,
                    )
                    comprehend_query_number += 1

                    # Start new batch
                    current_batch = word
                    batch_word_count = 1
                    batch_char_count = len(word)
                    current_batch_mapping = [
                        (
                            0,
                            i,
                            text_line,
                            line_characters[i],
                            word_start_positions[word_idx],
                        )
                    ]
                else:
                    if current_batch:
                        current_batch += " "
                        batch_char_count += 1
                    current_batch += word
                    batch_char_count += len(word)
                    batch_word_count += 1

                    if not current_batch_mapping or current_batch_mapping[-1][1] != i:
                        current_batch_mapping.append(
                            (
                                batch_char_count - len(word),
                                i,
                                text_line,
                                line_characters[i],
                                word_start_positions[
                                    word_idx
                                ],  # Add the word's start position within its line
                            )
                        )

        # Process final batch
        if current_batch:
            all_text_line_results = do_aws_comprehend_call(
                current_batch,
                current_batch_mapping,
                comprehend_client,
                language,
                allow_list,
                chosen_redact_comprehend_entities,
                all_text_line_results,
            )
            comprehend_query_number += 1

    # Process results for each line
    for i, text_line in enumerate(line_level_text_results_list):
        line_results = next(
            (results for idx, results in all_text_line_results if idx == i), []
        )

        if line_results:
            text_line_bounding_boxes = merge_text_bounding_boxes(
                line_results, line_characters[i]
            )

            page_analyser_results.extend(line_results)
            page_analysed_bounding_boxes.extend(text_line_bounding_boxes)

    return page_analysed_bounding_boxes


def merge_text_bounding_boxes(
    analyser_results: dict,
    characters: List[LTChar],
    combine_pixel_dist: int = 20,
    vertical_padding: int = 0,
):
    """
    Merge identified bounding boxes containing PII that are very close to one another
    """
    analysed_bounding_boxes = list()
    original_bounding_boxes = list()  # List to hold original bounding boxes

    if len(analyser_results) > 0 and len(characters) > 0:
        # Extract bounding box coordinates for sorting
        bounding_boxes = list()
        for result in analyser_results:
            char_boxes = [
                char.bbox
                for char in characters[result.start : result.end]
                if isinstance(char, LTChar)
            ]
            char_text = [
                char._text
                for char in characters[result.start : result.end]
                if isinstance(char, LTChar)
            ]
            if char_boxes:
                # Calculate the bounding box that encompasses all characters
                left = min(box[0] for box in char_boxes)
                bottom = min(box[1] for box in char_boxes)
                right = max(box[2] for box in char_boxes)
                top = max(box[3] for box in char_boxes) + vertical_padding
                bbox = [left, bottom, right, top]
                bounding_boxes.append(
                    (bottom, left, result, bbox, char_text)
                )  # (y, x, result, bbox, text)

                # Store original bounding boxes
                original_bounding_boxes.append(
                    {
                        "text": "".join(char_text),
                        "boundingBox": bbox,
                        "result": copy.deepcopy(result),
                    }
                )
                # print("Original bounding boxes:", original_bounding_boxes)

        # Sort the results by y-coordinate and then by x-coordinate
        bounding_boxes.sort()

        merged_bounding_boxes = list()
        current_box = None
        current_y = None
        current_result = None
        current_text = list()

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

                if (
                    vertical_diff_bboxes <= 5
                    and horizontal_diff_bboxes <= combine_pixel_dist
                ):
                    # Merge bounding boxes
                    # print("Merging boxes")
                    merged_box = current_box.copy()
                    merged_result = current_result
                    merged_text = current_text.copy()

                    merged_box[2] = next_box[2]  # Extend horizontally
                    merged_box[3] = max(current_box[3], next_box[3])  # Adjust the top
                    merged_result.end = max(
                        current_result.end, result.end
                    )  # Extend text range
                    try:
                        if current_result.entity_type != result.entity_type:
                            merged_result.entity_type = (
                                current_result.entity_type + " - " + result.entity_type
                            )
                        else:
                            merged_result.entity_type = current_result.entity_type
                    except Exception as e:
                        print("Unable to combine result entity types:", e)
                    if current_text:
                        merged_text.append(" ")  # Add space between texts
                    merged_text.extend(text)

                    merged_bounding_boxes.append(
                        {
                            "text": "".join(merged_text),
                            "boundingBox": merged_box,
                            "result": merged_result,
                        }
                    )

                else:
                    # Start a new bounding box
                    current_box = next_box
                    current_y = next_box[1]
                    current_result = result
                    current_text = list(text)

        # Combine original and merged bounding boxes
        analysed_bounding_boxes.extend(original_bounding_boxes)
        analysed_bounding_boxes.extend(merged_bounding_boxes)

        # print("Analysed bounding boxes:", analysed_bounding_boxes)

    return analysed_bounding_boxes


def recreate_page_line_level_ocr_results_with_page(
    page_line_level_ocr_results_with_words: dict,
):
    reconstructed_results = list()

    # Assume all lines belong to the same page, so we can just read it from one item
    # page = next(iter(page_line_level_ocr_results_with_words.values()))["page"]

    page = page_line_level_ocr_results_with_words["page"]

    for line_data in page_line_level_ocr_results_with_words["results"].values():
        bbox = line_data["bounding_box"]
        text = line_data["text"]
        if line_data["line"]:
            line_number = line_data["line"]
        if "conf" in line_data["words"][0]:
            conf = sum(word["conf"] for word in line_data["words"]) / len(
                line_data["words"]
            )
        else:
            conf = 0.0

        # Recreate the OCRResult
        line_result = OCRResult(
            text=text,
            left=bbox[0],
            top=bbox[1],
            width=bbox[2] - bbox[0],
            height=bbox[3] - bbox[1],
            line=line_number,
            conf=round(float(conf), 0),
        )
        reconstructed_results.append(line_result)

    page_line_level_ocr_results_with_page = {
        "page": page,
        "results": reconstructed_results,
    }

    return page_line_level_ocr_results_with_page


def split_words_and_punctuation_from_line(
    line_of_words: List[OCRResult],
) -> List[OCRResult]:
    """
    Takes a list of OCRResult objects and splits words with trailing/leading punctuation.

    For a word like "example.", it creates two new OCRResult objects for "example"
    and "." and estimates their bounding boxes. Words with internal hyphens like
    "high-tech" are preserved.
    """
    # Punctuation that will be split off. Hyphen is not included.

    new_word_list = list()

    for word_result in line_of_words:
        word_text = word_result.text

        # This regex finds a central "core" word, and captures leading and trailing punctuation
        # Handles cases like "(word)." -> group1='(', group2='word', group3='.'
        match = re.match(r"([(\[{]*)(.*?)_?([.,?!:;)\}\]]*)$", word_text)

        # Handle words with internal hyphens that might confuse the regex
        if "-" in word_text and not match.group(2):
            core_part_text = word_text
            leading_punc = ""
            trailing_punc = ""
        elif match:
            leading_punc, core_part_text, trailing_punc = match.groups()
        else:  # Failsafe
            new_word_list.append(word_result)
            continue

        # If no split is needed, just add the original and continue
        if not leading_punc and not trailing_punc:
            new_word_list.append(word_result)
            continue

        # --- A split is required ---
        # Estimate new bounding boxes by proportionally allocating width
        original_width = word_result.width
        if not word_text or original_width == 0:
            continue  # Failsafe

        avg_char_width = original_width / len(word_text)
        current_left = word_result.left

        # Add leading punctuation if it exists
        if leading_punc:
            punc_width = avg_char_width * len(leading_punc)
            new_word_list.append(
                OCRResult(
                    text=leading_punc,
                    left=current_left,
                    top=word_result.top,
                    width=punc_width,
                    height=word_result.height,
                    conf=word_result.conf,
                )
            )
            current_left += punc_width

        # Add the core part of the word
        if core_part_text:
            core_width = avg_char_width * len(core_part_text)
            new_word_list.append(
                OCRResult(
                    text=core_part_text,
                    left=current_left,
                    top=word_result.top,
                    width=core_width,
                    height=word_result.height,
                    conf=word_result.conf,
                )
            )
            current_left += core_width

        # Add trailing punctuation if it exists
        if trailing_punc:
            punc_width = avg_char_width * len(trailing_punc)
            new_word_list.append(
                OCRResult(
                    text=trailing_punc,
                    left=current_left,
                    top=word_result.top,
                    width=punc_width,
                    height=word_result.height,
                    conf=word_result.conf,
                )
            )

    return new_word_list


def create_ocr_result_with_children(
    combined_results: dict, i: int, current_bbox: dict, current_line: list
):
    combined_results["text_line_" + str(i)] = {
        "line": i,
        "text": current_bbox.text,
        "bounding_box": (
            current_bbox.left,
            current_bbox.top,
            current_bbox.left + current_bbox.width,
            current_bbox.top + current_bbox.height,
        ),
        "words": [
            {
                "text": word.text,
                "bounding_box": (
                    word.left,
                    word.top,
                    word.left + word.width,
                    word.top + word.height,
                ),
                "conf": word.conf,
                "model": word.model,
            }
            for word in current_line
        ],
        "conf": current_bbox.conf,
    }
    return combined_results["text_line_" + str(i)]


def combine_ocr_results(
    ocr_results: List[OCRResult],
    x_threshold: float = 50.0,
    y_threshold: float = 12.0,
    page: int = 1,
):
    """
    Group OCR results into lines, splitting words from punctuation.
    """
    if not ocr_results:
        return {"page": page, "results": []}, {"page": page, "results": {}}

    lines = list()
    current_line = list()

    for result in sorted(ocr_results, key=lambda x: (x.top, x.left)):
        if not current_line or abs(result.top - current_line[0].top) <= y_threshold:
            current_line.append(result)
        else:
            lines.append(sorted(current_line, key=lambda x: x.left))
            current_line = [result]
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x.left))

    page_line_level_ocr_results = list()
    page_line_level_ocr_results_with_words = {}
    line_counter = 1

    for line in lines:
        if not line:
            continue

        # Process the line to split punctuation from words
        processed_line = split_words_and_punctuation_from_line(line)

        # Re-calculate the line-level text and bounding box from the ORIGINAL words
        line_text = " ".join([word.text for word in line])
        line_left = line[0].left
        line_top = min(word.top for word in line)
        line_right = max(word.left + word.width for word in line)
        line_bottom = max(word.top + word.height for word in line)
        line_conf = round(
            sum(word.conf for word in line) / len(line), 0
        )  # This is mean confidence for the line

        final_line_bbox = OCRResult(
            text=line_text,
            left=line_left,
            top=line_top,
            width=line_right - line_left,
            height=line_bottom - line_top,
            line=line_counter,
            conf=line_conf,
        )

        page_line_level_ocr_results.append(final_line_bbox)

        # Use the PROCESSED line to create the children. Creates a result within page_line_level_ocr_results_with_words
        page_line_level_ocr_results_with_words["text_line_" + str(line_counter)] = (
            create_ocr_result_with_children(
                page_line_level_ocr_results_with_words,
                line_counter,
                final_line_bbox,
                processed_line,
            )
        )
        line_counter += 1

    page_level_results_with_page = {
        "page": page,
        "results": page_line_level_ocr_results,
    }
    page_level_results_with_words = {
        "page": page,
        "results": page_line_level_ocr_results_with_words,
    }

    return page_level_results_with_page, page_level_results_with_words
