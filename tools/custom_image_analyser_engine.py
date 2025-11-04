import copy
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
import pytesseract
from pdfminer.layout import LTChar
from PIL import Image
from presidio_analyzer import AnalyzerEngine, RecognizerResult

from tools.config import (
    AWS_PII_OPTION,
    CONVERT_LINE_TO_WORD_LEVEL,
    DEFAULT_LANGUAGE,
    HYBRID_OCR_CONFIDENCE_THRESHOLD,
    HYBRID_OCR_PADDING,
    LOCAL_OCR_MODEL_OPTIONS,
    LOCAL_PII_OPTION,
    OUTPUT_FOLDER,
    PADDLE_DET_DB_UNCLIP_RATIO,
    PADDLE_MODEL_PATH,
    PADDLE_USE_TEXTLINE_ORIENTATION,
    PREPROCESS_LOCAL_OCR_IMAGES,
    SAVE_EXAMPLE_HYBRID_IMAGES,
    SAVE_PADDLE_VISUALISATIONS,
    SAVE_PREPROCESS_IMAGES,
    SELECTED_MODEL,
    TESSERACT_SEGMENTATION_LEVEL,
)
from tools.helper_functions import clean_unicode_text
from tools.load_spacy_model_custom_recognisers import custom_entities
from tools.presidio_analyzer_custom import recognizer_result_from_dict
from tools.run_vlm import generate_image as vlm_generate_image
from tools.secure_path_utils import validate_folder_containment
from tools.secure_regex_utils import safe_sanitize_text
from tools.word_segmenter import AdaptiveSegmenter

if PREPROCESS_LOCAL_OCR_IMAGES == "True":
    PREPROCESS_LOCAL_OCR_IMAGES = True
else:
    PREPROCESS_LOCAL_OCR_IMAGES = False

try:
    from paddleocr import PaddleOCR
except ImportError:
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


def _vlm_ocr_predict(
    image: Image.Image,
    prompt: str = "Extract the text content from this image.",
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
        # Use the VLM to extract text
        # Pass None for parameters to prioritize model-specific defaults from run_vlm.py
        # If model defaults are not available, general defaults will be used (matching current values)
        extracted_text = vlm_generate_image(
            text=prompt,
            image=image,
            max_new_tokens=None,  # Use model default if available, otherwise MAX_NEW_TOKENS from config
            temperature=None,  # Use model default if available, otherwise 0.7
            top_p=None,  # Use model default if available, otherwise 0.9
            top_k=None,  # Use model default if available, otherwise 50
            repetition_penalty=None,  # Use model default if available, otherwise 1.3
        )

        if extracted_text and extracted_text.strip():
            # Clean the text
            cleaned_text = extracted_text.strip()

            # Split into words for compatibility with PaddleOCR format
            words = cleaned_text.split()

            # If text has more than 5 words, assume something went wrong and skip it
            if len(words) > 5:
                return {"rec_texts": [], "rec_scores": []}

            # Create PaddleOCR-compatible result
            result = {
                "rec_texts": words,
                "rec_scores": [0.95] * len(words),  # High confidence for VLM results
            }

            return result
        else:
            return {"rec_texts": [], "rec_scores": []}

    except Exception as e:
        print(f"VLM OCR error: {e}")
        return {"rec_texts": [], "rec_scores": []}


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

        :param ocr_engine: The OCR engine to use ("tesseract", "hybrid-paddle", "hybrid-vlm", "hybrid-paddle-vlm", or "paddle").
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
        ):
            if PaddleOCR is None:
                raise ImportError(
                    "paddleocr is not installed. Please run 'pip install paddleocr paddlepaddle' in your python environment and retry."
                )

            # Set PaddleOCR model directory environment variable (only if specified).
            if PADDLE_MODEL_PATH and PADDLE_MODEL_PATH.strip():
                os.environ["PADDLEOCR_MODEL_DIR"] = PADDLE_MODEL_PATH
                print(f"Setting PaddleOCR model path to: {PADDLE_MODEL_PATH}")
            else:
                print("Using default PaddleOCR model storage location")

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
            self.paddle_ocr = PaddleOCR(**paddle_kwargs)

        elif self.ocr_engine == "hybrid-vlm":
            # VLM-based hybrid OCR - no additional initialization needed
            # The VLM model is loaded when run_vlm.py is imported
            print(f"Initializing hybrid VLM OCR with model: {SELECTED_MODEL}")
            self.paddle_ocr = None  # Not using PaddleOCR

        if self.ocr_engine == "hybrid-paddle-vlm":
            # Hybrid PaddleOCR + VLM - requires both PaddleOCR and VLM
            # The VLM model is loaded when run_vlm.py is imported
            print(
                f"Initializing hybrid PaddleOCR + VLM OCR with model: {SELECTED_MODEL}"
            )

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
    ) -> Dict[str, List]:
        """Converts PaddleOCR result format to Tesseract's dictionary format using relative coordinates.

        This function uses a safer approach: converts PaddleOCR coordinates to relative (0-1) coordinates
        based on whatever coordinate space PaddleOCR uses, then scales them to the input image dimensions.
        This avoids issues with PaddleOCR's internal image resizing.

        Args:
            paddle_results: List of PaddleOCR result dictionaries
            input_image_width: Width of the input image passed to PaddleOCR (target dimensions for scaling)
            input_image_height: Height of the input image passed to PaddleOCR (target dimensions for scaling)
        """

        output = {
            "text": list(),
            "left": list(),
            "top": list(),
            "width": list(),
            "height": list(),
            "conf": list(),
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
            use_relative_coords = False

        for page_result in paddle_results:
            # Extract text recognition results from the new format
            rec_texts = page_result.get("rec_texts", list())
            rec_scores = page_result.get("rec_scores", list())
            rec_polys = page_result.get("rec_polys", list())

            # PaddleOCR may return image dimensions in the result - check for them
            # Some versions of PaddleOCR include this information
            result_image_width = page_result.get("image_width")
            result_image_height = page_result.get("image_height")

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
            # Priority: result metadata > detected from coordinates > input dimensions
            paddle_coord_width = (
                result_image_width
                if result_image_width is not None
                else max_x_coord if max_x_coord > 0 else input_image_width
            )
            paddle_coord_height = (
                result_image_height
                if result_image_height is not None
                else max_y_coord if max_y_coord > 0 else input_image_height
            )

            # If we couldn't determine PaddleOCR's coordinate space, fall back to input dimensions
            if paddle_coord_width is None or paddle_coord_height is None:
                paddle_coord_width = input_image_width
                paddle_coord_height = input_image_height
                use_relative_coords = False

            if paddle_coord_width <= 0 or paddle_coord_height <= 0:
                print(
                    f"Warning: Invalid PaddleOCR coordinate space dimensions ({paddle_coord_width}x{paddle_coord_height}). Using input dimensions."
                )
                paddle_coord_width = input_image_width or 1
                paddle_coord_height = input_image_height or 1
                use_relative_coords = False

            # Second pass: convert coordinates using relative coordinate approach
            for line_text, line_confidence, bounding_box in zip(
                rec_texts, rec_scores, rec_polys
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
            print(f"Using actual dimensions: {actual_width}x{actual_height}")
            # Update to use actual dimensions
            image_width = actual_width
            image_height = actual_height

        segmenter = AdaptiveSegmenter(output_folder=self.output_folder)

        # Process each line
        for i in range(len(line_data["text"])):
            line_text = line_data["text"][i]
            line_conf = line_data["conf"][i]

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

    def _perform_hybrid_ocr(
        self,
        image: Image.Image,
        confidence_threshold: int = HYBRID_OCR_CONFIDENCE_THRESHOLD,
        padding: int = HYBRID_OCR_PADDING,
        ocr: Optional[Any] = None,
        image_name: str = "unknown_image_name",
    ) -> Dict[str, list]:
        """
        Performs OCR using Tesseract for bounding boxes and PaddleOCR/VLM for low-confidence text.
        Returns data in the same dictionary format as pytesseract.image_to_data.
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

        print("Starting hybrid OCR process...")

        # 1. Get initial word-level results from Tesseract
        tesseract_data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config=self.tesseract_config,
            lang=self.tesseract_lang,
        )

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
            if conf < confidence_threshold:
                img_width, img_height = image.size
                crop_left = max(0, left - padding - 15)
                crop_top = max(0, top - padding)
                crop_right = min(img_width, left + width + padding + 15)
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
                    if new_conf > conf:
                        ocr_type = "VLM" if use_vlm else "Paddle"
                        print(
                            f"  Re-OCR'd word: '{text}' (conf: {conf}) -> '{new_text}' (conf: {new_conf:.0f}) [{ocr_type}]"
                        )

                        # For exporting example image comparisons, not used here
                        safe_filename = self._create_safe_filename_with_confidence(
                            text, new_text, conf, new_conf, ocr_type
                        )

                        if SAVE_EXAMPLE_HYBRID_IMAGES is True:
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
        confidence_threshold: int = HYBRID_OCR_CONFIDENCE_THRESHOLD,
        padding: int = HYBRID_OCR_PADDING,
        image_name: str = "unknown_image_name",
        input_image_width: int = None,
        input_image_height: int = None,
    ) -> Dict[str, list]:
        """
        Performs OCR using PaddleOCR at line level, then VLM for low-confidence lines.
        Returns data in the same dictionary format as pytesseract.image_to_data.

        Args:
            image: PIL Image to process
            ocr: PaddleOCR instance (optional, uses self.paddle_ocr if not provided)
            confidence_threshold: Confidence threshold below which VLM is used
            padding: Padding to add around line crops
            image_name: Name of the image for logging/debugging
            input_image_width: Original image width (before preprocessing)
            input_image_height: Original image height (before preprocessing)

        Returns:
            Dictionary with OCR results in Tesseract format
        """
        if ocr is None:
            if hasattr(self, "paddle_ocr") and self.paddle_ocr is not None:
                ocr = self.paddle_ocr
            else:
                raise ValueError(
                    "No OCR object provided and 'paddle_ocr' is not initialized."
                )

        print("Starting hybrid PaddleOCR + VLM OCR process...")

        # Get image dimensions
        img_width, img_height = image.size

        # Use original dimensions if provided, otherwise use current image dimensions
        if input_image_width is None:
            input_image_width = img_width
        if input_image_height is None:
            input_image_height = img_height

        # 1. Get initial line-level results from PaddleOCR
        image_np = np.array(image)
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)

        paddle_results = ocr.predict(image_np)

        # Convert PaddleOCR results to line-level format
        paddle_line_data = self._convert_paddle_to_tesseract_format(
            paddle_results,
            input_image_width=input_image_width,
            input_image_height=input_image_height,
        )

        # Prepare final output structure
        final_data = {
            "text": list(),
            "left": list(),
            "top": list(),
            "width": list(),
            "height": list(),
            "conf": list(),
            "model": list(),  # Track which model was used for each line
        }

        num_lines = len(paddle_line_data["text"])

        # Process each line
        for i in range(num_lines):
            line_text = paddle_line_data["text"][i]
            line_conf = int(paddle_line_data["conf"][i])
            line_left = float(paddle_line_data["left"][i])
            line_top = float(paddle_line_data["top"][i])
            line_width = float(paddle_line_data["width"][i])
            line_height = float(paddle_line_data["height"][i])

            # Skip empty lines
            if not line_text.strip():
                continue

            # Initialize model as PaddleOCR (default)
            model_used = "Paddle"

            # Count words in PaddleOCR output
            paddle_words = line_text.split()
            paddle_word_count = len(paddle_words)

            # If confidence is low, use VLM for a second opinion
            if line_conf < confidence_threshold:
                # Calculate crop coordinates with padding
                crop_left = max(0, int(line_left - padding))
                crop_top = max(0, int(line_top - padding))
                crop_right = min(img_width, int(line_left + line_width + padding))
                crop_bottom = min(img_height, int(line_top + line_height + padding))

                # Ensure crop dimensions are valid
                if crop_right <= crop_left or crop_bottom <= crop_top:
                    # Invalid crop, keep original PaddleOCR result
                    final_data["text"].append(clean_unicode_text(line_text))
                    final_data["left"].append(line_left)
                    final_data["top"].append(line_top)
                    final_data["width"].append(line_width)
                    final_data["height"].append(line_height)
                    final_data["conf"].append(line_conf)
                    final_data["model"].append(model_used)
                    continue

                # Crop the line image
                cropped_image = image.crop(
                    (crop_left, crop_top, crop_right, crop_bottom)
                )

                # Use VLM for OCR on this line
                vlm_result = _vlm_ocr_predict(cropped_image)
                vlm_rec_texts = vlm_result.get("rec_texts", [])
                vlm_rec_scores = vlm_result.get("rec_scores", [])

                if vlm_rec_texts and vlm_rec_scores:
                    # Combine VLM words into a single text string
                    vlm_text = " ".join(vlm_rec_texts)
                    vlm_word_count = len(vlm_rec_texts)
                    vlm_conf = int(round(np.median(vlm_rec_scores) * 100, 0))

                    # Only replace if word counts match
                    if vlm_word_count == paddle_word_count:
                        print(
                            f"  Re-OCR'd line: '{line_text}' (conf: {line_conf}, words: {paddle_word_count}) "
                            f"-> '{vlm_text}' (conf: {vlm_conf:.0f}, words: {vlm_word_count}) [VLM]"
                        )

                        # For exporting example image comparisons
                        safe_filename = self._create_safe_filename_with_confidence(
                            line_text, vlm_text, line_conf, vlm_conf, "VLM"
                        )

                        if SAVE_EXAMPLE_HYBRID_IMAGES is True:
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
                                self.output_folder
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
                                hybrid_ocr_examples_folder + f"/{safe_filename}.png"
                            )
                            print(f"Saving example image to {output_image_path}")
                            cropped_image.save(output_image_path)

                        # Replace with VLM result
                        line_text = vlm_text
                        line_conf = vlm_conf
                        model_used = "VLM"
                    else:
                        print(
                            f"  Line: '{line_text}' (conf: {line_conf}, words: {paddle_word_count}) -> "
                            f"VLM result '{vlm_text}' (conf: {vlm_conf:.0f}, words: {vlm_word_count}) "
                            f"word count mismatch. Keeping PaddleOCR result."
                        )

            # Append the final result (either original PaddleOCR or replaced VLM)
            final_data["text"].append(clean_unicode_text(line_text))
            final_data["left"].append(line_left)
            final_data["top"].append(line_top)
            final_data["width"].append(line_width)
            final_data["height"].append(line_height)
            final_data["conf"].append(int(line_conf))
            final_data["model"].append(model_used)

        return final_data

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

        if PREPROCESS_LOCAL_OCR_IMAGES:
            print("Pre-processing image...")
            # Get original dimensions before preprocessing
            original_image_width, original_image_height = image.size
            image, preprocessing_metadata = self.image_preprocessor.preprocess_image(
                image
            )
            if SAVE_PREPROCESS_IMAGES:
                print("Saving pre-processed image...")
                image_basename = os.path.basename(image_name)
                output_path = os.path.join(
                    self.output_folder,
                    "preprocessed_images",
                    image_basename + "_preprocessed_image.png",
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path)
                print(f"Pre-processed image saved to {output_path}")
        else:
            preprocessing_metadata = dict()
            original_image_width, original_image_height = image.size

        image_width, image_height = image.size

        # Note: In testing I haven't seen that this necessarily improves results
        if self.ocr_engine == "hybrid-paddle":
            # Try hybrid with original image for cropping:
            ocr_data = self._perform_hybrid_ocr(image, image_name=image_name)

        elif self.ocr_engine == "hybrid-vlm":
            # Try hybrid VLM with original image for cropping:
            ocr_data = self._perform_hybrid_ocr(image, image_name=image_name)

        elif self.ocr_engine == "hybrid-paddle-vlm":
            # Hybrid PaddleOCR + VLM: use PaddleOCR at line level, then VLM for low-confidence lines
            if ocr is None:
                if hasattr(self, "paddle_ocr") and self.paddle_ocr is not None:
                    ocr = self.paddle_ocr
                else:
                    raise ValueError(
                        "No OCR object provided and 'paddle_ocr' is not initialized."
                    )
            ocr_data = self._perform_hybrid_paddle_vlm_ocr(
                image,
                ocr=ocr,
                image_name=image_name,
                input_image_width=original_image_width,
                input_image_height=original_image_height,
            )

        elif self.ocr_engine == "tesseract":

            ocr_data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config=self.tesseract_config,
                lang=self.tesseract_lang,  # Ensure the Tesseract language data (e.g., fra.traineddata) is installed on your system.
            )

        elif self.ocr_engine == "paddle":

            if ocr is None:
                if hasattr(self, "paddle_ocr") and self.paddle_ocr is not None:
                    ocr = self.paddle_ocr
                else:
                    raise ValueError(
                        "No OCR object provided and 'paddle_ocr' is not initialised."
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

                # Store dimensions of the image we're passing to PaddleOCR (preprocessed dimensions)
                paddle_input_width = image_np.shape[1]
                paddle_input_height = image_np.shape[0]

                paddle_results = ocr.predict(image_np)
            else:
                # When using image path, load image to get dimensions
                temp_image = Image.open(image_path)
                paddle_input_width, paddle_input_height = temp_image.size
                # For file path, use the original dimensions (before preprocessing)
                # original_image_width and original_image_height are already set above
                paddle_results = ocr.predict(image_path)

            # Save PaddleOCR visualization with bounding boxes
            if paddle_results and SAVE_PADDLE_VISUALISATIONS is True:

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

            ocr_data = self._convert_paddle_to_tesseract_format(
                paddle_results,
                input_image_width=original_image_width,
                input_image_height=original_image_height,
            )

        else:
            raise RuntimeError(f"Unsupported OCR engine: {self.ocr_engine}")

        # Convert line-level results to word-level if configured and needed
        if CONVERT_LINE_TO_WORD_LEVEL and self._is_line_level_data(ocr_data):
            print("Converting line-level OCR results to word-level...")
            # Check if coordinates need to be scaled to match the preprocessed image
            # For PaddleOCR: _convert_paddle_to_tesseract_format converts coordinates to original image space,
            #   but we need to crop from the preprocessed image, so we need to scale coordinates up
            # For Tesseract: OCR runs on preprocessed image, so coordinates are already in preprocessed space,
            #   matching the preprocessed image we're cropping from - no scaling needed
            needs_scaling = False
            if (
                PREPROCESS_LOCAL_OCR_IMAGES
                and original_image_width
                and original_image_height
            ):
                if (
                    self.ocr_engine == "paddle"
                    or self.ocr_engine == "hybrid-paddle-vlm"
                ):
                    # PaddleOCR coordinates are converted to original space by _convert_paddle_to_tesseract_format
                    # hybrid-paddle-vlm also uses PaddleOCR and converts to original space
                    needs_scaling = True

            if needs_scaling:
                # Calculate scale factors from original to preprocessed
                scale_x = image_width / original_image_width
                scale_y = image_height / original_image_height
                print(
                    f"Scaling coordinates from original ({original_image_width}x{original_image_height}) to preprocessed ({image_width}x{image_height})"
                )
                print(f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
                # Scale coordinates to preprocessed image space for cropping
                scaled_ocr_data = {
                    "text": ocr_data["text"],
                    "left": [x * scale_x for x in ocr_data["left"]],
                    "top": [y * scale_y for y in ocr_data["top"]],
                    "width": [w * scale_x for w in ocr_data["width"]],
                    "height": [h * scale_y for h in ocr_data["height"]],
                    "conf": ocr_data["conf"],
                }
                ocr_data = self._convert_line_to_word_level(
                    scaled_ocr_data,
                    image_width,
                    image_height,
                    image,
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
                ocr_data = self._convert_line_to_word_level(
                    ocr_data, image_width, image_height, image, image_name=image_name
                )

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
            if self.ocr_engine == "paddle" or self.ocr_engine == "hybrid-paddle-vlm":
                pass
                # print(f"Skipping rescale_ocr_data for PaddleOCR (already scaled to original dimensions)")
            else:
                ocr_data = rescale_ocr_data(ocr_data, scale_factor)

        # The rest of your processing pipeline now works for both engines
        ocr_result = ocr_data

        # Filter out empty strings and low confidence results
        valid_indices = [
            i
            for i, text in enumerate(ocr_result["text"])
            if text.strip() and int(ocr_result["conf"][i]) > 0
        ]

        # Determine default model based on OCR engine if model field is not present
        if "model" in ocr_result and len(ocr_result["model"]) == len(
            ocr_result["text"]
        ):
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
                        "hybrid-paddle"
                        if self.ocr_engine == "hybrid-paddle"
                        else (
                            "VLM"
                            if self.ocr_engine == "hybrid-vlm"
                            else (
                                "hybrid-paddle-vlm"
                                if self.ocr_engine == "hybrid-paddle-vlm"
                                else None
                            )
                        )
                    )
                )
            )

            def get_model(idx):
                return default_model

        return [
            OCRResult(
                text=clean_unicode_text(ocr_result["text"][i]),
                left=ocr_result["left"][i],
                top=ocr_result["top"][i],
                width=ocr_result["width"][i],
                height=ocr_result["height"][i],
                conf=round(float(ocr_result["conf"][i]), 0),
                model=get_model(i),
                # line_number=ocr_result['abs_line_id'][i]
            )
            for i in valid_indices
        ]

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
                print(f"No relevant entities supported for language: {language}")
                raise Warning(
                    f"No relevant entities supported for language: {language}"
                )

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

            for i, text_line in enumerate(line_level_ocr_results):
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
            # print("ocr_results_with_words_child_info:", ocr_results_with_words_child_info)

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
                            # print(f"Matched word: {word_info['text']}")

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
            # print("Result:", result)
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
