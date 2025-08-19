import pytesseract
import numpy as np
import pandas as pd
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
import time
import cv2
import re
import copy
import botocore
from copy import deepcopy
from pdfminer.layout import LTChar
from PIL import Image
from typing import Optional, Tuple, Union
from tools.helper_functions import clean_unicode_text
from tools.presidio_analyzer_custom import recognizer_result_from_dict
from tools.load_spacy_model_custom_recognisers import custom_entities
from tools.config import PREPROCESS_LOCAL_OCR_IMAGES

if PREPROCESS_LOCAL_OCR_IMAGES == "True": PREPROCESS_LOCAL_OCR_IMAGES = True 
else: PREPROCESS_LOCAL_OCR_IMAGES = False

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

@dataclass
class OCRResult:
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float = None
    line_number: int = None

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
    def _get_bg_color(image: np.ndarray, is_greyscale: bool, invert: bool = False) -> Union[int, Tuple[int, int, int]]:
        # Note: Modified to expect numpy array for bincount
        if invert:
             image = 255 - image # Simple inversion for greyscale numpy array
        
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
    def __init__(self, diameter: int = 9, sigma_color: int = 75, sigma_space: int = 75) -> None:
        super().__init__(use_greyscale=True)
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        # Modified to accept and return numpy array for consistency in the pipeline
        filtered_image = cv2.bilateralFilter(image, self.diameter, self.sigma_color, self.sigma_space)
        metadata = {"diameter": self.diameter, "sigma_color": self.sigma_color, "sigma_space": self.sigma_space}
        return filtered_image, metadata
    
class SegmentedAdaptiveThreshold(ImagePreprocessor):
    """Applies adaptive thresholding."""
    def __init__(self, block_size: int = 21, contrast_threshold: int = 40, c_low_contrast: int = 5,
                 c_high_contrast: int = 10, bg_threshold: int = 127) -> None:
        super().__init__(use_greyscale=True)
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1 # Ensure odd
        self.c_low_contrast = c_low_contrast
        self.c_high_contrast = c_high_contrast
        self.bg_threshold = bg_threshold
        self.contrast_threshold = contrast_threshold

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        # Modified to accept and return numpy array
        background_color = self._get_bg_color(image, True)
        contrast, _ = self._get_image_contrast(image)
        c = self.c_low_contrast if contrast <= self.contrast_threshold else self.c_high_contrast

        if background_color < self.bg_threshold: # Dark background, light text
            adaptive_threshold_image = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self.block_size, -c
            )
        else: # Light background, dark text
            adaptive_threshold_image = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.block_size, c
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

    def preprocess_image(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """
        A corrected, logical pipeline for OCR preprocessing.
        Order: Greyscale -> Rescale -> Denoise -> Enhance Contrast -> Binarize
        """
        # 1. Convert to greyscale NumPy array
        image_np = self.convert_image_to_array(image)

        # 2. Rescale image to optimal DPI (while still greyscale)
        rescaled_image_np, scale_metadata = self.image_rescaling.preprocess_image(image_np)

        # 3. Apply bilateral filtering for noise reduction
        filtered_image_np, _ = self.bilateral_filter.preprocess_image(rescaled_image_np)

        # 4. Improve contrast
        adjusted_image_np, _, _ = self._improve_contrast(filtered_image_np)

        # 5. Adaptive Thresholding (Binarization) - This is the final step
        final_image_np, threshold_metadata = self.adaptive_threshold.preprocess_image(
            adjusted_image_np
        )
        
        # Combine metadata
        final_metadata = {**scale_metadata, **threshold_metadata}
        
        # Convert final numpy array back to PIL Image for return
        return Image.fromarray(final_image_np), final_metadata

def rescale_ocr_data(ocr_data, scale_factor:float):
    
    # We loop from 0 to the number of detected words.
    num_boxes = len(ocr_data['text'])
    for i in range(num_boxes):
        # We only want to process actual words, not empty boxes Tesseract might find
        if int(ocr_data['conf'][i]) > -1: # -1 confidence is for structural elements
            # Get coordinates from the processed image using the index 'i'
            x_proc = ocr_data['left'][i]
            y_proc = ocr_data['top'][i]
            w_proc = ocr_data['width'][i]
            h_proc = ocr_data['height'][i]

            # Apply the inverse transformation (division)
            x_orig = int(x_proc / scale_factor)
            y_orig = int(y_proc / scale_factor)
            w_orig = int(w_proc / scale_factor)
            h_orig = int(h_proc / scale_factor)

            # --- THE MAPPING STEP ---
            # Update the dictionary values in-place using the same index 'i'
            ocr_data['left'][i] = x_orig
            ocr_data['top'][i] = y_orig
            ocr_data['width'][i] = w_orig
            ocr_data['height'][i] = h_orig
    
    return ocr_data
class CustomImageAnalyzerEngine:
    def __init__(
        self,
        analyzer_engine: Optional[AnalyzerEngine] = None,
        ocr_engine: str = "tesseract",        
        tesseract_config: Optional[str] = None,
        paddle_kwargs: Optional[Dict[str, Any]] = None,
        image_preprocessor: Optional[ImagePreprocessor] = None
    ):
        """
        Initializes the CustomImageAnalyzerEngine.

        :param ocr_engine: The OCR engine to use ("tesseract" or "paddle").
        :param analyzer_engine: The Presidio AnalyzerEngine instance.
        :param tesseract_config: Configuration string for Tesseract.
        :param paddle_kwargs: Dictionary of keyword arguments for PaddleOCR constructor.
        :param image_preprocessor: Optional image preprocessor.
        """
        if ocr_engine not in ["tesseract", "paddle", "hybrid"]:
            raise ValueError("ocr_engine must be either 'tesseract', 'hybrid', or 'paddle'")

        self.ocr_engine = ocr_engine
        
        if self.ocr_engine == "paddle" or self.ocr_engine == "hybrid":
            if PaddleOCR is None:
                raise ImportError("paddleocr is not installed. Please run 'pip install paddleocr paddlepaddle'")
            # Default paddle configuration if none provided
            if paddle_kwargs is None:
                paddle_kwargs = {'use_textline_orientation': True, 'lang': 'en'}
            self.paddle_ocr = PaddleOCR(**paddle_kwargs)

        if not analyzer_engine:
            analyzer_engine = AnalyzerEngine()
        self.analyzer_engine = analyzer_engine

        self.tesseract_config = tesseract_config or '--oem 3 --psm 11'

        if not image_preprocessor:
            image_preprocessor = ContrastSegmentedImageEnhancer()
        self.image_preprocessor = image_preprocessor

    def _sanitize_filename(self, text: str, max_length: int = 20) -> str:
        """
        Sanitizes text for use in filenames by removing invalid characters and limiting length.
        
        :param text: The text to sanitize
        :param max_length: Maximum length of the sanitized text
        :return: Sanitized text safe for filenames
        """
       
        # Remove or replace invalid filename characters
        # Windows: < > : " | ? * \ /
        # Unix: / (forward slash)
        # Also remove control characters and other problematic chars
        invalid_chars = r'[<>:"|?*\\/\x00-\x1f\x7f-\x9f]'
        sanitized = re.sub(invalid_chars, '_', text)
        
        # Replace multiple consecutive underscores with a single one
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores and spaces
        sanitized = sanitized.strip('_ ')
        
        # If empty after sanitization, use a default value
        if not sanitized:
            sanitized = 'text'
        
        # Limit to max_length characters
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            # Ensure we don't end with an underscore if we cut in the middle
            sanitized = sanitized.rstrip('_')
        
        return sanitized

    def _convert_paddle_to_tesseract_format(self, paddle_results: List[Any]) -> Dict[str, List]:
        """Converts PaddleOCR result format to Tesseract's dictionary format. NOTE: This attempts to create word-level bounding boxes by estimating the distance between characters in sentence-level text output. This is currently quite inaccurate, and word-level bounding boxes should not be relied upon."""

        output = {'text': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': []}

        # paddle_results is now a list of dictionaries with detailed information
        if not paddle_results:
            return output
            
        for page_result in paddle_results:
            # Extract text recognition results from the new format
            rec_texts = page_result.get('rec_texts', [])
            rec_scores = page_result.get('rec_scores', [])
            rec_polys = page_result.get('rec_polys', [])
            
            for line_text, line_confidence, bounding_box in zip(rec_texts, rec_scores, rec_polys):
                # bounding_box is now a numpy array with shape (4, 2)
                # Convert to list of coordinates if it's a numpy array
                if hasattr(bounding_box, 'tolist'):
                    box = bounding_box.tolist()
                else:
                    box = bounding_box
                
                # box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                
                line_left = int(min(x_coords))
                line_top = int(min(y_coords))
                line_width = int(max(x_coords) - line_left)
                line_height = int(max(y_coords) - line_top)

                 # 2. Split the line into words
                words = line_text.split()
                if not words:
                    continue

                # 3. Estimate bounding box for each word
                total_chars = len(line_text)
                # Avoid division by zero for empty lines
                avg_char_width = line_width / total_chars if total_chars > 0 else 0

                current_char_offset = 0

                for word in words:
                    word_width = int(len(word) * avg_char_width)
                    word_left = line_left + int(current_char_offset * avg_char_width)
                    
                    output['text'].append(word)
                    output['left'].append(word_left)
                    output['top'].append(line_top)
                    output['width'].append(word_width)
                    output['height'].append(line_height)
                    # Use the line's confidence for each word derived from it
                    output['conf'].append(int(line_confidence * 100))

                    # Update offset for the next word (add word length + 1 for the space)
                    current_char_offset += len(word) + 1
            
        return output
    
    # def _perform_hybrid_ocr(
    #     self, 
    #     image: Image.Image, 
    #     confidence_threshold: int = 65, 
    #     padding: int = 5,
    #     ocr: Optional[Any] = None
    # ) -> List[OCRResult]:
    #     """
    #     Performs OCR using Tesseract for bounding boxes and PaddleOCR for low-confidence text.
    #     """
    #     if ocr is None:
    #         if hasattr(self, 'paddle_ocr') and self.paddle_ocr is not None:
    #             ocr = self.paddle_ocr
    #         else:
    #             raise ValueError("No OCR object provided and 'paddle_ocr' is not initialized.")
    #     """
    #     Performs OCR using Tesseract for bounding boxes and PaddleOCR for low-confidence text.
    #     """
    #     print("Starting hybrid OCR process...")
        
    #     # 1. Get initial word-level results from Tesseract
    #     tesseract_data = pytesseract.image_to_data(
    #         image,
    #         output_type=pytesseract.Output.DICT,
    #         config=self.tesseract_config
    #     )
        
    #     final_results = []
    #     num_words = len(tesseract_data['text'])
        
    #     for i in range(num_words):
    #         text = tesseract_data['text'][i]
    #         conf = int(tesseract_data['conf'][i])
            
    #         # Skip empty text boxes or non-word elements
    #         if not text.strip() or conf == -1:
    #             continue

    #         left = tesseract_data['left'][i]
    #         top = tesseract_data['top'][i]
    #         width = tesseract_data['width'][i]
    #         height = tesseract_data['height'][i]
            
    #         # 2. If confidence is low, use PaddleOCR for a second opinion
    #         if conf < confidence_threshold:         

    #             # 3. Crop the sub-image with padding
    #             img_width, img_height = image.size
                
    #             # Add padding but ensure it doesn't go out of bounds
    #             crop_left = max(0, left - padding - 15)
    #             crop_top = max(0, top - padding)
    #             crop_right = min(img_width, left + width + padding + 15)
    #             crop_bottom = min(img_height, top + height + padding)
                
    #             cropped_image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    #             cropped_image_np = np.array(cropped_image)                

    #             # PaddleOCR may need an RGB image. Ensure it has 3 channels.
    #             if len(cropped_image_np.shape) == 2:
    #                 cropped_image_np = np.stack([cropped_image_np] * 3, axis=-1)

    #             # 4. Run PaddleOCR on the small crop
                
    #             paddle_results = ocr.predict(cropped_image_np)
                
    #             if paddle_results[0]:                    
    #                 # Extract text recognition results from the new format
    #                 rec_texts = paddle_results[0].get('rec_texts', [])
    #                 rec_scores = paddle_results[0].get('rec_scores', [])
    #                 rec_polys = paddle_results[0].get('rec_polys', [])
                    
    #                 new_text = " ".join([line_text for line_text in rec_texts])                    
                
    #                 # 5. Process and replace the text                
    #                 # Concatenate results if Paddle splits the word into multiple parts
    #                 #new_text = " ".join([line[1][0] for line in paddle_result[0]])
    #                 new_conf = pd.Series(rec_scores).median() * 100

    #                 if new_conf > confidence_threshold:
                        
    #                     print(f"  Re-OCR'd word: '{text}' (conf: {conf}) -> '{new_text}' (conf: {new_conf})")

    #                     # For exporting example image comparisons, not used here
    #                     # safe_text = self._sanitize_filename(text, max_length=20)
    #                     # new_safe_text = self._sanitize_filename(new_text, max_length=20)
    #                     # output_image_path = f"examples/tess_vs_paddle_examples/{conf}_conf_{safe_text}_to_{new_safe_text}.png"
    #                     # cropped_image.save(output_image_path)

    #                     text = new_text
    #                     conf = new_conf

    #                 elif new_text:
    #                     text = new_text
    #                     print(f"  '{text}' (conf: {conf}) -> '{new_text}' (conf: {new_conf}) had too low confidence, keeping original")
    #                 else:
    #                     print(f"  '{text}' (conf: {conf}) -> No text found by Paddle, returning nothing.")

    #                     # For exporting example image comparisons, not used here
    #                     # safe_text = self._sanitize_filename(text, max_length=20)
    #                     # output_image_path = f"examples/tess_vs_paddle_examples/{conf}_conf_{safe_text}_to_blank.png"
    #                     # cropped_image.save(output_image_path)

    #                     text = ''

    #             else:
    #                 print(f"  '{text}' (conf: {conf}) -> No text found by Paddle, keeping original.")
    #                 text = ''

    #         # 6. Append the final result (either original or replaced)
    #         if text:
    #             final_results.append(OCRResult(
    #                 text=clean_unicode_text(text),
    #                 left=left,
    #                 top=top,
    #                 width=width,
    #                 height=height
    #             ))
            
    #     return final_results
    
    def _perform_hybrid_ocr(
    self,
    image: Image.Image,
    confidence_threshold: int = 65,
    padding: int = 5,
    ocr: Optional[Any] = None
) -> Dict[str, list]:
        """
        Performs OCR using Tesseract for bounding boxes and PaddleOCR for low-confidence text.
        Returns data in the same dictionary format as pytesseract.image_to_data.
        """
        if ocr is None:
            if hasattr(self, 'paddle_ocr') and self.paddle_ocr is not None:
                ocr = self.paddle_ocr
            else:
                raise ValueError("No OCR object provided and 'paddle_ocr' is not initialized.")
        
        print("Starting hybrid OCR process...")
        
        # 1. Get initial word-level results from Tesseract
        tesseract_data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config=self.tesseract_config
        )
        
        final_data = {'text': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': []}
        
        num_words = len(tesseract_data['text'])

        # This handles the "no text on page" case. If num_words is 0, the loop is skipped
        # and an empty dictionary with empty lists is returned, which is the correct behavior.
        for i in range(num_words):
            text = tesseract_data['text'][i]
            conf = int(tesseract_data['conf'][i])
            
            # Skip empty text boxes or non-word elements (like page/block markers)
            if not text.strip() or conf == -1:
                continue

            left = tesseract_data['left'][i]
            top = tesseract_data['top'][i]
            width = tesseract_data['width'][i]
            height = tesseract_data['height'][i]
            
            # If confidence is low, use PaddleOCR for a second opinion
            if conf < confidence_threshold:
                img_width, img_height = image.size
                crop_left = max(0, left - padding - 15)
                crop_top = max(0, top - padding)
                crop_right = min(img_width, left + width + padding + 15)
                crop_bottom = min(img_height, top + height + padding)
                
                # Ensure crop dimensions are valid
                if crop_right <= crop_left or crop_bottom <= crop_top:
                    continue # Skip invalid crops

                cropped_image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
                cropped_image_np = np.array(cropped_image)
                
                if len(cropped_image_np.shape) == 2:
                    cropped_image_np = np.stack([cropped_image_np] * 3, axis=-1)
                
                paddle_results = ocr.predict(cropped_image_np)
                
                if paddle_results and paddle_results[0]:
                    rec_texts = paddle_results[0].get('rec_texts', [])
                    rec_scores = paddle_results[0].get('rec_scores', [])
                    
                    if rec_texts and rec_scores:
                        new_text = " ".join(rec_texts)
                        new_conf = int(round(np.median(rec_scores) * 100,0))

                        # Only replace if Paddle's confidence is better
                        if new_conf > conf:
                            print(f"  Re-OCR'd word: '{text}' (conf: {conf}) -> '{new_text}' (conf: {new_conf:.0f})")

                            # For exporting example image comparisons, not used here
                            safe_text = self._sanitize_filename(text, max_length=20)
                            safe_new_text = self._sanitize_filename(new_text, max_length=20)
                            output_image_path = f"examples/tess_vs_paddle_examples/{conf}_conf_{safe_text}_to_{new_text}_{new_conf}.png"
                            cropped_image.save(output_image_path)

                            text = new_text
                            conf = new_conf
                            
                        else:
                            print(f"  '{text}' (conf: {conf}) -> Paddle result '{new_text}' (conf: {new_conf:.0f}) was not better. Keeping original.")
                    else:
                        # Paddle ran but found nothing, so discard the original low-confidence word
                        print(f"  '{text}' (conf: {conf}) -> No text found by Paddle. Discarding.")
                        text = ''
                else:
                    # Paddle found nothing, discard original word
                    print(f"  '{text}' (conf: {conf}) -> No text found by Paddle. Discarding.")
                    text = ''

            # Append the final result (either original, replaced, or skipped if empty)
            if text.strip():
                final_data['text'].append(clean_unicode_text(text))
                final_data['left'].append(left)
                final_data['top'].append(top)
                final_data['width'].append(width)
                final_data['height'].append(height)
                final_data['conf'].append(int(conf))
                
        return final_data
    
    def perform_ocr(self,        
        image: Union[str, Image.Image, np.ndarray],
        ocr: Optional[Any] = None) -> List[OCRResult]:
        """
        Performs OCR on the given image using the configured engine.
        """
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Pre-process image - currently seems to give worse results!
        if str(PREPROCESS_LOCAL_OCR_IMAGES).lower() == 'true':
            image, preprocessing_metadata = self.image_preprocessor.preprocess_image(image)
        else:
            preprocessing_metadata = {}

        # Note: In testing I haven't seen that this necessarily improves results
        if self.ocr_engine == "hybrid":
            # Try hybrid with original image for cropping:
            ocr_data = self._perform_hybrid_ocr(image)

        elif self.ocr_engine == "tesseract":

            ocr_data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config=self.tesseract_config
            )

        elif self.ocr_engine == "paddle":

            image_np = np.array(image) # image_processed
            
            # PaddleOCR may need an RGB image. Ensure it has 3 channels.
            if len(image_np.shape) == 2:
                image_np = np.stack([image_np] * 3, axis=-1)

            if ocr is None:
                if hasattr(self, 'paddle_ocr') and self.paddle_ocr is not None:
                    ocr = self.paddle_ocr
                else:
                    raise ValueError("No OCR object provided and 'paddle_ocr' is not initialised.")

            #ocr = PaddleOCR(use_textline_orientation=True, lang='en')
            paddle_results = ocr.predict(image_np)
            ocr_data = self._convert_paddle_to_tesseract_format(paddle_results)

        else:
            raise RuntimeError(f"Unsupported OCR engine: {self.ocr_engine}")
        
        if preprocessing_metadata:
            scale_factor = preprocessing_metadata.get('scale_factor', 1.0)
            ocr_data = rescale_ocr_data(ocr_data, scale_factor)

        # The rest of your processing pipeline now works for both engines
        ocr_result = ocr_data
        
        # Filter out empty strings and low confidence results
        valid_indices = [
            i for i, text in enumerate(ocr_result['text'])
            if text.strip() and int(ocr_result['conf'][i]) > 0
        ]
        
        return [
            OCRResult(
                text=clean_unicode_text(ocr_result['text'][i]),
                left=ocr_result['left'][i],
                top=ocr_result['top'][i],
                width=ocr_result['width'][i],
                height=ocr_result['height'][i]
            )
            for i in valid_indices
        ]


    def analyze_text(
        self, 
        line_level_ocr_results: List[OCRResult], 
        ocr_results_with_words: Dict[str, Dict],
        chosen_redact_comprehend_entities: List[str],
        pii_identification_method: str = "Local",
        comprehend_client = "",
        custom_entities:List[str]=custom_entities,   
        **text_analyzer_kwargs
    ) -> List[CustomImageRecognizerResult]:

        page_text = ""
        page_text_mapping = []
        all_text_line_results = []
        comprehend_query_number = 0

        # Collect all text and create mapping
        for i, line_level_ocr_result in enumerate(line_level_ocr_results):
            if page_text:
                page_text += " "
            start_pos = len(page_text)
            page_text += line_level_ocr_result.text
            # Note: We're not passing line_characters here since it's not needed for this use case
            page_text_mapping.append((start_pos, i, line_level_ocr_result, None))

        # Process using either Local or AWS Comprehend
        if pii_identification_method == "Local":
            analyzer_result = self.analyzer_engine.analyze(
                text=page_text,
                **text_analyzer_kwargs
            )
            all_text_line_results = map_back_entity_results(
                analyzer_result,
                page_text_mapping,
                all_text_line_results
            )

        elif pii_identification_method == "AWS Comprehend":
            # Handle custom entities first
            if custom_entities:
                custom_redact_entities = [
                    entity for entity in chosen_redact_comprehend_entities 
                    if entity in custom_entities
                ]
                if custom_redact_entities:
                    text_analyzer_kwargs["entities"] = custom_redact_entities
                    page_analyser_result = self.analyzer_engine.analyze(
                        text=page_text,
                        **text_analyzer_kwargs
                    )
                    all_text_line_results = map_back_entity_results(
                        page_analyser_result,
                        page_text_mapping,
                        all_text_line_results
                    )

            # Process text in batches for AWS Comprehend
            current_batch = ""
            current_batch_mapping = []
            batch_char_count = 0
            batch_word_count = 0

            for i, text_line in enumerate(line_level_ocr_results):
                words = text_line.text.split()
                word_start_positions = []
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
                            text_analyzer_kwargs["language"],
                            text_analyzer_kwargs.get('allow_list', []),
                            chosen_redact_comprehend_entities,
                            all_text_line_results
                        )
                        comprehend_query_number += 1
                        
                        # Reset batch
                        current_batch = word
                        batch_word_count = 1
                        batch_char_count = len(word)
                        current_batch_mapping = [(0, i, text_line, None, word_start_positions[word_idx])]
                    else:
                        if current_batch:
                            current_batch += " "
                            batch_char_count += 1
                        current_batch += word
                        batch_char_count += len(word)
                        batch_word_count += 1
                        
                        if not current_batch_mapping or current_batch_mapping[-1][1] != i:
                            current_batch_mapping.append((
                                batch_char_count - len(word),
                                i,
                                text_line,
                                None,
                                word_start_positions[word_idx]
                            ))

            # Process final batch if any
            if current_batch:
                all_text_line_results = do_aws_comprehend_call(
                    current_batch,
                    current_batch_mapping,
                    comprehend_client,
                    text_analyzer_kwargs["language"],
                    text_analyzer_kwargs.get('allow_list', []),
                    chosen_redact_comprehend_entities,
                    all_text_line_results
                )
                comprehend_query_number += 1        

        # Process results and create bounding boxes
        combined_results = []
        for i, text_line in enumerate(line_level_ocr_results):
            line_results = next((results for idx, results in all_text_line_results if idx == i), [])
            if line_results and i < len(ocr_results_with_words):
                child_level_key = list(ocr_results_with_words.keys())[i]
                ocr_results_with_words_line_level = ocr_results_with_words[child_level_key]
                
                for result in line_results:
                    bbox_results = self.map_analyzer_results_to_bounding_boxes(
                        [result],
                        [OCRResult(
                            text=text_line.text[result.start:result.end],
                            left=text_line.left,
                            top=text_line.top,
                            width=text_line.width,
                            height=text_line.height
                        )],
                        text_line.text,
                        text_analyzer_kwargs.get('allow_list', []),
                        ocr_results_with_words_line_level
                    )
                    combined_results.extend(bbox_results)

        return combined_results, comprehend_query_number

    @staticmethod
    def map_analyzer_results_to_bounding_boxes(
    text_analyzer_results: List[RecognizerResult],
    redaction_relevant_ocr_results: List[OCRResult],
    full_text: str,
    allow_list: List[str],
    ocr_results_with_words_child_info: Dict[str, Dict]
) -> List[CustomImageRecognizerResult]:
        redaction_bboxes = []

        for redaction_relevant_ocr_result in redaction_relevant_ocr_results:
            #print("ocr_results_with_words_child_info:", ocr_results_with_words_child_info)

            line_text = ocr_results_with_words_child_info['text']
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
                    matched_words = matched_text.split()
                    
                    # Find the corresponding words in the OCR results
                    matching_word_boxes = []

                    current_position = 0

                    for word_info in ocr_results_with_words_child_info.get('words', []):
                        word_text = word_info['text']
                        word_length = len(word_text)

                        word_start = current_position
                        word_end = current_position + word_length

                        # Update current position for the next word
                        current_position += word_length + 1  # +1 for the space after the word
                        
                        # Check if the word's bounding box is within the start and end bounds
                        if word_start >= start_in_line and word_end <= (end_in_line + 1):
                            matching_word_boxes.append(word_info['bounding_box'])
                            #print(f"Matched word: {word_info['text']}")
                    
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
                                score=redaction_result.score,
                                left=left,
                                top=top,
                                width=right - left,
                                height=bottom - top,
                                text=matched_text
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


def bounding_boxes_overlap(box1:List, box2:List):
    """Check if two bounding boxes overlap."""
    return (box1[0] < box2[2] and box2[0] < box1[2] and
            box1[1] < box2[3] and box2[1] < box1[3])
   
def map_back_entity_results(page_analyser_result:dict, page_text_mapping:dict, all_text_line_results:List[Tuple]):
    for entity in page_analyser_result:
        entity_start = entity.start
        entity_end = entity.end
        
        # Track if the entity has been added to any line
        added_to_line = False
        
        for batch_start, line_idx, original_line, chars in page_text_mapping:
            batch_end = batch_start + len(original_line.text)
            
            # Check if the entity overlaps with the current line
            if batch_start < entity_end and batch_end > entity_start:  # Overlap condition
                relative_start = max(0, entity_start - batch_start)  # Adjust start relative to the line
                relative_end = min(entity_end - batch_start, len(original_line.text))  # Adjust end relative to the line
                
                # Create a new adjusted entity
                adjusted_entity = copy.deepcopy(entity)
                adjusted_entity.start = relative_start
                adjusted_entity.end = relative_end
                
                # Check if this line already has an entry
                existing_entry = next((entry for idx, entry in all_text_line_results if idx == line_idx), None)
                
                if existing_entry is None:
                    all_text_line_results.append((line_idx, [adjusted_entity]))
                else:
                    existing_entry.append(adjusted_entity)  # Append to the existing list of entities
                
                added_to_line = True
        
        # If the entity spans multiple lines, you may want to handle that here
        if not added_to_line:
            # Handle cases where the entity does not fit in any line (optional)
            print(f"Entity '{entity}' does not fit in any line.")

    return all_text_line_results

def map_back_comprehend_entity_results(response:object, current_batch_mapping:List[Tuple], allow_list:List[str], chosen_redact_comprehend_entities:List[str], all_text_line_results:List[Tuple]):
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
        for batch_start, line_idx, original_line, chars, line_offset in current_batch_mapping:
            batch_end = batch_start + len(original_line.text[line_offset:])

            # Check if the entity overlaps with the current line
            if batch_start < entity_end and batch_end > entity_start:  # Overlap condition
                # Calculate the absolute position within the line
                relative_start = max(0, entity_start - batch_start + line_offset)
                relative_end = min(entity_end - batch_start + line_offset, len(original_line.text))

                result_text = original_line.text[relative_start:relative_end]

                if result_text not in allow_list:
                    adjusted_entity = entity.copy()
                    adjusted_entity["BeginOffset"] = relative_start  # Now relative to the full line
                    adjusted_entity["EndOffset"] = relative_end

                    recogniser_entity = recognizer_result_from_dict(adjusted_entity)

                    existing_entry = next((entry for idx, entry in all_text_line_results if idx == line_idx), None)
                    if existing_entry is None:
                        all_text_line_results.append((line_idx, [recogniser_entity]))
                    else:
                        existing_entry.append(recogniser_entity)  # Append to the existing list of entities

                added_to_line = True

        # Optional: Handle cases where the entity does not fit in any line
        if not added_to_line:
            print(f"Entity '{entity}' does not fit in any line.")

    return all_text_line_results

def do_aws_comprehend_call(current_batch:str, current_batch_mapping:List[Tuple], comprehend_client:botocore.client.BaseClient, language:str, allow_list:List[str], chosen_redact_comprehend_entities:List[str], all_text_line_results:List[Tuple]):
    if not current_batch:
        return all_text_line_results

    max_retries = 3
    retry_delay = 3

    for attempt in range(max_retries):
        try:
            response = comprehend_client.detect_pii_entities(
                Text=current_batch.strip(),
                LanguageCode=language
            )

            all_text_line_results = map_back_comprehend_entity_results(
                response, 
                current_batch_mapping, 
                allow_list, 
                chosen_redact_comprehend_entities, 
                all_text_line_results
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
    page_analyser_results: List = [],
    page_analysed_bounding_boxes: List = [],
    comprehend_client = None,
    allow_list: List[str] = None,
    pii_identification_method: str = "Local",
    nlp_analyser = None,
    score_threshold: float = 0.0,
    custom_entities: List[str] = None,
    comprehend_query_number:int = 0#,
    #merge_text_bounding_boxes_fn = merge_text_bounding_boxes
):
    #if not merge_text_bounding_boxes_fn:
    #    raise ValueError("merge_text_bounding_boxes_fn is required")
    
    page_text = ""
    page_text_mapping = []
    all_text_line_results = []
    comprehend_query_number = 0

    # Collect all text from the page
    for i, text_line in enumerate(line_level_text_results_list):
        #print("line_level_text_results_list:", line_level_text_results_list)
        if chosen_redact_entities:
            if page_text:
                #page_text += " | "
                page_text += " "
            
            start_pos = len(page_text)
            page_text += text_line.text
            page_text_mapping.append((start_pos, i, text_line, line_characters[i]))

    # Process based on identification method
    if pii_identification_method == "Local":
        if not nlp_analyser:
            raise ValueError("nlp_analyser is required for Local identification method")
        
        #print("page text:", page_text)

        page_analyser_result = nlp_analyser.analyze(
            text=page_text,
            language=language,
            entities=chosen_redact_entities,
            score_threshold=score_threshold,
            return_decision_process=True,
            allow_list=allow_list
        )

        
        all_text_line_results = map_back_entity_results(
            page_analyser_result, 
            page_text_mapping, 
            all_text_line_results
        )


    elif pii_identification_method == "AWS Comprehend":

        # Process custom entities if any
        if custom_entities:
            custom_redact_entities = [
                entity for entity in chosen_redact_comprehend_entities 
                if entity in custom_entities
            ]
            if custom_redact_entities:
                page_analyser_result = nlp_analyser.analyze(
                    text=page_text,
                    language=language,
                    entities=custom_redact_entities,
                    score_threshold=score_threshold,
                    return_decision_process=True,
                    allow_list=allow_list
                )

                all_text_line_results = map_back_entity_results(
                    page_analyser_result, 
                    page_text_mapping, 
                    all_text_line_results
                )

        current_batch = ""
        current_batch_mapping = []
        batch_char_count = 0
        batch_word_count = 0

        for i, text_line in enumerate(line_level_text_results_list):
            words = text_line.text.split()
            word_start_positions = []
            
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
                        all_text_line_results
                    )
                    comprehend_query_number += 1
                    
                    # Start new batch
                    current_batch = word
                    batch_word_count = 1
                    batch_char_count = len(word)
                    current_batch_mapping = [(0, i, text_line, line_characters[i], word_start_positions[word_idx])]
                else:
                    if current_batch:
                        current_batch += " "
                        batch_char_count += 1
                    current_batch += word
                    batch_char_count += len(word)
                    batch_word_count += 1
                    
                    if not current_batch_mapping or current_batch_mapping[-1][1] != i:
                        current_batch_mapping.append((
                            batch_char_count - len(word),
                            i,
                            text_line,
                            line_characters[i],
                            word_start_positions[word_idx]  # Add the word's start position within its line
                        ))

        # Process final batch
        if current_batch:
            all_text_line_results = do_aws_comprehend_call(
                current_batch,
                current_batch_mapping,
                comprehend_client,
                language,
                allow_list,
                chosen_redact_comprehend_entities,
                all_text_line_results
            )
            comprehend_query_number += 1

    # Process results for each line
    for i, text_line in enumerate(line_level_text_results_list):
        line_results = next((results for idx, results in all_text_line_results if idx == i), [])
        
        if line_results:
            text_line_bounding_boxes = merge_text_bounding_boxes(
                line_results,
                line_characters[i]
            )
            
            page_analyser_results.extend(line_results)
            page_analysed_bounding_boxes.extend(text_line_bounding_boxes)

    return page_analysed_bounding_boxes

def merge_text_bounding_boxes(analyser_results:dict, characters: List[LTChar], combine_pixel_dist: int = 20, vertical_padding: int = 0):
    '''
    Merge identified bounding boxes containing PII that are very close to one another
    '''
    analysed_bounding_boxes = []
    original_bounding_boxes = []  # List to hold original bounding boxes

    if len(analyser_results) > 0 and len(characters) > 0:
        # Extract bounding box coordinates for sorting
        bounding_boxes = []
        for result in analyser_results:
            #print("Result:", result)
            char_boxes = [char.bbox for char in characters[result.start:result.end] if isinstance(char, LTChar)]
            char_text = [char._text for char in characters[result.start:result.end] if isinstance(char, LTChar)]
            if char_boxes:
                # Calculate the bounding box that encompasses all characters
                left = min(box[0] for box in char_boxes)
                bottom = min(box[1] for box in char_boxes)
                right = max(box[2] for box in char_boxes)
                top = max(box[3] for box in char_boxes) + vertical_padding
                bbox = [left, bottom, right, top]
                bounding_boxes.append((bottom, left, result, bbox, char_text))  # (y, x, result, bbox, text)

                # Store original bounding boxes
                original_bounding_boxes.append({"text": "".join(char_text), "boundingBox": bbox, "result": copy.deepcopy(result)})
                #print("Original bounding boxes:", original_bounding_boxes)

        # Sort the results by y-coordinate and then by x-coordinate
        bounding_boxes.sort()

        merged_bounding_boxes = []
        current_box = None
        current_y = None
        current_result = None
        current_text = []

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

                if vertical_diff_bboxes <= 5 and horizontal_diff_bboxes <= combine_pixel_dist:
                    # Merge bounding boxes
                    #print("Merging boxes")
                    merged_box = current_box.copy()
                    merged_result = current_result
                    merged_text = current_text.copy()

                    merged_box[2] = next_box[2]  # Extend horizontally
                    merged_box[3] = max(current_box[3], next_box[3])  # Adjust the top
                    merged_result.end = max(current_result.end, result.end)  # Extend text range
                    try:
                        if current_result.entity_type != result.entity_type:
                            merged_result.entity_type = current_result.entity_type + " - " + result.entity_type
                        else:
                            merged_result.entity_type = current_result.entity_type
                    except Exception as e:
                        print("Unable to combine result entity types:", e)
                    if current_text:
                        merged_text.append(" ")  # Add space between texts
                    merged_text.extend(text)

                    merged_bounding_boxes.append({
                        "text": "".join(merged_text),
                        "boundingBox": merged_box,
                        "result": merged_result
                    })

                else:
                    # Start a new bounding box
                    current_box = next_box
                    current_y = next_box[1]
                    current_result = result
                    current_text = list(text)

        # Combine original and merged bounding boxes
        analysed_bounding_boxes.extend(original_bounding_boxes)
        analysed_bounding_boxes.extend(merged_bounding_boxes)

        #print("Analysed bounding boxes:", analysed_bounding_boxes)

    return analysed_bounding_boxes

def recreate_page_line_level_ocr_results_with_page(page_line_level_ocr_results_with_words: dict):
    reconstructed_results = []
    
    # Assume all lines belong to the same page, so we can just read it from one item
    #page = next(iter(page_line_level_ocr_results_with_words.values()))["page"]

    page = page_line_level_ocr_results_with_words["page"]
    
    for line_data in page_line_level_ocr_results_with_words["results"].values():
        bbox = line_data["bounding_box"]
        text = line_data["text"]

        # Recreate the OCRResult (you'll need the OCRResult class imported)
        line_result = OCRResult(
            text=text,
            left=bbox[0],
            top=bbox[1],
            width=bbox[2] - bbox[0],
            height=bbox[3] - bbox[1],
        )
        reconstructed_results.append(line_result)
    
    page_line_level_ocr_results_with_page = {"page": page, "results": reconstructed_results}
    
    return page_line_level_ocr_results_with_page

def split_words_and_punctuation_from_line(line_of_words: List[OCRResult]) -> List[OCRResult]:
    """
    Takes a list of OCRResult objects and splits words with trailing/leading punctuation.

    For a word like "example.", it creates two new OCRResult objects for "example"
    and "." and estimates their bounding boxes. Words with internal hyphens like
    "high-tech" are preserved.
    """
    # Punctuation that will be split off. Hyphen is not included.
    PUNCTUATION_TO_SPLIT = {'.', ',', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'}
    
    new_word_list = []
    
    for word_result in line_of_words:
        word_text = word_result.text
        
        # This regex finds a central "core" word, and captures leading and trailing punctuation
        # Handles cases like "(word)." -> group1='(', group2='word', group3='.'
        match = re.match(r"([(\[{]*)(.*?)_?([.,?!:;)\}\]]*)$", word_text)

        # Handle words with internal hyphens that might confuse the regex
        if '-' in word_text and not match.group(2):
             core_part_text = word_text
             leading_punc = ""
             trailing_punc = ""
        elif match:
            leading_punc, core_part_text, trailing_punc = match.groups()
        else: # Failsafe
            new_word_list.append(word_result)
            continue
            
        # If no split is needed, just add the original and continue
        if not leading_punc and not trailing_punc:
            new_word_list.append(word_result)
            continue
            
        # --- A split is required ---
        # Estimate new bounding boxes by proportionally allocating width
        original_width = word_result.width
        if not word_text or original_width == 0: continue # Failsafe
        
        avg_char_width = original_width / len(word_text)
        current_left = word_result.left

        # Add leading punctuation if it exists
        if leading_punc:
            punc_width = avg_char_width * len(leading_punc)
            new_word_list.append(OCRResult(
                text=leading_punc, left=current_left, top=word_result.top, 
                width=punc_width, height=word_result.height
            ))
            current_left += punc_width

        # Add the core part of the word
        if core_part_text:
            core_width = avg_char_width * len(core_part_text)
            new_word_list.append(OCRResult(
                text=core_part_text, left=current_left, top=word_result.top,
                width=core_width, height=word_result.height
            ))
            current_left += core_width

        # Add trailing punctuation if it exists
        if trailing_punc:
            punc_width = avg_char_width * len(trailing_punc)
            new_word_list.append(OCRResult(
                text=trailing_punc, left=current_left, top=word_result.top,
                width=punc_width, height=word_result.height
            ))
            
    return new_word_list

def create_ocr_result_with_children(combined_results:dict, i:int, current_bbox:dict, current_line:list):
        combined_results["text_line_" + str(i)] = {
        "line": i,
        'text': current_bbox.text,
        'bounding_box': (current_bbox.left, current_bbox.top, 
                            current_bbox.left + current_bbox.width, 
                            current_bbox.top + current_bbox.height),
        'words': [{'text': word.text, 
                    'bounding_box': (word.left, word.top, 
                                    word.left + word.width, 
                                    word.top + word.height)} 
                    for word in current_line]
    }
        return combined_results["text_line_" + str(i)]

def combine_ocr_results(ocr_results: List[OCRResult], x_threshold: float = 50.0, y_threshold: float = 12.0, page: int = 1):
    """
    Group OCR results into lines, splitting words from punctuation.
    """
    if not ocr_results:
        return {"page": page, "results": []}, {"page": page, "results": {}}

    lines = []
    current_line = []
    for result in sorted(ocr_results, key=lambda x: (x.top, x.left)):
        if not current_line or abs(result.top - current_line[0].top) <= y_threshold:
            current_line.append(result)
        else:
            lines.append(sorted(current_line, key=lambda x: x.left))
            current_line = [result]
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x.left))

    page_line_level_ocr_results = []
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
        
        final_line_bbox = OCRResult(
            text=line_text,
            left=line_left,
            top=line_top,
            width=line_right - line_left,
            height=line_bottom - line_top
        )
        
        page_line_level_ocr_results.append(final_line_bbox)
        
        # Use the PROCESSED line to create the children
        create_ocr_result_with_children(
            page_line_level_ocr_results_with_words, 
            line_counter, 
            final_line_bbox, 
            processed_line  # <-- Use the new, split list of words
        )
        line_counter += 1

    page_level_results_with_page = {"page": page, "results": page_line_level_ocr_results}
    page_level_results_with_words = {"page": page, "results": page_line_level_ocr_results_with_words}

    return page_level_results_with_page, page_level_results_with_words

