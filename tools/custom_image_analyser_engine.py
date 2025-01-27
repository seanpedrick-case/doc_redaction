import pytesseract
import numpy as np
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import time
import cv2
import copy
from copy import deepcopy
from pdfminer.layout import LTChar
import PIL
from PIL import Image
from typing import Optional, Tuple, Union
from tools.helper_functions import clean_unicode_text
from tools.presidio_analyzer_custom import recognizer_result_from_dict
from tools.load_spacy_model_custom_recognisers import custom_entities

@dataclass
class OCRResult:
    text: str
    left: int
    top: int
    width: int
    height: int

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
    """ImagePreprocessor class.

    Parent class for image preprocessing objects.
    """

    def __init__(self, use_greyscale: bool = True) -> None:
        """Initialize the ImagePreprocessor class.

        :param use_greyscale: Whether to convert the image to greyscale.
        """
        self.use_greyscale = use_greyscale

    def preprocess_image(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """Preprocess the image to be analyzed.

        :param image: Loaded PIL image.

        :return: The processed image and any metadata regarding the
             preprocessing approach.
        """
        return image, {}

    def convert_image_to_array(self, image: Image.Image) -> np.ndarray:
        """Convert PIL image to numpy array.

        :param image: Loaded PIL image.
        :param convert_to_greyscale: Whether to convert the image to greyscale.

        :return: image pixels as a numpy array.

        """

        if isinstance(image, np.ndarray):
            img = image
        else:
            if self.use_greyscale:
                image = image.convert("L")
            img = np.asarray(image)
        return img

    @staticmethod
    def _get_bg_color(
        image: Image.Image, is_greyscale: bool, invert: bool = False
    ) -> Union[int, Tuple[int, int, int]]:
        """Select most common color as background color.

        :param image: Loaded PIL image.
        :param is_greyscale: Whether the image is greyscale.
        :param invert: TRUE if you want to get the inverse of the bg color.

        :return: Background color.
        """
        # Invert colors if invert flag is True
        if invert:
            if image.mode == "RGBA":
                # Handle transparency as needed
                r, g, b, a = image.split()
                rgb_image = Image.merge("RGB", (r, g, b))
                inverted_image = PIL.ImageOps.invert(rgb_image)
                r2, g2, b2 = inverted_image.split()

                image = Image.merge("RGBA", (r2, g2, b2, a))

            else:
                image = PIL.ImageOps.invert(image)

        # Get background color
        if is_greyscale:
            # Select most common color as color
            bg_color = int(np.bincount(image.flatten()).argmax())
        else:
            # Reduce size of image to 1 pixel to get dominant color
            tmp_image = image.copy()
            tmp_image = tmp_image.resize((1, 1), resample=0)
            bg_color = tmp_image.getpixel((0, 0))

        return bg_color

    @staticmethod
    def _get_image_contrast(image: np.ndarray) -> Tuple[float, float]:
        """Compute the contrast level and mean intensity of an image.

        :param image: Input image pixels (as a numpy array).

        :return: A tuple containing the contrast level and mean intensity of the image.
        """
        contrast = np.std(image)
        mean_intensity = np.mean(image)
        return contrast, mean_intensity
    
class BilateralFilter(ImagePreprocessor):
    """BilateralFilter class.

    The class applies bilateral filtering to an image. and returns the filtered
    image and metadata.
    """

    def __init__(
        self, diameter: int = 3, sigma_color: int = 40, sigma_space: int = 40
    ) -> None:
        """Initialize the BilateralFilter class.

        :param diameter: Diameter of each pixel neighborhood.
        :param sigma_color: value of sigma in the color space.
        :param sigma_space: value of sigma in the coordinate space.
        """
        super().__init__(use_greyscale=True)

        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def preprocess_image(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """Preprocess the image to be analyzed.

        :param image: Loaded PIL image.

        :return: The processed image and metadata (diameter, sigma_color, sigma_space).
        """
        image = self.convert_image_to_array(image)

        # Apply bilateral filtering
        filtered_image = cv2.bilateralFilter(
            image,
            self.diameter,
            self.sigma_color,
            self.sigma_space,
        )

        metadata = {
            "diameter": self.diameter,
            "sigma_color": self.sigma_color,
            "sigma_space": self.sigma_space,
        }

        return Image.fromarray(filtered_image), metadata

class SegmentedAdaptiveThreshold(ImagePreprocessor):
    """SegmentedAdaptiveThreshold class.

    The class applies adaptive thresholding to an image
    and returns the thresholded image and metadata.
    The parameters used to run the adaptivethresholding are selected based on
    the contrast level of the image.
    """

    def __init__(
        self,
        block_size: int = 5,
        contrast_threshold: int = 40,
        c_low_contrast: int = 10,
        c_high_contrast: int = 40,
        bg_threshold: int = 122,
    ) -> None:
        """Initialize the SegmentedAdaptiveThreshold class.

        :param block_size: Size of the neighborhood area for threshold calculation.
        :param contrast_threshold: Threshold for low contrast images.
        :param C_low_contrast: Constant added to the mean for low contrast images.
        :param C_high_contrast: Constant added to the mean for high contrast images.
        :param bg_threshold: Threshold for background color.
        """

        super().__init__(use_greyscale=True)
        self.block_size = block_size
        self.c_low_contrast = c_low_contrast
        self.c_high_contrast = c_high_contrast
        self.bg_threshold = bg_threshold
        self.contrast_threshold = contrast_threshold

    def preprocess_image(
        self, image: Union[Image.Image, np.ndarray]
    ) -> Tuple[Image.Image, dict]:
        """Preprocess the image.

        :param image: Loaded PIL image.

        :return: The processed image and metadata (C, background_color, contrast).
        """
        if not isinstance(image, np.ndarray):
            image = self.convert_image_to_array(image)

        # Determine background color
        background_color = self._get_bg_color(image, True)
        contrast, _ = self._get_image_contrast(image)

        c = (
            self.c_low_contrast
            if contrast <= self.contrast_threshold
            else self.c_high_contrast
        )

        if background_color < self.bg_threshold:
            adaptive_threshold_image = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                self.block_size,
                -c,
            )
        else:
            adaptive_threshold_image = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                self.block_size,
                c,
            )

        metadata = {"C": c, "background_color": background_color, "contrast": contrast}
        return Image.fromarray(adaptive_threshold_image), metadata
    
class ImageRescaling(ImagePreprocessor):
    """ImageRescaling class. Rescales images based on their size."""

    def __init__(
        self,
        small_size: int = 1048576,
        large_size: int = 4000000,
        factor: int = 2,
        interpolation: int = cv2.INTER_AREA,
    ) -> None:
        """Initialize the ImageRescaling class.

        :param small_size: Threshold for small image size.
        :param large_size: Threshold for large image size.
        :param factor: Scaling factor for resizing.
        :param interpolation: Interpolation method for resizing.
        """
        super().__init__(use_greyscale=True)

        self.small_size = small_size
        self.large_size = large_size
        self.factor = factor
        self.interpolation = interpolation

    def preprocess_image(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """Preprocess the image to be analyzed.

        :param image: Loaded PIL image.

        :return: The processed image and metadata (scale_factor).
        """

        scale_factor = 1
        if image.size < self.small_size:
            scale_factor = self.factor
        elif image.size > self.large_size:
            scale_factor = 1 / self.factor

        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        dimensions = (width, height)

        # resize image
        rescaled_image = cv2.resize(image, dimensions, interpolation=self.interpolation)
        metadata = {"scale_factor": scale_factor}
        return Image.fromarray(rescaled_image), metadata

class ContrastSegmentedImageEnhancer(ImagePreprocessor):
    """Class containing all logic to perform contrastive segmentation.

    Contrastive segmentation is a preprocessing step that aims to enhance the
    text in an image by increasing the contrast between the text and the
    background. The parameters used to run the preprocessing are selected based
    on the contrast level of the image.
    """

    def __init__(
        self,
        bilateral_filter: Optional[BilateralFilter] = None,
        adaptive_threshold: Optional[SegmentedAdaptiveThreshold] = None,
        image_rescaling: Optional[ImageRescaling] = None,
        low_contrast_threshold: int = 40,
    ) -> None:
        """Initialize the class.

        :param bilateral_filter: Optional BilateralFilter instance.
        :param adaptive_threshold: Optional AdaptiveThreshold instance.
        :param image_rescaling: Optional ImageRescaling instance.
        :param low_contrast_threshold: Threshold for low contrast images.
        """

        super().__init__(use_greyscale=True)
        if not bilateral_filter:
            self.bilateral_filter = BilateralFilter()
        else:
            self.bilateral_filter = bilateral_filter

        if not adaptive_threshold:
            self.adaptive_threshold = SegmentedAdaptiveThreshold()
        else:
            self.adaptive_threshold = adaptive_threshold

        if not image_rescaling:
            self.image_rescaling = ImageRescaling()
        else:
            self.image_rescaling = image_rescaling

        self.low_contrast_threshold = low_contrast_threshold

    def preprocess_image(self, image: Image.Image) -> Tuple[Image.Image, dict]:
        """Preprocess the image to be analyzed.

        :param image: Loaded PIL image.

        :return: The processed image and metadata (background color, scale percentage,
             contrast level, and C value).
        """
        image = self.convert_image_to_array(image)

        # Apply bilateral filtering
        filtered_image, _ = self.bilateral_filter.preprocess_image(image)

        # Convert to grayscale
        pil_filtered_image = Image.fromarray(np.uint8(filtered_image))
        pil_grayscale_image = pil_filtered_image.convert("L")
        grayscale_image = np.asarray(pil_grayscale_image)

        # Improve contrast
        adjusted_image, _, adjusted_contrast = self._improve_contrast(grayscale_image)

        # Adaptive Thresholding
        adaptive_threshold_image, _ = self.adaptive_threshold.preprocess_image(
            adjusted_image
        )
        # Increase contrast
        _, threshold_image = cv2.threshold(
            np.asarray(adaptive_threshold_image),
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )

        # Rescale image
        rescaled_image, scale_metadata = self.image_rescaling.preprocess_image(
            threshold_image
        )

        return rescaled_image, scale_metadata

    def _improve_contrast(self, image: np.ndarray) -> Tuple[np.ndarray, str, str]:
        """Improve the contrast of an image based on its initial contrast level.

        :param image: Input image.

        :return: A tuple containing the improved image, the initial contrast level,
             and the adjusted contrast level.
        """
        contrast, mean_intensity = self._get_image_contrast(image)

        if contrast <= self.low_contrast_threshold:
            alpha = 1.5
            beta = -mean_intensity * alpha
            adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            adjusted_contrast, _ = self._get_image_contrast(adjusted_image)
        else:
            adjusted_image = image
            adjusted_contrast = contrast
        return adjusted_image, contrast, adjusted_contrast

def bounding_boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    return (box1[0] < box2[2] and box2[0] < box1[2] and
            box1[1] < box2[3] and box2[1] < box1[3])
   
def map_back_entity_results(page_analyser_result, page_text_mapping, all_text_line_results):
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

def map_back_comprehend_entity_results(response, current_batch_mapping, allow_list, chosen_redact_comprehend_entities, all_text_line_results):
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

def do_aws_comprehend_call(current_batch, current_batch_mapping, comprehend_client, language, allow_list, chosen_redact_comprehend_entities, all_text_line_results):
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

        #print("page_analyser_result:", page_analyser_result)
        
        all_text_line_results = map_back_entity_results(
            page_analyser_result, 
            page_text_mapping, 
            all_text_line_results
        )

        #print("all_text_line_results:", all_text_line_results)

    elif pii_identification_method == "AWS Comprehend":
        #print("page text:", page_text)

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

                print("page_analyser_result:", page_analyser_result)

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

def merge_text_bounding_boxes(analyser_results, characters: List[LTChar], combine_pixel_dist: int = 20, vertical_padding: int = 0):
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

# Function to combine OCR results into line-level results
def combine_ocr_results(ocr_results, x_threshold=50, y_threshold=12):
    # Group OCR results into lines based on y_threshold
    lines = []
    current_line = []
    for result in sorted(ocr_results, key=lambda x: x.top):
        if not current_line or abs(result.top - current_line[0].top) <= y_threshold:
            current_line.append(result)
        else:
            lines.append(current_line)
            current_line = [result]
    if current_line:
        lines.append(current_line)

    # Sort each line by left position
    for line in lines:
        line.sort(key=lambda x: x.left)

    # Flatten the sorted lines back into a single list
    sorted_results = [result for line in lines for result in line]

    combined_results = []
    new_format_results = {}
    current_line = []
    current_bbox = None
    line_counter = 1

    def create_ocr_result_with_children(combined_results, i, current_bbox, current_line):
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

    for result in sorted_results:
        if not current_line:
            # Start a new line
            current_line.append(result)
            current_bbox = result
        else:
            # Check if the result is on the same line (y-axis) and close horizontally (x-axis)
            last_result = current_line[-1]

            if abs(result.top - last_result.top) <= y_threshold and \
               (result.left - (last_result.left + last_result.width)) <= x_threshold:
                # Update the bounding box to include the new word
                new_right = max(current_bbox.left + current_bbox.width, result.left + result.width)
                current_bbox = OCRResult(
                    text=f"{current_bbox.text} {result.text}",
                    left=current_bbox.left,
                    top=current_bbox.top,
                    width=new_right - current_bbox.left,
                    height=max(current_bbox.height, result.height)
                )
                current_line.append(result)
            else:
                

                # Commit the current line and start a new one
                combined_results.append(current_bbox)

                new_format_results["text_line_" + str(line_counter)] = create_ocr_result_with_children(new_format_results, line_counter, current_bbox, current_line)

                line_counter += 1
                current_line = [result]
                current_bbox = result

    # Append the last line
    if current_bbox:
        combined_results.append(current_bbox)

        new_format_results["text_line_" + str(line_counter)] = create_ocr_result_with_children(new_format_results, line_counter, current_bbox, current_line)


    return combined_results, new_format_results

class CustomImageAnalyzerEngine:
    def __init__(
        self,
        analyzer_engine: Optional[AnalyzerEngine] = None,
        tesseract_config: Optional[str] = None,
        image_preprocessor: Optional[ImagePreprocessor] = None
    ):
        if not analyzer_engine:
            analyzer_engine = AnalyzerEngine()
        self.analyzer_engine = analyzer_engine
        self.tesseract_config = tesseract_config or '--oem 3 --psm 11'

        if not image_preprocessor:
            image_preprocessor = ContrastSegmentedImageEnhancer()
            #print(image_preprocessor)
        self.image_preprocessor = image_preprocessor

    def perform_ocr(self, image: Union[str, Image.Image, np.ndarray]) -> List[OCRResult]:
        # Ensure image is a PIL Image
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_processed, preprocessing_metadata = self.image_preprocessor.preprocess_image(image)

        ocr_data = pytesseract.image_to_data(image_processed, output_type=pytesseract.Output.DICT, config=self.tesseract_config)

        if preprocessing_metadata and ("scale_factor" in preprocessing_metadata):
            ocr_result = self._scale_bbox_results(
                ocr_data, preprocessing_metadata["scale_factor"]
            )

        ocr_result = self.remove_space_boxes(ocr_result)
        
        # Filter out empty strings and low confidence results
        valid_indices = [i for i, text in enumerate(ocr_result['text']) if text.strip() and int(ocr_result['conf'][i]) > 0]
        
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
        ocr_results_with_children: Dict[str, Dict],
        chosen_redact_comprehend_entities: List[str],
        pii_identification_method: str = "Local",
        comprehend_client = "",      
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
            if line_results and i < len(ocr_results_with_children):
                child_level_key = list(ocr_results_with_children.keys())[i]
                ocr_results_with_children_line_level = ocr_results_with_children[child_level_key]
                
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
                        ocr_results_with_children_line_level
                    )
                    combined_results.extend(bbox_results)

        return combined_results, comprehend_query_number

    @staticmethod
    def map_analyzer_results_to_bounding_boxes(
    text_analyzer_results: List[RecognizerResult],
    redaction_relevant_ocr_results: List[OCRResult],
    full_text: str,
    allow_list: List[str],
    ocr_results_with_children_child_info: Dict[str, Dict]
) -> List[CustomImageRecognizerResult]:
        redaction_bboxes = []

        for redaction_relevant_ocr_result in redaction_relevant_ocr_results:
            #print("ocr_results_with_children_child_info:", ocr_results_with_children_child_info)

            line_text = ocr_results_with_children_child_info['text']
            line_length = len(line_text)
            redaction_text = redaction_relevant_ocr_result.text

            #print(f"Processing line: '{line_text}'")
            
            for redaction_result in text_analyzer_results:
                #print(f"Checking redaction result: {redaction_result}")
                #print("redaction_text:", redaction_text)
                #print("line_length:", line_length)
                #print("line_text:", line_text)
                
                # Check if the redaction text is not in the allow list
                
                if redaction_text not in allow_list:
                    
                    # Adjust start and end to be within line bounds
                    start_in_line = max(0, redaction_result.start)
                    end_in_line = min(line_length, redaction_result.end)
                    
                    # Get the matched text from this line
                    matched_text = line_text[start_in_line:end_in_line]
                    matched_words = matched_text.split()
                    
                    # print(f"Found match: '{matched_text}' in line")

                    # for word_info in ocr_results_with_children_child_info.get('words', []):
                    #     # Check if this word is part of our match
                    #     if any(word.lower() in word_info['text'].lower() for word in matched_words):
                    #         matching_word_boxes.append(word_info['bounding_box'])
                    #         print(f"Matched word: {word_info['text']}")
                    
                    # Find the corresponding words in the OCR results
                    matching_word_boxes = []
                    
                    #print("ocr_results_with_children_child_info:", ocr_results_with_children_child_info)

                    current_position = 0

                    for word_info in ocr_results_with_children_child_info.get('words', []):
                        word_text = word_info['text']
                        word_length = len(word_text)

                        # Assign start and end character positions
                        #word_info['start_position'] = current_position
                        #word_info['end_position'] = current_position + word_length

                        word_start = current_position
                        word_end = current_position + word_length

                        # Update current position for the next word
                        current_position += word_length + 1  # +1 for the space after the word

                        #print("word_info['bounding_box']:", word_info['bounding_box'])
                        #print("word_start:", word_start)
                        #print("start_in_line:", start_in_line)

                        #print("word_end:", word_end)
                        #print("end_in_line:", end_in_line)
                        
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
                        #print(f"Added bounding box for: '{matched_text}'")

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
