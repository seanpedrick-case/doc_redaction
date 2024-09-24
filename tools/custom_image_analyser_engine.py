import pytesseract
import numpy as np
from presidio_analyzer import AnalyzerEngine, RecognizerResult
#from presidio_image_redactor import ImagePreprocessor
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import cv2
import PIL
from PIL import ImageDraw, ImageFont, Image
from typing import Optional, Tuple, Union
from copy import deepcopy

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
            # image_preprocessor = ImagePreprocessor(
            #     c_low_contrast=10,
            #     c_high_contrast=20,
            #     contrast_threshold=0.5,
            #     bg_threshold=128,
            #     block_size=11
            # )
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

        #print("pre-processing metadata:", preprocessing_metadata)
        #image_processed.save("image_processed.png")

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
                text=ocr_result['text'][i],
                left=ocr_result['left'][i],
                top=ocr_result['top'][i],
                width=ocr_result['width'][i],
                height=ocr_result['height'][i]
            )
            for i in valid_indices
        ]

    def analyze_text(
        self, 
        ocr_results: List[OCRResult], 
        ocr_results_with_children: Dict[str, Dict],
        **text_analyzer_kwargs
    ) -> List[CustomImageRecognizerResult]:
        # Define English as default language, if not specified
        if "language" not in text_analyzer_kwargs:
            text_analyzer_kwargs["language"] = "en"
        
        allow_list = text_analyzer_kwargs.get('allow_list', [])

        combined_results = []
        for ocr_result in ocr_results:
            # Analyze each OCR result (line) individually
            analyzer_result = self.analyzer_engine.analyze(
                text=ocr_result.text, **text_analyzer_kwargs
            )
            
            for result in analyzer_result:
                # Extract the relevant portion of text based on start and end
                relevant_text = ocr_result.text[result.start:result.end]
                
                # Find the corresponding entry in ocr_results_with_children
                child_info = ocr_results_with_children.get(ocr_result.text)
                if child_info:
                    # Calculate left and width based on child words
                    #print("Found in ocr_results_with_children")
                    child_words = child_info['words']
                    start_word = child_words[0]
                    end_word = child_words[-1]
                    left = start_word['bounding_box'][0]
                    width = end_word['bounding_box'][2] - left

                    relevant_ocr_result = OCRResult(
                        text=relevant_text,
                        left=left,
                        top=ocr_result.top,
                        width=width,
                        height=ocr_result.height
                    )
                else:
                    # Fallback to previous method if not found in ocr_results_with_children
                    #print("Couldn't find result in ocr_results_with_children")
                    relevant_ocr_result = OCRResult(
                        text=relevant_text,
                        left=ocr_result.left + self.estimate_x_offset(relevant_text, result.start),
                        top=ocr_result.top,
                        width=self.estimate_width(ocr_result=ocr_result, start=result.start, end=result.end),
                        height=ocr_result.height
                    )

                result_mod = result
                result.start = 0
                result.end = len(relevant_text)
                
                # Map the analyzer results to bounding boxes for this line
                line_results = self.map_analyzer_results_to_bounding_boxes(
                    [result_mod], [relevant_ocr_result], ocr_result.text, allow_list, ocr_results_with_children
                )
                
                combined_results.extend(line_results)

        return combined_results

    @staticmethod
    def map_analyzer_results_to_bounding_boxes(
        text_analyzer_results: List[RecognizerResult],
        ocr_results: List[OCRResult],
        full_text: str,
        allow_list: List[str],
        ocr_results_with_children: Dict[str, Dict]
    ) -> List[CustomImageRecognizerResult]:
        pii_bboxes = []
        text_position = 0

        for ocr_result in ocr_results:
            word_end = text_position + len(ocr_result.text)

            #print("Checking relevant OCR result:", ocr_result)
            
            for result in text_analyzer_results:
                max_of_current_text_pos_or_result_start_pos = max(text_position, result.start)
                min_of_result_end_pos_or_results_end = min(word_end, result.end)

                #print("max_of_current_text_pos_or_result_start_pos", str(max_of_current_text_pos_or_result_start_pos))
                #print("min_of_result_end_pos_or_results_end", str(min_of_result_end_pos_or_results_end))

                if (max_of_current_text_pos_or_result_start_pos < min_of_result_end_pos_or_results_end) and (ocr_result.text not in allow_list):
                    print("result", result, "made it through if statement")

                    # Find the corresponding entry in ocr_results_with_children
                    child_info = ocr_results_with_children.get(full_text)
                    if child_info:
                        # Use the bounding box from ocr_results_with_children
                        bbox = child_info['bounding_box']
                        left, top, right, bottom = bbox
                        width = right - left
                        height = bottom - top
                    else:
                        # Fallback to ocr_result if not found
                        left = ocr_result.left
                        top = ocr_result.top
                        width = ocr_result.width
                        height = ocr_result.height

                    pii_bboxes.append(
                        CustomImageRecognizerResult(
                            entity_type=result.entity_type,
                            start=result.start,
                            end=result.end,
                            score=result.score,
                            left=left,
                            top=top,
                            width=width,
                            height=height,
                            text=ocr_result.text
                        )
                    )
            
            text_position = word_end + 1  # +1 for the space between words

        return pii_bboxes

    # @staticmethod
    # def map_analyzer_results_to_bounding_boxes(
    #     text_analyzer_results: List[RecognizerResult],
    #     ocr_results: List[OCRResult],
    #     full_text: str,
    #     allow_list: List[str],
    # ) -> List[CustomImageRecognizerResult]:
    #     pii_bboxes = []
    #     text_position = 0

    #     for ocr_result in ocr_results:
    #         word_end = text_position + len(ocr_result.text)

    #         print("Checking relevant OCR result:", ocr_result)
            
    #         for result in text_analyzer_results:
    #             if (max(text_position, result.start) < min(word_end, result.end)) and (ocr_result.text not in allow_list):
    #                 print("result", result, "made it through if statement")

    #                 pii_bboxes.append(
    #                     CustomImageRecognizerResult(
    #                         entity_type=result.entity_type,
    #                         start=result.start,
    #                         end=result.end,
    #                         score=result.score,
    #                         left=ocr_result.left,
    #                         top=ocr_result.top,
    #                         width=ocr_result.width,
    #                         height=ocr_result.height,
    #                         text=ocr_result.text
    #                     )
    #                 )
            
    #         text_position = word_end + 1  # +1 for the space between words

    #     return pii_bboxes
    
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


    # def estimate_width(self, ocr_result: OCRResult, start: int, end: int) -> int:
    #     # Extract the relevant text portion
    #     relevant_text = ocr_result.text[start:end]
        
    #     # Check if the relevant text is the entire text of the OCR result
    #     if relevant_text == ocr_result.text:
    #         return ocr_result.width
        
    #     # Estimate the font size based on the height of the bounding box
    #     estimated_font_size = ocr_result.height + 4
        
    #     # Create a blank image with enough width to measure the text
    #     dummy_image = Image.new('RGB', (1000, 50), color=(255, 255, 255))
    #     draw = ImageDraw.Draw(dummy_image)
        
    #     # Specify the font and size
    #     try:
    #         font = ImageFont.truetype("arial.ttf", estimated_font_size)  # Adjust the font file as needed
    #     except IOError:
    #         font = ImageFont.load_default()  # Fallback to default font if the specified font is not found
        
    #     # Draw the relevant text on the image
    #     draw.text((0, 0), relevant_text, fill=(0, 0, 0), font=font)
        
    #     # Save the image for debugging purposes
    #     dummy_image.save("debug_image.png")
        
    #     # Use pytesseract to get the bounding box of the relevant text
    #     bbox = pytesseract.image_to_boxes(dummy_image, config=self.tesseract_config)
        
    #     # Print the bbox for debugging
    #     print("Bounding box:", bbox)
        
    #     # Calculate the width from the bounding box
    #     if bbox:
    #         try:
    #             # Initialize min_left and max_right with extreme values
    #             min_left = float('inf')
    #             max_right = float('-inf')
                
    #             # Split the bbox string into lines
    #             bbox_lines = bbox.splitlines()
                
    #             for line in bbox_lines:
    #                 parts = line.split()
    #                 if len(parts) == 6:
    #                     _, left, _, right, _, _ = parts
    #                     left = int(left)
    #                     right = int(right)
    #                     min_left = min(min_left, left)
    #                     max_right = max(max_right, right)
                
    #             width = max_right - min_left
    #         except ValueError as e:
    #             print("Error parsing bounding box:", e)
    #             width = 0
    #     else:
    #         width = 0

    #     print("Estimated width:", width)
        
    #     return width



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
                new_format_results[current_bbox.text] = { # f"combined_text_{line_counter}"
                    'bounding_box': (current_bbox.left, current_bbox.top, 
                                     current_bbox.left + current_bbox.width, 
                                     current_bbox.top + current_bbox.height),
                    'words': [{'text': word.text, 
                               'bounding_box': (word.left, word.top, 
                                                word.left + word.width, 
                                                word.top + word.height)} 
                              for word in current_line]
                }
                line_counter += 1
                current_line = [result]
                current_bbox = result

    # Append the last line
    if current_bbox:
        combined_results.append(current_bbox)
        new_format_results[current_bbox.text] = { # f"combined_text_{line_counter}"
            'bounding_box': (current_bbox.left, current_bbox.top, 
                             current_bbox.left + current_bbox.width, 
                             current_bbox.top + current_bbox.height),
            'words': [{'text': word.text, 
                       'bounding_box': (word.left, word.top, 
                                        word.left + word.width, 
                                        word.top + word.height)} 
                      for word in current_line]
        }

    return combined_results, new_format_results

