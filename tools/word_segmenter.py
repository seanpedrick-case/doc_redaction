import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from tools.config import OUTPUT_FOLDER, SAVE_WORD_SEGMENTER_OUTPUT_IMAGES

# Adaptive thresholding parameters
BLOCK_SIZE_FACTOR = 1.5  # Multiplier for adaptive threshold block size
C_VALUE = 4  # Constant subtracted from mean in adaptive thresholding

# Word segmentation search parameters
INITIAL_KERNEL_WIDTH_FACTOR = 0.0  # Starting kernel width factor for Stage 2 search
INITIAL_VALLEY_THRESHOLD_FACTOR = (
    0.0  # Starting valley threshold factor for Stage 1 search
)
MAIN_VALLEY_THRESHOLD_FACTOR = (
    0.15  # Primary valley threshold factor for word separation
)
MIN_SPACE_FACTOR = 0.3  # Minimum space width relative to character width
MATCH_TOLERANCE = 1  # Tolerance for word count matching

# Noise removal parameters
MIN_AREA_THRESHOLD = 6  # Minimum component area to be considered valid text
DEFAULT_TRIM_PERCENTAGE = (
    0.2  # Percentage to trim from top/bottom for vertical cropping
)

# Skew detection parameters
MIN_SKEW_THRESHOLD = 0.5  # Ignore angles smaller than this (likely noise)
MAX_SKEW_THRESHOLD = 15.0  # Angles larger than this are extreme and likely errors


def _sanitize_filename(filename: str, max_length: int = 100) -> str:
    """
    Sanitizes a string to be used as a valid filename.
    Removes or replaces invalid characters for Windows/Linux file systems.

    Args:
        filename: The string to sanitize
        max_length: Maximum length of the sanitized filename

    Returns:
        A sanitized string safe for use in file names
    """
    if not filename:
        return "unnamed"

    # Replace spaces with underscores
    sanitized = filename.replace(" ", "_")

    # Remove or replace invalid characters for Windows/Linux
    # Invalid: < > : " / \ | ? *
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        sanitized = sanitized.replace(char, "_")

    # Remove control characters
    sanitized = "".join(
        char for char in sanitized if ord(char) >= 32 or char in "\n\r\t"
    )

    # Remove leading/trailing dots and spaces (Windows doesn't allow these)
    sanitized = sanitized.strip(". ")

    # Replace multiple consecutive underscores with a single one
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    # Ensure it's not empty after sanitization
    if not sanitized:
        sanitized = "unnamed"

    return sanitized


class AdaptiveSegmenter:
    """
    Line to word segmentation pipeline. It features:
    1. Adaptive Thresholding.
    2. Targeted Noise Removal using Connected Component Analysis to isolate the main text body.
    3. The robust two-stage adaptive search (Valley -> Kernel).
    4. CCA for final pixel-perfect refinement.
    """

    def __init__(self, output_folder: str = OUTPUT_FOLDER):
        self.output_folder = output_folder

    def _correct_orientation(
        self, gray_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects and corrects 90-degree orientation issues (e.g., vertical text).
        This runs *before* the fine-grained _deskew_image function.

        Returns the oriented image and the transformation matrix.
        """
        h, w = gray_image.shape
        center = (w // 2, h // 2)

        # --- STEP 1: Binarization for orientation detection ---
        block_size = 21
        if h < block_size:
            block_size = h if h % 2 != 0 else h - 1

        if block_size > 3:
            binary = cv2.adaptiveThreshold(
                gray_image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                4,
            )
        else:
            _, binary = cv2.threshold(
                gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

        # Remove small noise artifacts
        opening_kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, opening_kernel)

        # --- STEP 2: Extract text pixel coordinates ---
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 50:
            # print(
            #     "Warning: Not enough text pixels for orientation. Assuming horizontal."
            # )
            M_orient = cv2.getRotationMatrix2D(center, 0, 1.0)
            return gray_image, M_orient

        # --- STEP 3: Determine orientation using bounding box dimensions ---
        # Use simple bounding box instead of minAreaRect to avoid angle ambiguity
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        box_height = ymax - ymin
        box_width = xmax - xmin

        # --- STEP 4: Apply 90-degree rotation if text is vertical ---
        orientation_angle = 0.0
        if box_height > box_width:
            orientation_angle = 90.0
        else:
            # Already horizontal, no correction needed
            M_orient = cv2.getRotationMatrix2D(center, 0, 1.0)
            return gray_image, M_orient

        # --- STEP 5: Create and apply rotation transformation ---
        M_orient = cv2.getRotationMatrix2D(center, orientation_angle, 1.0)

        # Calculate new image dimensions (width and height are swapped after 90° rotation)
        new_w, new_h = h, w

        # Adjust translation to center the rotated image
        M_orient[0, 2] += (new_w - w) / 2
        M_orient[1, 2] += (new_h - h) / 2

        oriented_gray = cv2.warpAffine(
            gray_image,
            M_orient,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return oriented_gray, M_orient

    def _deskew_image(self, gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects skew using a robust method that normalizes the output of
        cv2.minAreaRect to correctly handle its angle/dimension ambiguity.
        """
        h, w = gray_image.shape

        # --- STEP 1: Binarization for skew detection ---
        block_size = 21
        if h < block_size:
            block_size = h if h % 2 != 0 else h - 1

        if block_size > 3:
            binary = cv2.adaptiveThreshold(
                gray_image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                4,
            )
        else:
            _, binary = cv2.threshold(
                gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

        # Remove small noise artifacts
        opening_kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, opening_kernel)

        # --- STEP 2: Extract text pixel coordinates ---
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 50:
            # Not enough pixels for reliable skew detection
            M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, 1.0)
            return gray_image, M

        # --- STEP 3: Calculate minimum area rectangle ---
        rect = cv2.minAreaRect(coords[:, ::-1])
        rect_width, rect_height = rect[1]
        angle = rect[2]

        # --- STEP 4: Normalize rectangle orientation ---
        # minAreaRect can return vertical rectangles; normalize to horizontal
        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
            angle += 90

        # --- STEP 5: Normalize angle to [-45, 45] range ---
        # minAreaRect returns angles in [-90, 0), normalize to horizontal baseline
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

        correction_angle = angle

        # --- STEP 6: Apply sanity checks on detected angle ---
        if abs(correction_angle) < MIN_SKEW_THRESHOLD:
            # Angle too small, likely noise
            correction_angle = 0.0
        elif abs(correction_angle) > MAX_SKEW_THRESHOLD:
            # Angle too extreme, likely detection error
            correction_angle = 0.0

        # --- STEP 7: Create rotation matrix and apply correction ---
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, correction_angle, 1.0)

        deskewed_gray = cv2.warpAffine(
            gray_image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return deskewed_gray, M

    def _get_boxes_from_profile(
        self,
        binary_image: np.ndarray,
        stable_avg_char_width: float,
        min_space_factor: float,
        valley_threshold_factor: float,
    ) -> List:
        """
        Extracts word bounding boxes from vertical projection profile.

        Uses valley detection in the vertical projection to identify word boundaries,
        with gap patching to handle small intra-word spaces.
        """
        img_h, img_w = binary_image.shape
        vertical_projection = np.sum(binary_image, axis=0)
        peaks = vertical_projection[vertical_projection > 0]
        if len(peaks) == 0:
            return []
        avg_peak_height = np.mean(peaks)
        valley_threshold = int(avg_peak_height * valley_threshold_factor)
        min_space_width = int(stable_avg_char_width * min_space_factor)
        patched_projection = vertical_projection.copy()
        in_gap = False
        gap_start = 0
        for x, col_sum in enumerate(patched_projection):
            if col_sum <= valley_threshold and not in_gap:
                in_gap = True
                gap_start = x
            elif col_sum > valley_threshold and in_gap:
                in_gap = False
                if (x - gap_start) < min_space_width:
                    patched_projection[gap_start:x] = int(avg_peak_height)
        unlabeled_boxes = []
        in_word = False
        start_x = 0
        for x, col_sum in enumerate(patched_projection):
            if col_sum > valley_threshold and not in_word:
                start_x = x
                in_word = True
            elif col_sum <= valley_threshold and in_word:
                unlabeled_boxes.append((start_x, 0, x - start_x, img_h))
                in_word = False
        if in_word:
            unlabeled_boxes.append((start_x, 0, img_w - start_x, img_h))
        return unlabeled_boxes

    def _enforce_logical_constraints(
        self, output: Dict[str, List], image_width: int, image_height: int
    ) -> Dict[str, List]:
        """
        Enforces geometric sanity checks with 2D awareness.
        """
        if not output or not output["text"]:
            return output

        # --- STEP 1: Unpack input data into box dictionaries ---
        num_items = len(output["text"])
        boxes = []
        for i in range(num_items):
            boxes.append(
                {
                    "text": output["text"][i],
                    "left": int(output["left"][i]),
                    "top": int(output["top"][i]),
                    "width": int(output["width"][i]),
                    "height": int(output["height"][i]),
                    "conf": output["conf"][i],
                }
            )

        # --- STEP 2: Clamp boxes to image boundaries ---
        valid_boxes = []
        for box in boxes:
            x0 = max(0, box["left"])
            y0 = max(0, box["top"])
            x1 = min(image_width, box["left"] + box["width"])
            y1 = min(image_height, box["top"] + box["height"])

            w = x1 - x0
            h = y1 - y0

            if w > 0 and h > 0:
                box["left"] = x0
                box["top"] = y0
                box["width"] = w
                box["height"] = h
                valid_boxes.append(box)
        boxes = valid_boxes

        # --- STEP 3: Sort boxes by reading direction ---
        is_vertical = image_height > (image_width * 1.2)
        if is_vertical:
            boxes.sort(key=lambda b: (b["top"], b["left"]))
        else:
            boxes.sort(key=lambda b: (b["left"], -b["width"]))

        # --- STEP 4: Remove nested boxes (2D-aware) ---
        final_pass_boxes = []
        if boxes:
            keep_indices = [True] * len(boxes)

            for i in range(len(boxes)):
                for j in range(len(boxes)):
                    if i == j:
                        continue

                    b1 = boxes[i]
                    b2 = boxes[j]

                    # Check nesting X
                    x_nested = (b1["left"] >= b2["left"] - 2) and (
                        b1["left"] + b1["width"] <= b2["left"] + b2["width"] + 2
                    )
                    # Check nesting Y
                    y_nested = (b1["top"] >= b2["top"] - 2) and (
                        b1["top"] + b1["height"] <= b2["top"] + b2["height"] + 2
                    )

                    if x_nested and y_nested:
                        # Only remove if text is identical (duplicate)
                        # OR if the inner box is likely noise compared to container
                        if b1["text"] == b2["text"]:
                            if b1["width"] * b1["height"] <= b2["width"] * b2["height"]:
                                keep_indices[i] = False

            for i, keep in enumerate(keep_indices):
                if keep:
                    final_pass_boxes.append(boxes[i])

        boxes = final_pass_boxes

        # --- STEP 5: Resolve overlaps (Smart Engulfment Logic) ---
        # Re-sort to ensure processing order is correct after deletions
        if is_vertical:
            boxes.sort(key=lambda b: (b["top"], b["left"]))
        else:
            boxes.sort(key=lambda b: (b["left"], -b["width"]))

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                b1 = boxes[i]
                b2 = boxes[j]

                # Calculate intersection
                x_overlap = min(
                    b1["left"] + b1["width"], b2["left"] + b2["width"]
                ) - max(b1["left"], b2["left"])
                y_overlap = min(
                    b1["top"] + b1["height"], b2["top"] + b2["height"]
                ) - max(b1["top"], b2["top"])

                if x_overlap > 0 and y_overlap > 0:
                    if is_vertical:
                        # Vertical Logic (Simpler, top-down)
                        if b1["top"] < b2["top"]:
                            new_h = max(1, b2["top"] - b1["top"])
                            b1["height"] = new_h
                    else:
                        # Horizontal Logic with Engulfment Check
                        if b1["left"] < b2["left"]:
                            # b1 is the "Container" or "Left" box
                            # b2 is the "Inner" or "Right" box

                            b1_right = b1["left"] + b1["width"]
                            b2_right = b2["left"] + b2["width"]

                            # Calculate potential remaining pieces of b1
                            left_slice_width = max(0, b2["left"] - b1["left"])
                            right_slice_width = max(0, b1_right - b2_right)

                            # Check if b1 essentially "engulfs" b2 (extends significantly past it)
                            if (
                                b1_right > b2_right
                                and right_slice_width > left_slice_width
                            ):
                                # CASE: Engulfment -> The "meat" of b1 is actually AFTER b2
                                # Move b1 to start after b2 ends
                                b1["left"] = b2_right
                                b1["width"] = right_slice_width
                            else:
                                # CASE: Standard Overlap -> b1 is to the left of b2
                                # Trim b1 to stop at b2
                                b1["width"] = max(1, left_slice_width)

        # --- STEP 6: Repack ---
        cleaned_output = {
            k: [] for k in ["text", "left", "top", "width", "height", "conf"]
        }
        # Re-sort one last time to ensure final order is visually correct
        if is_vertical:
            boxes.sort(key=lambda b: (b["top"], b["left"]))
        else:
            boxes.sort(key=lambda b: (b["left"], -b["width"]))

        for box in boxes:
            for key in cleaned_output.keys():
                cleaned_output[key].append(box[key])

        return cleaned_output

    def _is_geometry_valid(
        self,
        boxes: List[Tuple[int, int, int, int]],
        words: List[str],
        expected_height: float = 0,
    ) -> bool:
        """
        Validates if the detected boxes are physically plausible.
        FIXED: Reduced threshold to 0.20 to allow for small words like "of" in mixed-case text.
        """
        if len(boxes) != len(words):
            return False

        baseline = expected_height
        if baseline < 5:
            heights = [b[3] for b in boxes]
            if heights:
                baseline = np.median(heights)

        if baseline < 5:
            return True

        for i, box in enumerate(boxes):
            word = words[i]
            num_chars = len(word)
            if num_chars < 1 or not any(c.isalnum() for c in word):
                continue

            width = box[2]
            height = box[3]

            # 0.20 to catch small words like 'of'
            # print(f"Checking box for '{word}' (H:{height}px, B:{baseline}px)")
            if height < (baseline * 0.20):
                # print(f"Rejecting segmentation: Box for '{word}' is too thin (H:{height}px, B:{baseline}px)")
                return False

            avg_char_width = width / num_chars
            min_expected = baseline * 0.20
            if avg_char_width < min_expected and avg_char_width < 4:
                # print(f"Rejecting segmentation: Box for '{word}' is too thin (W:{avg_char_width}px, B:{baseline}px)")
                return False

        # print(f"Geometry validation passed for {len(boxes)} boxes")
        return True

    def segment(
        self,
        line_data: Dict[str, List],
        line_image: np.ndarray,
        min_space_factor=MIN_SPACE_FACTOR,
        match_tolerance=MATCH_TOLERANCE,
        image_name: str = None,
    ) -> Tuple[Dict[str, List], bool]:

        if line_image is None:
            # print(
            #     f"Error: line_image is None in segment function (image_name: {image_name})"
            # )
            return ({}, False)

        # Validate line_image is a valid numpy array
        if not isinstance(line_image, np.ndarray):
            # print(
            #     f"Error: line_image is not a numpy array (type: {type(line_image)}, image_name: {image_name})"
            # )
            return ({}, False)

        # Validate line_image has valid shape and size
        if line_image.size == 0:
            # print(
            #     f"Error: line_image is empty (shape: {line_image.shape}, image_name: {image_name})"
            # )
            return ({}, False)

        if len(line_image.shape) < 2:
            # print(
            #     f"Error: line_image has invalid shape {line_image.shape} (image_name: {image_name})"
            # )
            return ({}, False)

        # Early return if 1 or fewer words
        if line_data and line_data.get("text") and len(line_data["text"]) > 0:
            line_text = line_data["text"][0]
            words = line_text.split()
            if len(words) <= 1:
                return ({}, False)
        else:
            # print(
            #     f"Error: line_data is empty or does not contain text (image_name: {image_name})"
            # )
            return ({}, False)

        line_number = line_data["line"][0]
        # Sanitize all filename components
        safe_image_name = _sanitize_filename(image_name or "image", max_length=50)
        safe_line_number = _sanitize_filename(str(line_number), max_length=10)
        safe_shortened_line_text = _sanitize_filename(line_text, max_length=10)

        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f"{self.output_folder}/word_segmentation/{safe_image_name}_{safe_line_number}_{safe_shortened_line_text}_original.png"
            os.makedirs(f"{self.output_folder}/word_segmentation", exist_ok=True)
            cv2.imwrite(output_path, line_image)
            # print(f"\nSaved original image to '{output_path}'")

        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

        # ========================================================================
        # IMAGE PREPROCESSING: ORIENTATION AND DESKEWING
        # ========================================================================

        # --- STEP 1: Correct major orientation (90-degree rotations) ---
        # M_orient: ORIGINAL -> ORIENTED
        oriented_gray, M_orient = self._correct_orientation(gray)

        # --- STEP 2: Correct minor skew (small angle corrections) ---
        # M_skew: ORIENTED -> DESKEWED
        deskewed_gray, M_skew = self._deskew_image(oriented_gray)

        # --- STEP 3: Combine transformations into single matrix ---
        # Combine M_orient and M_skew: M_total = M_skew * M_orient
        # This transforms directly from ORIGINAL -> DESKEWED
        M_orient_3x3 = np.vstack([M_orient, [0, 0, 1]])
        M_skew_3x3 = np.vstack([M_skew, [0, 0, 1]])
        M_total_3x3 = M_skew_3x3 @ M_orient_3x3
        M = M_total_3x3[0:2, :]  # Extract 2x3 affine matrix

        # --- STEP 4: Apply combined transformation to original color image ---
        h, w = deskewed_gray.shape

        deskewed_line_image = cv2.warpAffine(
            line_image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Validate deskewed image
        if (
            deskewed_line_image is None
            or not isinstance(deskewed_line_image, np.ndarray)
            or deskewed_line_image.size == 0
        ):
            return ({}, False)

        # Save deskewed image (if enabled)
        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f"{self.output_folder}/word_segmentation/{safe_image_name}_{safe_line_number}_{safe_shortened_line_text}_deskewed.png"
            os.makedirs(f"{self.output_folder}/word_segmentation", exist_ok=True)
            cv2.imwrite(output_path, deskewed_line_image)

        # ========================================================================
        # MAIN SEGMENTATION PIPELINE
        # ========================================================================

        # --- STEP 1: Calculate character width estimate for adaptive parameters ---
        approx_char_count = len(line_data["text"][0].replace(" ", ""))
        if approx_char_count == 0:
            return {}, False
        img_h, img_w = deskewed_gray.shape
        estimated_char_height = (
            img_h * 0.6
        )  # conservative estimate (60% of line height)
        avg_char_width_approx = img_w / approx_char_count
        block_size = int(avg_char_width_approx * BLOCK_SIZE_FACTOR)
        if block_size % 2 == 0:
            block_size += 1

        # Validate input image
        if deskewed_gray is None or not isinstance(deskewed_gray, np.ndarray):
            return ({}, False)

        if len(deskewed_gray.shape) != 2:
            return ({}, False)

        if block_size < 3:
            block_size = 3

        # --- STEP 2: Adaptive binarization ---
        binary = cv2.adaptiveThreshold(
            deskewed_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            C_VALUE,
        )

        # Validate binary image
        if binary is None or not isinstance(binary, np.ndarray) or binary.size == 0:
            return ({}, False)

        # Save intermediate binary image (if enabled)
        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f"{self.output_folder}/word_segmentation/{safe_image_name}_{safe_line_number}_{safe_shortened_line_text}_binary.png"
            os.makedirs(f"{self.output_folder}/word_segmentation", exist_ok=True)
            cv2.imwrite(output_path, binary)

        # --- STEP 3: Morphological closing to bridge gaps in handwriting ---
        # Goal: Connect broken strokes within words without merging words
        # Kernel is horizontally-biased (wide but short) to bridge intra-word gaps
        # while avoiding inter-word connections
        morph_width = max(3, int(avg_char_width_approx * 0.40))  # 40% of char width
        morph_height = max(
            2, int(avg_char_width_approx * 0.1)
        )  # Minimal height to avoid vertical merging

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_width, morph_height))
        closed_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Validate closed binary image
        if (
            closed_binary is None
            or not isinstance(closed_binary, np.ndarray)
            or closed_binary.size == 0
        ):
            return ({}, False)

        # Save intermediate closed binary image (if enabled)
        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f"{self.output_folder}/word_segmentation/{safe_image_name}_{safe_line_number}_{safe_shortened_line_text}_closed_binary.png"
            os.makedirs(f"{self.output_folder}/word_segmentation", exist_ok=True)
            cv2.imwrite(output_path, closed_binary)

        # --- STEP 4: Intelligent noise removal using Connected Component Analysis ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            closed_binary, 8, cv2.CV_32S
        )
        clean_binary = np.zeros_like(binary)

        # --- STEP 4.1: Early noise detection ---
        # If component count >> character count, image is too noisy for adaptive search
        # In this case, skip to fallback method which is more robust to noise
        force_fallback = False
        if approx_char_count > 0 and num_labels > (approx_char_count * 10):
            force_fallback = True
            clean_binary = binary  # Keep original binary for fallback

        if num_labels > 1:
            areas = stats[
                1:, cv2.CC_STAT_AREA
            ]  # Get all component areas, skip background (label 0)

            # Handle edge case of empty 'areas' array
            if len(areas) == 0:
                clean_binary = binary
                # print("Warning: No components found after binarization.")
                areas = np.array([0])  # Add a dummy value to prevent crashes

            # --- STEP 4.2: Calculate conservative threshold (safe for clean images) ---
            # Uses 1st percentile and estimated minimum letter area to protect small characters
            p1 = np.percentile(areas, 1)
            img_h, img_w = binary.shape
            estimated_char_height = img_h * 0.7
            estimated_min_letter_area = max(
                2, int(estimated_char_height * 0.2 * estimated_char_height * 0.15)
            )

            # Conservative threshold protects small letters on clean lines
            area_threshold = max(MIN_AREA_THRESHOLD, min(p1, estimated_min_letter_area))

            # --- STEP 4.3: Detect noise-to-text gap for adaptive threshold selection ---
            sorted_areas = np.sort(areas)
            has_clear_gap = False
            aggressive_threshold = -1
            area_before_gap = -1

            if len(sorted_areas) > 10:  # Need enough components for gap analysis
                area_diffs = np.diff(sorted_areas)
                if len(area_diffs) > 0:
                    # Find significant jumps in area distribution (3x the 95th percentile)
                    jump_threshold = np.percentile(area_diffs, 95)
                    significant_jump_thresh = max(10, jump_threshold * 3)

                    jump_indices = np.where(area_diffs > significant_jump_thresh)[0]

                    if len(jump_indices) > 0:
                        has_clear_gap = True
                        gap_idx = jump_indices[0]  # Index of last noise component
                        area_before_gap = sorted_areas[gap_idx]
                        aggressive_threshold = area_before_gap + 1

            # --- STEP 4.4: Adaptive threshold selection ---
            if has_clear_gap:
                # Only use aggressive threshold if conservative threshold is deep in noise cluster
                # (threshold < 80% of noise cluster end)
                if area_threshold < (area_before_gap * 0.8):
                    # Use minimal increment above noise cluster to preserve small legitimate components
                    small_increment = 2
                    moderate_threshold = area_before_gap + small_increment

                    # Adjust based on gap size between noise and text
                    if gap_idx + 1 < len(sorted_areas):
                        first_after_gap = sorted_areas[gap_idx + 1]
                        gap_size = first_after_gap - area_before_gap

                        if gap_size > 50:
                            # Large gap: safe to use noise_end + 2
                            final_threshold = moderate_threshold
                        else:
                            # Small gap: be more conservative (noise_end + 1)
                            final_threshold = area_before_gap + 1
                    else:
                        final_threshold = moderate_threshold

                    # Apply bounds
                    final_threshold = max(final_threshold, area_before_gap + 1)
                    final_threshold = min(final_threshold, aggressive_threshold, 15)
                    area_threshold = final_threshold

            # --- STEP 4.5: Apply final threshold to filter components ---
            for i in range(1, num_labels):
                # Use >= to be inclusive of the threshold itself
                if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
                    clean_binary[labels == i] = 255
        else:
            # No components found, use original binary
            clean_binary = binary

        # Validate clean_binary before proceeding
        if (
            clean_binary is None
            or not isinstance(clean_binary, np.ndarray)
            or clean_binary.size == 0
        ):
            # print(
            #     f"Error: clean_binary image is None or empty (image_name: {image_name})"
            # )
            return ({}, False)

        # --- STEP 5: Vertical cropping to isolate text region ---
        # Calculate horizontal projection to find text boundaries
        horizontal_projection = np.sum(clean_binary, axis=1)
        y_start = 0  # Track offset for coordinate system conversion

        # Find text boundaries using percentiles to ignore outlier noise
        non_zero_rows = np.where(horizontal_projection > 0)[0]

        if len(non_zero_rows) > 0:
            # Use 5th and 95th percentiles to ignore top/bottom noise
            p_top = int(np.percentile(non_zero_rows, 5))
            p_bottom = int(np.percentile(non_zero_rows, 95))
            core_height = p_bottom - p_top

            # Trim 15% from each end (keep middle 70%)
            trim_pixels = int(core_height * 0.15)
            y_start = max(0, p_top + trim_pixels)
            y_end = min(clean_binary.shape[0], p_bottom - trim_pixels)

            # Fallback if trimmed region is too small
            if y_end - y_start < 5:
                y_start = p_top
                y_end = p_bottom

            analysis_image = clean_binary[y_start:y_end, :]
        else:
            # No text found, use full image
            analysis_image = clean_binary

        # Validate analysis image
        if (
            analysis_image is None
            or not isinstance(analysis_image, np.ndarray)
            or analysis_image.size == 0
        ):
            return ({}, False)

        # Save cropped analysis image (if enabled)
        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f"{self.output_folder}/word_segmentation/{safe_image_name}_{safe_line_number}_{safe_shortened_line_text}_clean_binary.png"
            os.makedirs(f"{self.output_folder}/word_segmentation", exist_ok=True)
            cv2.imwrite(output_path, analysis_image)

        # --- STEP 6: Hierarchical adaptive search for word boundaries ---
        words = line_data["text"][0].split()
        len(words)

        best_boxes = None
        successful_binary_image = None

        if not force_fallback:
            words = line_data["text"][0].split()
            # print(f"Words: {words}")
            target = len(words)

            # --- STAGE 1: Valley threshold search ---
            # Search for perfect match by varying valley threshold factor
            backup_boxes_s1 = None

            for v_factor in np.arange(INITIAL_VALLEY_THRESHOLD_FACTOR, 0.45, 0.02):
                curr_boxes = self._get_boxes_from_profile(
                    analysis_image, avg_char_width_approx, min_space_factor, v_factor
                )
                diff = abs(target - len(curr_boxes))

                if diff == 0 and self._is_geometry_valid(
                    curr_boxes, words, estimated_char_height
                ):
                    # print(f"Stage 1: Found perfect match with valley threshold factor {v_factor} and passed geometry validation")
                    best_boxes = curr_boxes
                    successful_binary_image = analysis_image
                    break

                # Store first close match (diff=1) as backup
                if (
                    diff == 1
                    and backup_boxes_s1 is None
                    and self._is_geometry_valid(
                        curr_boxes, words, estimated_char_height
                    )
                ):
                    # f"Stage 1: Found close match with valley threshold factor {v_factor} and passed geometry validation")
                    backup_boxes_s1 = curr_boxes

            # --- STAGE 2: Morphological closing search (if Stage 1 failed) ---
            if best_boxes is None:
                backup_boxes_s2 = None

                # Try varying kernel widths for morphological closing
                for k_factor in np.arange(INITIAL_KERNEL_WIDTH_FACTOR, 0.5, 0.02):
                    k_w = max(1, int(avg_char_width_approx * k_factor))
                    s2_bin = cv2.morphologyEx(
                        clean_binary, cv2.MORPH_CLOSE, np.ones((1, k_w), np.uint8)
                    )

                    # Apply same vertical cropping as analysis_image
                    s2_img = (
                        s2_bin[y_start:y_end, :] if len(non_zero_rows) > 0 else s2_bin
                    )
                    if s2_img is None or s2_img.size == 0:
                        continue

                    curr_boxes = self._get_boxes_from_profile(
                        s2_img,
                        avg_char_width_approx,
                        min_space_factor,
                        MAIN_VALLEY_THRESHOLD_FACTOR,
                    )
                    diff = abs(target - len(curr_boxes))

                    if diff == 0 and self._is_geometry_valid(curr_boxes, words):
                        best_boxes = curr_boxes
                        successful_binary_image = s2_bin
                        break

                    # Store first close match as backup
                    if (
                        diff == 1
                        and backup_boxes_s2 is None
                        and self._is_geometry_valid(curr_boxes, words)
                    ):
                        backup_boxes_s2 = curr_boxes

                # Use backups if exact match not found
                if best_boxes is None:
                    if backup_boxes_s1 is not None:
                        best_boxes = backup_boxes_s1
                        successful_binary_image = analysis_image
                    elif backup_boxes_s2 is not None:
                        best_boxes = backup_boxes_s2
                        successful_binary_image = clean_binary

        final_output = None
        used_fallback = False

        if best_boxes is None:
            # --- FALLBACK: Use hybrid segmenter if adaptive search fails ---
            fallback_segmenter = HybridWordSegmenter()
            used_fallback = True
            final_output = fallback_segmenter.refine_words_bidirectional(
                line_data, deskewed_line_image
            )

        else:
            # --- STEP 7: CCA-based refinement for pixel-perfect word boundaries ---
            unlabeled_boxes = best_boxes

            # Determine which binary image to use for CCA
            # Stage 1 success: use clean_binary (full image, not cropped)
            # Stage 2 success: use the morphologically closed binary
            if successful_binary_image is analysis_image:
                # Stage 1 succeeded - use full clean_binary for CCA
                cca_source_image = clean_binary
            else:
                # Stage 2 succeeded - use the morphologically closed binary
                cca_source_image = successful_binary_image
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                cca_source_image, 8, cv2.CV_32S
            )

            num_to_process = min(len(words), len(unlabeled_boxes))

            # Get image dimensions for filtering components outside text region
            cca_img_h, cca_img_w = cca_source_image.shape[:2]

            # --- STEP 7.1: Assign components to word boxes ---
            # Use component center point as primary criterion, overlap as tie-breaker
            # This prevents components from being assigned to multiple boxes
            component_assignments = {}
            num_proc = min(len(words), len(unlabeled_boxes))

            # Calculate a minimum valid area to ignore specks (like the one over 'Beach')
            min_valid_component_area = estimated_char_height * 2

            for j in range(1, num_labels):  # Skip background
                comp_x = stats[j, cv2.CC_STAT_LEFT]
                comp_w = stats[j, cv2.CC_STAT_WIDTH]
                comp_area = stats[j, cv2.CC_STAT_AREA]
                comp_r = comp_x + comp_w
                comp_center_x = comp_x + comp_w / 2  # Component center x coordinate
                comp_y = stats[j, cv2.CC_STAT_TOP]
                comp_h = stats[j, cv2.CC_STAT_HEIGHT]

                # Filter out components that are clearly outside the text region
                # Components should be roughly in the middle 80% of the image height
                # (accounting for vertical cropping that was done)
                comp_center_y = comp_y + comp_h / 2
                # Filter out components outside text region (top/bottom 10%)
                if comp_center_y < cca_img_h * 0.1 or comp_center_y > cca_img_h * 0.9:
                    continue

                # Ignore tiny noise components
                if comp_area < min_valid_component_area:
                    continue

                best_box_idx = None
                max_overlap = 0
                best_center_distance = float("inf")
                component_center_in_box = False

                for i in range(num_to_process):
                    box_x, box_y, box_w, box_h = unlabeled_boxes[i]
                    box_r = box_x + box_w
                    box_center_x = box_x + box_w / 2

                    # Skip components that are too wide (likely merged blobs spanning multiple words)
                    if comp_w > box_w * 1.5:
                        continue

                    # Check horizontal overlap with this box
                    # Note: unlabeled_boxes use analysis_image coordinates (x matches, y=0)
                    # cca_source_image is full image, so only check x overlap
                    if comp_x < box_r and box_x < comp_r:
                        # Calculate horizontal overlap
                        overlap_start = max(comp_x, box_x)
                        overlap_end = min(comp_r, box_r)
                        overlap = overlap_end - overlap_start

                        if overlap > 0:
                            # Check if component center is within box boundaries
                            center_in_box = box_x <= comp_center_x < box_r
                            center_distance = abs(comp_center_x - box_center_x)

                            # Priority 1: Component center within box (best match)
                            # Priority 2: Closest center distance if no box contains center
                            if center_in_box:
                                if not component_center_in_box or overlap > max_overlap:
                                    component_center_in_box = True
                                    best_center_distance = center_distance
                                    max_overlap = overlap
                                    best_box_idx = i
                            elif not component_center_in_box:
                                if center_distance < best_center_distance or (
                                    center_distance == best_center_distance
                                    and overlap > max_overlap
                                ):
                                    best_center_distance = center_distance
                                    max_overlap = overlap
                                    best_box_idx = i

                if best_box_idx is not None:
                    component_assignments[j] = best_box_idx

            # --- STEP 7.2: Build refined boxes from assigned components ---
            refined_boxes_list = []
            for i in range(num_proc):
                word_label = words[i]
                components_in_box = [
                    stats[j] for j, b in component_assignments.items() if b == i
                ]

                # ----------------------------------------------------------
                # SMART CCA FALLBACK: If CCA shrinks box to noise, revert!
                # ----------------------------------------------------------

                use_original_box = False

                if not components_in_box:
                    use_original_box = True
                else:
                    # Calculate potential CCA box
                    min_x = min(c[cv2.CC_STAT_LEFT] for c in components_in_box)
                    min_y = min(c[cv2.CC_STAT_TOP] for c in components_in_box)
                    max_r = max(
                        c[cv2.CC_STAT_LEFT] + c[cv2.CC_STAT_WIDTH]
                        for c in components_in_box
                    )
                    max_b = max(
                        c[cv2.CC_STAT_TOP] + c[cv2.CC_STAT_HEIGHT]
                        for c in components_in_box
                    )
                    cca_h = max(1, max_b - min_y)

                    # CHECK: Is this box unreasonably small compared to expected font size?
                    # If the CCA result is < 35% of line height, it's likely a noise speck.
                    # But the Profile Box (Stage 1) was valid. So we trust the Profile Box.
                    if cca_h < (estimated_char_height * 0.35):
                        # print(f"Reverting '{word_label}' to Profile Box (CCA height {cca_h} vs Exp {estimated_char_height:.1f})")
                        use_original_box = True

                if use_original_box:
                    # Revert to Stage 1 Profile Box
                    box_x, box_y, box_w, box_h = unlabeled_boxes[i]
                    adjusted_box_y = y_start + box_y
                    refined_boxes_list.append(
                        {
                            "text": word_label,
                            "left": box_x,
                            "top": adjusted_box_y,
                            "width": box_w,
                            "height": box_h,
                            "conf": line_data["conf"][0],
                        }
                    )
                else:
                    # Use CCA Box
                    refined_boxes_list.append(
                        {
                            "text": word_label,
                            "left": min_x,
                            "top": min_y,
                            "width": max(1, max_r - min_x),
                            "height": cca_h,
                            "conf": line_data["conf"][0],
                        }
                    )

            # --- LATE VALIDATION ---
            # Now that we've attempted repairs, check the whole set one last time.
            cca_check_list = [
                (b["left"], b["top"], b["width"], b["height"])
                for b in refined_boxes_list
            ]

            if not self._is_geometry_valid(
                cca_check_list, words, estimated_char_height
            ):
                # print("Late Validation Failed even after repair. Falling back.")
                best_boxes = None  # Trigger fallback below
            else:
                final_output = {
                    k: [] for k in ["text", "left", "top", "width", "height", "conf"]
                }
                for box in refined_boxes_list:
                    for key in final_output.keys():
                        final_output[key].append(box[key])

        # --- FALLBACK CATCH (Repeated if Late Validation sets best_boxes=None) ---
        if best_boxes is None and not used_fallback:
            fallback_segmenter = HybridWordSegmenter()
            used_fallback = True
            final_output = fallback_segmenter.refine_words_bidirectional(
                line_data, deskewed_line_image
            )
        # else:
        #    print(f"Late Validation passed. Using CCA boxes.")

        # ========================================================================
        # COORDINATE TRANSFORMATION BACK TO ORIGINAL IMAGE
        # ========================================================================

        # --- STEP 8: Transform coordinates from deskewed space to original space ---
        # Get inverse transformation matrix
        M_inv = cv2.invertAffineTransform(M)

        # Transform each box by mapping its 4 corners back to original space
        remapped_boxes_list = []

        for i in range(len(final_output["text"])):
            left, top = final_output["left"][i], final_output["top"][i]
            width, height = final_output["width"][i], final_output["height"][i]

            # Define 4 corners of box in deskewed space
            corners = np.array(
                [
                    [left, top],
                    [left + width, top],
                    [left + width, top + height],
                    [left, top + height],
                ],
                dtype="float32",
            )

            # Expand dimensions for cv2.transform (requires shape (N, 1, 2))
            corners_expanded = np.expand_dims(corners, axis=1)

            # Apply inverse transformation
            original_corners = cv2.transform(corners_expanded, M_inv)

            # Squeeze to get shape (N, 2)
            squeezed_corners = original_corners.squeeze(axis=1)

            # Calculate axis-aligned bounding box in original space
            min_x = int(np.min(squeezed_corners[:, 0]))
            max_x = int(np.max(squeezed_corners[:, 0]))
            min_y = int(np.min(squeezed_corners[:, 1]))
            max_y = int(np.max(squeezed_corners[:, 1]))

            # Create remapped box
            remapped_box = {
                "text": final_output["text"][i],
                "left": min_x,
                "top": min_y,
                "width": max_x - min_x,
                "height": max_y - min_y,
                "conf": final_output["conf"][i],
            }
            remapped_boxes_list.append(remapped_box)

        # Convert back to dictionary format
        remapped_output = {k: [] for k in final_output.keys()}
        for box in remapped_boxes_list:
            for key in remapped_output.keys():
                remapped_output[key].append(box[key])

        # --- STEP 9: Apply final logical constraint checks ---
        img_h, img_w = line_image.shape[:2]
        remapped_output = self._enforce_logical_constraints(
            remapped_output, img_w, img_h
        )

        # --- STEP 10: Save visualization (if enabled) ---
        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            output_path = f"{self.output_folder}/word_segmentation/{safe_image_name}_{safe_shortened_line_text}_final_boxes.png"
            os.makedirs(f"{self.output_folder}/word_segmentation", exist_ok=True)
            output_image_vis = line_image.copy()

            if (
                output_image_vis is not None
                and isinstance(output_image_vis, np.ndarray)
                and output_image_vis.size > 0
            ):
                # Draw bounding boxes on original image
                for i in range(len(remapped_output["text"])):
                    remapped_output["text"][i]
                    x, y, w, h = (
                        int(remapped_output["left"][i]),
                        int(remapped_output["top"][i]),
                        int(remapped_output["width"][i]),
                        int(remapped_output["height"][i]),
                    )
                    cv2.rectangle(
                        output_image_vis, (x, y), (x + w, y + h), (0, 255, 0), 2
                    )
                cv2.imwrite(output_path, output_image_vis)

        return remapped_output, used_fallback


class HybridWordSegmenter:
    """
    Implements a two-step approach for word segmentation:
    1. Proportional estimation based on text.
    2. Image-based refinement with a "Bounded Scan" to prevent
       over-correction.
    """

    def _convert_line_to_word_level_improved(
        self, line_data: Dict[str, List], image_width: int, image_height: int
    ) -> Dict[str, List]:
        """
        Step 1: Converts line-level OCR results to word-level by using a
        robust proportional estimation method.
        (This function is unchanged from the previous version)
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

        i = 0  # Assuming a single line
        line_text = line_data["text"][i]
        line_left = float(line_data["left"][i])
        line_top = float(line_data["top"][i])
        line_width = float(line_data["width"][i])
        line_height = float(line_data["height"][i])
        line_conf = line_data["conf"][i]

        if not line_text.strip():
            return output
        words = line_text.split()
        if not words:
            return output
        num_chars = len("".join(words))
        num_spaces = len(words) - 1
        if num_chars == 0:
            return output

        if (num_chars * 2 + num_spaces) > 0:
            char_space_ratio = 2.0
            estimated_space_width = line_width / (
                num_chars * char_space_ratio + num_spaces
            )
            avg_char_width = estimated_space_width * char_space_ratio
        else:
            avg_char_width = line_width / (num_chars if num_chars > 0 else 1)
            estimated_space_width = avg_char_width

        current_left = line_left
        for word in words:
            word_width = len(word) * avg_char_width
            clamped_left = max(0, min(current_left, image_width))
            clamped_width = max(0, min(word_width, image_width - clamped_left))
            output["text"].append(word)
            output["left"].append(clamped_left)
            output["top"].append(line_top)
            output["width"].append(clamped_width)
            output["height"].append(line_height)
            output["conf"].append(line_conf)
            current_left += word_width + estimated_space_width
        return output

    def _run_single_pass(
        self,
        initial_boxes: List[Dict],
        vertical_projection: np.ndarray,
        max_scan_distance: int,
        img_w: int,
        direction: str = "ltr",
    ) -> List[Dict]:
        """
        Helper function to run one pass of refinement.
        IMPROVED: Uses local minima detection for cursive script where
        perfect zero-gaps (white space) might not exist.
        """

        refined_boxes = [box.copy() for box in initial_boxes]

        if direction == "ltr":
            last_corrected_right_edge = 0
            indices = range(len(refined_boxes))
        else:  # rtl
            next_corrected_left_edge = img_w
            indices = range(len(refined_boxes) - 1, -1, -1)

        for i in indices:
            box = refined_boxes[i]
            left = int(box["left"])
            right = int(box["left"] + box["width"])

            left = max(0, min(left, img_w - 1))
            right = max(0, min(right, img_w - 1))

            new_left, new_right = left, right

            # --- Boundary search with improved gap detection ---
            # Priority 1: True gap (zero projection)
            # Priority 2: Valley with lowest ink density (thinnest connection)

            if direction == "ltr" or direction == "both":  # Scan right logic
                if right < img_w:
                    scan_limit = min(img_w, right + max_scan_distance)
                    search_range = range(right, scan_limit)

                    best_x = right
                    min_density = float("inf")
                    found_zero = False

                    # Look for the best cut in the window
                    for x in search_range:
                        density = vertical_projection[x]
                        if density == 0:
                            new_right = x
                            found_zero = True
                            break
                        if density < min_density:
                            min_density = density
                            best_x = x

                    if not found_zero:
                        # No clear gap found, cut at thinnest point (minimum density)
                        new_right = best_x

            if direction == "rtl" or direction == "both":  # Scan left logic
                if left > 0:
                    scan_limit = max(0, left - max_scan_distance)
                    search_range = range(left, scan_limit, -1)

                    best_x = left
                    min_density = float("inf")
                    found_zero = False

                    for x in search_range:
                        density = vertical_projection[x]
                        if density == 0:
                            new_left = x
                            found_zero = True
                            break
                        if density < min_density:
                            min_density = density
                            best_x = x

                    if not found_zero:
                        new_left = best_x

            # --- Directional de-overlapping (strict stitching) ---
            if direction == "ltr":
                if new_left < last_corrected_right_edge:
                    new_left = last_corrected_right_edge
                # Ensure valid width
                if new_right <= new_left:
                    new_right = new_left + 1
                last_corrected_right_edge = new_right
            else:  # rtl
                if new_right > next_corrected_left_edge:
                    new_right = next_corrected_left_edge
                # Ensure valid width
                if new_left >= new_right:
                    new_left = new_right - 1
                next_corrected_left_edge = new_left

            box["left"] = new_left
            box["width"] = max(1, new_right - new_left)

        return refined_boxes

    def refine_words_bidirectional(
        self,
        line_data: Dict[str, List],
        line_image: np.ndarray,
    ) -> Dict[str, List]:
        """
        Refines boxes using a more robust bidirectional scan and averaging.
        """
        if line_image is None:
            return line_data

        # Early return if 1 or fewer words
        if line_data and line_data.get("text"):
            words = line_data["text"][0].split()
            if len(words) <= 1:
                img_h, img_w = line_image.shape[:2]
                return self._convert_line_to_word_level_improved(
                    line_data, img_w, img_h
                )

        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img_h, img_w = binary.shape
        vertical_projection = np.sum(binary, axis=0)

        char_blobs = []
        in_blob = False
        blob_start = 0
        for x, col_sum in enumerate(vertical_projection):
            if col_sum > 0 and not in_blob:
                blob_start = x
                in_blob = True
            elif col_sum == 0 and in_blob:
                char_blobs.append((blob_start, x))
                in_blob = False
        if in_blob:
            char_blobs.append((blob_start, img_w))

        if not char_blobs:
            return self._convert_line_to_word_level_improved(line_data, img_w, img_h)

        avg_char_width = np.mean([end - start for start, end in char_blobs])
        max_scan_distance = int(avg_char_width * 1.5)

        estimated_data = self._convert_line_to_word_level_improved(
            line_data, img_w, img_h
        )
        if not estimated_data["text"]:
            return estimated_data

        initial_boxes = []
        for i in range(len(estimated_data["text"])):
            initial_boxes.append(
                {
                    "text": estimated_data["text"][i],
                    "left": estimated_data["left"][i],
                    "top": estimated_data["top"][i],
                    "width": estimated_data["width"][i],
                    "height": estimated_data["height"][i],
                    "conf": estimated_data["conf"][i],
                }
            )

        # --- STEP 1 & 2: Perform bidirectional refinement passes ---
        ltr_boxes = self._run_single_pass(
            initial_boxes, vertical_projection, max_scan_distance, img_w, "ltr"
        )
        rtl_boxes = self._run_single_pass(
            initial_boxes, vertical_projection, max_scan_distance, img_w, "rtl"
        )

        # --- STEP 3: Combine results using best edge from each pass ---
        # LTR pass provides accurate left boundaries, RTL pass provides accurate right boundaries
        combined_boxes = [box.copy() for box in initial_boxes]
        for i in range(len(combined_boxes)):
            final_left = ltr_boxes[i]["left"]  # Best left edge from LTR scan
            rtl_right = (
                rtl_boxes[i]["left"] + rtl_boxes[i]["width"]
            )  # Best right edge from RTL scan

            combined_boxes[i]["left"] = final_left
            combined_boxes[i]["width"] = max(1, rtl_right - final_left)

        # --- STEP 4: Contiguous stitching to eliminate gaps and overlaps ---
        # Goal: Ensure box[i].right == box[i+1].left (no gaps, no overlaps)

        # First, ensure strictly increasing left edges (protect against merge artifacts)
        for i in range(len(combined_boxes) - 1):
            if combined_boxes[i + 1]["left"] <= combined_boxes[i]["left"]:
                combined_boxes[i + 1]["left"] = combined_boxes[i]["left"] + 1

        # Stitch boxes together (assign gaps to current word)
        for i in range(len(combined_boxes) - 1):
            curr = combined_boxes[i]
            nxt = combined_boxes[i + 1]
            curr["width"] = max(1, nxt["left"] - curr["left"])

        # Convert back to Tesseract-style output dict
        final_output = {k: [] for k in estimated_data.keys()}
        for box in combined_boxes:
            if box["width"] > 0:  # Ensure we don't add zero-width boxes
                for key in final_output.keys():
                    final_output[key].append(box[key])

        return final_output
