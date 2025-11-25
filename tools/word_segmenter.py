import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from tools.config import OUTPUT_FOLDER, SAVE_WORD_SEGMENTER_OUTPUT_IMAGES

# Adaptive thresholding parameters
BLOCK_SIZE_FACTOR = 1.5  # Multiplier for adaptive threshold block size
C_VALUE = 2  # Constant subtracted from mean in adaptive thresholding

# Word segmentation search parameters
INITIAL_KERNEL_WIDTH_FACTOR = 0.0  # Starting kernel width factor for Stage 2 search
INITIAL_VALLEY_THRESHOLD_FACTOR = (
    0.0  # Starting valley threshold factor for Stage 1 search
)
MAIN_VALLEY_THRESHOLD_FACTOR = (
    0.15  # Primary valley threshold factor for word separation
)
MIN_SPACE_FACTOR = 0.2  # Minimum space width relative to character width
MATCH_TOLERANCE = 0  # Tolerance for word count matching

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
    2. Targeted Noise Removal using Connected Component Analysis.
    3. The robust two-stage adaptive search (Valley -> Kernel).
    4. CCA for final pixel-perfect refinement.
    """

    def __init__(self, output_folder: str = OUTPUT_FOLDER):
        self.output_folder = output_folder
        self.fallback_segmenter = HybridWordSegmenter()

    def _correct_orientation(
        self, gray_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects and corrects 90-degree orientation issues.
        """
        h, w = gray_image.shape
        center = (w // 2, h // 2)

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

        opening_kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, opening_kernel)

        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 50:
            M_orient = cv2.getRotationMatrix2D(center, 0, 1.0)
            return gray_image, M_orient

        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        box_height = ymax - ymin
        box_width = xmax - xmin

        orientation_angle = 0.0
        if box_height > box_width:
            orientation_angle = 90.0
        else:
            M_orient = cv2.getRotationMatrix2D(center, 0, 1.0)
            return gray_image, M_orient

        M_orient = cv2.getRotationMatrix2D(center, orientation_angle, 1.0)
        new_w, new_h = h, w
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
        Detects skew using a robust method that normalizes minAreaRect.
        """
        h, w = gray_image.shape

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

        opening_kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, opening_kernel)

        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 50:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, 1.0)
            return gray_image, M

        rect = cv2.minAreaRect(coords[:, ::-1])
        rect_width, rect_height = rect[1]
        angle = rect[2]

        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
            angle += 90

        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

        correction_angle = angle

        if abs(correction_angle) < MIN_SKEW_THRESHOLD:
            correction_angle = 0.0
        elif abs(correction_angle) > MAX_SKEW_THRESHOLD:
            correction_angle = 0.0

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
                # [NOTE] Returns full height stripe
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

        is_vertical = image_height > (image_width * 1.2)
        if is_vertical:
            boxes.sort(key=lambda b: (b["top"], b["left"]))
        else:
            boxes.sort(key=lambda b: (b["left"], -b["width"]))

        final_pass_boxes = []
        if boxes:
            keep_indices = [True] * len(boxes)
            for i in range(len(boxes)):
                for j in range(len(boxes)):
                    if i == j:
                        continue
                    b1 = boxes[i]
                    b2 = boxes[j]

                    x_nested = (b1["left"] >= b2["left"] - 2) and (
                        b1["left"] + b1["width"] <= b2["left"] + b2["width"] + 2
                    )
                    y_nested = (b1["top"] >= b2["top"] - 2) and (
                        b1["top"] + b1["height"] <= b2["top"] + b2["height"] + 2
                    )

                    if x_nested and y_nested:
                        if b1["text"] == b2["text"]:
                            if b1["width"] * b1["height"] <= b2["width"] * b2["height"]:
                                keep_indices[i] = False

            for i, keep in enumerate(keep_indices):
                if keep:
                    final_pass_boxes.append(boxes[i])

        boxes = final_pass_boxes

        if is_vertical:
            boxes.sort(key=lambda b: (b["top"], b["left"]))
        else:
            boxes.sort(key=lambda b: (b["left"], -b["width"]))

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                b1 = boxes[i]
                b2 = boxes[j]

                x_overlap = min(
                    b1["left"] + b1["width"], b2["left"] + b2["width"]
                ) - max(b1["left"], b2["left"])
                y_overlap = min(
                    b1["top"] + b1["height"], b2["top"] + b2["height"]
                ) - max(b1["top"], b2["top"])

                if x_overlap > 0 and y_overlap > 0:
                    if is_vertical:
                        if b1["top"] < b2["top"]:
                            new_h = max(1, b2["top"] - b1["top"])
                            b1["height"] = new_h
                    else:
                        if b1["left"] < b2["left"]:
                            b1_right = b1["left"] + b1["width"]
                            b2_right = b2["left"] + b2["width"]
                            left_slice_width = max(0, b2["left"] - b1["left"])
                            right_slice_width = max(0, b1_right - b2_right)

                            if (
                                b1_right > b2_right
                                and right_slice_width > left_slice_width
                            ):
                                b1["left"] = b2_right
                                b1["width"] = right_slice_width
                            else:
                                b1["width"] = max(1, left_slice_width)

        cleaned_output = {
            k: [] for k in ["text", "left", "top", "width", "height", "conf"]
        }
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
        [FIX] Improved robustness for punctuation and mixed-case text.
        """
        if len(boxes) != len(words):
            return False

        baseline = expected_height
        # Use median only if provided expected height is unreliable
        if baseline < 5:
            heights = [b[3] for b in boxes]
            if heights:
                baseline = np.median(heights)

        if baseline < 5:
            return True

        for i, box in enumerate(boxes):
            word = words[i]

            # [FIX] Check for punctuation/symbols. They are allowed to be small.
            # If word is just punctuation, skip geometry checks
            is_punctuation = not any(c.isalnum() for c in word)
            if is_punctuation:
                continue

            # Standard checks for alphanumeric words
            num_chars = len(word)
            if num_chars < 1:
                continue

            width = box[2]
            height = box[3]

            # [FIX] Only reject height if it's REALLY small compared to baseline
            # A period might be small, but we skipped that check above.
            # This check ensures a real word like "The" isn't 2 pixels tall.
            if height < (baseline * 0.20):
                return False

            avg_char_width = width / num_chars
            min_expected = baseline * 0.20

            # Only reject if it fails BOTH absolute (4px) and relative checks
            if avg_char_width < min_expected and avg_char_width < 4:
                # Exception: If the word is 1 char long (e.g. "I", "l", "1"), allow it to be skinny.
                if num_chars == 1 and avg_char_width >= 2:
                    continue
                return False

        return True

    def segment(
        self,
        line_data: Dict[str, List],
        line_image: np.ndarray,
        min_space_factor=MIN_SPACE_FACTOR,
        match_tolerance=MATCH_TOLERANCE,
        image_name: str = None,
    ) -> Tuple[Dict[str, List], bool]:

        if (
            line_image is None
            or not isinstance(line_image, np.ndarray)
            or line_image.size == 0
        ):
            return ({}, False)
        # Allow grayscale (2 dims) or color (3 dims)
        if len(line_image.shape) < 2:
            return ({}, False)
        if not line_data or not line_data.get("text") or len(line_data["text"]) == 0:
            return ({}, False)

        line_text = line_data["text"][0]
        words = line_text.split()

        # Early return if 1 or fewer words
        if len(words) <= 1:
            img_h, img_w = line_image.shape[:2]
            one_word_result = self.fallback_segmenter.convert_line_to_word_level(
                line_data, img_w, img_h
            )
            return (one_word_result, False)

        line_number = line_data["line"][0]
        safe_image_name = _sanitize_filename(image_name or "image", max_length=50)
        safe_line_number = _sanitize_filename(str(line_number), max_length=10)
        safe_shortened_line_text = _sanitize_filename(line_text, max_length=10)

        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f"{self.output_folder}/word_segmentation/{safe_image_name}_{safe_line_number}_{safe_shortened_line_text}_original.png"
            os.makedirs(f"{self.output_folder}/word_segmentation", exist_ok=True)
            cv2.imwrite(output_path, line_image)

        if len(line_image.shape) == 3:
            gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_image.copy()

        # ========================================================================
        # IMAGE PREPROCESSING (Deskew / Rotate)
        # ========================================================================
        oriented_gray, M_orient = self._correct_orientation(gray)
        deskewed_gray, M_skew = self._deskew_image(oriented_gray)

        # Combine matrices: M_total = M_skew * M_orient
        M_orient_3x3 = np.vstack([M_orient, [0, 0, 1]])
        M_skew_3x3 = np.vstack([M_skew, [0, 0, 1]])
        M_total_3x3 = M_skew_3x3 @ M_orient_3x3
        M = M_total_3x3[0:2, :]  # Extract 2x3 affine matrix

        # Apply transformation to the original color image
        h, w = deskewed_gray.shape
        deskewed_line_image = cv2.warpAffine(
            line_image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # [FIX] Create Local Line Data that matches the deskewed/rotated image dimensions.
        # This prevents the fallback segmenter from using vertical dimensions on a horizontal image.
        local_line_data = {
            "text": line_data["text"],
            "conf": line_data["conf"],
            "left": [0],  # Local coordinate system starts at 0
            "top": [0],
            "width": [w],  # Use the ROTATED width
            "height": [h],  # Use the ROTATED height
            "line": line_data.get("line", [0]),
        }

        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f"{self.output_folder}/word_segmentation/{safe_image_name}_{safe_line_number}_{safe_shortened_line_text}_deskewed.png"
            cv2.imwrite(output_path, deskewed_line_image)

        # ========================================================================
        # MAIN SEGMENTATION PIPELINE
        # ========================================================================
        approx_char_count = len(line_data["text"][0].replace(" ", ""))
        if approx_char_count == 0:
            return {}, False

        img_h, img_w = deskewed_gray.shape
        estimated_char_height = img_h * 0.6
        avg_char_width_approx = img_w / approx_char_count

        block_size = int(avg_char_width_approx * BLOCK_SIZE_FACTOR)
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3

        # --- Binarization ---
        binary_adaptive = cv2.adaptiveThreshold(
            deskewed_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            C_VALUE,
        )
        otsu_thresh_val, _ = cv2.threshold(
            deskewed_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        strict_thresh_val = otsu_thresh_val * 0.75
        _, binary_strict = cv2.threshold(
            deskewed_gray, strict_thresh_val, 255, cv2.THRESH_BINARY_INV
        )
        binary = cv2.bitwise_and(binary_adaptive, binary_strict)

        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            output_path = f"{self.output_folder}/word_segmentation/{safe_image_name}_{safe_line_number}_{safe_shortened_line_text}_binary.png"
            cv2.imwrite(output_path, binary)

        # --- Morphological Closing ---
        morph_width = max(3, int(avg_char_width_approx * 0.40))
        morph_height = max(2, int(avg_char_width_approx * 0.1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_width, morph_height))
        closed_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # --- Noise Removal ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            closed_binary, 8, cv2.CV_32S
        )
        clean_binary = np.zeros_like(binary)

        force_fallback = False
        significant_labels = 0
        if num_labels > 1:
            # Only count components with area > 3 pixels
            significant_labels = np.sum(stats[1:, cv2.CC_STAT_AREA] > 3)

        if approx_char_count > 0 and significant_labels > (approx_char_count * 12):
            force_fallback = True

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) == 0:
                clean_binary = binary
                areas = np.array([0])
            else:
                p1 = np.percentile(areas, 1)
                img_h, img_w = binary.shape
                estimated_char_height = img_h * 0.7
                estimated_min_letter_area = max(
                    2, int(estimated_char_height * 0.2 * estimated_char_height * 0.15)
                )
                area_threshold = max(
                    MIN_AREA_THRESHOLD, min(p1, estimated_min_letter_area)
                )

                # Gap detection logic...
                sorted_areas = np.sort(areas)
                area_diffs = np.diff(sorted_areas)
                if len(sorted_areas) > 10 and len(area_diffs) > 0:
                    jump_threshold = np.percentile(area_diffs, 95)
                    significant_jump_thresh = max(10, jump_threshold * 3)
                    jump_indices = np.where(area_diffs > significant_jump_thresh)[0]
                    if len(jump_indices) > 0:
                        gap_idx = jump_indices[0]
                        area_before_gap = sorted_areas[gap_idx]
                        final_threshold = max(area_before_gap + 1, area_threshold)
                        final_threshold = min(final_threshold, 15)
                        area_threshold = final_threshold

                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
                        clean_binary[labels == i] = 255
        else:
            clean_binary = binary

        # --- Vertical Cropping ---
        horizontal_projection = np.sum(clean_binary, axis=1)
        y_start = 0
        non_zero_rows = np.where(horizontal_projection > 0)[0]
        if len(non_zero_rows) > 0:
            p_top = int(np.percentile(non_zero_rows, 5))
            p_bottom = int(np.percentile(non_zero_rows, 95))
            core_height = p_bottom - p_top
            trim_pixels = int(core_height * 0.1)
            y_start = max(0, p_top + trim_pixels)
            y_end = min(clean_binary.shape[0], p_bottom - trim_pixels)
            if y_end - y_start < 5:
                y_start = p_top
                y_end = p_bottom
            analysis_image = clean_binary[y_start:y_end, :]
        else:
            analysis_image = clean_binary

        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            output_path = f"{self.output_folder}/word_segmentation/{safe_image_name}_{safe_line_number}_{safe_shortened_line_text}_clean_binary.png"
            cv2.imwrite(output_path, analysis_image)

        # --- Adaptive Search ---
        best_boxes = None
        successful_binary_image = None

        if not force_fallback:
            words = line_data["text"][0].split()
            target = len(words)
            backup_boxes_s1 = None

            # STAGE 1
            for v_factor in np.arange(INITIAL_VALLEY_THRESHOLD_FACTOR, 0.60, 0.02):
                curr_boxes = self._get_boxes_from_profile(
                    analysis_image, avg_char_width_approx, min_space_factor, v_factor
                )
                diff = abs(target - len(curr_boxes))
                is_geom_valid = self._is_geometry_valid(
                    curr_boxes, words, estimated_char_height
                )

                if diff == 0:
                    if is_geom_valid:
                        best_boxes = curr_boxes
                        successful_binary_image = analysis_image
                        break
                    else:
                        if backup_boxes_s1 is None:
                            backup_boxes_s1 = curr_boxes
                if diff == 1 and backup_boxes_s1 is None and is_geom_valid:
                    backup_boxes_s1 = curr_boxes

            # STAGE 2 (if needed)
            if best_boxes is None:
                backup_boxes_s2 = None
                for k_factor in np.arange(INITIAL_KERNEL_WIDTH_FACTOR, 0.5, 0.02):
                    k_w = max(1, int(avg_char_width_approx * k_factor))
                    s2_bin = cv2.morphologyEx(
                        clean_binary, cv2.MORPH_CLOSE, np.ones((1, k_w), np.uint8)
                    )
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
                    is_geom_valid = self._is_geometry_valid(
                        curr_boxes, words, estimated_char_height
                    )

                    if diff == 0 and is_geom_valid:
                        best_boxes = curr_boxes
                        successful_binary_image = s2_bin
                        break

                    if diff == 1 and backup_boxes_s2 is None and is_geom_valid:
                        backup_boxes_s2 = curr_boxes

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
            # --- FALLBACK WITH ROTATED DATA ---
            used_fallback = True
            # [FIX] Use local_line_data (rotated dims) instead of line_data (original dims)
            final_output = self.fallback_segmenter.refine_words_bidirectional(
                local_line_data, deskewed_line_image
            )
        else:
            # --- CCA Refinement ---
            unlabeled_boxes = best_boxes
            if successful_binary_image is analysis_image:
                cca_source_image = clean_binary
            else:
                cca_source_image = successful_binary_image

            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                cca_source_image, 8, cv2.CV_32S
            )
            cca_img_h, cca_img_w = cca_source_image.shape[:2]

            component_assignments = {}
            num_proc = min(len(words), len(unlabeled_boxes))
            min_valid_component_area = estimated_char_height * 2

            for j in range(1, num_labels):
                comp_x = stats[j, cv2.CC_STAT_LEFT]
                comp_w = stats[j, cv2.CC_STAT_WIDTH]
                comp_area = stats[j, cv2.CC_STAT_AREA]
                comp_r = comp_x + comp_w
                comp_center_x = comp_x + comp_w / 2
                comp_y = stats[j, cv2.CC_STAT_TOP]
                comp_h = stats[j, cv2.CC_STAT_HEIGHT]
                comp_center_y = comp_y + comp_h / 2

                if comp_center_y < cca_img_h * 0.1 or comp_center_y > cca_img_h * 0.9:
                    continue
                if comp_area < min_valid_component_area:
                    continue

                best_box_idx = None
                max_overlap = 0
                best_center_distance = float("inf")
                component_center_in_box = False

                num_to_process = min(len(words), len(unlabeled_boxes))

                # Assign components to boxes...
                for i in range(
                    num_to_process
                ):  # Note: ensure num_to_process is defined
                    box_x, box_y, box_w, box_h = unlabeled_boxes[i]
                    box_r = box_x + box_w
                    box_center_x = box_x + box_w / 2

                    if comp_w > box_w * 1.5:
                        continue

                    if comp_x < box_r and box_x < comp_r:
                        overlap_start = max(comp_x, box_x)
                        overlap_end = min(comp_r, box_r)
                        overlap = overlap_end - overlap_start

                        if overlap > 0:
                            center_in_box = box_x <= comp_center_x < box_r
                            center_distance = abs(comp_center_x - box_center_x)

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

            refined_boxes_list = []
            for i in range(num_proc):
                word_label = words[i]
                components_in_box = [
                    stats[j] for j, b in component_assignments.items() if b == i
                ]

                use_original_box = False
                if not components_in_box:
                    use_original_box = True
                else:
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
                    if cca_h < (estimated_char_height * 0.35):
                        use_original_box = True

                if use_original_box:
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

            # Check validity
            cca_check_list = [
                (b["left"], b["top"], b["width"], b["height"])
                for b in refined_boxes_list
            ]
            if not self._is_geometry_valid(
                cca_check_list, words, estimated_char_height
            ):
                if abs(len(refined_boxes_list) - len(words)) > 1:
                    best_boxes = None  # Trigger fallback
                else:
                    final_output = {
                        k: []
                        for k in ["text", "left", "top", "width", "height", "conf"]
                    }
                    for box in refined_boxes_list:
                        for key in final_output.keys():
                            final_output[key].append(box[key])
            else:
                final_output = {
                    k: [] for k in ["text", "left", "top", "width", "height", "conf"]
                }
                for box in refined_boxes_list:
                    for key in final_output.keys():
                        final_output[key].append(box[key])

        # --- REPEAT FALLBACK IF VALIDATION FAILED ---
        if best_boxes is None and not used_fallback:
            used_fallback = True
            # [FIX] Use local_line_data here too
            final_output = self.fallback_segmenter.refine_words_bidirectional(
                local_line_data, deskewed_line_image
            )

        # ========================================================================
        # COORDINATE TRANSFORMATION (Map back to Original)
        # ========================================================================
        M_inv = cv2.invertAffineTransform(M)
        remapped_boxes_list = []
        for i in range(len(final_output["text"])):
            left, top = final_output["left"][i], final_output["top"][i]
            width, height = final_output["width"][i], final_output["height"][i]

            # Map the 4 corners
            corners = np.array(
                [
                    [left, top],
                    [left + width, top],
                    [left + width, top + height],
                    [left, top + height],
                ],
                dtype="float32",
            )
            corners_expanded = np.expand_dims(corners, axis=1)
            original_corners = cv2.transform(corners_expanded, M_inv)
            squeezed_corners = original_corners.squeeze(axis=1)

            # Get axis aligned bounding box in original space
            min_x = int(np.min(squeezed_corners[:, 0]))
            max_x = int(np.max(squeezed_corners[:, 0]))
            min_y = int(np.min(squeezed_corners[:, 1]))
            max_y = int(np.max(squeezed_corners[:, 1]))

            remapped_boxes_list.append(
                {
                    "text": final_output["text"][i],
                    "left": min_x,
                    "top": min_y,
                    "width": max_x - min_x,
                    "height": max_y - min_y,
                    "conf": final_output["conf"][i],
                }
            )

        remapped_output = {k: [] for k in final_output.keys()}
        for box in remapped_boxes_list:
            for key in remapped_output.keys():
                remapped_output[key].append(box[key])

        img_h, img_w = line_image.shape[:2]
        remapped_output = self._enforce_logical_constraints(
            remapped_output, img_w, img_h
        )

        # ========================================================================
        # FINAL SAFETY NET
        # ========================================================================
        words = line_data["text"][0].split()
        target_count = len(words)
        current_count = len(remapped_output["text"])
        has_collapsed_boxes = any(w < 3 for w in remapped_output["width"])

        if current_count > 0:
            total_text_len = sum(len(t) for t in remapped_output["text"])
            total_box_width = sum(remapped_output["width"])
            avg_width_pixels = total_box_width / max(1, total_text_len)
        else:
            avg_width_pixels = 0
        is_suspiciously_thin = avg_width_pixels < 4

        if current_count != target_count or is_suspiciously_thin or has_collapsed_boxes:
            used_fallback = True

            # [FIX] Do NOT use original line_image/line_data here.
            # Use the local_line_data + deskewed_line_image pipeline,
            # then transform back using M_inv (same as above).

            # 1. Run fallback on rotated data
            temp_local_output = self.fallback_segmenter.refine_words_bidirectional(
                local_line_data, deskewed_line_image
            )

            # 2. If bidirectional failed to split correctly, use purely mathematical split on rotated data
            if len(temp_local_output["text"]) != target_count:
                h, w = deskewed_line_image.shape[:2]
                temp_local_output = self.fallback_segmenter.convert_line_to_word_level(
                    local_line_data, w, h
                )

            # 3. Transform the result back to original coordinates (M_inv)
            # (Repeating the transformation logic for the safety net result)
            remapped_boxes_list = []
            for i in range(len(temp_local_output["text"])):
                left, top = temp_local_output["left"][i], temp_local_output["top"][i]
                width, height = (
                    temp_local_output["width"][i],
                    temp_local_output["height"][i],
                )

                corners = np.array(
                    [
                        [left, top],
                        [left + width, top],
                        [left + width, top + height],
                        [left, top + height],
                    ],
                    dtype="float32",
                )
                corners_expanded = np.expand_dims(corners, axis=1)
                original_corners = cv2.transform(corners_expanded, M_inv)
                squeezed_corners = original_corners.squeeze(axis=1)

                min_x = int(np.min(squeezed_corners[:, 0]))
                max_x = int(np.max(squeezed_corners[:, 0]))
                min_y = int(np.min(squeezed_corners[:, 1]))
                max_y = int(np.max(squeezed_corners[:, 1]))

                remapped_boxes_list.append(
                    {
                        "text": temp_local_output["text"][i],
                        "left": min_x,
                        "top": min_y,
                        "width": max_x - min_x,
                        "height": max_y - min_y,
                        "conf": temp_local_output["conf"][i],
                    }
                )

            remapped_output = {k: [] for k in temp_local_output.keys()}
            for box in remapped_boxes_list:
                for key in remapped_output.keys():
                    remapped_output[key].append(box[key])

        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            output_path = f"{self.output_folder}/word_segmentation/{safe_image_name}_{safe_shortened_line_text}_final_boxes.png"
            os.makedirs(f"{self.output_folder}/word_segmentation", exist_ok=True)
            output_image_vis = line_image.copy()
            for i in range(len(remapped_output["text"])):
                x, y, w, h = (
                    int(remapped_output["left"][i]),
                    int(remapped_output["top"][i]),
                    int(remapped_output["width"][i]),
                    int(remapped_output["height"][i]),
                )
                cv2.rectangle(output_image_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(output_path, output_image_vis)

        return remapped_output, used_fallback


class HybridWordSegmenter:
    """
    Implements a two-step approach for word segmentation:
    1. Proportional estimation based on text.
    2. Image-based refinement with a "Bounded Scan" to prevent
       over-correction.
    """

    def convert_line_to_word_level(
        self, line_data: Dict[str, List], image_width: int, image_height: int
    ) -> Dict[str, List]:
        """
        Step 1: Converts line-level OCR results to word-level by using a
        robust proportional estimation method.
        Guarantees output box count equals input word count.
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

        # [SAFETY CHECK] Ensure we never estimate a character width of ~0
        avg_char_width = max(3.0, avg_char_width)
        min_word_width = max(5.0, avg_char_width * 0.5)

        current_left = line_left
        for word in words:
            raw_word_width = len(word) * avg_char_width

            # Force the box to have a legible size
            word_width = max(min_word_width, raw_word_width)

            clamped_left = max(0, min(current_left, image_width))
            # We do NOT clamp the width against image_width here because that
            # causes the "0 width" bug if current_left is at the edge.
            # It is better to have a box go off-screen than be 0-width.

            output["text"].append(word)
            output["left"].append(clamped_left)
            output["top"].append(line_top)
            output["width"].append(word_width)
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
        Includes ADAPTIVE NOISE REMOVAL to filter specks based on font size.
        """
        if line_image is None:
            return line_data

        # Early return if 1 or fewer words
        if line_data and line_data.get("text"):
            words = line_data["text"][0].split()
            if len(words) <= 1:
                img_h, img_w = line_image.shape[:2]
                return self.convert_line_to_word_level(line_data, img_w, img_h)

        # --- PRE-PROCESSING: Stricter Binarization ---
        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

        # 1. Calculate standard Otsu threshold first
        otsu_thresh_val, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 2. Apply "Strictness Factor" to remove dark noise
        # 0.75 means "Only keep pixels that are in the darkest 75% of what Otsu thought was foreground"
        # This effectively filters out light-gray noise shadows.
        strict_thresh_val = otsu_thresh_val * 0.75
        _, binary = cv2.threshold(gray, strict_thresh_val, 255, cv2.THRESH_BINARY_INV)

        img_h, img_w = binary.shape

        # [NEW STEP 1] Morphological Opening
        # Physically erodes small protrusions and dust (2x2 pixels or smaller)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # [NEW STEP 2] Adaptive Component Filtering
        # Instead of hardcoded pixels, we filter relative to the line's text size.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_clean, 8, cv2.CV_32S
        )

        # Get heights of all components (excluding background)
        heights = stats[1:, cv2.CC_STAT_HEIGHT]

        if len(heights) > 0:
            # Calculate Median Height of "significant" parts (ignore tiny noise for the median calculation)
            # We assume valid text is at least 20% of the image height
            significant_heights = heights[heights > img_h * 0.2]
            if len(significant_heights) > 0:
                median_h = np.median(significant_heights)
            else:
                median_h = np.median(heights)

            # Define Thresholds based on Text Size
            # 1. Main Threshold: Keep parts taller than 30% of median letter height
            min_height_thresh = median_h * 0.30

            clean_binary = np.zeros_like(binary)
            for i in range(1, num_labels):
                h = stats[i, cv2.CC_STAT_HEIGHT]
                w = stats[i, cv2.CC_STAT_WIDTH]
                area = stats[i, cv2.CC_STAT_AREA]

                # Logic: Keep the component IF:
                # A. It is tall enough to be a letter part (h > threshold)
                # B. OR it is a "Dot" (Period / i-dot):
                #    - Height is small (< threshold)
                #    - Width is ALSO small (roughly square, prevents flat dash/scratch noise)
                #    - Area is reasonable (> 2px)

                is_tall_enough = h > min_height_thresh
                is_dot = (
                    (h <= min_height_thresh) and (w <= min_height_thresh) and (area > 2)
                )

                if is_tall_enough or is_dot:
                    clean_binary[labels == i] = 255

            # Use the adaptively cleaned image for projection
            vertical_projection = np.sum(clean_binary, axis=0)
        else:
            # Fallback if no components found (unlikely)
            vertical_projection = np.sum(binary, axis=0)

        # --- Rest of logic remains the same ---
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
            return self.convert_line_to_word_level(line_data, img_w, img_h)

        # [PREVIOUS FIX] Bounded Scan Distance
        total_chars = len("".join(words))
        if total_chars > 0:
            geom_avg_char_width = img_w / total_chars
        else:
            geom_avg_char_width = 10

        blob_avg_char_width = np.mean([end - start for start, end in char_blobs])
        safe_avg_char_width = min(blob_avg_char_width, geom_avg_char_width * 1.5)
        max_scan_distance = int(safe_avg_char_width * 2.0)

        # [PREVIOUS FIX] Safety Floor
        min_safe_box_width = max(4, int(safe_avg_char_width * 0.5))

        estimated_data = self.convert_line_to_word_level(line_data, img_w, img_h)
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
        combined_boxes = [box.copy() for box in initial_boxes]
        for i in range(len(combined_boxes)):
            final_left = ltr_boxes[i]["left"]
            rtl_right = rtl_boxes[i]["left"] + rtl_boxes[i]["width"]

            combined_boxes[i]["left"] = final_left
            combined_boxes[i]["width"] = max(min_safe_box_width, rtl_right - final_left)

        # --- STEP 4: Contiguous stitching to eliminate gaps ---
        for i in range(len(combined_boxes) - 1):
            if combined_boxes[i + 1]["left"] <= combined_boxes[i]["left"]:
                combined_boxes[i + 1]["left"] = (
                    combined_boxes[i]["left"] + min_safe_box_width
                )

        for i in range(len(combined_boxes) - 1):
            curr = combined_boxes[i]
            nxt = combined_boxes[i + 1]
            gap_width = nxt["left"] - curr["left"]
            curr["width"] = max(min_safe_box_width, gap_width)

        # Convert back to output dict
        final_output = {k: [] for k in estimated_data.keys()}
        for box in combined_boxes:
            if box["width"] >= min_safe_box_width:
                for key in final_output.keys():
                    final_output[key].append(box[key])

        return final_output
