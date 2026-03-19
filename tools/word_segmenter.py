import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from tools.config import OUTPUT_FOLDER, SAVE_WORD_SEGMENTER_OUTPUT_IMAGES

# Adaptive thresholding parameters (resolution-independent via line_height / median CC height)
BLOCK_SIZE_FACTOR = 0.5  # Fraction of line_height when median CC height unavailable
BLOCK_SIZE_MEDIAN_CC_FACTOR = 1.2  # Block size = median_cc_height * this when available
C_VALUE = 2  # Constant subtracted from mean in adaptive thresholding
REFERENCE_LINE_HEIGHT = 50  # Line height (px) at which NOISE_THRESHOLD is defined

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

# Noise removal parameters (resolution-independent: derived from line_height)
MIN_AREA_HEIGHT_FRACTION = 0.05  # MIN_AREA = (line_height * this)^2
MIN_AREA_FLOOR = 2  # Minimum pixel area floor for very low-res lines
DEFAULT_TRIM_PERCENTAGE = (
    0.2  # Percentage to trim from top/bottom for vertical cropping
)

# Skew detection parameters
MIN_SKEW_THRESHOLD = 0.5  # Ignore angles smaller than this (likely noise)
MAX_SKEW_THRESHOLD = 15.0  # Angles larger than this are extreme and likely errors
# Baseline (Hough) skew: minimum bottom points to use baseline method; Hough threshold
SKEW_BASELINE_MIN_POINTS = 20
SKEW_HOUGH_THRESHOLD = 25  # Min votes for a line to be considered

ALLOWED_WORD_MISMATCH_COUNT = 0  # Maximum allowed difference in word count between the target and the detected words during the word segmentation process. If above this, it will use the fallback segmenter.

# Noise detection: if estimated noise (Laplacian variance) is above this (at REFERENCE_LINE_HEIGHT),
# skip primary segmentation and use fallback. Scaled by line_height for resolution independence.
NOISE_THRESHOLD = 800

# Polarity: binarization assumes dark text on light background. If estimated background
# mean is below this, the image is treated as light-on-dark and inverted before binarization.
POLARITY_MEAN_THRESHOLD = 128
POLARITY_CORNER_FRACTION = (
    0.15  # Fraction of width/height used for corner/edge sampling
)


def _find_widest_zero_gaps(
    vertical_projection: np.ndarray,
    n: int,
    gap_threshold: float = 0.0,
) -> List[Tuple[int, int]]:
    """
    Find the N widest contiguous zero-gaps (or near-zero) in the vertical projection.
    Used for justified text: anchor word cut points to the centers of these gaps.
    Returns list of (start, end) in left-to-right order, or empty if not enough gaps.
    """
    if vertical_projection is None or n <= 0:
        return []
    w = len(vertical_projection)
    gaps = []
    in_gap = False
    start = 0
    for x in range(w):
        val = vertical_projection[x] if x < w else 0
        if val <= gap_threshold and not in_gap:
            start = x
            in_gap = True
        elif val > gap_threshold and in_gap:
            gaps.append((start, x))
            in_gap = False
    if in_gap:
        gaps.append((start, w))
    if not gaps:
        return []
    # Sort by width descending, take first n
    gaps_by_width = sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)
    selected = gaps_by_width[:n]
    # Sort by position (left-to-right) for cutting
    selected.sort(key=lambda g: g[0])
    return selected


# Punctuation that often sits after a word with a visible gap (anchor to include in word box)
TRAILING_PUNCTUATION_CHARS = frozenset(".,:;\"'!?)]}")


def _word_ends_with_punctuation(word: str) -> bool:
    """True if word ends with a punctuation character that may have a gap before it."""
    return bool(word and word[-1] in TRAILING_PUNCTUATION_CHARS)


def get_weighted_length(text: str) -> float:
    """
    Proportional-font heuristic: sum character width weights instead of counting chars.
    Narrow chars (i, l, 1, punctuation) get < 1.0; wide chars (W, M, w) get > 1.0.
    Used by HybridWordSegmenter.convert_line_to_word_level for better blind estimation.
    """
    width = 0.0
    weights = {
        "i": 0.4,
        "l": 0.4,
        "1": 0.4,
        "t": 0.6,
        "j": 0.4,
        ".": 0.3,
        ",": 0.3,
        "!": 0.3,
        "'": 0.3,
        "W": 1.3,
        "M": 1.3,
        "m": 1.3,
        "w": 1.2,
        "@": 1.2,
        "%": 1.2,
        " ": 0.5,  # space between words
    }
    for char in text:
        base = 1.1 if char.isupper() else 1.0
        width += weights.get(char, base)
    return width


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

    def _skew_angle_from_baseline(self, binary: np.ndarray) -> float:
        """
        Estimate skew angle from the text baseline using bottom points of foreground
        and Hough line transform. More stable than minAreaRect for short words or
        lines with ascenders/descenders (e.g. "all"). Returns correction angle in
        degrees, or None if baseline cannot be reliably estimated.
        """
        h, w = binary.shape
        # For each column, take the bottom-most foreground pixel (baseline point)
        bottom_points = []
        for x in range(w):
            col = binary[:, x]
            on_pixels = np.where(col > 0)[0]
            if len(on_pixels) > 0:
                y_bottom = int(np.max(on_pixels))
                bottom_points.append((x, y_bottom))
        if len(bottom_points) < SKEW_BASELINE_MIN_POINTS:
            return None
        # Draw baseline points on a blank image for Hough
        baseline_img = np.zeros((h, w), dtype=np.uint8)
        for x, y in bottom_points:
            baseline_img[y, x] = 255
        # Slight dilation so Hough sees a denser line
        kernel = np.ones((2, 2), np.uint8)
        baseline_img = cv2.dilate(baseline_img, kernel)
        lines = cv2.HoughLines(
            baseline_img,
            rho=1,
            theta=np.pi / 180,
            threshold=SKEW_HOUGH_THRESHOLD,
        )
        if lines is None or len(lines) == 0:
            return None
        # Score each line by number of bottom points near it; take best
        best_angle = None
        best_score = 0
        dist_thresh = max(2, h // 30)
        for line in lines:
            rho, theta = line[0]
            # Line equation: rho = x*cos(theta) + y*sin(theta). Perpendicular is at angle theta.
            # Baseline angle from horizontal = theta - 90°. To level it we rotate by -(theta - 90°) = 90° - theta.
            correction_deg = 90.0 - np.degrees(theta)
            # Normalize to [-90, 90] for comparison
            if correction_deg > 90:
                correction_deg -= 180
            elif correction_deg < -90:
                correction_deg += 180
            score = 0
            for x, y in bottom_points:
                # Distance from (x,y) to line rho = x*cos(theta)+y*sin(theta)
                d = abs(x * np.cos(theta) + y * np.sin(theta) - rho)
                if d <= dist_thresh:
                    score += 1
            if score > best_score:
                best_score = score
                best_angle = correction_deg
        if best_angle is None:
            return None
        return float(best_angle)

    def _skew_angle_from_min_area_rect(
        self, coords: np.ndarray, w: int, h: int
    ) -> float:
        """Fallback: skew angle from minAreaRect of all foreground pixels."""
        if len(coords) < 50:
            return 0.0
        rect = cv2.minAreaRect(coords[:, ::-1])
        rect_width, rect_height = rect[1]
        angle = rect[2]
        if rect_width < rect_height:
            angle += 90
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        return float(angle)

    def _deskew_image(self, gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detects skew using baseline (Hough on bottom points of letters) when possible,
        which is more stable for short words and ascenders/descenders; falls back to
        minAreaRect otherwise.
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

        # Prefer baseline-based skew (stable for short words / ascenders-descenders)
        correction_angle = self._skew_angle_from_baseline(binary)
        if correction_angle is None:
            correction_angle = self._skew_angle_from_min_area_rect(coords, w, h)

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

    def _estimate_noise(self, gray: np.ndarray) -> float:
        """
        Estimate image noisiness using Laplacian variance. Noisy images tend to have
        high high-frequency content, so higher values indicate more noise (or very
        sharp edges). Used to skip the primary segmentation pipeline when above
        NOISE_THRESHOLD and use the fallback segmenter instead.
        """
        if gray is None or gray.size == 0:
            return 0.0
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        return float(lap.var())

    def _block_size_from_median_cc_height(
        self, gray: np.ndarray, line_height: int, fallback_block_size: int
    ) -> int:
        """
        Determine adaptive threshold block size from median height of connected components
        (resolution-independent). Uses an Otsu pre-pass to get CCs; if median height is
        valid, returns block_size = median_cc_height * BLOCK_SIZE_MEDIAN_CC_FACTOR.
        Otherwise returns fallback_block_size (e.g. from line_height).
        """
        if gray is None or gray.size == 0 or line_height < 3:
            return fallback_block_size
        _, otsu_binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            otsu_binary, 8, cv2.CV_32S
        )
        if num_labels < 3:  # background + need at least 2 components
            return fallback_block_size
        areas = stats[1:, cv2.CC_STAT_AREA]
        heights = stats[1:, cv2.CC_STAT_HEIGHT]
        min_area_cc = max(2, int((line_height * 0.02) ** 2))
        valid = areas >= min_area_cc
        if not np.any(valid):
            return fallback_block_size
        median_h = np.median(heights[valid])
        if np.isnan(median_h) or median_h < 2:
            return fallback_block_size
        block = max(3, int(median_h * BLOCK_SIZE_MEDIAN_CC_FACTOR))
        if block % 2 == 0:
            block += 1
        return block

    def _normalize_polarity_for_binarization(self, gray: np.ndarray) -> np.ndarray:
        """
        Ensure we work with dark-text-on-light-background for binarization. If the
        image is mostly dark (light text on dark background), invert it so that
        adaptive threshold and projection profile logic behave correctly.

        Uses corner/edge regions to estimate background (typical in documents);
        falls back to global mean for very small or full-page line crops.
        """
        if gray is None or gray.size == 0:
            return gray
        h, w = gray.shape
        frac = POLARITY_CORNER_FRACTION
        # Sample corners and edges (background is often visible there)
        margin_w = max(1, int(w * frac))
        margin_h = max(1, int(h * frac))
        corner_pixels = []
        if margin_w < w and margin_h < h:
            top_left = gray[:margin_h, :margin_w]
            top_right = gray[:margin_h, -margin_w:]
            bottom_left = gray[-margin_h:, :margin_w]
            bottom_right = gray[-margin_h:, -margin_w:]
            for region in (top_left, top_right, bottom_left, bottom_right):
                corner_pixels.append(region.ravel())
            if corner_pixels:
                corner_pixels = np.concatenate(corner_pixels)
                background_mean = float(np.mean(corner_pixels))
            else:
                background_mean = float(np.mean(gray))
        else:
            background_mean = float(np.mean(gray))
        if background_mean < POLARITY_MEAN_THRESHOLD:
            return cv2.bitwise_not(gray)
        return gray

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

        # Validate that line_image is not empty before processing
        if line_image is None or line_image.size == 0 or len(line_image.shape) < 2:
            # If line_image is empty, fall back to proportional estimation
            return {}, False

        line_number = line_data["line"][0]
        safe_image_name = _sanitize_filename(image_name or "image", max_length=50)
        safe_line_number = _sanitize_filename(str(line_number), max_length=10)
        safe_shortened_line_text = _sanitize_filename(line_text, max_length=10)

        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f"{self.output_folder}/word_segmentation/{safe_image_name}_{safe_line_number}_{safe_shortened_line_text}_original.png"
            os.makedirs(f"{self.output_folder}/word_segmentation", exist_ok=True)
            # Only write if image is valid
            if line_image.size > 0 and len(line_image.shape) >= 2:
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
        line_height = img_h
        estimated_char_height = img_h * 0.6
        avg_char_width_approx = img_w / approx_char_count

        # Block size from line height (resolution-independent); could be refined from median CC height in two-pass
        block_size = max(3, int(line_height * BLOCK_SIZE_FACTOR))
        if block_size % 2 == 0:
            block_size += 1

        # Noise threshold scaled by line height so behavior is resolution-independent
        effective_noise_threshold = NOISE_THRESHOLD * (
            line_height / REFERENCE_LINE_HEIGHT
        )

        # --- Noise check: skip primary pipeline if image is too noisy ---
        noise_level = self._estimate_noise(deskewed_gray)
        if noise_level > effective_noise_threshold:
            used_fallback = True
            final_output = self.fallback_segmenter.refine_words_bidirectional(
                local_line_data, deskewed_line_image
            )
        else:
            # --- Polarity: ensure dark text on light background for binarization ---
            gray_for_binary = self._normalize_polarity_for_binarization(deskewed_gray)

            # Refine block size from median CC height (Otsu pre-pass) when possible
            block_size = self._block_size_from_median_cc_height(
                gray_for_binary, line_height, block_size
            )

            # --- Binarization ---
            binary_adaptive = cv2.adaptiveThreshold(
                gray_for_binary,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                C_VALUE,
            )
            otsu_thresh_val, _ = cv2.threshold(
                gray_for_binary, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            strict_thresh_val = otsu_thresh_val * 0.75
            _, binary_strict = cv2.threshold(
                gray_for_binary, strict_thresh_val, 255, cv2.THRESH_BINARY_INV
            )
            binary = cv2.bitwise_and(binary_adaptive, binary_strict)

            if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
                output_path = f"{self.output_folder}/word_segmentation/{safe_image_name}_{safe_line_number}_{safe_shortened_line_text}_binary.png"
                cv2.imwrite(output_path, binary)

            # --- Morphological Closing ---
            morph_width = max(3, int(avg_char_width_approx * 0.40))
            morph_height = max(2, int(avg_char_width_approx * 0.1))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (morph_width, morph_height)
            )
            closed_binary = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE, kernel, iterations=1
            )

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
                    line_h = img_h
                    estimated_char_height = img_h * 0.7
                    # Resolution-independent min area: (line_height * 0.05)^2 with floor
                    min_area_threshold = max(
                        MIN_AREA_FLOOR,
                        int((line_h * MIN_AREA_HEIGHT_FRACTION) ** 2),
                    )
                    estimated_min_letter_area = max(
                        2,
                        int(estimated_char_height * 0.2 * estimated_char_height * 0.15),
                    )
                    area_threshold = max(
                        min_area_threshold, min(p1, estimated_min_letter_area)
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

            # Validate clean_binary is not empty before proceeding
            if (
                clean_binary is None
                or clean_binary.size == 0
                or len(clean_binary.shape) < 2
            ):
                # If clean_binary is empty, fall back to proportional estimation
                return {}, False

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
                # Ensure y_end > y_start to avoid empty slice
                if y_end > y_start:
                    analysis_image = clean_binary[y_start:y_end, :]
                else:
                    # If slice would be empty, use the full image
                    analysis_image = clean_binary
            else:
                analysis_image = clean_binary

            # Validate that analysis_image is not empty before proceeding
            if (
                analysis_image is None
                or analysis_image.size == 0
                or len(analysis_image.shape) < 2
            ):
                # If analysis_image is empty, fall back to proportional estimation
                return {}, False

            if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
                # Validate that analysis_image is not empty before writing
                if analysis_image.size > 0 and len(analysis_image.shape) >= 2:
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
                        analysis_image,
                        avg_char_width_approx,
                        min_space_factor,
                        v_factor,
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
                    if (
                        diff <= ALLOWED_WORD_MISMATCH_COUNT
                        and backup_boxes_s1 is None
                        and is_geom_valid
                    ):
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
                            s2_bin[y_start:y_end, :]
                            if len(non_zero_rows) > 0
                            else s2_bin
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

                        if (
                            diff <= ALLOWED_WORD_MISMATCH_COUNT
                            and backup_boxes_s2 is None
                            and is_geom_valid
                        ):
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

                    if (
                        comp_center_y < cca_img_h * 0.1
                        or comp_center_y > cca_img_h * 0.9
                    ):
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
                                    if (
                                        not component_center_in_box
                                        or overlap > max_overlap
                                    ):
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
                        k: []
                        for k in ["text", "left", "top", "width", "height", "conf"]
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
    1. Proportional estimation based on text (primary; avoids image noise).
    2. Image-based refinement with a "Bounded Scan" that cannot shrink boxes
       beyond a fraction of the text-based width.

    Design: Relies more on expected character spacing from the text than on
    image analysis, so noisy images are less likely to produce tiny or
    missing boxes.

    Situations that could otherwise cause very small boxes (and how we mitigate):
    - False gaps in the vertical projection (noise/speckle) -> refinement is
      bounded by shrink_limit_fraction; initial boxes use proportional only.
    - Image-based "justified" gap anchoring picking wrong cuts -> we do not
      use vertical_projection for initial segmentation here; only proportional.
    - Bidirectional scan snapping to a thin low-density strip inside a word ->
      same shrink bound; fallback "thinnest point" also clamped.
    - De-overlapping stealing space from the next word -> shrink bound keeps
      each box at least (1 - shrink_limit_fraction) of initial width.

    ROBUSTNESS UPGRADES:
    - Uses Horizontal Smearing to prevent cutting inside noisy characters.
    - Uses Gaussian Blur to suppress speckle noise.
    - Implements 'Noise Floors' for gap detection (never assumes perfect 0).
    """

    def convert_line_to_word_level(
        self,
        line_data: Dict[str, List],
        image_width: int,
        image_height: int,
        vertical_projection: np.ndarray = None,
    ) -> Dict[str, List]:
        """
        Step 1: Converts line-level OCR results to word-level using proportional estimation.
        Includes noise-tolerant gap anchoring for justified text.
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

        i = 0
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

        # --- Justified text: anchor cut points to widest zero-gaps in projection ---
        if (
            vertical_projection is not None
            and len(vertical_projection) == image_width
            and num_spaces > 0
        ):
            # ROBUSTNESS: Allow significantly more noise in gaps for justified text detection.
            # Allow up to 3% of the column height to be noise and still count as a "gap".
            dynamic_gap_threshold = max(255.0 * 0.03 * image_height, 255.0 * 2)
            gaps = _find_widest_zero_gaps(
                vertical_projection, n=num_spaces, gap_threshold=dynamic_gap_threshold
            )
            if len(gaps) == num_spaces:
                cuts = [0]
                for start, end in gaps:
                    cuts.append((start + end) // 2)
                cuts.append(image_width)

                for idx, word in enumerate(words):
                    left_px = cuts[idx]
                    right_px = cuts[idx + 1]
                    width_px = max(1, right_px - left_px)
                    output["text"].append(word)
                    output["left"].append(line_left + left_px)
                    output["top"].append(line_top)
                    output["width"].append(width_px)
                    output["height"].append(line_height)
                    output["conf"].append(line_conf)
                return output

        # --- Proportional estimation ---
        total_line_weight = get_weighted_length(line_text)
        if total_line_weight <= 0:
            total_line_weight = 1.0
        avg_weight_unit = line_width / total_line_weight
        estimated_space_width = get_weighted_length(" ") * avg_weight_unit

        avg_char_width = line_width / (num_chars if num_chars > 0 else 1)
        avg_char_width = max(3.0, avg_char_width)
        min_word_width = max(5.0, avg_char_width * 0.5)

        current_left = line_left
        for word in words:
            word_weight = get_weighted_length(word)
            raw_word_width = word_weight * avg_weight_unit
            word_width = max(min_word_width, raw_word_width)

            clamped_left = max(0, min(current_left, image_width))
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
        img_h: int,
        direction: str = "ltr",
        trailing_punctuation: List[bool] = None,
        shrink_limit_fraction: float = 0.5,
    ) -> List[Dict]:
        """
        Helper function to run one pass of refinement.
        ROBUSTNESS UPGRADE:
        - Uses a 'gap_noise_floor' instead of looking for 0.
        - Enforces 'safety_density_limit': if the "thinnest" point is still thick (ink),
          it refuses to cut there (prevents cutting bold letters).
        - shrink_limit_fraction: Refinement cannot shrink a box by more than this fraction
          of its initial (text-based) width from either edge. Prevents noise from creating
          tiny boxes; keeps segmentation anchored to expected character spacing.
        """

        refined_boxes = [box.copy() for box in initial_boxes]
        if trailing_punctuation is None:
            trailing_punctuation = [False] * len(initial_boxes)

        # ROBUSTNESS: Define what constitutes a "gap" vs "ink"
        # 1. Gap Floor: Anything below 5% of image height is treated as empty space (noise tolerance)
        gap_noise_floor = 255.0 * (img_h * 0.05)

        # 2. Ink Safety Limit: If the "thinnest" point has > 25% ink density, it is NOT a gap.
        # It's a character. Do not cut.
        safety_density_limit = 255.0 * (img_h * 0.25)

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
            init_width = max(1, int(box["width"]))
            # Bounds from initial (text-based) box: don't let image refinement shrink too much
            min_right = right - int(shrink_limit_fraction * init_width)
            max_left = left + int(shrink_limit_fraction * init_width)

            left = max(0, min(left, img_w - 1))
            right = max(0, min(right, img_w - 1))

            new_left, new_right = left, right

            if direction == "ltr" or direction == "both":  # Scan right
                if right < img_w:
                    scan_limit = min(img_w, right + max_scan_distance)
                    search_range = range(right, scan_limit)

                    best_x = right
                    min_density = float("inf")
                    found_gap = False
                    first_gap_x = None

                    for x in search_range:
                        density = vertical_projection[x]

                        # Check for Gap
                        if density <= gap_noise_floor:
                            first_gap_x = x
                            found_gap = True
                            break

                        # Track minimum density for fallback
                        if density < min_density:
                            min_density = density
                            best_x = x

                    if found_gap and first_gap_x is not None:
                        if trailing_punctuation[i]:
                            # Logic to jump over the gap and include the punctuation blob
                            # ... (same safety limits as before) ...
                            proj_len = len(vertical_projection)
                            x_pos = first_gap_x

                            # 1. Cross the gap
                            gap_safety_limit = x_pos + (max_scan_distance // 2)
                            while (
                                x_pos < scan_limit
                                and x_pos < proj_len
                                and vertical_projection[x_pos] <= gap_noise_floor
                            ):
                                if x_pos >= gap_safety_limit:
                                    break
                                x_pos += 1

                            # 2. Consume blob
                            blob_start = x_pos
                            blob_safety_limit = blob_start + max(1, int(img_h * 0.5))
                            while (
                                x_pos < scan_limit
                                and x_pos < proj_len
                                and vertical_projection[x_pos] > gap_noise_floor
                            ):
                                if x_pos >= blob_safety_limit:
                                    x_pos = first_gap_x  # Revert
                                    break
                                x_pos += 1
                            new_right = min(x_pos, scan_limit)
                        else:
                            new_right = first_gap_x

                    elif not found_gap:
                        # Fallback: No clear gap found.
                        # ROBUSTNESS CHECK: Is the "thinnest" point actually thin?
                        if min_density < safety_density_limit:
                            new_right = best_x
                        else:
                            # The thinnest point is still very dark (ink).
                            # Don't cut through a letter. Keep original guess or limit.
                            new_right = right

            if direction == "rtl" or direction == "both":  # Scan left
                if left > 0:
                    scan_limit = max(0, left - max_scan_distance)
                    search_range = range(left, scan_limit, -1)

                    best_x = left
                    min_density = float("inf")
                    found_gap = False

                    for x in search_range:
                        density = vertical_projection[x]

                        if density <= gap_noise_floor:
                            new_left = x
                            found_gap = True
                            break

                        if density < min_density:
                            min_density = density
                            best_x = x

                    if not found_gap:
                        # ROBUSTNESS CHECK
                        if min_density < safety_density_limit:
                            new_left = best_x
                        else:
                            # Refuse to cut through dense ink
                            new_left = left

            # --- Anchor to text: don't shrink past allowed fraction of initial width ---
            new_right = max(new_right, min_right)
            new_left = min(new_left, max_left)

            # --- Directional de-overlapping ---
            if direction == "ltr":
                if new_left < last_corrected_right_edge:
                    new_left = last_corrected_right_edge
                if new_right <= new_left:
                    new_right = new_left + 1
                last_corrected_right_edge = new_right
            else:  # rtl
                if new_right > next_corrected_left_edge:
                    new_right = next_corrected_left_edge
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
        Refines boxes using a robust bidirectional scan.
        DIFFERENCE FROM MAIN SEGMENTER: Uses aggressive smoothing and horizontal
        smearing to force-merge characters, prioritizing word separation over
        character detail.
        """
        if line_image is None:
            return line_data

        # Handle grayscale (2D) or BGR (3D) line images
        if len(line_image.shape) == 2:
            gray = np.ascontiguousarray(line_image)
        else:
            gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape[:2]

        # OpenCV GaussianBlur(5,5) and later adaptiveThreshold need minimum dimensions.
        # Avoid "Unknown C++ exception" on very small line crops (e.g. 1–4 px).
        if img_h < 5 or img_w < 5:
            return self.convert_line_to_word_level(line_data, img_w, img_h)

        if line_data and line_data.get("text"):
            words = line_data["text"][0].split()
            if len(words) <= 1:
                return self.convert_line_to_word_level(line_data, img_w, img_h)

        # --- PRE-PROCESSING: The "Bulldozer" Approach ---
        # 1. Gaussian Blur: Suppress high-frequency speckle noise that confuses the main segmenter
        # We accept slight edge blurring for the sake of noise reduction.
        # OpenCV can intermittently throw low-information C++ exceptions on some
        # page crops (often due to dtype/range/nan/inf issues). If that happens,
        # fall back to the non-image-based word conversion to keep OCR flowing.
        try:
            # Guard against NaN/Inf propagating into OpenCV internals.
            if gray.dtype.kind in ("f", "c"):
                gray = np.nan_to_num(gray, nan=0.0, posinf=255.0, neginf=0.0)

            # GaussianBlur is most stable on uint8 or float32. If we have another
            # dtype (e.g. int16/float64/object), normalize and cast.
            if gray.dtype != np.uint8 and gray.dtype != np.float32:
                # Normalize to 0..255 if range looks unusual.
                gmin = float(np.min(gray)) if gray.size else 0.0
                gmax = float(np.max(gray)) if gray.size else 255.0
                if gmax > 255.0 or gmin < 0.0:
                    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
                gray = np.clip(gray, 0, 255).astype(np.uint8)

            blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        except Exception:
            return self.convert_line_to_word_level(line_data, img_w, img_h)

        # 2. Aggressive Thresholding
        # We use a larger block size here to be less sensitive to local texture variations
        block_size = max(25, int(img_h * 0.5))
        if block_size % 2 == 0:
            block_size += 1

        binary = cv2.adaptiveThreshold(
            blurred_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            10,
        )

        # 3. Horizontal Smearing (The critical difference)
        # We intentionally smear mostly horizontally to bridge gaps inside noisy letters.
        # Kernel width: ~15-20% of line height.
        smear_w = max(3, int(img_h * 0.20))
        smear_h = max(1, int(img_h * 0.05))
        kernel_smear = cv2.getStructuringElement(cv2.MORPH_RECT, (smear_w, smear_h))

        # Apply Morphological Closing
        binary_smeared = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_smear)

        # Calculate projection on the SMEARED image
        vertical_projection = np.sum(binary_smeared, axis=0)

        # --- Setup for Scan ---
        # Detect blobs to estimate character width for scan limiting
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

        total_chars = len("".join(words))
        if total_chars > 0:
            geom_avg_char_width = img_w / total_chars
        else:
            geom_avg_char_width = 10

        blob_avg_char_width = np.mean([end - start for start, end in char_blobs])
        safe_avg_char_width = min(blob_avg_char_width, geom_avg_char_width * 1.5)

        # Scan distance parameters
        max_scan_distance = max(int(safe_avg_char_width * 2.5), int(img_h * 0.6))
        min_safe_box_width = max(4, int(safe_avg_char_width * 0.5))

        # --- Standard Logic Continues ---
        # Use proportional estimation only (no vertical_projection) so initial boxes
        # are driven by text/character spacing. Image-based gap anchoring on noisy
        # images can produce tiny slices; refinement will still run but is bounded.
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

        trailing_punctuation = [
            _word_ends_with_punctuation(estimated_data["text"][j])
            for j in range(len(estimated_data["text"]))
        ]

        # Run passes (ensure _run_single_pass uses the robust gap logic)
        ltr_boxes = self._run_single_pass(
            initial_boxes,
            vertical_projection,
            max_scan_distance,
            img_w,
            img_h,
            "ltr",
            trailing_punctuation,
        )
        rtl_boxes = self._run_single_pass(
            initial_boxes,
            vertical_projection,
            max_scan_distance,
            img_w,
            img_h,
            "rtl",
            trailing_punctuation,
        )

        # [Re-use stitching logic from previous code...]
        combined_boxes = [box.copy() for box in initial_boxes]
        for i in range(len(combined_boxes)):
            final_left = ltr_boxes[i]["left"]
            rtl_right = rtl_boxes[i]["left"] + rtl_boxes[i]["width"]
            combined_boxes[i]["left"] = final_left
            combined_boxes[i]["width"] = max(min_safe_box_width, rtl_right - final_left)

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

        final_output = {k: [] for k in estimated_data.keys()}
        for box in combined_boxes:
            # Always keep one box per word; enforce minimum width 1 for valid geometry
            box_width = max(1, box["width"])
            box["width"] = box_width
            for key in final_output.keys():
                final_output[key].append(box[key])

        return final_output
