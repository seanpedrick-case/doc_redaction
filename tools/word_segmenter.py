import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from tools.config import OUTPUT_FOLDER, SAVE_WORD_SEGMENTER_OUTPUT_IMAGES

INITIAL_KERNEL_WIDTH_FACTOR = 0.05  # Default 0.05
INITIAL_VALLEY_THRESHOLD_FACTOR = 0.05  # Default 0.05
MAIN_VALLEY_THRESHOLD_FACTOR = 0.15  # Default 0.15
C_VALUE = 4  # Default 4
BLOCK_SIZE_FACTOR = 1.5  # Default 1.5
MIN_SPACE_FACTOR = 0.3  # Default 0.4
MATCH_TOLERANCE = 0  # Default 0
MIN_AREA_THRESHOLD = 6  # Default 6
DEFAULT_TRIM_PERCENTAGE = 0.2  # Default 0.2


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

        # --- Binarization (copied from _deskew_image) ---
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

        # Small noise removal
        opening_kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, opening_kernel)

        # --- Extract text pixel coordinates ---
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 50:
            # print(
            #     "Warning: Not enough text pixels for orientation. Assuming horizontal."
            # )
            M_orient = cv2.getRotationMatrix2D(center, 0, 1.0)
            return gray_image, M_orient

        # --- Robust bounding-box check (no minAreaRect quirks) ---
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        box_height = ymax - ymin
        box_width = xmax - xmin

        orientation_angle = 0.0
        if box_height > box_width:
            # print(
            #     f"Detected vertical orientation (W:{box_width} < H:{box_height}). Applying 90-degree correction."
            # )
            orientation_angle = 90.0
        else:
            # print(
            #     f"Detected horizontal orientation (W:{box_width} >= H:{box_height}). No orientation correction."
            # )
            M_orient = cv2.getRotationMatrix2D(center, 0, 1.0)
            return gray_image, M_orient

        # --- Apply 90-degree correction ---
        M_orient = cv2.getRotationMatrix2D(center, orientation_angle, 1.0)

        # Calculate new image bounds (they will be swapped)
        new_w, new_h = h, w

        # Adjust translation part of M_orient to center the new image
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

        # Use a single, reliable binarization method for detection.
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
            # print("Warning: Not enough text pixels to detect skew. Skipping.")
            M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, 1.0)
            return gray_image, M

        rect = cv2.minAreaRect(coords[:, ::-1])

        rect_width, rect_height = rect[1]
        angle = rect[2]

        # If the rectangle is described as vertical, normalize it
        if rect_width < rect_height:
            # Swap dimensions
            rect_width, rect_height = rect_height, rect_width
            # Correct the angle
            angle += 90

        # The angle from minAreaRect is in [-90, 0). After normalization,
        # our angle for a horizontal line will be close to 0 or -90/90.
        # We need one last correction for angles near +/- 90.
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

        correction_angle = angle

        # print(f"Normalized shape (W:{rect_width:.0f}, H:{rect_height:.0f}). Detected angle: {correction_angle:.2f} degrees.")

        # Final sanity checks on the angle
        MIN_SKEW_THRESHOLD = 0.5  # Ignore angles smaller than this (likely noise)
        MAX_SKEW_THRESHOLD = (
            15.0  # Angles larger than this are extreme and likely errors
        )

        if abs(correction_angle) < MIN_SKEW_THRESHOLD:
            # print(f"Detected angle {correction_angle:.2f}° is too small (likely noise). Skipping deskew.")
            correction_angle = 0.0
        elif abs(correction_angle) > MAX_SKEW_THRESHOLD:
            # print(f"Warning: Corrected angle {correction_angle:.2f}° is extreme. Skipping deskew.")
            correction_angle = 0.0

        # Create rotation matrix and apply the final correction
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
        # This helper function remains IDENTICAL. No changes needed.
        # ... (code from the previous version)
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
        Enforces geometric sanity checks with improved robustness:
        1. VERTICAL SNAP: Expands small/floating boxes to match the line's median vertical bounds.
        2. INFLATE: Expands 'THIN' widths.
        3. SORT: Sorts by position AND size.
        4. CLEAN: Removes nested boxes.
        5. RESOLVE: Fixes sequential overlaps.
        6. BOUND: Clamps to image.
        """
        if not output or not output["text"]:
            return output

        # --- 1. UNPACK ---
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

        # --- 2. VERTICAL STANDARDIZATION (New Step) ---
        if boxes:
            # Calculate the median vertical bounds for this line
            tops = [b["top"] for b in boxes]
            bottoms = [b["top"] + b["height"] for b in boxes]

            median_top = int(np.median(tops))
            median_bottom = int(np.median(bottoms))
            median_height = median_bottom - median_top

            # Heuristic: If a box is 'short' (e.g., < 60% of median height),
            # it is likely a noise speck, punctuation, or diacritic floating high/low.
            # We expand it to match the line's median bounds.
            for box in boxes:
                box_bottom = box["top"] + box["height"]

                # Check if box is too short relative to the line
                # (We use 0.6 as a safe threshold; real words are usually > 60% of line height)
                if median_height > 0 and box["height"] < (0.6 * median_height):
                    # Snap to the median bounds
                    # We use min/max to ensure we extend, never shrink
                    new_top = min(box["top"], median_top)
                    new_bottom = max(box_bottom, median_bottom)

                    box["top"] = new_top
                    box["height"] = max(1, new_bottom - new_top)

        # --- 3. INFLATE THIN WIDTHS ---
        if boxes:
            widths = [b["width"] for b in boxes]
            median_width = np.median(widths)

            # Rule: Box must be at least 12px OR 35% of median width.
            min_viable_width = max(12, int(median_width * 0.35))

            for box in boxes:
                # Skip punctuation from aggressive width inflation
                is_punctuation = len(box["text"]) == 1 and not box["text"].isalnum()

                if not is_punctuation and box["width"] < min_viable_width:
                    target_width = min_viable_width
                    diff = target_width - box["width"]
                    box["left"] -= diff // 2
                    box["width"] = target_width

        # --- 4. ROBUST SORT ---
        # Sort by 'left' (ascending), then by 'width' (descending).
        boxes.sort(key=lambda b: (b["left"], -b["width"]))

        # --- 5. REMOVE NESTED BOXES ---
        valid_boxes = []
        if boxes:
            current_base = boxes[0]
            valid_boxes.append(current_base)

            for i in range(1, len(boxes)):
                candidate = boxes[i]
                base_x1 = current_base["left"] + current_base["width"]
                cand_x1 = candidate["left"] + candidate["width"]

                # Check for nesting with a small buffer
                if (
                    candidate["left"] >= current_base["left"] - 2
                    and cand_x1 <= base_x1 + 2
                ):
                    continue
                else:
                    valid_boxes.append(candidate)
                    current_base = candidate

        boxes = valid_boxes

        # --- 6. RESOLVE SEQUENTIAL OVERLAPS & BOUNDARIES ---
        for i in range(len(boxes)):
            curr = boxes[i]

            # Boundary Check
            curr["left"] = max(0, curr["left"])
            curr["top"] = max(0, curr["top"])
            if curr["left"] + curr["width"] > image_width:
                curr["width"] = max(1, image_width - curr["left"])

            # Overlap Check
            if i < len(boxes) - 1:
                next_box = boxes[i + 1]
                curr_x1 = curr["left"] + curr["width"]
                next_x0 = next_box["left"]

                if curr_x1 > next_x0:
                    overlap = curr_x1 - next_x0
                    new_width = curr["width"] - overlap
                    curr["width"] = max(1, new_width)

        # --- 7. REPACK ---
        cleaned_output = {
            k: [] for k in ["text", "left", "top", "width", "height", "conf"]
        }
        for box in boxes:
            for key in cleaned_output.keys():
                cleaned_output[key].append(box[key])

        return cleaned_output

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

        # print(f"line_text: {line_text}")
        shortened_line_text = line_text.replace(" ", "_")[:10]

        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f"{self.output_folder}/word_segmentation/{image_name}_{shortened_line_text}_original.png"
            os.makedirs(f"{self.output_folder}/word_segmentation", exist_ok=True)
            cv2.imwrite(output_path, line_image)
            # print(f"\nSaved original image to '{output_path}'")

        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

        # --- STEP 1: Correct major orientation (e.g., 90 degrees) ---
        # M_orient transforms from ORIGINAL -> ORIENTED
        oriented_gray, M_orient = self._correct_orientation(gray)

        # --- STEP 2: Correct minor skew (e.g., -2 degrees) ---
        # M_skew transforms from ORIENTED -> DESKEWED
        deskewed_gray, M_skew = self._deskew_image(oriented_gray)

        # --- STEP 3: Combine Transformations ---
        # We need a single matrix 'M' that transforms from ORIGINAL -> DESKEWED
        # We do this by converting to 3x3 matrices and multiplying: M = M_skew * M_orient

        # Convert to 3x3
        M_orient_3x3 = np.vstack([M_orient, [0, 0, 1]])
        M_skew_3x3 = np.vstack([M_skew, [0, 0, 1]])

        # Combine transformations
        M_total_3x3 = M_skew_3x3 @ M_orient_3x3

        # Get the final 2x3 transformation matrix
        M = M_total_3x3[0:2, :]

        # --- Apply TOTAL transformation to the original color image ---
        # The final dimensions are those of the *last* image in the chain: deskewed_gray
        h, w = deskewed_gray.shape

        deskewed_line_image = cv2.warpAffine(
            line_image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Validate deskewed_line_image before saving
        if (
            deskewed_line_image is None
            or not isinstance(deskewed_line_image, np.ndarray)
            or deskewed_line_image.size == 0
        ):
            # print(
            #     f"Error: deskewed_line_image is None or empty (image_name: {image_name})"
            # )
            return ({}, False)

        # Save deskewed image (optional, only if image_name is provided)
        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f"{self.output_folder}/word_segmentation/{image_name}_{shortened_line_text}_deskewed.png"
            os.makedirs(f"{self.output_folder}/word_segmentation", exist_ok=True)
            cv2.imwrite(output_path, deskewed_line_image)
            # print(f"\nSaved deskewed image to '{output_path}'")

        # --- Step 1: Binarization and Stable Width Calculation (Unchanged) ---
        approx_char_count = len(line_data["text"][0].replace(" ", ""))
        if approx_char_count == 0:
            return {}, False
        img_h, img_w = deskewed_gray.shape
        avg_char_width_approx = img_w / approx_char_count
        block_size = int(avg_char_width_approx * BLOCK_SIZE_FACTOR)
        if block_size % 2 == 0:
            block_size += 1

        # Validate deskewed_gray and ensure block_size is valid
        if deskewed_gray is None or not isinstance(deskewed_gray, np.ndarray):
            # print(
            #     f"Error: deskewed_gray is None or not a numpy array (image_name: {image_name})"
            # )
            return ({}, False)

        if len(deskewed_gray.shape) != 2:
            # print(
            #     f"Error: deskewed_gray must be a 2D grayscale image (shape: {deskewed_gray.shape}, image_name: {image_name})"
            # )
            return ({}, False)

        if block_size < 3:
            # print(
            #     f"Warning: block_size ({block_size}) is too small for adaptiveThreshold. "
            #     f"Using minimum value of 3. (image_name: {image_name}, "
            #     f"img_w: {img_w}, approx_char_count: {approx_char_count}, "
            #     f"avg_char_width_approx: {avg_char_width_approx:.2f})"
            # )
            block_size = 3

        binary = cv2.adaptiveThreshold(
            deskewed_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            C_VALUE,
        )

        # Validate binary image before saving
        if binary is None or not isinstance(binary, np.ndarray) or binary.size == 0:
            # print(
            #     f"Error: binary image is None or empty (image_name: {image_name})"
            # )
            return ({}, False)

        # Save cropped image (optional, only if image_name is provided)
        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f"{self.output_folder}/word_segmentation/{image_name}_{shortened_line_text}_binary.png"
            os.makedirs(f"{self.output_folder}/word_segmentation", exist_ok=True)
            cv2.imwrite(output_path, binary)
            # print(f"\nSaved cropped image to '{output_path}'")

        # --- NEW STEP 1.5: Post-processing with Morphology ---
        # This "closes" gaps in letters and joins nearby components.

        # Create a small kernel (e.g., 3x3 rectangle)
        # You may need to tune this size.
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Use MORPH_CLOSE to close small holes and gaps within the letters
        # It's a dilation followed by an erosion
        closed_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Validate closed_binary image before saving
        if (
            closed_binary is None
            or not isinstance(closed_binary, np.ndarray)
            or closed_binary.size == 0
        ):
            # print(
            #     f"Error: closed_binary image is None or empty (image_name: {image_name})"
            # )
            return ({}, False)

        # (Optional) You could also use a DILATE to make letters thicker
        # dilated_binary = cv2.dilate(closed_binary, kernel, iterations=1)
        # Use 'closed_binary' (or 'dilated_binary') from now on.

        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f"{self.output_folder}/word_segmentation/{image_name}_{shortened_line_text}_closed_binary.png"
            os.makedirs(f"{self.output_folder}/word_segmentation", exist_ok=True)
            cv2.imwrite(output_path, closed_binary)
            # print(f"\nSaved dilated binary image to '{output_path}'")

        # --- Step 2: Intelligent Noise Removal (Improved) ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            closed_binary, 8, cv2.CV_32S
        )
        clean_binary = np.zeros_like(binary)

        if num_labels > 1:
            areas = stats[
                1:, cv2.CC_STAT_AREA
            ]  # Get all component areas, skip background (label 0)

            # Handle edge case of empty 'areas' array
            if len(areas) == 0:
                clean_binary = binary
                # print("Warning: No components found after binarization.")
                areas = np.array([0])  # Add a dummy value to prevent crashes

            # --- 1. Calculate the DEFAULT CONSERVATIVE threshold ---
            # This is your existing logic, which works well for *clean* lines.
            p1 = np.percentile(areas, 1)
            img_h, img_w = binary.shape
            estimated_char_height = img_h * 0.7
            estimated_min_letter_area = max(
                2, int(estimated_char_height * 0.2 * estimated_char_height * 0.15)
            )

            # This is the "safe" threshold that protects small letters on clean lines.
            area_threshold = max(MIN_AREA_THRESHOLD, min(p1, estimated_min_letter_area))
            # print(f"Noise Removal: Initial conservative threshold: {area_threshold:.1f} (p1={p1:.1f}, est_min={estimated_min_letter_area:.1f})")

            # --- 2. Find a "Noise-to-Text" Gap (to enable AGGRESSIVE mode) ---
            sorted_areas = np.sort(areas)
            has_clear_gap = False
            aggressive_threshold = -1
            area_before_gap = -1

            if len(sorted_areas) > 10:  # Need enough components to analyze
                area_diffs = np.diff(sorted_areas)
                if len(area_diffs) > 0:
                    # Use your "gap" logic: find a jump > 3x the 95th percentile jump
                    jump_threshold = np.percentile(area_diffs, 95)
                    significant_jump_thresh = max(
                        10, jump_threshold * 3
                    )  # Add a 10px minimum jump

                    jump_indices = np.where(area_diffs > significant_jump_thresh)[0]

                    if len(jump_indices) > 0:
                        has_clear_gap = True
                        # This is the index of the *last noise component*
                        gap_idx = jump_indices[0]
                        area_before_gap = sorted_areas[gap_idx]

                        # The aggressive threshold is 1 pixel *larger* than the biggest noise component
                        aggressive_threshold = area_before_gap + 1

            # --- 3. ADAPTIVE DECISION: Override if conservative threshold is clearly noise ---
            if has_clear_gap:
                # print(
                #     f"Noise Removal: Gap detected. Noise cluster ends at {area_before_gap}px. Aggressive threshold = {aggressive_threshold:.1f}"
                # )

                # Only use a more aggressive threshold IF our "safe" threshold is clearly
                # stuck *inside* the noise cluster.
                # e.g., Safe threshold = 1, but noise goes up to 10.
                # (We use 0.8 as a buffer, so if thresh=7 and gap=8, we don't switch)
                if area_threshold < (area_before_gap * 0.8):
                    # print(
                    #     f"Noise Removal: Conservative threshold ({area_threshold:.1f}) is deep in noise cluster (ends at {area_before_gap}px)."
                    # )

                    # Instead of using large percentage increases, use a very small absolute increment
                    # This preserves legitimate small letters/words that might be just above the noise
                    # Use a minimal fixed offset (2-3 pixels) above the noise cluster end
                    # This ensures we only remove noise, not legitimate small components
                    small_increment = (
                        2  # Fixed small increment - just 2 pixels above noise
                    )

                    moderate_threshold = area_before_gap + small_increment

                    # Also check what the actual first component after the gap is
                    # This gives us insight into where real text starts
                    # If the gap is very large (e.g., noise ends at 229, text starts at 500),
                    # we want to use a threshold closer to the noise end, not the text start
                    if gap_idx + 1 < len(sorted_areas):
                        first_after_gap = sorted_areas[gap_idx + 1]
                        gap_size = first_after_gap - area_before_gap

                        # If there's a large gap, stick close to the noise end (2 pixels above)
                        # If the gap is small, we might be cutting into text, so be even more conservative
                        if gap_size > 50:  # Large gap - safe to use noise_end + 2
                            final_threshold = moderate_threshold
                        else:  # Small gap - might be cutting into text, use just 1 pixel above noise
                            final_threshold = area_before_gap + 1
                    else:
                        final_threshold = moderate_threshold

                    # Ensure we're at least 1 pixel above the noise cluster
                    final_threshold = max(final_threshold, area_before_gap + 1)

                    # Cap at aggressive threshold as absolute upper bound (shouldn't be needed)
                    final_threshold = min(final_threshold, aggressive_threshold)

                    # Cap at 15 pixels as absolute upper bound
                    final_threshold = min(final_threshold, 15)

                    # print(
                    #     f"Noise Removal: Using MODERATE threshold: {final_threshold:.1f} (noise ends at {area_before_gap}px, increment: {small_increment}px)"
                    # )
                    area_threshold = final_threshold
                else:
                    # print(
                    #     f"Noise Removal: Gap found, but conservative threshold ({area_threshold:.1f}) is sufficient. Sticking with conservative."
                    # )
                    pass

            # --- 4. Apply the final, determined threshold ---
            # print(f"Noise Removal: Final area threshold: {area_threshold:.1f}")
            for i in range(1, num_labels):
                # Use >= to be inclusive of the threshold itself
                if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
                    clean_binary[labels == i] = 255
        else:
            # No components found, or only background
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

        # Calculate the horizontal projection profile on the cleaned image
        horizontal_projection = np.sum(clean_binary, axis=1)

        # Track y_start offset for coordinate system conversion
        y_start = 0  # Default: no vertical cropping

        # Find the top and bottom boundaries of the text
        non_zero_rows = np.where(horizontal_projection > 0)[0]
        if len(non_zero_rows) > 0:
            text_top = non_zero_rows[0]
            text_bottom = non_zero_rows[-1]
            text_height = text_bottom - text_top

            # Define a percentage to trim off the top and bottom
            # This is a tunable parameter. 15% is a good starting point.
            trim_percentage = DEFAULT_TRIM_PERCENTAGE
            trim_pixels = int(text_height * trim_percentage)

            # Calculate new, tighter boundaries
            y_start = text_top + trim_pixels
            y_end = text_bottom - trim_pixels

            # Ensure the crop is valid
            if y_start < y_end:
                # print(
                #     f"Original text height: {text_height}px. Cropping to middle {100 - (2*trim_percentage*100):.0f}% region."
                # )
                # Slice the image to get the vertically cropped ROI
                analysis_image = clean_binary[y_start:y_end, :]
            else:
                # If trimming would result in an empty image, use the full text region
                y_start = text_top
                analysis_image = clean_binary[text_top:text_bottom, :]
        else:
            # If no text is found, use the original cleaned image
            analysis_image = clean_binary

        # Validate analysis_image before proceeding
        if (
            analysis_image is None
            or not isinstance(analysis_image, np.ndarray)
            or analysis_image.size == 0
        ):
            # print(
            #     f"Error: analysis_image is None or empty (image_name: {image_name})"
            # )
            return ({}, False)

        # --- Step 3: Hierarchical Adaptive Search (using the new clean_binary) ---
        # The rest of the pipeline is identical but now operates on a superior image.
        words = line_data["text"][0].split()
        target_word_count = len(words)

        # print(f"Target word count: {target_word_count}")

        # Save cropped image (optional, only if image_name is provided)
        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f"{self.output_folder}/word_segmentation/{image_name}_{shortened_line_text}_clean_binary.png"
            os.makedirs(f"{self.output_folder}/word_segmentation", exist_ok=True)
            cv2.imwrite(output_path, analysis_image)
            # print(f"\nSaved cropped image to '{output_path}'")

        best_boxes = None
        successful_binary_image = None

        # --- Step 3: Hierarchical Adaptive Search (using the CROPPED analysis_image) ---
        words = line_data["text"][0].split()
        target_word_count = len(words)
        stage1_succeeded = False

        # print("--- Stage 1: Searching with adaptive valley threshold ---")
        valley_factors_to_try = np.arange(INITIAL_VALLEY_THRESHOLD_FACTOR, 0.45, 0.05)
        for v_factor in valley_factors_to_try:
            # Pass the cropped image to the helper
            unlabeled_boxes = self._get_boxes_from_profile(
                analysis_image, avg_char_width_approx, min_space_factor, v_factor
            )
            # ... (The rest of the Stage 1 loop is the same)
            if abs(target_word_count - len(unlabeled_boxes)) <= match_tolerance:
                best_boxes = unlabeled_boxes
                successful_binary_image = analysis_image
                stage1_succeeded = True
                break

        if not stage1_succeeded:
            # print(
            #     "\n--- Stage 1 failed. Starting Stage 2: Searching with adaptive kernel ---"
            # )
            kernel_factors_to_try = np.arange(INITIAL_KERNEL_WIDTH_FACTOR, 0.5, 0.05)
            fixed_valley_factor = MAIN_VALLEY_THRESHOLD_FACTOR
            for k_factor in kernel_factors_to_try:
                kernel_width = max(1, int(avg_char_width_approx * k_factor))
                closing_kernel = np.ones((1, kernel_width), np.uint8)
                # Apply closing on the original clean_binary, then crop it
                closed_binary = cv2.morphologyEx(
                    clean_binary, cv2.MORPH_CLOSE, closing_kernel
                )
                # Validate closed_binary before proceeding
                if (
                    closed_binary is None
                    or not isinstance(closed_binary, np.ndarray)
                    or closed_binary.size == 0
                ):
                    # print(
                    #     f"Error: closed_binary in Stage 2 is None or empty (image_name: {image_name}, k_factor: {k_factor:.2f})"
                    # )
                    continue  # Skip this iteration and try next kernel factor

                # We need to re-apply the same vertical crop to this new image
                if len(non_zero_rows) > 0 and y_start < y_end:
                    analysis_image = closed_binary[y_start:y_end, :]
                else:
                    analysis_image = closed_binary

                # Validate analysis_image before using it
                if (
                    analysis_image is None
                    or not isinstance(analysis_image, np.ndarray)
                    or analysis_image.size == 0
                ):
                    # print(
                    #     f"Error: analysis_image in Stage 2 is None or empty (image_name: {image_name}, k_factor: {k_factor:.2f})"
                    # )
                    continue  # Skip this iteration and try next kernel factor

                unlabeled_boxes = self._get_boxes_from_profile(
                    analysis_image,
                    avg_char_width_approx,
                    min_space_factor,
                    fixed_valley_factor,
                )

                # print(
                #     f"Testing kernel factor {k_factor:.2f} ({kernel_width}px): Found {len(unlabeled_boxes)} boxes."
                # )
                if abs(target_word_count - len(unlabeled_boxes)) <= match_tolerance:
                    # print("SUCCESS (Stage 2): Found a match.")
                    best_boxes = unlabeled_boxes
                    successful_binary_image = (
                        closed_binary  # For Stage 2, the source is the closed_binary
                    )
                    break

        final_output = None
        used_fallback = False

        if best_boxes is None:
            # print("\nWarning: All adaptive searches failed. Falling back.")
            fallback_segmenter = HybridWordSegmenter()
            used_fallback = True
            final_output = fallback_segmenter.refine_words_bidirectional(
                line_data, deskewed_line_image
            )

        else:
            # --- CCA Refinement using the determined successful_binary_image ---
            unlabeled_boxes = best_boxes
            cca_source_image = successful_binary_image

            if (
                successful_binary_image is analysis_image
            ):  # This comparison might not work as intended
                # A safer way is to check if Stage 1 succeeded
                if any(
                    v_factor in locals()
                    and abs(
                        target_word_count
                        - len(
                            self._get_boxes_from_profile(
                                analysis_image,
                                avg_char_width_approx,
                                min_space_factor,
                                v_factor,
                            )
                        )
                    )
                    <= match_tolerance
                    for v_factor in np.arange(
                        INITIAL_VALLEY_THRESHOLD_FACTOR, 0.45, 0.05
                    )
                ):
                    cca_source_image = clean_binary
                else:  # Stage 2 must have succeeded
                    # Recreate the successful closed_binary for CCA
                    successful_k_factor = locals().get("k_factor")
                    if successful_k_factor is not None:
                        kernel_width = max(
                            1, int(avg_char_width_approx * successful_k_factor)
                        )
                        closing_kernel = np.ones((1, kernel_width), np.uint8)
                        cca_source_image = cv2.morphologyEx(
                            clean_binary, cv2.MORPH_CLOSE, closing_kernel
                        )
                    else:
                        cca_source_image = clean_binary  # Fallback
            else:
                cca_source_image = successful_binary_image

            # --- Proceed with CCA Refinement ---
            unlabeled_boxes = best_boxes
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                cca_source_image, 8, cv2.CV_32S
            )

            num_to_process = min(len(words), len(unlabeled_boxes))

            # Get the height of cca_source_image to filter components outside text region
            cca_img_h, cca_img_w = cca_source_image.shape[:2]

            # First pass: assign each component to the box it overlaps with most
            # This prevents components from being assigned to multiple boxes, which causes overlapping words
            # We use component center point as the primary criterion, with overlap as a tie-breaker
            component_assignments = {}  # component_index -> box_index
            for j in range(1, num_labels):  # Skip background
                comp_x = stats[j, cv2.CC_STAT_LEFT]
                comp_w = stats[j, cv2.CC_STAT_WIDTH]
                comp_r = comp_x + comp_w
                comp_center_x = comp_x + comp_w / 2  # Component center x coordinate
                comp_y = stats[j, cv2.CC_STAT_TOP]
                comp_h = stats[j, cv2.CC_STAT_HEIGHT]

                # Filter out components that are clearly outside the text region
                # Components should be roughly in the middle 80% of the image height
                # (accounting for vertical cropping that was done)
                comp_center_y = comp_y + comp_h / 2
                if comp_center_y < cca_img_h * 0.1 or comp_center_y > cca_img_h * 0.9:
                    continue  # Skip components too far from text region

                best_box_idx = None
                max_overlap = 0
                best_center_distance = float(
                    "inf"
                )  # Distance from component center to box center
                component_center_in_box = (
                    False  # Track if component center is within any box
                )

                for i in range(num_to_process):
                    box_x, box_y, box_w, box_h = unlabeled_boxes[i]
                    box_r = box_x + box_w
                    box_center_x = box_x + box_w / 2  # Box center x coordinate

                    # Heuristic: If component is more than 1.5x the width of the box it's matching,
                    # it's likely a merged blob spanning multiple words. Do NOT assign it.
                    if comp_w > box_w * 1.5:
                        continue

                    # Check if component overlaps with this box horizontally
                    # Note: unlabeled_boxes are in analysis_image coordinate system (x matches, y=0)
                    # but cca_source_image might be full image, so we only check x overlap
                    if comp_x < box_r and box_x < comp_r:
                        # Calculate horizontal overlap amount
                        overlap_start = max(comp_x, box_x)
                        overlap_end = min(comp_r, box_r)
                        overlap = overlap_end - overlap_start

                        # Only consider boxes with actual overlap (not just touching)
                        if overlap > 0:
                            # Check if component center falls within this box's boundaries
                            center_in_box = box_x <= comp_center_x < box_r

                            # Calculate distance from component center to box center
                            center_distance = abs(comp_center_x - box_center_x)

                            # Priority 1: If component center is within box boundaries, prefer it
                            # Priority 2: If no box contains the center, use closest center distance
                            if center_in_box:
                                # Component center is within this box - this is the best match
                                if not component_center_in_box or overlap > max_overlap:
                                    component_center_in_box = True
                                    best_center_distance = center_distance
                                    max_overlap = overlap
                                    best_box_idx = i
                            elif not component_center_in_box:
                                # Component center not in any box yet - use closest center distance
                                if center_distance < best_center_distance or (
                                    center_distance == best_center_distance
                                    and overlap > max_overlap
                                ):
                                    best_center_distance = center_distance
                                    max_overlap = overlap
                                    best_box_idx = i

                if best_box_idx is not None:
                    component_assignments[j] = best_box_idx

            # Second pass: build refined boxes from assigned components
            refined_boxes_list = []
            for i in range(num_to_process):
                word_label = words[i]

                # Find all components assigned to this box
                components_in_box = []
                for j, box_idx in component_assignments.items():
                    if box_idx == i:
                        components_in_box.append(stats[j])

                if not components_in_box:
                    # Fallback: use the original box if no components assigned
                    # Adjust y coordinate from analysis_image space to cca_source_image space
                    box_x, box_y, box_w, box_h = unlabeled_boxes[i]
                    # unlabeled_boxes have y=0 relative to analysis_image (vertically cropped)
                    # cca_source_image is always the full image (clean_binary or closed_binary)
                    # So we need to adjust y coordinate by adding y_start offset
                    adjusted_box_y = (
                        y_start + box_y
                    )  # box_y is typically 0, but add offset for safety
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
                    continue

                # Calculate bounding box from assigned components
                # Components are already in cca_source_image coordinate system
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

                # Validate dimensions
                box_width = max(1, max_r - min_x)
                box_height = max(1, max_b - min_y)

                refined_boxes_list.append(
                    {
                        "text": word_label,
                        "left": min_x,
                        "top": min_y,
                        "width": box_width,
                        "height": box_height,
                        "conf": line_data["conf"][0],
                    }
                )

            # Convert to dict format
            final_output = {
                k: [] for k in ["text", "left", "top", "width", "height", "conf"]
            }
            for box in refined_boxes_list:
                for key in final_output.keys():
                    final_output[key].append(box[key])

        # --- TRANSFORM COORDINATES BACK ---

        # Get the inverse transformation matrix
        M_inv = cv2.invertAffineTransform(M)

        # Create a new list for the re-mapped boxes
        remapped_boxes_list = []

        # Iterate through the boxes found on the deskewed image
        for i in range(len(final_output["text"])):
            # Get the box coordinates from the deskewed image
            left, top = final_output["left"][i], final_output["top"][i]
            width, height = final_output["width"][i], final_output["height"][i]

            # Define the 4 corners of this box
            # Use float for accurate transformation
            corners = np.array(
                [
                    [left, top],
                    [left + width, top],
                    [left + width, top + height],
                    [left, top + height],
                ],
                dtype="float32",
            )

            # Add a '1' to each coordinate for the 2x3 affine matrix
            # shape (4, 1, 2)
            corners_expanded = np.expand_dims(corners, axis=1)

            # Apply the inverse transformation
            # shape (4, 1, 2)
            original_corners = cv2.transform(corners_expanded, M_inv)

            # Find the new axis-aligned bounding box in the original image
            # original_corners is now [[ [x1,y1] ], [ [x2,y2] ], ...]
            # We need to squeeze it to get [ [x1,y1], [x2,y2], ...]
            squeezed_corners = original_corners.squeeze(axis=1)

            # Find the min/max x and y
            min_x = int(np.min(squeezed_corners[:, 0]))
            max_x = int(np.max(squeezed_corners[:, 0]))
            min_y = int(np.min(squeezed_corners[:, 1]))
            max_y = int(np.max(squeezed_corners[:, 1]))

            # Create the re-mapped box
            remapped_box = {
                "text": final_output["text"][i],
                "left": min_x,
                "top": min_y,
                "width": max_x - min_x,
                "height": max_y - min_y,
                "conf": final_output["conf"][i],
            }
            remapped_boxes_list.append(remapped_box)

        # Convert the remapped list back to the dictionary format
        remapped_output = {k: [] for k in final_output.keys()}
        for box in remapped_boxes_list:
            for key in remapped_output.keys():
                remapped_output[key].append(box[key])

        # Apply Final Logical Constraint Checks
        img_h, img_w = line_image.shape[:2]
        remapped_output = self._enforce_logical_constraints(
            remapped_output, img_w, img_h
        )

        # Visualisation
        if SAVE_WORD_SEGMENTER_OUTPUT_IMAGES:
            output_path = f"{self.output_folder}/word_segmentation/{image_name}_{shortened_line_text}_final_boxes.png"
            os.makedirs(f"{self.output_folder}/word_segmentation", exist_ok=True)
            output_image_vis = line_image.copy()
            # Validate output_image_vis before saving
            if (
                output_image_vis is None
                or not isinstance(output_image_vis, np.ndarray)
                or output_image_vis.size == 0
            ):
                pass
                # print(
                #     f"Error: output_image_vis is None or empty (image_name: {image_name})"
                # )
            else:
                # print(f"\nFinal refined {len(remapped_output['text'])} words:")
                for i in range(len(remapped_output["text"])):
                    word = remapped_output["text"][i]
                    x, y, w, h = (
                        int(remapped_output["left"][i]),
                        int(remapped_output["top"][i]),
                        int(remapped_output["width"][i]),
                        int(remapped_output["height"][i]),
                    )
                    # print(f"- '{word}' at ({x}, {y}, {w}, {h})")
                    cv2.rectangle(
                        output_image_vis, (x, y), (x + w, y + h), (0, 255, 0), 2
                    )
                cv2.imwrite(output_path, output_image_vis)
                # print(f"\nSaved visualisation to '{output_path}'")

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
        """Helper function to run one pass of refinement (either LTR or RTL)."""

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

            # Bounded Scan (logic is the same for both directions)
            if right < img_w and vertical_projection[right] > 0:
                scan_limit = min(img_w, right + max_scan_distance)
                for x in range(right + 1, scan_limit):
                    if vertical_projection[x] == 0:
                        new_right = x
                        break

            if left > 0 and vertical_projection[left] > 0:
                scan_limit = max(0, left - max_scan_distance)
                for x in range(left - 1, scan_limit, -1):
                    if vertical_projection[x] == 0:
                        new_left = x
                        break

            # Directional De-overlapping
            if direction == "ltr":
                if new_left < last_corrected_right_edge:
                    new_left = last_corrected_right_edge
                last_corrected_right_edge = max(last_corrected_right_edge, new_right)
            else:  # rtl
                if new_right > next_corrected_left_edge:
                    new_right = next_corrected_left_edge
                next_corrected_left_edge = min(next_corrected_left_edge, new_left)

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

        # 1. & 2. Perform both passes
        ltr_boxes = self._run_single_pass(
            initial_boxes, vertical_projection, max_scan_distance, img_w, "ltr"
        )
        rtl_boxes = self._run_single_pass(
            initial_boxes, vertical_projection, max_scan_distance, img_w, "rtl"
        )

        # 3. Combine the results by taking the best edge from each pass
        combined_boxes = [box.copy() for box in initial_boxes]
        for i in range(len(combined_boxes)):

            # Get the "expert" left boundary from the LTR pass
            final_left = ltr_boxes[i]["left"]

            # Get the "expert" right boundary from the RTL pass
            rtl_right = rtl_boxes[i]["left"] + rtl_boxes[i]["width"]

            combined_boxes[i]["left"] = final_left
            combined_boxes[i]["width"] = max(1, rtl_right - final_left)

        # 4. Final De-overlap Pass
        last_corrected_right_edge = 0
        for i, box in enumerate(combined_boxes):
            if box["left"] < last_corrected_right_edge:
                box["width"] = max(
                    1, box["width"] - (last_corrected_right_edge - box["left"])
                )
                box["left"] = last_corrected_right_edge

            if box["width"] < 1:
                # Handle edge case where a box is completely eliminated
                if i < len(combined_boxes) - 1:
                    next_left = combined_boxes[i + 1]["left"]
                    box["width"] = max(1, next_left - box["left"])
                else:
                    box["width"] = 1

            last_corrected_right_edge = box["left"] + box["width"]

        # Convert back to Tesseract-style output dict
        final_output = {k: [] for k in estimated_data.keys()}
        for box in combined_boxes:
            if box["width"] > 0:  # Ensure we don't add zero-width boxes
                for key in final_output.keys():
                    final_output[key].append(box[key])

        return final_output
