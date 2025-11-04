import cv2
import numpy as np
from typing import Dict, List, Tuple
import os
from tools.config import OUTPUT_FOLDER

INITIAL_KERNEL_WIDTH_FACTOR = 0.05 # Default 0.05
INITIAL_VALLEY_THRESHOLD_FACTOR = 0.05 # Default 0.05
MAIN_VALLEY_THRESHOLD_FACTOR = 0.15 # Default 0.15
C_VALUE = 4 # Default 4
BLOCK_SIZE_FACTOR = 1.5 # Default 1.5
MIN_SPACE_FACTOR = 0.3 # Default 0.4
MATCH_TOLERANCE = 0 # Default 0
MIN_AREA_THRESHOLD = 6 # Default 6
DEFAULT_TRIM_PERCENTAGE = 0.2 # Default 0.2
SHOW_OUTPUT_IMAGES = True # Default False

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

    def _correct_orientation(self, gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
                gray_image, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size, 4
            )
        else:
            _, binary = cv2.threshold(
                gray_image, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

        # Small noise removal
        opening_kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, opening_kernel)

        # --- Extract text pixel coordinates ---
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 50:
            print("Warning: Not enough text pixels for orientation. Assuming horizontal.")
            M_orient = cv2.getRotationMatrix2D(center, 0, 1.0)
            return gray_image, M_orient

        # --- Robust bounding-box check (no minAreaRect quirks) ---
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        box_height = ymax - ymin
        box_width = xmax - xmin

        orientation_angle = 0.0
        if box_height > box_width:
            print(f"Detected vertical orientation (W:{box_width} < H:{box_height}). Applying 90-degree correction.")
            orientation_angle = 90.0
        else:
            print(f"Detected horizontal orientation (W:{box_width} >= H:{box_height}). No orientation correction.")
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
            gray_image, M_orient, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
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
            binary = cv2.adaptiveThreshold(gray_image, 255, 
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, block_size, 4)
        else:
            _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        opening_kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, opening_kernel)
        
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) < 50:
            #print("Warning: Not enough text pixels to detect skew. Skipping.")
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
        
        #print(f"Normalized shape (W:{rect_width:.0f}, H:{rect_height:.0f}). Detected angle: {correction_angle:.2f} degrees.")
        
        # Final sanity checks on the angle
        MIN_SKEW_THRESHOLD = 0.5  # Ignore angles smaller than this (likely noise)
        MAX_SKEW_THRESHOLD = 15.0  # Angles larger than this are extreme and likely errors
        
        if abs(correction_angle) < MIN_SKEW_THRESHOLD:
            #print(f"Detected angle {correction_angle:.2f}° is too small (likely noise). Skipping deskew.")
            correction_angle = 0.0
        elif abs(correction_angle) > MAX_SKEW_THRESHOLD:
            #print(f"Warning: Corrected angle {correction_angle:.2f}° is extreme. Skipping deskew.")
            correction_angle = 0.0

        # Create rotation matrix and apply the final correction
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
        
        deskewed_gray = cv2.warpAffine(gray_image, M, (w, h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
        
        return deskewed_gray, M

    def _get_boxes_from_profile(self, binary_image: np.ndarray, stable_avg_char_width: float, min_space_factor: float, valley_threshold_factor: float) -> List:
        # This helper function remains IDENTICAL. No changes needed.
        # ... (code from the previous version)
        img_h, img_w = binary_image.shape
        vertical_projection = np.sum(binary_image, axis=0)
        peaks = vertical_projection[vertical_projection > 0]
        if len(peaks) == 0: return []
        avg_peak_height = np.mean(peaks)
        valley_threshold = int(avg_peak_height * valley_threshold_factor)
        min_space_width = int(stable_avg_char_width * min_space_factor)
        patched_projection = vertical_projection.copy()
        in_gap = False; gap_start = 0
        for x, col_sum in enumerate(patched_projection):
            if col_sum <= valley_threshold and not in_gap: in_gap = True; gap_start = x
            elif col_sum > valley_threshold and in_gap:
                in_gap = False
                if (x - gap_start) < min_space_width: patched_projection[gap_start:x] = int(avg_peak_height)
        unlabeled_boxes = []
        in_word = False; start_x = 0
        for x, col_sum in enumerate(patched_projection):
            if col_sum > valley_threshold and not in_word: start_x = x; in_word = True
            elif col_sum <= valley_threshold and in_word: unlabeled_boxes.append((start_x, 0, x - start_x, img_h)); in_word = False
        if in_word: unlabeled_boxes.append((start_x, 0, img_w - start_x, img_h))
        return unlabeled_boxes

    def segment(self, line_data: Dict[str, List], line_image: np.ndarray, min_space_factor=MIN_SPACE_FACTOR, match_tolerance=MATCH_TOLERANCE, image_name: str = None) -> Tuple[Dict[str, List], bool]:
        
        if line_image is None:
            print(f"Error: line_image is None in segment function (image_name: {image_name})")
            return ({}, False)
        
        # Validate line_image is a valid numpy array
        if not isinstance(line_image, np.ndarray):
            print(f"Error: line_image is not a numpy array (type: {type(line_image)}, image_name: {image_name})")
            return ({}, False)
        
        # Validate line_image has valid shape and size
        if line_image.size == 0:
            print(f"Error: line_image is empty (shape: {line_image.shape}, image_name: {image_name})")
            return ({}, False)
        
        if len(line_image.shape) < 2:
            print(f"Error: line_image has invalid shape {line_image.shape} (image_name: {image_name})")
            return ({}, False)
        
        # Early return if 1 or fewer words
        if line_data and line_data.get("text") and len(line_data["text"]) > 0:
            line_text = line_data["text"][0]
            words = line_text.split()
            if len(words) <= 1:
                return ({}, False)
        else:
            print(f"Error: line_data is empty or does not contain text (image_name: {image_name})")
            return ({}, False)

        
        print(f"line_text: {line_text}")
        shortened_line_text = line_text.replace(" ", "_")[:10]

        if SHOW_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f'{self.output_folder}/paddle_visualisations/{image_name}_{shortened_line_text}_original.png'
            os.makedirs(f'{self.output_folder}/paddle_visualisations', exist_ok=True)
            cv2.imwrite(output_path, line_image)
            print(f"\nSaved original image to '{output_path}'")


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
        
        deskewed_line_image = cv2.warpAffine(line_image, M, (w, h),
                                             flags=cv2.INTER_CUBIC,
                                             borderMode=cv2.BORDER_REPLICATE)


        # Save deskewed image (optional, only if image_name is provided)
        if SHOW_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f'{self.output_folder}/paddle_visualisations/{image_name}_{shortened_line_text}_deskewed.png'
            os.makedirs(f'{self.output_folder}/paddle_visualisations', exist_ok=True)
            cv2.imwrite(output_path, deskewed_line_image)
            #print(f"\nSaved deskewed image to '{output_path}'")

        # --- Step 1: Binarization and Stable Width Calculation (Unchanged) ---
        approx_char_count = len(line_data["text"][0].replace(" ", ""))
        if approx_char_count == 0: return {}, False
        img_h, img_w = deskewed_gray.shape
        avg_char_width_approx = img_w / approx_char_count
        block_size = int(avg_char_width_approx * BLOCK_SIZE_FACTOR)
        if block_size % 2 == 0: block_size += 1
        binary = cv2.adaptiveThreshold(deskewed_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C_VALUE)

        # Save cropped image (optional, only if image_name is provided)
        if SHOW_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f'{self.output_folder}/paddle_visualisations/{image_name}_{shortened_line_text}_binary.png'
            os.makedirs(f'{self.output_folder}/paddle_visualisations', exist_ok=True)
            cv2.imwrite(output_path, binary)
            #print(f"\nSaved cropped image to '{output_path}'")

        # --- NEW STEP 1.5: Post-processing with Morphology ---
        # This "closes" gaps in letters and joins nearby components.

        # Create a small kernel (e.g., 3x3 rectangle)
        # You may need to tune this size.
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Use MORPH_CLOSE to close small holes and gaps within the letters
        # It's a dilation followed by an erosion
        closed_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # (Optional) You could also use a DILATE to make letters thicker
        #dilated_binary = cv2.dilate(closed_binary, kernel, iterations=1)
        # Use 'closed_binary' (or 'dilated_binary') from now on.

        if SHOW_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f'{self.output_folder}/paddle_visualisations/{image_name}_{shortened_line_text}_closed_binary.png'
            os.makedirs(f'{self.output_folder}/paddle_visualisations', exist_ok=True)
            cv2.imwrite(output_path, closed_binary)
            #print(f"\nSaved dilated binary image to '{output_path}'")
        
        # --- Step 2: Intelligent Noise Removal (Improved) ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_binary, 8, cv2.CV_32S)
        clean_binary = np.zeros_like(binary)
        
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA] # Get all component areas, skip background (label 0)
            
            # Handle edge case of empty 'areas' array
            if len(areas) == 0:
                clean_binary = binary
                print("Warning: No components found after binarization.")
                areas = np.array([0]) # Add a dummy value to prevent crashes
            
            # --- 1. Calculate the DEFAULT CONSERVATIVE threshold ---
            # This is your existing logic, which works well for *clean* lines.
            p1 = np.percentile(areas, 1)
            img_h, img_w = binary.shape
            estimated_char_height = img_h * 0.7
            estimated_min_letter_area = max(2, int(estimated_char_height * 0.2 * estimated_char_height * 0.15))
            
            # This is the "safe" threshold that protects small letters on clean lines.
            area_threshold = max(MIN_AREA_THRESHOLD, min(p1, estimated_min_letter_area))
            #print(f"Noise Removal: Initial conservative threshold: {area_threshold:.1f} (p1={p1:.1f}, est_min={estimated_min_letter_area:.1f})")

            # --- 2. Find a "Noise-to-Text" Gap (to enable AGGRESSIVE mode) ---
            sorted_areas = np.sort(areas)
            has_clear_gap = False
            aggressive_threshold = -1
            area_before_gap = -1

            if len(sorted_areas) > 10: # Need enough components to analyze
                area_diffs = np.diff(sorted_areas)
                if len(area_diffs) > 0:
                    # Use your "gap" logic: find a jump > 3x the 95th percentile jump
                    jump_threshold = np.percentile(area_diffs, 95)
                    significant_jump_thresh = max(10, jump_threshold * 3) # Add a 10px minimum jump
                    
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
                print(f"Noise Removal: Gap detected. Noise cluster ends at {area_before_gap}px. Aggressive threshold = {aggressive_threshold:.1f}")

                # Only use a more aggressive threshold IF our "safe" threshold is clearly
                # stuck *inside* the noise cluster.
                # e.g., Safe threshold = 1, but noise goes up to 10.
                # (We use 0.8 as a buffer, so if thresh=7 and gap=8, we don't switch)
                if area_threshold < (area_before_gap * 0.8):
                    print(f"Noise Removal: Conservative threshold ({area_threshold:.1f}) is deep in noise cluster (ends at {area_before_gap}px).")
                    
                    # Instead of using large percentage increases, use a very small absolute increment
                    # This preserves legitimate small letters/words that might be just above the noise
                    # Use a minimal fixed offset (2-3 pixels) above the noise cluster end
                    # This ensures we only remove noise, not legitimate small components
                    small_increment = 2  # Fixed small increment - just 2 pixels above noise
                    
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
                    
                    print(f"Noise Removal: Using MODERATE threshold: {final_threshold:.1f} (noise ends at {area_before_gap}px, increment: {small_increment}px)")
                    area_threshold = final_threshold
                else:
                    print(f"Noise Removal: Gap found, but conservative threshold ({area_threshold:.1f}) is sufficient. Sticking with conservative.")
                    pass

            # --- 4. Apply the final, determined threshold ---
            print(f"Noise Removal: Final area threshold: {area_threshold:.1f}")
            for i in range(1, num_labels):
                # Use >= to be inclusive of the threshold itself
                if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
                    clean_binary[labels == i] = 255
        else:
            # No components found, or only background
            clean_binary = binary

        # Calculate the horizontal projection profile on the cleaned image
        horizontal_projection = np.sum(clean_binary, axis=1)
        
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
                print(f"Original text height: {text_height}px. Cropping to middle {100 - (2*trim_percentage*100):.0f}% region.")
                # Slice the image to get the vertically cropped ROI
                analysis_image = clean_binary[y_start:y_end, :]
            else:
                # If trimming would result in an empty image, use the full text region
                analysis_image = clean_binary[text_top:text_bottom, :]
        else:
            # If no text is found, use the original cleaned image
            analysis_image = clean_binary
        
        # --- Step 3: Hierarchical Adaptive Search (using the new clean_binary) ---
        # The rest of the pipeline is identical but now operates on a superior image.
        words = line_data["text"][0].split()
        target_word_count = len(words)

        print(f"Target word count: {target_word_count}")

        # Save cropped image (optional, only if image_name is provided)
        if SHOW_OUTPUT_IMAGES:
            os.makedirs(self.output_folder, exist_ok=True)
            output_path = f'{self.output_folder}/paddle_visualisations/{image_name}_{shortened_line_text}_clean_binary.png'
            os.makedirs(f'{self.output_folder}/paddle_visualisations', exist_ok=True)
            cv2.imwrite(output_path, analysis_image)
            #print(f"\nSaved cropped image to '{output_path}'")

        best_boxes = None
        successful_binary_image = None

       # --- Step 3: Hierarchical Adaptive Search (using the CROPPED analysis_image) ---
        words = line_data["text"][0].split()
        target_word_count = len(words)
        stage1_succeeded = False

        print("--- Stage 1: Searching with adaptive valley threshold ---")
        valley_factors_to_try = np.arange(INITIAL_VALLEY_THRESHOLD_FACTOR, 0.45, 0.05)
        for v_factor in valley_factors_to_try:
            # Pass the cropped image to the helper
            unlabeled_boxes = self._get_boxes_from_profile(analysis_image, avg_char_width_approx, min_space_factor, v_factor)
            # ... (The rest of the Stage 1 loop is the same)
            if abs(target_word_count - len(unlabeled_boxes)) <= match_tolerance:
                 best_boxes = unlabeled_boxes
                 successful_binary_image = analysis_image
                 stage1_succeeded = True
                 break
        
        if not stage1_succeeded:
            print("\n--- Stage 1 failed. Starting Stage 2: Searching with adaptive kernel ---")
            kernel_factors_to_try = np.arange(INITIAL_KERNEL_WIDTH_FACTOR, 0.5, 0.05)
            fixed_valley_factor = MAIN_VALLEY_THRESHOLD_FACTOR
            for k_factor in kernel_factors_to_try:
                kernel_width = max(1, int(avg_char_width_approx * k_factor))
                closing_kernel = np.ones((1, kernel_width), np.uint8)
                # Apply closing on the original clean_binary, then crop it
                closed_binary = cv2.morphologyEx(clean_binary, cv2.MORPH_CLOSE, closing_kernel)
                # We need to re-apply the same vertical crop to this new image
                if len(non_zero_rows) > 0 and y_start < y_end:
                    analysis_image = closed_binary[y_start:y_end, :]
                else:
                    analysis_image = closed_binary
                
                unlabeled_boxes = self._get_boxes_from_profile(analysis_image, avg_char_width_approx, min_space_factor, fixed_valley_factor)


                print(f"Testing kernel factor {k_factor:.2f} ({kernel_width}px): Found {len(unlabeled_boxes)} boxes.")
                if abs(target_word_count - len(unlabeled_boxes)) <= match_tolerance:
                    print(f"SUCCESS (Stage 2): Found a match.")
                    best_boxes = unlabeled_boxes
                    successful_binary_image = closed_binary # For Stage 2, the source is the closed_binary
                    break        
        
        final_output = None
        used_fallback = False

        if best_boxes is None:
            print(f"\nWarning: All adaptive searches failed. Falling back.")
            fallback_segmenter = HybridWordSegmenter()
            used_fallback = True
            final_output = fallback_segmenter.refine_words_bidirectional(line_data, deskewed_line_image)

        else:
            # --- CCA Refinement using the determined successful_binary_image ---
            unlabeled_boxes = best_boxes
            cca_source_image = successful_binary_image

            if successful_binary_image is analysis_image: # This comparison might not work as intended
                # A safer way is to check if Stage 1 succeeded
                if any(v_factor in locals() and abs(target_word_count - len(self._get_boxes_from_profile(analysis_image, avg_char_width_approx, min_space_factor, v_factor))) <= match_tolerance for v_factor in np.arange(INITIAL_VALLEY_THRESHOLD_FACTOR, 0.45, 0.05)):
                    cca_source_image = clean_binary
                else: # Stage 2 must have succeeded
                    # Recreate the successful closed_binary for CCA
                    successful_k_factor = locals().get('k_factor')
                    if successful_k_factor is not None:
                        kernel_width = max(1, int(avg_char_width_approx * successful_k_factor))
                        closing_kernel = np.ones((1, kernel_width), np.uint8)
                        cca_source_image = cv2.morphologyEx(clean_binary, cv2.MORPH_CLOSE, closing_kernel)
                    else:
                        cca_source_image = clean_binary # Fallback
            else:
                cca_source_image = successful_binary_image
            
            # --- Proceed with CCA Refinement ---
            unlabeled_boxes = best_boxes
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cca_source_image, 8, cv2.CV_32S)
            
            refined_boxes_list = []
            num_to_process = min(len(words), len(unlabeled_boxes))
            for i in range(num_to_process):
                word_label = words[i]
                box_x, _, box_w, _ = unlabeled_boxes[i]
                box_r = box_x + box_w # Box right edge
                
                components_in_box = []
                for j in range(1, num_labels): # Skip background
                    comp_x = stats[j, cv2.CC_STAT_LEFT]
                    comp_w = stats[j, cv2.CC_STAT_WIDTH]
                    comp_r = comp_x + comp_w # Component right edge

                    # --- THE CRITICAL FIX: Check for OVERLAP, not strict containment ---
                    # Old logic: if box_x <= comp_x < box_r:
                    # New logic:
                    if comp_x < box_r and box_x < comp_r:
                        components_in_box.append(stats[j])
                
                if not components_in_box: continue

                # The rest of the CCA union logic is unchanged
                min_x = min(c[cv2.CC_STAT_LEFT] for c in components_in_box)
                min_y = min(c[cv2.CC_STAT_TOP] for c in components_in_box)
                max_r = max(c[cv2.CC_STAT_LEFT] + c[cv2.CC_STAT_WIDTH] for c in components_in_box)
                max_b = max(c[cv2.CC_STAT_TOP] + c[cv2.CC_STAT_HEIGHT] for c in components_in_box)
                
                refined_boxes_list.append({
                    "text": word_label, "left": min_x, "top": min_y, "width": max_r - min_x, "height": max_b - min_y, "conf": line_data["conf"][0],
                })

            # Convert to dict format
            final_output = {k: [] for k in ["text", "left", "top", "width", "height", "conf"]}
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
            l, t = final_output["left"][i], final_output["top"][i]
            w, h = final_output["width"][i], final_output["height"][i]
            
            # Define the 4 corners of this box
            # Use float for accurate transformation
            corners = np.array([
                [l, t],
                [l + w, t],
                [l + w, t + h],
                [l, t + h]
            ], dtype="float32")
            
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
                "conf": final_output["conf"][i]
            }
            remapped_boxes_list.append(remapped_box)

        # Convert the remapped list back to the dictionary format
        remapped_output = {k: [] for k in final_output.keys()}
        for box in remapped_boxes_list:
            for key in remapped_output.keys():
                remapped_output[key].append(box[key])

        # Visualisation
        if SHOW_OUTPUT_IMAGES:
            output_path = f'{self.output_folder}/paddle_visualisations/{image_name}_{shortened_line_text}_final_boxes.png'
            os.makedirs(f'{self.output_folder}/paddle_visualisations', exist_ok=True)
            output_image_vis = line_image.copy()
            print(f"\nFinal refined {len(remapped_output['text'])} words:")
            for i in range(len(remapped_output['text'])):
                word = remapped_output['text'][i]
                x, y, w, h = (
                    int(remapped_output['left'][i]), int(remapped_output['top'][i]),
                    int(remapped_output['width'][i]), int(remapped_output['height'][i])
                )
                print(f"- '{word}' at ({x}, {y}, {w}, {h})")
                cv2.rectangle(output_image_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(output_path, output_image_vis)
            print(f"\nSaved visualisation to '{output_path}'")
        
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
            "text": list(), "left": list(), "top": list(), "width": list(),
            "height": list(), "conf": list(),
        }

        if not line_data or not line_data.get("text"):
            return output
        
        i = 0 # Assuming a single line
        line_text = line_data["text"][i]
        line_left = float(line_data["left"][i])
        line_top = float(line_data["top"][i])
        line_width = float(line_data["width"][i])
        line_height = float(line_data["height"][i])
        line_conf = line_data["conf"][i]

        if not line_text.strip(): return output
        words = line_text.split()
        if not words: return output
        num_chars = len("".join(words))
        num_spaces = len(words) - 1
        if num_chars == 0: return output

        if (num_chars * 2 + num_spaces) > 0:
            char_space_ratio = 2.0
            estimated_space_width = line_width / (num_chars * char_space_ratio + num_spaces)
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
        direction: str = 'ltr'
    ) -> List[Dict]:
        """Helper function to run one pass of refinement (either LTR or RTL)."""
        
        refined_boxes = [box.copy() for box in initial_boxes]
        
        if direction == 'ltr':
            last_corrected_right_edge = 0
            indices = range(len(refined_boxes))
        else: # rtl
            next_corrected_left_edge = img_w
            indices = range(len(refined_boxes) - 1, -1, -1)

        for i in indices:
            box = refined_boxes[i]
            left = int(box['left'])
            right = int(box['left'] + box['width'])
            
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
            if direction == 'ltr':
                if new_left < last_corrected_right_edge:
                    new_left = last_corrected_right_edge
                last_corrected_right_edge = max(last_corrected_right_edge, new_right)
            else: # rtl
                if new_right > next_corrected_left_edge:
                    new_right = next_corrected_left_edge
                next_corrected_left_edge = min(next_corrected_left_edge, new_left)

            box['left'] = new_left
            box['width'] = max(1, new_right - new_left)
            
        return refined_boxes

    def refine_words_bidirectional(
        self,
        line_data: Dict[str, List],
        line_image: np.ndarray,
    ) -> Dict[str, List]:
        """
        Refines boxes using a more robust bidirectional scan and averaging.
        """
        if line_image is None: return line_data
        
        # Early return if 1 or fewer words
        if line_data and line_data.get("text"):
            words = line_data["text"][0].split()
            if len(words) <= 1:
                img_h, img_w = line_image.shape[:2]
                return self._convert_line_to_word_level_improved(line_data, img_w, img_h)
        
        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img_h, img_w = binary.shape
        vertical_projection = np.sum(binary, axis=0)
        
        char_blobs = []
        in_blob = False; blob_start = 0
        for x, col_sum in enumerate(vertical_projection):
            if col_sum > 0 and not in_blob: blob_start = x; in_blob = True
            elif col_sum == 0 and in_blob: char_blobs.append((blob_start, x)); in_blob = False
        if in_blob: char_blobs.append((blob_start, img_w))

        if not char_blobs:
            return self._convert_line_to_word_level_improved(line_data, img_w, img_h)

        avg_char_width = np.mean([end - start for start, end in char_blobs])
        max_scan_distance = int(avg_char_width * 1.5)
        
        estimated_data = self._convert_line_to_word_level_improved(line_data, img_w, img_h)
        if not estimated_data["text"]: return estimated_data

        initial_boxes = []
        for i in range(len(estimated_data["text"])):
            initial_boxes.append({
                "text": estimated_data["text"][i], "left": estimated_data["left"][i],
                "top": estimated_data["top"][i], "width": estimated_data["width"][i],
                "height": estimated_data["height"][i], "conf": estimated_data["conf"][i],
            })

        # 1. & 2. Perform both passes
        ltr_boxes = self._run_single_pass(initial_boxes, vertical_projection, max_scan_distance, img_w, 'ltr')
        rtl_boxes = self._run_single_pass(initial_boxes, vertical_projection, max_scan_distance, img_w, 'rtl')

        # 3. Combine the results by taking the best edge from each pass
        combined_boxes = [box.copy() for box in initial_boxes]
        for i in range(len(combined_boxes)):
            
            # Get the "expert" left boundary from the LTR pass
            final_left = ltr_boxes[i]['left']
            
            # Get the "expert" right boundary from the RTL pass
            rtl_right = rtl_boxes[i]['left'] + rtl_boxes[i]['width']
            
            combined_boxes[i]['left'] = final_left
            combined_boxes[i]['width'] = max(1, rtl_right - final_left)

        # 4. Final De-overlap Pass
        last_corrected_right_edge = 0
        for i, box in enumerate(combined_boxes):
            if box['left'] < last_corrected_right_edge:
                box['width'] = max(1, box['width'] - (last_corrected_right_edge - box['left']))
                box['left'] = last_corrected_right_edge
            
            if box['width'] < 1:
                # Handle edge case where a box is completely eliminated
                if i < len(combined_boxes) - 1:
                     next_left = combined_boxes[i+1]['left']
                     box['width'] = max(1, next_left - box['left'])
                else:
                    box['width'] = 1

            last_corrected_right_edge = box['left'] + box['width']

        # Convert back to Tesseract-style output dict
        final_output = {k: [] for k in estimated_data.keys()}
        for box in combined_boxes:
            if box['width'] > 0: # Ensure we don't add zero-width boxes
                for key in final_output.keys():
                    final_output[key].append(box[key])
        
        return final_output

# --- Example Usage ---
if __name__ == '__main__':
    # Make sure you have the previous class available to import for the fallback
    image_path = 'input/example_partnership_p6_1.PNG'
    #image_path = 'input/example_partnership_p6_2.PNG'
    #image_path = 'input/example_partnership_p4_1.PNG'
    #image_path = 'input/line_image_3.png'
    #image_path = 'input/cora_fuller.png'
    #image_path = 'input/london_borough_of_lambeth.png'
    image_basename = os.path.basename(image_path)
    image_name = os.path.splitext(image_basename)[0]
    output_path = f'outputs/{image_name}_refined_morph.png'
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    line_image_cv = cv2.imread(image_path)
    h, w, _ = line_image_cv.shape

    # Read in related text
    with open(f'input/{image_name}_text.txt', 'r') as file:
        text = file.read()
    line_data = {
        "text": [text],
        "left": [0], "top": [0], "width": [w], "height": [h], "conf": [95.0]
    }
    segmenter = AdaptiveSegmenter()
    final_word_data, used_fallback = segmenter.segment(line_data, line_image_cv, image_name=image_name)

    # Visualisation
    output_image_vis = line_image_cv.copy()
    print(f"\nFinal refined {len(final_word_data['text'])} words:")
    for i in range(len(final_word_data['text'])):
        word = final_word_data['text'][i]
        x, y, w, h = (
            int(final_word_data['left'][i]), int(final_word_data['top'][i]),
            int(final_word_data['width'][i]), int(final_word_data['height'][i])
        )
        print(f"- '{word}' at ({x}, {y}, {w}, {h})")
        cv2.rectangle(output_image_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imwrite(output_path, output_image_vis)
    print(f"\nSaved visualisation to '{output_path}'")

    # You can also use matplotlib to display it in a notebook
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(output_image_vis, cv2.COLOR_BGR2RGB))

    if used_fallback:
        plt.title("Refined with Bounded Scan")
    else:
        plt.title("Refined with Morphological Closing")
    plt.axis('off')
    plt.show()