import pandas as pd
import os
import re
from tools.helper_functions import OUTPUT_FOLDER
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import gradio as gr
from gradio import Progress
from pathlib import Path
from pymupdf import Document
from tools.file_conversion import redact_whole_pymupdf_page, convert_annotation_data_to_dataframe
import en_core_web_lg
nlp = en_core_web_lg.load()

similarity_threshold = 0.95

def combine_ocr_output_text(input_files:List[str], output_folder:str=OUTPUT_FOLDER):
    """
    Combines text from multiple CSV files containing page and text columns.
    Groups text by file and page number, concatenating text within these groups.
    
    Args:
        input_files (list): List of paths to CSV files
    
    Returns:
        pd.DataFrame: Combined dataframe with columns [file, page, text]
    """
    all_data = []
    output_files = []

    if isinstance(input_files, str):
        file_paths_list = [input_files]
    else:
        file_paths_list = input_files
    
    for file in file_paths_list:

        if isinstance(file, str):
            file_path = file
        else:
            file_path = file.name

        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        if 'page' not in df.columns or 'text' not in df.columns:
            print(f"Warning: Skipping {file_path} - missing required columns 'page' and 'text'")
            continue

        df['text'] = df['text'].fillna('').astype(str)
        
        # Group by page and concatenate text
        grouped = df.groupby('page')['text'].apply(' '.join).reset_index()
        
        # Add filename column
        grouped['file'] = os.path.basename(file_path)
        
        all_data.append(grouped)
    
    if not all_data:
        raise ValueError("No valid CSV files were processed")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns
    combined_df = combined_df[['file', 'page', 'text']]

    output_combined_file_path = output_folder + "combined_ocr_output_files.csv"
    combined_df.to_csv(output_combined_file_path, index=None)

    output_files.append(output_combined_file_path)
    
    return combined_df, output_files

def process_data(df:pd.DataFrame, column:str):
    '''
    Clean and stem text columns in a data frame
    '''
    
    def _clean_text(raw_text):
        # Remove HTML tags
        clean = re.sub(r'<.*?>', '', raw_text)
        clean = ' '.join(clean.split())
        # Join the cleaned words back into a string
        return clean

    # Function to apply lemmatisation and remove stopwords
    def _apply_lemmatization(text):
        doc = nlp(text)
        # Keep only alphabetic tokens and remove stopwords
        lemmatized_words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        return ' '.join(lemmatized_words)
    
    df['text_clean'] = df[column].apply(_clean_text)

    df['text_clean'] = df['text_clean'].apply(_apply_lemmatization)
    
    return df

def map_metadata_single_page(similarity_df:pd.DataFrame, metadata_source_df:pd.DataFrame, preview_length:int=200):
    """Helper to map metadata for single page results."""
    metadata_df = metadata_source_df[['file', 'page', 'text']]
    results_df = similarity_df.merge(metadata_df, left_on='Page1_Index', right_index=True)\
                            .rename(columns={'file': 'Page1_File', 'page': 'Page1_Page', 'text': 'Page1_Text'})
    results_df = results_df.merge(metadata_df, left_on='Page2_Index', right_index=True, suffixes=('_1', '_2'))\
                            .rename(columns={'file': 'Page2_File', 'page': 'Page2_Page', 'text': 'Page2_Text'})
    results_df["Similarity_Score"] = results_df["Similarity_Score"].round(3)
    final_df = results_df[['Page1_File', 'Page1_Page', 'Page2_File', 'Page2_Page', 'Similarity_Score', 'Page1_Text', 'Page2_Text']]
    final_df = final_df.sort_values(["Page1_File", "Page1_Page", "Page2_File", "Page2_Page"])
    final_df['Page1_Text'] = final_df['Page1_Text'].str[:preview_length]
    final_df['Page2_Text'] = final_df['Page2_Text'].str[:preview_length]
    return final_df

def map_metadata_subdocument(subdocument_df:pd.DataFrame, metadata_source_df:pd.DataFrame, preview_length:int=200):
    """Helper to map metadata for subdocument results."""
    metadata_df = metadata_source_df[['file', 'page', 'text']]
    
    subdocument_df = subdocument_df.merge(metadata_df, left_on='Page1_Start_Index', right_index=True)\
                                   .rename(columns={'file': 'Page1_File', 'page': 'Page1_Start_Page', 'text': 'Page1_Text'})
    subdocument_df = subdocument_df.merge(metadata_df[['page']], left_on='Page1_End_Index', right_index=True)\
                                   .rename(columns={'page': 'Page1_End_Page'})
    subdocument_df = subdocument_df.merge(metadata_df, left_on='Page2_Start_Index', right_index=True)\
                                   .rename(columns={'file': 'Page2_File', 'page': 'Page2_Start_Page', 'text': 'Page2_Text'})
    subdocument_df = subdocument_df.merge(metadata_df[['page']], left_on='Page2_End_Index', right_index=True)\
                                   .rename(columns={'page': 'Page2_End_Page'})

    cols = ['Page1_File', 'Page1_Start_Page', 'Page1_End_Page',
            'Page2_File', 'Page2_Start_Page', 'Page2_End_Page',
            'Match_Length', 'Page1_Text', 'Page2_Text']
            
    # Add Avg_Similarity if it exists (it won't for greedy match unless we add it)
    if 'Avg_Similarity' in subdocument_df.columns:
        subdocument_df['Avg_Similarity'] = subdocument_df['Avg_Similarity'].round(3)
        cols.insert(7, 'Avg_Similarity')

    final_df = subdocument_df[cols]
    final_df = final_df.sort_values(['Page1_File', 'Page1_Start_Page', 'Page2_File', 'Page2_Start_Page'])
    final_df['Page1_Text'] = final_df['Page1_Text'].str[:preview_length]
    final_df['Page2_Text'] = final_df['Page2_Text'].str[:preview_length]

    return final_df

def save_results_and_redaction_lists(final_df: pd.DataFrame, output_folder: str) -> list:
    """
    Saves the main results DataFrame and generates per-file redaction lists.
    This function is extracted to be reusable.

    Args:
        final_df (pd.DataFrame): The DataFrame containing the final match results.
        output_folder (str): The folder to save the output files.

    Returns:
        list: A list of paths to all generated files.
    """
    output_paths = []
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(exist_ok=True)

    if final_df.empty:
        print("No matches to save.")
        return []

    # 1. Save the main results DataFrame
    similarity_file_output_path = output_folder_path / 'page_similarity_results.csv'
    final_df.to_csv(similarity_file_output_path, index=False)

    output_paths.append(str(similarity_file_output_path))
    print(f"Main results saved to {similarity_file_output_path}")

    # 2. Save per-file redaction lists
    # Use 'Page2_File' as the source of duplicate content
    grouping_col = 'Page2_File'
    if grouping_col not in final_df.columns:
        print("Warning: 'Page2_File' column not found. Cannot generate redaction lists.")
        return output_paths

    for redact_file, group in final_df.groupby(grouping_col):
        output_file_name_stem = Path(redact_file).stem
        output_file_path = output_folder_path / f"{output_file_name_stem}_pages_to_redact.csv"
        
        all_pages_to_redact = set()
        is_subdocument_match = 'Page2_Start_Page' in group.columns

        if is_subdocument_match:
            for _, row in group.iterrows():
                pages_in_range = range(int(row['Page2_Start_Page']), int(row['Page2_End_Page']) + 1)
                all_pages_to_redact.update(pages_in_range)
        else:
            pages = group['Page2_Page'].unique()
            all_pages_to_redact.update(pages)
        
        if all_pages_to_redact:
            redaction_df = pd.DataFrame(sorted(list(all_pages_to_redact)), columns=['Page_to_Redact'])
            redaction_df.to_csv(output_file_path, header=False, index=False)

            output_paths.append(str(output_file_path))
            print(f"Redaction list for {redact_file} saved to {output_file_path}")
            
    return output_paths

def identify_similar_pages(
    df_combined: pd.DataFrame,
    similarity_threshold: float = 0.9,
    min_word_count: int = 10,
    min_consecutive_pages: int = 1,
    greedy_match: bool = False, # NEW parameter
    output_folder: str = OUTPUT_FOLDER,
    progress=Progress(track_tqdm=True)
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Identifies similar pages with three possible strategies:
    1. Single Page: If greedy_match=False and min_consecutive_pages=1.
    2. Fixed-Length Subdocument: If greedy_match=False and min_consecutive_pages > 1.
    3. Greedy Consecutive Match: If greedy_match=True.
    """

    output_paths = []
    progress(0.1, desc="Processing and filtering text")
    df = process_data(df_combined, 'text')
    df['word_count'] = df['text_clean'].str.split().str.len().fillna(0)
    original_row_count = len(df)
    df_filtered = df[df['word_count'] >= min_word_count].copy()
    df_filtered.reset_index(drop=True, inplace=True)
    
    print(f"Filtered out {original_row_count - len(df_filtered)} pages with fewer than {min_word_count} words.")

    if len(df_filtered) < 2:
        return pd.DataFrame(), [], df_combined
        
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_filtered['text_clean'])

    progress(0.3, desc="Calculating text similarity")
    similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=False)
    coo_matrix = similarity_matrix.tocoo()
    
    # Create a DataFrame of all individual page pairs above the threshold.
    # This is the base for all three matching strategies.
    similar_pages = [
        (r, c, v) for r, c, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
        if r < c and v >= similarity_threshold
    ]

    if not similar_pages:
        return pd.DataFrame(), [], df_combined
    
    base_similarity_df = pd.DataFrame(similar_pages, columns=['Page1_Index', 'Page2_Index', 'Similarity_Score'])

    progress(0.6, desc="Aggregating results based on matching strategy")
    
    if greedy_match:
        print("Finding matches using greedy consecutive strategy.")
        
        # A set of pairs for fast lookups of (page1_idx, page2_idx)
        valid_pairs_set = set(zip(base_similarity_df['Page1_Index'], base_similarity_df['Page2_Index']))
        
        # Keep track of indices that have been used in a sequence
        consumed_indices_1 = set()
        consumed_indices_2 = set()
        
        all_sequences = []

        # Iterate through all potential starting pairs, sorted for consistent results
        sorted_pairs = base_similarity_df.sort_values(['Page1_Index', 'Page2_Index'])

        for _, row in sorted_pairs.iterrows():
            start_idx1, start_idx2 = int(row['Page1_Index']), int(row['Page2_Index'])
            
            # If this pair has already been consumed by a previous sequence, skip it
            if start_idx1 in consumed_indices_1 or start_idx2 in consumed_indices_2:
                continue

            # This is a new sequence, start expanding it
            current_sequence = [(start_idx1, start_idx2)]
            k = 1
            while True:
                next_idx1 = start_idx1 + k
                next_idx2 = start_idx2 + k
                
                # Check if the next pair in the sequence is a valid match
                if (next_idx1, next_idx2) in valid_pairs_set and \
                   next_idx1 not in consumed_indices_1 and \
                   next_idx2 not in consumed_indices_2:
                    current_sequence.append((next_idx1, next_idx2))
                    k += 1
                else:
                    # The sequence has ended
                    break
            
            # Record the found sequence and mark all its pages as consumed
            sequence_indices_1 = [p[0] for p in current_sequence]
            sequence_indices_2 = [p[1] for p in current_sequence]
            
            all_sequences.append({
                'Page1_Start_Index': sequence_indices_1[0], 'Page1_End_Index': sequence_indices_1[-1],
                'Page2_Start_Index': sequence_indices_2[0], 'Page2_End_Index': sequence_indices_2[-1],
                'Match_Length': len(current_sequence)
            })

            consumed_indices_1.update(sequence_indices_1)
            consumed_indices_2.update(sequence_indices_2)

        if not all_sequences:
            return pd.DataFrame(), [], df_combined

        subdocument_df = pd.DataFrame(all_sequences)
        # We can add back the average similarity if needed, but it requires more lookups.
        # For now, we'll omit it for simplicity in the greedy approach.
        # ... (The rest is metadata mapping, same as the subdocument case)

    elif min_consecutive_pages > 1:
        # --- STRATEGY 2: Fixed-Length Subdocument Matching ---
        print(f"Finding consecutive page matches (min_consecutive_pages > 1)")
        similarity_df = base_similarity_df.copy()
        similarity_df.sort_values(['Page1_Index', 'Page2_Index'], inplace=True)
        is_consecutive = (similarity_df['Page1_Index'].diff() == 1) & (similarity_df['Page2_Index'].diff() == 1)
        block_id = is_consecutive.eq(False).cumsum()
        grouped = similarity_df.groupby(block_id)
        agg_results = grouped.agg(
            Page1_Start_Index=('Page1_Index', 'first'), Page2_Start_Index=('Page2_Index', 'first'),
            Page1_End_Index=('Page1_Index', 'last'), Page2_End_Index=('Page2_Index', 'last'),
            Match_Length=('Page1_Index', 'size'), Avg_Similarity=('Similarity_Score', 'mean')
        ).reset_index(drop=True)
        subdocument_df = agg_results[agg_results['Match_Length'] >= min_consecutive_pages].copy()
        if subdocument_df.empty: return pd.DataFrame(), [], df_combined

    else:
        # --- STRATEGY 1: Single Page Matching ---
        print(f"Finding single page matches (min_consecutive_pages=1)")
        final_df = map_metadata_single_page(base_similarity_df, df_filtered)
        # The rest of the logic (saving files) is handled after this if/else block
        pass # The final_df is already prepared

    # --- Map metadata and format output ---
    # This block now handles the output for both subdocument strategies (2 and 3)
    if greedy_match or min_consecutive_pages > 1:
        final_df = map_metadata_subdocument(subdocument_df, df_filtered)
    
    progress(0.8, desc="Saving output files")
    
    output_paths = save_results_and_redaction_lists(final_df, output_folder)

    return final_df, output_paths, df_combined

# ==============================================================================
# GRADIO HELPER FUNCTIONS
# ==============================================================================

# full_data:pd.DataFrame, 
def handle_selection_and_preview(evt: gr.SelectData, results_df:pd.DataFrame, full_duplicate_data_by_file: dict):
    """
    This single function handles a user selecting a row. It:
    1. Determines the selected row index.
    2. Calls the show_page_previews function to get the text data.
    3. Returns all the necessary outputs for the UI.
    """
    # If the user deselects, the event might be None.
    if not evt:
        return None, None, None # Clear state and both preview panes

    # 1. Get the selected index
    selected_index = evt.index[0]

    # 2. Get the preview data
    page1_data, page2_data = show_page_previews(full_duplicate_data_by_file, results_df, evt)

    # 3. Return all three outputs in the correct order
    return selected_index, page1_data, page2_data

def exclude_match(results_df:pd.DataFrame, selected_index:int, output_folder="./output/"):
    """
    Removes a selected row from the results DataFrame, regenerates output files,
    and clears the text preview panes.
    """
    if selected_index is None:
        gr.Warning("No match selected. Please click on a row in the table first.")
        # Return the original dataframe and update=False for the files
        return results_df, gr.update(), None, None
    
    if results_df.empty:
        gr.Warning("No duplicate page results found, nothing to exclude.")
        return results_df, gr.update(), None, None

    # Drop the selected row
    updated_df = results_df.drop(selected_index).reset_index(drop=True)
    
    # Recalculate all output files using the helper function
    new_output_paths = save_results_and_redaction_lists(updated_df, output_folder)
        
    gr.Info(f"Match at row {selected_index} excluded. Output files have been updated.")
    
    # Return the updated dataframe, the new file list, and clear the preview panes
    return updated_df, new_output_paths, None, None

def run_duplicate_analysis(files:list[pd.DataFrame], threshold:float, min_words:int, min_consecutive:int, greedy_match:bool, preview_length:int=500, progress=gr.Progress(track_tqdm=True)):
    """
    Wrapper function updated to include the 'greedy_match' boolean.
    """
    if not files:
        gr.Warning("Please upload files to analyze.")
        return None, None, None
        
    progress(0, desc="Combining input files...")
    df_combined, _ = combine_ocr_output_text(files)

    if df_combined.empty:
        gr.Warning("No data found in the uploaded files.")
        return None, None, None

    # Call the main analysis function with the new parameter
    results_df, output_paths, full_df = identify_similar_pages(
        df_combined=df_combined,
        similarity_threshold=threshold,
        min_word_count=min_words,
        min_consecutive_pages=int(min_consecutive),
        greedy_match=greedy_match,
        progress=progress
    )

    # Clip text to first 200 characters
    full_df['text'] = full_df['text'].str[:preview_length]

    # Preprocess full_data (without preview text) for fast access (run once)
    full_data_by_file = {
    file: df.sort_values('page').set_index('page')
    for file, df in full_df.drop(["text_clean"],axis=1).groupby('file')
    }

    if results_df.empty:
        gr.Info(f"No duplicate pages found, no results returned.")
    
    return results_df, output_paths, full_data_by_file # full_df, 

def show_page_previews(full_data_by_file: dict, results_df: pd.DataFrame, evt: gr.SelectData, preview_length:int=500):
    """
    Optimized version using pre-partitioned and indexed full_data.
    Triggered when a user selects a row in the results DataFrame.
    """
    if not full_data_by_file or results_df is None or not evt:
        return None, None

    selected_row = results_df.iloc[evt.index[0], :]

    is_subdocument_match = 'Page1_Start_Page' in selected_row

    if is_subdocument_match:
        file1, start1, end1 = selected_row['Page1_File'], selected_row['Page1_Start_Page'], selected_row['Page1_End_Page']
        file2, start2, end2 = selected_row['Page2_File'], selected_row['Page2_Start_Page'], selected_row['Page2_End_Page']

        page1_data = full_data_by_file[file1].loc[start1:end1, ['text']].reset_index()
        page2_data = full_data_by_file[file2].loc[start2:end2, ['text']].reset_index()

    else:
        file1, page1 = selected_row['Page1_File'], selected_row['Page1_Page']
        file2, page2 = selected_row['Page2_File'], selected_row['Page2_Page']

        page1_data = full_data_by_file[file1].loc[[page1], ['text']].reset_index()
        page2_data = full_data_by_file[file2].loc[[page2], ['text']].reset_index()

    page1_data['text'] = page1_data['text'].str[:preview_length]
    page2_data['text'] = page2_data['text'].str[:preview_length]

    return page1_data[['page', 'text']], page2_data[['page', 'text']]

def apply_whole_page_redactions_from_list(duplicate_page_numbers_df:pd.DataFrame, doc_file_name_with_extension_textbox:str, review_file_state:pd.DataFrame, duplicate_output_paths:list[str], pymupdf_doc:object, page_sizes:list[dict], all_existing_annotations:list[dict]):
    '''
    Take a list of suggested whole pages to redact and apply it to review file data currently available from an existing PDF under review
    '''
    # Create a copy of annotations to avoid modifying the original
    all_annotations = all_existing_annotations.copy()

    if not pymupdf_doc:     
        print("Warning: No document file currently under review. Please upload a document on the 'Review redactions' tab to apply whole page redactions.")
        raise Warning("No document file currently under review. Please upload a document on the 'Review redactions' tab to apply whole page redactions.")
        return review_file_state, all_annotations

    # Initialize list of pages to redact
    list_whole_pages_to_redact = []
    
    # Get list of pages to redact from either dataframe or file
    if not duplicate_page_numbers_df.empty:
        list_whole_pages_to_redact = duplicate_page_numbers_df.iloc[:, 0].tolist()
    elif duplicate_output_paths:
        expected_duplicate_pages_to_redact_name = f"{doc_file_name_with_extension_textbox}"
        whole_pages_list = pd.DataFrame()  # Initialize empty DataFrame
        
        for output_file in duplicate_output_paths:
            # Note: output_file.name might not be available if output_file is just a string path
            # If it's a Path object or similar, .name is fine. Otherwise, parse from string.
            file_name_from_path = output_file.split('/')[-1] if isinstance(output_file, str) else output_file.name
            if expected_duplicate_pages_to_redact_name in file_name_from_path:
                whole_pages_list = pd.read_csv(output_file, header=None) # Use output_file directly if it's a path
                break
        
        if not whole_pages_list.empty:
            list_whole_pages_to_redact = whole_pages_list.iloc[:, 0].tolist()
    
    # Convert to set to remove duplicates, then back to list
    list_whole_pages_to_redact = list(set(list_whole_pages_to_redact))
    
    if not list_whole_pages_to_redact:
        # Assuming gr is defined (e.g., gradio)
        print("No relevant list of whole pages to redact found, returning inputs.")
        raise Warning("Warning: No relevant list of whole pages to redact found, returning inputs.")
        return review_file_state, all_existing_annotations
    
    new_annotations = []

    # Process each page for redaction
    for page in list_whole_pages_to_redact:
        try:
            page_index = int(page) - 1
            if page_index < 0 or page_index >= len(pymupdf_doc):
                print(f"Page {page} is out of bounds for a document with {len(pymupdf_doc)} pages, skipping.")
                continue
                
            pymupdf_page = pymupdf_doc[page_index]

            # Find the matching page size dictionary
            page_size = next((size for size in page_sizes if size["page"] == int(page)), None)
            
            if not page_size:
                print(f"Page {page} not found in page_sizes object, skipping.")
                continue

            rect_height = page_size["cropbox_height"]
            rect_width = page_size["cropbox_width"]
            image = page_size["image_path"] # This `image` likely represents the page identifier

            # Create the whole page redaction box
            annotation_box = redact_whole_pymupdf_page(rect_height, rect_width, pymupdf_page, border=0.005, redact_pdf=False)
            
            # Find existing annotation for this image/page
            current_page_existing_boxes_group = next((annot_group for annot_group in all_annotations if annot_group["image"] == image), None)

            new_annotation_group = {
                    "image": image,
                    "boxes": [annotation_box]
                }

            if current_page_existing_boxes_group:
                # Check if we already have a whole page redaction for this page
                if not any(box.get("label", "Whole page") for box in current_page_existing_boxes_group["boxes"]):
                    current_page_existing_boxes_group["boxes"].append(annotation_box)

                else:
                    # Optional: Print a message if a whole-page redaction already exists for this page
                    print(f"Whole page redaction for page {page} already exists in annotations, skipping addition.")
                    pass
            else:
                # Create new annotation entry
                                
                all_annotations.append(new_annotation_group)

            new_annotations.append(new_annotation_group)
                
        except Exception as e:
            print(f"Error processing page {page}: {str(e)}")
            continue

    # Convert annotations to dataframe and combine with existing review file
    whole_page_review_file = convert_annotation_data_to_dataframe(new_annotations)
    
    # Ensure all required columns are present in both DataFrames before concat
    # This is a common point of error if DFs have different schemas
    expected_cols = ['image', 'page', 'label', 'color', 'xmin', 'ymin', 'xmax', 'ymax', 'text', 'id']
    
    for col in expected_cols:
        if col not in review_file_state.columns:
            review_file_state[col] = None # Or an appropriate default value
        if col not in whole_page_review_file.columns:
            whole_page_review_file[col] = None

    review_file_out = pd.concat([review_file_state, whole_page_review_file], ignore_index=True)
    review_file_out = review_file_out.sort_values(by=["page", "ymin", "xmin"])

    # --- Remove duplicate entries from the final DataFrame ---
    dedup_subset_cols = ['page', 'label', 'text', 'id']
    
    # Ensure these columns exist before trying to use them as subset for drop_duplicates
    if all(col in review_file_out.columns for col in dedup_subset_cols):
        review_file_out = review_file_out.drop_duplicates(
            subset=dedup_subset_cols,
            keep='first' # Keep the first occurrence of a duplicate redaction
        )
    else:
        print(f"Warning: Not all columns required for de-duplication ({dedup_subset_cols}) are present in review_file_out. Skipping specific de-duplication.")
        # You might want a fallback or to inspect what's missing

    review_file_out.to_csv(OUTPUT_FOLDER + "review_file_out_after_whole_page.csv")

    gr.Info("Successfully created whole page redactions. Go to the 'Review redactions' tab to see them.")

    return review_file_out, all_annotations