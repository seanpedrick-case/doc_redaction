import pandas as pd
import os
import re
from tools.helper_functions import OUTPUT_FOLDER
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string
from typing import List, Tuple
import gradio as gr
from gradio import Progress
from pathlib import Path

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
        # clean = re.sub(r'&nbsp;', ' ', clean)
        # clean = re.sub(r'\r\n', ' ', clean)
        # clean = re.sub(r'&lt;', ' ', clean)
        # clean = re.sub(r'&gt;', ' ', clean)
        # clean = re.sub(r'<strong>', ' ', clean)
        # clean = re.sub(r'</strong>', ' ', clean)

        # Replace non-breaking space \xa0 with a space
        # clean = clean.replace(u'\xa0', u' ')
        # Remove extra whitespace
        clean = ' '.join(clean.split())

        # # Tokenize the text
        # words = word_tokenize(clean.lower())

        # # Remove punctuation and numbers
        # words = [word for word in words if word.isalpha()]

        # # Remove stopwords
        # words = [word for word in words if word not in stop_words]

        # Join the cleaned words back into a string
        return clean

    # Function to apply lemmatization and remove stopwords
    def _apply_lemmatization(text):
        doc = nlp(text)
        # Keep only alphabetic tokens and remove stopwords
        lemmatized_words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        return ' '.join(lemmatized_words)
    
    df['text_clean'] = df[column].apply(_clean_text)

    df['text_clean'] = df['text_clean'].apply(_apply_lemmatization)
    
    return df

def map_metadata_single_page(similarity_df, metadata_source_df):
    """Helper to map metadata for single page results."""
    metadata_df = metadata_source_df[['file', 'page', 'text']]
    results_df = similarity_df.merge(metadata_df, left_on='Page1_Index', right_index=True)\
                            .rename(columns={'file': 'Page1_File', 'page': 'Page1_Page', 'text': 'Page1_Text'})
    results_df = results_df.merge(metadata_df, left_on='Page2_Index', right_index=True, suffixes=('_1', '_2'))\
                            .rename(columns={'file': 'Page2_File', 'page': 'Page2_Page', 'text': 'Page2_Text'})
    results_df["Similarity_Score"] = results_df["Similarity_Score"].round(3)
    final_df = results_df[['Page1_File', 'Page1_Page', 'Page2_File', 'Page2_Page', 'Similarity_Score', 'Page1_Text', 'Page2_Text']]
    final_df = final_df.sort_values(["Page1_File", "Page1_Page", "Page2_File", "Page2_Page"])
    final_df['Page1_Text'] = final_df['Page1_Text'].str[:200]
    final_df['Page2_Text'] = final_df['Page2_Text'].str[:200]
    return final_df


def map_metadata_subdocument(subdocument_df, metadata_source_df):
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
    final_df['Page1_Text'] = final_df['Page1_Text'].str[:200]
    final_df['Page2_Text'] = final_df['Page2_Text'].str[:200]
    return final_df

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
    # ... (Initial setup: progress, data loading/processing, word count filter) ...
    # This part remains the same as before.
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
    
    # --- NEW: Logic to select matching strategy ---

    if greedy_match:
        # --- STRATEGY 3: Greedy Consecutive Matching ---
        print("Finding matches using GREEDY consecutive strategy.")
        
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
    
    # If no matches were found, final_df could be empty.
    if final_df.empty:
        print("No matches found, no output files to save.")
        return final_df, [], df_combined

    # --- 1. Save the main results DataFrame ---
    # This file contains the detailed summary of all matches found.
    similarity_file_output_path = Path(output_folder) / 'page_similarity_results.csv'
    final_df.to_csv(similarity_file_output_path, index=False)
    output_paths.append(str(similarity_file_output_path))
    print(f"Main results saved to {similarity_file_output_path}")

    # --- 2. Save per-file redaction lists ---
    # These files contain a simple list of page numbers to redact for each document
    # that contains duplicate content.
    
    # We group by the file containing the duplicates ('Page2_File')
    for redact_file, group in final_df.groupby('Page2_File'):
        output_file_name_stem = Path(redact_file).stem
        output_file_path = Path(output_folder) / f"{output_file_name_stem}_pages_to_redact.csv"
        
        all_pages_to_redact = set()
        
        # Check if the results are for single pages or subdocuments
        is_subdocument_match = 'Page2_Start_Page' in group.columns

        if is_subdocument_match:
            # For subdocument matches, create a range of pages for each match
            for _, row in group.iterrows():
                # Generate all page numbers from the start to the end of the match
                pages_in_range = range(int(row['Page2_Start_Page']), int(row['Page2_End_Page']) + 1)
                all_pages_to_redact.update(pages_in_range)
        else:
            # For single-page matches, just add the page number
            pages = group['Page2_Page'].unique()
            all_pages_to_redact.update(pages)
        
        if all_pages_to_redact:
            # Create a DataFrame from the sorted list of pages to redact
            redaction_df = pd.DataFrame(sorted(list(all_pages_to_redact)), columns=['Page_to_Redact'])
            redaction_df.to_csv(output_file_path, header=False, index=False)
            output_paths.append(str(output_file_path))
            print(f"Redaction list for {redact_file} saved to {output_file_path}")

    # Note: The 'combined ocr output' csv was part of the original data loading function,
    # not the analysis function itself. If you need that, it should be saved within
    # your `combine_ocr_output_text` function.

    return final_df, output_paths, df_combined

# ==============================================================================
# GRADIO HELPER FUNCTIONS
# ==============================================================================

def run_analysis(files, threshold, min_words, min_consecutive, greedy_match, progress=gr.Progress(track_tqdm=True)):
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
        greedy_match=greedy_match, # Pass the new boolean
        progress=progress
    )
    
    return results_df, output_paths, full_df

def show_page_previews(full_data, results_df, evt: gr.SelectData):
    """
    Triggered when a user selects a row in the results DataFrame.
    It uses the stored 'full_data' to find and display the complete text.
    """
    if full_data is None or results_df is None:
        return None, None # Return empty dataframes if no analysis has been run

    selected_row = results_df.iloc[evt.index[0]]
    
    # Determine if it's a single page or a multi-page (subdocument) match
    is_subdocument_match = 'Page1_Start_Page' in selected_row

    if is_subdocument_match:
        # --- Handle Subdocument Match ---
        file1, start1, end1 = selected_row['Page1_File'], selected_row['Page1_Start_Page'], selected_row['Page1_End_Page']
        file2, start2, end2 = selected_row['Page2_File'], selected_row['Page2_Start_Page'], selected_row['Page2_End_Page']

        page1_data = full_data[
            (full_data['file'] == file1) &
            (full_data['page'].between(start1, end1))
        ].sort_values('page')[['page', 'text']]
        
        page2_data = full_data[
            (full_data['file'] == file2) &
            (full_data['page'].between(start2, end2))
        ].sort_values('page')[['page', 'text']]
        
    else:
        # --- Handle Single Page Match ---
        file1, page1 = selected_row['Page1_File'], selected_row['Page1_Page']
        file2, page2 = selected_row['Page2_File'], selected_row['Page2_Page']

        page1_data = full_data[
            (full_data['file'] == file1) & (full_data['page'] == page1)
        ][['page', 'text']]

        page2_data = full_data[
            (full_data['file'] == file2) & (full_data['page'] == page2)
        ][['page', 'text']]

    return page1_data, page2_data


# Perturb text
# Apply the perturbation function with a 10% error probability
def perturb_text_with_errors(series:pd.Series):

    def _perturb_text(text, error_probability=0.1):
        words = text.split()  # Split text into words
        perturbed_words = []
        
        for word in words:
            if random.random() < error_probability:  # Add a random error
                perturbation_type = random.choice(['char_error', 'extra_space', 'extra_punctuation'])
                
                if perturbation_type == 'char_error':  # Introduce a character error
                    idx = random.randint(0, len(word) - 1)
                    char = random.choice(string.ascii_lowercase)  # Add a random letter
                    word = word[:idx] + char + word[idx:]
                
                elif perturbation_type == 'extra_space':  # Add extra space around a word
                    word = ' ' + word + ' '
                
                elif perturbation_type == 'extra_punctuation':  # Add punctuation to the word
                    punctuation = random.choice(string.punctuation)
                    idx = random.randint(0, len(word))  # Insert punctuation randomly
                    word = word[:idx] + punctuation + word[idx:]
            
            perturbed_words.append(word)
        
        return ' '.join(perturbed_words)

    series = series.apply(lambda x: _perturb_text(x, error_probability=0.1))

    return series
