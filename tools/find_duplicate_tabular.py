import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import gradio as gr
from gradio import Progress
from pathlib import Path
from tools.helper_functions import OUTPUT_FOLDER, read_file
from tools.data_anonymise import initial_clean
from tools.load_spacy_model_custom_recognisers import nlp
from tools.config import DO_INITIAL_TABULAR_DATA_CLEAN

similarity_threshold = 0.95

def clean_and_stem_text_series(df: pd.DataFrame, column: str, do_initial_clean_dup: bool = DO_INITIAL_TABULAR_DATA_CLEAN):
    """
    Clean and stem text columns in a data frame for tabular data
    """

    # Function to apply lemmatisation and remove stopwords
    def _apply_lemmatization(text):
        doc = nlp(text)
        # Keep only alphabetic tokens and remove stopwords
        lemmatized_words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        return ' '.join(lemmatized_words)
    
    if do_initial_clean_dup:
        df['text_clean'] = initial_clean(df[column])

    df['text_clean'] = df['text_clean'].apply(_apply_lemmatization)
    df['text_clean'] = df[column].str.lower()#.str.replace(r'[^\w\s]', '', regex=True)
    
    return df

def convert_tabular_data_to_analysis_format(
    df: pd.DataFrame, 
    file_name: str,
    text_columns: List[str] = None
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Convert tabular data (CSV/XLSX) to the format needed for duplicate analysis.
    
    Args:
        df (pd.DataFrame): The input DataFrame
        file_name (str): Name of the file
        text_columns (List[str], optional): Columns to analyze for duplicates. 
                                          If None, uses all string columns.
    
    Returns:
        List[Tuple[str, pd.DataFrame]]: List containing (file_name, processed_df) tuple
    """
    if text_columns is None:
        # Auto-detect text columns (string type columns)
        text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    if not text_columns:
        print(f"No text columns found in {file_name}")
        return []
    
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Create a combined text column from all text columns
    df_copy['combined_text'] = df_copy[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
    
    # Add row identifier
    df_copy['row_id'] = df_copy.index
    
    # Create the format expected by the duplicate detection system
    # Using 'page' as row number and 'text' as the combined text
    processed_df = pd.DataFrame({
        'page': df_copy['row_id'],
        'text': df_copy['combined_text'],
        'file': file_name
    })
    
    # Add original row data for reference
    for col in text_columns:
        processed_df[f'original_{col}'] = df_copy[col]
    
    return [(file_name, processed_df)]

def find_duplicate_cells_in_tabular_data(
    input_files: List[str],
    similarity_threshold: float = 0.95,
    min_word_count: int = 3,
    text_columns: List[str] = None,
    output_folder: str = OUTPUT_FOLDER,
    do_initial_clean_dup: bool = DO_INITIAL_TABULAR_DATA_CLEAN,
    progress: Progress = Progress(track_tqdm=True)
) -> Tuple[pd.DataFrame, List[str], Dict[str, pd.DataFrame]]:
    """
    Find duplicate cells/text in tabular data files (CSV, XLSX).
    
    Args:
        input_files (List[str]): List of file paths to analyze
        similarity_threshold (float): Minimum similarity score to consider duplicates
        min_word_count (int): Minimum word count for text to be considered
        text_columns (List[str], optional): Specific columns to analyze
        output_folder (str, optional): Output folder for results
        do_initial_clean_dup (bool, optional): Whether to do initial clean of text
        progress (Progress): Progress tracking object
    
    Returns:
        Tuple containing:
        - results_df: DataFrame with duplicate matches
        - output_paths: List of output file paths
        - full_data_by_file: Dictionary of processed data by file
    """
    
    if not input_files:
        raise gr.Error("Please upload files to analyze.")
    
    progress(0.1, desc="Loading and processing files...")
    
    all_data_to_process = []
    full_data_by_file = {}
    file_paths = []
    
    # Process each file
    for file_path in input_files:
        try:
            df = read_file(file_path)
            
            file_name = os.path.basename(file_path)
            file_paths.append(file_path)
            
            # Convert to analysis format
            processed_data = convert_tabular_data_to_analysis_format(
                df, file_name, text_columns
            )
            
            if processed_data:
                all_data_to_process.extend(processed_data)
                full_data_by_file[file_name] = processed_data[0][1]
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not all_data_to_process:
        raise gr.Error("No valid data found in uploaded files.")
    
    progress(0.2, desc="Combining data...")
    
    # Combine all data
    combined_df = pd.concat([data[1] for data in all_data_to_process], ignore_index=True)
    
    progress(0.3, desc="Cleaning and preparing text...")
    
    # Clean and prepare text
    combined_df = clean_and_stem_text_series(combined_df, 'text', do_initial_clean_dup=do_initial_clean_dup)
    
    # Filter by minimum word count
    combined_df['word_count'] = combined_df['text_clean'].str.split().str.len().fillna(0)
    combined_df = combined_df[combined_df['word_count'] >= min_word_count].copy()
    
    if len(combined_df) < 2:
        return pd.DataFrame(), [], full_data_by_file
    
    progress(0.4, desc="Calculating similarities...")
    
    # Calculate similarities
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_df['text_clean'])
    similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=False)
    
    # Find similar pairs
    coo_matrix = similarity_matrix.tocoo()
    similar_pairs = [
        (r, c, v) for r, c, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
        if r < c and v >= similarity_threshold
    ]
    
    if not similar_pairs:
        gr.Info("No duplicate cells found.")
        return pd.DataFrame(), [], full_data_by_file
    
    progress(0.7, desc="Processing results...")
    
    # Create results DataFrame
    results_data = []
    for row1, row2, similarity in similar_pairs:
        row1_data = combined_df.iloc[row1]
        row2_data = combined_df.iloc[row2]
        
        results_data.append({
            'File1': row1_data['file'],
            'Row1': int(row1_data['page']),
            'File2': row2_data['file'],
            'Row2': int(row2_data['page']),
            'Similarity_Score': round(similarity, 3),
            'Text1': row1_data['text'][:200] + '...' if len(row1_data['text']) > 200 else row1_data['text'],
            'Text2': row2_data['text'][:200] + '...' if len(row2_data['text']) > 200 else row2_data['text'],
            'Original_Index1': row1,
            'Original_Index2': row2
        })
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values(['File1', 'Row1', 'File2', 'Row2'])
    
    progress(0.9, desc="Saving results...")
    
    # Save results
    output_paths = save_tabular_duplicate_results(results_df, output_folder, file_paths, file_replaced_index=0)
    
    gr.Info(f"Found {len(results_df)} duplicate cell matches")
    
    return results_df, output_paths, full_data_by_file

def save_tabular_duplicate_results(results_df: pd.DataFrame, output_folder: str, file_paths: List[str], file_replaced_index: int = 0) -> List[str]:
    """
    Save tabular duplicate detection results to files.
    
    Args:
        results_df (pd.DataFrame): Results DataFrame
        output_folder (str): Output folder path
        file_paths (List[str]): List of file paths
        file_replaced_index (int): Index of the file to replace with duplicate rows removed
            (0 is the first file in the list)
    Returns:
        List[str]: List of output file paths
    """
    output_paths = []
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(exist_ok=True)
    
    if results_df.empty:
        print("No duplicate matches to save.")
        return []
    
    # Save main results
    results_file = output_folder_path / 'tabular_duplicate_results.csv'
    results_df.to_csv(results_file, index=False, encoding="utf-8-sig")
    output_paths.append(str(results_file))
    
    # Save per-file duplicate lists
    for file_name, group in results_df.groupby('File1'):
        file_stem = Path(file_name).stem
        duplicate_rows_file = output_folder_path / f"{file_stem}_duplicate_rows.csv"
        
        # Get unique row numbers to remove
        rows_to_remove = sorted(group['Row1'].unique())
        duplicate_df = pd.DataFrame({'Row_to_Remove': rows_to_remove})
        duplicate_df.to_csv(duplicate_rows_file, index=False)
        output_paths.append(str(duplicate_rows_file))

        # Save also original file (first file in list) with duplicate rows removed
        file_path = file_paths[file_replaced_index]
        file_base_name = os.path.basename(file_path)
        df = read_file(file_path)
        df_cleaned = df.drop(index=rows_to_remove).reset_index(drop=True)

        output_path = os.path.join(output_folder, f"{file_base_name}_deduplicated.csv")
        df_cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        output_paths.append(str(output_path))
    
    return output_paths

def remove_duplicate_rows_from_tabular_data(
    file_path: str,
    duplicate_rows: List[int],
    output_folder: str = OUTPUT_FOLDER
) -> str:
    """
    Remove duplicate rows from a tabular data file.
    
    Args:
        file_path (str): Path to the input file
        duplicate_rows (List[int]): List of row indices to remove
        output_folder (str): Output folder for cleaned file
    
    Returns:
        str: Path to the cleaned file
    """
    try:
        # Load the file
        df = read_file(file_path)
        
        # Remove duplicate rows (0-indexed)
        df_cleaned = df.drop(index=duplicate_rows).reset_index(drop=True)
        
        # Save cleaned file
        file_name = os.path.basename(file_path)
        file_stem = os.path.splitext(file_name)[0]
        file_ext = os.path.splitext(file_name)[1]
        
        output_path = os.path.join(output_folder, f"{file_stem}_deduplicated{file_ext}")
        
        if file_ext in ['.xlsx', '.xls']:
            df_cleaned.to_excel(output_path, index=False)
        elif file_ext in ['.parquet']:
            df_cleaned.to_parquet(output_path, index=False)
        else:
            df_cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        return output_path
        
    except Exception as e:
        print(f"Error removing duplicates from {file_path}: {e}")
        raise

def run_tabular_duplicate_analysis(
    files: List[str],
    threshold: float,
    min_words: int,
    text_columns: List[str] = None,
    output_folder: str = OUTPUT_FOLDER,
    do_initial_clean_dup: bool = DO_INITIAL_TABULAR_DATA_CLEAN,
    progress: Progress = Progress(track_tqdm=True)
) -> Tuple[pd.DataFrame, List[str], Dict[str, pd.DataFrame]]:
    """
    Main function to run tabular duplicate analysis.
    
    Args:
        files (List[str]): List of file paths
        threshold (float): Similarity threshold
        min_words (int): Minimum word count
        text_columns (List[str], optional): Specific columns to analyze
        output_folder (str, optional): Output folder for results
        progress (Progress): Progress tracking
    
    Returns:
        Tuple containing results DataFrame, output paths, and full data by file
    """
    return find_duplicate_cells_in_tabular_data(
        input_files=files,
        similarity_threshold=threshold,
        min_word_count=min_words,
        text_columns=text_columns,
        output_folder=output_folder,
        do_initial_clean_dup=do_initial_clean_dup,
        progress=progress
    )



# Function to update column choices when files are uploaded
def update_tabular_column_choices(files):
    if not files:
        return gr.update(choices=[])
    
    all_columns = set()
    for file in files:
        try:
            df = read_file(file.name)
            
            # Get text columns
            text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            all_columns.update(text_cols)
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
            continue
    
    return gr.Dropdown(choices=sorted(list(all_columns)))

# Function to handle tabular duplicate detection
def run_tabular_duplicate_detection(files, threshold, min_words, text_columns, output_folder: str = OUTPUT_FOLDER, do_initial_clean_dup: bool = DO_INITIAL_TABULAR_DATA_CLEAN):
    if not files:
        return pd.DataFrame(), [], gr.Dropdown(choices=[])
    
    file_paths = [f.name for f in files]
    results_df, output_paths, full_data = run_tabular_duplicate_analysis(
        files=file_paths,
        threshold=threshold,
        min_words=min_words,
        text_columns=text_columns if text_columns else None,
        output_folder=output_folder,
        do_initial_clean_dup=do_initial_clean_dup
    )

    print("output_paths:", output_paths)
    
    # Update file choices for cleaning
    file_choices = list(set([f for f in file_paths]))
    
    return results_df, output_paths, gr.Dropdown(choices=file_choices)

# Function to handle row selection for preview
def handle_tabular_row_selection(results_df, evt:gr.SelectData):
    
    if not evt:
        return None, "", ""

    if not isinstance(results_df, pd.DataFrame):
        return None, "", ""
    elif results_df.empty:
        return None, "", ""
    
    selected_index = evt.index[0]
    if selected_index >= len(results_df):
        return None, "", ""
    
    row = results_df.iloc[selected_index]
    return selected_index, row['Text1'], row['Text2']

# Function to clean duplicates from selected file
def clean_tabular_duplicates(file_name, results_df, output_folder):
    if not file_name or results_df.empty:
        return None
    
    # Get duplicate rows for this file
    file_duplicates = results_df[results_df['File1'] == file_name]['Row1'].tolist()
    
    if not file_duplicates:
        return None
    
    try:
        # Find the original file path
        # This is a simplified approach - in practice you might want to store file paths
        cleaned_file = remove_duplicate_rows_from_tabular_data(
            file_path=file_name,
            duplicate_rows=file_duplicates,
            output_folder=output_folder
        )
        return cleaned_file
    except Exception as e:
        print(f"Error cleaning duplicates: {e}")
        return None