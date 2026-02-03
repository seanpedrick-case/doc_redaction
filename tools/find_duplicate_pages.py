import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import pandas as pd
from gradio import Progress
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tools.config import MAX_SIMULTANEOUS_FILES
from tools.file_conversion import (
    convert_annotation_data_to_dataframe,
    fill_missing_box_ids_each_box,
    word_level_ocr_df_to_line_level_ocr_df,
)
from tools.file_redaction import redact_whole_pymupdf_page
from tools.helper_functions import OUTPUT_FOLDER
from tools.load_spacy_model_custom_recognisers import nlp
from tools.secure_path_utils import (
    secure_path_join,
    validate_folder_containment,
    validate_path_safety,
)

number_of_zeros_to_add_to_index = 7  # Number of zeroes to add between page number and line numbers to get a unique page/line index value
ID_MULTIPLIER = 100000
# Define the set of punctuation characters for efficient lookup
PUNCTUATION_TO_STRIP = {".", ",", "?", "!", ":", ";"}


def _normalize_page_to_int(page: Any) -> Optional[int]:
    """
    Convert a page identifier to an integer page number.
    Handles: int, numeric string, 'PageN_...' labels, and path/filenames.
    Returns None if no page number can be determined.
    """
    if page is None:
        return None
    try:
        return int(page)
    except (ValueError, TypeError):
        pass
    s = str(page).strip()
    # "Page1_File" style label
    m = re.search(r"Page(\d+)_", s, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Path/filename with _N.png
    from tools.secure_regex_utils import safe_extract_page_number_from_path

    n = safe_extract_page_number_from_path(s)
    if n is not None:
        return n
    # First contiguous digit sequence (e.g. "A17_FlightPlan_..." -> 17)
    m = re.search(r"(\d{1,10})", s)
    if m:
        return int(m.group(1))
    return None


def split_text_with_punctuation(text: str) -> List[str]:
    """
    A more concise version of the tokenization function using a single
    powerful regex with re.findall.
    """
    # This single regex pattern finds either:
    # 1. A sequence of one or more punctuation marks `[.,?!:;]+`
    # 2. OR a sequence of one or more characters that are NOT punctuation or whitespace `[^.,?!:;\s]+`
    pattern = re.compile(r"([.,?!:;]+|[^.,?!:;\s]+)")

    final_list = list()
    # We first split by whitespace to handle sentences correctly
    for word in text.split():
        # Then, for each whitespace-separated word, we tokenize it further
        final_list.extend(pattern.findall(word))

    return final_list


def extract_indices_from_page_ranges(
    results_df: pd.DataFrame,
    start_col: str = "Page2_Start_Page",
    end_col: str = "Page2_End_Page",
    modulo_divisor_number_of_zeros: int = number_of_zeros_to_add_to_index,  # Search for number of added
    converted_index: bool = False,  # Has the index been converted to the page_no + 0000 + line number format that needs the modulo divisor to convert back?
) -> List[int]:
    all_indices = set()
    int("1" + modulo_divisor_number_of_zeros * "0")

    for _, row in results_df.iterrows():
        start_page = row[start_col]
        end_page = row[end_col]
        for encoded_page_id in range(start_page, end_page + 1):
            if converted_index is True:
                original_page, original_index = _parse_page_line_id(
                    encoded_page_id
                )  # (encoded_page_id % modulo_divisor) - 1
            else:
                original_index = encoded_page_id

            all_indices.add(original_index)
    return sorted(list(all_indices))


def punctuation_at_word_text_end(word_level_df_orig: pd.DataFrame) -> bool:
    """
    Check the first 1000 rows of word_level_df_orig to see if any of the strings
    in 'word_text' end with a full stop '.', exclamation mark '!', or question mark '?',
    for strings that do not contain these characters alone.

    Args:
        word_level_df_orig (pd.DataFrame): DataFrame containing word-level OCR data with 'word_text' column

    Returns:
        bool: True if any strings end with punctuation marks, False otherwise
    """
    # Get the first 1000 rows or all rows if less than 1000
    sample_df = word_level_df_orig.head(1000)

    # Check if 'word_text' column exists
    if "word_text" not in sample_df.columns:
        return False

    # Define punctuation marks to check for
    punctuation_marks = [".", "!", "?"]

    # Check each word_text string
    for word_text in sample_df["word_text"]:
        if pd.isna(word_text) or not isinstance(word_text, str):
            continue

        # Skip strings that contain only punctuation marks
        if word_text.strip() in punctuation_marks:
            continue

        # Check if the string ends with any of the punctuation marks
        if any(word_text.rstrip().endswith(punct) for punct in punctuation_marks):
            return True

    return False


def run_full_search_and_analysis(
    search_query_text: str,
    word_level_df_orig: pd.DataFrame,
    similarity_threshold: float = 1,
    combine_pages: bool = False,
    min_word_count: int = 1,
    min_consecutive_pages: int = 1,
    greedy_match: bool = True,
    remake_index: bool = False,
    use_regex: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    """
    This function orchestrates the entire pipeline for finding duplicate pages based on a user's search query. It takes in the search query text, the original word-level OCR data, and various parameters to control the analysis. The function then:

    1. Converts the user's search query into a DataFrame format suitable for analysis.
    2. Prepares the main word-level OCR data for processing by converting it into the required format.
    3. Combines the search query DataFrame with the prepared OCR data DataFrame.
    4. Executes the similarity analysis on the combined data using the specified parameters such as similarity threshold, minimum word count, minimum consecutive pages, and greedy match strategy.

    Parameters:
    - search_query_text (str): The text entered by the user to search for in the OCR data. If use_regex=True, this is treated as a regex pattern.
    - word_level_df_orig (pd.DataFrame): The original DataFrame containing word-level OCR data.
    - similarity_threshold (float, optional): The minimum similarity score required for two pages to be considered duplicates. Defaults to 1.
    - combine_pages (bool, optional): A flag indicating whether to combine text from the same page number within a file. Defaults to False.
    - min_word_count (int, optional): The minimum number of words required for a page to be considered in the analysis. Defaults to 1.
    - min_consecutive_pages (int, optional): The minimum number of consecutive pages required to be considered a match. Defaults to 1.
    - greedy_match (bool, optional): A flag indicating whether to use a greedy strategy for matching consecutive pages. Defaults to True.
    - remake_index (bool, optional): A flag indicating whether to remake the index of the DataFrame during processing. Defaults to False.
    - use_regex (bool, optional): If True, treats search_query_text as a regex pattern instead of literal text. Defaults to False.
    - progress (gr.Progress, optional): A Progress object to track the progress of the operation. Defaults to a Progress object with track_tqdm set to True.
    """

    if len(search_query_text) < 3:
        raise Warning("Please use a search query with at least three letters.")
    if len(search_query_text) > 100:
        raise Warning("Please use a search query with at less than 100 characters.")

    # For regex mode, we handle the query differently
    if use_regex:
        # Validate regex pattern
        try:
            re.compile(search_query_text)
        except re.error as e:
            raise Warning(f"Invalid regex pattern: {e}")

        # For regex, we don't split into words - treat as single pattern
        # Create a minimal DataFrame structure for the regex pattern
        search_query_data = [
            (
                "user_search_query",
                pd.DataFrame({"page": [1], "text": [search_query_text], "line": [1]}),
            )
        ]
        query_word_length = 1  # For regex, we'll handle matching differently
        min_consecutive_pages = 1  # Regex matches can be variable length
    else:
        # Original literal text matching logic
        if punctuation_at_word_text_end(word_level_df_orig) is True:
            do_punctuation_split = False
        else:
            do_punctuation_split = True

        # Step 1: Process the user's search query string
        search_query_data, query_word_length = create_dataframe_from_string(
            search_query_text,
            file_name="user_search_query",
            split_words=True,
            split_punctuation=do_punctuation_split,
        )
        if not search_query_data:
            # Handle case where user submits an empty search string
            raise Warning("Could not convert search string to required format")

        if query_word_length > 25:
            # Handle case where user submits an empty search string
            raise Warning("Please use a query with less than 25 words")

        # Overwrite min_consecutive_pages with the search string length
        min_consecutive_pages = query_word_length

    # Create word index from reference table

    if word_level_df_orig.empty:
        raise gr.Error(
            "No word-level data to process. Please check that you have loaded in OCR data."
        )

    word_level_df_orig["index"] = word_level_df_orig.index
    word_level_df = word_level_df_orig.copy()

    # Step 2: Process the main word-level OCR DataFrame
    word_level_data = convert_word_level_df(word_level_df, file_name="source_document")

    # Step 3: Combine both data sources into one list
    all_data_to_process = search_query_data + word_level_data
    if not all_data_to_process:
        raise gr.Error("No data to process. Please check your inputs.")

    # Step 4: Run the combination logic
    combined_df, _, full_out_ocr_df = combine_ocr_dataframes(
        input_data=all_data_to_process,
        combine_pages=combine_pages,
        output_folder=None,  # No need to save this intermediate file
        remake_index=remake_index,
    )

    # Step 5: Run the final similarity analysis on the combined data
    results_df, duplicate_files, full_data = identify_similar_text_sequences(
        df_combined=combined_df,
        similarity_threshold=similarity_threshold,
        min_word_count=min_word_count,
        min_consecutive_pages=min_consecutive_pages,
        greedy_match=greedy_match,
        combine_pages=combine_pages,
        inter_file_only=True,
        do_text_clean=False,
        file1_name="user_search_query",
        file2_name="source_document",
        use_regex=use_regex,
        progress=progress,
    )

    print("Finished text search")

    # Map the results back to the reference data file
    if remake_index is True:
        results_df_index_list = extract_indices_from_page_ranges(
            results_df, converted_index=True
        )
    else:
        results_df_index_list = extract_indices_from_page_ranges(
            results_df, converted_index=False
        )

    word_level_df_out = word_level_df_orig.loc[
        word_level_df_orig["index"].isin(results_df_index_list)
    ]

    return word_level_df_out, duplicate_files, full_data


def create_all_data_to_process(
    converted_data: pd.DataFrame, other_data_list: List[Tuple]
):
    all_data_to_process = converted_data + other_data_list
    return all_data_to_process


def convert_word_level_df(
    word_level_df: pd.DataFrame, file_name: str = "converted_dataframe"
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Converts a word-level OCR DataFrame to the format for
    combine_ocr_dataframes.

    A simple renaming and selection of relevant columns

    Args:
        word_level_df (pd.DataFrame):
            A DataFrame containing detailed OCR output. Must include at least
            the columns: 'page', 'line', and 'word_text'.
        file_name (str, optional):
            A unique identifier or "dummy" filename to assign to the resulting
            data. Defaults to "converted_dataframe".

    Returns:
        List[Tuple[str, pd.DataFrame]]:
            A list containing a single tuple of (file_name, DataFrame), ready
            to be used as input for the combine_ocr_dataframes function. The
            DataFrame will have 'page' and 'text' columns.
    """
    # --- 1. Validate Input ---
    required_columns = ["page", "line", "word_text"]
    if not all(col in word_level_df.columns for col in required_columns):
        raise ValueError(
            f"Input DataFrame must contain all of the following columns: {required_columns}"
        )

    df = word_level_df.copy()

    # --- 2. Process the DataFrame ---
    # Ensure word_text is a string to allow for joining
    df["word_text"] = df["word_text"].astype(str)

    # Group by page and line number, then join the words with a space (not needed for word level search)
    # The result is a Series with a MultiIndex (page, line)
    # line_text_series = df.groupby(['page', 'line'])['word_text'].apply(' '.join)

    # Convert the Series back to a DataFrame and reset the index
    # line_level_df = line_text_series.reset_index()

    # Rename the aggregated column from 'word_text' to the required 'text'
    df = df.rename(columns={"word_text": "text"})

    # --- 3. Finalise the structure ---
    # We now have a DataFrame with columns [page, line, text].
    final_df = df[["page", "text"]]

    # --- 4. Package for output ---
    # Return in the required List[Tuple[str, DataFrame]] format
    return [(file_name, final_df)]


def create_dataframe_from_string(
    text_string: str,
    file_name: str = "user_search_query",
    page_number: int = 1,
    split_words: bool = False,
    split_punctuation: bool = True,
) -> Tuple[List[Tuple[str, pd.DataFrame]], int]:
    """
    Converts a string into a DataFrame compatible with combine_ocr_dataframes.

    Can operate in two modes:
    1. As a single-line document (default).
    2. As a multi-line document where each word from the string is a separate line.

    Args:
        text_string (str): The input text to be placed in the DataFrame.
        file_name (str, optional): A dummy filename to assign to this text.
                                   Defaults to "user_search_query".
        page_number (int, optional): A dummy page number to assign. Defaults to 1.
        split_words (bool, optional): If True, splits the input string by
                                      whitespace and creates a row for each word.
                                      If False (default), the entire string is
                                      treated as a single text entry.
        split_punctuation (bool, optional): If True, splits the 'end of sentence' punctuation off the end
                                      of the search query to match the reference data.

    Returns:
        Tuple[List[Tuple[str, pd.DataFrame]], int]:
            A list containing a single tuple: (file_name, DataFrame).
            The DataFrame has 'page' and 'text' columns. Also, an integer value indicating the number of words in the search string.
            Returns an empty list if the input string is empty or whitespace.
    """
    # Handle empty input gracefully, this works for both modes.
    if not text_string or not text_string.strip():
        print("Warning: Input string is empty. Returning an empty list.")
        return [], 0

    if split_words:
        # --- Split string into words, one per row, based on similar punctuation split technique used to create ocr_results_with_words objects ---
        if split_punctuation is True:
            words = split_text_with_punctuation(text_string)
        else:
            words = text_string.split()

        # words = text_string.split()
        len_words = len(words)
        data = {
            "page": [page_number]
            * len_words,  # Assign the same page number to every word
            "text": words,  # The list of words becomes the text column
        }
    else:
        # --- Entire string in one row ---
        len_words = 1
        data = {"page": [page_number], "text": [text_string]}

    # Create the DataFrame from the prepared data
    df = pd.DataFrame(data)

    df["line"] = df.index + 1

    # Return it in the required format: a list containing one (name, df) tuple
    return [(file_name, df)], len_words


def combine_ocr_dataframes(
    input_data: List[Tuple[str, pd.DataFrame]],
    combine_pages: bool = True,
    output_folder: str = OUTPUT_FOLDER,
    output_filename: str = "combined_ocr_output.csv",
    number_of_added_zeros: int = number_of_zeros_to_add_to_index,
    remake_index: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Combines text from multiple pandas DataFrames containing page and text columns.

    This function takes a list of (name, DataFrame) tuples, processes each DataFrame
    by grouping and concatenating text, and then combines them into a single DataFrame.

    Args:
        input_data (List[Tuple[str, pd.DataFrame]]):
            A list of tuples, where each tuple contains a unique identifier (like a filename)
            and a pandas DataFrame. Each DataFrame must have 'page' and 'text' columns.
        combine_pages (bool, optional):
            If True, text from the same page number within a file is joined into a
            single row. If False, each line of text gets its own row with a unique
            page identifier. Defaults to True.
        output_folder (str, optional):
            The folder where the combined CSV file will be saved. Defaults to OUTPUT_FOLDER.
        output_filename (str, optional):
            The name of the output CSV file. Defaults to "combined_ocr_output.csv".

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            A tuple containing:
            - The final combined and processed DataFrame.
            - A list containing the path to the saved output CSV file.
    """
    all_data = list()

    for file_identifier, df_initial in input_data:
        df = df_initial.copy()  # Work on a copy to avoid side effects

        # --- Validation ---
        if "page" not in df.columns or "text" not in df.columns:
            print(
                f"Warning: Skipping data for '{file_identifier}' - missing required columns 'page' and 'text'."
            )
            continue

        # --- Processing ---
        df["text"] = df["text"].fillna("").astype(str)

        if combine_pages:
            # Group by page and concatenate text into a single string
            processed_df = df.groupby("page")["text"].apply(" ".join).reset_index()
        else:
            if remake_index is True:
                # # Create a unique, sortable page ID for each line without combining
                # df['line_number_by_page'] = df.groupby('page').cumcount() + 1
                # df['original_page'] = df['page']
                # # Create a new page ID that combines page and line number for uniqueness
                # df['page'] = (
                #     df['page'].astype(str).str.zfill(number_of_added_zeros) +
                #     df['line_number_by_page'].astype(str).str.zfill(number_of_added_zeros)
                # ).astype(int)

                # Define the multiplier based on the max expected lines per page.
                # If you expect up to 99,999 lines, use 100,000.

                df["line_number_by_page"] = df.groupby("page").cumcount() + 1
                df["original_page"] = df["page"]

                # Create the new combined ID using arithmetic
                df["page"] = (df["original_page"] * ID_MULTIPLIER) + df[
                    "line_number_by_page"
                ]

            else:
                if "index" not in df.columns:
                    df["index"] = df.index
                df["page"] = df["index"]

            processed_df = df

        # Add the file identifier column
        processed_df["file"] = file_identifier
        all_data.append(processed_df)

    if not all_data:
        raise ValueError(
            "No valid DataFrames were processed. Ensure input data is not empty and DataFrames have 'page' and 'text' columns."
        )

    # --- Final Combination ---
    combined_df = pd.concat(all_data, ignore_index=True)

    # Reorder columns to a standard format, dropping intermediate columns
    final_columns = ["file", "page", "text"]
    if "original_page" in combined_df.columns:
        final_columns.append("original_page")  # Keep for context if created

    # Ensure all final columns exist before trying to select them
    existing_final_columns = [
        col for col in final_columns if col in combined_df.columns
    ]

    full_out_ocr_df = combined_df
    combined_df = combined_df.copy()[existing_final_columns]

    # --- Save Output ---
    output_files = list()
    if output_folder and output_filename:
        # Validate path safety before creating directories and files
        if not validate_folder_containment(output_folder, OUTPUT_FOLDER):
            raise ValueError(f"Unsafe output folder path: {output_folder}")
        if not validate_path_safety(output_filename):
            raise ValueError(f"Unsafe output filename: {output_filename}")

        # Normalize and validate the output folder path before using in os.makedirs
        normalized_output_folder = os.path.normpath(os.path.abspath(output_folder))
        # Double-check containment after normalization
        if not validate_folder_containment(normalized_output_folder, OUTPUT_FOLDER):
            raise ValueError(
                f"Unsafe normalized output folder path: {normalized_output_folder}"
            )

        # Assign the validated path back to output_folder to ensure all subsequent
        # operations use the secure, validated value
        output_folder = normalized_output_folder

        os.makedirs(output_folder, exist_ok=True)
        output_path = secure_path_join(output_folder, output_filename)
        combined_df.to_csv(output_path, index=False)
        output_files.append(output_path)
        print(f"Successfully combined data and saved to: {output_path}")

    return combined_df, output_files, full_out_ocr_df


def combine_ocr_output_text(
    input_files: Union[str, List[str]],
    combine_pages: bool = True,
    remake_index: bool = True,
    output_folder: str = OUTPUT_FOLDER,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Reads multiple OCR CSV files, combines them, and saves the result.

    This function serves as a wrapper that reads CSV files from paths and then
    uses the `combine_ocr_dataframes` function to perform the combination logic.

    Args:
        input_files (Union[str, List[str]]): A single file path or a list of file paths.
        combine_pages (bool, optional): See `combine_ocr_dataframes`. Defaults to True.
        output_folder (str, optional): See `combine_ocr_dataframes`. Defaults to OUTPUT_FOLDER.

    Returns:
        Tuple[pd.DataFrame, List[str]]: The combined DataFrame and the path to the output file.
    """
    if isinstance(input_files, str):
        file_paths_list = [input_files]
    else:
        file_paths_list = input_files

    data_to_process = list()
    i = 0
    first_ocr_df = pd.DataFrame()
    for file_path in file_paths_list:
        try:
            df = pd.read_csv(file_path)
            # Convert word-level OCR to line-level if user uploaded word-level file
            if "ocr_results_with_words" in os.path.basename(file_path) and (
                "word_text" in df.columns and "text" not in df.columns
            ):
                df = word_level_ocr_df_to_line_level_ocr_df(df)
            # Save the first OCR dataframe to save to the GUI
            if i == 0:
                first_ocr_df = df
                first_ocr_df_path = file_path
            i += 1
            # Use the base filename as the identifier
            file_identifier = os.path.basename(file_path)
            data_to_process.append((file_identifier, df))
        except FileNotFoundError:
            print(f"Warning: File not found, skipping: {file_path}")
        except Exception as e:
            print(f"Warning: Failed to read or process {file_path}. Error: {e}")

    if not data_to_process:
        raise ValueError("No valid CSV files could be read or processed.")

    df_combined, _, full_out_ocr_df = combine_ocr_dataframes(
        input_data=data_to_process,
        combine_pages=combine_pages,
        output_folder=output_folder,
        output_filename="combined_ocr_from_files.csv",  # Specific name for this path
        remake_index=remake_index,
    )

    # Call the core function with the loaded data
    return df_combined, first_ocr_df, first_ocr_df_path


def clean_and_stem_text_series(df: pd.DataFrame, column: str):
    """
    Clean and stem text columns in a data frame
    """

    def _clean_text(raw_text):
        from tools.secure_regex_utils import safe_clean_text

        clean = safe_clean_text(raw_text, remove_html=True)
        clean = " ".join(clean.split())
        # Join the cleaned words back into a string
        return clean

    # Function to apply lemmatisation and remove stopwords
    def _apply_lemmatization(text):
        doc = nlp(text)
        # Keep only alphabetic tokens and remove stopwords
        lemmatized_words = [
            token.lemma_ for token in doc if token.is_alpha and not token.is_stop
        ]
        return " ".join(lemmatized_words)

    df["text_clean"] = df[column].apply(_clean_text)

    df["text_clean"] = df["text_clean"].apply(_apply_lemmatization)

    return df


def map_metadata_single_page(
    similarity_df: pd.DataFrame,
    metadata_source_df: pd.DataFrame,
    preview_length: int = 200,
):
    """Helper to map metadata for single page results."""
    metadata_df = metadata_source_df[["file", "page", "text"]]
    results_df = similarity_df.merge(
        metadata_df, left_on="Page1_Index", right_index=True
    ).rename(columns={"file": "Page1_File", "page": "Page1_Page", "text": "Page1_Text"})
    results_df = results_df.merge(
        metadata_df, left_on="Page2_Index", right_index=True, suffixes=("_1", "_2")
    ).rename(columns={"file": "Page2_File", "page": "Page2_Page", "text": "Page2_Text"})
    results_df["Similarity_Score"] = results_df["Similarity_Score"].round(3)
    final_df = results_df[
        [
            "Page1_File",
            "Page1_Page",
            "Page2_File",
            "Page2_Page",
            "Similarity_Score",
            "Page1_Text",
            "Page2_Text",
        ]
    ]
    final_df = final_df.sort_values(
        ["Page1_File", "Page1_Page", "Page2_File", "Page2_Page"]
    )
    final_df["Page1_Text"] = final_df["Page1_Text"].str[:preview_length]
    final_df["Page2_Text"] = final_df["Page2_Text"].str[:preview_length]
    return final_df


def map_metadata_subdocument(
    subdocument_df: pd.DataFrame,
    metadata_source_df: pd.DataFrame,
    preview_length: int = 200,
):
    """Helper to map metadata for subdocument results."""
    metadata_df = metadata_source_df[["file", "page", "text"]]

    subdocument_df = subdocument_df.merge(
        metadata_df, left_on="Page1_Start_Index", right_index=True
    ).rename(
        columns={"file": "Page1_File", "page": "Page1_Start_Page", "text": "Page1_Text"}
    )
    subdocument_df = subdocument_df.merge(
        metadata_df[["page"]], left_on="Page1_End_Index", right_index=True
    ).rename(columns={"page": "Page1_End_Page"})
    subdocument_df = subdocument_df.merge(
        metadata_df, left_on="Page2_Start_Index", right_index=True
    ).rename(
        columns={"file": "Page2_File", "page": "Page2_Start_Page", "text": "Page2_Text"}
    )
    subdocument_df = subdocument_df.merge(
        metadata_df[["page"]], left_on="Page2_End_Index", right_index=True
    ).rename(columns={"page": "Page2_End_Page"})

    cols = [
        "Page1_File",
        "Page1_Start_Page",
        "Page1_End_Page",
        "Page2_File",
        "Page2_Start_Page",
        "Page2_End_Page",
        "Match_Length",
        "Page1_Text",
        "Page2_Text",
    ]

    # Add Avg_Similarity if it exists (it won't for greedy match unless we add it)
    if "Avg_Similarity" in subdocument_df.columns:
        subdocument_df["Avg_Similarity"] = subdocument_df["Avg_Similarity"].round(3)
        cols.insert(7, "Avg_Similarity")

    final_df = subdocument_df[cols]
    final_df = final_df.sort_values(
        ["Page1_File", "Page1_Start_Page", "Page2_File", "Page2_Start_Page"]
    )
    final_df["Page1_Text"] = final_df["Page1_Text"].str[:preview_length]
    final_df["Page2_Text"] = final_df["Page2_Text"].str[:preview_length]

    return final_df


def save_results_and_redaction_lists(
    final_df: pd.DataFrame, output_folder: str, combine_pages: bool = True
) -> list:
    """
    Saves the main results DataFrame and generates per-file redaction lists.
    This function is extracted to be reusable.

    Args:
        final_df (pd.DataFrame): The DataFrame containing the final match results.
        output_folder (str): The folder to save the output files.
        combine_pages (bool, optional): Boolean to check whether the text from pages have been combined into one, or if instead the duplicate match has been conducted line by line.

    Returns:
        list: A list of paths to all generated files.
    """
    # Validate the output_folder path for security
    if not validate_folder_containment(output_folder, OUTPUT_FOLDER):
        raise ValueError(f"Invalid or unsafe output folder path: {output_folder}")

    output_paths = list()

    # Use secure path operations to prevent path injection
    try:
        output_folder_path = Path(output_folder).resolve()
        # Validate that the resolved path is within the trusted OUTPUT_FOLDER using robust containment check
        if not validate_folder_containment(str(output_folder_path), OUTPUT_FOLDER):
            raise ValueError(
                f"Output folder path {output_folder} is outside the trusted directory {OUTPUT_FOLDER}"
            )
        output_folder_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise ValueError(f"Cannot create output directory {output_folder}: {e}")

    if final_df.empty:
        print("No matches to save.")
        return []

    # 1. Save the main results DataFrame using secure path operations
    similarity_file_output_path = secure_path_join(
        output_folder_path, "page_similarity_results.csv"
    )
    final_df.to_csv(similarity_file_output_path, index=False, encoding="utf-8-sig")

    output_paths.append(str(similarity_file_output_path))

    # 2. Save per-file redaction lists
    # Use 'Page2_File' as the source of duplicate content
    if combine_pages is True:
        grouping_col = "Page2_File"
        if grouping_col not in final_df.columns:
            print(
                "Warning: 'Page2_File' column not found. Cannot generate redaction lists."
            )
            return output_paths

        for redact_file, group in final_df.groupby(grouping_col):
            # Sanitize the filename to prevent path injection
            output_file_name_stem = Path(redact_file).stem
            output_file_name = output_file_name_stem + "_pages_to_redact.csv"
            # Use secure path operations for the output file
            output_file_path = secure_path_join(output_folder_path, output_file_name)

            all_pages_to_redact = set()
            is_subdocument_match = "Page2_Start_Page" in group.columns

            if is_subdocument_match:
                for _, row in group.iterrows():
                    pages_in_range = range(
                        int(row["Page2_Start_Page"]), int(row["Page2_End_Page"]) + 1
                    )
                    all_pages_to_redact.update(pages_in_range)
            else:
                pages = group["Page2_Page"].unique()
                all_pages_to_redact.update(pages)

            if all_pages_to_redact:
                redaction_df = pd.DataFrame(
                    sorted(list(all_pages_to_redact)), columns=["Page_to_Redact"]
                )
                redaction_df.to_csv(output_file_path, header=False, index=False)

                output_paths.append(str(output_file_path))
                print(f"Redaction list for {redact_file} saved to {output_file_name}")

    return output_paths


def _sequences_match(query_seq: List[str], ref_seq: List[str]) -> bool:
    """
    Helper function to compare two sequences of tokens with punctuation flexibility.

    Returns True if the sequences match according to the rules:
    1. An exact match is a match.
    2. A reference token also matches a query token if it is the query token
       followed by a single character from PUNCTUATION_TO_STRIP. This rule does not
       apply if the reference token consists only of punctuation.
    """
    if len(query_seq) != len(ref_seq):
        return False

    for query_token, ref_token in zip(query_seq, ref_seq):
        # Rule 1: Check for a direct, exact match first (most common case)
        if query_token == ref_token:
            continue

        # Rule 2: Check for the flexible punctuation match
        # - The reference token must be longer than 1 character
        # - Its last character must be in our punctuation set
        # - The token without its last character must match the query token
        if (
            len(ref_token) > 1
            and ref_token[-1] in PUNCTUATION_TO_STRIP
            and ref_token[:-1] == query_token
        ):
            continue

        # If neither rule applies, the tokens don't match, so the sequence doesn't match.
        return False

    # If the loop completes, every token has matched.
    return True


def find_consecutive_sequence_matches(
    df_filtered: pd.DataFrame,
    search_file_name: str,
    reference_file_name: str,
    use_regex: bool = False,
) -> pd.DataFrame:
    """
    Finds all occurrences of a consecutive sequence of tokens from a search file
    within a larger reference file.

    This function is designed for order-dependent matching, not "bag-of-words" similarity.

    Args:
        df_filtered: The DataFrame containing all tokens, with 'file' and 'text_clean' columns.
        search_file_name: The name of the file containing the search query sequence.
        reference_file_name: The name of the file to search within.
        use_regex: If True, treats the search query as a regex pattern instead of literal tokens.

    Returns:
        A DataFrame with two columns ('Page1_Index', 'Page2_Index') mapping the
        consecutive match, or an empty DataFrame if no match is found.
    """

    # Step 1: Isolate the data for each file
    search_df = df_filtered[df_filtered["file"] == search_file_name]
    reference_df = df_filtered[df_filtered["file"] == reference_file_name]

    if search_df.empty or reference_df.empty:
        print("Error: One or both files not found or are empty.")
        return pd.DataFrame(columns=["Page1_Index", "Page2_Index"])

    if use_regex:
        # Regex mode: Extract pattern and search in combined text
        # Get the regex pattern from the search query (should be in 'text' column, not 'text_clean')
        # We need to get it from the original 'text' column if available, otherwise use 'text_clean'
        if "text" in search_df.columns:
            regex_pattern = search_df["text"].iloc[0]
        else:
            regex_pattern = search_df["text_clean"].iloc[0]

        # Join reference tokens back into text for regex searching
        # Use original 'text' column if available to preserve original formatting (important for emails, etc.)
        # Otherwise fall back to 'text_clean'
        if "text" in reference_df.columns:
            reference_tokens = reference_df["text"].tolist()
        else:
            reference_tokens = reference_df["text_clean"].tolist()
        reference_indices = reference_df.index.tolist()

        # Concatenate ALL tokens into a single continuous string with smart spacing
        # Rules:
        # - Words are joined with single spaces
        # - Punctuation (periods, commas, etc.) touches adjacent tokens directly (no spaces)
        # Example: ["Hi", ".", "How", "are", "you", "?", "Great"] -> "Hi.How are you?Great"
        # This allows regex patterns to span multiple tokens naturally while preserving word boundaries

        def is_punctuation_only(token):
            """Check if token contains only punctuation characters"""
            if not token:
                return False
            # Check if all characters are punctuation (using string.punctuation or our set)
            import string

            return all(c in string.punctuation for c in token)

        def starts_with_punctuation(token):
            """Check if token starts with punctuation"""
            if not token:
                return False
            import string

            return token[0] in string.punctuation

        def ends_with_punctuation(token):
            """Check if token ends with punctuation"""
            if not token:
                return False
            import string

            return token[-1] in string.punctuation

        # Build the concatenated string and position mapping
        reference_text_parts = []
        char_to_token_map = []
        current_pos = 0

        for idx, token in enumerate(reference_tokens):
            # Determine if we need a space before this token
            needs_space_before = False
            if idx > 0:  # Not the first token
                prev_token = reference_tokens[idx - 1]
                # Add space if:
                # - Current token is not punctuation-only AND
                # - Previous token is not punctuation-only AND
                # - Previous token didn't end with punctuation AND
                # - Current token doesn't start with punctuation
                if (
                    not is_punctuation_only(token)
                    and not is_punctuation_only(prev_token)
                    and not ends_with_punctuation(prev_token)
                    and not starts_with_punctuation(token)
                ):
                    needs_space_before = True

            # Add space if needed
            if needs_space_before:
                current_pos += 1  # Account for the space

            # Record token position in the concatenated string
            token_start_in_text = current_pos
            token_end_in_text = current_pos + len(token)
            char_to_token_map.append(
                (token_start_in_text, token_end_in_text, reference_indices[idx])
            )

            # Add token to the concatenated string
            if needs_space_before:
                reference_text_parts.append(" " + token)
            else:
                reference_text_parts.append(token)

            # Move position forward by token length (and space if added)
            current_pos = token_end_in_text

        # Join all parts to create the final concatenated string
        reference_text = "".join(reference_text_parts)

        # Find all regex matches
        try:
            pattern = re.compile(regex_pattern, re.IGNORECASE)
            matches = list(pattern.finditer(reference_text))
        except re.error as e:
            print(f"Error compiling regex pattern: {e}")
            gr.Warning(f"Invalid regex pattern: {e}")
            return pd.DataFrame(
                columns=["Page1_Index", "Page2_Index", "Similarity_Score"]
            )

        if not matches:
            print("No regex matches found")
            gr.Info("No regex matches found")
            return pd.DataFrame(
                columns=["Page1_Index", "Page2_Index", "Similarity_Score"]
            )

        all_found_matches = []
        query_index = search_df.index[0]  # Use the first (and only) query index

        # Optimize overlap detection for large documents
        # Instead of checking every token for every match (O(m*n)), we can use the fact that
        # char_to_token_map is sorted by position. For each match, we only need to check
        # tokens that could possibly overlap.

        # For each regex match found in the concatenated string:
        # 1. Get the match's start and end character positions
        # 2. Find all tokens whose character ranges overlap with the match
        # 3. Include all overlapping tokens in the results
        # This ensures patterns spanning multiple tokens are captured correctly

        # Optimization: Use a set to track which tokens we've already found
        # This prevents duplicates if multiple matches overlap the same tokens
        found_token_indices = set()

        for match in matches:
            match_start = match.start()
            match_end = match.end()

            # Find all tokens that overlap with this match
            # A token overlaps if: token_start < match_end AND token_end > match_start
            # Optimization: Since char_to_token_map is sorted by start position,
            # we can stop early once we pass match_end, but we still need to check
            # tokens that start before match_end (they might extend into the match)
            matching_token_indices = []
            for token_start, token_end, token_idx in char_to_token_map:
                # Early exit optimization: if token starts after match ends, no more overlaps possible
                # (This works because tokens are processed in order)
                if token_start >= match_end:
                    break

                # Check if token overlaps with match (not disjoint)
                if (
                    token_end > match_start
                ):  # token_start < match_end already checked by break above
                    matching_token_indices.append(token_idx)

            # Create matches for all tokens that overlap with the regex match
            # This ensures patterns spanning multiple tokens are captured
            for token_idx in matching_token_indices:
                if token_idx not in found_token_indices:
                    all_found_matches.append((query_index, token_idx, 1))
                    found_token_indices.add(token_idx)

        print(
            f"Found {len(matches)} regex match(es) spanning {len(set(idx for _, idx, _ in all_found_matches))} token(s)"
        )

    else:
        # Original literal token matching logic
        # Step 2: Convert the token data into lists for easy comparison.
        # We need both the text tokens and their original global indices.
        query_tokens = search_df["text_clean"].tolist()
        query_indices = search_df.index.tolist()

        reference_tokens = reference_df["text_clean"].tolist()
        reference_indices = reference_df.index.tolist()

        query_len = len(query_tokens)
        all_found_matches = list()

        print(f"Searching for a sequence of {query_len} tokens...")

        # Step 3: Use a "sliding window" to search for the query sequence in the reference list.
        for i in range(len(reference_tokens) - query_len + 1):
            # The "window" is a slice of the reference list that is the same size as the query
            window = reference_tokens[i : i + query_len]

            # Step 4: If the window matches the query with or without punctuation on end
            if _sequences_match(query_tokens, window):

                # Get the global indices for this entire matching block
                matching_reference_indices = reference_indices[i : i + query_len]

                # Create the mapping between query indices and the found reference indices
                for j in range(query_len):
                    all_found_matches.append(
                        (query_indices[j], matching_reference_indices[j], 1)
                    )

                # If you only want the *first* match, you can uncomment the next line:
                # break

    if not all_found_matches:
        print("No matches found")
        gr.Info("No matches found")
        return pd.DataFrame(columns=["Page1_Index", "Page2_Index", "Similarity_Score"])

    # Step 5: Create the final DataFrame in the desired format
    result_df = pd.DataFrame(
        all_found_matches, columns=["Page1_Index", "Page2_Index", "Similarity_Score"]
    )
    return result_df


def identify_similar_text_sequences(
    df_combined: pd.DataFrame,
    similarity_threshold: float = 1,
    min_word_count: int = 1,
    min_consecutive_pages: int = 1,
    greedy_match: bool = True,
    combine_pages: bool = False,
    inter_file_only: bool = False,
    do_text_clean: bool = True,
    file1_name: str = "",
    file2_name: str = "",
    output_folder: str = OUTPUT_FOLDER,
    use_regex: bool = False,
    progress=Progress(track_tqdm=True),
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Identifies similar pages. Uses a highly optimized path for inter_file_only=True.
    """
    progress(0.1, desc="Processing and filtering text")

    if do_text_clean:
        df = clean_and_stem_text_series(
            df_combined, "text"
        )  # Will produce the column 'text_clean'
    else:
        df = df_combined.copy()
        df["text_clean"] = df[
            "text"
        ].str.lower()  # .str.replace(r'[^\w\s]', '', regex=True)

    df["word_count"] = df["text_clean"].str.split().str.len().fillna(0)
    # df['word_count'] = pd.to_numeric(df['word_count'], errors='coerce').fillna(0).astype('int64')

    # ensure min_word_count is an int (e.g., from Gradio/text input)
    try:
        min_word_count = int(min_word_count)
    except (TypeError, ValueError):
        min_word_count = 0  # or raise/log, depending on your preference

    original_row_count = len(df)
    df_filtered = df[df["word_count"] >= min_word_count].copy()
    df_filtered.reset_index(drop=True, inplace=True)

    print(
        f"Filtered out {original_row_count - len(df_filtered)} pages with fewer than {min_word_count} words."
    )
    if len(df_filtered) < 2:
        return pd.DataFrame(), [], df_combined

    # Similarity calculated differently if comparing between files only (inter_file_only==True), or within the same file
    if inter_file_only:

        progress(0.2, desc="Finding direct text matches...")

        # base_similarity_df = _debug_similarity_between_two_files(df_filtered, vectorizer, similarity_threshold, file1_name, file2_name)
        base_similarity_df = find_consecutive_sequence_matches(
            df_filtered, file1_name, file2_name, use_regex=use_regex
        )
        if base_similarity_df.empty:
            return pd.DataFrame(), [], df_combined

    else:
        # Use the original, simpler path for all-to-all comparisons (including intra-file).
        vectorizer = TfidfVectorizer()
        print("Standard Path: Calculating all-to-all similarity.")
        progress(0.2, desc="Vectorising text...")
        tfidf_matrix = vectorizer.fit_transform(df_filtered["text_clean"])

        progress(0.3, desc="Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=False)
        coo_matrix = similarity_matrix.tocoo()

        similar_pages = [
            (r, c, v)
            for r, c, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
            if r < c and v >= similarity_threshold
        ]

        if not similar_pages:
            return pd.DataFrame(), [], df_combined

        base_similarity_df = pd.DataFrame(
            similar_pages, columns=["Page1_Index", "Page2_Index", "Similarity_Score"]
        )

    progress(0.7, desc="Aggregating results based on matching strategy")

    if greedy_match or min_consecutive_pages > 1:
        # Sort the dataframe to ensure consecutive pages are adjacent
        similarity_df = base_similarity_df

        # A new sequence starts if the difference from the previous row is not (1, 1)
        # is_consecutive will be True if a row continues the sequence, False if it's a new one.
        is_consecutive = (similarity_df["Page1_Index"].diff() == 1) & (
            similarity_df["Page2_Index"].diff() == 1
        )

        # Use cumsum() on the inverted boolean series to create a unique ID for each block.
        # Every time a 'False' appears (a new block starts), the sum increases.
        block_id = is_consecutive.eq(False).cumsum()

        # Group by this block ID
        grouped = similarity_df.groupby(block_id)

        # Aggregate each group to get the start, end, and length of the match
        agg_results = grouped.agg(
            Page1_Start_Index=("Page1_Index", "first"),
            Page2_Start_Index=("Page2_Index", "first"),
            Page1_End_Index=("Page1_Index", "last"),
            Page2_End_Index=("Page2_Index", "last"),
            Match_Length=("Page1_Index", "size"),
            Avg_Similarity=("Similarity_Score", "mean"),
        ).reset_index(drop=True)

        # If greedy_match=True, we keep all matches. If min_consecutive_pages > 1, we filter.
        if greedy_match and min_consecutive_pages <= 1:
            subdocument_df = agg_results
        else:
            # This handles the case for min_consecutive_pages > 1
            subdocument_df = agg_results[
                agg_results["Match_Length"] >= min_consecutive_pages
            ].copy()

        if subdocument_df.empty:
            gr.Info("No matches found")
            return pd.DataFrame(), [], df_combined

        final_df = map_metadata_subdocument(subdocument_df, df_filtered)
    else:
        print("Finding single page matches, not greedy (min_consecutive_pages=1)")
        # This part of your code would handle the non-sequential case
        final_df = map_metadata_single_page(base_similarity_df, df_filtered)
        # subdocument_df = final_df # To align variable names for saving

        if final_df.empty:
            gr.Info("No matches found")
            return pd.DataFrame(), [], df_combined

    progress(0.9, desc="Saving output files")

    output_paths = save_results_and_redaction_lists(
        final_df, output_folder, combine_pages
    )

    gr.Info(f"Found {final_df.shape[0]} match(es)")
    print(f"Found {final_df.shape[0]} match(es)")

    return final_df, output_paths, df_combined


def handle_selection_and_preview(
    evt: gr.SelectData, results_df: pd.DataFrame, full_duplicate_data_by_file: dict
):
    """
    This single function handles a user selecting a row. It:
    1. Determines the selected row index.
    2. Calls the show_page_previews function to get the text data.
    3. Returns all the necessary outputs for the UI.
    """
    # If the user deselects, the event might be None.
    if not evt:
        return None, None, None  # Clear state and both preview panes

    # 1. Get the selected index
    selected_index = evt.index[0]

    # 2. Get the preview data
    page1_data, page2_data = show_page_previews(
        full_duplicate_data_by_file, results_df, evt
    )

    # 3. Return all three outputs in the correct order
    return selected_index, page1_data, page2_data


def exclude_match(
    results_df: pd.DataFrame, selected_index: int, output_folder=OUTPUT_FOLDER
):
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


def run_duplicate_analysis(
    files: list[str],
    threshold: float,
    min_words: int,
    min_consecutive: int,
    greedy_match: bool,
    all_page_line_level_ocr_results_df_base: pd.DataFrame,
    ocr_df_paths_list: list[str],
    combine_pages: bool = True,
    output_folder: str = OUTPUT_FOLDER,
    preview_length: int = 500,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Main wrapper function to orchestrate the duplicate page analysis process.
    It handles file loading, text combination, similarity identification,
    and result saving.

    Args:
        files (list[str]): A list of file paths (PDFs, etc.) to be analyzed for duplicate content.
        threshold (float): The similarity threshold (0.0 to 1.0) above which text segments are considered duplicates.
        min_words (int): The minimum number of words a text segment must contain to be included in the analysis.
        min_consecutive (int): The minimum number of consecutive pages that must match for a sequence to be considered a duplicate.
        greedy_match (bool): If True, uses a greedy matching strategy for identifying consecutive sequences.
        combine_pages (bool, optional): If True, text from multiple pages is combined into larger segments for analysis. Defaults to True.
        all_page_line_level_ocr_results_df_base (pd.DataFrame): The base dataframe containing the OCR results.
        ocr_df_paths_list (list[str]): A list of file paths to the OCR results.
        output_folder (str, optional): The directory where the similarity results and redaction lists will be saved. Defaults to OUTPUT_FOLDER.
        preview_length (int, optional): The maximum number of characters to display in the text preview panes. Defaults to 500.
        progress (gr.Progress, optional): A Gradio progress tracker object to display progress in the UI.
    """

    if not files:
        raise Warning("Please upload files to analyse.")

    if isinstance(files, str):
        files = [files]

    if not ocr_df_paths_list:
        ocr_df_paths_list = []

    if len(files) > MAX_SIMULTANEOUS_FILES:
        out_message = f"Number of files to deduplicate is greater than {MAX_SIMULTANEOUS_FILES}. Please submit a smaller number of files."
        print(out_message)
        raise Exception(out_message)

    start_time = time.time()

    task_textbox = "deduplicate"

    progress(0, desc="Combining input files...")
    df_combined, first_ocr_df, first_ocr_df_path = combine_ocr_output_text(
        files, combine_pages=combine_pages, output_folder=output_folder
    )

    # Replace current OCR app components if currently empty
    if all_page_line_level_ocr_results_df_base.empty:
        all_page_line_level_ocr_results_df_base = first_ocr_df
        if first_ocr_df_path not in ocr_df_paths_list:
            ocr_df_paths_list.append(first_ocr_df_path)

    if df_combined.empty:
        raise Warning("No data found in the uploaded files.")

    # Call the main analysis function with the new parameter
    results_df, output_paths, full_df = identify_similar_text_sequences(
        df_combined=df_combined,
        similarity_threshold=threshold,
        min_word_count=min_words,
        min_consecutive_pages=int(min_consecutive),
        greedy_match=greedy_match,
        combine_pages=combine_pages,
        output_folder=output_folder,
        progress=progress,
    )

    full_df["text"] = full_df["text"].astype(str)

    # Clip text to first 200 characters
    full_df["text"] = full_df["text"].str[:preview_length]

    # Preprocess full_data (without preview text) for fast access (run once)
    full_data_by_file = {
        file: df.sort_values("page").set_index("page")
        for file, df in full_df.drop(["text_clean"], axis=1).groupby("file")
    }

    if results_df.empty:
        gr.Info("No duplicate pages found, no results returned.")

    end_time = time.time()
    processing_time = round(end_time - start_time, 2)

    return (
        results_df,
        output_paths,
        full_data_by_file,
        processing_time,
        task_textbox,
        all_page_line_level_ocr_results_df_base,
        ocr_df_paths_list,
    )


def show_page_previews(
    full_data_by_file: dict,
    results_df: pd.DataFrame,
    evt: gr.SelectData,
    preview_length: int = 500,
):
    """
    Optimized version using pre-partitioned and indexed full_data.
    Triggered when a user selects a row in the results DataFrame.
    """
    if not full_data_by_file or results_df is None or not evt:
        return None, None

    selected_row = results_df.iloc[evt.index[0], :]

    is_subdocument_match = "Page1_Start_Page" in selected_row

    if is_subdocument_match:
        file1, start1, end1 = (
            selected_row["Page1_File"],
            selected_row["Page1_Start_Page"],
            selected_row["Page1_End_Page"],
        )
        file2, start2, end2 = (
            selected_row["Page2_File"],
            selected_row["Page2_Start_Page"],
            selected_row["Page2_End_Page"],
        )

        page1_data = full_data_by_file[file1].loc[start1:end1, ["text"]].reset_index()
        page2_data = full_data_by_file[file2].loc[start2:end2, ["text"]].reset_index()

    else:
        file1, page1 = selected_row["Page1_File"], selected_row["Page1_Page"]
        file2, page2 = selected_row["Page2_File"], selected_row["Page2_Page"]

        page1_data = full_data_by_file[file1].loc[[page1], ["text"]].reset_index()
        page2_data = full_data_by_file[file2].loc[[page2], ["text"]].reset_index()

    page1_data["text"] = page1_data["text"].astype(str)
    page2_data["text"] = page2_data["text"].astype(str)

    page1_data["text"] = page1_data["text"].str[:preview_length]
    page2_data["text"] = page2_data["text"].str[:preview_length]

    return page1_data[["page", "text"]], page2_data[["page", "text"]]


def get_page_image_info(page_num: int, page_sizes: List[Dict]) -> Optional[Dict]:
    """
    Finds and returns the size and path information for a specific page.
    """
    return next((size for size in page_sizes if size["page"] == page_num), None)


def add_new_annotations_to_existing_page_annotations(
    all_annotations: List[Dict], image_path: str, new_annotation_boxes: List[Dict]
) -> Tuple[List[Dict], Dict]:
    """
    Adds a list of new annotation boxes to the annotations for a specific page.

    If the page already has annotations, it extends the list of boxes. If not,
    it creates a new entry for the page.

    Args:
        all_annotations (List[Dict]): The current list of all annotation groups.
        image_path (str): The identifier for the image/page.
        new_annotation_boxes (List[Dict]): A list of new annotation boxes to add.

    Returns:
        Tuple[List[Dict], Dict]: A tuple containing:
            - The updated list of all annotation groups.
            - The annotation group representing the newly added boxes.
    """
    # Find the annotation group for the current page/image
    current_page_group = next(
        (
            annot_group
            for annot_group in all_annotations
            if annot_group["image"] == image_path
        ),
        None,
    )

    if current_page_group:
        # Page already has annotations, so extend the list with the new boxes
        current_page_group["boxes"].extend(new_annotation_boxes)
    else:
        # This is the first set of annotations for this page, create a new group
        new_group = {"image": image_path, "boxes": new_annotation_boxes}
        all_annotations.append(new_group)

    # This object represents all annotations that were just added for this page
    newly_added_annotation_group = {"image": image_path, "boxes": new_annotation_boxes}

    return all_annotations, newly_added_annotation_group


def apply_whole_page_redactions_from_list(
    duplicate_page_numbers_df_or_list: pd.DataFrame | list[str],
    doc_file_name_with_extension_textbox: str,
    review_file_state: pd.DataFrame,
    duplicate_output_paths: list[str],
    pymupdf_doc: object,
    page_sizes: list[dict],
    all_existing_annotations: list[dict],
    combine_pages: bool = True,
    new_annotations_with_bounding_boxes: List[dict] = list(),
):
    """
    This function applies redactions to whole pages based on a provided list of duplicate page numbers. It supports two modes of operation: combining pages and not combining pages. When combining pages is enabled, it attempts to identify duplicate pages across different files and applies redactions accordingly. If combining pages is disabled, it relies on new annotations with bounding boxes to determine which pages to redact. The function utilises a PyMuPDF document object to manipulate the PDF file, and it also considers the sizes of pages to ensure accurate redaction application.

    Args:
        duplicate_page_numbers_df_or_list: A DataFrame or list containing page numbers identified as duplicates (supports both legacy DataFrame and new Dropdown list format).
        doc_file_name_with_extension_textbox (str): The name of the document file with its extension.
        review_file_state (pd.DataFrame): The current state of the review file.
        duplicate_output_paths (list[str]): A list of paths to files containing duplicate page information.
        pymupdf_doc (object): A PyMuPDF document object representing the PDF file.
        page_sizes (list[dict]): A list of dictionaries containing page size information.
        all_existing_annotations (list[dict]): A list of all existing annotations in the document.
        combine_pages (bool, optional): A flag indicating whether to combine pages for redaction. Defaults to True.
        new_annotations_with_bounding_boxes (List[dict], optional): A list of new annotations with bounding boxes. Defaults to an empty list.
    """
    if all_existing_annotations is None:
        all_existing_annotations = list()

    if new_annotations_with_bounding_boxes is None:
        new_annotations_with_bounding_boxes = list()

    all_annotations = all_existing_annotations.copy()

    if not pymupdf_doc:
        message = "No document file currently under review. Please upload the relevant PDF on the Review redactions tab under the top 'Upload PDFs/images...' section to apply duplicate redactions."
        print(f"Warning: {message}")
        raise Warning(message)

    list_whole_pages_to_redact = list()

    if combine_pages is True:
        # Get list of pages to redact from either dataframe, list, or file
        # Handle both DataFrame (legacy) and list (new Dropdown format)
        if isinstance(duplicate_page_numbers_df_or_list, pd.DataFrame):
            if not duplicate_page_numbers_df_or_list.empty:
                df = duplicate_page_numbers_df_or_list
                # Only redact Page2 (duplicate) pages, not Page1 (original)
                page_cols = [
                    c
                    for c in (
                        "Page2_Start_Page",
                        "Page2_End_Page",
                    )
                    if c in df.columns
                ]
                if page_cols:
                    list_whole_pages_to_redact = []
                    if (
                        "Page2_Start_Page" in df.columns
                        and "Page2_End_Page" in df.columns
                    ):
                        for _, row in df.iterrows():
                            try:
                                start_p = int(row["Page2_Start_Page"])
                                end_p = int(row["Page2_End_Page"])
                                list_whole_pages_to_redact.extend(
                                    range(start_p, end_p + 1)
                                )
                            except (ValueError, TypeError):
                                pass
                    else:
                        for col in page_cols:
                            list_whole_pages_to_redact.extend(
                                df[col].dropna().astype(int).tolist()
                            )
                else:
                    list_whole_pages_to_redact = df.iloc[:, 0].tolist()
        elif (
            isinstance(duplicate_page_numbers_df_or_list, list)
            and len(duplicate_page_numbers_df_or_list) > 0
        ):
            # Dropdown component returns a list directly
            try:
                # Try to convert to integers for page numbers
                list_whole_pages_to_redact = [
                    int(item) for item in duplicate_page_numbers_df_or_list if item
                ]
            except (ValueError, TypeError):
                # Fall back to string list if conversion fails
                list_whole_pages_to_redact = [
                    str(item) for item in duplicate_page_numbers_df_or_list if item
                ]
        elif duplicate_output_paths:
            expected_duplicate_pages_to_redact_name = (
                f"{doc_file_name_with_extension_textbox}"
            )
            # Substitute out '_for_review' from the expected filename to successfully modify existing redactions files
            expected_duplicate_pages_to_redact_name = (
                expected_duplicate_pages_to_redact_name.replace(
                    "_redactions_for_review", ""
                ).replace(".pdf", "")
            )

            whole_pages_list = pd.DataFrame()  # Initialize empty DataFrame
            chosen_output_file = None

            # Prefer *_pages_to_redact.csv (only Page2/duplicate pages); avoid using
            # page_similarity_results.csv as first match, which would otherwise add
            # both Page1 and Page2 and redact the wrong pages.
            PAGES_TO_REDACT_SUFFIX = "_pages_to_redact"
            for output_file in duplicate_output_paths:
                if isinstance(output_file, str):
                    file_name_from_path = os.path.basename(output_file)
                else:
                    file_name_from_path = getattr(
                        output_file, "name", str(output_file).split(os.sep)[-1]
                    )
                file_name_from_path = file_name_from_path.replace(
                    "_redactions_for_review", ""
                )
                if expected_duplicate_pages_to_redact_name in file_name_from_path:
                    if PAGES_TO_REDACT_SUFFIX in file_name_from_path:
                        whole_pages_list = pd.read_csv(output_file, header=None)
                        chosen_output_file = output_file
                        break
                    if chosen_output_file is None:
                        chosen_output_file = output_file
            if whole_pages_list.empty and chosen_output_file is not None:
                whole_pages_list = pd.read_csv(chosen_output_file, header=None)

            if not whole_pages_list.empty:
                output_file = chosen_output_file
                # Support both formats:
                # 1) page_similarity_results.csv: has header row. Only Page2_* columns
                #    are the duplicate pages to redact; Page1_* is the "original" and
                #    must not be redacted (same semantics as _pages_to_redact.csv).
                # 2) *_pages_to_redact.csv: no header, first column is page numbers
                first_col = whole_pages_list.iloc[:, 0]
                if (
                    len(whole_pages_list.columns) >= 6
                    and first_col.iloc[0] == "Page1_File"
                ):
                    # Likely page_similarity_results.csv with header in first row
                    df_with_header = pd.read_csv(output_file, header=0)
                    # Only use Page2 columns (duplicate side); do not redact Page1 (original)
                    page_cols = [
                        c
                        for c in (
                            "Page2_Start_Page",
                            "Page2_End_Page",
                        )
                        if c in df_with_header.columns
                    ]
                    if page_cols:
                        list_whole_pages_to_redact = []
                        if (
                            "Page2_Start_Page" in df_with_header.columns
                            and "Page2_End_Page" in df_with_header.columns
                        ):
                            for _, row in df_with_header.iterrows():
                                try:
                                    start_p = int(row["Page2_Start_Page"])
                                    end_p = int(row["Page2_End_Page"])
                                    list_whole_pages_to_redact.extend(
                                        range(start_p, end_p + 1)
                                    )
                                except (ValueError, TypeError):
                                    pass
                        else:
                            for col in page_cols:
                                list_whole_pages_to_redact.extend(
                                    df_with_header[col].dropna().astype(int).tolist()
                                )
                    else:
                        list_whole_pages_to_redact = first_col.tolist()
                else:
                    list_whole_pages_to_redact = whole_pages_list.iloc[:, 0].tolist()
        else:
            message = "No relevant list of whole pages to redact found."
            print(message)
            raise Warning(message)

        list_whole_pages_to_redact = list(set(list_whole_pages_to_redact))

        # When called from "Apply match" with combine_pages=True, the page list may come from
        # in_fully_redacted_list_state (upload), which can be empty. Fall back to deriving
        # pages from new_annotations_with_bounding_boxes so suggested duplicate pages are applied.
        if not list_whole_pages_to_redact and new_annotations_with_bounding_boxes:
            from tools.secure_regex_utils import safe_extract_page_number_from_path

            list_whole_pages_to_redact = list()
            for annotation in new_annotations_with_bounding_boxes:
                page_num = safe_extract_page_number_from_path(annotation.get("image"))
                if page_num is not None:
                    list_whole_pages_to_redact.append(page_num + 1)
                else:
                    img = annotation.get("image", "")
                    print(f"Warning: Could not extract page number from {img!r}")
            list_whole_pages_to_redact = list(set(list_whole_pages_to_redact))

    else:
        if not new_annotations_with_bounding_boxes:
            message = "Can't find any new annotations to add"
            print(message)
            raise Warning(message)

        list_whole_pages_to_redact = list()
        for annotation in new_annotations_with_bounding_boxes:
            from tools.secure_regex_utils import safe_extract_page_number_from_path

            page_num = safe_extract_page_number_from_path(annotation["image"])
            if page_num is not None:
                page = page_num + 1
                list_whole_pages_to_redact.append(page)
            else:
                print(
                    f"Warning: Could not extract page number from {annotation['image']}"
                )

        list_whole_pages_to_redact = list(set(list_whole_pages_to_redact))

    new_annotations = list()
    # Process each page for redaction (page may be int or string label/filename)
    for page in list_whole_pages_to_redact:
        try:
            page_num = _normalize_page_to_int(page)
            if page_num is None:
                print(f"Warning: Could not parse page number from {page!r}, skipping.")
                continue
            page_index = page_num - 1
            if not (0 <= page_index < len(pymupdf_doc)):
                print(f"Page {page_num} is out of bounds, skipping.")
                continue

            page_info = get_page_image_info(page_num, page_sizes)
            if not page_info:
                print(f"Page {page_num} not found in page_sizes, skipping.")
                continue

            image_path = page_info["image_path"]
            page_annotation_group = next(
                (g for g in all_annotations if g["image"] == image_path), None
            )
            if page_annotation_group and any(
                box["label"] == "Whole page" for box in page_annotation_group["boxes"]
            ):
                print(
                    f"Whole page redaction for page {page_num} already exists, skipping."
                )
                continue

            boxes_to_add = list()

            pymupdf_page = pymupdf_doc[page_index]

            if combine_pages is True:
                whole_page_box = redact_whole_pymupdf_page(
                    rect_height=page_info["cropbox_height"],
                    rect_width=page_info["cropbox_width"],
                    page=pymupdf_page,
                    border=0.005,
                    redact_pdf=False,
                )
                boxes_to_add.append(whole_page_box)
            else:
                # Find the specific annotation group that matches the current page's image path
                relevant_box_group = next(
                    (
                        group
                        for group in new_annotations_with_bounding_boxes
                        if group.get("image") == image_path
                    ),
                    None,  # Default to None if no match is found
                )

                # Check if we found a matching group of boxes for this page
                if relevant_box_group:
                    boxes_to_add.extend(relevant_box_group["boxes"])
                else:
                    # This case would be unexpected, but it's good to handle.
                    # It means a page was in list_whole_pages_to_redact but had no
                    # corresponding boxes generated in new_annotations_with_bounding_boxes.
                    print(
                        f"Warning: No new annotation boxes found for page {page_num} ({image_path})."
                    )

            # === Use the modified helper function to add a LIST of boxes ===
            all_annotations, new_annotations_for_page = (
                add_new_annotations_to_existing_page_annotations(
                    all_annotations=all_annotations,
                    image_path=image_path,
                    new_annotation_boxes=boxes_to_add,  # Pass the list here
                )
            )

            new_annotations_for_page = fill_missing_box_ids_each_box(
                new_annotations_for_page
            )
            new_annotations.append(new_annotations_for_page)

        except Exception as e:
            print(f"Error processing page {page}: {str(e)}")
            continue

    whole_page_review_file = convert_annotation_data_to_dataframe(new_annotations)

    if whole_page_review_file.empty:
        message = "No new whole page redactions were added."
        print(message)
        gr.Info(message)
        return review_file_state, all_annotations

    expected_cols = [
        "image",
        "page",
        "label",
        "color",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "text",
        "id",
    ]
    for col in expected_cols:
        if col not in review_file_state.columns:
            review_file_state[col] = pd.NA
        if col not in whole_page_review_file.columns:
            whole_page_review_file[col] = pd.NA

    review_file_out = pd.concat(
        [review_file_state, whole_page_review_file], ignore_index=True
    )
    review_file_out = review_file_out.sort_values(
        by=["page", "ymin", "xmin"]
    ).reset_index(drop=True)
    review_file_out = review_file_out.drop_duplicates(
        subset=["page", "label", "text", "id"], keep="first"
    )

    out_message = (
        f"Successfully created {whole_page_review_file.shape[0]} whole page redactions."
    )
    print(out_message)
    gr.Info(out_message)

    return review_file_out, all_annotations


def _parse_page_line_id(combined_id: int) -> Tuple[int, int]:
    """Parses a combined ID using modular arithmetic."""
    if int(combined_id) < ID_MULTIPLIER:
        # Handle cases where page is 0 (or just an edge case)
        return 0, combined_id

    page = combined_id // ID_MULTIPLIER
    line = combined_id % ID_MULTIPLIER
    return page, line


def create_annotation_objects_from_duplicates(
    duplicates_df: pd.DataFrame,
    ocr_results_df: pd.DataFrame,
    page_sizes: List[Dict],
    combine_pages: bool = False,
) -> List[Dict]:
    """
    Creates structured annotation objects from duplicate line ranges, mapping
    page numbers to image paths.

    Args:
        duplicates_df (pd.DataFrame): DataFrame with duplicate ranges.
        ocr_results_df (pd.DataFrame): DataFrame with OCR results.
        page_sizes (List[Dict]): A list of dictionaries mapping page numbers to image paths and other metadata. Expected format: [{"page": 1, "image_path": "path/to/img.png", ...}]
        combine_pages (bool): A boolean that determines whether in previous functions, all text from a page was combined (True). This function will only run if this is False.

    Returns:
        List[Dict]: A list of dictionaries, where each dict represents a page and its list of annotation boxes, in the format: [{"image": "path/to/img.png", "boxes": [...]}, ...]
    """
    final_output = list()

    if duplicates_df.empty:
        gr.Info("No duplicates found")
        return final_output
    if ocr_results_df.empty:
        raise Warning(
            "No OCR results found for file under review. Please upload relevant OCR_output file and original PDF document on the review tab."
        )

    if combine_pages is False:
        page_to_image_map = {item["page"]: item["image_path"] for item in page_sizes}

        # Prepare OCR Data: line_number_by_page must match the duplicate detection
        # pipeline. That pipeline uses raw CSV row order (no sort) and cumcount()+1,
        # so "line 4" = 4th row on the page in CSV order. If the OCR has a "line"
        # column (1-based per page), use it so we redact the same line the match
        # refers to. Otherwise fall back to (page, top, left) order + cumcount()+1.
        if "line" in ocr_results_df.columns:
            ocr_results_df = ocr_results_df.copy()
            ocr_results_df["line_number_by_page"] = (
                pd.to_numeric(ocr_results_df["line"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
        else:
            ocr_results_df = ocr_results_df.sort_values(
                by=["page", "top", "left"]
            ).reset_index(drop=True)
            ocr_results_df["line_number_by_page"] = (
                ocr_results_df.groupby("page").cumcount() + 1
            )

        annotations_by_page = defaultdict(list)

        # Detect format: subdocument (Page2_Start_Page / Page2_End_Page) vs single-page (Page2_Page)
        is_subdocument_format = (
            "Page2_Start_Page" in duplicates_df.columns
            and "Page2_End_Page" in duplicates_df.columns
        )

        for _, row in duplicates_df.iterrows():
            if is_subdocument_format:
                start_page, start_line = _parse_page_line_id(row["Page2_Start_Page"])
                end_page, end_line = _parse_page_line_id(row["Page2_End_Page"])

                if start_page == end_page:
                    condition = (ocr_results_df["page"] == start_page) & (
                        ocr_results_df["line_number_by_page"].between(
                            start_line, end_line
                        )
                    )
                else:
                    cond_start = (ocr_results_df["page"] == start_page) & (
                        ocr_results_df["line_number_by_page"] >= start_line
                    )
                    cond_middle = ocr_results_df["page"].between(
                        start_page + 1, end_page - 1
                    )
                    cond_end = (ocr_results_df["page"] == end_page) & (
                        ocr_results_df["line_number_by_page"] <= end_line
                    )
                    condition = cond_start | cond_middle | cond_end
            else:
                # Single-page format (min_consecutive_pages=1, not greedy): Page2_Page only
                if "Page2_Page" not in row.index:
                    print(
                        "Warning: duplicates_df has neither Page2_Start_Page/Page2_End_Page nor Page2_Page; skipping row."
                    )
                    continue
                page_num = int(row["Page2_Page"])
                condition = ocr_results_df["page"] == page_num

            lines_to_annotate = ocr_results_df[condition]

            # Build and group annotation boxes by page number (this logic is unchanged)
            for _, line_row in lines_to_annotate.iterrows():
                box = {
                    "label": "Duplicate text",
                    "color": (0, 0, 0),
                    "xmin": line_row["left"],
                    "ymin": line_row["top"],
                    "xmax": line_row["left"] + line_row["width"],
                    "ymax": line_row["top"] + line_row["height"],
                    "text": line_row["text"],
                    "id": "",  # to be filled in after
                }
                page_number = line_row["page"]

                annotations_by_page[page_number].append(box)

        # --- Format the final output list using the page-to-image map ---

        # Sort by page number for a predictable order
        for page_num, boxes in sorted(annotations_by_page.items()):
            # Look up the image path using the page number
            image_path = page_to_image_map.get(page_num)

            if image_path:
                page_boxes = {"image": image_path, "boxes": boxes}

                # Fill in missing IDs for the new data entries
                page_boxes = fill_missing_box_ids_each_box(page_boxes)

                # Add the annotation group using 'image' as the key
                final_output.append(page_boxes)
            else:
                # Handle cases where a page might not have a corresponding image path
                print(
                    f"Warning: Page {page_num} found in OCR data but has no corresponding "
                    f"entry in the 'page_sizes' object. This page's annotations will be skipped."
                )

    return final_output
