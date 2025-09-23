import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import pandas as pd
from gradio import Progress
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tools.config import (
    DO_INITIAL_TABULAR_DATA_CLEAN,
    MAX_SIMULTANEOUS_FILES,
    MAX_TABLE_ROWS,
    REMOVE_DUPLICATE_ROWS,
)
from tools.data_anonymise import initial_clean
from tools.helper_functions import OUTPUT_FOLDER, read_file
from tools.load_spacy_model_custom_recognisers import nlp

if REMOVE_DUPLICATE_ROWS == "True":
    REMOVE_DUPLICATE_ROWS = True
else:
    REMOVE_DUPLICATE_ROWS = False


def clean_and_stem_text_series(
    df: pd.DataFrame,
    column: str,
    do_initial_clean_dup: bool = DO_INITIAL_TABULAR_DATA_CLEAN,
):
    """
    Clean and stem text columns in a data frame for tabular data
    """

    # Function to apply lemmatisation and remove stopwords
    def _apply_lemmatization(text):
        doc = nlp(text)
        # Keep only alphabetic tokens and remove stopwords
        lemmatized_words = [
            token.lemma_ for token in doc if token.is_alpha and not token.is_stop
        ]
        return " ".join(lemmatized_words)

    if do_initial_clean_dup:
        df["text_clean"] = initial_clean(df[column])

    df["text_clean"] = df["text_clean"].apply(_apply_lemmatization)
    df["text_clean"] = df[
        column
    ].str.lower()  # .str.replace(r'[^\w\s]', '', regex=True)

    return df


def convert_tabular_data_to_analysis_format(
    df: pd.DataFrame, file_name: str, text_columns: List[str] = None
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
    # if text_columns is None:
    #     # Auto-detect text columns (string type columns)
    #     print(f"No text columns given for {file_name}")
    #     return []
    #     text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()

    text_columns = [col for col in text_columns if col in df.columns]

    if not text_columns:
        print(f"No text columns found in {file_name}")
        return list()

    # Create a copy to avoid modifying original
    df_copy = df.copy()

    # Create a combined text column from all text columns
    df_copy["combined_text"] = (
        df_copy[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    )

    # Add row identifier
    df_copy["row_id"] = df_copy.index

    # Create the format expected by the duplicate detection system
    # Using 'row_number' as row number and 'text' as the combined text
    processed_df = pd.DataFrame(
        {
            "row_number": df_copy["row_id"],
            "text": df_copy["combined_text"],
            "file": file_name,
        }
    )

    # Add original row data for reference
    for col in text_columns:
        processed_df[f"original_{col}"] = df_copy[col]

    return [(file_name, processed_df)]


def find_duplicate_cells_in_tabular_data(
    input_files: List[str],
    similarity_threshold: float = 0.95,
    min_word_count: int = 3,
    text_columns: List[str] = [],
    output_folder: str = OUTPUT_FOLDER,
    do_initial_clean_dup: bool = DO_INITIAL_TABULAR_DATA_CLEAN,
    remove_duplicate_rows: bool = REMOVE_DUPLICATE_ROWS,
    in_excel_tabular_sheets: str = "",
    progress: Progress = Progress(track_tqdm=True),
) -> Tuple[pd.DataFrame, List[str], Dict[str, pd.DataFrame]]:
    """
    Find duplicate cells/text in tabular data files (CSV, XLSX, Parquet).

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

    all_data_to_process = list()
    full_data_by_file = dict()
    file_paths = list()

    # Process each file
    for file_path in input_files:
        try:
            if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                temp_df = pd.DataFrame()

                # Try finding each sheet in the given list until a match is found
                for sheet_name in in_excel_tabular_sheets:
                    temp_df = read_file(file_path, excel_sheet_name=sheet_name)

                    # If sheet was successfully_loaded
                    if not temp_df.empty:

                        if temp_df.shape[0] > MAX_TABLE_ROWS:
                            out_message = f"Number of rows in {file_path} for sheet {sheet_name} is greater than {MAX_TABLE_ROWS}. Please submit a smaller file."
                            print(out_message)
                            raise Exception(out_message)

                        file_name = os.path.basename(file_path) + "_" + sheet_name
                        file_paths.append(file_path)

                        # Convert to analysis format
                        processed_data = convert_tabular_data_to_analysis_format(
                            temp_df, file_name, text_columns
                        )

                        if processed_data:
                            all_data_to_process.extend(processed_data)
                            full_data_by_file[file_name] = processed_data[0][1]

                    temp_df = pd.DataFrame()
            else:
                temp_df = read_file(file_path)

                if temp_df.shape[0] > MAX_TABLE_ROWS:
                    out_message = f"Number of rows in {file_path} is greater than {MAX_TABLE_ROWS}. Please submit a smaller file."
                    print(out_message)
                    raise Exception(out_message)

                file_name = os.path.basename(file_path)
                file_paths.append(file_path)

                # Convert to analysis format
                processed_data = convert_tabular_data_to_analysis_format(
                    temp_df, file_name, text_columns
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
    combined_df = pd.concat(
        [data[1] for data in all_data_to_process], ignore_index=True
    )

    combined_df = combined_df.drop_duplicates(subset=["row_number", "file"])

    progress(0.3, desc="Cleaning and preparing text...")

    # Clean and prepare text
    combined_df = clean_and_stem_text_series(
        combined_df, "text", do_initial_clean_dup=do_initial_clean_dup
    )

    # Filter by minimum word count
    combined_df["word_count"] = (
        combined_df["text_clean"].str.split().str.len().fillna(0)
    )
    combined_df = combined_df[combined_df["word_count"] >= min_word_count].copy()

    if len(combined_df) < 2:
        return pd.DataFrame(), [], full_data_by_file

    progress(0.4, desc="Calculating similarities...")

    # Calculate similarities
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_df["text_clean"])
    similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=False)

    # Find similar pairs
    coo_matrix = similarity_matrix.tocoo()
    similar_pairs = [
        (r, c, v)
        for r, c, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
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

        results_data.append(
            {
                "File1": row1_data["file"],
                "Row1": int(row1_data["row_number"]),
                "File2": row2_data["file"],
                "Row2": int(row2_data["row_number"]),
                "Similarity_Score": round(similarity, 3),
                "Text1": (
                    row1_data["text"][:200] + "..."
                    if len(row1_data["text"]) > 200
                    else row1_data["text"]
                ),
                "Text2": (
                    row2_data["text"][:200] + "..."
                    if len(row2_data["text"]) > 200
                    else row2_data["text"]
                ),
                "Original_Index1": row1,
                "Original_Index2": row2,
            }
        )

    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values(["File1", "Row1", "File2", "Row2"])

    progress(0.9, desc="Saving results...")

    # Save results
    output_paths = save_tabular_duplicate_results(
        results_df,
        output_folder,
        file_paths,
        remove_duplicate_rows=remove_duplicate_rows,
        in_excel_tabular_sheets=in_excel_tabular_sheets,
    )

    gr.Info(f"Found {len(results_df)} duplicate cell matches")

    return results_df, output_paths, full_data_by_file


def save_tabular_duplicate_results(
    results_df: pd.DataFrame,
    output_folder: str,
    file_paths: List[str],
    remove_duplicate_rows: bool = REMOVE_DUPLICATE_ROWS,
    in_excel_tabular_sheets: List[str] = [],
) -> List[str]:
    """
    Save tabular duplicate detection results to files.

    Args:
        results_df (pd.DataFrame): Results DataFrame
        output_folder (str): Output folder path
        file_paths (List[str]): List of file paths
        remove_duplicate_rows (bool): Whether to remove duplicate rows
        in_excel_tabular_sheets (str): Name of the Excel sheet to save the results to
    Returns:
        List[str]: List of output file paths
    """
    output_paths = list()
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(exist_ok=True)

    if results_df.empty:
        print("No duplicate matches to save.")
        return list()

    # Save main results
    results_file = output_folder_path / "tabular_duplicate_results.csv"
    results_df.to_csv(results_file, index=False, encoding="utf-8-sig")
    output_paths.append(str(results_file))

    # Group results by original file to handle Excel files properly
    excel_files_processed = dict()  # Track which Excel files have been processed

    # Save per-file duplicate lists
    for file_name, group in results_df.groupby("File2"):
        # Check for matches with original file names
        for original_file in file_paths:
            original_file_name = os.path.basename(original_file)

            if original_file_name in file_name:
                original_file_extension = os.path.splitext(original_file)[-1]
                if original_file_extension in [".xlsx", ".xls"]:

                    # Split the string using a regex to handle both .xlsx_ and .xls_ delimiters
                    # The regex r'\.xlsx_|\.xls_' correctly matches either ".xlsx_" or ".xls_" as a delimiter.
                    parts = re.split(r"\.xlsx_|\.xls_", os.path.basename(file_name))
                    # The sheet name is the last part after splitting
                    file_sheet_name = parts[-1]

                    file_path = original_file

                    # Initialize Excel file tracking if not already done
                    if file_path not in excel_files_processed:
                        excel_files_processed[file_path] = {
                            "sheets_data": dict(),
                            "all_sheets": list(),
                            "processed_sheets": set(),
                        }

                    # Read the original Excel file to get all sheet names
                    if not excel_files_processed[file_path]["all_sheets"]:
                        try:
                            excel_file = pd.ExcelFile(file_path)
                            excel_files_processed[file_path][
                                "all_sheets"
                            ] = excel_file.sheet_names
                        except Exception as e:
                            print(f"Error reading Excel file {file_path}: {e}")
                            continue

                    # Read the current sheet
                    df = read_file(file_path, excel_sheet_name=file_sheet_name)

                    # Create duplicate rows file for this sheet
                    file_stem = Path(file_name).stem
                    duplicate_rows_file = (
                        output_folder_path
                        / f"{file_stem}_{file_sheet_name}_duplicate_rows.csv"
                    )

                    # Get unique row numbers to remove
                    rows_to_remove = sorted(group["Row2"].unique())
                    duplicate_df = pd.DataFrame({"Row_to_Remove": rows_to_remove})
                    duplicate_df.to_csv(duplicate_rows_file, index=False)
                    output_paths.append(str(duplicate_rows_file))

                    # Process the sheet data
                    df_cleaned = df.copy()
                    df_cleaned["duplicated"] = False
                    df_cleaned.loc[rows_to_remove, "duplicated"] = True
                    if remove_duplicate_rows:
                        df_cleaned = df_cleaned.drop(index=rows_to_remove)

                    # Store the processed sheet data
                    excel_files_processed[file_path]["sheets_data"][
                        file_sheet_name
                    ] = df_cleaned
                    excel_files_processed[file_path]["processed_sheets"].add(
                        file_sheet_name
                    )

                else:
                    file_sheet_name = ""
                    file_path = original_file
                    print("file_path after match:", file_path)
                    file_base_name = os.path.basename(file_path)
                    df = read_file(file_path)

                    file_stem = Path(file_name).stem
                    duplicate_rows_file = (
                        output_folder_path / f"{file_stem}_duplicate_rows.csv"
                    )

                    # Get unique row numbers to remove
                    rows_to_remove = sorted(group["Row2"].unique())
                    duplicate_df = pd.DataFrame({"Row_to_Remove": rows_to_remove})
                    duplicate_df.to_csv(duplicate_rows_file, index=False)
                    output_paths.append(str(duplicate_rows_file))

                    df_cleaned = df.copy()
                    df_cleaned["duplicated"] = False
                    df_cleaned.loc[rows_to_remove, "duplicated"] = True
                    if remove_duplicate_rows:
                        df_cleaned = df_cleaned.drop(index=rows_to_remove)

                    file_ext = os.path.splitext(file_name)[-1]

                    if file_ext in [".parquet"]:
                        output_path = os.path.join(
                            output_folder, f"{file_base_name}_deduplicated.parquet"
                        )
                        df_cleaned.to_parquet(output_path, index=False)
                    else:
                        output_path = os.path.join(
                            output_folder, f"{file_base_name}_deduplicated.csv"
                        )
                        df_cleaned.to_csv(
                            output_path, index=False, encoding="utf-8-sig"
                        )

                    output_paths.append(str(output_path))
                break

    # Process Excel files to create complete deduplicated files
    for file_path, file_data in excel_files_processed.items():
        try:
            # Create output filename
            file_base_name = os.path.splitext(os.path.basename(file_path))[0]
            file_ext = os.path.splitext(file_path)[-1]
            output_path = os.path.join(
                output_folder, f"{file_base_name}_deduplicated{file_ext}"
            )

            # Create Excel writer
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # Write all sheets
                for sheet_name in file_data["all_sheets"]:
                    if sheet_name in file_data["processed_sheets"]:
                        # Use the processed (deduplicated) version
                        file_data["sheets_data"][sheet_name].to_excel(
                            writer, sheet_name=sheet_name, index=False
                        )
                    else:
                        # Use the original sheet (no duplicates found)
                        original_df = read_file(file_path, excel_sheet_name=sheet_name)
                        original_df.to_excel(writer, sheet_name=sheet_name, index=False)

            output_paths.append(str(output_path))
            print(f"Created deduplicated Excel file: {output_path}")

        except Exception as e:
            print(f"Error creating deduplicated Excel file for {file_path}: {e}")
            continue

    return output_paths


def remove_duplicate_rows_from_tabular_data(
    file_path: str,
    duplicate_rows: List[int],
    output_folder: str = OUTPUT_FOLDER,
    in_excel_tabular_sheets: List[str] = [],
    remove_duplicate_rows: bool = REMOVE_DUPLICATE_ROWS,
) -> str:
    """
    Remove duplicate rows from a tabular data file.

    Args:
        file_path (str): Path to the input file
        duplicate_rows (List[int]): List of row indices to remove
        output_folder (str): Output folder for cleaned file
        in_excel_tabular_sheets (str): Name of the Excel sheet to save the results to
        remove_duplicate_rows (bool): Whether to remove duplicate rows
    Returns:
        str: Path to the cleaned file
    """
    try:
        # Load the file
        df = read_file(
            file_path,
            excel_sheet_name=in_excel_tabular_sheets if in_excel_tabular_sheets else "",
        )

        # Remove duplicate rows (0-indexed)
        df_cleaned = df.drop(index=duplicate_rows).reset_index(drop=True)

        # Save cleaned file
        file_name = os.path.basename(file_path)
        file_stem = os.path.splitext(file_name)[0]
        file_ext = os.path.splitext(file_name)[-1]

        output_path = os.path.join(output_folder, f"{file_stem}_deduplicated{file_ext}")

        if file_ext in [".xlsx", ".xls"]:
            df_cleaned.to_excel(
                output_path,
                index=False,
                sheet_name=in_excel_tabular_sheets if in_excel_tabular_sheets else [],
            )
        elif file_ext in [".parquet"]:
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
    text_columns: List[str] = [],
    output_folder: str = OUTPUT_FOLDER,
    do_initial_clean_dup: bool = DO_INITIAL_TABULAR_DATA_CLEAN,
    remove_duplicate_rows: bool = REMOVE_DUPLICATE_ROWS,
    in_excel_tabular_sheets: List[str] = [],
    progress: Progress = Progress(track_tqdm=True),
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
        text_columns=text_columns if text_columns else [],
        output_folder=output_folder,
        do_initial_clean_dup=do_initial_clean_dup,
        in_excel_tabular_sheets=(
            in_excel_tabular_sheets if in_excel_tabular_sheets else []
        ),
        remove_duplicate_rows=remove_duplicate_rows,
    )


# Function to update column choices when files are uploaded
def update_tabular_column_choices(files, in_excel_tabular_sheets: List[str] = []):
    if not files:
        return gr.update(choices=[])

    all_columns = set()
    for file in files:
        try:
            file_extension = os.path.splitext(file.name)[-1]
            if file_extension in [".xlsx", ".xls"]:
                for sheet_name in in_excel_tabular_sheets:
                    df = read_file(file.name, excel_sheet_name=sheet_name)
                    text_cols = df.select_dtypes(
                        include=["object", "string"]
                    ).columns.tolist()
                    all_columns.update(text_cols)
            else:
                df = read_file(file.name)
                text_cols = df.select_dtypes(
                    include=["object", "string"]
                ).columns.tolist()
                all_columns.update(text_cols)

            # Get text columns
            text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

            all_columns.update(text_cols)
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
            continue

    return gr.Dropdown(choices=sorted(list(all_columns)))


# Function to handle tabular duplicate detection
def run_tabular_duplicate_detection(
    files,
    threshold,
    min_words,
    text_columns,
    output_folder: str = OUTPUT_FOLDER,
    do_initial_clean_dup: bool = DO_INITIAL_TABULAR_DATA_CLEAN,
    in_excel_tabular_sheets: List[str] = [],
    remove_duplicate_rows: bool = REMOVE_DUPLICATE_ROWS,
):
    if not files:
        print("No files uploaded")
        return pd.DataFrame(), [], gr.Dropdown(choices=[]), 0, "deduplicate"

    start_time = time.time()

    task_textbox = "deduplicate"

    # If output folder doesn't end with a forward slash, add one
    if not output_folder.endswith("/"):
        output_folder = output_folder + "/"

    file_paths = list()
    if isinstance(files, str):
        # If 'files' is a single string, treat it as a list with one element
        file_paths.append(files)
    elif isinstance(files, list):
        # If 'files' is a list, iterate through its elements
        for f_item in files:
            if isinstance(f_item, str):
                # If an element is a string, it's a direct file path
                file_paths.append(f_item)
            elif hasattr(f_item, "name"):
                # If an element has a '.name' attribute (e.g., a Gradio File object), use its name
                file_paths.append(f_item.name)
            else:
                # Log a warning for unexpected element types within the list
                print(
                    f"Warning: Skipping an element in 'files' list that is neither a string nor has a '.name' attribute: {type(f_item)}"
                )
    elif hasattr(files, "name"):
        # Handle the case where a single file object (e.g., gr.File) is passed directly, not in a list
        file_paths.append(files.name)
    else:
        # Raise an error for any other unexpected type of the 'files' argument itself
        raise TypeError(
            f"Unexpected type for 'files' argument: {type(files)}. Expected str, list of str/file objects, or a single file object."
        )

    if len(file_paths) > MAX_SIMULTANEOUS_FILES:
        out_message = f"Number of files to deduplicate is greater than {MAX_SIMULTANEOUS_FILES}. Please submit a smaller number of files."
        print(out_message)
        raise Exception(out_message)

    results_df, output_paths, full_data = run_tabular_duplicate_analysis(
        files=file_paths,
        threshold=threshold,
        min_words=min_words,
        text_columns=text_columns if text_columns else [],
        output_folder=output_folder,
        do_initial_clean_dup=do_initial_clean_dup,
        in_excel_tabular_sheets=(
            in_excel_tabular_sheets if in_excel_tabular_sheets else None
        ),
        remove_duplicate_rows=remove_duplicate_rows,
    )

    # Update file choices for cleaning
    file_choices = list(set([f for f in file_paths]))

    end_time = time.time()
    processing_time = round(end_time - start_time, 2)

    return (
        results_df,
        output_paths,
        gr.Dropdown(choices=file_choices),
        processing_time,
        task_textbox,
    )


# Function to handle row selection for preview
def handle_tabular_row_selection(results_df, evt: gr.SelectData):

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
    return selected_index, row["Text1"], row["Text2"]


# Function to clean duplicates from selected file
def clean_tabular_duplicates(
    file_name,
    results_df,
    output_folder,
    in_excel_tabular_sheets: str = "",
    remove_duplicate_rows: bool = REMOVE_DUPLICATE_ROWS,
):
    if not file_name or results_df.empty:
        return None

    # Get duplicate rows for this file
    file_duplicates = results_df[results_df["File2"] == file_name]["Row2"].tolist()

    if not file_duplicates:
        return None

    try:
        # Find the original file path
        # This is a simplified approach - in practice you might want to store file paths
        cleaned_file = remove_duplicate_rows_from_tabular_data(
            file_path=file_name,
            duplicate_rows=file_duplicates,
            output_folder=output_folder,
            in_excel_tabular_sheets=in_excel_tabular_sheets,
            remove_duplicate_rows=remove_duplicate_rows,
        )
        return cleaned_file
    except Exception as e:
        print(f"Error cleaning duplicates: {e}")
        return None
