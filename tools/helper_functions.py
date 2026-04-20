import logging
import os
import platform
import random
import re
import string
import sys
import tempfile
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import List, Set

import boto3
import gradio as gr
import numpy as np
import pandas as pd
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
)
from fastapi import FastAPI

from tools.aws_functions import download_file_from_s3, upload_file_to_s3
from tools.config import (
    AWS_LLM_PII_OPTION,
    AWS_PII_OPTION,
    AWS_USER_POOL_ID,
    BEDROCK_LLM_INPUT_COST,
    BEDROCK_LLM_INPUT_TOKENS_PER_PAGE,
    BEDROCK_LLM_OUTPUT_COST,
    BEDROCK_LLM_OUTPUT_TOKENS_PER_PAGE,
    BEDROCK_VLM_INPUT_COST,
    BEDROCK_VLM_OUTPUT_COST,
    BEDROCK_VLM_PIXELS_PER_INPUT_TOKEN,
    BEDROCK_VLM_TEXT_EXTRACT_OPTION,
    COST_CODES_PATH,
    CUSTOM_HEADER,
    CUSTOM_HEADER_VALUE,
    DEFAULT_COST_CODE,
    DEFAULT_LANGUAGE,
    DEFAULT_LOCAL_OCR_MODEL,
    DOCUMENT_REDACTION_BUCKET,
    INFERENCE_SERVER_PII_OPTION,
    INPUT_FOLDER,
    LANGUAGE_CHOICES,
    LANGUAGE_MAP,
    LOCAL_OCR_MODEL_OPTIONS,
    LOCAL_OCR_MODEL_TEXT_EXTRACT_OPTION,
    LOCAL_PII_OPTION,
    LOCAL_TRANSFORMERS_LLM_PII_OPTION,
    NO_REDACTION_PII_OPTION,
    OUTPUT_COST_CODES_PATH,
    OUTPUT_FOLDER,
    RUN_AWS_FUNCTIONS,
    S3_COST_CODES_PATH,
    S3_OUTPUTS_FOLDER,
    SAVE_OUTPUTS_TO_S3,
    SELECTABLE_TEXT_EXTRACT_OPTION,
    SESSION_OUTPUT_FOLDER,
    SHOW_FEEDBACK_BUTTONS,
    TEXTRACT_JOBS_LOCAL_LOC,
    TEXTRACT_JOBS_S3_LOC,
    TEXTRACT_TEXT_EXTRACT_OPTION,
    TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER,
    TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER,
    VLM_MAX_IMAGE_SIZE,
    aws_comprehend_language_choices,
    convert_string_to_boolean,
    ensure_folder_within_app_directory,
    textract_language_choices,
)
from tools.secure_path_utils import sanitize_filename, secure_path_join


def reset_state_vars():
    return (
        [],
        pd.DataFrame(),
        pd.DataFrame(),
        0,
        "",
        None,
        [],
        [],
        pd.DataFrame(),
        pd.DataFrame(),
        [],
        [],
        "",
        False,
        0,
        [],
        [],
        0,  # latest_file_completed_num: reset to 0 at start of document redaction
        0,  # LLM total input tokens
        0,  # LLM total output tokens
        0,  # VLM total input tokens
        0,  # VLM total output tokens
    )


def reset_ocr_results_state():
    return pd.DataFrame(), pd.DataFrame(), []


def reset_review_vars():
    return pd.DataFrame(), pd.DataFrame()


def reset_data_vars():
    return 0, [], 0


def reset_aws_call_vars():
    return 0, 0, 0, 0, 0, 0, "", ""


### functions related to summarisation ###


def clean_column_name(
    column_name: str, max_length: int = 20, front_characters: bool = True
):
    # Convert to string
    column_name = str(column_name)
    # Replace non-alphanumeric characters (except underscores) with underscores
    column_name = re.sub(r"\W+", "_", column_name)
    # Remove leading/trailing underscores
    column_name = column_name.strip("_")
    # Ensure the result is not empty; fall back to "column" if necessary
    column_name = column_name if column_name else "column"
    # Truncate to max_length
    if front_characters is True:
        output_text = column_name[:max_length]
    else:
        output_text = column_name[-max_length:]
    return output_text


def create_batch_file_path_details(
    reference_data_file_name: str,
    latest_batch_completed: int = None,
    batch_size_number: int = None,
    in_column: str = None,
) -> str:
    """
    Creates a standardised batch file path detail string from a reference data filename.

    Args:
        reference_data_file_name (str): Name of the reference data file
        latest_batch_completed (int, optional): Latest batch completed. Defaults to None.
        batch_size_number (int, optional): Batch size number. Defaults to None.
        in_column (str, optional): In column. Defaults to None.
    Returns:
        str: Formatted batch file path detail string
    """

    # Extract components from filename using regex
    file_name = (
        re.search(
            r"(.*?)(?:_all_|_final_|_batch_|_col_)", reference_data_file_name
        ).group(1)
        if re.search(r"(.*?)(?:_all_|_final_|_batch_|_col_)", reference_data_file_name)
        else reference_data_file_name
    )

    # Clean the extracted names
    file_name_cleaned = clean_column_name(file_name, max_length=20)

    return f"{file_name_cleaned}_"


def ensure_model_in_map(model_choice: str, model_name_map_dict: dict = None) -> dict:
    """
    Ensures that a model_choice is registered in model_name_map.
    If the model_choice is not found, it assumes it's an inference-server model
    and adds it to the map with source "inference-server".

    Args:
        model_choice (str): The model name to check/register
        model_name_map_dict (dict, optional): The model_name_map dictionary to update.
            If None, uses the global model_name_map from config.

    Returns:
        dict: The model_name_map dictionary (updated if needed)
    """
    # Use provided dict or global one
    if model_name_map_dict is None:
        from tools.config import model_name_map

        model_name_map_dict = model_name_map

    # If model_choice is not in the map, assume it's an inference-server model
    if model_choice not in model_name_map_dict:
        model_name_map_dict[model_choice] = {
            "short_name": model_choice,
            "source": "inference-server",
        }
        print(f"Registered custom model '{model_choice}' as inference-server model")

    return model_name_map_dict


def get_file_name_no_ext(file_path: str):
    # First, get the basename of the file (e.g., "example.txt" from "/path/to/example.txt")
    basename = os.path.basename(file_path)

    # Then, split the basename and its extension and return only the basename without the extension
    filename_without_extension, _ = os.path.splitext(basename)

    # print(filename_without_extension)

    return filename_without_extension


def _file_name_from_pdf_path(full_file_name):
    """Derive a safe file_name prefix from a PDF path (for summary output naming)."""
    if not full_file_name or not str(full_file_name).strip():
        return "document"
    basename = os.path.basename(full_file_name)
    name_without_ext, _ = os.path.splitext(basename)
    filename_prefix = (name_without_ext or "document")[:20]
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename_prefix = filename_prefix.replace(char, "_")
    return filename_prefix if filename_prefix else "document"


###


def load_in_default_allow_list(allow_list_file_path):
    if isinstance(allow_list_file_path, str):
        allow_list_file_path = [allow_list_file_path]
    return allow_list_file_path


def load_in_default_cost_codes(cost_codes_path: str, default_cost_code: str = ""):
    """
    Load in the cost codes list from file.
    """
    cost_codes_df = pd.read_csv(cost_codes_path)
    dropdown_choices = cost_codes_df.iloc[:, 0].astype(str).tolist()

    # Avoid inserting duplicate or empty cost code values
    if default_cost_code and default_cost_code not in dropdown_choices:
        dropdown_choices.insert(0, default_cost_code)

    # Always have a blank option at the top
    if "" not in dropdown_choices:
        dropdown_choices.insert(0, "")

    # Use passed default if in choices, else fall back to DEFAULT_COST_CODE so dropdown shows app default on load
    if default_cost_code and default_cost_code in dropdown_choices:
        value = default_cost_code
    elif DEFAULT_COST_CODE and DEFAULT_COST_CODE in dropdown_choices:
        value = DEFAULT_COST_CODE
    else:
        value = ""

    out_dropdown = gr.Dropdown(
        value=value,
        label="Choose cost code for analysis",
        choices=dropdown_choices,
        allow_custom_value=False,
    )

    return cost_codes_df, cost_codes_df, out_dropdown


def enforce_cost_codes(
    enforce_cost_code_bool: bool,
    cost_code_choice: str,
    cost_code_df: pd.DataFrame,
    verify_cost_codes: bool = True,
):
    """
    Check if the enforce cost codes variable is set to True, and then check that a cost cost has been chosen. If not, raise an error. Then, check against the values in the cost code dataframe to ensure that the cost code exists.
    """

    if enforce_cost_code_bool:
        if not cost_code_choice:
            raise Exception("Please choose a cost code before continuing")

        if verify_cost_codes:
            if cost_code_df.empty:
                raise Exception("No cost codes present in dataframe for verification")
            else:
                valid_cost_codes_list = list(cost_code_df.iloc[:, 0].unique())

                if cost_code_choice not in valid_cost_codes_list:
                    raise Exception(
                        "Selected cost code not found in list. Please contact your system administrator if you cannot find the correct cost code from the given list of suggestions."
                    )
    return


def update_cost_code_dataframe_from_dropdown_select(
    cost_dropdown_selection: str, cost_code_df: pd.DataFrame
):
    cost_code_df = cost_code_df.loc[
        cost_code_df.iloc[:, 0] == cost_dropdown_selection, :
    ]
    return cost_code_df


SESSION_DEFAULT_COST_CODES_FILENAME = "session_default_cost_codes.csv"

# Reasonable email pattern for validating session_hash (saves/upload require email format)
_EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def _is_valid_session_hash_email(session_hash: str) -> bool:
    """Return True if session_hash is in valid email format (required for saving defaults)."""
    if not session_hash or not isinstance(session_hash, str):
        return False
    return bool(_EMAIL_PATTERN.match(session_hash.strip()))


def _dedupe_session_default_cost_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate session_hashes, keeping only the latest row by saved_at.
    If saved_at is missing or empty, those rows are treated as oldest.
    """
    if df.empty or "session_hash" not in df.columns:
        return df
    if "saved_at" not in df.columns:
        return df.drop_duplicates(subset=["session_hash"], keep="last")
    # Sort so latest saved_at is last; empty string sorts last in ascending order
    df = df.sort_values("saved_at", ascending=True, na_position="last")
    return df.drop_duplicates(subset=["session_hash"], keep="last")


def _get_session_default_cost_codes_s3_key_prefix():
    """
    Return the S3 key prefix (folder) for the session default cost codes file,
    same bucket and folder as S3_COST_CODES_PATH. Empty string if no S3 path.
    """
    if not S3_COST_CODES_PATH or not str(S3_COST_CODES_PATH).strip():
        return ""
    # Use forward slash for S3; path can be "config/COST_CENTRES.csv" or "file.csv"
    parts = S3_COST_CODES_PATH.replace("\\", "/").rstrip("/").split("/")
    if len(parts) <= 1:
        return ""
    return "/".join(parts[:-1]) + "/"


def get_session_default_cost_codes_csv_path(folder: str | None = None):
    """
    Return the path to the CSV file that stores session_hash -> default_cost_code.
    If folder is provided (e.g. session-specific input folder), derive path segments
    only from the portion of that folder under INPUT_FOLDER (each segment
    sanitized, then secure_path_join). If the folder is not under INPUT_FOLDER,
    try a single sanitized basename under INPUT_FOLDER. Otherwise fall back to
    the cost codes directory (COST_CODES_PATH or OUTPUT_COST_CODES_PATH).
    """
    if folder is not None and str(folder).strip():
        raw_folder = str(folder).strip()
        try:
            input_root = os.path.normpath(
                os.path.abspath(str(Path(INPUT_FOLDER).resolve()))
            )
        except OSError:
            input_root = os.path.normpath(os.path.abspath(str(INPUT_FOLDER)))

        candidate = os.path.normpath(os.path.abspath(raw_folder.rstrip("/\\")))
        try:
            rel = os.path.relpath(candidate, input_root)
        except ValueError:
            rel = None

        def _join_sanitized_under_input(parts: List[str]) -> str:
            safe_base: Path | str = INPUT_FOLDER
            for p in parts:
                safe_folder_name = sanitize_filename(p)
                if safe_folder_name in {"", ".", ".."}:
                    raise ValueError(
                        "Invalid folder name for session default cost codes path"
                    )
                safe_base = secure_path_join(safe_base, safe_folder_name)
            return os.path.join(str(safe_base), SESSION_DEFAULT_COST_CODES_FILENAME)

        if rel is not None and rel != ".." and not rel.startswith(".." + os.sep):
            part_tokens = [
                p
                for p in rel.replace("\\", "/").split("/")
                if p and p not in (".", "..")
            ]
            if part_tokens:
                try:
                    return _join_sanitized_under_input(part_tokens)
                except (ValueError, PermissionError, OSError):
                    pass
            else:
                return os.path.join(input_root, SESSION_DEFAULT_COST_CODES_FILENAME)

        try:
            tail = raw_folder.rstrip("/\\")
            safe_folder_name = sanitize_filename(os.path.basename(tail))
            if safe_folder_name in {"", ".", ".."}:
                raise ValueError(
                    "Invalid folder name for session default cost codes path"
                )
            safe_base = secure_path_join(INPUT_FOLDER, safe_folder_name)
            return os.path.join(str(safe_base), SESSION_DEFAULT_COST_CODES_FILENAME)
        except (ValueError, PermissionError, OSError):
            pass
    if COST_CODES_PATH:
        folder = os.path.dirname(COST_CODES_PATH)
    else:
        folder = os.path.dirname(OUTPUT_COST_CODES_PATH)
    if not folder:
        folder = "."
    return os.path.join(folder, SESSION_DEFAULT_COST_CODES_FILENAME)


def _session_default_cost_codes_parent_dirs() -> List[Path]:
    """Resolved directories where session_default_cost_codes.csv may be stored locally."""
    dirs: List[Path] = []
    for folder in (
        INPUT_FOLDER,
        (os.path.dirname(COST_CODES_PATH) if COST_CODES_PATH else ""),
        (os.path.dirname(OUTPUT_COST_CODES_PATH) if OUTPUT_COST_CODES_PATH else ""),
        os.getcwd(),
    ):
        s = str(folder).strip() if folder is not None else ""
        if not s:
            continue
        try:
            dirs.append(Path(folder).resolve())
        except OSError:
            continue
    seen: Set[str] = set()
    out: List[Path] = []
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _validate_session_default_cost_codes_csv_path(
    csv_path: str, *, allow_download_temp: bool = False
) -> Path:
    """
    Ensure csv_path is safe for local read/write (CodeQL py/path-injection).
    When allow_download_temp is True, only paths under the system temp directory
    are accepted (S3 download scratch file).
    """
    if not csv_path or not str(csv_path).strip():
        raise ValueError("Missing session default cost codes CSV path")
    path_str = str(csv_path).strip()
    # Do not call Path.resolve() on untrusted paths (CodeQL py/path-injection);
    # use normpath + abspath + commonpath containment like validate_path_safety.
    candidate = os.path.normpath(os.path.abspath(path_str))
    if allow_download_temp:
        temp_root = os.path.normpath(os.path.abspath(tempfile.gettempdir()))
        try:
            common = os.path.commonpath([candidate, temp_root])
        except ValueError:
            common = None
        if common != temp_root:
            raise PermissionError(
                f"Session cost codes download path outside system temp: {candidate}"
            )
        return Path(candidate)
    if os.path.basename(candidate) != SESSION_DEFAULT_COST_CODES_FILENAME:
        raise PermissionError(
            f"Unexpected session default cost codes filename: {os.path.basename(candidate)!r}"
        )
    roots = _session_default_cost_codes_parent_dirs()
    for root in roots:
        root_norm = os.path.normpath(os.path.abspath(str(root)))
        try:
            common = os.path.commonpath([candidate, root_norm])
        except ValueError:
            continue
        if common == root_norm:
            return Path(candidate)
    raise PermissionError(
        f"Session default cost codes CSV outside allowed directories: {candidate}"
    )


def save_default_cost_code_for_session(
    session_hash: str,
    cost_code_choice: str,
    cost_code_df: pd.DataFrame,
    output_folder: str | None = None,
) -> str:
    """
    Validate cost_code_choice against cost_code_df 'Cost code' column; if valid,
    save or update session_hash -> default_cost_code in the session defaults CSV.
    If output_folder is provided (e.g. input folder path), save there; else use
    cost codes folder. Returns a message string for the user.
    """
    if not session_hash or not str(session_hash).strip():
        return "No session identifier available."
    if not _is_valid_session_hash_email(session_hash):
        return (
            "Default cost code can only be saved when the session identifier is an "
            "email address (e.g. when signed in with Cognito)."
        )
    if not cost_code_choice or not str(cost_code_choice).strip():
        return "Please choose a cost code first."
    if cost_code_df is None or cost_code_df.empty:
        return "No cost codes loaded for validation."
    valid_codes = list(cost_code_df.iloc[:, 0].astype(str).unique())
    if cost_code_choice not in valid_codes:
        return "Selected cost code not in the list. Choose a cost code from the table."
    try:
        csv_path = str(
            _validate_session_default_cost_codes_csv_path(
                get_session_default_cost_codes_csv_path(output_folder)
            )
        )
    except (ValueError, PermissionError, OSError):
        return "Cannot save default cost code: invalid or unsafe storage path."
    ensure_folder_exists(os.path.dirname(csv_path))
    saved_at = datetime.now().isoformat()
    row = {
        "session_hash": session_hash,
        "default_cost_code": cost_code_choice,
        "saved_at": saved_at,
    }
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        if (
            "session_hash" in existing.columns
            and "default_cost_code" in existing.columns
        ):
            if "saved_at" not in existing.columns:
                existing["saved_at"] = ""
            # Replace any existing row for this session (one row per session_hash)
            existing = existing[
                existing["session_hash"].astype(str) != str(session_hash)
            ]
            updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        else:
            updated = pd.DataFrame([row])
    else:
        updated = pd.DataFrame([row])
    # Remove duplicate session_hashes, keeping only the latest by saved_at
    updated = _dedupe_session_default_cost_codes(updated)
    updated.to_csv(csv_path, index=False)

    # If S3 cost codes path is set, save to the same bucket and folder in S3
    if (
        S3_COST_CODES_PATH
        and str(S3_COST_CODES_PATH).strip()
        and RUN_AWS_FUNCTIONS
        and DOCUMENT_REDACTION_BUCKET
    ):
        # Ensure the file exists before upload (e.g. in case of path or write edge cases)
        if not os.path.exists(csv_path):
            ensure_folder_exists(os.path.dirname(csv_path))
            updated.to_csv(csv_path, index=False)
        s3_prefix = _get_session_default_cost_codes_s3_key_prefix()
        s3_full_key = s3_prefix + SESSION_DEFAULT_COST_CODES_FILENAME
        upload_result = upload_file_to_s3(
            local_file_paths=[csv_path],
            s3_key=s3_prefix,
            s3_bucket=DOCUMENT_REDACTION_BUCKET,
            RUN_AWS_FUNCTIONS=RUN_AWS_FUNCTIONS,
        )
        if upload_result and "successfully" in upload_result.lower():
            print(
                f"Session default cost codes CSV uploaded to "
                f"s3://{DOCUMENT_REDACTION_BUCKET}/{s3_full_key}"
            )
        else:
            print(
                f"Session default cost codes S3 upload issue "
                f"(target s3://{DOCUMENT_REDACTION_BUCKET}/{s3_full_key}): {upload_result!r}"
            )

    return "Default cost code saved"


def _read_session_default_from_csv_path(
    csv_path: str,
    session_hash: str,
    *,
    allow_download_temp: bool = False,
) -> str:
    """Read CSV at csv_path and return default_cost_code for session_hash, or "".
    If saved_at exists, uses the latest row by saved_at for that session_hash.
    """
    try:
        csv_path = str(
            _validate_session_default_cost_codes_csv_path(
                csv_path, allow_download_temp=allow_download_temp
            )
        )
    except (ValueError, PermissionError, OSError):
        return ""
    if not os.path.exists(csv_path):
        return ""
    try:
        df = pd.read_csv(csv_path)
        if "session_hash" not in df.columns or "default_cost_code" not in df.columns:
            return ""
        match = df[df["session_hash"].astype(str) == str(session_hash)]
        if match.empty:
            return ""
        if "saved_at" in match.columns:
            match = match.sort_values("saved_at", ascending=True, na_position="last")
        return str(match.iloc[-1]["default_cost_code"]).strip()
    except Exception:
        return ""


def load_session_default_cost_code(
    session_hash: str, input_folder: str | None = None
) -> str:
    """
    Load the default cost code for the given session_hash from the session defaults CSV.
    If S3_COST_CODES_PATH is set, tries the same bucket and folder in S3 first, then local.
    If input_folder is provided, tries that path first for local file, then fallback.
    Returns the default cost code string if found, else "".
    """
    if not session_hash or not str(session_hash).strip():
        return ""

    # Only download from S3 when session_hash is an email (same rule as for saves)
    if _is_valid_session_hash_email(session_hash) and (
        S3_COST_CODES_PATH
        and str(S3_COST_CODES_PATH).strip()
        and RUN_AWS_FUNCTIONS
        and DOCUMENT_REDACTION_BUCKET
    ):
        s3_prefix = _get_session_default_cost_codes_s3_key_prefix()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as tmp:
                tmp_path = tmp.name
            download_file_from_s3(
                bucket_name=DOCUMENT_REDACTION_BUCKET,
                key=s3_prefix + SESSION_DEFAULT_COST_CODES_FILENAME,
                local_file_path_and_name=tmp_path,
                RUN_AWS_FUNCTIONS=RUN_AWS_FUNCTIONS,
            )
            if tmp_path and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                result = _read_session_default_from_csv_path(
                    tmp_path, session_hash, allow_download_temp=True
                )
                if result:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    return result
        except Exception:
            pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # Local: try input_folder first if provided, then default cost codes folder
    if input_folder is not None and str(input_folder).strip():
        csv_path = get_session_default_cost_codes_csv_path(input_folder)
        result = _read_session_default_from_csv_path(csv_path, session_hash)
        if result:
            return result
    csv_path = get_session_default_cost_codes_csv_path()
    return _read_session_default_from_csv_path(csv_path, session_hash)


def apply_session_default_cost_code(
    session_hash: str,
    cost_code_dataframe: pd.DataFrame,
    input_folder: str | None = None,
    current_default_cost_code: str | None = None,
    current_dropdown_value: str | None = None,
) -> tuple[str, str]:
    """
    Look up the saved default cost code for session_hash. If found and valid in
    cost_code_dataframe, return (default_cost_code, default_cost_code) for
    default_cost_code_textbox and cost_code_choice_drop. When no saved default
    applies, return (current_default_cost_code, current_dropdown_value) so the
    dropdown keeps showing e.g. DEFAULT_COST_CODE instead of being cleared.
    """
    default_code = load_session_default_cost_code(session_hash, input_folder)
    if not default_code:
        # Preserve current values so we don't overwrite with "" on app load
        cur = current_default_cost_code if current_default_cost_code is not None else ""
        drop = current_dropdown_value if current_dropdown_value is not None else ""
        return cur, drop
    if cost_code_dataframe is None or cost_code_dataframe.empty:
        return default_code, default_code
    choices = list(cost_code_dataframe.iloc[:, 0].astype(str).unique())
    if default_code not in choices:
        cur = current_default_cost_code if current_default_cost_code is not None else ""
        drop = current_dropdown_value if current_dropdown_value is not None else ""
        return cur, drop
    return default_code, default_code


def ensure_folder_exists(output_folder: str):
    """Checks if the specified folder exists, creates it if not.

    Resolves ``output_folder`` under the app directory via
    ``ensure_folder_within_app_directory`` so user-influenced paths cannot escape
    the workspace (CodeQL py/path-injection).
    """
    if output_folder is None or not str(output_folder).strip():
        return
    try:
        safe_folder = ensure_folder_within_app_directory(str(output_folder).strip())
    except (ValueError, PermissionError, OSError) as e:
        logging.getLogger(__name__).warning(
            "ensure_folder_exists: refused or could not normalize path %r: %s",
            output_folder,
            e,
        )
        return
    if not safe_folder or not str(safe_folder).strip():
        return

    if not os.path.exists(safe_folder):
        os.makedirs(safe_folder, exist_ok=True)


def update_dataframe(df_or_list):
    """
    Update function for both DataFrame and list inputs.
    For Dropdown components (list), return the list as-is.
    For DataFrame components, return a copy.
    """
    if isinstance(df_or_list, list):
        return df_or_list
    elif isinstance(df_or_list, pd.DataFrame):
        return df_or_list.copy()
    else:
        return df_or_list


def get_file_name_without_type(file_path):
    # First, get the basename of the file (e.g., "example.txt" from "/path/to/example.txt")
    basename = os.path.basename(file_path)

    # Then, split the basename and its extension and return only the basename without the extension
    filename_without_extension, _ = os.path.splitext(basename)

    # print(filename_without_extension)

    return filename_without_extension


def detect_file_type(filename: str):
    """Detect the file type based on its extension."""
    if not isinstance(filename, str):
        filename = str(filename)

    if (
        (filename.endswith(".csv"))
        | (filename.endswith(".csv.gz"))
        | (filename.endswith(".zip"))
    ):
        return "csv"
    elif filename.endswith(".xlsx"):
        return "xlsx"
    elif filename.endswith(".xls"):
        return "xls"
    elif filename.endswith(".parquet"):
        return "parquet"
    elif filename.endswith(".pdf"):
        return "pdf"
    elif filename.endswith(".jpg"):
        return "jpg"
    elif filename.endswith(".jpeg"):
        return "jpeg"
    elif filename.endswith(".png"):
        return "png"
    elif filename.endswith(".xfdf"):
        return "xfdf"
    elif filename.endswith(".docx"):
        return "docx"
    else:
        raise ValueError("Unsupported file type.")


def read_file(filename: str, excel_sheet_name: str = ""):
    """Read the file based on its detected type."""
    file_type = detect_file_type(filename)

    if file_type == "csv":
        return pd.read_csv(filename, low_memory=False)
    elif file_type == "xlsx":
        if excel_sheet_name:
            try:
                return pd.read_excel(filename, sheet_name=excel_sheet_name)
            except Exception as e:
                print(
                    f"Error reading {filename} with sheet name {excel_sheet_name}: {e}"
                )
                return pd.DataFrame()
        else:
            return pd.read_excel(filename)
    elif file_type == "parquet":
        return pd.read_parquet(filename)


def ensure_output_folder_exists(output_folder: str):
    """Checks if the specified folder exists, creates it if not."""

    if not os.path.exists(output_folder):
        # Create the folder if it doesn't exist
        os.makedirs(output_folder)
        print(f"Created the {output_folder} folder.")
    else:
        print(f"The {output_folder} folder already exists.")


def custom_regex_load(in_file: List[str], file_type: str = "allow_list"):
    """
    When file is loaded, update the column dropdown choices and write to relevant data states.
    Returns a list for Dropdown components (instead of DataFrame).
    """
    custom_regex_list = list()

    if in_file:
        file_list = [string.name for string in in_file]

        regex_file_names = [string for string in file_list if "csv" in string.lower()]
        if regex_file_names:
            regex_file_name = regex_file_names[0]
            custom_regex_df = pd.read_csv(
                regex_file_name, low_memory=False, header=None
            )

            # Select just first column and convert to list for Dropdown component
            if not custom_regex_df.empty:
                custom_regex_list = (
                    custom_regex_df.iloc[:, 0].dropna().astype(str).tolist()
                )

            # substitute underscores in file type
            file_type_output = file_type.replace("_", " ")

            output_text = file_type_output + " file loaded."
            print(output_text)
    else:
        output_text = "No file provided."
        # print(output_text)
        return output_text, custom_regex_list

    return output_text, custom_regex_list


def put_columns_in_df(in_file: List[str]):
    new_choices = []
    concat_choices = []
    all_sheet_names = []
    number_of_excel_files = 0

    for file in in_file:
        file_name = file.name
        file_type = detect_file_type(file_name)
        # print("File type is:", file_type)

        if (file_type == "xlsx") | (file_type == "xls"):
            number_of_excel_files += 1
            new_choices = []
            anon_xlsx = pd.ExcelFile(file_name)
            new_sheet_names = anon_xlsx.sheet_names
            # Iterate through the sheet names
            for sheet_name in new_sheet_names:
                # Read each sheet into a DataFrame
                df = pd.read_excel(file_name, sheet_name=sheet_name)

                # Process the DataFrame (e.g., print its contents)
                new_choices.extend(list(df.columns))

            all_sheet_names.extend(new_sheet_names)

        elif (file_type == "csv") | (file_type == "parquet"):
            df = read_file(file_name)
            new_choices = list(df.columns)

        else:
            new_choices = []

        concat_choices.extend(new_choices)

    # Drop duplicate columns
    concat_choices = list(set(concat_choices))

    if number_of_excel_files > 0:
        return gr.Dropdown(
            choices=concat_choices, value=concat_choices, visible=True
        ), gr.Dropdown(choices=all_sheet_names, value=all_sheet_names, visible=True)
    else:
        return gr.Dropdown(
            choices=concat_choices, value=concat_choices, visible=True
        ), gr.Dropdown(visible=False)


def get_textract_file_suffix(handwrite_signature_checkbox: List[str] = list()) -> str:
    """
    Generate a suffix for textract JSON files based on the selected feature types.

    Args:
        handwrite_signature_checkbox: List of selected Textract feature types.
            Options: "Extract signatures", "Extract forms", "Extract layout", "Extract tables"
            "Extract handwriting" is the default and doesn't add a suffix.

    Returns:
        A suffix string like "_sig", "_form", "_sig_form", etc., or empty string if only handwriting is selected.
    """
    if not handwrite_signature_checkbox:
        return ""

    # Map feature types to short suffixes
    feature_map = {
        "Extract signatures": "sig",
        "Extract forms": "form",
        "Extract layout": "layout",
        "Extract tables": "table",
    }

    # Collect suffixes for selected features (excluding handwriting which is default)
    suffixes = []
    for feature in handwrite_signature_checkbox:
        if feature in feature_map:
            suffixes.append(feature_map[feature])

    # Sort alphabetically for consistent naming
    suffixes.sort()

    # Return suffix with underscore prefix if any features selected
    if suffixes:
        return "_" + "_".join(suffixes)
    return ""


def check_for_existing_textract_file(
    doc_file_name_no_extension_textbox: str,
    output_folder: str = OUTPUT_FOLDER,
    handwrite_signature_checkbox: List[str] = list(),
):
    # Generate suffix based on checkbox options
    suffix = get_textract_file_suffix(handwrite_signature_checkbox)
    try:
        safe_stem = sanitize_filename(doc_file_name_no_extension_textbox)
        filename = safe_stem + suffix + "_textract.json"
        textract_output_path = secure_path_join(output_folder, filename)
    except (ValueError, PermissionError, OSError):
        return False

    if textract_output_path.exists():
        # print("Existing Textract analysis output file found.")
        return True

    else:
        return False


def check_for_relevant_ocr_output_with_words(
    doc_file_name_no_extension_textbox: str,
    text_extraction_method: str,
    output_folder: str = OUTPUT_FOLDER,
):
    if text_extraction_method == SELECTABLE_TEXT_EXTRACT_OPTION:
        file_ending = "_ocr_results_with_words_local_text.json"
    elif text_extraction_method == LOCAL_OCR_MODEL_TEXT_EXTRACT_OPTION:
        file_ending = "_ocr_results_with_words_local_ocr.json"
    elif text_extraction_method == TEXTRACT_TEXT_EXTRACT_OPTION:
        file_ending = "_ocr_results_with_words_textract.json"
    else:
        # print("No valid text extraction method found. Returning False")
        return False

    try:
        safe_stem = sanitize_filename(doc_file_name_no_extension_textbox)
        doc_file_with_ending = safe_stem + file_ending
        local_ocr_output_path = secure_path_join(output_folder, doc_file_with_ending)
    except (ValueError, PermissionError, OSError):
        return False

    if local_ocr_output_path.exists():
        print("Existing OCR with words analysis output file found.")
        return True
    else:
        return False


def add_folder_to_path(folder_path: str):
    """
    Check if a folder exists on your system. If so, get the absolute path and then add it to the system Path variable if it doesn't already exist. Function is only relevant for locally-created executable files based on this app (when using pyinstaller it creates a _internal folder that contains tesseract and poppler. These need to be added to the system path to enable the app to run)
    """

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(folder_path, "folder exists.")

        # Resolve relative path to absolute path
        absolute_path = os.path.abspath(folder_path)

        current_path = os.environ["PATH"]
        if absolute_path not in current_path.split(os.pathsep):
            full_path_extension = absolute_path + os.pathsep + current_path
            os.environ["PATH"] = full_path_extension
            # print(f"Updated PATH with: ", full_path_extension)
        else:
            print(f"Directory {folder_path} already exists in PATH.")
    else:
        print(f"Folder not found at {folder_path} - not added to PATH")


# Upon running a process, the feedback buttons are revealed
def reveal_feedback_buttons():
    if SHOW_FEEDBACK_BUTTONS:
        is_visible = True
    else:
        is_visible = False
    return (
        gr.Radio(
            visible=is_visible,
            label="Please give some feedback about the results of the redaction. A reminder that the app is only expected to identify about 80% of personally identifiable information in a given (typed) document.",
        ),
        gr.Textbox(visible=is_visible),
        gr.Button(visible=is_visible),
        gr.Markdown(visible=is_visible),
    )


def wipe_logs(feedback_logs_loc: str, usage_logs_loc: str):
    try:
        os.remove(feedback_logs_loc)
    except Exception as e:
        print("Could not remove feedback logs file", e)
    try:
        os.remove(usage_logs_loc)
    except Exception as e:
        print("Could not remove usage logs file", e)


def merge_csv_files(file_list: List[str], output_folder: str = OUTPUT_FOLDER):

    # Initialise an empty list to hold DataFrames
    dataframes = []
    output_files = []

    # Loop through each file in the file list
    for file in file_list:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file.name)
        dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)

    for col in ["xmin", "xmax", "ymin", "ymax"]:
        merged_df[col] = np.floor(merged_df[col])

    merged_df = merged_df.drop_duplicates(
        subset=["page", "label", "color", "xmin", "ymin", "xmax", "ymax"]
    )

    merged_df = merged_df.sort_values(["page", "ymin", "xmin", "label"])

    file_out_name = os.path.basename(file_list[0])

    merged_csv_path = output_folder + file_out_name + "_merged.csv"

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(merged_csv_path, index=False, encoding="utf-8-sig")
    output_files.append(merged_csv_path)

    return output_files


async def get_connection_params(
    request: gr.Request,
    output_folder_textbox: str = OUTPUT_FOLDER,
    input_folder_textbox: str = INPUT_FOLDER,
    session_output_folder: bool = SESSION_OUTPUT_FOLDER,
    s3_outputs_folder_textbox: str = S3_OUTPUTS_FOLDER,
    textract_document_upload_input_folder: str = TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_INPUT_SUBFOLDER,
    textract_document_upload_output_folder: str = TEXTRACT_WHOLE_DOCUMENT_ANALYSIS_OUTPUT_SUBFOLDER,
    s3_textract_document_logs_subfolder: str = TEXTRACT_JOBS_S3_LOC,
    local_textract_document_logs_subfolder: str = TEXTRACT_JOBS_LOCAL_LOC,
):
    # Convert session_output_folder to boolean if it's a string (from Gradio Textbox)
    if isinstance(session_output_folder, str):
        session_output_folder = convert_string_to_boolean(session_output_folder)

    if CUSTOM_HEADER and CUSTOM_HEADER_VALUE:
        if CUSTOM_HEADER in request.headers:
            supplied_custom_header_value = request.headers[CUSTOM_HEADER]
            if supplied_custom_header_value == CUSTOM_HEADER_VALUE:
                print("Custom header supplied and matches CUSTOM_HEADER_VALUE")
            else:
                print("Custom header value does not match expected value.")
                raise ValueError("Custom header value does not match expected value.")
        else:
            print("Custom header value not found.")
            raise ValueError("Custom header value not found.")

    # Get output save folder from 1 - username passed in from direct Cognito login, 2 - Cognito ID header passed through a Lambda authenticator, 3 - the session hash.

    if request.username:
        out_session_hash = request.username
        # print("Request username found:", out_session_hash)

    elif "x-cognito-id" in request.headers:
        out_session_hash = request.headers["x-cognito-id"]
        print("Cognito ID found:", out_session_hash)

    elif "x-amzn-oidc-identity" in request.headers:
        out_session_hash = request.headers["x-amzn-oidc-identity"]

        if AWS_USER_POOL_ID:
            try:
                # Fetch email address using Cognito client
                cognito_client = boto3.client("cognito-idp")

                response = cognito_client.admin_get_user(
                    UserPoolId=AWS_USER_POOL_ID,  # Replace with your User Pool ID
                    Username=out_session_hash,
                )
                email = next(
                    attr["Value"]
                    for attr in response["UserAttributes"]
                    if attr["Name"] == "email"
                )
                print("Cognito email address found, will be used as session hash")

                out_session_hash = email
            except (
                ClientError,
                NoCredentialsError,
                PartialCredentialsError,
                BotoCoreError,
            ) as e:
                print(f"Error fetching Cognito user details: {e}")
                print("Falling back to using AWS ID as session hash")
                # out_session_hash already set to the AWS ID from header, so no need to change it
            except Exception as e:
                print(f"Unexpected error when fetching Cognito user details: {e}")
                print("Falling back to using AWS ID as session hash")
                # out_session_hash already set to the AWS ID from header, so no need to change it

        print("AWS ID found, will be used as username for session:", out_session_hash)

    else:
        out_session_hash = request.session_hash

    if session_output_folder:
        output_folder = output_folder_textbox + out_session_hash + "/"
        input_folder = input_folder_textbox + out_session_hash + "/"

        # If configured, create a session-specific S3 outputs folder using the same pattern
        if SAVE_OUTPUTS_TO_S3 and s3_outputs_folder_textbox:
            s3_outputs_folder = (
                s3_outputs_folder_textbox.rstrip("/") + "/" + out_session_hash + "/"
            )
        else:
            s3_outputs_folder = s3_outputs_folder_textbox

        textract_document_upload_input_folder = (
            textract_document_upload_input_folder + "/" + out_session_hash
        )
        textract_document_upload_output_folder = (
            textract_document_upload_output_folder + "/" + out_session_hash
        )

        s3_textract_document_logs_subfolder = (
            s3_textract_document_logs_subfolder + "/" + out_session_hash
        )
        local_textract_document_logs_subfolder = (
            local_textract_document_logs_subfolder + "/" + out_session_hash + "/"
        )

    else:
        output_folder = output_folder_textbox
        input_folder = input_folder_textbox
        # Keep S3 outputs folder as configured (no per-session subfolder)
        s3_outputs_folder = s3_outputs_folder_textbox

    # Append today's date (YYYYMMDD/) to the final S3 outputs folder when enabled
    if SAVE_OUTPUTS_TO_S3 and s3_outputs_folder:
        today_suffix = datetime.now().strftime("%Y%m%d") + "/"
        s3_outputs_folder = s3_outputs_folder.rstrip("/") + "/" + today_suffix

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    if not os.path.exists(input_folder):
        os.makedirs(input_folder, exist_ok=True)

    return (
        out_session_hash,
        output_folder,
        out_session_hash,
        input_folder,
        textract_document_upload_input_folder,
        textract_document_upload_output_folder,
        s3_textract_document_logs_subfolder,
        local_textract_document_logs_subfolder,
        s3_outputs_folder,
    )


def clean_unicode_text(text: str, preserve_international_scripts: bool = False) -> str:
    """
    Normalise Unicode (NFKC), replace common smart punctuation with ASCII.

    By default, non-ASCII characters are stripped (legacy behaviour for some pipelines).
    Set ``preserve_international_scripts=True`` for OCR and extracted document text so
    Cyrillic, Arabic, CJK, etc. are retained in outputs (CSV, JSON, redaction targets).
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Step 1: Normalise unicode characters to decompose any special forms
    normalized_text = unicodedata.normalize("NFKC", text)

    # Step 2: Replace smart quotes and special punctuation with standard ASCII equivalents
    replacements = {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
        "•": "*",
    }

    for old_char, new_char in replacements.items():
        normalized_text = normalized_text.replace(old_char, new_char)

    if preserve_international_scripts:
        return normalized_text

    from tools.secure_regex_utils import safe_remove_non_ascii

    return safe_remove_non_ascii(normalized_text)


# --- Helper Function for ID Generation ---
# This function encapsulates your ID logic in a performant, batch-oriented way.
def _generate_unique_ids(
    num_ids_to_generate: int, existing_ids_set: Set[str]
) -> List[str]:
    """
    Generates a specified number of unique, 12-character alphanumeric IDs.

    This is a batch-oriented, performant version of the original
    `fill_missing_ids_in_list` logic, designed to work efficiently
    with DataFrames.

    Args:
        num_ids_to_generate (int): The number of unique IDs to create.
        existing_ids_set (Set[str]): A set of IDs that are already in use and
                                     should be avoided.

    Returns:
        List[str]: A list of newly generated unique IDs.
    """
    id_length = 12
    character_set = string.ascii_letters + string.digits

    newly_generated_ids = set()

    # The while loop ensures we generate exactly the number of IDs required,
    # automatically handling the astronomically rare case of a collision.
    while len(newly_generated_ids) < num_ids_to_generate:
        candidate_id = "".join(random.choices(character_set, k=id_length))

        # Check against both pre-existing IDs and IDs generated in this batch
        if (
            candidate_id not in existing_ids_set
            and candidate_id not in newly_generated_ids
        ):
            newly_generated_ids.add(candidate_id)

    return list(newly_generated_ids)


def load_all_output_files(folder_path: str = OUTPUT_FOLDER) -> List[str]:
    """Get the file paths of all files in the given folder and its subfolders."""

    # Gradio can sometimes pass None here during chained load/update events
    # (e.g. if a prior event errored and the textbox/state wasn't populated).
    folder_path = folder_path or OUTPUT_FOLDER

    safe_folder_path_resolved = Path(folder_path).resolve()

    return gr.FileExplorer(
        root_dir=safe_folder_path_resolved,
    )


def update_file_explorer_object():
    return gr.FileExplorer()


def _is_file_path(path: str) -> bool:
    """True if path looks like a file (has a file-type suffix), not a folder."""
    if not path or not path.strip():
        return False
    name = os.path.basename(path.rstrip("/\\"))
    if not name or "." not in name:
        return False
    ext = name.rsplit(".", 1)[-1]
    return bool(ext and len(ext) <= 10 and ext.isalnum())


def all_outputs_file_download_fn(file_explorer_object: list[str]):
    """Return only paths that are files (have a suffix like .csv, .txt), not folder paths."""
    if not file_explorer_object:
        return file_explorer_object
    return [p for p in file_explorer_object if _is_file_path(p)]


def calculate_aws_costs(
    number_of_pages: str,
    text_extract_method_radio: str,
    handwrite_signature_checkbox: List[str],
    pii_identification_method: str,
    textract_output_found_checkbox: bool,
    only_extract_text_radio: bool,
    convert_to_gbp: bool = True,
    usd_gbp_conversion_rate: float = 0.76,
    textract_page_cost: float = 1.5 / 1000,
    textract_signature_cost: float = 2.0 / 1000,
    textract_forms_cost: float = 50.0 / 1000,
    textract_layout_cost: float = 4.0 / 1000,
    textract_tables_cost: float = 15.0 / 1000,
    comprehend_unit_cost: float = 0.0001,
    comprehend_size_unit_average: float = 250,
    average_characters_per_page: float = 2000,
    bedrock_vlm_output_token_ratio: float = 0.08,
    bedrock_vlm_face_output_token_ratio: float = 0.03,
    TEXTRACT_TEXT_EXTRACT_OPTION: str = TEXTRACT_TEXT_EXTRACT_OPTION,
    BEDROCK_VLM_TEXT_EXTRACT_OPTION: str = BEDROCK_VLM_TEXT_EXTRACT_OPTION,
    NO_REDACTION_PII_OPTION: str = NO_REDACTION_PII_OPTION,
    AWS_PII_OPTION: str = AWS_PII_OPTION,
    AWS_LLM_PII_OPTION: str = AWS_LLM_PII_OPTION,
    VLM_MAX_IMAGE_SIZE: int = VLM_MAX_IMAGE_SIZE,
    BEDROCK_VLM_INPUT_COST: float = BEDROCK_VLM_INPUT_COST,
    BEDROCK_VLM_OUTPUT_COST: float = BEDROCK_VLM_OUTPUT_COST,
    BEDROCK_VLM_PIXELS_PER_INPUT_TOKEN: int = BEDROCK_VLM_PIXELS_PER_INPUT_TOKEN,
    BEDROCK_LLM_INPUT_COST: float = BEDROCK_LLM_INPUT_COST,
    BEDROCK_LLM_OUTPUT_COST: float = BEDROCK_LLM_OUTPUT_COST,
    BEDROCK_LLM_INPUT_TOKENS_PER_PAGE: int = BEDROCK_LLM_INPUT_TOKENS_PER_PAGE,
    BEDROCK_LLM_OUTPUT_TOKENS_PER_PAGE: int = BEDROCK_LLM_OUTPUT_TOKENS_PER_PAGE,
):
    """
    Calculate the approximate cost of submitting a document to AWS Textract and/or AWS Comprehend, assuming that Textract outputs do not already exist in the output folder.

    - number_of_pages: The number of pages in the uploaded document(s).
    - text_extract_method_radio: The method of text extraction.
    - handwrite_signature_checkbox: Whether signatures are being extracted or not.
    - pii_identification_method_drop: The method of personally-identifiable information removal.
    - textract_output_found_checkbox: Whether existing Textract results have been found in the output folder. Assumes that results exist for all pages and files in the output folder.
    - only_extract_text_radio (bool, optional): Option to only extract text from the document rather than redact.
    - convert_to_gbp (bool, optional): Should suggested costs be converted from USD to GBP.
    - usd_gbp_conversion_rate (float, optional): Conversion rate used for USD to GBP. Last changed 14th April 2025.
    - textract_page_cost (float, optional): AWS pricing for Textract text extraction per page ($).
    - textract_signature_cost (float, optional): Additional AWS cost above standard AWS Textract extraction for extracting signatures.
    - textract_forms_cost (float, optional): AWS Textract cost per page for "Extract forms" ($50/1000 pages).
    - textract_layout_cost (float, optional): AWS Textract cost per page for "Extract layout" ($4/1000 pages).
    - textract_tables_cost (float, optional): AWS Textract cost per page for "Extract tables" ($15/1000 pages).
    - comprehend_unit_cost (float, optional): Cost per 'unit' (300 character minimum) for identifying PII in text with AWS Comprehend.
    - comprehend_size_unit_average (float, optional): Average size of a 'unit' of text passed to AWS Comprehend by the app through the batching process
    - average_characters_per_page (float, optional): Average number of characters on an A4 page.
    - bedrock_vlm_output_token_ratio (float, optional): Ratio of output to input tokens for Bedrock VLM OCR (~0.08 in practice).
    - bedrock_vlm_face_output_token_ratio (float, optional): Ratio of output to input tokens for the face-identification second run (~0.03 in practice).
    - TEXTRACT_TEXT_EXTRACT_OPTION (str, optional): String label for the text_extract_method_radio button for AWS Textract.
    - BEDROCK_VLM_TEXT_EXTRACT_OPTION (str, optional): String label for AWS Bedrock VLM OCR text extraction.
    - NO_REDACTION_PII_OPTION (str, optional): String label for pii_identification_method_drop for no redaction.
    - AWS_PII_OPTION (str, optional): String label for pii_identification_method_drop for AWS Comprehend.
    - AWS_LLM_PII_OPTION (str, optional): String label for PII identification via LLM (AWS Bedrock).
    - VLM_MAX_IMAGE_SIZE, BEDROCK_VLM_*: used for Bedrock VLM OCR cost estimate.
    - BEDROCK_LLM_*: used for Bedrock LLM (e.g. PII detection) cost estimate when that method is selected (2000 input / 250 output tokens per page).
    """
    text_extraction_cost = 0
    pii_identification_cost = 0
    calculated_aws_cost = 0
    number_of_pages = int(number_of_pages)

    if textract_output_found_checkbox is not True:
        if text_extract_method_radio == TEXTRACT_TEXT_EXTRACT_OPTION:
            text_extraction_cost = number_of_pages * textract_page_cost

            if "Extract signatures" in handwrite_signature_checkbox:
                text_extraction_cost += textract_signature_cost * number_of_pages
            if "Extract forms" in handwrite_signature_checkbox:
                text_extraction_cost += textract_forms_cost * number_of_pages
            if "Extract layout" in handwrite_signature_checkbox:
                text_extraction_cost += textract_layout_cost * number_of_pages
            if "Extract tables" in handwrite_signature_checkbox:
                text_extraction_cost += textract_tables_cost * number_of_pages

        elif text_extract_method_radio == BEDROCK_VLM_TEXT_EXTRACT_OPTION:
            # Estimate input tokens per page from max image size; output tokens ~8% of input
            input_tokens_per_page = ceil(
                VLM_MAX_IMAGE_SIZE / BEDROCK_VLM_PIXELS_PER_INPUT_TOKEN
            )
            output_tokens_per_page = (
                input_tokens_per_page * bedrock_vlm_output_token_ratio
            )
            total_input_tokens = number_of_pages * input_tokens_per_page
            total_output_tokens = number_of_pages * output_tokens_per_page
            text_extraction_cost = total_input_tokens * (
                BEDROCK_VLM_INPUT_COST / 1_000_000
            ) + total_output_tokens * (BEDROCK_VLM_OUTPUT_COST / 1_000_000)
            # Face detection does a second run per page: same input tokens, output ~3% of input
            if "Face detection" in handwrite_signature_checkbox:
                face_input_tokens = total_input_tokens
                face_output_tokens = int(
                    total_input_tokens * bedrock_vlm_face_output_token_ratio
                )
                text_extraction_cost += face_input_tokens * (
                    BEDROCK_VLM_INPUT_COST / 1_000_000
                ) + face_output_tokens * (BEDROCK_VLM_OUTPUT_COST / 1_000_000)

    if pii_identification_method != NO_REDACTION_PII_OPTION:
        if pii_identification_method == AWS_PII_OPTION:
            comprehend_page_cost = (
                ceil(average_characters_per_page / comprehend_size_unit_average)
                * comprehend_unit_cost
            )
            pii_identification_cost = comprehend_page_cost * number_of_pages

        elif pii_identification_method == AWS_LLM_PII_OPTION:
            # Bedrock LLM (e.g. PII detection): 2000 input tokens, 250 output tokens per page
            llm_input_tokens = number_of_pages * BEDROCK_LLM_INPUT_TOKENS_PER_PAGE
            llm_output_tokens = number_of_pages * BEDROCK_LLM_OUTPUT_TOKENS_PER_PAGE
            pii_identification_cost = llm_input_tokens * (
                BEDROCK_LLM_INPUT_COST / 1_000_000
            ) + llm_output_tokens * (BEDROCK_LLM_OUTPUT_COST / 1_000_000)

    calculated_aws_cost = (
        calculated_aws_cost + text_extraction_cost + pii_identification_cost
    )

    if convert_to_gbp is True:
        calculated_aws_cost *= usd_gbp_conversion_rate

    return calculated_aws_cost


def calculate_time_taken(
    number_of_pages: str,
    text_extract_method_radio: str,
    pii_identification_method: str,
    textract_output_found_checkbox: bool,
    only_extract_text_radio: bool,
    local_ocr_output_found_checkbox: bool,
    handwrite_signature_checkbox: List[str],
    convert_page_time: float = 0.3,
    textract_page_time: float = 0.4,
    comprehend_page_time: float = 0.4,
    local_text_extraction_page_time: float = 0.2,
    local_pii_redaction_page_time: float = 0.4,
    local_ocr_extraction_page_time: float = 1.5,
    TEXTRACT_TEXT_EXTRACT_OPTION: str = TEXTRACT_TEXT_EXTRACT_OPTION,
    BEDROCK_VLM_TEXT_EXTRACT_OPTION: str = BEDROCK_VLM_TEXT_EXTRACT_OPTION,
    SELECTABLE_TEXT_EXTRACT_OPTION: str = SELECTABLE_TEXT_EXTRACT_OPTION,
    local_ocr_option: str = LOCAL_OCR_MODEL_TEXT_EXTRACT_OPTION,
    NO_REDACTION_PII_OPTION: str = NO_REDACTION_PII_OPTION,
    AWS_PII_OPTION: str = AWS_PII_OPTION,
    AWS_LLM_PII_OPTION: str = AWS_LLM_PII_OPTION,
):
    """
    Calculate the approximate time to redact a document.

    - number_of_pages: The number of pages in the uploaded document(s).
    - text_extract_method_radio: The method of text extraction.
    - pii_identification_method_drop: The method of personally-identifiable information removal.
    - textract_output_found_checkbox (bool, optional): Boolean indicating if AWS Textract text extraction outputs have been found.
    - only_extract_text_radio (bool, optional): Option to only extract text from the document rather than redact.
    - local_ocr_output_found_checkbox (bool, optional): Boolean indicating if local OCR text extraction outputs have been found.
    - handwrite_signature_checkbox: List of selected options (e.g. "Face detection"); when Face detection is selected with Bedrock VLM, extraction time is doubled.
    - textract_page_time (float, optional): Approximate time to query AWS Textract (also used for Bedrock VLM OCR).
    - comprehend_page_time (float, optional): Approximate time to query text on a page with AWS Comprehend.
    - local_text_redaction_page_time (float, optional): Approximate time to extract text on a page with the local text redaction option.
    - local_pii_redaction_page_time (float, optional): Approximate time to redact text on a page with the local text redaction option.
    - local_ocr_extraction_page_time (float, optional): Approximate time to extract text from a page with the local OCR redaction option.
    - TEXTRACT_TEXT_EXTRACT_OPTION (str, optional): String label for the text_extract_method_radio button for AWS Textract.
    - SELECTABLE_TEXT_EXTRACT_OPTION (str, optional): String label for text_extract_method_radio for text extraction.
    - local_ocr_option (str, optional): String label for text_extract_method_radio for local OCR.
    - NO_REDACTION_PII_OPTION (str, optional): String label for pii_identification_method_drop for no redaction.
    - AWS_PII_OPTION (str, optional): String label for pii_identification_method_drop for AWS Comprehend.
    - BEDROCK_VLM_TEXT_EXTRACT_OPTION, AWS_LLM_PII_OPTION (str, optional): Labels for Bedrock VLM OCR and LLM PII; times match Textract and Comprehend respectively.
    """
    calculated_time_taken = 0
    page_conversion_time_taken = 0
    page_extraction_time_taken = 0
    page_redaction_time_taken = 0

    number_of_pages = int(number_of_pages)

    # Page preparation/conversion to image time
    if (text_extract_method_radio != SELECTABLE_TEXT_EXTRACT_OPTION) and (
        textract_output_found_checkbox is not True
    ):
        page_conversion_time_taken = number_of_pages * convert_page_time

    # Page text extraction time
    if text_extract_method_radio == TEXTRACT_TEXT_EXTRACT_OPTION:
        if textract_output_found_checkbox is not True:
            page_extraction_time_taken = number_of_pages * textract_page_time
    elif text_extract_method_radio == BEDROCK_VLM_TEXT_EXTRACT_OPTION:
        if textract_output_found_checkbox is not True:
            page_extraction_time_taken = number_of_pages * textract_page_time
            if "Face detection" in (handwrite_signature_checkbox or []):
                page_extraction_time_taken *= 2
    elif text_extract_method_radio == local_ocr_option:
        if local_ocr_output_found_checkbox is not True:
            page_extraction_time_taken = (
                number_of_pages * local_ocr_extraction_page_time
            )
    elif text_extract_method_radio == SELECTABLE_TEXT_EXTRACT_OPTION:
        page_conversion_time_taken = number_of_pages * local_text_extraction_page_time

    # Page redaction time (Bedrock LLM PII uses same time as AWS Comprehend)
    if pii_identification_method != NO_REDACTION_PII_OPTION:
        if pii_identification_method in (AWS_PII_OPTION, AWS_LLM_PII_OPTION):
            page_redaction_time_taken = number_of_pages * comprehend_page_time
        else:
            page_redaction_time_taken = number_of_pages * local_pii_redaction_page_time

    calculated_time_taken = (
        page_conversion_time_taken
        + page_extraction_time_taken
        + page_redaction_time_taken
    ) / 60

    return calculated_time_taken


def reset_base_dataframe(df: pd.DataFrame):
    return df


def reset_ocr_base_dataframe(df: pd.DataFrame):
    if df.empty:
        print("OCR base dataframe is empty, returning empty dataframe")
        return pd.DataFrame(columns=["page", "line", "text"])
    else:
        return df.loc[:, ["page", "line", "text"]]


def reset_ocr_with_words_base_dataframe(
    df: pd.DataFrame, page_entity_dropdown_redaction_value: str
):
    """
    Resets and prepares the OCR dataframe with word-level details for a specific page.

    Args:
        df (pd.DataFrame): The original dataframe that may contain word-level OCR results,
                           with at least a 'page' column.
        page_entity_dropdown_redaction_value (str): The currently selected page value
            to filter the dataframe on, for redaction view.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of (filtered dataframe for selected page, full dataframe after processing).
    """

    if "page" not in df.columns:
        print(
            "OCR with words dataframe does not contain page column, returning empty dataframe"
        )
        df_out = pd.DataFrame(
            columns=[
                "page",
                "line",
                "word_text",
                "word_x0",
                "word_y0",
                "word_x1",
                "word_y1",
                "index",
            ]
        )
        df_filtered_out = pd.DataFrame(
            columns=[
                "page",
                "line",
                "word_text",
                "index",
            ]
        )
        return df_filtered_out, df_out

    df.reset_index(drop=True, inplace=True)
    df["index"] = df.index

    output_df_base = df.copy()

    if output_df_base.empty:
        print("OCR with words dataframe is empty, returning empty dataframe")

    df["page"] = df["page"].astype(str)

    output_df_filtered = df.loc[
        df["page"] == str(page_entity_dropdown_redaction_value),
        [
            "page",
            "line",
            "word_text",
            "index",
        ],
    ]

    if output_df_filtered.empty:
        print("No OCR results found for page, returning empty dataframe")

    return output_df_filtered, output_df_base


def update_language_dropdown(
    chosen_language_full_name_drop,
    textract_language_choices=textract_language_choices,
    aws_comprehend_language_choices=aws_comprehend_language_choices,
    LANGUAGE_MAP=LANGUAGE_MAP,
):

    try:
        full_language_name = chosen_language_full_name_drop.lower()
        matched_language = LANGUAGE_MAP[full_language_name]

        chosen_language_drop = gr.Dropdown(
            value=matched_language,
            choices=LANGUAGE_CHOICES,
            label="Chosen language short code",
            multiselect=False,
            visible=True,
        )

        if (
            matched_language not in aws_comprehend_language_choices
            and matched_language not in textract_language_choices
        ):
            gr.Info(
                f"Note that {full_language_name} is not supported by AWS Comprehend or AWS Textract"
            )
        elif matched_language not in aws_comprehend_language_choices:
            gr.Info(
                f"Note that {full_language_name} is not supported by AWS Comprehend"
            )
        elif matched_language not in textract_language_choices:
            gr.Info(f"Note that {full_language_name} is not supported by AWS Textract")
    except Exception as e:
        print(e)
        gr.Info("Could not find language in list")
        chosen_language_drop = gr.Dropdown(
            value=DEFAULT_LANGUAGE,
            choices=LANGUAGE_CHOICES,
            label="Chosen language short code",
            multiselect=False,
        )

    return chosen_language_drop


def get_system_font_path():
    """
    Returns the path to a standard font that exists on most operating systems.
    Used to replace PaddleOCR's default fonts (simfang.ttf, PingFang-SC-Regular.ttf).

    Returns:
        str: Path to a system font, or None if no suitable font found
    """
    system = platform.system()

    # Windows font paths
    if system == "Windows":
        windows_fonts = [
            os.path.join(
                os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "simsun.ttc"
            ),  # SimSun
            os.path.join(
                os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "msyh.ttc"
            ),  # Microsoft YaHei
            os.path.join(
                os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf"
            ),  # Arial (fallback)
        ]
        for font_path in windows_fonts:
            if os.path.exists(font_path):
                return font_path

    # macOS font paths
    elif system == "Darwin":
        mac_fonts = [
            "/System/Library/Fonts/STSong.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        for font_path in mac_fonts:
            if os.path.exists(font_path):
                return font_path

    # Linux font paths
    elif system == "Linux":
        linux_fonts = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for font_path in linux_fonts:
            if os.path.exists(font_path):
                return font_path

    return None


def get_ocr_visualisation_font_path():
    """
    Path to a TrueType/OpenType font for drawing recognised OCR text on debug images.

    OpenCV's Hershey fonts (cv2.putText) are effectively Latin-only; other scripts show as '?'.
    PIL + this font supports Cyrillic, CJK, Arabic in typical OS installs (Segoe UI, Noto, etc.).
    """
    system = platform.system()

    if system == "Windows":
        fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        for name in (
            "segoeui.ttf",
            "calibri.ttf",
            "msyh.ttc",
            "simsun.ttc",
            "arial.ttf",
            "times.ttf",
        ):
            p = os.path.join(fonts_dir, name)
            if os.path.isfile(p):
                return p

    elif system == "Darwin":
        for p in (
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
        ):
            if os.path.isfile(p):
                return p

    elif system == "Linux":
        for p in (
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            if os.path.isfile(p):
                return p

    return get_system_font_path()


# Custom logging filter to remove logs from healthiness/readiness endpoints so they don't fill up application log flow
class EndpointFilter(logging.Filter):
    def __init__(self, path: str, *args, **kwargs):
        self._path = path
        super().__init__(*args, **kwargs)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find(self._path) == -1


# 2. Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    # Filter out /health logging to declutter ECS logs
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(EndpointFilter(path="/health"))

    # Yield control back to the application
    yield

    pass


def check_duplicate_pages_checkbox(redact_duplicate_pages_checkbox_value: bool):
    if not redact_duplicate_pages_checkbox_value:
        # Silently raise an error to avoid showing a popup
        return
    if redact_duplicate_pages_checkbox_value:
        print("Identifying duplicates")
        sys.tracebacklimit = 0  # Suppress traceback
        gr.Info("Redact duplicate pages checkbox is enabled, Identifying duplicates")
        raise gr.Error(
            message="Redact duplicate pages checkbox is enabled, identifying duplicates.",
            title="Finding duplicates...",
            visible=False,
            print_exception=False,
        )


# Tab switch functions
def change_tab_to_tabular_or_document_redactions(is_data_file):
    if is_data_file:
        return gr.Tabs(selected=5)
    else:
        return gr.Tabs(selected=1)


def change_tab_to_review_redactions():
    return gr.Tabs(selected=2)


### Examples functions
def show_info_box_on_click(
    in_doc_files,
    text_extract_method_radio,
    pii_identification_method_drop,
    handwrite_signature_checkbox,
    in_redact_entities,
    in_redact_comprehend_entities,
    prepared_pdf_state,
    doc_full_file_name_textbox,
    doc_file_name_no_extension_textbox,
    in_deny_list,
    in_deny_list_state,
    in_fully_redacted_list,
    in_fully_redacted_list_state,
    total_pdf_page_count,
):
    gr.Info(
        "Example data loaded. Now click on 'Extract text and redact document' on the Redact PDFs/images tab to run the example redaction."
    )

    # Convert deny_list_state, allow_list_state, and fully_redacted_list_state to lists if they are DataFrames
    # Handle deny_list_state
    deny_list_walkthrough = []
    if isinstance(in_deny_list_state, pd.DataFrame):
        # Explicitly convert empty DataFrame to empty list
        if in_deny_list_state.empty:
            deny_list_walkthrough = []
        else:
            deny_list_walkthrough = (
                in_deny_list_state.iloc[:, 0].dropna().astype(str).tolist()
            )
    elif isinstance(in_deny_list_state, list):
        deny_list_walkthrough = (
            [str(item) for item in in_deny_list_state if item]
            if in_deny_list_state
            else []
        )
    else:
        # Default to empty list for any other type
        deny_list_walkthrough = []

    # Handle fully_redacted_list_state
    fully_redacted_list_walkthrough = []
    if isinstance(in_fully_redacted_list_state, pd.DataFrame):
        # Explicitly convert empty DataFrame to empty list
        if in_fully_redacted_list_state.empty:
            fully_redacted_list_walkthrough = []
        else:
            fully_redacted_list_walkthrough = (
                in_fully_redacted_list_state.iloc[:, 0].dropna().astype(str).tolist()
            )
    elif isinstance(in_fully_redacted_list_state, list):
        fully_redacted_list_walkthrough = (
            [str(item) for item in in_fully_redacted_list_state if item]
            if in_fully_redacted_list_state
            else []
        )
    else:
        # Default to empty list for any other type
        fully_redacted_list_walkthrough = []

    # Allow list is not in examples, so always set to empty list
    allow_list_walkthrough = []

    # Use default local OCR method - examples don't set this directly
    local_ocr_method = DEFAULT_LOCAL_OCR_MODEL

    # Update visibility of main PII entity components based on selected PII method
    # This ensures visibility is correct even when clicking examples with the same PII method
    # Determine visibility based on PII method (same logic as handle_main_pii_method_selection)
    is_no_redaction = pii_identification_method_drop == NO_REDACTION_PII_OPTION
    show_local_entities = (
        not is_no_redaction and pii_identification_method_drop == LOCAL_PII_OPTION
    )
    show_comprehend_entities = (
        not is_no_redaction and pii_identification_method_drop == AWS_PII_OPTION
    )
    is_llm_method = not is_no_redaction and (
        pii_identification_method_drop == LOCAL_TRANSFORMERS_LLM_PII_OPTION
        or pii_identification_method_drop == INFERENCE_SERVER_PII_OPTION
        or pii_identification_method_drop == AWS_LLM_PII_OPTION
    )

    # Create updates with both value and visibility for main components
    main_local_entities_update = gr.update(
        value=in_redact_entities,
        visible=show_local_entities,
    )
    main_comprehend_entities_update = gr.update(
        value=in_redact_comprehend_entities,
        visible=show_comprehend_entities,
    )
    main_llm_entities_update = gr.update(
        visible=is_llm_method,
    )
    main_llm_instructions_update = gr.update(
        visible=True,
    )

    # Set visibility on walkthrough entity dropdowns so they match PII method after example load
    walkthrough_local_update = gr.update(
        value=in_redact_entities, visible=show_local_entities
    )
    walkthrough_comprehend_update = gr.update(
        value=in_redact_comprehend_entities, visible=show_comprehend_entities
    )

    return (
        gr.File(value=in_doc_files, visible=True),  # walkthrough_file_input
        walkthrough_local_update,  # walkthrough_in_redact_entities
        walkthrough_comprehend_update,  # walkthrough_in_redact_comprehend_entities
        gr.Radio(
            value=text_extract_method_radio, visible=True
        ),  # walkthrough_text_extract_method_radio
        gr.Radio(
            value=local_ocr_method, visible=True
        ),  # walkthrough_local_ocr_method_radio
        gr.CheckboxGroup(
            value=handwrite_signature_checkbox, visible=True
        ),  # walkthrough_handwrite_signature_checkbox
        gr.Radio(
            value=pii_identification_method_drop, visible=True
        ),  # walkthrough_pii_identification_method_drop
        gr.Dropdown(
            value=allow_list_walkthrough, visible=True
        ),  # walkthrough_allow_list_state
        gr.Dropdown(
            value=deny_list_walkthrough, visible=True
        ),  # walkthrough_deny_list_state
        gr.Dropdown(
            value=fully_redacted_list_walkthrough, visible=True
        ),  # walkthrough_fully_redacted_list_state
        main_local_entities_update,  # in_redact_entities (main component)
        main_comprehend_entities_update,  # in_redact_comprehend_entities (main component)
        main_llm_entities_update,  # in_redact_llm_entities (main component)
        main_llm_instructions_update,  # custom_llm_instructions_textbox (main component)
    )


def show_info_box_on_click_ocr_examples(
    in_doc_files,
    text_extract_method_radio,
    pii_identification_method_drop,
    handwrite_signature_checkbox,
    prepared_pdf_state,
    doc_full_file_name_textbox,
    doc_file_name_no_extension_textbox,
    total_pdf_page_count,
    page_min,
    page_max,
    local_ocr_method_radio,
    in_redact_entities,
    in_redact_llm_entities,
    custom_llm_instructions_textbox,
):
    gr.Info(
        "Example OCR data loaded. Now click on 'Extract text and redact document' on the Redact PDFs/images tab to run the OCR analysis."
    )

    is_no_redaction = pii_identification_method_drop == NO_REDACTION_PII_OPTION
    show_local_entities = (
        not is_no_redaction and pii_identification_method_drop == LOCAL_PII_OPTION
    )
    is_llm_method = not is_no_redaction and (
        pii_identification_method_drop == LOCAL_TRANSFORMERS_LLM_PII_OPTION
        or pii_identification_method_drop == INFERENCE_SERVER_PII_OPTION
        or pii_identification_method_drop == AWS_LLM_PII_OPTION
    )

    main_local_entities_update = gr.update(
        value=in_redact_entities,
        visible=show_local_entities,
    )

    main_llm_entities_update = gr.update(
        value=in_redact_llm_entities,
        visible=is_llm_method,
    )
    main_llm_instructions_update = gr.update(
        value=custom_llm_instructions_textbox,
        visible=True,
    )

    return (
        gr.File(value=in_doc_files, visible=True),  # walkthrough_file_input
        main_local_entities_update,  # walkthrough_in_redact_entities
        gr.Radio(
            value=text_extract_method_radio, visible=True
        ),  # walkthrough_text_extract_method_radio
        gr.Radio(
            value=local_ocr_method_radio, visible=True
        ),  # walkthrough_local_ocr_method_radio
        gr.CheckboxGroup(
            value=handwrite_signature_checkbox, visible=True
        ),  # walkthrough_handwrite_signature_checkbox
        gr.Radio(
            value=pii_identification_method_drop, visible=True
        ),  # walkthrough_pii_identification_method_drop
        main_llm_entities_update,  # walkthrough_in_redact_llm_entities
        main_llm_instructions_update,  # walkthrough_custom_llm_instructions_textbox
        main_llm_entities_update,  # in_redact_llm_entities (main component)
        main_llm_instructions_update,  # custom_llm_instructions_textbox (main component)
    )


def show_duplicate_info_box_on_click(
    in_duplicate_pages,
    duplicate_threshold_input,
    min_word_count_input,
    combine_page_text_for_duplicates_bool,
):
    gr.Info(
        "Example data loaded. Now click on 'Identify duplicate pages/subdocuments' on the Identify duplicate pages tab to run the example duplicate detection."
    )


def show_tabular_info_box_on_click(
    in_data_files,
    in_colnames,
    pii_identification_method_drop_tabular,
    anon_strategy,
    in_tabular_duplicate_files,
    tabular_text_columns,
    tabular_min_word_count,
):
    gr.Info(
        "Example data loaded. Now click on 'Redact text/data files' or 'Find duplicate cells/rows' on the Word or Excel/CSV files tab to run the example."
    )

    return (
        gr.File(value=in_data_files),  # walkthrough_file_input
        gr.Radio(
            value=pii_identification_method_drop_tabular
        ),  # walkthrough_pii_identification_method_drop_tabular
        gr.Radio(value=anon_strategy),  # walkthrough_anon_strategy
    )


# Dynamic visibility handlers for main redaction tab (run regardless of SHOW_COSTS)
# Automatically set local_ocr_method_radio to "bedrock-vlm" when AWS Bedrock VLM is selected
def auto_set_local_ocr_for_bedrock_vlm(text_extract_method):
    """Automatically set local OCR method to bedrock-vlm when AWS Bedrock VLM is selected."""
    if text_extract_method == BEDROCK_VLM_TEXT_EXTRACT_OPTION:
        # Only set if "bedrock-vlm" is a valid option
        if "bedrock-vlm" in LOCAL_OCR_MODEL_OPTIONS:
            return gr.update(value="bedrock-vlm")
    return gr.update()
