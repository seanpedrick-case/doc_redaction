import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import boto3
import gradio as gr
import markdown
import pandas as pd
import spaces
from gradio import Progress as progress
from tqdm import tqdm

from tools.config import (
    API_URL,
    AWS_ACCESS_KEY,
    AWS_LLM_PII_OPTION,
    AWS_REGION,
    AWS_SECRET_KEY,
    BATCH_SIZE_DEFAULT,
    CLOUD_LLM_PII_MODEL_CHOICE,
    CLOUD_SUMMARISATION_MODEL_CHOICE,
    DEDUPLICATION_THRESHOLD,
    DEFAULT_INFERENCE_SERVER_PII_MODEL,
    INFERENCE_SERVER_PII_OPTION,
    LLM_CONTEXT_LENGTH,
    LLM_MAX_NEW_TOKENS,
    LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE,
    LOCAL_TRANSFORMERS_LLM_PII_OPTION,
    MAX_COMMENT_CHARS,
    MAX_SPACES_GPU_RUN_TIME,
    MAX_TIME_FOR_LOOP,
    NUMBER_OF_RETRY_ATTEMPTS,
    OUTPUT_FOLDER,
    PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS,
    REASONING_SUFFIX,
    RUN_AWS_FUNCTIONS,
    SUMMARY_PAGE_GROUP_MAX_WORKERS,
    TIMEOUT_WAIT,
    model_name_map,
)
from tools.file_conversion import word_level_ocr_df_to_line_level_ocr_df
from tools.helper_functions import (
    clean_column_name,
    create_batch_file_path_details,
    get_file_name_no_ext,
)
from tools.llm_funcs import (
    calculate_tokens_from_metadata,
    construct_azure_client,
    construct_gemini_generative_model,
    load_model,
    process_requests,
)

max_tokens = LLM_MAX_NEW_TOKENS
timeout_wait = TIMEOUT_WAIT
number_of_api_retry_attempts = NUMBER_OF_RETRY_ATTEMPTS
max_time_for_loop = MAX_TIME_FOR_LOOP
batch_size_default = BATCH_SIZE_DEFAULT
deduplication_threshold = DEDUPLICATION_THRESHOLD
max_comment_character_length = MAX_COMMENT_CHARS
reasoning_suffix = REASONING_SUFFIX
max_text_length = 500

###
# System prompt
###

generic_system_prompt = """You are a researcher analysing a document. Use British English spelling and grammar."""

system_prompt = """You are a researcher analysing a document. Use British English spelling and grammar."""

markdown_additional_prompt = """ You will be given a request for a markdown table. You must respond with ONLY the markdown table. Do not include any introduction, explanation, or concluding text."""

###
# SUMMARISE TOPICS PROMPT
###

summary_assistant_prefill = ""

summarise_topic_descriptions_system_prompt = system_prompt

summarise_topic_descriptions_prompt = """Your task is to make a consolidated summary of the text below. {summary_format}
Return only the summary and no other text. Do not mention specific response numbers in the summary.{additional_summary_instructions}

Text to summarise:
{summaries}

Summary:"""

concise_summary_format_prompt = "Return a concise summary that summarises only the most important themes from the original text"

detailed_summary_format_prompt = (
    "Return a summary that includes as much detail as possible from the original text"
)

###
# OVERALL SUMMARY PROMPTS
###

summarise_everything_system_prompt = system_prompt

summarise_everything_prompt = """Below is a table that gives an overview of the main issues related to a document. 
Your task is to summarise the text in the table below. {summary_format}. Return only the summary and no other text. Use headers and paragraphs to structure the summary where appropriate. Format the output for Excel display using: **bold text** for main headings, • bullet points for sub-items, and line breaks between sections. Avoid markdown symbols like # or ##. {additional_summary_instructions}

Table to summarise:
{topic_summary_table}

Summary:"""

# comprehensive_summary_format_prompt = "Return a comprehensive summary that covers all the important topics and themes described in the summaries below. Structure the summary with Main issues as headings, with significant topics described in bullet points below them in order of relative significance. Format the output for Excel display using: **bold text** for main headings, • bullet points for sub-items, and line breaks between sections. Avoid markdown symbols like # or ##."

# comprehensive_summary_format_prompt_by_group = "Return a comprehensive summary that covers all the important topics and themes described in the summaries below. Structure the summary with main issues as headings, with significant Subtopics described in bullet points below them in order of relative significance. Compare and contrast differences between the topics and themes from each Group. Format the output for Excel display using: **bold text** for main headings, • bullet points for sub-items, and line breaks between sections. Avoid markdown symbols like # or ##."

# # Alternative Excel formatting options
# excel_rich_text_format_prompt = "Return a comprehensive summary that covers all the important topics and themes described in the summaries below. Structure the summary with main issues as headings, with significant topics described in bullet points below them in order of relative significance. Format for Excel using: BOLD for main headings, bullet points (•) for sub-items, and line breaks between sections. Use simple text formatting that Excel can interpret."

# excel_plain_text_format_prompt = "Return a comprehensive summary that covers all the important topics and themes described in the summaries below. Structure the summary with main issues as headings, with significant topics described in bullet points below them in order of relative significance. Format as plain text with clear structure: use ALL CAPS for main headings, bullet points (•) for sub-items, and line breaks between sections. Avoid any special formatting symbols."


###
# Document Summarisation Functions
###
def get_model_choice_from_inference_method(inference_method: str) -> str:
    """
    Get the default model choice for a given inference method (for summarisation).
    Uses the default values defined in config.py (CLOUD_SUMMARISATION_MODEL_CHOICE for cloud).

    Args:
        inference_method: One of "aws-bedrock", "local", "inference-server"

    Returns:
        str: The model choice string to use
    """
    # Map inference method to model choice using defaults from config.py
    if inference_method == "aws-bedrock":
        return CLOUD_SUMMARISATION_MODEL_CHOICE
    elif inference_method == "local":
        return LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE
    elif inference_method == "inference-server":
        return DEFAULT_INFERENCE_SERVER_PII_MODEL
    else:
        raise ValueError(
            f"Unknown inference method: {inference_method}. "
            f"Expected one of: 'aws-bedrock', 'local', 'inference-server'"
        )


def get_model_source_from_model_choice(model_choice: str) -> str:
    """
    Determine model source from model_choice by comparing to defaults from config.py.
    Does not check model_name_map - uses the defined defaults.

    Args:
        model_choice: The model choice string

    Returns:
        str: The model source ("AWS", "Local", or "inference-server")
    """
    # Compare model_choice to the default config values to determine source
    if model_choice == LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE:
        return "Local"
    elif model_choice == DEFAULT_INFERENCE_SERVER_PII_MODEL:
        return "inference-server"
    elif (
        model_choice == CLOUD_LLM_PII_MODEL_CHOICE
        or model_choice == CLOUD_SUMMARISATION_MODEL_CHOICE
    ):
        return "AWS"
    else:
        # If it doesn't match any default, infer from common patterns
        # AWS Bedrock models typically have "amazon." or "anthropic." prefix
        if model_choice.startswith("amazon.") or model_choice.startswith("anthropic."):
            return "AWS"
        # Inference server models are often custom names
        # Default to AWS for backward compatibility, but could be inference-server
        # Since we're using defaults, assume AWS if it's not clearly local
        return "AWS"


def load_csv_files_to_dataframe(file_input):
    """
    Load CSV files from Gradio file input and combine them into a single DataFrame.
    Similar to how duplicate pages function handles file input.

    Args:
        file_input: Gradio file input (can be a single file, list of files, or file objects)

    Returns:
        pd.DataFrame: Combined DataFrame with columns page, line, and text
    """
    if not file_input:
        return pd.DataFrame(columns=["page", "line", "text"])

    # Handle different input types (similar to run_tabular_duplicate_detection)
    file_paths = []
    if isinstance(file_input, str):
        file_paths.append(file_input)
    elif isinstance(file_input, list):
        for f_item in file_input:
            if isinstance(f_item, str):
                file_paths.append(f_item)
            elif hasattr(f_item, "name"):
                file_paths.append(f_item.name)
    elif hasattr(file_input, "name"):
        file_paths.append(file_input.name)

    # Load and combine all CSV files
    all_dfs = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            # Convert word-level OCR to line-level if user uploaded word-level file
            if "ocr_results_with_words" in os.path.basename(file_path) and (
                "word_text" in df.columns and "text" not in df.columns
            ):
                df = word_level_ocr_df_to_line_level_ocr_df(df)
            # Ensure required columns exist
            if "page" in df.columns and "line" in df.columns and "text" in df.columns:
                all_dfs.append(df[["page", "line", "text"]])
            else:
                print(
                    f"Warning: {file_path} does not have required columns (page, line, text)"
                )
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not all_dfs:
        return pd.DataFrame(columns=["page", "line", "text"])

    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


# Wrapper function to convert inference method to model choice
@spaces.GPU(duration=MAX_SPACES_GPU_RUN_TIME)
def summarise_document_wrapper(
    all_page_line_level_ocr_results_df,
    output_folder,
    summarisation_inference_method,
    summarisation_api_key,
    summarisation_temperature,
    file_name,
    summarisation_context,
    summarisation_aws_access_key,
    summarisation_aws_secret_key,
    summarisation_hf_api_key,
    summarisation_azure_endpoint,
    summarisation_format,
    summarisation_additional_instructions,
    summarisation_max_pages_per_group,
    in_summarisation_ocr_files=None,
):
    """
    Wrapper to select the correct model and format for document summarization, and optionally
    load input OCR CSV files if they are provided.

    Args:
        all_page_line_level_ocr_results_df (pd.DataFrame): Pre-loaded DataFrame containing the line-level OCR results.
        output_folder (str): Path to folder where outputs should be saved.
        summarisation_inference_method (str): String specifying which inference/LLM method to use ('aws-bedrock', etc).
        summarisation_api_key (str): API key for the selected inference method, if required.
        summarisation_temperature (float): The temperature parameter for the model (controls randomness).
        file_name (str): Name to use as a base for output files.
        summarisation_context (str): Additional context string to include in the summarization.
        summarisation_aws_access_key (str): AWS access key if using AWS inference.
        summarisation_aws_secret_key (str): AWS secret key if using AWS inference.
        summarisation_hf_api_key (str): HuggingFace API key if required.
        summarisation_azure_endpoint (str): Endpoint string if using Azure inference.
        summarisation_format (str): Format for the summary output (e.g., "bullets", "structured").
        summarisation_additional_instructions (str): Extra instructions to pass to the summarization LLM.
        summarisation_max_pages_per_group (int): Maximum number of pages to group per LLM summarization pass.
        in_summarisation_ocr_files (str | list | object, optional): One or more file paths or file-like objects to OCR results in CSV format.

    Returns:
        Output of the downstream summarisation process (see next code section for details).
    """
    """Wrapper to convert inference method selection to model choice and load CSV files."""
    # Map inference method option to inference method string
    inference_method_map = {
        AWS_LLM_PII_OPTION: "aws-bedrock",
        LOCAL_TRANSFORMERS_LLM_PII_OPTION: "local",
        INFERENCE_SERVER_PII_OPTION: "inference-server",
    }

    inference_method = inference_method_map.get(
        summarisation_inference_method, "aws-bedrock"
    )

    # Use config default for region
    summarisation_aws_region = AWS_REGION
    summarisation_api_url = API_URL

    # Get model choice from inference method
    model_choice = get_model_choice_from_inference_method(inference_method)

    # Load CSV files if provided, otherwise use the dataframe
    if in_summarisation_ocr_files:
        ocr_df = load_csv_files_to_dataframe(in_summarisation_ocr_files)
    else:
        ocr_df = all_page_line_level_ocr_results_df

    # If file_name is None or empty, derive it from in_summarisation_ocr_files
    if not file_name or file_name.strip() == "":
        if in_summarisation_ocr_files:
            # Extract file path from in_summarisation_ocr_files (similar to load_csv_files_to_dataframe)
            file_paths = []
            if isinstance(in_summarisation_ocr_files, str):
                file_paths.append(in_summarisation_ocr_files)
            elif isinstance(in_summarisation_ocr_files, list):
                for f_item in in_summarisation_ocr_files:
                    if isinstance(f_item, str):
                        file_paths.append(f_item)
                    elif hasattr(f_item, "name"):
                        file_paths.append(f_item.name)
            elif hasattr(in_summarisation_ocr_files, "name"):
                file_paths.append(in_summarisation_ocr_files.name)

            # Get the first file path and extract filename prefix
            if file_paths:
                first_file_path = file_paths[0]
                # Get basename without extension
                basename = os.path.basename(first_file_path)
                filename_without_ext, _ = os.path.splitext(basename)
                # Take first 20 characters, removing any invalid filename characters
                filename_prefix = filename_without_ext[:20]
                # Remove any invalid characters for filenames
                invalid_chars = '<>:"/\\|?*'
                for char in invalid_chars:
                    filename_prefix = filename_prefix.replace(char, "_")
                file_name = filename_prefix if filename_prefix else "document"
            else:
                file_name = "document"
        else:
            file_name = "document"

    # Call the actual summarise_document function (timed for usage logs)
    start_time = time.perf_counter()
    (
        output_files,
        status_message,
        llm_model_name,
        llm_total_input_tokens,
        llm_total_output_tokens,
        summary_display_text,
    ) = summarise_document(
        ocr_df,
        output_folder,
        model_choice,
        summarisation_api_key,
        summarisation_temperature,
        file_name,
        summarisation_context,
        summarisation_aws_access_key,
        summarisation_aws_secret_key,
        summarisation_aws_region,
        summarisation_hf_api_key,
        summarisation_azure_endpoint,
        summarisation_api_url,
        summarisation_format,
        summarisation_additional_instructions,
        max_pages_per_group=summarisation_max_pages_per_group,
    )
    elapsed_seconds = round(time.perf_counter() - start_time, 1)

    return (
        output_files,
        status_message,
        llm_model_name,
        llm_total_input_tokens,
        llm_total_output_tokens,
        summary_display_text,
        elapsed_seconds,
    )


def group_pages_by_context_length(
    all_page_line_level_ocr_results_df: pd.DataFrame,
    context_length: int = LLM_CONTEXT_LENGTH,
    tokenizer=None,
    model_source: str = "Local",
    max_pages_per_group: int = 30,
) -> List[Tuple[List[int], str]]:
    """
    Group pages into chunks that fit within the LLM context length.
    Splits pages into roughly equal-sized groups (e.g. 56 pages with room for 50
    per context -> two groups of 28, not 50 and 6). Each page is prefixed with
    '=== Page x ==='.

    Args:
        all_page_line_level_ocr_results_df: DataFrame with columns 'page', 'line', 'text'
        context_length: Maximum context length in tokens
        tokenizer: Tokenizer for accurate token counting
        model_source: Source of the model for token counting

    Returns:
        List of tuples: (list of page numbers, formatted text for that group)
    """
    if (
        all_page_line_level_ocr_results_df is None
        or all_page_line_level_ocr_results_df.empty
    ):
        return []

    # Group by page and concatenate text
    page_texts = {}
    for _, row in all_page_line_level_ocr_results_df.iterrows():
        page = int(row["page"])
        text = str(row.get("text", ""))
        if page not in page_texts:
            page_texts[page] = []
        page_texts[page].append(text)

    # Format each page with header and get token count per page
    page_list = []  # (page_num, formatted_page, page_tokens)
    for page_num in sorted(page_texts.keys()):
        page_text = " ".join(page_texts[page_num])
        formatted_page = f"=== Page {page_num} ===\n{page_text}"
        page_tokens = count_tokens_in_text(formatted_page, tokenizer, model_source)
        page_list.append((page_num, formatted_page, page_tokens))

    # Reserve some tokens for the prompt template
    reserved_tokens = 500
    available_tokens = context_length - reserved_tokens

    if not page_list:
        return []

    # Sanitise max_pages_per_group
    try:
        max_pages_per_group_int = int(max_pages_per_group)
    except Exception:
        max_pages_per_group_int = 30
    if max_pages_per_group_int < 1:
        max_pages_per_group_int = 1

    # Step 1: Greedy pass to determine minimum number of groups by tokens
    k_token = 0
    cur_tokens = 0
    for _, _, pt in page_list:
        if cur_tokens + pt > available_tokens and cur_tokens > 0:
            k_token += 1
            cur_tokens = 0
        cur_tokens += pt
    k_token += 1  # last group
    n = len(page_list)

    # Also enforce a maximum pages-per-group cap
    k_pages = (n + max_pages_per_group_int - 1) // max_pages_per_group_int

    # Final number of groups must satisfy both token limit and max-pages limit
    k = max(k_token, k_pages)

    # Step 2: Target pages per group for roughly equal split (e.g. 56 pages, 2 groups -> 28, 28)
    q, r = n // k, n % k
    target_per_group = [q + 1] * r + [q] * (k - r)

    # Step 3: Assign pages to groups with target sizes, respecting token limit
    groups = []
    page_idx = 0
    for group_idx in range(k):
        target = min(target_per_group[group_idx], max_pages_per_group_int)
        current_group_pages = []
        current_group_text = ""
        current_tokens = 0
        while page_idx < n and len(current_group_pages) < target:
            page_num, formatted_page, page_tokens = page_list[page_idx]
            if current_tokens + page_tokens > available_tokens and current_group_pages:
                break  # full by token limit; start next group
            current_group_pages.append(page_num)
            if current_group_text:
                current_group_text += "\n\n" + formatted_page
            else:
                current_group_text = formatted_page
            current_tokens += page_tokens
            page_idx += 1
        if current_group_pages:
            groups.append((current_group_pages, current_group_text))

    # Any remaining pages (e.g. group hit token limit before target) go into final group(s)
    while page_idx < n:
        current_group_pages = []
        current_group_text = ""
        current_tokens = 0
        while page_idx < n and len(current_group_pages) < max_pages_per_group_int:
            page_num, formatted_page, page_tokens = page_list[page_idx]
            if current_tokens + page_tokens > available_tokens and current_group_pages:
                break
            # If even a single page exceeds limit, add it anyway to avoid infinite loop
            current_group_pages.append(page_num)
            if current_group_text:
                current_group_text += "\n\n" + formatted_page
            else:
                current_group_text = formatted_page
            current_tokens += page_tokens
            page_idx += 1
        if current_group_pages:
            groups.append((current_group_pages, current_group_text))

    return groups


def summarise_text_chunk(
    text_chunk: str,
    model_choice: str,
    in_api_key: str,
    temperature: float,
    context_textbox: str = "",
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    aws_region_textbox: str = "",
    model_name_map: dict = None,
    hf_api_key_textbox: str = "",
    azure_endpoint_textbox: str = "",
    api_url: str = None,
    reasoning_suffix: str = "",
    local_model=None,
    tokenizer=None,
    assistant_model=None,
    summarise_format_radio: str = "Return a summary up to two paragraphs long that includes as much detail as possible from the original text",
    additional_summary_instructions: str = "",
) -> Tuple[str, str, dict]:
    """
    Summarise a single text chunk using the summarise_output_topics_query function.

    Returns:
        Tuple of (summary_text, full_prompt, metadata)
    """
    from tools.config import (
        model_name_map as default_model_name_map,
    )

    # Note: load_model is already imported at the top of the file

    if model_name_map is None:
        model_name_map = default_model_name_map

    if additional_summary_instructions:
        additional_summary_instructions = (
            "Important additional instructions to follow closely: "
            + additional_summary_instructions
        )

    formatted_summary_prompt = [
        summarise_topic_descriptions_prompt.format(
            summaries=text_chunk,
            summary_format=summarise_format_radio,
            additional_summary_instructions=additional_summary_instructions,
        )
    ]

    # Format system prompt
    formatted_system_prompt = summarise_topic_descriptions_system_prompt.format(
        column_name="document text",
        consultation_context=context_textbox if context_textbox else "",
    )

    # Determine model source from model_choice using defaults from config.py
    # Does not check model_name_map - uses the defined defaults
    model_source = get_model_source_from_model_choice(model_choice)

    # Setup model based on model source
    # Load model and tokenizer together to ensure they're from the same source
    # This prevents mismatches that could occur if they're loaded separately
    # Similar to llm_funcs.py pattern (lines 830-839) and llm_entity_detection.py (lines 519-533)
    if (model_source == "Local") & (local_model is None or tokenizer is None):
        progress(0.1, f"Using model: {LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE}")
        # Use load_model() to ensure both are loaded atomically
        # This is safer than calling get_pii_model() and get_pii_tokenizer() separately
        loaded_model, loaded_tokenizer, loaded_assistant_model = load_model()
        if local_model is None:
            local_model = loaded_model
        if tokenizer is None:
            tokenizer = loaded_tokenizer
        if assistant_model is None:
            assistant_model = loaded_assistant_model

    # Setup bedrock for AWS models
    # Use the same approach as file_redaction.py (lines 939-969) for consistency
    bedrock_runtime = None
    if model_source == "AWS":
        # Use aws_region_textbox if provided, otherwise fall back to AWS_REGION from config
        region = aws_region_textbox if aws_region_textbox else AWS_REGION

        if RUN_AWS_FUNCTIONS and PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS:
            print("Connecting to Bedrock via existing SSO connection")
            bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
        elif aws_access_key_textbox and aws_secret_key_textbox:
            print(
                "Connecting to Bedrock using AWS access key and secret keys from user input."
            )
            bedrock_runtime = boto3.client(
                "bedrock-runtime",
                aws_access_key_id=aws_access_key_textbox,
                aws_secret_access_key=aws_secret_key_textbox,
                region_name=region,
            )
        elif RUN_AWS_FUNCTIONS:
            print("Connecting to Bedrock via existing SSO connection")
            bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
        elif AWS_ACCESS_KEY and AWS_SECRET_KEY:
            print("Getting Bedrock credentials from environment variables")
            bedrock_runtime = boto3.client(
                "bedrock-runtime",
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY,
                region_name=region,
            )
        else:
            bedrock_runtime = None
            out_message = "Cannot connect to AWS Bedrock service. Please provide access keys under LLM settings, or choose another model type."
            print(out_message)
            raise Exception(out_message)

    # Note: Gemini and Azure/OpenAI clients are handled within summarise_output_topics_query
    # via the process_requests function, so we don't need to set them up here
    # Similar to how llm_entity_detection.py handles them (lines 554-584)

    # Apply reasoning suffix if needed
    if reasoning_suffix:
        is_gpt_oss_model = (
            "gpt-oss" in model_choice.lower() or "gpt_oss" in model_choice.lower()
        )
        if is_gpt_oss_model or ("Local" in model_source and reasoning_suffix):
            formatted_system_prompt = formatted_system_prompt + "\n" + reasoning_suffix

    # Call the summarisation function
    try:
        response, conversation_history, metadata, response_text = (
            summarise_output_topics_query(
                model_choice,
                in_api_key,
                temperature,
                formatted_summary_prompt,
                formatted_system_prompt,
                model_source,
                bedrock_runtime,
                local_model if local_model else [],
                tokenizer if tokenizer else [],
                assistant_model if assistant_model else [],
                azure_endpoint_textbox,
                api_url,
            )
        )

        full_prompt = formatted_system_prompt + "\n" + formatted_summary_prompt[0]
        return response_text, full_prompt, metadata
    except Exception as e:
        print(f"Error summarising text chunk: {e}")
        full_prompt = formatted_system_prompt + "\n" + formatted_summary_prompt[0]
        return "", full_prompt, {}


def recursively_summarise(
    summaries: List[str],
    model_choice: str,
    in_api_key: str,
    temperature: float,
    context_length: int = LLM_CONTEXT_LENGTH,
    tokenizer=None,
    model_source: str = "Local",
    token_accumulator=None,
    **kwargs,
) -> List[str]:
    """
    Recursively summarise summaries until they fit within context length.

    Args:
        token_accumulator: Optional list to accumulate [input_tokens, output_tokens] from metadata
    """
    # Check total length
    combined_summaries = "\n\n".join(summaries)
    total_tokens = count_tokens_in_text(combined_summaries, tokenizer, model_source)

    # Reserve tokens for prompt
    reserved_tokens = 500
    available_tokens = context_length - reserved_tokens

    if total_tokens <= available_tokens:
        return summaries

    # Need to summarise further - group summaries into chunks
    groups = []
    current_group = []
    current_tokens = 0

    for summary in summaries:
        summary_tokens = count_tokens_in_text(summary, tokenizer, model_source)
        if current_tokens + summary_tokens > available_tokens and current_group:
            groups.append("\n\n".join(current_group))
            current_group = [summary]
            current_tokens = summary_tokens
        else:
            current_group.append(summary)
            current_tokens += summary_tokens

    if current_group:
        groups.append("\n\n".join(current_group))

    # Summarise each group
    new_summaries = []
    for group_text in groups:
        summary_text, _, metadata = summarise_text_chunk(
            group_text,
            model_choice,
            in_api_key,
            temperature,
            tokenizer=tokenizer,
            model_source=model_source,
            **kwargs,
        )
        if summary_text:
            new_summaries.append(summary_text)
            # Accumulate tokens if accumulator provided
            if token_accumulator is not None and metadata:
                # Convert metadata to string if it's a list
                metadata_string = (
                    str(metadata) if not isinstance(metadata, str) else metadata
                )
                input_tokens, output_tokens, _ = calculate_tokens_from_metadata(
                    metadata_string, model_choice, model_name_map
                )
                token_accumulator[0] += input_tokens
                token_accumulator[1] += output_tokens

    # Recursively call if still too long
    if len(new_summaries) > 1:
        return recursively_summarise(
            new_summaries,
            model_choice,
            in_api_key,
            temperature,
            context_length,
            tokenizer,
            model_source,
            token_accumulator=token_accumulator,
            **kwargs,
        )

    return new_summaries


def summarise_document(
    all_page_line_level_ocr_results_df: pd.DataFrame,
    output_folder: str,
    model_choice: str,
    in_api_key: str,
    temperature: float,
    file_name: str = "document",
    context_textbox: str = "",
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    aws_region_textbox: str = "",
    hf_api_key_textbox: str = "",
    azure_endpoint_textbox: str = "",
    api_url: str = None,
    summarise_format_radio: str = "Return a summary up to two paragraphs long that includes as much detail as possible from the original text",
    additional_summary_instructions: str = "",
    max_pages_per_group: int = 30,
    summary_page_group_max_workers: Optional[int] = None,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[List[str], str]:
    """
    Main function to summarise a document from OCR results.

    Args:
        all_page_line_level_ocr_results_df (pd.DataFrame): DataFrame containing line-level OCR results.
        output_folder (str): The folder where outputs will be saved.
        model_choice (str): The model to use for summarization.
        in_api_key (str): API key for the selected model/inference method.
        temperature (float): LLM temperature hyperparameter.
        file_name (str, optional): Name to use for the output files. Default is "document".
        context_textbox (str, optional): Extra context for summarization. Default is "".
        aws_access_key_textbox (str, optional): AWS access key, if using AWS. Default is "".
        aws_secret_key_textbox (str, optional): AWS secret key, if using AWS. Default is "".
        aws_region_textbox (str, optional): AWS region string. Default is "".
        hf_api_key_textbox (str, optional): HuggingFace API key, if used. Default is "".
        azure_endpoint_textbox (str, optional): Azure endpoint, if used. Default is "".
        api_url (str, optional): API URL. Default is None.
        summarise_format_radio (str, optional): Summary output format instructions. Default is detailed summary.
        additional_summary_instructions (str, optional): Extra instructions for the summarization. Default is "".
        max_pages_per_group (int, optional): Maximum number of pages to group per LLM pass. Default is 30.
        progress (gr.Progress, optional): Gradio progress tracker. Default is Gradio Progress with tqdm.

    Returns:
        Tuple of (output_file_paths, status_message)
    """
    import os
    from datetime import datetime

    from tools.llm_funcs import load_model

    output_files = []
    all_prompts = []
    all_responses = []
    all_token_counts = (
        []
    )  # Store (input_tokens, output_tokens) for each prompt/response
    page_group_page_ranges = (
        []
    )  # Store (min_page, max_page) for each saved prompt/response
    page_group_summaries = []

    # Initialize token tracking variables
    llm_total_input_tokens = 0
    llm_total_output_tokens = 0
    llm_model_name = ""

    try:
        # Determine model source from model_choice using defaults from config.py
        # Does not check model_name_map - uses the defined defaults
        model_source = get_model_source_from_model_choice(model_choice)

        local_model = None
        tokenizer = None
        assistant_model = None

        # Setup model based on model source - check for Local models
        # Load model and tokenizer together to ensure they're from the same source
        # This prevents mismatches that could occur if they're loaded separately
        # Similar to llm_funcs.py pattern (lines 830-839) and llm_entity_detection.py (lines 519-533)
        if model_source == "Local":
            if local_model is None or tokenizer is None:
                progress(0.05, "Loading local model...")
                # Use load_model() to ensure both are loaded atomically
                # This is safer than calling get_pii_model() and get_pii_tokenizer() separately
                loaded_model, loaded_tokenizer, loaded_assistant_model = load_model()
                if local_model is None:
                    local_model = loaded_model
                if tokenizer is None:
                    tokenizer = loaded_tokenizer
                if assistant_model is None:
                    assistant_model = loaded_assistant_model

        # Step 1: Group pages by context length
        progress(0.1, "Grouping pages by context length...")
        page_groups = group_pages_by_context_length(
            all_page_line_level_ocr_results_df,
            LLM_CONTEXT_LENGTH,
            tokenizer,
            model_source,
            max_pages_per_group=max_pages_per_group,
        )

        if not page_groups:
            return [], "No OCR results found. Please run text extraction first."

        # Step 2: Summarise each page group (optionally in parallel)
        _summary_page_group_max_workers = (
            summary_page_group_max_workers
            if summary_page_group_max_workers is not None
            else SUMMARY_PAGE_GROUP_MAX_WORKERS
        )
        use_parallel_page_groups = (
            _summary_page_group_max_workers > 1 and len(page_groups) > 1
        )
        progress(0.2, f"Summarising {len(page_groups)} page groups...")

        def _summarise_one_group(args):
            i, page_nums, group_text = args
            summary_text, full_prompt, metadata = summarise_text_chunk(
                group_text,
                model_choice,
                in_api_key,
                temperature,
                context_textbox=context_textbox,
                aws_access_key_textbox=aws_access_key_textbox,
                aws_secret_key_textbox=aws_secret_key_textbox,
                aws_region_textbox=aws_region_textbox,
                hf_api_key_textbox=hf_api_key_textbox,
                azure_endpoint_textbox=azure_endpoint_textbox,
                api_url=api_url,
                local_model=local_model,
                tokenizer=tokenizer,
                assistant_model=assistant_model,
                summarise_format_radio=summarise_format_radio,
                additional_summary_instructions=additional_summary_instructions,
            )
            return (i, page_nums, summary_text, full_prompt, metadata)

        if use_parallel_page_groups:
            max_workers = min(_summary_page_group_max_workers, len(page_groups))
            tasks = [
                (i, page_nums, group_text)
                for i, (page_nums, group_text) in enumerate(page_groups)
            ]
            results_by_index = [None] * len(page_groups)
            pbar = tqdm(
                total=len(page_groups),
                unit="groups",
                desc="Summarising page groups",
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_summarise_one_group, t): t[0] for t in tasks
                }
                completed = 0
                for future in as_completed(futures):
                    i, page_nums, summary_text, full_prompt, metadata = future.result()
                    results_by_index[i] = (
                        page_nums,
                        summary_text,
                        full_prompt,
                        metadata,
                    )
                    completed += 1
                    pbar.update(1)
                    progress(
                        0.2 + (completed / len(page_groups)) * 0.5,
                        f"Summarising page group {completed}/{len(page_groups)} (pages {min(page_nums)}-{max(page_nums)})...",
                    )
            pbar.close()
            # Build lists in page-group order
            for i in range(len(page_groups)):
                if results_by_index[i] is None:
                    continue
                page_nums, summary_text, full_prompt, metadata = results_by_index[i]
                if summary_text:
                    try:
                        min_page = int(min(page_nums)) if page_nums else 0
                        max_page = int(max(page_nums)) if page_nums else 0
                    except Exception:
                        min_page, max_page = 0, 0
                    page_group_page_ranges.append((min_page, max_page))
                    page_group_summaries.append(summary_text)
                    all_prompts.append(full_prompt)
                    all_responses.append(summary_text)
                    input_tokens, output_tokens = 0, 0
                    if metadata:
                        metadata_string = (
                            str(metadata) if not isinstance(metadata, str) else metadata
                        )
                        input_tokens, output_tokens, _ = calculate_tokens_from_metadata(
                            metadata_string, model_choice, model_name_map
                        )
                        llm_total_input_tokens += input_tokens
                        llm_total_output_tokens += output_tokens
                        if not llm_model_name and model_choice:
                            llm_model_name = model_choice
                    all_token_counts.append((input_tokens, output_tokens))
        else:
            seq_pbar = tqdm(
                page_groups,
                unit="groups",
                desc="Summarising page groups",
            )
            for i, (page_nums, group_text) in enumerate(seq_pbar):
                progress(
                    0.2 + (i / len(page_groups)) * 0.5,
                    f"Summarising page group {i+1}/{len(page_groups)} (pages {min(page_nums)}-{max(page_nums)})...",
                )
                summary_text, full_prompt, metadata = summarise_text_chunk(
                    group_text,
                    model_choice,
                    in_api_key,
                    temperature,
                    context_textbox=context_textbox,
                    aws_access_key_textbox=aws_access_key_textbox,
                    aws_secret_key_textbox=aws_secret_key_textbox,
                    aws_region_textbox=aws_region_textbox,
                    hf_api_key_textbox=hf_api_key_textbox,
                    azure_endpoint_textbox=azure_endpoint_textbox,
                    api_url=api_url,
                    local_model=local_model,
                    tokenizer=tokenizer,
                    assistant_model=assistant_model,
                    summarise_format_radio=summarise_format_radio,
                    additional_summary_instructions=additional_summary_instructions,
                )
                if summary_text:
                    try:
                        min_page = int(min(page_nums)) if page_nums else 0
                        max_page = int(max(page_nums)) if page_nums else 0
                    except Exception:
                        min_page, max_page = 0, 0
                    page_group_page_ranges.append((min_page, max_page))
                    page_group_summaries.append(summary_text)
                    all_prompts.append(full_prompt)
                    all_responses.append(summary_text)
                    input_tokens, output_tokens = 0, 0
                    if metadata:
                        metadata_string = (
                            str(metadata) if not isinstance(metadata, str) else metadata
                        )
                        input_tokens, output_tokens, _ = calculate_tokens_from_metadata(
                            metadata_string, model_choice, model_name_map
                        )
                        llm_total_input_tokens += input_tokens
                        llm_total_output_tokens += output_tokens
                        if not llm_model_name and model_choice:
                            llm_model_name = model_choice
                    all_token_counts.append((input_tokens, output_tokens))
            seq_pbar.close()

        # Step 3: Recursively summarise if needed
        progress(0.7, "Checking if recursive summarisation is needed...")
        # Create token accumulator for recursive summarization
        recursive_token_accumulator = [0, 0]  # [input_tokens, output_tokens]
        final_summaries = recursively_summarise(
            page_group_summaries,
            model_choice,
            in_api_key,
            temperature,
            context_length=LLM_CONTEXT_LENGTH,
            tokenizer=tokenizer,
            model_source=model_source,
            token_accumulator=recursive_token_accumulator,
            context_textbox=context_textbox,
            aws_access_key_textbox=aws_access_key_textbox,
            aws_secret_key_textbox=aws_secret_key_textbox,
            aws_region_textbox=aws_region_textbox,
            hf_api_key_textbox=hf_api_key_textbox,
            azure_endpoint_textbox=azure_endpoint_textbox,
            api_url=api_url,
            local_model=local_model,
            assistant_model=assistant_model,
            summarise_format_radio=summarise_format_radio,
            additional_summary_instructions=additional_summary_instructions,
        )

        # Add tokens from recursive summarization
        llm_total_input_tokens += recursive_token_accumulator[0]
        llm_total_output_tokens += recursive_token_accumulator[1]

        # Step 4: Create overall summary
        progress(0.85, "Creating overall summary...")
        # Create a topic summary DataFrame for overall_summary: three columns only
        summary_numbers = list(range(1, len(final_summaries) + 1))
        if len(final_summaries) == len(page_groups):
            page_ranges = [f"Pages {min(pg[0])}-{max(pg[0])}" for pg in page_groups]
        else:
            # Recursion combined some summaries - use "All" or full range
            if len(final_summaries) == 1 and page_groups:
                all_pages = [p for pg in page_groups for p in pg[0]]
                page_ranges = [f"Pages {min(all_pages)}-{max(all_pages)}"]
            else:
                page_ranges = ["All"] * len(final_summaries)
        topic_summary_df = pd.DataFrame(
            {
                "Summary number": summary_numbers,
                "Page range": page_ranges,
                "Summary": final_summaries,
            }
        )

        # Call overall_summary
        (
            output_files,
            html_output_table,
            overall_summarised_outputs_df,
            out_metadata_str,
            overall_input_tokens,
            overall_output_tokens,
            number_of_calls_num,
            time_taken,
            out_message,
            overall_logged_content,
            overall_prompt,
            overall_response,
        ) = overall_summary(
            topic_summary_df=topic_summary_df,
            model_choice=model_choice,
            in_api_key=in_api_key,
            temperature=temperature,
            reference_data_file_name=file_name,
            output_folder=output_folder,
            context_textbox=context_textbox,
            aws_access_key_textbox=aws_access_key_textbox,
            aws_secret_key_textbox=aws_secret_key_textbox,
            aws_region_textbox=aws_region_textbox,
            hf_api_key_textbox=hf_api_key_textbox,
            azure_endpoint_textbox=azure_endpoint_textbox,
            api_url=api_url,
            local_model=local_model,
            tokenizer=tokenizer,
            assistant_model=assistant_model,
            summarise_format_radio=summarise_format_radio,
            additional_summary_instructions=additional_summary_instructions,
            progress=progress,
        )

        llm_total_input_tokens += overall_input_tokens
        llm_total_output_tokens += overall_output_tokens

        # Extract summary texts from the DataFrame
        if (
            overall_summarised_outputs_df is not None
            and not overall_summarised_outputs_df.empty
        ):
            if "Summary" in overall_summarised_outputs_df.columns:
                overall_summary_texts = overall_summarised_outputs_df[
                    "Summary"
                ].tolist()
            else:
                # Fallback: get from first column if "Summary" column doesn't exist
                overall_summary_texts = overall_summarised_outputs_df.iloc[
                    :, 0
                ].tolist()
        else:
            overall_summary_texts = []

        # Step 5: Save outputs
        progress(0.95, "Saving output files...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name_clean = get_file_name_no_ext(file_name) if file_name else "document"
        # Ensure file_name_clean is not empty
        if not file_name_clean or file_name_clean.strip() == "":
            file_name_clean = "document"

        summaries_folder = os.path.join(output_folder, "summaries")
        os.makedirs(summaries_folder, exist_ok=True)

        # Save prompts and responses as .txt files for page group summaries
        for i, (prompt, response) in enumerate(zip(all_prompts, all_responses)):
            # Page range for this prompt/response pair
            min_page, max_page = (
                page_group_page_ranges[i] if i < len(page_group_page_ranges) else (0, 0)
            )
            page_range_slug = f"pages_{min_page}_{max_page}"
            txt_file_path = os.path.join(
                summaries_folder,
                f"{file_name_clean}_{page_range_slug}_prompt_response_{timestamp}.txt",
            )
            # Get token counts for this prompt/response pair
            input_tokens, output_tokens = (
                all_token_counts[i] if i < len(all_token_counts) else (0, 0)
            )

            with open(txt_file_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("TOKEN INFORMATION\n")
                f.write("=" * 80 + "\n")
                f.write(f"Page Range: {min_page}-{max_page}\n")
                f.write(f"Input Tokens: {input_tokens}\n")
                f.write(f"Output Tokens: {output_tokens}\n")
                f.write(f"Maximum Context Length: {LLM_CONTEXT_LENGTH}\n")
                f.write(f"Model: {model_choice}\n")
                f.write(f"Temperature: {temperature}\n")
                f.write("=" * 80 + "\n\n")
                f.write("=" * 80 + "\n")
                f.write("PROMPT\n")
                f.write("=" * 80 + "\n")
                f.write(prompt)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("RESPONSE\n")
                f.write("=" * 80 + "\n")
                f.write(response)
            output_files.append(txt_file_path)

        # Save overall summary prompt/response

        # Fallback: If we don't have prompt/response from logged_content, use summary texts
        # This should rarely happen, but provides a safety net
        if not overall_prompt and overall_summary_texts:
            # Construct a basic prompt representation (this is a fallback, not ideal)
            overall_prompt = (
                f"Overall summary request for document: {file_name_clean}\n"
            )
            overall_prompt += f"Input: {len(final_summaries)} summary group(s) to combine into overall summary\n"
            overall_prompt += f"Summary format: {summarise_format_radio}\n"
            if additional_summary_instructions:
                overall_prompt += (
                    f"Additional instructions: {additional_summary_instructions}\n"
                )

        # If we still don't have a response, use summary texts
        if not overall_response and overall_summary_texts:
            overall_response = (
                "\n\n".join(overall_summary_texts)
                if isinstance(overall_summary_texts, list)
                else str(overall_summary_texts)
            )

        # Save overall summary .txt file if we have response content (always create if we have summary texts)
        if overall_response or overall_summary_texts:
            txt_file_path = os.path.join(
                summaries_folder,
                f"{file_name_clean}_overall_summary_prompt_response_{timestamp}.txt",
            )
            with open(txt_file_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("TOKEN INFORMATION\n")
                f.write("=" * 80 + "\n")
                f.write(f"Input Tokens: {overall_input_tokens}\n")
                f.write(f"Output Tokens: {overall_output_tokens}\n")
                f.write(f"Maximum Context Length: {LLM_CONTEXT_LENGTH}\n")
                f.write(f"Model: {model_choice}\n")
                f.write(f"Temperature: {temperature}\n")
                f.write("=" * 80 + "\n\n")
                f.write("=" * 80 + "\n")
                f.write("PROMPT\n")
                f.write("=" * 80 + "\n")
                f.write(overall_prompt)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("RESPONSE\n")
                f.write("=" * 80 + "\n")
                f.write(overall_response)
            output_files.append(txt_file_path)

        # Save summaries as CSV
        summary_data = {"Type": [], "Page_Range": [], "Summary": []}

        # Add page group summaries
        for i, (page_nums, summary) in enumerate(
            zip([pg[0] for pg in page_groups], page_group_summaries)
        ):
            summary_data["Type"].append("Page Group Summary")
            summary_data["Page_Range"].append(f"{min(page_nums)}-{max(page_nums)}")
            summary_data["Summary"].append(summary)

        # Add final summaries if different from page group summaries
        if final_summaries != page_group_summaries:
            for i, summary in enumerate(final_summaries):
                summary_data["Type"].append("Final Summary")
                summary_data["Page_Range"].append(f"Group {i+1}")
                summary_data["Summary"].append(summary)

        # Add overall summary - ensure overall_summary_texts is a list of strings
        if overall_summary_texts:
            # Handle case where overall_summary_texts might be a single string
            if isinstance(overall_summary_texts, str):
                overall_summary_texts = [overall_summary_texts]
            # Ensure each item is a string, not being iterated character by character
            for summary in overall_summary_texts:
                if isinstance(summary, str):
                    summary_data["Type"].append("Overall Summary")
                    summary_data["Page_Range"].append("All")
                    summary_data["Summary"].append(summary)
                elif hasattr(summary, "__iter__") and not isinstance(summary, str):
                    # If it's iterable but not a string, convert to string
                    summary_str = str(summary)
                    summary_data["Type"].append("Overall Summary")
                    summary_data["Page_Range"].append("All")
                    summary_data["Summary"].append(summary_str)

        summary_df = pd.DataFrame(summary_data)
        csv_file_path = os.path.join(
            summaries_folder, f"{file_name_clean}_summaries_{timestamp}.csv"
        )
        summary_df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")
        output_files.append(csv_file_path)

        progress(1.0, "Summarisation complete!")
        status_message = (
            f"Summarisation complete! Generated {len(output_files)} output files."
        )

        # Prepare summary text for display (combine all overall summary texts)
        summary_display_text = ""
        if overall_summary_texts:
            if isinstance(overall_summary_texts, list):
                summary_display_text = "\n\n".join(overall_summary_texts)
            else:
                summary_display_text = str(overall_summary_texts)

        return (
            output_files,
            status_message,
            llm_model_name,
            llm_total_input_tokens,
            llm_total_output_tokens,
            summary_display_text,
        )

    except Exception as e:
        error_message = f"Error during summarisation: {str(e)}"
        print(error_message)
        import traceback

        traceback.print_exc()
        return (
            output_files,
            error_message,
            llm_model_name,
            llm_total_input_tokens,
            llm_total_output_tokens,
            "",  # Empty summary display text on error
        )


def join_unique_summaries(x):
    unique_summaries = []
    seen = set()

    for s in x:
        if pd.isna(s):
            continue

        # 1. Normalize whitespace and split lines
        s_str = str(s).strip()
        lines = s_str.split("\n")

        for line in lines:
            # 2. Aggressive Cleaning
            # Remove "Rows X to Y:" prefix
            line = re.sub(
                r"^Rows\s+\d+\s+to\s+\d+:\s*", "", line, flags=re.IGNORECASE
            ).strip()

            # Remove generic "Prefix:" if it exists (e.g., "Summary: ...")
            if ": " in line:
                parts = line.split(": ", 1)
                if len(parts[0]) < 50 and " " not in parts[0]:
                    line = parts[1].strip()

            # 3. Handle Invisible Characters (Crucial)
            # Replace non-breaking spaces (\xa0) and multiple spaces with a single standard space
            normalized_line = re.sub(r"\s+", " ", line).strip()

            # 4. Check against Seen
            if normalized_line and normalized_line not in seen:
                unique_summaries.append(normalized_line)
                seen.add(normalized_line)

    return "\n".join(unique_summaries)


def sample_reference_table_summaries(
    reference_df: pd.DataFrame,
    random_seed: int,
    no_of_sampled_summaries: int = 100,
    sample_reference_table_checkbox: bool = False,
):
    """
    Sample x number of summaries from which to produce summaries, so that the input token length is not too long.
    """

    if sample_reference_table_checkbox:

        all_summaries = pd.DataFrame(
            columns=[
                "General topic",
                "Subtopic",
                "Sentiment",
                "Group",
                "Response References",
                "Summary",
            ]
        )

        if "Group" not in reference_df.columns:
            reference_df["Group"] = "All"

        reference_df_grouped = reference_df.groupby(
            ["General topic", "Subtopic", "Sentiment", "Group"]
        )

        if "Revised summary" in reference_df.columns:
            out_message = "Summary has already been created for this file"
            print(out_message)
            raise Exception(out_message)

        for group_keys, reference_df_group in reference_df_grouped:
            if len(reference_df_group["General topic"]) > 1:

                filtered_reference_df = reference_df_group.reset_index()

                filtered_reference_df_unique = filtered_reference_df.drop_duplicates(
                    [
                        "General topic",
                        "Subtopic",
                        "Sentiment",
                        "Group",
                        "Start row of group",
                    ]
                )

                # Sample n of the unique topic summaries PER GROUP. To limit the length of the text going into the summarisation tool
                # This ensures each group gets up to no_of_sampled_summaries summaries, not the total across all groups
                number_of_summaries_to_sample = min(
                    no_of_sampled_summaries, len(filtered_reference_df_unique)
                )
                print(
                    f"Sampling {number_of_summaries_to_sample} summaries from group {group_keys}, from dataframe filtered_reference_df_unique.head(5):\n{filtered_reference_df_unique.head(5)}"
                )
                filtered_reference_df_unique_sampled = (
                    filtered_reference_df_unique.sample(
                        number_of_summaries_to_sample, random_state=random_seed
                    )
                )

                all_summaries = pd.concat(
                    [all_summaries, filtered_reference_df_unique_sampled]
                )

                print("all_summaries.tail(5):\n", all_summaries.tail(5))

        # If no responses/topics qualify, just go ahead with the original reference dataframe
        if all_summaries.empty:
            sampled_reference_table_df = reference_df
            # Filter by sentiment only (Response References is a string in original df, not a count)
            sampled_reference_table_df = sampled_reference_table_df.loc[
                sampled_reference_table_df["Sentiment"] != "Not Mentioned"
            ]
        else:
            # Deduplicate summaries within each group before joining to prevent repeated summaries

            sampled_reference_table_df = (
                all_summaries.groupby(
                    ["General topic", "Subtopic", "Sentiment", "Group"]
                )
                .agg(
                    {
                        "Response References": "size",  # Count the number of references
                        "Summary": join_unique_summaries,  # Join unique summaries only
                    }
                )
                .reset_index()
            )
            # Filter by sentiment and count (Response References is now a numeric count after aggregation)
            sampled_reference_table_df = sampled_reference_table_df.loc[
                (sampled_reference_table_df["Sentiment"] != "Not Mentioned")
                & (sampled_reference_table_df["Response References"] > 1)
            ]
    else:
        sampled_reference_table_df = reference_df

    summarised_references_markdown = sampled_reference_table_df.to_markdown(index=False)

    return sampled_reference_table_df, summarised_references_markdown


def count_tokens_in_text(text: str, tokenizer=None, model_source: str = "Local") -> int:
    """
    Count the number of tokens in the given text.

    Args:
        text (str): The text to count tokens for
        tokenizer (object, optional): Tokenizer object for local models. Defaults to None.
        model_source (str): Source of the model to determine tokenization method. Defaults to "Local".

    Returns:
        int: Number of tokens in the text
    """
    if not text:
        return 0

    try:
        if model_source == "Local" and tokenizer and len(tokenizer) > 0:
            # Use local tokenizer if available
            tokens = tokenizer[0].encode(text, add_special_tokens=False)
            return len(tokens)
        else:
            # Fallback: rough estimation using word count (approximately 1.3 tokens per word)
            word_count = len(text.split())
            return int(word_count * 1.3)
    except Exception as e:
        print(f"Error counting tokens: {e}. Using word count estimation.")
        # Fallback: rough estimation using word count
        word_count = len(text.split())
        return int(word_count * 1.3)


def clean_markdown_table_whitespace(markdown_text: str) -> str:
    if not markdown_text:
        return markdown_text

    lines = markdown_text.splitlines()
    cleaned_lines = []

    for line in lines:
        # 1. Clean all types of whitespace (including non-breaking spaces \u00A0)
        # This turns every cell into a single-spaced string
        cells = [re.sub(r"[\s\u00A0]+", " ", cell.strip()) for cell in line.split("|")]

        # 2. Check if the row is effectively empty (only pipes or whitespace)
        # We join the content; if nothing is left, it's a "ghost" row.
        if not "".join(cells).strip():
            continue

        # 3. Handle the separator row specifically (e.g., |:---|---:|)
        # We reset these to a small fixed width so they don't stretch the table.
        if re.match(r"^[|\s\-:]+$", line):
            new_separator = []
            for cell in cells:
                if not cell:  # Outer pipes
                    new_separator.append("")
                elif ":" in cell:  # Alignment markers
                    left = ":" if cell.startswith(":") else "-"
                    right = ":" if cell.endswith(":") else "-"
                    new_separator.append(f"{left}---{right}")
                else:
                    new_separator.append("---")
            cleaned_lines.append("|".join(new_separator))
            continue

        # 4. Standard data row: Rejoin with single padding
        # We filter out empty outer parts caused by leading/trailing pipes
        formatted_row = (
            "| "
            + " | ".join(
                c for c in cells if c or cells.index(c) not in [0, len(cells) - 1]
            )
            + " |"
        )

        # Simple fallback if the logic above is too aggressive for your specific table style:
        # formatted_row = "|".join(f" {c} " if c else "" for c in cells)

        cleaned_lines.append(formatted_row)

    return "\n".join(cleaned_lines)


def summarise_output_topics_query(
    model_choice: str,
    in_api_key: str,
    temperature: float,
    formatted_summary_prompt: str,
    summarise_topic_descriptions_system_prompt: str,
    model_source: str,
    bedrock_runtime: boto3.Session.client,
    local_model=list(),
    tokenizer=list(),
    assistant_model=list(),
    azure_endpoint: str = "",
    api_url: str = None,
):
    """
    Query an LLM to generate a summary of topics based on the provided prompts.

    Args:
        model_choice (str): The name/type of model to use for generation
        in_api_key (str): API key for accessing the model service
        temperature (float): Temperature parameter for controlling randomness in generation
        formatted_summary_prompt (str): The formatted prompt containing topics to summarise
        summarise_topic_descriptions_system_prompt (str): System prompt providing context and instructions
        model_source (str): Source of the model (e.g. "AWS", "Gemini", "Local")
        bedrock_runtime (boto3.Session.client): AWS Bedrock runtime client for AWS models
        local_model (object, optional): Local model object if using local inference. Defaults to empty list.
        tokenizer (object, optional): Tokenizer object if using local inference. Defaults to empty list.
    Returns:
        tuple: Contains:
            - response_text (str): The generated summary text
            - conversation_history (list): History of the conversation with the model
            - whole_conversation_metadata (list): Metadata about the conversation
    """
    conversation_history = list()
    whole_conversation_metadata = list()
    client = list()
    client_config = {}

    # Combine system prompt and user prompt for token counting
    full_input_text = (
        summarise_topic_descriptions_system_prompt + "\n" + formatted_summary_prompt[0]
        if isinstance(formatted_summary_prompt, list)
        else summarise_topic_descriptions_system_prompt
        + "\n"
        + formatted_summary_prompt
    )

    # Count tokens in the input text
    input_token_count = count_tokens_in_text(full_input_text, tokenizer, model_source)

    # Check if input exceeds context length
    if input_token_count > LLM_CONTEXT_LENGTH:
        error_message = f"Input text exceeds LLM context length. Input tokens: {input_token_count}, Max context length: {LLM_CONTEXT_LENGTH}. Please reduce the input text size."
        print(error_message)
        raise ValueError(error_message)

    print(f"Input token count: {input_token_count} (Max: {LLM_CONTEXT_LENGTH})")

    # Prepare Gemini models before query
    if "Gemini" in model_source:
        # print("Using Gemini model:", model_choice)
        client, config = construct_gemini_generative_model(
            in_api_key=in_api_key,
            temperature=temperature,
            model_choice=model_choice,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
    elif "Azure/OpenAI" in model_source:
        client, config = construct_azure_client(
            in_api_key=os.environ.get("AZURE_INFERENCE_CREDENTIAL", ""),
            endpoint=azure_endpoint,
        )
    elif "Local" in model_source:
        pass
        # print("Using local model: ", model_choice)
    elif "AWS" in model_source:
        pass
        # print("Using AWS Bedrock model:", model_choice)

    whole_conversation = [summarise_topic_descriptions_system_prompt]

    # Process requests to large language model
    (
        responses,
        conversation_history,
        whole_conversation,
        whole_conversation_metadata,
        response_text,
    ) = process_requests(
        formatted_summary_prompt,
        summarise_topic_descriptions_system_prompt,
        conversation_history,
        whole_conversation,
        whole_conversation_metadata,
        client,
        client_config,
        model_choice,
        temperature,
        bedrock_runtime=bedrock_runtime,
        model_source=model_source,
        local_model=local_model,
        tokenizer=tokenizer,
        assistant_model=assistant_model,
        assistant_prefill=summary_assistant_prefill,
        api_url=api_url,
    )

    summarised_output = re.sub(
        r"\n{2,}", "\n", response_text
    )  # Replace multiple line breaks with a single line break
    summarised_output = re.sub(
        r"^\n{1,}", "", summarised_output
    )  # Remove one or more line breaks at the start
    summarised_output = re.sub(
        r"\n", "<br>", summarised_output
    )  # Replace \n with more html friendly <br> tags
    summarised_output = summarised_output.strip()

    print("Finished summary query")

    # Ensure the system prompt is included in the conversation history
    try:
        if isinstance(conversation_history, list):
            has_system_prompt = False

            if conversation_history:
                first_entry = conversation_history[0]
                if isinstance(first_entry, dict):
                    role_is_system = first_entry.get("role") == "system"
                    parts = first_entry.get("parts")
                    content_matches = (
                        parts == summarise_topic_descriptions_system_prompt
                        or (
                            isinstance(parts, list)
                            and summarise_topic_descriptions_system_prompt in parts
                        )
                    )
                    has_system_prompt = role_is_system and content_matches
                elif isinstance(first_entry, str):
                    has_system_prompt = (
                        first_entry.strip().lower().startswith("system:")
                    )

            if not has_system_prompt:
                conversation_history.insert(
                    0,
                    {
                        "role": "system",
                        "parts": [summarise_topic_descriptions_system_prompt],
                    },
                )
    except Exception as _e:
        # Non-fatal: if anything goes wrong, return the original conversation history
        pass

    return (
        summarised_output,
        conversation_history,
        whole_conversation_metadata,
        response_text,
    )


def process_debug_output_iteration(
    output_debug_files: str,
    summaries_folder: str,
    batch_file_path_details: str,
    model_choice_clean_short: str,
    final_system_prompt: str,
    summarised_output: str,
    conversation_history: list,
    metadata: list,
    log_output_files: list,
    task_type: str,
) -> tuple[str, str, str, str]:
    """
    Writes debug files for summary generation if output_debug_files is "True",
    and returns the content of the prompt, summary, conversation, and metadata for the current iteration.

    Args:
        output_debug_files (str): Flag to indicate if debug files should be written.
        summaries_folder (str): The folder where output files are saved.
        batch_file_path_details (str): Details for the batch file path.
        model_choice_clean_short (str): Shortened cleaned model choice.
        final_system_prompt (str): The system prompt content.
        summarised_output (str): The summarised output content.
        conversation_history (list): The full conversation history.
        metadata (list): The metadata for the conversation.
        log_output_files (list): A list to append paths of written log files. This list is modified in-place.
        task_type (str): The type of task being performed.
    Returns:
        tuple[str, str, str, str]: A tuple containing the content of the prompt,
                                    summarised output, conversation history (as string),
                                    and metadata (as string) for the current iteration.
    """
    current_prompt_content = final_system_prompt
    current_summary_content = summarised_output

    if isinstance(conversation_history, list):

        # Handle both list of strings and list of dicts
        if conversation_history and isinstance(conversation_history[0], dict):
            # Convert list of dicts to list of strings
            conversation_strings = list()
            for entry in conversation_history:
                if "role" in entry and "parts" in entry:
                    role = entry["role"].capitalize()
                    message = (
                        " ".join(entry["parts"])
                        if isinstance(entry["parts"], list)
                        else str(entry["parts"])
                    )
                    conversation_strings.append(f"{role}: {message}")
                else:
                    # Fallback for unexpected dict format
                    conversation_strings.append(str(entry))
            current_conversation_content = "\n".join(conversation_strings)
        else:
            # Handle list of strings
            current_conversation_content = "\n".join(conversation_history)
    else:
        current_conversation_content = str(conversation_history)
    current_metadata_content = str(metadata)
    current_task_type = task_type

    if output_debug_files == "True":
        try:
            formatted_prompt_output_path = (
                summaries_folder
                + batch_file_path_details
                + "_full_prompt_"
                + model_choice_clean_short
                + "_"
                + current_task_type
                + ".txt"
            )
            final_table_output_path = (
                summaries_folder
                + batch_file_path_details
                + "_full_response_"
                + model_choice_clean_short
                + "_"
                + current_task_type
                + ".txt"
            )
            whole_conversation_path = (
                summaries_folder
                + batch_file_path_details
                + "_full_conversation_"
                + model_choice_clean_short
                + "_"
                + current_task_type
                + ".txt"
            )
            whole_conversation_path_meta = (
                summaries_folder
                + batch_file_path_details
                + "_metadata_"
                + model_choice_clean_short
                + "_"
                + current_task_type
                + ".txt"
            )

            with open(
                formatted_prompt_output_path,
                "w",
                encoding="utf-8-sig",
                errors="replace",
            ) as f:
                f.write(current_prompt_content)
            with open(
                final_table_output_path, "w", encoding="utf-8-sig", errors="replace"
            ) as f:
                f.write(current_summary_content)
            with open(
                whole_conversation_path, "w", encoding="utf-8-sig", errors="replace"
            ) as f:
                f.write(current_conversation_content)
            with open(
                whole_conversation_path_meta,
                "w",
                encoding="utf-8-sig",
                errors="replace",
            ) as f:
                f.write(current_metadata_content)

            log_output_files.append(formatted_prompt_output_path)
            log_output_files.append(final_table_output_path)
            log_output_files.append(whole_conversation_path)
            log_output_files.append(whole_conversation_path_meta)
        except Exception as e:
            print(f"Error in writing debug files for summary: {e}")

    # Return the content of the objects for the current iteration.
    # The caller can then append these to separate lists if accumulation is desired.
    return (
        current_prompt_content,
        current_summary_content,
        current_conversation_content,
        current_metadata_content,
    )


def convert_markdown_headers_to_excel_format(text: str) -> str:
    """
    Convert markdown headers to Excel-friendly format that preserves hierarchy.

    Converts:
    - # Header (H1) -> === HEADER === (most prominent)
    - ## Header (H2) -> --- Header --- (medium)
    - ### Header (H3) -> ── Header ── (less prominent)
    - #### Header (H4) -> • Header (with bullet)
    - ##### Header (H5) ->   • Header (indented)
    - ###### Header (H6) ->     • Header (more indented)

    Args:
        text (str): Text containing markdown headers

    Returns:
        str: Text with markdown headers converted to Excel-friendly format
    """
    if not text:
        return text

    lines = text.split("\n")
    converted_lines = []

    for line in lines:
        # Match markdown headers (# through ######)
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if header_match:
            header_level = len(header_match.group(1))  # Number of # characters
            header_text = header_match.group(2).strip()

            if header_level == 1:
                # H1: Most prominent - uppercase with double equals
                converted_line = f"=== {header_text.upper()} ==="
            elif header_level == 2:
                # H2: Medium prominence - title case with dashes
                converted_line = f"--- {header_text.title()} ---"
            elif header_level == 3:
                # H3: Less prominent - title case with single dashes
                converted_line = f"── {header_text.title()} ──"
            elif header_level == 4:
                # H4: Bullet with no indentation
                converted_line = f"• {header_text}"
            elif header_level == 5:
                # H5: Bullet with indentation
                converted_line = f"  • {header_text}"
            else:  # header_level == 6
                # H6: Bullet with more indentation
                converted_line = f"    • {header_text}"

            converted_lines.append(converted_line)
        else:
            converted_lines.append(line)

    return "\n".join(converted_lines)


@spaces.GPU(duration=MAX_SPACES_GPU_RUN_TIME)
def overall_summary(
    topic_summary_df: pd.DataFrame,
    model_choice: str,
    in_api_key: str,
    temperature: float,
    reference_data_file_name: str,
    output_folder: str = OUTPUT_FOLDER,
    context_textbox: str = "",
    aws_access_key_textbox: str = "",
    aws_secret_key_textbox: str = "",
    aws_region_textbox: str = "",
    model_name_map: dict = model_name_map,
    hf_api_key_textbox: str = "",
    azure_endpoint_textbox: str = "",
    existing_logged_content: list = list(),
    api_url: str = None,
    output_debug_files: str = "False",
    log_output_files: list = list(),
    reasoning_suffix: str = reasoning_suffix,
    local_model: object = None,
    tokenizer: object = None,
    assistant_model: object = None,
    summarise_everything_prompt: str = summarise_everything_prompt,
    summarise_everything_system_prompt: str = summarise_everything_system_prompt,
    summarise_format_radio: str = detailed_summary_format_prompt,
    additional_summary_instructions: str = "",
    do_summaries: str = "Yes",
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[
    List[str],
    List[str],
    int,
    str,
    List[str],
    List[str],
    int,
    int,
    int,
    float,
    List[dict],
]:
    """
    Create an overall summary of all responses based on a topic summary table.

    Args:
        topic_summary_df (pd.DataFrame): DataFrame with columns "Summary number", "Page range", "Summary"
        model_choice (str): Name of the LLM model to use
        in_api_key (str): API key for model access
        temperature (float): Temperature parameter for model generation
        reference_data_file_name (str): Name of reference data file
        output_folder (str, optional): Folder to save outputs. Defaults to OUTPUT_FOLDER.
        context_textbox (str, optional): Additional context. Defaults to empty string.
        aws_access_key_textbox (str, optional): AWS access key. Defaults to empty string.
        aws_secret_key_textbox (str, optional): AWS secret key. Defaults to empty string.
        aws_region_textbox (str, optional): AWS region. Defaults to empty string.
        model_name_map (dict, optional): Mapping of model names. Defaults to model_name_map.
        hf_api_key_textbox (str, optional): Hugging Face API key. Defaults to empty string.
        existing_logged_content (list, optional): List of existing logged content. Defaults to empty list.
        output_debug_files (str, optional): Flag to indicate if debug files should be written. Defaults to "False".
        log_output_files (list, optional): List of existing logged content. Defaults to empty list.
        api_url (str, optional): API URL for inference-server models. Defaults to None.
        reasoning_suffix (str, optional): Suffix for reasoning. Defaults to reasoning_suffix.
        local_model (object, optional): Local model object. Defaults to None.
        tokenizer (object, optional): Tokenizer object. Defaults to None.
        assistant_model (object, optional): Assistant model object. Defaults to None.
        summarise_everything_prompt (str, optional): Prompt for overall summary
        summarise_everything_system_prompt (str, optional): System prompt for overall summary
        summarise_format_radio (str, optional): Summary format radio. Defaults to summarise_format_radio.
        additional_summary_instructions (str, optional): Additional summary instructions. Defaults to additional_summary_instructions.
        do_summaries (str, optional): Whether to generate summaries. Defaults to "Yes".
        progress (gr.Progress, optional): Progress tracker. Defaults to gr.Progress(track_tqdm=True).

    Returns:
        Tuple containing:
            List[str]: Output files
            List[str]: Text summarised outputs
            int: Latest summary completed
            str: Output metadata
            List[str]: Summarised outputs
            List[str]: Summarised outputs for DataFrame
            int: Number of input tokens
            int: Number of output tokens
            int: Number of API calls
            float: Time taken
            List[dict]: List of logged content
    """

    out_metadata = list()
    latest_summary_completed = 0
    output_files = list()
    txt_summarised_outputs = list()
    summarised_outputs = list()
    summarised_outputs_for_df = list()
    input_tokens_num = 0
    output_tokens_num = 0
    number_of_calls_num = 0
    time_taken = 0
    out_message = list()
    all_logged_content = list()
    all_prompts_content = list()
    all_summaries_content = list()
    all_metadata_content = list()
    all_groups_content = list()
    all_batches_content = list()
    all_model_choice_content = list()
    all_validated_content = list()
    task_type = "Overall summary"
    all_task_type_content = list()
    log_output_files = list()
    all_logged_content = list()
    all_file_names_content = list()
    tic = time.perf_counter()

    summaries_folder = os.path.join(output_folder, "summaries")
    os.makedirs(summaries_folder, exist_ok=True)

    # Expect three columns: Summary number, Page range, Summary
    required_cols = ["Summary number", "Page range", "Summary"]
    if not all(c in topic_summary_df.columns for c in required_cols):
        raise ValueError(
            "topic_summary_df must have columns: Summary number, Page range, Summary"
        )
    topic_summary_df = topic_summary_df[required_cols].copy()
    topic_summary_df = topic_summary_df.sort_values(by="Summary number", ascending=True)

    # Single "group" containing the whole table (no grouping by Group column)
    unique_groups = ["All"]

    len(unique_groups)

    if context_textbox and "The context of this analysis is" not in context_textbox:
        context_textbox = "The context of this analysis is '" + context_textbox + "'."

    # if length_groups > 1:
    #     comprehensive_summary_format_prompt = (
    #         comprehensive_summary_format_prompt_by_group
    #     )
    # else:
    #     comprehensive_summary_format_prompt = comprehensive_summary_format_prompt

    batch_file_path_details = create_batch_file_path_details(reference_data_file_name)
    # Use model_choice directly as short_name, or try to get from model_name_map if available
    if model_name_map and model_choice in model_name_map:
        model_choice_clean = model_name_map[model_choice]["short_name"]
    else:
        # Use model_choice directly if not in model_name_map
        model_choice_clean = model_choice
    model_choice_clean_short = clean_column_name(
        model_choice_clean, max_length=20, front_characters=False
    )

    tic = time.perf_counter()

    # Determine model source from model_choice using defaults from config.py
    # Does not check model_name_map - uses the defined defaults
    model_source = get_model_source_from_model_choice(model_choice)

    # Load model and tokenizer together to ensure they're from the same source
    # This prevents mismatches that could occur if they're loaded separately
    # Similar to llm_funcs.py pattern (lines 830-839) and llm_entity_detection.py (lines 519-533)
    if (model_source == "Local") & (local_model is None or tokenizer is None):
        progress(0.1, f"Using model: {LOCAL_TRANSFORMERS_LLM_PII_MODEL_CHOICE}")
        # Use load_model() to ensure both are loaded atomically
        # This is safer than calling get_pii_model() and get_pii_tokenizer() separately
        loaded_model, loaded_tokenizer, loaded_assistant_model = load_model()
        if local_model is None:
            local_model = loaded_model
        if tokenizer is None:
            tokenizer = loaded_tokenizer
        if assistant_model is None:
            assistant_model = loaded_assistant_model

    summary_loop = tqdm(
        unique_groups, desc="Creating overall summary for groups", unit="groups"
    )

    if do_summaries == "Yes":
        # Determine model source from model_choice using defaults from config.py
        # Does not check model_name_map - uses the defined defaults
        model_source = get_model_source_from_model_choice(model_choice)

        # Setup bedrock for AWS models only
        # Use the same approach as file_redaction.py (lines 939-969) for consistency
        bedrock_runtime = None
        if model_source == "AWS":
            # Use aws_region_textbox if provided, otherwise fall back to AWS_REGION from config
            region = aws_region_textbox if aws_region_textbox else AWS_REGION

            if RUN_AWS_FUNCTIONS and PRIORITISE_SSO_OVER_AWS_ENV_ACCESS_KEYS:
                print("Connecting to Bedrock via existing SSO connection")
                bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
            elif aws_access_key_textbox and aws_secret_key_textbox:
                print(
                    "Connecting to Bedrock using AWS access key and secret keys from user input."
                )
                bedrock_runtime = boto3.client(
                    "bedrock-runtime",
                    aws_access_key_id=aws_access_key_textbox,
                    aws_secret_access_key=aws_secret_key_textbox,
                    region_name=region,
                )
            elif RUN_AWS_FUNCTIONS:
                print("Connecting to Bedrock via existing SSO connection")
                bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
            elif AWS_ACCESS_KEY and AWS_SECRET_KEY:
                print("Getting Bedrock credentials from environment variables")
                bedrock_runtime = boto3.client(
                    "bedrock-runtime",
                    aws_access_key_id=AWS_ACCESS_KEY,
                    aws_secret_access_key=AWS_SECRET_KEY,
                    region_name=region,
                )
            else:
                bedrock_runtime = None
                out_message = "Cannot connect to AWS Bedrock service. Please provide access keys under LLM settings, or choose another model type."
                print(out_message)
                raise Exception(out_message)

        for summary_group in summary_loop:

            print("Creating overall summary for group:", summary_group)

            # Use the full table (three columns: Summary number, Page range, Summary)
            group_df = topic_summary_df.copy()

            # Prepare the system prompt first (needed for token counting)
            formatted_summarise_everything_system_prompt = (
                summarise_everything_system_prompt.format(
                    consultation_context=context_textbox
                )
            )

            # Apply reasoning suffix for GPT-OSS models (Local, inference-server, or AWS)
            is_gpt_oss_model = (
                "gpt-oss" in model_choice.lower() or "gpt_oss" in model_choice.lower()
            )

            if is_gpt_oss_model:
                # Use default reasoning suffix if not set
                effective_reasoning_suffix = (
                    reasoning_suffix if reasoning_suffix else "Reasoning: low"
                )
                if effective_reasoning_suffix:
                    formatted_summarise_everything_system_prompt = (
                        formatted_summarise_everything_system_prompt
                        + "\n"
                        + effective_reasoning_suffix
                    )
            elif "Local" in model_source and reasoning_suffix:
                # For other local models, use reasoning_suffix if provided
                formatted_summarise_everything_system_prompt = (
                    formatted_summarise_everything_system_prompt
                    + "\n"
                    + reasoning_suffix
                )

            if additional_summary_instructions:
                additional_summary_instructions = (
                    "Important additional instructions to follow closely: "
                    + additional_summary_instructions
                )

            # Create a test prompt with empty table to get base token count
            test_summary_text = ""
            test_formatted_summary_prompt = [
                summarise_everything_prompt.format(
                    topic_summary_table=test_summary_text,
                    summary_format=summarise_format_radio,
                    additional_summary_instructions=additional_summary_instructions,
                )
            ]

            # Calculate base token count (system prompt + prompt template without table)
            full_test_text = (
                formatted_summarise_everything_system_prompt
                + "\n"
                + test_formatted_summary_prompt[0]
            )
            base_token_count = count_tokens_in_text(
                full_test_text, tokenizer, model_source
            )

            # Calculate available tokens for the summary table
            available_tokens = LLM_CONTEXT_LENGTH - base_token_count

            # Ensure markdown table rows don't get visually "split" by newlines inside cells.
            # Markdown tables don't reliably support multiline cells, so we replace internal
            # newlines with a single-line representation before calling `to_markdown()`.
            def _escape_markdown_table_cell(value):
                if not isinstance(value, str):
                    return value
                s = value.replace("\r\n", "\n").replace("\r", "\n")
                # Keep content in a single cell/row in markdown output
                s = s.replace("\n", "\\n")
                # Avoid breaking markdown table syntax
                s = s.replace("|", "\\|")
                return s

            if "Summary" in group_df.columns:
                group_df["Summary"] = group_df["Summary"].apply(
                    _escape_markdown_table_cell
                )

            # Truncate DataFrame rows if needed to fit within context limit
            if len(group_df) > 0:
                # Start with all rows and check if they fit
                current_summary_text = group_df.to_markdown(index=False)
                current_summary_text = clean_markdown_table_whitespace(
                    current_summary_text
                )
                current_token_count = count_tokens_in_text(
                    current_summary_text, tokenizer, model_source
                )

                # If the full table exceeds available tokens, truncate rows
                if current_token_count > available_tokens:
                    print(
                        f"Warning: Summary table for group '{summary_group}' exceeds context limit. "
                        f"Truncating rows. Table tokens: {current_token_count}, Available: {available_tokens}"
                    )

                    # Binary search approach: find the maximum number of rows that fit
                    # Start with all rows and reduce until we fit
                    num_rows = len(group_df)
                    min_rows = 0
                    max_rows = num_rows
                    best_df = group_df.iloc[:0]  # Empty DataFrame as fallback

                    # Try to find the maximum number of rows that fit
                    while min_rows < max_rows:
                        mid_rows = (min_rows + max_rows + 1) // 2
                        test_df = group_df.iloc[:mid_rows]
                        test_summary = test_df.to_markdown(index=False)
                        test_summary = clean_markdown_table_whitespace(test_summary)
                        test_token_count = count_tokens_in_text(
                            test_summary, tokenizer, model_source
                        )

                        if test_token_count <= available_tokens:
                            best_df = test_df
                            min_rows = mid_rows
                        else:
                            max_rows = mid_rows - 1

                    # Use the best fitting DataFrame
                    group_df = best_df
                    print(
                        f"Truncated to {len(group_df)} rows (from {num_rows} original rows) "
                        f"to fit within context limit."
                    )

            # Create summary_text from (possibly truncated) DataFrame
            summary_text = group_df.to_markdown(index=False)
            # Clean extraneous whitespace from markdown table cells
            summary_text = clean_markdown_table_whitespace(summary_text)

            formatted_summary_prompt = [
                summarise_everything_prompt.format(
                    topic_summary_table=summary_text,
                    summary_format=summarise_format_radio,
                    additional_summary_instructions=additional_summary_instructions,
                )
            ]

            combined_prompt = (
                formatted_summarise_everything_system_prompt
                + "\n"
                + formatted_summary_prompt[0]
            )

            try:
                response, conversation_history, metadata, response_text = (
                    summarise_output_topics_query(
                        model_choice,
                        in_api_key,
                        temperature,
                        formatted_summary_prompt,
                        formatted_summarise_everything_system_prompt,
                        model_source,
                        bedrock_runtime,
                        local_model,
                        tokenizer=tokenizer,
                        assistant_model=assistant_model,
                        azure_endpoint=azure_endpoint_textbox,
                        api_url=api_url,
                    )
                )
                summarised_output_for_df = response_text
                summarised_output = response
            except Exception as e:
                print(
                    "Cannot create overall summary for group:",
                    summary_group,
                    "due to:",
                    e,
                )
                summarised_output = ""
                summarised_output_for_df = ""

            # Remove multiple consecutive line breaks (2 or more) and replace with single line break
            if summarised_output_for_df:
                summarised_output_for_df = re.sub(
                    r"\n{2,}", "\n", summarised_output_for_df
                )
                # Convert markdown headers to Excel-friendly format
                summarised_output_for_df = convert_markdown_headers_to_excel_format(
                    summarised_output_for_df
                )
            if summarised_output:
                summarised_output = re.sub(r"\n{2,}", "\n", summarised_output)

            summarised_outputs_for_df.append(summarised_output_for_df)
            summarised_outputs.append(summarised_output)
            txt_summarised_outputs.append(
                f"""Group name: {summary_group}\n""" + summarised_output
            )

            out_metadata.extend(metadata)
            out_metadata_str = ". ".join(out_metadata)

            full_prompt = (
                formatted_summarise_everything_system_prompt
                + "\n"
                + formatted_summary_prompt[0]
            )

            (
                current_prompt_content_logged,
                current_summary_content_logged,
                current_conversation_content_logged,
                current_metadata_content_logged,
            ) = process_debug_output_iteration(
                output_debug_files,
                summaries_folder,
                batch_file_path_details,
                model_choice_clean_short,
                full_prompt,
                summarised_output,
                conversation_history,
                metadata,
                log_output_files,
                task_type=task_type,
            )

            all_prompts_content.append(current_prompt_content_logged)
            all_summaries_content.append(current_summary_content_logged)
            # all_conversation_content.append(current_conversation_content_logged)
            all_metadata_content.append(current_metadata_content_logged)
            all_groups_content.append(summary_group)
            all_batches_content.append("1")
            all_model_choice_content.append(model_choice_clean_short)
            all_validated_content.append("No")
            all_task_type_content.append(task_type)
            all_file_names_content.append(reference_data_file_name)
            latest_summary_completed += 1
            clean_column_name(summary_group)

        # Write overall outputs to csv
        overall_summary_output_csv_path = (
            output_folder
            + "summaries/"
            + batch_file_path_details
            + "_overall_summary_"
            + model_choice_clean_short
            + ".csv"
        )
        summarised_outputs_df = pd.DataFrame(
            data={"Group": unique_groups, "Summary": summarised_outputs_for_df}
        )
        if output_debug_files == "True":
            summarised_outputs_df.drop(["1", "2", "3"], axis=1, errors="ignore").to_csv(
                overall_summary_output_csv_path, index=None, encoding="utf-8-sig"
            )
            output_files.append(overall_summary_output_csv_path)

        summarised_outputs_df_for_display = pd.DataFrame(
            data={"Group": unique_groups, "Summary": summarised_outputs}
        )
        summarised_outputs_df_for_display["Summary"] = (
            summarised_outputs_df_for_display["Summary"]
            .apply(lambda x: markdown.markdown(x) if isinstance(x, str) else x)
            .str.replace(r"\n", "<br>", regex=False)
            .str.replace(r"(<br>\s*){2,}", "<br>", regex=True)
        )
        html_output_table = summarised_outputs_df_for_display.to_html(
            index=False, escape=False
        )

        output_files = list(set(output_files))

        input_tokens_num, output_tokens_num, number_of_calls_num = (
            calculate_tokens_from_metadata(
                out_metadata_str, model_choice, model_name_map
            )
        )

        # Check if beyond max time allowed for processing and break if necessary
        toc = time.perf_counter()
        time_taken = toc - tic

        out_message = "\n".join(out_message)
        out_message = (
            out_message
            + " "
            + f"Overall summary finished processing. Total time: {time_taken:.2f}s"
        )
        print(out_message)

        # Combine the logged content into a list of dictionaries
        all_logged_content = [
            {
                "prompt": prompt,
                "response": summary,
                "metadata": metadata,
                "batch": batch,
                "model_choice": model_choice,
                "validated": validated,
                "group": group,
                "task_type": task_type,
                "file_name": file_name,
            }
            for prompt, summary, metadata, batch, model_choice, validated, group, task_type, file_name in zip(
                all_prompts_content,
                all_summaries_content,
                all_metadata_content,
                all_batches_content,
                all_model_choice_content,
                all_validated_content,
                all_groups_content,
                all_task_type_content,
                all_file_names_content,
            )
        ]

        if isinstance(existing_logged_content, pd.DataFrame):
            existing_logged_content = existing_logged_content.to_dict(orient="records")

        out_logged_content = existing_logged_content + all_logged_content

    return (
        output_files,
        html_output_table,
        summarised_outputs_df,
        out_metadata_str,
        input_tokens_num,
        output_tokens_num,
        number_of_calls_num,
        time_taken,
        out_message,
        out_logged_content,
        combined_prompt,
        response_text,
    )
