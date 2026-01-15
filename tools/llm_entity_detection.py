"""
LLM-based entity detection using AWS Bedrock.
This module provides functions to detect PII entities using LLMs instead of AWS Comprehend.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import boto3

from tools.config import (
    CHOSEN_LLM_PII_INFERENCE_METHOD,
    CLOUD_LLM_PII_MODEL_CHOICE,
    INFERENCE_SERVER_API_URL,
    LLM_PII_MAX_TOKENS,
    LLM_PII_NUMBER_OF_RETRY_ATTEMPTS,
    LLM_PII_TEMPERATURE,
    LLM_PII_TIMEOUT_WAIT,
    model_name_map,
)
from tools.llm_entity_detection_prompts import (
    create_entity_detection_prompt,
    create_entity_detection_system_prompt,
)

# Import LLM functions from local tools.llm_funcs
try:
    from tools.llm_funcs import (
        ResponseObject,
        call_aws_bedrock,
        construct_azure_client,
    )
except ImportError as e:
    print(f"Warning: Could not import LLM functions: {e}")
    print("LLM-based entity detection will not be available.")
    print("Please ensure llm_funcs.py is in the tools folder.")
    call_aws_bedrock = None
    construct_azure_client = None
    ResponseObject = None


def _find_text_in_passage(
    search_text: str, original_text: str, reported_offset: Optional[int] = None
) -> Optional[Tuple[int, int]]:
    """
    Find the position of search_text in original_text and return (begin, end) offsets.

    Args:
        search_text: The text to search for
        original_text: The text to search in
        reported_offset: Optional offset reported by LLM (used to disambiguate multiple matches)

    Returns:
        Tuple of (begin_offset, end_offset) if found, None otherwise
    """
    if not search_text:
        return None

    # Clean search text - remove trailing ellipsis that LLM might add
    search_text_clean = search_text.rstrip("...").strip()

    # Find all occurrences of the exact text
    all_positions = []
    start = 0
    while True:
        pos = original_text.find(search_text, start)
        if pos == -1:
            break
        all_positions.append(pos)
        start = pos + 1

    if all_positions:
        # If we have a reported offset, prefer the match closest to it
        if reported_offset is not None:
            closest_pos = min(all_positions, key=lambda p: abs(p - reported_offset))
            return (closest_pos, closest_pos + len(search_text))
        else:
            # Otherwise, use the first occurrence
            return (all_positions[0], all_positions[0] + len(search_text))

    # Try with cleaned text (without ellipsis) if original didn't match
    if search_text_clean != search_text:
        all_positions_clean = []
        start = 0
        while True:
            pos = original_text.find(search_text_clean, start)
            if pos == -1:
                break
            all_positions_clean.append(pos)
            start = pos + 1

        if all_positions_clean:
            # If we have a reported offset, prefer the match closest to it
            if reported_offset is not None:
                closest_pos = min(
                    all_positions_clean, key=lambda p: abs(p - reported_offset)
                )
                return (closest_pos, closest_pos + len(search_text_clean))
            else:
                # Otherwise, use the first occurrence
                return (
                    all_positions_clean[0],
                    all_positions_clean[0] + len(search_text_clean),
                )

    # Try case-insensitive match
    search_text_lower = search_text.lower()
    original_text_lower = original_text.lower()
    all_positions_lower = []
    start = 0
    while True:
        pos = original_text_lower.find(search_text_lower, start)
        if pos == -1:
            break
        all_positions_lower.append(pos)
        start = pos + 1

    if all_positions_lower:
        # If we have a reported offset, prefer the match closest to it
        if reported_offset is not None:
            closest_pos = min(
                all_positions_lower, key=lambda p: abs(p - reported_offset)
            )
            return (closest_pos, closest_pos + len(search_text))
        else:
            # Otherwise, use the first occurrence
            return (all_positions_lower[0], all_positions_lower[0] + len(search_text))

    # Try case-insensitive match with cleaned text
    if search_text_clean != search_text:
        search_text_clean_lower = search_text_clean.lower()
        all_positions_clean_lower = []
        start = 0
        while True:
            pos = original_text_lower.find(search_text_clean_lower, start)
            if pos == -1:
                break
            all_positions_clean_lower.append(pos)
            start = pos + 1

        if all_positions_clean_lower:
            # If we have a reported offset, prefer the match closest to it
            if reported_offset is not None:
                closest_pos = min(
                    all_positions_clean_lower, key=lambda p: abs(p - reported_offset)
                )
                return (closest_pos, closest_pos + len(search_text_clean))
            else:
                # Otherwise, use the first occurrence
                return (
                    all_positions_clean_lower[0],
                    all_positions_clean_lower[0] + len(search_text_clean),
                )

    return None


def parse_llm_entity_response(
    response_text: str,
    original_text: str,
) -> List[Dict[str, Any]]:
    """
    Parse LLM response and extract entity information.
    Instead of using LLM-provided offsets, searches for the entity text
    in the original passage and derives offsets from the match position.

    Args:
        response_text: The LLM response text (should contain JSON)
        original_text: The original text that was analyzed (for validation)

    Returns:
        List of entity dictionaries with keys: Type, BeginOffset, EndOffset, Score, Text
    """
    entities = []

    # Try to extract JSON from the response
    # LLMs sometimes wrap JSON in markdown code blocks or add explanatory text
    json_match = re.search(
        r'\{[^{}]*"entities"[^{}]*\[.*?\].*?\}', response_text, re.DOTALL
    )
    if not json_match:
        # Try to find any JSON object
        json_match = re.search(r'\{.*?"entities".*?\}', response_text, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        try:
            # Clean up the JSON string
            json_str = json_str.strip()
            # Remove markdown code block markers if present
            json_str = re.sub(r"^```json\s*", "", json_str, flags=re.MULTILINE)
            json_str = re.sub(r"^```\s*", "", json_str, flags=re.MULTILINE)
            json_str = json_str.strip()

            data = json.loads(json_str)

            if "entities" in data and isinstance(data["entities"], list):
                for entity in data["entities"]:
                    # Validate entity has at least Type
                    if "Type" not in entity:
                        print(f"Warning: Entity missing Type field: {entity}")
                        continue

                    entity_type = str(entity["Type"])

                    # Get the text value from LLM response, or extract using offsets as fallback
                    entity_text = entity.get("Text", "")
                    reported_begin_offset = None
                    reported_end_offset = None

                    # Try to get reported offsets for disambiguation if text appears multiple times
                    if "BeginOffset" in entity and "EndOffset" in entity:
                        try:
                            reported_begin_offset = int(entity["BeginOffset"])
                            reported_end_offset = int(entity["EndOffset"])

                            # If no Text provided, extract it using the offsets
                            if (
                                not entity_text
                                and 0
                                <= reported_begin_offset
                                < reported_end_offset
                                <= len(original_text)
                            ):
                                entity_text = original_text[
                                    reported_begin_offset:reported_end_offset
                                ]
                        except (ValueError, TypeError):
                            pass

                    # If we still don't have text, skip this entity
                    if not entity_text:
                        print(
                            f"Warning: Entity of type '{entity_type}' has no Text value and invalid offsets"
                        )
                        continue

                    # Search for the text in the original passage
                    offsets = _find_text_in_passage(
                        entity_text, original_text, reported_begin_offset
                    )

                    if offsets:
                        begin_offset, end_offset = offsets
                        entity_dict = {
                            "Type": entity_type,
                            "BeginOffset": begin_offset,
                            "EndOffset": end_offset,
                            "Score": float(entity.get("Score", 0.8)),
                            "Text": original_text[
                                begin_offset:end_offset
                            ],  # Use actual text from passage
                        }
                        entities.append(entity_dict)
                    else:
                        print(
                            f"Warning: Could not find text '{entity_text[:50]}...' in original passage for entity type '{entity_type}'"
                        )
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from LLM response: {e}")
            print(f"Response text: {response_text[:500]}")
        except (ValueError, KeyError) as e:
            print(f"Error processing entity data: {e}")
    else:
        print("Warning: Could not find JSON in LLM response")
        print(f"Response text: {response_text[:500]}")

    return entities


def save_llm_prompt_response(
    system_prompt: str,
    user_prompt: str,
    response_text: str,
    output_folder: str,
    batch_number: int,
    model_choice: str,
    entities_to_detect: List[str],
    language: str,
    temperature: float,
    max_tokens: int,
    file_name: Optional[str] = None,
    page_number: Optional[int] = None,
) -> str:
    """
    Save LLM prompt and response to a text file for traceability.

    Args:
        system_prompt: System prompt sent to LLM
        user_prompt: User prompt sent to LLM
        response_text: Response text from LLM
        output_folder: Output folder path
        batch_number: Batch number for this call
        model_choice: Model used
        entities_to_detect: List of entities being detected
        language: Language code
        temperature: Temperature used
        max_tokens: Max tokens used
        file_name: Optional file name (without extension) for the filename
        page_number: Optional page number for the filename

    Returns:
        Path to the saved file
    """
    # Create LLM logs subfolder
    llm_logs_folder = os.path.join(output_folder, "llm_prompts_responses")
    os.makedirs(llm_logs_folder, exist_ok=True)

    # Create filename with file name and page number if available, otherwise use timestamp
    if file_name and page_number is not None:
        # Sanitize file name for use in filename (remove invalid characters)
        safe_file_name = "".join(
            c for c in file_name if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        safe_file_name = safe_file_name.replace(" ", "_")
        filename = (
            f"llm_{safe_file_name}_page_{page_number:04d}_batch_{batch_number:04d}.txt"
        )
    else:
        # Fallback to timestamp if file/page info not available
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_batch_{batch_number:04d}_{timestamp}.txt"
    filepath = os.path.join(llm_logs_folder, filename)

    # Write prompt and response to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("LLM ENTITY DETECTION - PROMPT AND RESPONSE LOG\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if file_name:
            f.write(f"File: {file_name}\n")
        if page_number is not None:
            f.write(f"Page: {page_number}\n")
        f.write(f"Batch Number: {batch_number}\n")
        f.write(f"Model: {model_choice}\n")
        f.write(f"Language: {language}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Max Tokens: {max_tokens}\n")
        f.write(f"Entities to Detect: {', '.join(entities_to_detect)}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("SYSTEM PROMPT\n")
        f.write("=" * 80 + "\n\n")
        f.write(system_prompt)
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("USER PROMPT\n")
        f.write("=" * 80 + "\n\n")
        f.write(user_prompt)
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("LLM RESPONSE\n")
        f.write("=" * 80 + "\n\n")
        f.write(response_text)
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("END OF LOG\n")
        f.write("=" * 80 + "\n")

    return filepath


def call_llm_for_entity_detection(
    text: str,
    entities_to_detect: List[str],
    language: str,
    bedrock_runtime: Optional[boto3.Session.client] = None,
    model_choice: str = CLOUD_LLM_PII_MODEL_CHOICE,
    temperature: float = LLM_PII_TEMPERATURE,
    max_tokens: int = LLM_PII_MAX_TOKENS,
    max_retries: int = LLM_PII_NUMBER_OF_RETRY_ATTEMPTS,
    retry_delay: int = LLM_PII_TIMEOUT_WAIT,
    output_folder: Optional[str] = None,
    batch_number: int = 0,
    custom_instructions: str = "",
    file_name: Optional[str] = None,
    page_number: Optional[int] = None,
    inference_method: Optional[str] = None,
    local_model=None,
    tokenizer=None,
    assistant_model=None,
    client=None,
    client_config=None,
    api_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Call LLM to detect entities in text using various inference methods.

    Args:
        text: Text to analyze
        entities_to_detect: List of entity types to detect
        language: Language code
        bedrock_runtime: AWS Bedrock runtime client (required for AWS method)
        model_choice: Model identifier (varies by inference method)
        temperature: Temperature for LLM generation (lower = more deterministic)
        max_tokens: Maximum tokens in response
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries (seconds)
        output_folder: Optional folder to save prompt/response logs
        batch_number: Batch number for logging
        custom_instructions: Optional custom instructions to include in the prompt
        file_name: Optional file name (without extension) for saving logs
        page_number: Optional page number for saving logs
        inference_method: Inference method to use ("aws-bedrock", "local", "inference-server", "azure-openai", "gemini")
                         If None, uses CHOSEN_LLM_PII_INFERENCE_METHOD from config
        local_model: Local model instance (required for "local" method)
        tokenizer: Tokenizer instance (required for "local" method with transformers)
        assistant_model: Assistant model for speculative decoding (optional)
        client: API client (required for "azure-openai" or "gemini" methods)
        client_config: Client config (required for "gemini" method)
        api_url: API URL for inference-server (required for "inference-server" method)

    Returns:
        List of entity dictionaries
    """
    # Determine inference method
    if inference_method is None:
        inference_method = CHOSEN_LLM_PII_INFERENCE_METHOD

    # Determine model source from model_choice if using model_name_map
    model_source = None
    if model_choice and model_name_map and model_choice in model_name_map:
        model_source = model_name_map[model_choice].get("source", "AWS")
        # Map model source to inference method
        if model_source == "Local":
            inference_method = "local"
        elif model_source == "inference-server":
            inference_method = "inference-server"
        elif model_source == "Azure/OpenAI":
            inference_method = "azure-openai"
        elif model_source == "Gemini":
            inference_method = "gemini"
        elif model_source == "AWS":
            inference_method = "aws-bedrock"

    system_prompt = create_entity_detection_system_prompt(
        entities_to_detect, language, custom_instructions
    )
    user_prompt = create_entity_detection_prompt(
        text, entities_to_detect, language, custom_instructions
    )

    # Use send_request from llm_funcs.py which handles all model sources, retries, and response parsing
    from gradio import Progress

    import tools.llm_funcs as llm_funcs_module
    from tools.llm_funcs import max_tokens as global_max_tokens
    from tools.llm_funcs import send_request

    # Map inference_method to model_source format expected by send_request
    model_source_map = {
        "aws-bedrock": "AWS",
        "local": "Local",
        "inference-server": "inference-server",
        "azure-openai": "Azure/OpenAI",
        "gemini": "Gemini",
    }

    model_source = model_source_map.get(inference_method, "AWS")

    # Prepare client and config for Gemini if needed
    if inference_method == "gemini" and (client is None or client_config is None):
        from tools.llm_funcs import construct_gemini_generative_model

        try:
            client, client_config = construct_gemini_generative_model(
                in_api_key="",  # Will use environment variable
                temperature=temperature,
                model_choice=model_choice,
                system_prompt=system_prompt,
                max_tokens=max_tokens,  # Use our specific max_tokens for entity detection
            )
        except Exception as e:
            raise ValueError(
                f"Failed to construct Gemini client: {e}. "
                f"Ensure GEMINI_API_KEY is set or pass client and client_config."
            )

    # Prepare client for Azure/OpenAI if needed
    if inference_method == "azure-openai" and client is None:
        from tools.llm_funcs import construct_azure_client

        try:
            client, _ = construct_azure_client(
                in_api_key="",  # Will use environment variable
                endpoint="",  # Will use environment variable
            )
        except Exception as e:
            raise ValueError(
                f"Failed to construct Azure/OpenAI client: {e}. "
                f"Ensure AZURE_OPENAI_API_KEY is set or pass client."
            )

    # Prepare local model if needed
    if inference_method == "local":
        from tools.llm_funcs import USE_LLAMA_CPP

        if local_model is None:
            from tools.llm_funcs import get_model

            try:
                local_model = get_model()
            except Exception as e:
                raise ValueError(
                    f"Failed to get local model: {e}. "
                    f"Ensure LOAD_LOCAL_MODEL_AT_START is True or pass local_model."
                )
        if tokenizer is None and USE_LLAMA_CPP != "True":
            from tools.llm_funcs import get_tokenizer

            try:
                tokenizer = get_tokenizer()
            except Exception as e:
                raise ValueError(
                    f"Failed to get tokenizer: {e}. "
                    f"Ensure LOAD_LOCAL_MODEL_AT_START is True or pass tokenizer."
                )

    # Set up API URL for inference-server if needed
    if inference_method == "inference-server" and api_url is None:
        api_url = INFERENCE_SERVER_API_URL
        if not api_url:
            raise ValueError(
                "api_url is required when using inference-server method. "
                "Set INFERENCE_SERVER_API_URL in config or pass api_url parameter."
            )

    # Temporarily override global max_tokens for send_request if it's different
    # (send_request uses global max_tokens variable, not a parameter)
    original_max_tokens = None
    if max_tokens != global_max_tokens:
        original_max_tokens = llm_funcs_module.max_tokens
        llm_funcs_module.max_tokens = max_tokens

    try:
        # Call send_request which handles all routing, retries, and response parsing
        # Note: send_request signature shows local_model=list() but it's actually used as a single model object
        response, conversation_history, response_text, _, _ = send_request(
            prompt=user_prompt,
            conversation_history=[],  # Empty for entity detection (no conversation history needed)
            client=client,
            config=client_config,
            model_choice=model_choice,
            system_prompt=system_prompt,
            temperature=temperature,
            bedrock_runtime=bedrock_runtime,
            model_source=model_source,
            local_model=(
                local_model if local_model else []
            ),  # Pass model directly (signature shows list but uses as single object)
            tokenizer=tokenizer,
            assistant_model=assistant_model,
            progress=Progress(
                track_tqdm=False
            ),  # Disable progress bar for entity detection
            api_url=api_url,
        )
    except Exception as e:
        print(f"LLM entity detection failed: {e}")
        raise
    finally:
        # Restore original max_tokens if we changed it
        if original_max_tokens is not None:
            llm_funcs_module.max_tokens = original_max_tokens

    # Save prompt and response if output_folder is provided
    if output_folder and response_text:
        try:
            saved_file = save_llm_prompt_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=response_text,
                output_folder=output_folder,
                batch_number=batch_number,
                model_choice=model_choice,
                entities_to_detect=entities_to_detect,
                language=language,
                temperature=temperature,
                max_tokens=max_tokens,
                file_name=file_name,
                page_number=page_number,
            )
            print(f"Saved LLM prompt/response to: {saved_file}")
        except Exception as e:
            print(f"Warning: Could not save LLM prompt/response: {e}")

    # Parse the response
    entities = parse_llm_entity_response(response_text, text)

    return entities


def map_back_llm_entity_results(
    entities: List[Dict[str, Any]],
    current_batch_mapping: List[Tuple],
    allow_list: List[str],
    chosen_redact_comprehend_entities: List[str],
    all_text_line_results: List[Tuple],
) -> List[Tuple]:
    """
    Map LLM-detected entities back to line-level results.
    Similar to map_back_comprehend_entity_results but for LLM responses.

    Args:
        entities: List of entity dictionaries from LLM
        current_batch_mapping: Mapping of batch positions to line indices
        allow_list: List of allowed text values (to skip)
        chosen_redact_comprehend_entities: List of entity types to include
        all_text_line_results: Existing line-level results to append to

    Returns:
        Updated all_text_line_results
    """
    if not entities:
        return all_text_line_results

    for entity in entities:
        entity_type = entity.get("Type")
        # Allow all entity types returned by LLM, including custom types from custom instructions
        # Log when a custom entity type (not in the original list) is found
        if entity_type not in chosen_redact_comprehend_entities:
            print(
                f"Info: Found custom entity type '{entity_type}' (not in original detection list). "
                f"Including it in results as it was returned by LLM."
            )

        entity_start = entity["BeginOffset"]
        entity_end = entity["EndOffset"]
        entity.get("Text", "")

        # Track if the entity has been added to any line
        added_to_line = False

        # Find the correct line and offset within that line
        for (
            batch_start,
            line_idx,
            original_line,
            chars,
            line_offset,
        ) in current_batch_mapping:
            # Calculate the end position of this line segment in the batch
            if line_offset is not None:
                # Line offset is the start position within the line
                line_text_length = len(original_line.text[line_offset:])
            else:
                line_text_length = len(original_line.text)

            batch_end = batch_start + line_text_length

            # Check if the entity overlaps with the current line
            if batch_start < entity_end and batch_end > entity_start:
                # Calculate the relative position within the line
                if line_offset is not None:
                    relative_start = max(0, entity_start - batch_start + line_offset)
                    relative_end = min(
                        entity_end - batch_start + line_offset, len(original_line.text)
                    )
                else:
                    relative_start = max(0, entity_start - batch_start)
                    relative_end = min(
                        entity_end - batch_start, len(original_line.text)
                    )

                result_text = original_line.text[relative_start:relative_end]

                if result_text not in allow_list:
                    # Create entity dict in Comprehend-like format
                    adjusted_entity = {
                        "Type": entity_type,
                        "BeginOffset": relative_start,
                        "EndOffset": relative_end,
                        "Score": entity.get("Score", 0.8),
                    }

                    # Import here to avoid circular imports
                    from tools.presidio_analyzer_custom import (
                        recognizer_result_from_dict,
                    )

                    recogniser_entity = recognizer_result_from_dict(adjusted_entity)

                    # Check if this line already has an entry
                    existing_entry = next(
                        (
                            entry
                            for idx, entry in all_text_line_results
                            if idx == line_idx
                        ),
                        None,
                    )
                    if existing_entry is None:
                        all_text_line_results.append((line_idx, [recogniser_entity]))
                    else:
                        existing_entry.append(recogniser_entity)

                    added_to_line = True

        # Optional: Handle cases where the entity does not fit in any line
        if not added_to_line:
            print(
                f"Entity '{entity_type}' at position {entity_start}-{entity_end} does not fit in any line."
            )

    return all_text_line_results


def do_llm_entity_detection_call(
    current_batch: str,
    current_batch_mapping: List[Tuple],
    bedrock_runtime: Optional[boto3.Session.client] = None,
    language: str = "en",
    allow_list: List[str] = None,
    chosen_redact_comprehend_entities: List[str] = None,
    all_text_line_results: List[Tuple] = None,
    model_choice: str = CLOUD_LLM_PII_MODEL_CHOICE,
    temperature: float = LLM_PII_TEMPERATURE,
    max_tokens: int = LLM_PII_MAX_TOKENS,
    output_folder: Optional[str] = None,
    batch_number: int = 0,
    custom_instructions: str = "",
    file_name: Optional[str] = None,
    page_number: Optional[int] = None,
    inference_method: Optional[str] = None,
    local_model=None,
    tokenizer=None,
    assistant_model=None,
    client=None,
    client_config=None,
    api_url: Optional[str] = None,
) -> List[Tuple]:
    """
    Call LLM for entity detection on a batch of text.
    Similar interface to do_aws_comprehend_call.

    Args:
        current_batch: Text batch to analyze
        current_batch_mapping: Mapping of batch positions to line indices
        bedrock_runtime: AWS Bedrock runtime client (required for AWS method)
        language: Language code
        allow_list: List of allowed text values
        chosen_redact_comprehend_entities: List of entity types to detect
        all_text_line_results: Existing line-level results
        model_choice: Model identifier (varies by inference method)
        temperature: Temperature for LLM generation
        max_tokens: Maximum tokens in response
        output_folder: Optional folder to save prompt/response logs
        batch_number: Batch number for logging
        custom_instructions: Optional custom instructions to include in the prompt
        file_name: Optional file name (without extension) for saving logs
        page_number: Optional page number for saving logs
        inference_method: Inference method to use (if None, uses config default)
        local_model: Local model instance (required for "local" method)
        tokenizer: Tokenizer instance (required for "local" method with transformers)
        assistant_model: Assistant model for speculative decoding (optional)
        client: API client (required for "azure-openai" or "gemini" methods)
        client_config: Client config (required for "gemini" method)
        api_url: API URL for inference-server (required for "inference-server" method)

    Returns:
        Updated all_text_line_results
    """
    if not current_batch:
        return all_text_line_results or []

    if allow_list is None:
        allow_list = []
    if chosen_redact_comprehend_entities is None:
        chosen_redact_comprehend_entities = []
    if all_text_line_results is None:
        all_text_line_results = []

    try:
        entities = call_llm_for_entity_detection(
            text=current_batch.strip(),
            entities_to_detect=chosen_redact_comprehend_entities,
            language=language,
            bedrock_runtime=bedrock_runtime,
            model_choice=model_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            output_folder=output_folder,
            batch_number=batch_number,
            custom_instructions=custom_instructions,
            file_name=file_name,
            page_number=page_number,
            inference_method=inference_method,
            local_model=local_model,
            tokenizer=tokenizer,
            assistant_model=assistant_model,
            client=client,
            client_config=client_config,
            api_url=api_url,
        )

        all_text_line_results = map_back_llm_entity_results(
            entities,
            current_batch_mapping,
            allow_list,
            chosen_redact_comprehend_entities,
            all_text_line_results,
        )

        return all_text_line_results

    except Exception as e:
        print(f"LLM entity detection call failed: {e}")
        raise
