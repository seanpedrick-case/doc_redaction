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
from gradio import Progress

from tools.config import (
    CHOSEN_LLM_PII_INFERENCE_METHOD,
    CLOUD_LLM_PII_CUSTOM_INSTRUCTIONS_MODEL_CHOICE,
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
    # Use send_request from llm_funcs.py which handles all model sources, retries, and response parsing

    from tools.llm_funcs import (
        ResponseObject,
        call_aws_bedrock,
        construct_azure_client,
        send_request,
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


def _find_all_text_in_passage(
    search_text: str, original_text: str
) -> List[Tuple[int, int]]:
    """
    Find all positions of search_text in original_text and return a list of (begin, end) offsets.
    Uses the same search strategy as _find_text_in_passage (exact, then cleaned, then case-insensitive).
    LLM offset values are never used; positions come only from search.

    Returns:
        List of (begin_offset, end_offset) tuples, sorted by begin_offset (ascending).
    """
    if not search_text:
        return []

    search_text_clean = search_text.rstrip("...").strip()

    def find_all_exact(needle: str, haystack: str) -> List[Tuple[int, int]]:
        result = []
        start = 0
        while True:
            pos = haystack.find(needle, start)
            if pos == -1:
                break
            result.append((pos, pos + len(needle)))
            start = pos + 1
        return result

    positions = find_all_exact(search_text, original_text)
    if positions:
        return sorted(positions, key=lambda p: p[0])

    if search_text_clean != search_text:
        positions = find_all_exact(search_text_clean, original_text)
        if positions:
            return sorted(positions, key=lambda p: p[0])

    # Case-insensitive
    needle_lower = search_text.lower()
    haystack_lower = original_text.lower()
    positions = find_all_exact(needle_lower, haystack_lower)
    if positions:
        # Return (start, start + len(search_text)) so length matches original entity text
        return sorted(
            [(p[0], p[0] + len(search_text)) for p in positions], key=lambda p: p[0]
        )

    if search_text_clean != search_text:
        needle_clean_lower = search_text_clean.lower()
        positions = find_all_exact(needle_clean_lower, haystack_lower)
        if positions:
            return sorted(
                [(p[0], p[0] + len(search_text_clean)) for p in positions],
                key=lambda p: p[0],
            )

    return []


def _entity_get(obj: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get value from entity dict with case-insensitive key lookup (e.g. BeginOffset vs beginOffset)."""
    key_lower = key.lower()
    for k, v in obj.items():
        if k.lower() == key_lower:
            return v
    return default


def parse_llm_entity_response(
    response_text: str,
    original_text: str,
) -> List[Dict[str, Any]]:
    """
    Parse LLM response and extract entity information.
    LLM BeginOffset/EndOffset are never used as character positions. Positions are
    taken only from searching for the entity text in the passage. When the same
    text is returned multiple times (e.g. "University of Notre Dame" three times),
    entities are sorted by their reported BeginOffset to define order; the i-th
    entity (by that order) is assigned to the i-th search result (by position).

    Args:
        response_text: The LLM response text (should contain JSON)
        original_text: The original text that was analyzed (for validation)

    Returns:
        List of entity dictionaries with keys: Type, BeginOffset, EndOffset, Score, Text
    """
    entities_out: List[Dict[str, Any]] = []

    # Remove <think> tags and their content (common in some LLM outputs)
    # This handles cases where LLMs include thinking/reasoning tags
    response_text = re.sub(
        r"<think>.*?</think>", "", response_text, flags=re.DOTALL | re.IGNORECASE
    )
    response_text = re.sub(
        r"<thinking>.*?</thinking>", "", response_text, flags=re.DOTALL | re.IGNORECASE
    )

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

            # Fix common JSON issues:
            # 1. Remove trailing commas before closing brackets/braces
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*]", "]", json_str)

            # 2. Fix unquoted string values (e.g., "Type": NAME should be "Type": "NAME")
            # This handles cases where LLMs output unquoted identifiers as values
            # Pattern: "key": VALUE where VALUE is an unquoted identifier
            def fix_unquoted_value(match):
                key_part = match.group(1)  # The key (e.g., "Type")
                value = match.group(2)  # The unquoted value
                separator = match.group(3)  # The separator (comma, closing brace, etc.)
                # Only fix if it looks like an identifier (alphanumeric/underscore, not a number or boolean)
                if re.match(
                    r"^[A-Za-z_][A-Za-z0-9_]*$", value
                ) and value.lower() not in ["true", "false", "null"]:
                    return f'{key_part}: "{value}"{separator}'
                return match.group(0)  # Return original if it doesn't need fixing

            # Fix unquoted string values after colons (common in LLM outputs)
            # Match: "key": VALUE where VALUE is unquoted identifier followed by comma, }, or ]
            # This pattern handles: "Type": NAME, or "Type": EMAIL_ADDRESS}
            json_str = re.sub(
                r'("[\w]+")\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*([,}\]])',
                fix_unquoted_value,
                json_str,
            )

            # Also handle cases where unquoted value is at end of line or followed by newline
            json_str = re.sub(
                r'("[\w]+")\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\n)',
                r'\1: "\2"\3',
                json_str,
            )

            # Try to parse the JSON
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                # If parsing still fails, try a more aggressive fix for unquoted values
                # This is a fallback that quotes any unquoted identifiers after colons
                print(
                    f"Initial JSON parse failed: {e}. Attempting more aggressive fixes..."
                )

                # More aggressive fix: quote any unquoted word after a colon that's not already quoted
                # Pattern: ": WORD" where WORD is not in quotes and not a number/boolean
                def quote_unquoted_identifier(match):
                    prefix = match.group(1)  # Everything before the colon
                    value = match.group(2)  # The unquoted value
                    suffix = match.group(3)  # Everything after (comma, brace, etc.)
                    # Only quote if it's a valid identifier and not a boolean/null
                    if re.match(
                        r"^[A-Za-z_][A-Za-z0-9_]*$", value
                    ) and value.lower() not in ["true", "false", "null"]:
                        return f'{prefix}: "{value}"{suffix}'
                    return match.group(0)

                # Try fixing unquoted values more aggressively
                json_str = re.sub(
                    r"(:\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*[,}\]])",
                    quote_unquoted_identifier,
                    json_str,
                )

                # Try parsing again
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as e2:
                    print(f"JSON parsing failed after fixes: {e2}")
                    print(f"Cleaned JSON string (first 1000 chars): {json_str[:1000]}")
                    raise e2

            if "entities" in data and isinstance(data["entities"], list):
                # Collect raw entity records (Type, Text, Score, reported BeginOffset for order only)
                raw_entities: List[Dict[str, Any]] = []
                for entity in data["entities"]:
                    entity_type_val = _entity_get(entity, "Type")
                    if entity_type_val is None:
                        print(f"Warning: Entity missing Type field: {entity}")
                        continue
                    entity_text = _entity_get(entity, "Text", "")
                    reported_begin = _entity_get(entity, "BeginOffset")
                    if reported_begin is not None:
                        try:
                            reported_begin = int(reported_begin)
                        except (ValueError, TypeError):
                            reported_begin = None
                    reported_end = _entity_get(entity, "EndOffset")
                    if reported_end is not None:
                        try:
                            reported_end = int(reported_end)
                        except (ValueError, TypeError):
                            reported_end = None
                    # If no Text, try to derive from reported offsets (for display/grouping only)
                    if (
                        not entity_text
                        and reported_begin is not None
                        and reported_end is not None
                        and 0 <= reported_begin < reported_end <= len(original_text)
                    ):
                        entity_text = original_text[reported_begin:reported_end]
                    if not entity_text:
                        print(
                            f"Warning: Entity of type '{entity_type_val}' has no Text value and invalid offsets"
                        )
                        continue
                    raw_entities.append(
                        {
                            "Type": str(entity_type_val),
                            "Text": entity_text,
                            "Score": float(_entity_get(entity, "Score", 0.8)),
                            "reported_begin": reported_begin,
                        }
                    )

                # Group by entity text (normalised) so we can assign search positions by order
                groups: Dict[str, List[Dict[str, Any]]] = {}
                for rec in raw_entities:
                    key = rec["Text"].strip().lower()
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(rec)

                # For each group: find all positions by search only; sort entities by reported BeginOffset; assign 1:1
                for _key, group in groups.items():
                    # Use the first entity's Text for searching (exact string to find)
                    search_text = group[0]["Text"]
                    positions = _find_all_text_in_passage(search_text, original_text)
                    if not positions:
                        print(
                            f"Warning: Could not find text '{search_text[:50]}...' in original passage"
                        )
                        continue
                    # Order entities by reported BeginOffset (ascending); None goes last
                    ordered = sorted(
                        group,
                        key=lambda r: (
                            r["reported_begin"] is None,
                            r["reported_begin"] or 0,
                        ),
                    )
                    # Assign i-th entity (by reported order) to i-th search result (by position)
                    for i in range(min(len(ordered), len(positions))):
                        start, end = positions[i]
                        rec = ordered[i]
                        entities_out.append(
                            {
                                "Type": rec["Type"],
                                "BeginOffset": start,
                                "EndOffset": end,
                                "Score": rec["Score"],
                                "Text": original_text[start:end],
                            }
                        )
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from LLM response: {e}")
            print(f"Response text: {response_text[:500]}")
        except (ValueError, KeyError) as e:
            print(f"Error processing entity data: {e}")
    else:
        print("Warning: Could not find JSON in LLM response")
        print(f"Response text: {response_text[:500]}")

    return entities_out


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

    Writes the exact system prompt and user prompt that were sent to the model
    (e.g. for local transformers, inference-server, AWS, etc.). Each section is
    clearly delimited so the log never duplicates or conflates system vs user.

    Args:
        system_prompt: System prompt sent to LLM (exactly as passed to the model).
        user_prompt: User prompt sent to LLM (exactly as passed to the model).
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
    # Normalise to strings so we never write "None" or non-string types
    system_prompt_str = (system_prompt if system_prompt is not None else "").strip()
    user_prompt_str = (user_prompt if user_prompt is not None else "").strip()

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

    # Write prompt and response to file with explicit section boundaries
    # so system and user prompts are never duplicated or mixed.
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
        f.write("SYSTEM PROMPT (sent as system role)\n")
        f.write("=" * 80 + "\n")
        f.write("--- BEGIN SYSTEM PROMPT ---\n")
        f.write(system_prompt_str)
        f.write("\n--- END SYSTEM PROMPT ---\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("USER PROMPT (sent as user role)\n")
        f.write("=" * 80 + "\n")
        if (
            system_prompt_str
            and user_prompt_str
            and system_prompt_str == user_prompt_str
        ):
            f.write(
                "[NOTE: System and user prompt content were identical - check caller.]\n"
            )
        f.write("--- BEGIN USER PROMPT ---\n")
        f.write(user_prompt_str)
        f.write("\n--- END USER PROMPT ---\n")

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
    # Ensure custom_instructions is a string (callers may pass bool or other types).
    # Treat boolean True and the string "True" as empty (e.g. from an unchecked/empty Gradio box).
    if not isinstance(custom_instructions, str):
        custom_instructions = (
            ""
            if custom_instructions is True or not custom_instructions
            else str(custom_instructions)
        )
    if (
        isinstance(custom_instructions, str)
        and custom_instructions.strip().lower() == "true"
    ):
        custom_instructions = ""

    # Determine inference method
    if inference_method is None:
        inference_method = CHOSEN_LLM_PII_INFERENCE_METHOD

    # When custom instructions are provided, use the upgraded model if configured
    custom_instructions_model = (
        CLOUD_LLM_PII_CUSTOM_INSTRUCTIONS_MODEL_CHOICE.strip()
        if isinstance(CLOUD_LLM_PII_CUSTOM_INSTRUCTIONS_MODEL_CHOICE, str)
        and CLOUD_LLM_PII_CUSTOM_INSTRUCTIONS_MODEL_CHOICE.strip()
        else ""
    )
    if (
        custom_instructions.strip()
        and model_choice == CLOUD_LLM_PII_MODEL_CHOICE
        and custom_instructions_model
    ):
        model_choice = custom_instructions_model

    # Filter out CUSTOM_VLM_* entities (these are handled separately via VLM)
    filtered_entities = [
        entity for entity in entities_to_detect if not entity.startswith("CUSTOM_VLM_")
    ]

    # Validate that we have either entities or custom instructions
    if not filtered_entities and (
        not custom_instructions or not custom_instructions.strip()
    ):
        raise ValueError(
            "No standard entities selected and no custom instructions provided. "
            "Please select at least one entity type (excluding CUSTOM_VLM_* entities) or provide custom instructions for LLM-based PII detection."
        )

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
        filtered_entities, language, custom_instructions
    )
    user_prompt = create_entity_detection_prompt(
        text, filtered_entities, language, custom_instructions
    )

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

    # Set up API URL for inference-server if needed
    if inference_method == "inference-server" and api_url is None:
        api_url = INFERENCE_SERVER_API_URL
        if not api_url:
            raise ValueError(
                "api_url is required when using inference-server method. "
                "Set INFERENCE_SERVER_API_URL in config or pass api_url parameter."
            )

    try:
        # Call send_request which handles all routing, retries, and response parsing
        # Note: send_request signature shows local_model=list() but it's actually used as a single model object
        (
            response,
            conversation_history,
            response_text,
            num_transformer_input_tokens,
            num_transformer_generated_tokens,
        ) = send_request(
            prompt=user_prompt,
            conversation_history=[],  # Empty for entity detection (no conversation history needed)
            client=client,
            config=client_config,
            model_choice=model_choice,
            system_prompt=system_prompt,
            temperature=temperature,
            bedrock_runtime=bedrock_runtime,
            model_source=model_source,
            # local_model=(
            #     local_model if local_model else []
            # ),  # Pass model directly (signature shows list but uses as single object)
            # tokenizer=tokenizer,
            # assistant_model=assistant_model,
            progress=Progress(
                track_tqdm=False
            ),  # Disable progress bar for entity detection
            api_url=api_url,
        )
    except Exception as e:
        print(f"LLM entity detection failed: {e}")
        raise

    # Save prompt and response if output_folder is provided.
    # Use the same system_prompt and user_prompt that were sent to the model
    # (no combined/rendered version) so the log correctly shows system vs user.
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

    # Extract token usage from response
    input_tokens = 0
    output_tokens = 0

    try:
        if isinstance(response, dict) and "usage" in response:
            # inference-server or llama-cpp format
            input_tokens = response["usage"].get("prompt_tokens", 0)
            output_tokens = response["usage"].get("completion_tokens", 0)
        elif hasattr(response, "usage_metadata"):
            # Check if it's AWS Bedrock format
            if isinstance(response.usage_metadata, dict):
                input_tokens = response.usage_metadata.get("inputTokens", 0)
                output_tokens = response.usage_metadata.get("outputTokens", 0)
            # Check if it's Gemini format
            elif hasattr(response.usage_metadata, "prompt_token_count"):
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
    except (KeyError, AttributeError) as e:
        print(f"Warning: Could not extract token usage from response: {e}")
        # Fallback: use transformer token counts if available
        if num_transformer_input_tokens and num_transformer_input_tokens > 0:
            input_tokens = num_transformer_input_tokens
        if num_transformer_generated_tokens and num_transformer_generated_tokens > 0:
            output_tokens = num_transformer_generated_tokens

    return entities, input_tokens, output_tokens


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
        allow_list: List of allowed text values (to skip) - case-insensitive matching
        chosen_redact_comprehend_entities: List of entity types to include
        all_text_line_results: Existing line-level results to append to

    Returns:
        Updated all_text_line_results
    """
    if not entities:
        return all_text_line_results

    # Normalize allow_list for case-insensitive matching
    if allow_list:
        allow_list_normalized = [item.strip().lower() for item in allow_list if item]
    else:
        allow_list_normalized = []

    for entity in entities:
        entity_type = entity.get("Type")
        # Allow all entity types returned by LLM, including custom types from custom instructions
        # Log when a custom entity type (not in the original list) is found
        # if entity_type not in chosen_redact_comprehend_entities:
        #     print(
        #         f"Info: Found custom entity type '{entity_type}' (not in original detection list). "
        #         f"Including it in results as it was returned by LLM."
        #     )

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

                # Check if result_text is in allow_list (case-insensitive)
                # If allow_list contains this text, skip adding it as a PII entity
                # This allows allow_list terms to "overrule" LLM PII detection
                result_text_normalized = result_text.strip().lower()
                if result_text_normalized not in allow_list_normalized:
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
) -> Tuple[List[Tuple], int, int]:
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
        Tuple of (updated all_text_line_results, input_tokens, output_tokens)
    """
    if not current_batch:
        return (all_text_line_results or [], 0, 0)

    if allow_list is None:
        allow_list = []
    if chosen_redact_comprehend_entities is None:
        chosen_redact_comprehend_entities = []
    if all_text_line_results is None:
        all_text_line_results = []

    try:
        entities, input_tokens, output_tokens = call_llm_for_entity_detection(
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

        return all_text_line_results, input_tokens, output_tokens

    except Exception as e:
        print(f"LLM entity detection call failed: {e}")
        raise
