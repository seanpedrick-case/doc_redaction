"""
LLM-based entity detection using AWS Bedrock.
This module provides functions to detect PII entities using LLMs instead of AWS Comprehend.
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import boto3

from tools.config import (
    LLM_MODEL_CHOICE,
    LLM_PII_MAX_TOKENS,
    LLM_PII_NUMBER_OF_RETRY_ATTEMPTS,
    LLM_PII_TEMPERATURE,
    LLM_PII_TIMEOUT_WAIT,
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


def parse_llm_entity_response(
    response_text: str,
    original_text: str,
) -> List[Dict[str, Any]]:
    """
    Parse LLM response and extract entity information.

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
                    # Validate entity structure
                    if all(
                        key in entity for key in ["Type", "BeginOffset", "EndOffset"]
                    ):
                        begin_offset = int(entity["BeginOffset"])
                        end_offset = int(entity["EndOffset"])

                        # Validate offsets are within text bounds
                        if 0 <= begin_offset < end_offset <= len(original_text):
                            # Extract the actual text to verify
                            extracted_text = original_text[begin_offset:end_offset]

                            entity_dict = {
                                "Type": str(entity["Type"]),
                                "BeginOffset": begin_offset,
                                "EndOffset": end_offset,
                                "Score": float(entity.get("Score", 0.8)),
                                "Text": extracted_text,
                            }
                            entities.append(entity_dict)
                        else:
                            print(
                                f"Warning: Invalid offsets {begin_offset}-{end_offset} for text of length {len(original_text)}"
                            )
                    else:
                        print(f"Warning: Entity missing required fields: {entity}")
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

    Returns:
        Path to the saved file
    """
    # Create LLM logs subfolder
    llm_logs_folder = os.path.join(output_folder, "llm_prompts_responses")
    os.makedirs(llm_logs_folder, exist_ok=True)

    # Create filename with timestamp and batch number
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"llm_batch_{batch_number:04d}_{timestamp}.txt"
    filepath = os.path.join(llm_logs_folder, filename)

    # Write prompt and response to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("LLM ENTITY DETECTION - PROMPT AND RESPONSE LOG\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
    bedrock_runtime: boto3.Session.client,
    model_choice: str = LLM_MODEL_CHOICE,
    temperature: float = LLM_PII_TEMPERATURE,
    max_tokens: int = LLM_PII_MAX_TOKENS,
    max_retries: int = LLM_PII_NUMBER_OF_RETRY_ATTEMPTS,
    retry_delay: int = LLM_PII_TIMEOUT_WAIT,
    output_folder: Optional[str] = None,
    batch_number: int = 0,
    custom_instructions: str = "",
) -> List[Dict[str, Any]]:
    """
    Call LLM (via AWS Bedrock) to detect entities in text.

    Args:
        text: Text to analyze
        entities_to_detect: List of entity types to detect
        language: Language code
        bedrock_runtime: AWS Bedrock runtime client
        model_choice: Model identifier for Bedrock
        temperature: Temperature for LLM generation (lower = more deterministic)
        max_tokens: Maximum tokens in response
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries (seconds)
        custom_instructions: Optional custom instructions to include in the prompt

    Returns:
        List of entity dictionaries
    """
    if call_aws_bedrock is None:
        raise ImportError(
            "LLM functions not available. Cannot perform LLM-based entity detection."
        )

    system_prompt = create_entity_detection_system_prompt(
        entities_to_detect, language, custom_instructions
    )
    user_prompt = create_entity_detection_prompt(
        text, entities_to_detect, language, custom_instructions
    )

    for attempt in range(max_retries):
        try:
            # Call AWS Bedrock
            response: ResponseObject = call_aws_bedrock(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                model_choice=model_choice,
                bedrock_runtime=bedrock_runtime,
            )

            # Save prompt and response if output_folder is provided
            if output_folder:
                try:
                    saved_file = save_llm_prompt_response(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        response_text=response.text,
                        output_folder=output_folder,
                        batch_number=batch_number,
                        model_choice=model_choice,
                        entities_to_detect=entities_to_detect,
                        language=language,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    print(f"Saved LLM prompt/response to: {saved_file}")
                except Exception as e:
                    print(f"Warning: Could not save LLM prompt/response: {e}")

            # Parse the response
            entities = parse_llm_entity_response(response.text, text)

            return entities

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"LLM entity detection failed after {max_retries} attempts: {e}")
                raise
            print(
                f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds..."
            )
            time.sleep(retry_delay)

    return []


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
        if entity_type not in chosen_redact_comprehend_entities:
            continue

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
    bedrock_runtime: boto3.Session.client,
    language: str,
    allow_list: List[str],
    chosen_redact_comprehend_entities: List[str],
    all_text_line_results: List[Tuple],
    model_choice: str = LLM_MODEL_CHOICE,
    temperature: float = LLM_PII_TEMPERATURE,
    max_tokens: int = LLM_PII_MAX_TOKENS,
    output_folder: Optional[str] = None,
    batch_number: int = 0,
    custom_instructions: str = "",
) -> List[Tuple]:
    """
    Call LLM for entity detection on a batch of text.
    Similar interface to do_aws_comprehend_call.

    Args:
        current_batch: Text batch to analyze
        current_batch_mapping: Mapping of batch positions to line indices
        bedrock_runtime: AWS Bedrock runtime client
        language: Language code
        allow_list: List of allowed text values
        chosen_redact_comprehend_entities: List of entity types to detect
        all_text_line_results: Existing line-level results
        model_choice: Model identifier for Bedrock
        temperature: Temperature for LLM generation
        max_tokens: Maximum tokens in response
        custom_instructions: Optional custom instructions to include in the prompt

    Returns:
        Updated all_text_line_results
    """
    if not current_batch:
        return all_text_line_results

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
