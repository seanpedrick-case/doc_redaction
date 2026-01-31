"""
Prompts for LLM-based entity detection.
These prompts are designed to extract PII entities with character positions.
"""


def create_entity_detection_prompt(
    text: str,
    entities_to_detect: list[str],
    language: str = "en",
    custom_instructions: str = "",
) -> str:
    """
    Create a prompt for LLM-based entity detection.

    Args:
        text: The text to analyze
        entities_to_detect: List of entity types to detect (e.g., ["EMAIL", "PHONE_NUMBER", "NAME"])
        language: Language code (e.g., "en", "es", "fr")
        custom_instructions: Optional custom instructions to include in the prompt (e.g., "don't redact anything related to Mark Wilson")

    Returns:
        Formatted prompt string
    """
    # Filter out CUSTOM_VLM_* entities (these are handled separately via VLM)
    filtered_entities = [
        entity for entity in entities_to_detect if not entity.startswith("CUSTOM_VLM_")
    ]

    # custom_instructions_section = ""
    if custom_instructions and custom_instructions.strip():
        # custom_instructions_section = (
        #     f"\n\nIMPORTANT CUSTOM INSTRUCTIONS:\n{custom_instructions.strip()}\n"
        # )
        pass

    # Handle case where no standard entities are selected
    if filtered_entities:
        pass
        # entities_list = ", ".join(filtered_entities)
        # entity_section = f"Your task is to analyse the provided text and identify all instances of the following entity types/labels: {entities_list}."
        # entity_rule = f"Only return entities/labels that match the requested types/labels: {entities_list}"
    else:
        # No standard entities selected - only use custom instructions
        # entity_section = "Your task is to analyse the provided text and identify PII entities based on the custom instructions provided below."
        # entity_rule = "Return entities based on the custom instructions provided."
        if not custom_instructions or not custom_instructions.strip():
            raise ValueError(
                "No standard entities selected and no custom instructions provided. "
                "Please select at least one entity type or provide custom instructions for LLM-based PII detection."
            )

    #     prompt = f"""You are an expert at identifying Personally Identifiable Information (PII) in text. {entity_section}{custom_instructions_section}

    # IMPORTANT: You must return your response as a valid JSON object with the following structure:
    # {{
    #   "entities": [
    #     {{
    #       "Type": "ENTITY_TYPE",
    #       "BeginOffset": <start_character_position>,
    #       "EndOffset": <end_character_position>,
    #       "Score": <confidence_score_0_to_1>,
    #       "Text": "<the_actual_text_found>"
    #     }}
    #   ]
    # }}

    # Rules:
    # 1. Character positions (BeginOffset and EndOffset) must be exact character indices in the original text (0-based indexing)
    # 2. BeginOffset is the position of the first character of the entity/label
    # 3. EndOffset is the position AFTER the last character of the entity/label (exclusive, like Python string slicing)
    # 4. Score should be a decimal between 0.0 and 1.0 representing your confidence
    # 5. Text should be the exact substring from the original text
    # 6. {entity_rule}
    # 7. If no entities/labels are found, return: {{"entities": []}}
    # 8. The JSON must be valid and parseable - do not include any explanatory text outside the JSON
    # 9. If IMPORTANT CUSTOM INSTRUCTIONS are provided, follow them carefully. They override all other instructions.

    # Text to analyse:
    # {text}

    # Return only the JSON object, nothing else:"""

    prompt = f"""### Task
Analyse the following text according to the instructions provided in the system prompt.
    
### Text:
{text}

### Final instruction:
Return only the JSON object, nothing else:"""

    return prompt


def create_entity_detection_system_prompt(
    entities_to_detect: list[str],
    language: str = "en",
    custom_instructions: str = "",
) -> str:
    """
    Create a system prompt for LLM-based entity detection.

    Args:
        entities_to_detect: List of entity types to detect
        language: Language code
        custom_instructions: Optional custom instructions to include in the system prompt

    Returns:
        System prompt string
    """
    # Filter out CUSTOM_VLM_* entities (these are handled separately via VLM)
    filtered_entities = [
        entity for entity in entities_to_detect if not entity.startswith("CUSTOM_VLM_")
    ]

    custom_instructions_section = ""
    # if custom_instructions and custom_instructions.strip():
    #     custom_instructions_section = (
    #         f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_instructions.strip()}\n"
    #     )
    if custom_instructions and custom_instructions.strip():
        custom_instructions_section = (
            f"\n\n## USER-SPECIFIC CONSTRAINTS:\n{custom_instructions.strip()}\n"
        )
    else:
        custom_instructions_section = "No specific user constraints provided."

    # Handle case where no standard entities are selected
    if filtered_entities:
        entity_types_section = f"Entity types to detect: {filtered_entities}"
    else:
        entity_types_section = "Entity types to detect: Based on custom instructions provided (no standard entity types selected)."
        if not custom_instructions or not custom_instructions.strip():
            raise ValueError(
                "No standard entities selected and no custom instructions provided. "
                "Please select at least one entity type or provide custom instructions for LLM-based PII detection."
            )

    #     system_prompt = f"""You are a precise PII (Personally Identifiable Information) detection system. Your role is to identify specific entity types in text and return them in a structured JSON format with exact character positions.{custom_instructions_section}

    # {entity_types_section}

    # For each entity found, you must provide:
    # - Type: The entity type (must match one of the requested types)
    # - BeginOffset: The starting character position (0-based index)
    # - EndOffset: The ending character position (exclusive, like Python string slicing)
    # - Score: Confidence score between 0.0 and 1.0
    # - Text: The exact text substring that was identified

    # Be precise with character positions - they must match the exact location in the original text. Handle edge cases like:
    # - Entities at the start of text (BeginOffset = 0)
    # - Entities at the end of text (EndOffset = text length)
    # - Entities with punctuation (include punctuation if it's part of the entity)
    # - Multi-word entities (include all words and spaces)

    # Always return valid JSON. If no entities are found, return an empty entities array."""

    system_prompt = f"""## Role
    You are a PII Detection System. Extract entities into a JSON array with the following structure: `{{"entities": [{{"Type": "", "BeginOffset": 0, "EndOffset": 0, "Score": 1.0, "Text": ""}}]}}`.

    ## Standard Entities
    {entity_types_section}

    ## CRITICAL HIERARCHY (Follow in order)
    1. USER OVERRIDES: If the "USER-SPECIFIC CONSTRAINTS" section below contains instructions, they supersede ALL standard rules.
    2. If a user instruction contradicts a standard entity rule (e.g., "Only detect Lauren"), the user instruction is the final authority.
    3. OFFSETS: 0-based index; EndOffset is exclusive.
    4. SCORE: A confidence score between 0.0 and 1.0. 1.0 is the highest confidence score.
    5. TEXT: The exact text substring that was identified.
    6. OUTPUT: Return ONLY valid JSON. No preamble.

    {custom_instructions_section}
"""

    return system_prompt
