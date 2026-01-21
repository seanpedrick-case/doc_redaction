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
    entities_list = ", ".join(entities_to_detect)

    custom_instructions_section = ""
    if custom_instructions and custom_instructions.strip():
        custom_instructions_section = (
            f"\n\nIMPORTANT CUSTOM INSTRUCTIONS:\n{custom_instructions.strip()}\n"
        )

    prompt = f"""You are an expert at identifying Personally Identifiable Information (PII) in text. Your task is to analyse the provided text and identify all instances of the following entity types/labels: {entities_list}.{custom_instructions_section}

IMPORTANT: You must return your response as a valid JSON object with the following structure:
{{
  "entities": [
    {{
      "Type": "ENTITY_TYPE",
      "BeginOffset": <start_character_position>,
      "EndOffset": <end_character_position>,
      "Score": <confidence_score_0_to_1>,
      "Text": "<the_actual_text_found>"
    }}
  ]
}}

Rules:
1. Character positions (BeginOffset and EndOffset) must be exact character indices in the original text (0-based indexing)
2. BeginOffset is the position of the first character of the entity/label
3. EndOffset is the position AFTER the last character of the entity/label (exclusive, like Python string slicing)
4. Score should be a decimal between 0.0 and 1.0 representing your confidence
5. Text should be the exact substring from the original text
6. Only return entities/labels that match the requested types/labels: {entities_list}
7. If no entities/labels are found, return: {{"entities": []}}
8. The JSON must be valid and parseable - do not include any explanatory text outside the JSON
9. If IMPORTANT CUSTOM INSTRUCTIONS are provided, follow them carefully. They override all other instructions.

Text to analyse:
{text}

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
    ", ".join(entities_to_detect)

    custom_instructions_section = ""
    if custom_instructions and custom_instructions.strip():
        custom_instructions_section = (
            f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_instructions.strip()}\n"
        )

    system_prompt = f"""You are a precise PII (Personally Identifiable Information) detection system. Your role is to identify specific entity types in text and return them in a structured JSON format with exact character positions.{custom_instructions_section}

Entity types to detect: {entities_to_detect}

For each entity found, you must provide:
- Type: The entity type (must match one of the requested types)
- BeginOffset: The starting character position (0-based index)
- EndOffset: The ending character position (exclusive, like Python string slicing)
- Score: Confidence score between 0.0 and 1.0
- Text: The exact text substring that was identified

Be precise with character positions - they must match the exact location in the original text. Handle edge cases like:
- Entities at the start of text (BeginOffset = 0)
- Entities at the end of text (EndOffset = text length)
- Entities with punctuation (include punctuation if it's part of the entity)
- Multi-word entities (include all words and spaces)

Always return valid JSON. If no entities are found, return an empty entities array."""

    return system_prompt
