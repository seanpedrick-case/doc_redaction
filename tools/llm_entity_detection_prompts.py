"""
Prompts for LLM-based entity detection.
These prompts are designed to extract PII entities with character positions.
Text passed to the LLM is chunked using the same limits as AWS Comprehend batching:
up to BATCH_CHAR_LIMIT characters or BATCH_WORD_LIMIT words, with boundaries at
phrase-ending punctuation, newlines, or end of page (never mid-sentence).
"""

# Must match custom_image_analyser_engine.DEFAULT_NEW_BATCH_* so prompts describe actual chunking
BATCH_CHAR_LIMIT = 1000
BATCH_WORD_LIMIT = 200


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
        entities_to_detect: List of entity types to detect (e.g., ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON_NAME"])
        language: Language code (e.g., "en", "es", "fr")
        custom_instructions: Optional custom instructions to include in the prompt (e.g., "don't redact anything related to Mark Wilson")

    Returns:
        Formatted prompt string
    """

    prompt = f"""### User prompt
Analyse the following text according to the provided instructions in the system prompt.
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
    # Ensure custom_instructions is a string (callers may pass bool or other types).
    # Treat boolean True and the string "True" as empty (e.g. from an unchecked/empty Gradio box).
    if not isinstance(custom_instructions, str):
        custom_instructions = ""
    elif custom_instructions.strip().lower() == "true":
        custom_instructions = ""

    # Filter out CUSTOM_VLM_* entities (these are handled separately via VLM)
    filtered_entities = [
        entity for entity in entities_to_detect if not entity.startswith("CUSTOM_VLM_")
    ]

    custom_instructions_section = ""

    if custom_instructions and custom_instructions.strip():
        custom_instructions_section = (
            f"\n## ADDITIONAL USER INSTRUCTIONS:\n{custom_instructions.strip()}\n"
        )
    else:
        custom_instructions_section = "No specific user constraints provided."

    # Handle case where no standard entities are selected
    if filtered_entities:
        entity_types_section = f"Standard entity types to detect: {filtered_entities}"
    else:
        entity_types_section = "No standard entity types selected - analyse text based on ADDITIONAL USER INSTRUCTIONS provided."
        if not custom_instructions or not custom_instructions.strip():
            raise ValueError(
                "No standard entities selected and no custom instructions provided. "
                "Please select at least one entity type or provide custom instructions for LLM-based PII detection."
            )

    system_prompt = f"""# System Prompt
You are a personal information detection system. Extract entities from the standard entity list, according to the INSTRUCTIONS HIERARCHY rules below, into a JSON array with the following structure: `{{"entities": [{{"Type": "", "BeginOffset": 0, "EndOffset": 0, "Score": 1.0, "Text": ""}}]}}`.

## Standard entity list
{entity_types_section}
{custom_instructions_section}
## INSTRUCTIONS HIERARCHY (Follow in order)
1. Use the standard entity list as the baseline list of entities to analyse the text.
2. ADDITIONAL USER INSTRUCTIONS (if available): these provide additional instructions for the analysis, and override the standard entity list if they contradict it. Users may suggest new entity types to identify - they may refer to them as labels, redactions, entity types, or other similar terms. Be sure to follow each instruction closely.
3. If a USER INSTRUCTION contradicts a standard entity rule, the user instruction is the final authority. For example, with the USER INSTRUCTION "Do not redact information related to John", exclude all entities where the text mentions or is related to John in the output.
4. If text could be assigned to multiple entity types then assign it to all relevant entity types in separate JSON entries, this includes entity types in the standard entity list or the ADDITIONAL USER INSTRUCTIONS.
4. OFFSETS: The text position of the start and end of the text for a given entity type. A 0-based index.
5. SCORE: A confidence score between 0.0 and 1.0. 1.0 is the highest confidence score.
6. TEXT: The exact text substring that was identified.
7. Return every relevant instance of a valid entity type after taking into account the ADDITIONAL USER INSTRUCTIONS, no matter how many times they appear in the text.
8. OUTPUT: Return ONLY valid JSON. No additional text or commentary.
"""

    return system_prompt
