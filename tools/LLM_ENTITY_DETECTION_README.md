# LLM-Based Entity Detection Implementation

## Overview

This implementation adds LLM-based entity detection as an alternative to AWS Comprehend for identifying PII (Personally Identifiable Information) entities in text. The system uses AWS Bedrock to call LLM models (such as Claude) to detect entities with character positions, maintaining compatibility with the existing entity detection pipeline.

## Files Created

1. **`tools/llm_entity_detection_prompts.py`**
   - Contains prompt templates for LLM-based entity detection
   - Creates system and user prompts that instruct the LLM to return JSON with entity positions

2. **`tools/llm_entity_detection.py`**
   - Main module for LLM-based entity detection
   - Functions:
     - `call_llm_for_entity_detection()`: Calls AWS Bedrock to detect entities
     - `parse_llm_entity_response()`: Parses LLM JSON response
     - `map_back_llm_entity_results()`: Maps entities back to line-level results
     - `do_llm_entity_detection_call()`: Main function for batch processing (similar to `do_aws_comprehend_call`)

## Files Modified

1. **`tools/config.py`**
   - Added `AWS_LLM_PII_OPTION` constant: `"LLM (AWS Bedrock)"`
   - Added `SHOW_TRANSFORMERS_LLM_PII_DETECTION_OPTIONS` config flag (default: `False`)
   - Added LLM option to `aws_model_options` when enabled

2. **`tools/custom_image_analyser_engine.py`**
   - Updated `analyze_text()` method to support `AWS_LLM_PII_OPTION`
   - Added `bedrock_runtime` and `model_choice` parameters
   - Implemented LLM-based detection branch with same batching logic as AWS Comprehend

3. **`tools/file_redaction.py`**
   - Added `bedrock_runtime` client creation (similar to `comprehend_client`)
   - Updated `analyze_text()` call to pass `bedrock_runtime` parameter

## How It Works

### 1. Prompt Engineering

The LLM is instructed to:
- Analyze text for specific entity types
- Return results in JSON format with:
  - `Type`: Entity type (e.g., "EMAIL", "PHONE_NUMBER")
  - `BeginOffset`: Starting character position (0-based)
  - `EndOffset`: Ending character position (exclusive)
  - `Score`: Confidence score (0.0 to 1.0)
  - `Text`: The actual text found

### 2. Batching

The implementation uses the same batching logic as AWS Comprehend:
- Batches are created when:
  - Word count reaches 50 words, OR
  - Character count reaches 200 characters
- This ensures efficient processing and stays within LLM context limits

### 3. Response Parsing

The LLM response is parsed to:
- Extract JSON from the response (handles markdown code blocks)
- Validate character positions against original text
- Convert to `RecognizerResult` format compatible with existing pipeline

### 4. Mapping Back to Lines

Entities are mapped back to line-level results using the same logic as AWS Comprehend:
- Batch positions are mapped to line indices
- Character offsets are adjusted relative to each line
- Results are stored in the same format as other detection methods

## Configuration

### Enable LLM-Based Detection

1. Set environment variable or config:
   ```python
   SHOW_TRANSFORMERS_LLM_PII_DETECTION_OPTIONS = "True"
   ```

2. Select "LLM (AWS Bedrock)" as the PII identification method in the UI

### Model Selection

Default model: `anthropic.claude-3-5-sonnet-20241022-v2:0`

You can customize the model by passing `model_choice` parameter to `analyze_text()`:
```python
analyze_text(
    ...,
    model_choice="anthropic.claude-3-haiku-20240307-v1:0",  # Faster, cheaper model
    ...
)
```

### Temperature and Max Tokens

You can customize LLM generation parameters via `text_analyzer_kwargs`:
```python
analyze_text(
    ...,
    temperature=0.1,  # Lower = more deterministic
    max_tokens=4096,  # Maximum response length
    ...
)
```

## Dependencies

The implementation requires `tools/llm_funcs.py` to be present in the local `tools` folder. This file has been copied from the `llm_topic_modeller` project and adapted to work with the local project's configuration.

The `llm_funcs.py` file includes default values for any missing configuration variables, so it will work even if some LLM-related config variables are not defined in your local `tools/config.py`.

## AWS Bedrock Setup

1. Ensure AWS credentials are configured (same as for AWS Comprehend/Textract)
2. Ensure Bedrock access is enabled in your AWS account
3. Ensure the selected model is available in your region

## Usage Example

```python
from tools.custom_image_analyser_engine import CustomImageAnalyzerEngine
import boto3

# Create analyzer
analyzer = CustomImageAnalyzerEngine()

# Create Bedrock client
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# Analyze text with LLM
results, query_count = analyzer.analyze_text(
    line_level_ocr_results=ocr_results,
    ocr_results_with_words=word_results,
    chosen_redact_comprehend_entities=["EMAIL", "PHONE_NUMBER", "NAME"],
    pii_identification_method="LLM (AWS Bedrock)",
    bedrock_runtime=bedrock_runtime,
    model_choice="anthropic.claude-3-5-sonnet-20241022-v2:0",
    language="en",
    temperature=0.1,
    max_tokens=4096,
)
```

## Advantages Over AWS Comprehend

1. **Flexibility**: Can use different models (Claude, Llama, etc.)
2. **Customization**: Can adjust temperature, max tokens, and other parameters
3. **Extensibility**: Easy to add custom entity types or modify prompts
4. **Cost Control**: Can choose cheaper/faster models for different use cases

## Limitations

1. **Cost**: LLM API calls may be more expensive than AWS Comprehend
2. **Latency**: May be slower than AWS Comprehend for large batches
3. **Accuracy**: Depends on the LLM model and prompt quality
4. **JSON Parsing**: Relies on LLM returning valid JSON (with error handling)

## Error Handling

- Invalid JSON responses are caught and logged
- Invalid character positions are validated against text length
- Retry logic (3 attempts) handles transient failures
- Falls back gracefully if LLM functions are not available

## Future Enhancements

Potential improvements:
1. Support for other LLM providers (OpenAI, Azure, etc.)
2. Prompt optimization based on entity types
3. Caching of common entity patterns
4. Batch optimization for better throughput
5. Confidence score calibration
