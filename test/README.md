# CLI Redaction Test Suite

This test suite provides comprehensive testing for the `cli_redact.py` script based on all the examples shown in the CLI epilog.

## Overview

The test suite includes tests for:

1. **PDF Redaction Examples**
   - Default settings (local OCR)
   - Text extraction only (no redaction)
   - Text extraction with whole page redaction
   - Redaction with allow lists
   - Limited pages with custom fuzzy matching
   - Custom deny/allow/whole page lists
   - Image redaction

2. **Tabular Anonymisation Examples**
   - CSV anonymisation with specific columns
   - Different anonymisation strategies
   - Word document anonymisation

3. **AWS Services Examples**
   - Textract and Comprehend redaction
   - Signature extraction
   - Layout extraction

4. **Duplicate Detection Examples**
   - Duplicate pages in OCR files
   - Line-level duplicate detection
   - Tabular duplicate detection

5. **Textract Batch Operations**
   - Submit documents for analysis
   - Retrieve results by job ID
   - List recent jobs

## Running the Tests

### Method 1: Run the test suite directly
```bash
cd test
python test.py
```

### Method 2: Use the convenience script
```bash
cd test
python run_tests.py
```

### Method 3: Run with unittest
```bash
cd test
python -m unittest test.test.TestCLIRedactExamples -v
```

## Test Behavior

- **File Dependencies**: Tests will be skipped if required example files are not found in the `example_data/` directory
- **AWS Tests**: AWS-related tests may fail if credentials are not configured, but this is expected
- **Temporary Output**: All tests use temporary output directories that are cleaned up automatically
- **Timeout**: Each test has a 10-minute timeout to prevent hanging

## Test Structure

The test suite uses Python's `unittest` framework with the following structure:

- `TestCLIRedactExamples`: Main test class containing all test methods
- `run_cli_redact()`: Helper function that executes the CLI script with specified parameters
- `run_all_tests()`: Main function that runs all tests and provides a summary

## Example Output

```
================================================================================
DOCUMENT REDACTION CLI TEST SUITE
================================================================================
This test suite runs through all the examples from the CLI epilog.
Tests will be skipped if required example files are not found.
AWS-related tests may fail if credentials are not configured.
================================================================================

Test setup complete. Script: /path/to/cli_redact.py
Example data directory: /path/to/example_data
Temp output directory: /tmp/test_output_xyz

=== Testing PDF redaction with default settings ===
✅ PDF redaction with default settings passed

=== Testing PDF text extraction only ===
✅ PDF text extraction only passed

...

================================================================================
TEST SUMMARY
================================================================================
Tests run: 20
Failures: 0
Errors: 0
Skipped: 2

Overall result: ✅ PASSED
================================================================================
```

## Requirements

- Python 3.6+
- All dependencies for the main CLI script
- Example data files in the `example_data/` directory (for full test coverage)
- AWS credentials (for AWS-related tests)

## Notes

- Tests are designed to be robust and will skip gracefully if files are missing
- AWS tests are marked as completed even if they fail due to missing credentials
- The test suite provides detailed output for debugging
- All temporary files are cleaned up automatically
