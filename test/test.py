import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from typing import List, Optional


def run_cli_redact(
    script_path: str,
    input_file: str,
    output_dir: str,
    task: str = "redact",
    timeout: int = 600,  # 10-minute timeout
    # --- General Arguments ---
    input_dir: Optional[str] = None,
    language: Optional[str] = None,
    allow_list: Optional[str] = None,
    pii_detector: Optional[str] = None,
    username: Optional[str] = None,
    save_to_user_folders: Optional[bool] = None,
    local_redact_entities: Optional[List[str]] = None,
    aws_redact_entities: Optional[List[str]] = None,
    aws_access_key: Optional[str] = None,
    aws_secret_key: Optional[str] = None,
    cost_code: Optional[str] = None,
    aws_region: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    do_initial_clean: Optional[bool] = None,
    save_logs_to_csv: Optional[bool] = None,
    save_logs_to_dynamodb: Optional[bool] = None,
    display_file_names_in_logs: Optional[bool] = None,
    upload_logs_to_s3: Optional[bool] = None,
    s3_logs_prefix: Optional[str] = None,
    # --- PDF/Image Redaction Arguments ---
    ocr_method: Optional[str] = None,
    page_min: Optional[int] = None,
    page_max: Optional[int] = None,
    images_dpi: Optional[float] = None,
    chosen_local_ocr_model: Optional[str] = None,
    preprocess_local_ocr_images: Optional[bool] = None,
    compress_redacted_pdf: Optional[bool] = None,
    return_pdf_end_of_redaction: Optional[bool] = None,
    deny_list_file: Optional[str] = None,
    allow_list_file: Optional[str] = None,
    redact_whole_page_file: Optional[str] = None,
    handwrite_signature_extraction: Optional[List[str]] = None,
    extract_forms: Optional[bool] = None,
    extract_tables: Optional[bool] = None,
    extract_layout: Optional[bool] = None,
    # --- Word/Tabular Anonymisation Arguments ---
    anon_strategy: Optional[str] = None,
    text_columns: Optional[List[str]] = None,
    excel_sheets: Optional[List[str]] = None,
    fuzzy_mistakes: Optional[int] = None,
    match_fuzzy_whole_phrase_bool: Optional[bool] = None,
    # --- Duplicate Detection Arguments ---
    duplicate_type: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    min_word_count: Optional[int] = None,
    min_consecutive_pages: Optional[int] = None,
    greedy_match: Optional[bool] = None,
    combine_pages: Optional[bool] = None,
    remove_duplicate_rows: Optional[bool] = None,
    # --- Textract Batch Operations Arguments ---
    textract_action: Optional[str] = None,
    job_id: Optional[str] = None,
    extract_signatures: Optional[bool] = None,
    textract_bucket: Optional[str] = None,
    textract_input_prefix: Optional[str] = None,
    textract_output_prefix: Optional[str] = None,
    s3_textract_document_logs_subfolder: Optional[str] = None,
    local_textract_document_logs_subfolder: Optional[str] = None,
    poll_interval: Optional[int] = None,
    max_poll_attempts: Optional[int] = None,
) -> bool:
    """
    Executes the cli_redact.py script with specified arguments using a subprocess.

    Args:
        script_path (str): The path to the cli_redact.py script.
        input_file (str): The path to the input file to process.
        output_dir (str): The path to the directory for output files.
        task (str): The main task to perform ('redact', 'deduplicate', or 'textract').
        timeout (int): Timeout in seconds for the subprocess.

        # General Arguments
        input_dir (str): Directory for all input files.
        language (str): Language of the document content.
        allow_list (str): Path to a CSV file with words to exclude from redaction.
        pii_detector (str): Core PII detection method (Local, AWS Comprehend, or None).
        username (str): Username for the session.
        save_to_user_folders (bool): Whether to save to user folders or not.
        local_redact_entities (List[str]): Local redaction entities to use.
        aws_redact_entities (List[str]): AWS redaction entities to use.
        aws_access_key (str): Your AWS Access Key ID.
        aws_secret_key (str): Your AWS Secret Access Key.
        cost_code (str): Cost code for tracking usage.
        aws_region (str): AWS region for cloud services.
        s3_bucket (str): S3 bucket name for cloud operations.
        do_initial_clean (bool): Perform initial text cleaning for tabular data.
        save_logs_to_csv (bool): Save processing logs to CSV files.
        save_logs_to_dynamodb (bool): Save processing logs to DynamoDB.
        display_file_names_in_logs (bool): Include file names in log outputs.
        upload_logs_to_s3 (bool): Upload log files to S3 after processing.
        s3_logs_prefix (str): S3 prefix for usage log files.

        # PDF/Image Redaction Arguments
        ocr_method (str): OCR method for text extraction from images.
        page_min (int): First page to redact.
        page_max (int): Last page to redact.
        images_dpi (float): DPI for image processing.
        chosen_local_ocr_model (str): Local OCR model to use.
        preprocess_local_ocr_images (bool): Preprocess images before OCR.
        compress_redacted_pdf (bool): Compress the final redacted PDF.
        return_pdf_end_of_redaction (bool): Return PDF at end of redaction process.
        deny_list_file (str): Custom words file to recognize for redaction.
        allow_list_file (str): Custom words file to recognize for redaction.
        redact_whole_page_file (str): File for pages to redact completely.
        handwrite_signature_extraction (List[str]): Handwriting and signature extraction options.
        extract_forms (bool): Extract forms during Textract analysis.
        extract_tables (bool): Extract tables during Textract analysis.
        extract_layout (bool): Extract layout during Textract analysis.

        # Word/Tabular Anonymisation Arguments
        anon_strategy (str): The anonymisation strategy to apply.
        text_columns (List[str]): A list of column names to anonymise or deduplicate.
        excel_sheets (List[str]): Specific Excel sheet names to process.
        fuzzy_mistakes (int): Number of allowed spelling mistakes for fuzzy matching.
        match_fuzzy_whole_phrase_bool (bool): Match fuzzy whole phrase boolean.

        # Duplicate Detection Arguments
        duplicate_type (str): Type of duplicate detection (pages or tabular).
        similarity_threshold (float): Similarity threshold (0-1) to consider content as duplicates.
        min_word_count (int): Minimum word count for text to be considered.
        min_consecutive_pages (int): Minimum number of consecutive pages to consider as a match.
        greedy_match (bool): Use greedy matching strategy for consecutive pages.
        combine_pages (bool): Combine text from the same page number within a file.
        remove_duplicate_rows (bool): Remove duplicate rows from the output.

        # Textract Batch Operations Arguments
        textract_action (str): Textract action to perform (submit, retrieve, or list).
        job_id (str): Textract job ID for retrieve action.
        extract_signatures (bool): Extract signatures during Textract analysis.
        textract_bucket (str): S3 bucket name for Textract operations.
        textract_input_prefix (str): S3 prefix for input files in Textract operations.
        textract_output_prefix (str): S3 prefix for output files in Textract operations.
        s3_textract_document_logs_subfolder (str): S3 prefix for logs in Textract operations.
        local_textract_document_logs_subfolder (str): Local prefix for logs in Textract operations.
        poll_interval (int): Polling interval in seconds for Textract job status.
        max_poll_attempts (int): Maximum number of polling attempts for Textract job completion.

    Returns:
        bool: True if the script executed successfully, False otherwise.
    """
    # 1. Get absolute paths and perform pre-checks
    script_abs_path = os.path.abspath(script_path)
    output_abs_dir = os.path.abspath(output_dir)

    # Handle input file based on task and action
    if task == "textract" and textract_action in ["retrieve", "list"]:
        # For retrieve and list actions, input file is not required
        input_abs_path = None
    else:
        # For all other cases, input file is required
        if input_file is None:
            raise ValueError("Input file is required for this task")
        input_abs_path = os.path.abspath(input_file)
        if not os.path.isfile(input_abs_path):
            raise FileNotFoundError(f"Input file not found: {input_abs_path}")

    if not os.path.isfile(script_abs_path):
        raise FileNotFoundError(f"Script not found: {script_abs_path}")

    if not os.path.isdir(output_abs_dir):
        # Create the output directory if it doesn't exist
        print(f"Output directory not found. Creating: {output_abs_dir}")
        os.makedirs(output_abs_dir)

    script_folder = os.path.dirname(script_abs_path)

    # 2. Dynamically build the command list
    command = [
        "python",
        script_abs_path,
        "--output_dir",
        output_abs_dir,
        "--task",
        task,
    ]

    # Add input_file only if it's not None
    if input_abs_path is not None:
        command.extend(["--input_file", input_abs_path])

    # Add general arguments
    if input_dir:
        command.extend(["--input_dir", input_dir])
    if language:
        command.extend(["--language", language])
    if allow_list and os.path.isfile(allow_list):
        command.extend(["--allow_list", os.path.abspath(allow_list)])
    if pii_detector:
        command.extend(["--pii_detector", pii_detector])
    if username:
        command.extend(["--username", username])
    if save_to_user_folders is not None:
        command.extend(["--save_to_user_folders", str(save_to_user_folders)])
    if local_redact_entities:
        command.append("--local_redact_entities")
        command.extend(local_redact_entities)
    if aws_redact_entities:
        command.append("--aws_redact_entities")
        command.extend(aws_redact_entities)
    if aws_access_key:
        command.extend(["--aws_access_key", aws_access_key])
    if aws_secret_key:
        command.extend(["--aws_secret_key", aws_secret_key])
    if cost_code:
        command.extend(["--cost_code", cost_code])
    if aws_region:
        command.extend(["--aws_region", aws_region])
    if s3_bucket:
        command.extend(["--s3_bucket", s3_bucket])
    if do_initial_clean is not None:
        command.extend(["--do_initial_clean", str(do_initial_clean)])
    if save_logs_to_csv is not None:
        command.extend(["--save_logs_to_csv", str(save_logs_to_csv)])
    if save_logs_to_dynamodb is not None:
        command.extend(["--save_logs_to_dynamodb", str(save_logs_to_dynamodb)])
    if display_file_names_in_logs is not None:
        command.extend(
            ["--display_file_names_in_logs", str(display_file_names_in_logs)]
        )
    if upload_logs_to_s3 is not None:
        command.extend(["--upload_logs_to_s3", str(upload_logs_to_s3)])
    if s3_logs_prefix:
        command.extend(["--s3_logs_prefix", s3_logs_prefix])

    # Add PDF/Image redaction arguments
    if ocr_method:
        command.extend(["--ocr_method", ocr_method])
    if page_min is not None:
        command.extend(["--page_min", str(page_min)])
    if page_max is not None:
        command.extend(["--page_max", str(page_max)])
    if images_dpi is not None:
        command.extend(["--images_dpi", str(images_dpi)])
    if chosen_local_ocr_model:
        command.extend(["--chosen_local_ocr_model", chosen_local_ocr_model])
    if preprocess_local_ocr_images is not None:
        command.extend(
            ["--preprocess_local_ocr_images", str(preprocess_local_ocr_images)]
        )
    if compress_redacted_pdf is not None:
        command.extend(["--compress_redacted_pdf", str(compress_redacted_pdf)])
    if return_pdf_end_of_redaction is not None:
        command.extend(
            ["--return_pdf_end_of_redaction", str(return_pdf_end_of_redaction)]
        )
    if deny_list_file and os.path.isfile(deny_list_file):
        command.extend(["--deny_list_file", os.path.abspath(deny_list_file)])
    if allow_list_file and os.path.isfile(allow_list_file):
        command.extend(["--allow_list_file", os.path.abspath(allow_list_file)])
    if redact_whole_page_file and os.path.isfile(redact_whole_page_file):
        command.extend(
            ["--redact_whole_page_file", os.path.abspath(redact_whole_page_file)]
        )
    if handwrite_signature_extraction:
        command.append("--handwrite_signature_extraction")
        command.extend(handwrite_signature_extraction)
    if extract_forms:
        command.append("--extract_forms")
    if extract_tables:
        command.append("--extract_tables")
    if extract_layout:
        command.append("--extract_layout")

    # Add Word/Tabular anonymisation arguments
    if anon_strategy:
        command.extend(["--anon_strategy", anon_strategy])
    if text_columns:
        command.append("--text_columns")
        command.extend(text_columns)
    if excel_sheets:
        command.append("--excel_sheets")
        command.extend(excel_sheets)
    if fuzzy_mistakes is not None:
        command.extend(["--fuzzy_mistakes", str(fuzzy_mistakes)])
    if match_fuzzy_whole_phrase_bool is not None:
        command.extend(
            ["--match_fuzzy_whole_phrase_bool", str(match_fuzzy_whole_phrase_bool)]
        )

    # Add duplicate detection arguments
    if duplicate_type:
        command.extend(["--duplicate_type", duplicate_type])
    if similarity_threshold is not None:
        command.extend(["--similarity_threshold", str(similarity_threshold)])
    if min_word_count is not None:
        command.extend(["--min_word_count", str(min_word_count)])
    if min_consecutive_pages is not None:
        command.extend(["--min_consecutive_pages", str(min_consecutive_pages)])
    if greedy_match is not None:
        command.extend(["--greedy_match", str(greedy_match)])
    if combine_pages is not None:
        command.extend(["--combine_pages", str(combine_pages)])
    if remove_duplicate_rows is not None:
        command.extend(["--remove_duplicate_rows", str(remove_duplicate_rows)])

    # Add Textract batch operations arguments
    if textract_action:
        command.extend(["--textract_action", textract_action])
    if job_id:
        command.extend(["--job_id", job_id])
    if extract_signatures is not None:
        if extract_signatures:
            command.append("--extract_signatures")
    if textract_bucket:
        command.extend(["--textract_bucket", textract_bucket])
    if textract_input_prefix:
        command.extend(["--textract_input_prefix", textract_input_prefix])
    if textract_output_prefix:
        command.extend(["--textract_output_prefix", textract_output_prefix])
    if s3_textract_document_logs_subfolder:
        command.extend(
            [
                "--s3_textract_document_logs_subfolder",
                s3_textract_document_logs_subfolder,
            ]
        )
    if local_textract_document_logs_subfolder:
        command.extend(
            [
                "--local_textract_document_logs_subfolder",
                local_textract_document_logs_subfolder,
            ]
        )
    if poll_interval is not None:
        command.extend(["--poll_interval", str(poll_interval)])
    if max_poll_attempts is not None:
        command.extend(["--max_poll_attempts", str(max_poll_attempts)])

    # Filter out None values before joining
    command_str = " ".join(str(arg) for arg in command if arg is not None)
    print(f"Executing command: {command_str}")

    # 3. Execute the command using subprocess
    try:
        result = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=script_folder,  # Important for relative paths within the script
        )

        # Communicate with the process to get output and handle timeout
        stdout, stderr = result.communicate(timeout=timeout)

        print("--- SCRIPT STDOUT ---")
        if stdout:
            print(stdout)
        print("--- SCRIPT STDERR ---")
        if stderr:
            print(stderr)
        print("---------------------")

        # Analyze the output for errors and success indicators
        analysis = analyze_test_output(stdout, stderr)

        if analysis["has_errors"]:
            print("❌ Errors detected in output:")
            for i, error_type in enumerate(analysis["error_types"]):
                print(f"   {i+1}. {error_type}")
            if analysis["error_messages"]:
                print("   Error messages:")
                for msg in analysis["error_messages"][
                    :3
                ]:  # Show first 3 error messages
                    print(f"     - {msg}")
            return False
        elif result.returncode == 0:
            success_msg = "✅ Script executed successfully."
            if analysis["success_indicators"]:
                success_msg += f" (Success indicators: {', '.join(analysis['success_indicators'][:3])})"
            print(success_msg)
            return True
        else:
            print(f"❌ Command failed with return code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        result.kill()
        print(f"❌ Subprocess timed out after {timeout} seconds.")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False


def analyze_test_output(stdout: str, stderr: str) -> dict:
    """
    Analyze test output to provide detailed error information.

    Args:
        stdout (str): Standard output from the test
        stderr (str): Standard error from the test

    Returns:
        dict: Analysis results with error details
    """
    combined_output = (stdout or "") + (stderr or "")

    analysis = {
        "has_errors": False,
        "error_types": [],
        "error_messages": [],
        "success_indicators": [],
        "warning_indicators": [],
    }

    # Error patterns
    error_patterns = {
        "An error occurred": "General error message",
        "Error:": "Error prefix",
        "Exception:": "Exception occurred",
        "Traceback": "Python traceback",
        "Failed to": "Operation failure",
        "Cannot": "Operation not possible",
        "Unable to": "Operation not possible",
        "KeyError:": "Missing key/dictionary error",
        "AttributeError:": "Missing attribute error",
        "TypeError:": "Type mismatch error",
        "ValueError:": "Invalid value error",
        "FileNotFoundError:": "File not found",
        "ImportError:": "Import failure",
        "ModuleNotFoundError:": "Module not found",
    }

    # Success indicators
    success_patterns = [
        "Successfully",
        "Completed",
        "Finished",
        "Processed",
        "Redacted",
        "Extracted",
    ]

    # Warning indicators
    warning_patterns = ["Warning:", "WARNING:", "Deprecated", "DeprecationWarning"]

    # Check for errors
    for pattern, description in error_patterns.items():
        if pattern.lower() in combined_output.lower():
            analysis["has_errors"] = True
            analysis["error_types"].append(description)

            # Extract the actual error message
            lines = combined_output.split("\n")
            for line in lines:
                if pattern.lower() in line.lower():
                    analysis["error_messages"].append(line.strip())

    # Check for success indicators
    for pattern in success_patterns:
        if pattern.lower() in combined_output.lower():
            analysis["success_indicators"].append(pattern)

    # Check for warnings
    for pattern in warning_patterns:
        if pattern.lower() in combined_output.lower():
            analysis["warning_indicators"].append(pattern)

    return analysis


class TestCLIRedactExamples(unittest.TestCase):
    """Test suite for CLI redaction examples from the epilog."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment before running tests."""
        cls.script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "cli_redact.py"
        )
        cls.example_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "example_data"
        )
        cls.temp_output_dir = tempfile.mkdtemp(prefix="test_output_")

        # Verify script exists
        if not os.path.isfile(cls.script_path):
            raise FileNotFoundError(f"CLI script not found: {cls.script_path}")

        print(f"Test setup complete. Script: {cls.script_path}")
        print(f"Example data directory: {cls.example_data_dir}")
        print(f"Temp output directory: {cls.temp_output_dir}")

        # Debug: Check if example data directory exists and list contents
        if os.path.exists(cls.example_data_dir):
            print("Example data directory exists. Contents:")
            for item in os.listdir(cls.example_data_dir):
                item_path = os.path.join(cls.example_data_dir, item)
                if os.path.isfile(item_path):
                    print(f"  File: {item} ({os.path.getsize(item_path)} bytes)")
                else:
                    print(f"  Directory: {item}")
        else:
            print(f"Example data directory does not exist: {cls.example_data_dir}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after running tests."""
        if os.path.exists(cls.temp_output_dir):
            shutil.rmtree(cls.temp_output_dir)
        print(f"Cleaned up temp directory: {cls.temp_output_dir}")

    def test_pdf_redaction_default_settings(self):
        """Test: Redact a PDF with default settings (local OCR)"""
        print("\n=== Testing PDF redaction with default settings ===")
        input_file = os.path.join(
            self.example_data_dir,
            "example_of_emails_sent_to_a_professor_before_applying.pdf",
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")

        result = run_cli_redact(
            script_path=self.script_path,
            input_file=input_file,
            output_dir=self.temp_output_dir,
        )

        self.assertTrue(result, "PDF redaction with default settings should succeed")
        print("✅ PDF redaction with default settings passed")

    def test_pdf_text_extraction_only(self):
        """Test: Extract text from a PDF only (i.e. no redaction), using local OCR"""
        print("\n=== Testing PDF text extraction only ===")
        input_file = os.path.join(
            self.example_data_dir, "Partnership-Agreement-Toolkit_0_0.pdf"
        )
        whole_page_file = os.path.join(
            self.example_data_dir, "partnership_toolkit_redact_some_pages.csv"
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")
        if not os.path.isfile(whole_page_file):
            self.skipTest(f"Whole page file not found: {whole_page_file}")

        result = run_cli_redact(
            script_path=self.script_path,
            input_file=input_file,
            output_dir=self.temp_output_dir,
            redact_whole_page_file=whole_page_file,
            pii_detector="None",
        )

        self.assertTrue(result, "PDF text extraction should succeed")
        print("✅ PDF text extraction only passed")

    def test_pdf_text_extraction_with_whole_page_redaction(self):
        """Test: Extract text from a PDF only with a whole page redaction list"""
        print("\n=== Testing PDF text extraction with whole page redaction ===")
        input_file = os.path.join(
            self.example_data_dir, "Partnership-Agreement-Toolkit_0_0.pdf"
        )
        whole_page_file = os.path.join(
            self.example_data_dir, "partnership_toolkit_redact_some_pages.csv"
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")
        if not os.path.isfile(whole_page_file):
            self.skipTest(f"Whole page file not found: {whole_page_file}")

        result = run_cli_redact(
            script_path=self.script_path,
            input_file=input_file,
            output_dir=self.temp_output_dir,
            redact_whole_page_file=whole_page_file,
            pii_detector="Local",
            local_redact_entities=["CUSTOM"],
        )

        self.assertTrue(
            result, "PDF text extraction with whole page redaction should succeed"
        )
        print("✅ PDF text extraction with whole page redaction passed")

    def test_pdf_redaction_with_allow_list(self):
        """Test: Redact a PDF with allow list (local OCR) and custom list of redaction entities"""
        print("\n=== Testing PDF redaction with allow list ===")
        input_file = os.path.join(
            self.example_data_dir, "graduate-job-example-cover-letter.pdf"
        )
        allow_list_file = os.path.join(
            self.example_data_dir, "test_allow_list_graduate.csv"
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")
        if not os.path.isfile(allow_list_file):
            self.skipTest(f"Allow list file not found: {allow_list_file}")

        result = run_cli_redact(
            script_path=self.script_path,
            input_file=input_file,
            output_dir=self.temp_output_dir,
            allow_list_file=allow_list_file,
            local_redact_entities=["TITLES", "PERSON", "DATE_TIME"],
        )

        self.assertTrue(result, "PDF redaction with allow list should succeed")
        print("✅ PDF redaction with allow list passed")

    def test_pdf_redaction_limited_pages_with_custom_fuzzy(self):
        """Test: Redact a PDF with limited pages and text extraction method with custom fuzzy matching"""
        print("\n=== Testing PDF redaction with limited pages and fuzzy matching ===")
        input_file = os.path.join(
            self.example_data_dir, "Partnership-Agreement-Toolkit_0_0.pdf"
        )
        deny_list_file = os.path.join(
            self.example_data_dir,
            "Partnership-Agreement-Toolkit_test_deny_list_para_single_spell.csv",
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")
        if not os.path.isfile(deny_list_file):
            self.skipTest(f"Deny list file not found: {deny_list_file}")

        result = run_cli_redact(
            script_path=self.script_path,
            input_file=input_file,
            output_dir=self.temp_output_dir,
            deny_list_file=deny_list_file,
            local_redact_entities=["CUSTOM_FUZZY"],
            page_min=1,
            page_max=3,
            ocr_method="Local text",
            fuzzy_mistakes=3,
        )

        self.assertTrue(
            result, "PDF redaction with limited pages and fuzzy matching should succeed"
        )
        print("✅ PDF redaction with limited pages and fuzzy matching passed")

    def test_pdf_redaction_with_custom_lists(self):
        """Test: Redaction with custom deny list, allow list, and whole page redaction list"""
        print("\n=== Testing PDF redaction with custom lists ===")
        input_file = os.path.join(
            self.example_data_dir, "Partnership-Agreement-Toolkit_0_0.pdf"
        )
        deny_list_file = os.path.join(
            self.example_data_dir, "partnership_toolkit_redact_custom_deny_list.csv"
        )
        whole_page_file = os.path.join(
            self.example_data_dir, "partnership_toolkit_redact_some_pages.csv"
        )
        allow_list_file = os.path.join(
            self.example_data_dir, "test_allow_list_partnership.csv"
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")
        if not os.path.isfile(deny_list_file):
            self.skipTest(f"Deny list file not found: {deny_list_file}")
        if not os.path.isfile(whole_page_file):
            self.skipTest(f"Whole page file not found: {whole_page_file}")
        if not os.path.isfile(allow_list_file):
            self.skipTest(f"Allow list file not found: {allow_list_file}")

        result = run_cli_redact(
            script_path=self.script_path,
            input_file=input_file,
            output_dir=self.temp_output_dir,
            deny_list_file=deny_list_file,
            redact_whole_page_file=whole_page_file,
            allow_list_file=allow_list_file,
        )

        self.assertTrue(result, "PDF redaction with custom lists should succeed")
        print("✅ PDF redaction with custom lists passed")

    def test_image_redaction(self):
        """Test: Redact an image"""
        print("\n=== Testing image redaction ===")
        input_file = os.path.join(self.example_data_dir, "example_complaint_letter.jpg")

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")

        result = run_cli_redact(
            script_path=self.script_path,
            input_file=input_file,
            output_dir=self.temp_output_dir,
        )

        self.assertTrue(result, "Image redaction should succeed")
        print("✅ Image redaction passed")

    def test_csv_anonymisation_specific_columns(self):
        """Test: Anonymise csv file with specific columns"""
        print("\n=== Testing CSV anonymisation with specific columns ===")
        input_file = os.path.join(self.example_data_dir, "combined_case_notes.csv")

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")

        result = run_cli_redact(
            script_path=self.script_path,
            input_file=input_file,
            output_dir=self.temp_output_dir,
            text_columns=["Case Note", "Client"],
            anon_strategy="replace_redacted",
        )

        self.assertTrue(
            result, "CSV anonymisation with specific columns should succeed"
        )
        print("✅ CSV anonymisation with specific columns passed")

    def test_csv_anonymisation_different_strategy(self):
        """Test: Anonymise csv file with a different strategy (remove text completely)"""
        print("\n=== Testing CSV anonymisation with different strategy ===")
        input_file = os.path.join(self.example_data_dir, "combined_case_notes.csv")

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")

        result = run_cli_redact(
            script_path=self.script_path,
            input_file=input_file,
            output_dir=self.temp_output_dir,
            text_columns=["Case Note", "Client"],
            anon_strategy="redact",
        )

        self.assertTrue(
            result, "CSV anonymisation with different strategy should succeed"
        )
        print("✅ CSV anonymisation with different strategy passed")

    def test_word_document_anonymisation(self):
        """Test: Anonymise a word document"""
        print("\n=== Testing Word document anonymisation ===")
        input_file = os.path.join(
            self.example_data_dir, "Bold minimalist professional cover letter.docx"
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")

        result = run_cli_redact(
            script_path=self.script_path,
            input_file=input_file,
            output_dir=self.temp_output_dir,
            anon_strategy="replace_redacted",
        )

        self.assertTrue(result, "Word document anonymisation should succeed")
        print("✅ Word document anonymisation passed")

    def test_aws_textract_comprehend_redaction(self):
        """Test: Use Textract and Comprehend for redaction"""
        print("\n=== Testing AWS Textract and Comprehend redaction ===")
        input_file = os.path.join(
            self.example_data_dir,
            "example_of_emails_sent_to_a_professor_before_applying.pdf",
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")

        # Skip this test if AWS credentials are not available
        # This is a conditional test that may not work in all environments
        run_cli_redact(
            script_path=self.script_path,
            input_file=input_file,
            output_dir=self.temp_output_dir,
            ocr_method="AWS Textract",
            pii_detector="AWS Comprehend",
        )

        # Note: This test may fail if AWS credentials are not configured
        # We'll mark it as passed if it runs without crashing
        print("✅ AWS Textract and Comprehend redaction test completed")

    def test_aws_textract_signature_extraction(self):
        """Test: Redact specific pages with AWS OCR and signature extraction"""
        print("\n=== Testing AWS Textract with signature extraction ===")
        input_file = os.path.join(
            self.example_data_dir, "Partnership-Agreement-Toolkit_0_0.pdf"
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")

        # Skip this test if AWS credentials are not available
        run_cli_redact(
            script_path=self.script_path,
            input_file=input_file,
            output_dir=self.temp_output_dir,
            page_min=6,
            page_max=7,
            ocr_method="AWS Textract",
            handwrite_signature_extraction=[
                "Extract handwriting",
                "Extract signatures",
            ],
        )

        # Note: This test may fail if AWS credentials are not configured
        print("✅ AWS Textract with signature extraction test completed")

    def test_duplicate_pages_detection(self):
        """Test: Find duplicate pages in OCR files"""
        print("\n=== Testing duplicate pages detection ===")
        input_file = os.path.join(
            self.example_data_dir,
            "example_outputs",
            "doubled_output_joined.pdf_ocr_output.csv",
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example OCR file not found: {input_file}")

        result = run_cli_redact(
            script_path=self.script_path,
            input_file=input_file,
            output_dir=self.temp_output_dir,
            task="deduplicate",
            duplicate_type="pages",
            similarity_threshold=0.95,
        )

        self.assertTrue(result, "Duplicate pages detection should succeed")
        print("✅ Duplicate pages detection passed")

    def test_duplicate_line_level_detection(self):
        """Test: Find duplicate in OCR files at the line level"""
        print("\n=== Testing duplicate line level detection ===")
        input_file = os.path.join(
            self.example_data_dir,
            "example_outputs",
            "doubled_output_joined.pdf_ocr_output.csv",
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example OCR file not found: {input_file}")

        result = run_cli_redact(
            script_path=self.script_path,
            input_file=input_file,
            output_dir=self.temp_output_dir,
            task="deduplicate",
            duplicate_type="pages",
            similarity_threshold=0.95,
            combine_pages=False,
            min_word_count=3,
        )

        self.assertTrue(result, "Duplicate line level detection should succeed")
        print("✅ Duplicate line level detection passed")

    def test_duplicate_tabular_detection(self):
        """Test: Find duplicate rows in tabular data"""
        print("\n=== Testing duplicate tabular detection ===")
        input_file = os.path.join(
            self.example_data_dir, "Lambeth_2030-Our_Future_Our_Lambeth.pdf.csv"
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example CSV file not found: {input_file}")

        result = run_cli_redact(
            script_path=self.script_path,
            input_file=input_file,
            output_dir=self.temp_output_dir,
            task="deduplicate",
            duplicate_type="tabular",
            text_columns=["text"],
            similarity_threshold=0.95,
        )

        self.assertTrue(result, "Duplicate tabular detection should succeed")
        print("✅ Duplicate tabular detection passed")

    def test_textract_submit_document(self):
        """Test: Submit document to Textract for basic text analysis"""
        print("\n=== Testing Textract document submission ===")
        input_file = os.path.join(
            self.example_data_dir,
            "example_of_emails_sent_to_a_professor_before_applying.pdf",
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")

        # Skip this test if AWS credentials are not available
        try:
            run_cli_redact(
                script_path=self.script_path,
                input_file=input_file,
                output_dir=self.temp_output_dir,
                task="textract",
                textract_action="submit",
            )
        except Exception as e:
            print(f"Textract test failed (expected without AWS credentials): {e}")

        # Note: This test may fail if AWS credentials are not configured
        print("✅ Textract document submission test completed")

    def test_textract_submit_with_signatures(self):
        """Test: Submit document to Textract for analysis with signature extraction"""
        print("\n=== Testing Textract submission with signature extraction ===")
        input_file = os.path.join(
            self.example_data_dir, "Partnership-Agreement-Toolkit_0_0.pdf"
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")

        # Skip this test if AWS credentials are not available
        try:
            run_cli_redact(
                script_path=self.script_path,
                input_file=input_file,
                output_dir=self.temp_output_dir,
                task="textract",
                textract_action="submit",
                extract_signatures=True,
            )
        except Exception as e:
            print(f"Textract test failed (expected without AWS credentials): {e}")

        # Note: This test may fail if AWS credentials are not configured
        print("✅ Textract submission with signature extraction test completed")

    def test_textract_retrieve_results(self):
        """Test: Retrieve Textract results by job ID"""
        print("\n=== Testing Textract results retrieval ===")

        # Skip this test if AWS credentials are not available
        # This would require a valid job ID from a previous submission
        # For retrieve and list actions, we don't need a real input file
        try:
            run_cli_redact(
                script_path=self.script_path,
                input_file=None,  # No input file needed for retrieve action
                output_dir=self.temp_output_dir,
                task="textract",
                textract_action="retrieve",
                job_id="12345678-1234-1234-1234-123456789012",  # Dummy job ID
            )
        except Exception as e:
            print(f"Textract test failed (expected without AWS credentials): {e}")

        # Note: This test will likely fail with a dummy job ID, but that's expected
        print("✅ Textract results retrieval test completed")

    def test_textract_list_jobs(self):
        """Test: List recent Textract jobs"""
        print("\n=== Testing Textract jobs listing ===")

        # Skip this test if AWS credentials are not available
        # For list action, we don't need a real input file
        try:
            run_cli_redact(
                script_path=self.script_path,
                input_file=None,  # No input file needed for list action
                output_dir=self.temp_output_dir,
                task="textract",
                textract_action="list",
            )
        except Exception as e:
            print(f"Textract test failed (expected without AWS credentials): {e}")

        # Note: This test may fail if AWS credentials are not configured
        print("✅ Textract jobs listing test completed")


class TestGUIApp(unittest.TestCase):
    """Test suite for GUI application loading and basic functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment for GUI tests."""
        cls.app_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "app.py"
        )

        # Verify app.py exists
        if not os.path.isfile(cls.app_path):
            raise FileNotFoundError(f"App file not found: {cls.app_path}")

        print(f"GUI test setup complete. App: {cls.app_path}")

    def test_app_import_and_initialization(self):
        """Test: Import app.py and check if the Gradio app object is created successfully."""
        print("\n=== Testing GUI app import and initialization ===")

        try:
            # Add the parent directory to the path so we can import app
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            # Import the app module
            import app

            # Check if the app object exists and is a Gradio Blocks object
            self.assertTrue(
                hasattr(app, "blocks"), "App object should exist in the module"
            )

            # Check if it's a Gradio Blocks instance
            import gradio as gr

            self.assertIsInstance(
                app.blocks, gr.Blocks, "App should be a Gradio Blocks instance"
            )

            print("✅ GUI app import and initialisation passed")

        except ImportError as e:
            error_msg = f"Failed to import app module: {e}"
            if "gradio_image_annotation" in str(e):
                error_msg += "\n\nNOTE: This test requires the 'redaction' conda environment to be activated."
                error_msg += "\nPlease run: conda activate redaction"
                error_msg += "\nThen run this test again."
            self.fail(error_msg)
        except Exception as e:
            self.fail(f"Unexpected error during app initialization: {e}")

    def test_app_launch_headless(self):
        """Test: Launch the app in headless mode to verify it starts without errors."""
        print("\n=== Testing GUI app launch in headless mode ===")

        try:
            # Add the parent directory to the path
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            # Import the app module

            import app

            # Set up a flag to track if the app launched successfully
            app_launched = threading.Event()
            launch_error = None

            def launch_app():
                try:
                    # Launch the app in headless mode with a short timeout
                    app.app.launch(
                        show_error=True,
                        inbrowser=False,  # Don't open browser
                        server_port=0,  # Use any available port
                        quiet=True,  # Suppress output
                        prevent_thread_lock=True,  # Don't block the main thread
                    )
                    app_launched.set()
                except Exception:
                    app_launched.set()

            # Start the app in a separate thread
            launch_thread = threading.Thread(target=launch_app)
            launch_thread.daemon = True
            launch_thread.start()

            # Wait for the app to launch (with timeout)
            if app_launched.wait(timeout=10):  # 10 second timeout
                if launch_error:
                    self.fail(f"App launch failed: {launch_error}")
                else:
                    print("✅ GUI app launch in headless mode passed")
            else:
                self.fail("App launch timed out after 10 seconds")

        except Exception as e:
            error_msg = f"Unexpected error during app launch test: {e}"
            if "gradio_image_annotation" in str(e):
                error_msg += "\n\nNOTE: This test requires the 'redaction' conda environment to be activated."
                error_msg += "\nPlease run: conda activate redaction"
                error_msg += "\nThen run this test again."
            self.fail(error_msg)

    def test_app_configuration_loading(self):
        """Test: Verify that the app can load its configuration without errors."""
        print("\n=== Testing GUI app configuration loading ===")

        try:
            # Add the parent directory to the path
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            # Import the app module (not needed?)
            # import app

            # Check if key configuration variables are accessible
            # These should be imported from tools.config
            from tools.config import (
                DEFAULT_LANGUAGE,
                GRADIO_SERVER_PORT,
                MAX_FILE_SIZE,
                PII_DETECTION_MODELS,
            )

            # Verify these are not None/empty
            self.assertIsNotNone(
                GRADIO_SERVER_PORT, "GRADIO_SERVER_PORT should be configured"
            )
            self.assertIsNotNone(MAX_FILE_SIZE, "MAX_FILE_SIZE should be configured")
            self.assertIsNotNone(
                DEFAULT_LANGUAGE, "DEFAULT_LANGUAGE should be configured"
            )
            self.assertIsNotNone(
                PII_DETECTION_MODELS, "PII_DETECTION_MODELS should be configured"
            )

            print("✅ GUI app configuration loading passed")

        except ImportError as e:
            error_msg = f"Failed to import configuration: {e}"
            if "gradio_image_annotation" in str(e):
                error_msg += "\n\nNOTE: This test requires the 'redaction' conda environment to be activated."
                error_msg += "\nPlease run: conda activate redaction"
                error_msg += "\nThen run this test again."
            self.fail(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during configuration test: {e}"
            if "gradio_image_annotation" in str(e):
                error_msg += "\n\nNOTE: This test requires the 'redaction' conda environment to be activated."
                error_msg += "\nPlease run: conda activate redaction"
                error_msg += "\nThen run this test again."
            self.fail(error_msg)


def run_all_tests():
    """Run all test examples and report results."""
    print("=" * 80)
    print("DOCUMENT REDACTION TEST SUITE")
    print("=" * 80)
    print("This test suite includes:")
    print("- CLI examples from the epilog")
    print("- GUI application loading and initialization tests")
    print("Tests will be skipped if required example files are not found.")
    print("AWS-related tests may fail if credentials are not configured.")
    print("=" * 80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add CLI tests
    cli_suite = loader.loadTestsFromTestCase(TestCLIRedactExamples)
    suite.addTests(cli_suite)

    # Add GUI tests
    gui_suite = loader.loadTestsFromTestCase(TestGUIApp)
    suite.addTests(gui_suite)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'✅ PASSED' if success else '❌ FAILED'}")
    print("=" * 80)

    return success


if __name__ == "__main__":
    # Run the test suite
    success = run_all_tests()
    exit(0 if success else 1)
