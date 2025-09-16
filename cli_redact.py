import argparse
import os
import pandas as pd
from tools.config import get_or_create_env_var, LOCAL_PII_OPTION, AWS_PII_OPTION, SELECTABLE_TEXT_EXTRACT_OPTION, TESSERACT_TEXT_EXTRACT_OPTION, TEXTRACT_TEXT_EXTRACT_OPTION, INPUT_FOLDER, OUTPUT_FOLDER, DEFAULT_LANGUAGE, CHOSEN_COMPREHEND_ENTITIES, FULL_COMPREHEND_ENTITY_LIST, CHOSEN_REDACT_ENTITIES, FULL_ENTITY_LIST
from tools.helper_functions import ensure_output_folder_exists
from tools.file_conversion import prepare_image_or_pdf
from tools.file_redaction import choose_and_run_redactor
from tools.data_anonymise import anonymise_files_with_open_text
from tools.helper_functions import _get_env_list
from tools.load_spacy_model_custom_recognisers import custom_entities
from tools.find_duplicate_pages import run_duplicate_analysis, run_full_search_and_analysis
from tools.find_duplicate_tabular import run_tabular_duplicate_analysis

# --- Constants and Configuration ---

if CHOSEN_COMPREHEND_ENTITIES: CHOSEN_COMPREHEND_ENTITIES = _get_env_list(CHOSEN_COMPREHEND_ENTITIES)
if FULL_COMPREHEND_ENTITY_LIST: FULL_COMPREHEND_ENTITY_LIST = _get_env_list(FULL_COMPREHEND_ENTITY_LIST)
if CHOSEN_REDACT_ENTITIES: CHOSEN_REDACT_ENTITIES = _get_env_list(CHOSEN_REDACT_ENTITIES)
if FULL_ENTITY_LIST: FULL_ENTITY_LIST = _get_env_list(FULL_ENTITY_LIST)

# Add custom spacy recognisers to the Comprehend list, so that local Spacy model can be used to pick up e.g. titles, streetnames, UK postcodes that are sometimes missed by comprehend
CHOSEN_COMPREHEND_ENTITIES.extend(custom_entities)
FULL_COMPREHEND_ENTITY_LIST.extend(custom_entities)

chosen_redact_entities = CHOSEN_REDACT_ENTITIES
full_entity_list = FULL_ENTITY_LIST
chosen_comprehend_entities = CHOSEN_COMPREHEND_ENTITIES
full_comprehend_entity_list = FULL_COMPREHEND_ENTITY_LIST

# --- Main CLI Function ---
def main(direct_mode_args=None):
    """
    A unified command-line interface to prepare, redact, and anonymise various document types.
    
    Args:
        direct_mode_args (dict, optional): Dictionary of arguments for direct mode execution.
                                          If provided, uses these instead of parsing command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='A versatile CLI for redacting PII from PDF/image files and anonymising Word/tabular data.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='''
Examples:
  # Redact a PDF with default settings:
  python cli_redact.py --input_file document.pdf

  # Redact specific pages with custom OCR:
  python cli_redact.py --input_file document.pdf --page_min 1 --page_max 10 --ocr_method "AWS Textract service - all PDF types"

  # Anonymize Excel file with specific columns:
  python cli_redact.py --input_file data.xlsx --columns "Name" "Email" --anon_strat "replace with 'REDACTED'"

  # Use AWS services with custom settings:
  python cli_redact.py --input_file document.pdf --pii_detector "AWS Comprehend" --aws_access_key YOUR_KEY --aws_secret_key YOUR_SECRET

  # Advanced redaction with custom word list:
  python cli_redact.py --input_file document.pdf --in_deny_list "CompanyName" "ProjectCode" --deny_list custom_terms.csv

  # Find duplicate pages in OCR files:
  python cli_redact.py --task deduplicate --input_file ocr_output.csv --duplicate_type pages --similarity_threshold 0.95

  # Find duplicate content with search query:
  python cli_redact.py --task deduplicate --input_file ocr_output.csv --duplicate_type pages --search_query "confidential information"

  # Find duplicate rows in tabular data:
  python cli_redact.py --task deduplicate --input_file data.csv --duplicate_type tabular --text_columns "Name" "Description"
        '''
    )

    # --- Task Selection ---
    task_group = parser.add_argument_group('Task Selection')
    task_group.add_argument('--task', 
    choices=['redact', 'deduplicate'], 
    default='redact', 
    help='Task to perform: redact (PII redaction/anonymization) or deduplicate (find duplicate content).')

    # --- General Arguments (apply to all file types) ---
    general_group = parser.add_argument_group('General Options')
    general_group.add_argument('--input_file', required=True, help='Path to the input file to process.')
    general_group.add_argument('--output_dir', default=OUTPUT_FOLDER, help='Directory for all output files.')
    general_group.add_argument('--language', default=DEFAULT_LANGUAGE, help='Language of the document content.')
    general_group.add_argument('--allow_list', help='Path to a CSV file with words to exclude from redaction.')
    general_group.add_argument('--pii_detector', 
    choices=[LOCAL_PII_OPTION, AWS_PII_OPTION], 
    default=LOCAL_PII_OPTION, 
    help='Core PII detection method (Local or AWS).')
    general_group.add_argument('--aws_access_key', default='', help='Your AWS Access Key ID.')
    general_group.add_argument('--aws_secret_key', default='', help='Your AWS Secret Access Key.')
    general_group.add_argument('--aws_region', default='', help='AWS region for cloud services.')
    general_group.add_argument('--s3_bucket', default='', help='S3 bucket name for cloud operations.')
    general_group.add_argument('--do_initial_clean', action='store_true', help='Perform initial text cleaning for tabular data.')
    general_group.add_argument('--save_logs_to_csv', action='store_true', help='Save processing logs to CSV files.')
    general_group.add_argument('--display_file_names_in_logs', action='store_true', help='Include file names in log outputs.')

    # --- PDF/Image Redaction Arguments ---
    pdf_group = parser.add_argument_group('PDF/Image Redaction Options (.pdf, .png, .jpg)')
    pdf_group.add_argument('--ocr_method', 
    choices=[SELECTABLE_TEXT_EXTRACT_OPTION, TESSERACT_TEXT_EXTRACT_OPTION, TEXTRACT_TEXT_EXTRACT_OPTION], 
    default=TESSERACT_TEXT_EXTRACT_OPTION, 
    help='OCR method for text extraction from images.')
    pdf_group.add_argument('--page_min', type=int, default=0, help='First page to redact.')
    pdf_group.add_argument('--page_max', type=int, default=999, help='Last page to redact.')
    pdf_group.add_argument('--prepare_for_review', action='store_true', help='Prepare files for reviewing redactions.')
    pdf_group.add_argument('--prepare_images', action='store_true', default=True, help='Enable image creation for PDF pages.')
    pdf_group.add_argument('--no_images', action='store_false', dest='prepare_images', help='Disable image creation for PDF pages.')
    pdf_group.add_argument('--images_dpi', type=float, default=300.0, help='DPI for image processing.')
    pdf_group.add_argument('--max_image_pixels', type=int, help='Maximum image pixels for processing.')
    pdf_group.add_argument('--load_truncated_images', action='store_true', help='Load truncated images during processing.')
    pdf_group.add_argument('--chosen_local_ocr_model', choices=['tesseract', 'hybrid', 'paddle'], default='tesseract', help='Local OCR model to use.')
    pdf_group.add_argument('--preprocess_local_ocr_images', action='store_true', help='Preprocess images before OCR.')
    pdf_group.add_argument('--compress_redacted_pdf', action='store_true', help='Compress the final redacted PDF.')
    pdf_group.add_argument('--return_pdf_end_of_redaction', action='store_true', default=True, help='Return PDF at end of redaction process.')
    pdf_group.add_argument('--in_deny_list', nargs='+', default=list(), help='Custom words to recognize for redaction.')
    pdf_group.add_argument('--redact_whole_page_list', nargs='+', default=list(), help='Pages to redact completely.')
    pdf_group.add_argument('--handwrite_signature_checkbox', nargs='+', default=['Extract handwriting', 'Extract signatures'], help='Handwriting and signature extraction options.')

    # --- Word/Tabular Anonymisation Arguments ---
    tabular_group = parser.add_argument_group('Word/Tabular Anonymisation Options (.docx, .csv, .xlsx)')
    tabular_group.add_argument('--anon_strat', choices=['redact', 'encrypt', 'hash', 'replace with \'REDACTED\'', 'replace with <ENTITY_NAME>', 'redact completely', 'mask', 'fake_first_name'], default='redact', help='The anonymisation strategy to apply.')
    tabular_group.add_argument('--columns', nargs='+', default=list(), help='A list of column names to anonymise in tabular data.')
    tabular_group.add_argument('--excel_sheets', nargs='+', default=list(), help='Specific Excel sheet names to process.')
    tabular_group.add_argument('--deny_list', help='Path to a CSV file with specific terms/phrases to redact.')
    tabular_group.add_argument('--fuzzy_mistakes', type=int, default=1, help='Number of allowed spelling mistakes for fuzzy matching.')

    # --- Duplicate Detection Arguments ---
    duplicate_group = parser.add_argument_group('Duplicate Detection Options')
    duplicate_group.add_argument('--duplicate_type', choices=['pages', 'tabular'], default='pages', help='Type of duplicate detection: pages (for OCR files) or tabular (for CSV/Excel files).')
    duplicate_group.add_argument('--similarity_threshold', type=float, default=0.95, help='Similarity threshold (0-1) to consider content as duplicates.')
    duplicate_group.add_argument('--min_word_count', type=int, default=3, help='Minimum word count for text to be considered in duplicate analysis.')
    duplicate_group.add_argument('--min_consecutive_pages', type=int, default=1, help='Minimum number of consecutive pages to consider as a match.')
    duplicate_group.add_argument('--greedy_match', action='store_true', default=True, help='Use greedy matching strategy for consecutive pages.')
    duplicate_group.add_argument('--combine_pages', action='store_true', default=True, help='Combine text from the same page number within a file.')
    duplicate_group.add_argument('--search_query', help='Search query text to find specific duplicate content (for page duplicates).')
    duplicate_group.add_argument('--text_columns', nargs='+', default=list(), help='Specific text columns to analyze for duplicates (for tabular data).')

    # Parse arguments - either from command line or direct mode
    if direct_mode_args:
        # Use direct mode arguments
        args = argparse.Namespace(**direct_mode_args)
    else:
        # Parse command line arguments
        args = parser.parse_args()

    # --- Initial Setup ---
    ensure_output_folder_exists(args.output_dir)
    _, file_extension = os.path.splitext(args.input_file)
    file_extension = file_extension.lower()
    
    # Load allow/deny lists
    allow_list = pd.read_csv(args.allow_list) if args.allow_list else pd.DataFrame()
    deny_list = pd.read_csv(args.deny_list).iloc[:, 0].tolist() if args.deny_list else []

    # --- Route to the Correct Workflow Based on Task and File Type ---

    # Task 1: Redaction/Anonymization
    if args.task == 'redact':
        # Workflow 1: PDF/Image Redaction
        if file_extension in ['.pdf', '.png', '.jpg', '.jpeg']:
            print("--- Detected PDF/Image file. Starting Redaction Workflow... ---")
            try:
                # Step 1: Prepare the document
                print("\nStep 1: Preparing document...")
                (
                    prep_summary, prepared_pdf_paths, image_file_paths, _, _, pdf_doc,
                    image_annotations, _, original_cropboxes, page_sizes, textract_output_found, _, _, _, _
                ) = prepare_image_or_pdf(
                    file_paths=[args.input_file], text_extract_method=args.ocr_method,
                    all_line_level_ocr_results_df=pd.DataFrame(), all_page_line_level_ocr_results_with_words_df=pd.DataFrame(),
                    first_loop_state=True, prepare_for_review=args.prepare_for_review,
                    output_folder=args.output_dir, prepare_images=args.prepare_images
                )
                print(f"Preparation complete. {prep_summary}")

                # Step 2: Redact the prepared document
                print("\nStep 2: Running redaction...")
                (
                    output_summary, output_files, _, _, log_files, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _
                ) = choose_and_run_redactor(
                    file_paths=[args.input_file], prepared_pdf_file_paths=prepared_pdf_paths,
                    pdf_image_file_paths=image_file_paths, chosen_redact_entities=chosen_redact_entities,
                    chosen_redact_comprehend_entities=chosen_comprehend_entities, text_extraction_method=args.ocr_method,
                    in_allow_list=allow_list, in_deny_list=args.in_deny_list,
                    redact_whole_page_list=args.redact_whole_page_list, first_loop_state=True, 
                    page_min=args.page_min, page_max=args.page_max, handwrite_signature_checkbox=args.handwrite_signature_checkbox,
                    pymupdf_doc=pdf_doc, annotations_all_pages=image_annotations, page_sizes=page_sizes,
                    document_cropboxes=original_cropboxes, pii_identification_method=args.pii_detector,
                    aws_access_key_textbox=args.aws_access_key, aws_secret_key_textbox=args.aws_secret_key,
                    language=args.language, output_folder=args.output_dir
                )
                
                print("\n--- Redaction Process Complete ---")
                print(f"Summary: {output_summary}")
                print(f"\nOutput files saved to: {args.output_dir}")
                print("Generated Files:", sorted(output_files))
                if log_files: print("Log Files:", sorted(log_files))

            except Exception as e:
                print(f"\nAn error occurred during the PDF/Image redaction workflow: {e}")

        # Workflow 2: Word/Tabular Data Anonymisation
        elif file_extension in ['.docx', '.xlsx', '.xls', '.csv', '.parquet']:
            print("--- Detected Word/Tabular file. Starting Anonymisation Workflow... ---")
            try:
                # Run the anonymisation function directly
                output_summary, output_files, _, _, log_files, _, _ = anonymise_files_with_open_text(
                    file_paths=[args.input_file],
                    in_text="", # Not used for file-based operations
                    anon_strat=args.anon_strat,
                    chosen_cols=args.columns,
                    chosen_redact_entities=chosen_redact_entities,
                    in_allow_list=allow_list,
                    in_excel_sheets=args.excel_sheets,
                    first_loop_state=True,
                    output_folder=args.output_dir,
                    in_deny_list=deny_list,
                    max_fuzzy_spelling_mistakes_num=args.fuzzy_mistakes,
                    pii_identification_method=args.pii_detector,
                    chosen_redact_comprehend_entities=chosen_comprehend_entities,
                    aws_access_key_textbox=args.aws_access_key,
                    aws_secret_key_textbox=args.aws_secret_key,
                    language=args.language,
                    do_initial_clean=args.do_initial_clean
                )

                print("\n--- Anonymisation Process Complete ---")
                print(f"Summary: {output_summary}")
                print(f"\nOutput files saved to: {args.output_dir}")
                print("Generated Files:", sorted(output_files))
                if log_files: print("Log Files:", sorted(log_files))

            except Exception as e:
                print(f"\nAn error occurred during the Word/Tabular anonymisation workflow: {e}")
                
        else:
            print(f"Error: Unsupported file type '{file_extension}' for redaction.")
            print("Supported types for redaction: .pdf, .png, .jpg, .jpeg")
            print("Supported types for anonymisation: .docx, .xlsx, .xls, .csv, .parquet")

    # Task 2: Duplicate Detection
    elif args.task == 'deduplicate':
        print("--- Starting Duplicate Detection Workflow... ---")
        try:
            if args.duplicate_type == 'pages':
                # Page duplicate detection
                if file_extension == '.csv':
                    print("--- Detected OCR CSV file. Starting Page Duplicate Detection... ---")
                    
                    if args.search_query:
                        # Use search-based duplicate detection
                        print(f"Searching for duplicates of: '{args.search_query}'")
                        # Note: This would require the OCR data to be loaded first
                        # For now, we'll use the general duplicate analysis
                        print("Note: Search-based duplicate detection requires OCR data preparation.")
                        print("Using general duplicate analysis instead.")
                    
                    # Load the CSV file as a list for the duplicate analysis function
                    results_df, output_paths, full_data_by_file = run_duplicate_analysis(
                        files=[args.input_file],
                        threshold=args.similarity_threshold,
                        min_words=args.min_word_count,
                        min_consecutive=args.min_consecutive_pages,
                        greedy_match=args.greedy_match,
                        combine_pages=args.combine_pages
                    )
                    
                    print("\n--- Page Duplicate Detection Complete ---")
                    print(f"Found {len(results_df)} duplicate matches")
                    print(f"\nOutput files saved to: {args.output_dir}")
                    if output_paths: print("Generated Files:", sorted(output_paths))
                    
                else:
                    print(f"Error: Page duplicate detection requires CSV files with OCR data.")
                    print("Please provide a CSV file containing OCR output data.")
                    
            elif args.duplicate_type == 'tabular':
                # Tabular duplicate detection
                if file_extension in ['.csv', '.xlsx', '.xls', '.parquet']:
                    print("--- Detected tabular file. Starting Tabular Duplicate Detection... ---")
                    
                    results_df, output_paths, full_data_by_file = run_tabular_duplicate_analysis(
                        files=[args.input_file],
                        threshold=args.similarity_threshold,
                        min_words=args.min_word_count,
                        text_columns=args.text_columns if args.text_columns else None,
                        output_folder=args.output_dir
                    )
                    
                    print("\n--- Tabular Duplicate Detection Complete ---")
                    print(f"Found {len(results_df)} duplicate matches")
                    print(f"\nOutput files saved to: {args.output_dir}")
                    if output_paths: print("Generated Files:", sorted(output_paths))
                    
                else:
                    print(f"Error: Tabular duplicate detection requires CSV, Excel, or Parquet files.")
                    print("Supported types: .csv, .xlsx, .xls, .parquet")
            else:
                print(f"Error: Invalid duplicate type '{args.duplicate_type}'.")
                print("Valid options: 'pages' or 'tabular'")
                
        except Exception as e:
            print(f"\nAn error occurred during the duplicate detection workflow: {e}")
    
    else:
        print(f"Error: Invalid task '{args.task}'.")
        print("Valid options: 'redact' or 'deduplicate'")

if __name__ == "__main__":
    main()