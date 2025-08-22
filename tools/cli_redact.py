import argparse
import os
import pandas as pd
from tools.config import get_or_create_env_var, LOCAL_PII_OPTION, AWS_PII_OPTION, SELECTABLE_TEXT_EXTRACT_OPTION, TESSERACT_TEXT_EXTRACT_OPTION, TEXTRACT_TEXT_EXTRACT_OPTION
from tools.helper_functions import ensure_output_folder_exists
from tools.file_conversion import get_input_file_names, prepare_image_or_pdf
from tools.file_redaction import choose_and_run_redactor
from tools.anonymisation import anonymise_files_with_open_text

# --- Constants and Configuration ---
INPUT_FOLDER = 'input/'
OUTPUT_FOLDER = 'output/'
DEFAULT_LANGUAGE = 'en'

# Define entities for redaction
chosen_comprehend_entities = [
    'BANK_ACCOUNT_NUMBER', 'BANK_ROUTING', 'CREDIT_DEBIT_NUMBER', 
    'CREDIT_DEBIT_CVV', 'CREDIT_DEBIT_EXPIRY', 'PIN', 'EMAIL', 'ADDRESS',
    'NAME', 'PHONE', 'PASSPORT_NUMBER', 'DRIVER_ID', 'USERNAME', 'PASSWORD',
    'IP_ADDRESS', 'MAC_ADDRESS', 'LICENSE_PLATE', 'VEHICLE_IDENTIFICATION_NUMBER',
    'UK_NATIONAL_INSURANCE_NUMBER', 'INTERNATIONAL_BANK_ACCOUNT_NUMBER',
    'SWIFT_CODE', 'UK_NATIONAL_HEALTH_SERVICE_NUMBER'
]
chosen_redact_entities = [
    "TITLES", "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "STREETNAME", "UKPOSTCODE"
]

# --- Main CLI Function ---
def main():
    """
    A unified command-line interface to prepare, redact, and anonymise various document types.
    """
    parser = argparse.ArgumentParser(
        description='A versatile CLI for redacting PII from PDF/image files and anonymising Word/tabular data.',
        formatter_class=argparse.RawTextHelpFormatter
    )

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

    # --- PDF/Image Redaction Arguments ---
    pdf_group = parser.add_argument_group('PDF/Image Redaction Options (.pdf, .png, .jpg)')
    pdf_group.add_argument('--ocr_method', 
    choices=[SELECTABLE_TEXT_EXTRACT_OPTION, TESSERACT_TEXT_EXTRACT_OPTION, TEXTRACT_TEXT_EXTRACT_OPTION], 
    default=TESSERACT_TEXT_EXTRACT_OPTION, 
    help='OCR method for text extraction from images.')
    pdf_group.add_argument('--page_min', type=int, default=0, help='First page to redact.')
    pdf_group.add_argument('--page_max', type=int, default=999, help='Last page to redact.')
    pdf_group.add_argument('--prepare_for_review', action='store_true', help='Prepare files for reviewing redactions.')
    pdf_group.add_argument('--no_images', action='store_false', dest='prepare_images', help='Disable image creation for PDF pages.')

    # --- Word/Tabular Anonymisation Arguments ---
    tabular_group = parser.add_argument_group('Word/Tabular Anonymisation Options (.docx, .csv, .xlsx)')
    tabular_group.add_argument('--anon_strat', choices=['redact', 'encrypt', 'hash'], default='redact', help='The anonymisation strategy to apply.')
    tabular_group.add_argument('--columns', nargs='+', default=[], help='A list of column names to anonymise in tabular data.')
    tabular_group.add_argument('--excel_sheets', nargs='+', default=[], help='Specific Excel sheet names to process.')
    tabular_group.add_argument('--deny_list', help='Path to a CSV file with specific terms/phrases to redact.')
    tabular_group.add_argument('--fuzzy_mistakes', type=int, default=1, help='Number of allowed spelling mistakes for fuzzy matching.')

    args = parser.parse_args()

    # --- Initial Setup ---
    ensure_output_folder_exists(args.output_dir)
    _, file_extension = os.path.splitext(args.input_file)
    file_extension = file_extension.lower()
    
    # Load allow/deny lists
    allow_list = pd.read_csv(args.allow_list) if args.allow_list else pd.DataFrame()
    deny_list = pd.read_csv(args.deny_list).iloc[:, 0].tolist() if args.deny_list else []


    # --- Route to the Correct Workflow Based on File Type ---

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
                in_allow_list=allow_list, first_loop_state=True, page_min=args.page_min, page_max=args.page_max,
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
                language=args.language
            )

            print("\n--- Anonymisation Process Complete ---")
            print(f"Summary: {output_summary}")
            print(f"\nOutput files saved to: {args.output_dir}")
            print("Generated Files:", sorted(output_files))
            if log_files: print("Log Files:", sorted(log_files))

        except Exception as e:
            print(f"\nAn error occurred during the Word/Tabular anonymisation workflow: {e}")
            
    else:
        print(f"Error: Unsupported file type '{file_extension}'.")
        print("Supported types for redaction: .pdf, .png, .jpg, .jpeg")
        print("Supported types for anonymisation: .docx, .xlsx, .xls, .csv, .parquet")

if __name__ == "__main__":
    main()