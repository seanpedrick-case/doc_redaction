import argparse
import os
from tools.helper_functions import ensure_output_folder_exists, get_or_create_env_var, tesseract_ocr_option, text_ocr_option, textract_option, local_pii_detector, aws_pii_detector
from tools.file_conversion import get_input_file_names, prepare_image_or_pdf
from tools.file_redaction import choose_and_run_redactor
import pandas as pd
from datetime import datetime

chosen_comprehend_entities = ['BANK_ACCOUNT_NUMBER','BANK_ROUTING','CREDIT_DEBIT_NUMBER', 'CREDIT_DEBIT_CVV',          'CREDIT_DEBIT_EXPIRY','PIN','EMAIL','ADDRESS',
                                'NAME','PHONE', 'PASSPORT_NUMBER','DRIVER_ID', 'USERNAME','PASSWORD',
                                'IP_ADDRESS','MAC_ADDRESS','LICENSE_PLATE',
                                'VEHICLE_IDENTIFICATION_NUMBER','UK_NATIONAL_INSURANCE_NUMBER',
                                'INTERNATIONAL_BANK_ACCOUNT_NUMBER','SWIFT_CODE',
                                'UK_NATIONAL_HEALTH_SERVICE_NUMBER']
chosen_redact_entities = ["TITLES", "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", 
                            "STREETNAME", "UKPOSTCODE"]

def main(first_loop_state=True, latest_file_completed=0, output_summary="", output_file_list=None, 
         log_files_list=None, estimated_time=0, textract_metadata="", comprehend_query_num=0, 
         current_loop_page=0, page_break=False, pdf_doc_state = [], all_image_annotations = [], all_line_level_ocr_results = pd.DataFrame(), all_decision_process_table = pd.DataFrame(),chosen_comprehend_entities = chosen_comprehend_entities, chosen_redact_entities = chosen_redact_entities, handwrite_signature_checkbox = ["Redact all identified handwriting", "Redact all identified signatures"]):
    
    if output_file_list is None:
        output_file_list = []
    if log_files_list is None:
        log_files_list = []

    parser = argparse.ArgumentParser(description='Redact PII from documents via command line')
    
    # Required arguments
    parser.add_argument('--input_file', help='Path to input file (PDF, JPG, or PNG)')
    
    # Optional arguments with defaults matching the GUI app
    parser.add_argument('--ocr_method', choices=[text_ocr_option, tesseract_ocr_option, textract_option],
                       default='Quick image analysis', help='OCR method to use')
    parser.add_argument('--pii_detector', choices=[local_pii_detector, aws_pii_detector],
                       default='Local', help='PII detection method')
    parser.add_argument('--page_min', type=int, default=0, help='First page to redact')
    parser.add_argument('--page_max', type=int, default=0, help='Last page to redact')
    parser.add_argument('--allow_list', help='Path to allow list CSV file')
    parser.add_argument('--output_dir', default='output/', help='Output directory')

    args = parser.parse_args()

    # Ensure output directory exists
    ensure_output_folder_exists()

    # Create file object similar to what Gradio provides
    file_obj = {"name": args.input_file}

    # Load allow list if provided
    allow_list_df = pd.DataFrame()
    if args.allow_list:
        allow_list_df = pd.read_csv(args.allow_list)

    # Get file names
    file_name_no_ext, file_name_with_ext, full_file_name = get_input_file_names(file_obj)

    # Initialize empty states for PDF processing    
    
    # Prepare PDF/image
    output_summary, prepared_pdf, images_pdf, max_pages, annotate_max_pages_bottom, pdf_doc_state, all_image_annotations = prepare_image_or_pdf(
        file_obj, args.ocr_method, allow_list_df, latest_file_completed, 
        output_summary, first_loop_state, args.page_max, current_loop_page, all_image_annotations
    )
        
    output_summary, output_files, output_file_list, latest_file_completed, log_files, \
    log_files_list, estimated_time, textract_metadata, pdf_doc_state, all_image_annotations, \
    current_loop_page, page_break, all_line_level_ocr_results, all_decision_process_table, \
    comprehend_query_num = choose_and_run_redactor(
        file_obj, prepared_pdf, images_pdf, "en", chosen_redact_entities,
        chosen_comprehend_entities, args.ocr_method, allow_list_df,
        latest_file_completed, output_summary, output_file_list, log_files_list,
        first_loop_state, args.page_min, args.page_max, estimated_time,
        handwrite_signature_checkbox, textract_metadata, all_image_annotations,
        all_line_level_ocr_results, all_decision_process_table, pdf_doc_state,
        current_loop_page, page_break, args.pii_detector, comprehend_query_num, args.output_dir
    )

    print(f"\nRedaction complete. Output file_list:\n{output_file_list}")
    print(f"\nOutput files saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 