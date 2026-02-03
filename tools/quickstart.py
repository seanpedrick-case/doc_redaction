"""Helper functions for the quickstart walkthrough in the redaction app."""

import os

import gradio as gr
import pandas as pd

from tools.config import (
    AWS_LLM_PII_OPTION,
    AWS_PII_OPTION,
    CHOSEN_COMPREHEND_ENTITIES,
    CHOSEN_LLM_ENTITIES,
    CHOSEN_REDACT_ENTITIES,
    DEFAULT_PII_DETECTION_MODEL,
    INFERENCE_SERVER_PII_OPTION,
    LOCAL_PII_OPTION,
    LOCAL_TRANSFORMERS_LLM_PII_OPTION,
    NO_REDACTION_PII_OPTION,
    SHOW_AWS_TEXT_EXTRACTION_OPTIONS,
    SHOW_INFERENCE_SERVER_VLM_MODEL_OPTIONS,
    SHOW_OCR_GUI_OPTIONS,
    SHOW_PII_IDENTIFICATION_OPTIONS,
    TESSERACT_TEXT_EXTRACT_OPTION,
    TEXTRACT_TEXT_EXTRACT_OPTION,
)
from tools.helper_functions import put_columns_in_df


def is_data_file_type_walkthrough(files):
    """Check if files are data file types (xlsx, xls, csv, parquet, docx)."""
    if not files:
        return False
    data_file_extensions = {".xlsx", ".xls", ".csv", ".parquet", ".docx"}
    for file in files:
        if file:
            file_path = file.name if hasattr(file, "name") else str(file)
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in data_file_extensions:
                return True
    return False


def route_walkthrough_files(files):
    """Route files from walkthrough to appropriate component and determine if data file."""
    if not files:
        return None, None, False, gr.Walkthrough(selected=2)

    is_data = is_data_file_type_walkthrough(files)
    doc_files = []
    data_files = []

    data_file_extensions = {".xlsx", ".xls", ".csv", ".parquet", ".docx"}

    for file in files:
        if file:
            file_path = file.name if hasattr(file, "name") else str(file)
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in data_file_extensions:
                data_files.append(file)
            else:
                doc_files.append(file)

    # Return files for appropriate component (None for the other)
    if is_data:
        return None, data_files, True, gr.Walkthrough(selected=2)
    else:
        return doc_files, None, False, gr.Walkthrough(selected=2)


def handle_step_2_next(
    files,
    is_data_file,
    walkthrough_colnames_val,
    walkthrough_excel_sheets_val,
    text_extract_method_val,
):
    """Handle step 2 next button - populate dropdowns if data files and sync with main components."""
    # Show text extraction method radio in Step 2 if SHOW_OCR_GUI_OPTIONS is True
    show_text_extract_method = SHOW_OCR_GUI_OPTIONS

    if is_data_file and files:
        # Use put_columns_in_df to populate dropdowns
        colnames_dropdown, excel_sheets_dropdown = put_columns_in_df(files)
        # Use the selected values from walkthrough if available, otherwise use the populated values
        if (
            walkthrough_colnames_val
            and len(walkthrough_colnames_val) > 0
            and walkthrough_colnames_val[0] != "Choose columns to anonymise"
        ):
            main_colnames_update = gr.Dropdown(value=walkthrough_colnames_val)
        else:
            main_colnames_update = colnames_dropdown

        if (
            walkthrough_excel_sheets_val
            and len(walkthrough_excel_sheets_val) > 0
            and walkthrough_excel_sheets_val[0] != "Choose Excel sheets to anonymise"
        ):
            main_excel_sheets_update = gr.Dropdown(
                value=walkthrough_excel_sheets_val, visible=True
            )
        else:
            main_excel_sheets_update = excel_sheets_dropdown

        # Return updates for both walkthrough and main components, and advance walkthrough
        # Note: walkthrough_local_ocr_method_radio and walkthrough_handwrite_signature_checkbox visibility
        # are controlled by event handler on walkthrough_text_extract_method_radio
        # Note: Step 3 PII components visibility is controlled by event handler on walkthrough_redaction_method_dropdown
        return (
            colnames_dropdown,  # walkthrough_colnames
            excel_sheets_dropdown,  # walkthrough_excel_sheets
            main_colnames_update,  # in_colnames
            main_excel_sheets_update,  # in_excel_sheets (defined in "Word or Excel/csv files" tab)
            gr.Radio(
                visible=show_text_extract_method
            ),  # walkthrough_text_extract_method_radio
            gr.Walkthrough(selected=3),  # walkthrough
        )
    else:
        # Return unchanged dropdowns and advance
        # Note: walkthrough_local_ocr_method_radio and walkthrough_handwrite_signature_checkbox visibility
        # are controlled by event handler on walkthrough_text_extract_method_radio
        # Note: Step 3 PII components visibility is controlled by event handler on walkthrough_redaction_method_dropdown
        return (
            gr.Dropdown(visible=False),  # walkthrough_colnames
            gr.Dropdown(visible=False),  # walkthrough_excel_sheets
            gr.Dropdown(),  # in_colnames (no change)
            gr.Dropdown(visible=False),  # in_excel_sheets (no change)
            gr.Radio(
                visible=show_text_extract_method
            ),  # walkthrough_text_extract_method_radio
            gr.Walkthrough(selected=3),  # walkthrough
        )


def update_step_2_on_data_file_upload(files, is_data_file):
    """Update Step 2 components when data files are uploaded."""
    if is_data_file and files:
        # Use put_columns_in_df to populate dropdowns
        colnames_dropdown, excel_sheets_dropdown = put_columns_in_df(files)
        return colnames_dropdown, excel_sheets_dropdown
    else:
        return gr.Dropdown(visible=False), gr.Dropdown(visible=False)


def handle_text_extract_method_selection(text_extract_method):
    """Handle text extraction method selection - show local OCR radio only if Local OCR model is selected,
    and show AWS Textract settings only if AWS Textract is selected."""
    # Show local OCR method radio only if "Local OCR model - PDFs without selectable text" is selected
    # When "AWS Bedrock VLM OCR" is selected, the local OCR method is automatically set to "bedrock-vlm" but the component is hidden
    show_local_ocr = text_extract_method == TESSERACT_TEXT_EXTRACT_OPTION
    # Show AWS Textract settings only if "AWS Textract service - all PDF types" is selected
    show_aws_textract = text_extract_method == TEXTRACT_TEXT_EXTRACT_OPTION

    return (
        gr.Radio(visible=show_local_ocr),  # walkthrough_local_ocr_method_radio
        gr.CheckboxGroup(
            visible=show_aws_textract
        ),  # walkthrough_handwrite_signature_checkbox
    )


def handle_redaction_method_selection(redaction_method):
    """Handle redaction method selection in Step 3 - show appropriate components based on selection."""
    # Check which redaction method is selected
    is_redact_all_pii = redaction_method == "Redact all PII"
    is_redact_selected_terms = redaction_method == "Redact selected terms"

    # Show PII detection settings if "Redact all PII" OR "Redact selected terms" is selected
    # Both options need PII detection method to determine what to redact
    show_pii_method = (
        is_redact_all_pii or is_redact_selected_terms
    ) and SHOW_PII_IDENTIFICATION_OPTIONS

    # Show deny/allow/fully redacted lists only if "Redact selected terms" is selected
    # These lists are essential for "Redact selected terms" mode, so show them regardless of SHOW_PII_IDENTIFICATION_OPTIONS
    show_selected_terms_lists = is_redact_selected_terms

    # Determine initial visibility of entity dropdowns based on default PII method
    default_pii_method = DEFAULT_PII_DETECTION_MODEL
    show_local_entities_init = show_pii_method and (
        default_pii_method == LOCAL_PII_OPTION
    )
    show_comprehend_entities_init = show_pii_method and (
        default_pii_method == AWS_PII_OPTION
    )
    is_llm_method_init = show_pii_method and (
        default_pii_method == LOCAL_TRANSFORMERS_LLM_PII_OPTION
        or default_pii_method == INFERENCE_SERVER_PII_OPTION
        or default_pii_method == AWS_LLM_PII_OPTION
    )

    # For "Extract text only", hide all components
    # For "Redact all PII", show PII detection components
    # For "Redact selected terms", show both PII detection components AND deny/allow/fully redacted list components

    # Set entity values based on redaction method
    if is_redact_selected_terms:
        # For "Redact selected terms", only show CUSTOM entity
        local_entities_update = gr.update(
            visible=show_local_entities_init, value=["CUSTOM"]
        )
        comprehend_entities_update = gr.update(
            visible=show_comprehend_entities_init, value=["CUSTOM"]
        )
        llm_entities_update = gr.update(visible=is_llm_method_init, value=["CUSTOM"])
    elif is_redact_all_pii:
        # For "Redact all PII", use default entities
        # Ensure entities are lists (they should already be parsed in config.py)
        local_entities_val = (
            CHOSEN_REDACT_ENTITIES
            if isinstance(CHOSEN_REDACT_ENTITIES, list)
            else ["CUSTOM"]
        )
        comprehend_entities_val = (
            CHOSEN_COMPREHEND_ENTITIES
            if isinstance(CHOSEN_COMPREHEND_ENTITIES, list)
            else ["CUSTOM"]
        )
        llm_entities_val = (
            CHOSEN_LLM_ENTITIES if isinstance(CHOSEN_LLM_ENTITIES, list) else ["CUSTOM"]
        )
        local_entities_update = gr.update(
            visible=show_local_entities_init, value=local_entities_val
        )
        comprehend_entities_update = gr.update(
            visible=show_comprehend_entities_init, value=comprehend_entities_val
        )
        llm_entities_update = gr.update(
            visible=is_llm_method_init, value=llm_entities_val
        )
    else:
        # For "Extract text only", just update visibility without changing value
        local_entities_update = gr.update(visible=show_local_entities_init)
        comprehend_entities_update = gr.update(visible=show_comprehend_entities_init)
        llm_entities_update = gr.update(visible=is_llm_method_init)

    return (
        gr.update(
            visible=show_pii_method
        ),  # walkthrough_pii_identification_method_drop
        local_entities_update,  # walkthrough_in_redact_entities
        comprehend_entities_update,  # walkthrough_in_redact_comprehend_entities
        llm_entities_update,  # walkthrough_in_redact_llm_entities
        gr.update(
            visible=is_llm_method_init
        ),  # walkthrough_custom_llm_instructions_textbox
        gr.update(visible=show_selected_terms_lists),  # walkthrough_deny_list_state
        gr.update(visible=show_selected_terms_lists),  # walkthrough_allow_list_state
        gr.update(
            visible=show_selected_terms_lists
        ),  # walkthrough_fully_redacted_list_state
    )


def handle_pii_method_selection(pii_method):
    """Handle PII method selection - show appropriate entity dropdowns."""
    # Check if method is Local
    show_local_entities = pii_method == LOCAL_PII_OPTION
    # Check if method is AWS Comprehend
    show_comprehend_entities = pii_method == AWS_PII_OPTION
    # Check if method is an LLM option
    is_llm_method = (
        pii_method == LOCAL_TRANSFORMERS_LLM_PII_OPTION
        or pii_method == INFERENCE_SERVER_PII_OPTION
        or pii_method == AWS_LLM_PII_OPTION
    )

    return (
        gr.Dropdown(visible=show_local_entities),  # walkthrough_in_redact_entities
        gr.Dropdown(
            visible=show_comprehend_entities
        ),  # walkthrough_in_redact_comprehend_entities
        gr.Dropdown(visible=is_llm_method),  # walkthrough_in_redact_llm_entities
        gr.Textbox(
            visible=is_llm_method
        ),  # walkthrough_custom_llm_instructions_textbox
    )


def handle_step_3_next(
    text_extract_method_val,
    local_ocr_method_val,
    handwrite_signature_val,
    pii_method_val,
    redact_entities_val,
    redact_comprehend_entities_val,
    redact_llm_entities_val,
    custom_llm_instructions_val,
    deny_list_val,
    allow_list_val,
    fully_redacted_list_val,
    pii_method_tabular_val,
    anon_strategy_val,
    do_initial_clean_val,
    redact_duplicate_pages_val,
    max_fuzzy_spelling_mistakes_num_val,
):
    """Handle step 3 next button - write values to main components."""
    # Update text extraction method with walkthrough value
    text_extract_method_update = (
        gr.Radio(value=text_extract_method_val)
        if text_extract_method_val
        else gr.Radio()
    )

    # Update OCR components with walkthrough values
    local_ocr_update = (
        gr.Radio(value=local_ocr_method_val) if local_ocr_method_val else gr.Radio()
    )
    handwrite_signature_update = (
        gr.CheckboxGroup(value=handwrite_signature_val)
        if handwrite_signature_val
        else gr.CheckboxGroup()
    )

    # Update PII components with walkthrough values
    pii_method_update = gr.Radio(value=pii_method_val) if pii_method_val else gr.Radio()
    # Always update dropdowns with the value, even if it's an empty list
    # This ensures that empty selections are correctly written to main components
    redact_entities_update = (
        gr.Dropdown(value=redact_entities_val)
        if redact_entities_val is not None
        else gr.Dropdown()
    )
    redact_comprehend_entities_update = (
        gr.Dropdown(value=redact_comprehend_entities_val)
        if redact_comprehend_entities_val is not None
        else gr.Dropdown()
    )
    redact_llm_entities_update = (
        gr.Dropdown(value=redact_llm_entities_val)
        if redact_llm_entities_val is not None
        else gr.Dropdown()
    )
    custom_llm_instructions_update = (
        gr.Textbox(value=custom_llm_instructions_val)
        if custom_llm_instructions_val is not None
        else gr.Textbox()
    )

    # Update deny/allow/fully redacted list components with walkthrough values
    # Convert DataFrame to list if needed (for backward compatibility)
    # Ensure all items are strings for Dropdown components
    if deny_list_val is not None:
        if isinstance(deny_list_val, pd.DataFrame):
            deny_list_val = (
                deny_list_val.iloc[:, 0].tolist() if not deny_list_val.empty else []
            )
        # Ensure all items are strings
        if isinstance(deny_list_val, list):
            deny_list_val = (
                [str(item) for item in deny_list_val if item] if deny_list_val else []
            )
        deny_list_update = (
            gr.Dropdown(value=deny_list_val) if deny_list_val else gr.Dropdown()
        )
    else:
        deny_list_update = gr.Dropdown()

    if allow_list_val is not None:
        if isinstance(allow_list_val, pd.DataFrame):
            allow_list_val = (
                allow_list_val.iloc[:, 0].tolist() if not allow_list_val.empty else []
            )
        # Ensure all items are strings
        if isinstance(allow_list_val, list):
            allow_list_val = (
                [str(item) for item in allow_list_val if item] if allow_list_val else []
            )
        allow_list_update = (
            gr.Dropdown(value=allow_list_val) if allow_list_val else gr.Dropdown()
        )
    else:
        allow_list_update = gr.Dropdown()

    if fully_redacted_list_val is not None:
        if isinstance(fully_redacted_list_val, pd.DataFrame):
            fully_redacted_list_val = (
                fully_redacted_list_val.iloc[:, 0].tolist()
                if not fully_redacted_list_val.empty
                else []
            )
        # Ensure all items are strings
        if isinstance(fully_redacted_list_val, list):
            fully_redacted_list_val = (
                [str(item) for item in fully_redacted_list_val if item]
                if fully_redacted_list_val
                else []
            )
        fully_redacted_list_update = (
            gr.Dropdown(value=fully_redacted_list_val)
            if fully_redacted_list_val
            else gr.Dropdown()
        )
    else:
        fully_redacted_list_update = gr.Dropdown()

    # Update tabular data components with walkthrough values
    pii_method_tabular_update = (
        gr.Radio(value=pii_method_tabular_val)
        if pii_method_tabular_val is not None
        else gr.Radio()
    )
    anon_strategy_update = (
        gr.Radio(value=anon_strategy_val)
        if anon_strategy_val is not None
        else gr.Radio()
    )
    do_initial_clean_update = (
        gr.Checkbox(value=do_initial_clean_val)
        if do_initial_clean_val is not None
        else gr.Checkbox()
    )

    # Update redact duplicate pages checkbox with walkthrough value
    redact_duplicate_pages_update = (
        gr.Checkbox(value=redact_duplicate_pages_val)
        if redact_duplicate_pages_val is not None
        else gr.Checkbox()
    )

    # Update max fuzzy spelling mistakes number with walkthrough value
    max_fuzzy_spelling_mistakes_num_update = (
        gr.Number(value=max_fuzzy_spelling_mistakes_num_val)
        if max_fuzzy_spelling_mistakes_num_val is not None
        else gr.Number()
    )

    return (
        text_extract_method_update,  # text_extract_method_radio
        local_ocr_update,  # local_ocr_method_radio
        handwrite_signature_update,  # handwrite_signature_checkbox
        pii_method_update,  # pii_identification_method_drop
        redact_entities_update,  # in_redact_entities
        redact_comprehend_entities_update,  # in_redact_comprehend_entities
        redact_llm_entities_update,  # in_redact_llm_entities
        custom_llm_instructions_update,  # custom_llm_instructions_textbox
        deny_list_update,  # in_deny_list_state
        allow_list_update,  # in_allow_list_state
        fully_redacted_list_update,  # in_fully_redacted_list_state
        pii_method_tabular_update,  # pii_identification_method_drop_tabular
        anon_strategy_update,  # anon_strategy
        do_initial_clean_update,  # do_initial_clean
        redact_duplicate_pages_update,  # redact_duplicate_pages_checkbox
        gr.Walkthrough(selected=4),  # walkthrough
        max_fuzzy_spelling_mistakes_num_update,  # max_fuzzy_spelling_mistakes_num
    )


def handle_step_4_next(
    page_min_val,
    page_max_val,
    textract_output_found_val,
    relevant_ocr_output_with_words_found_val,
    total_pdf_page_count_val,
    estimated_aws_costs_val,
    estimated_time_taken_val,
    cost_code_dataframe_val,
    cost_code_choice_val,
):
    """Handle step 4 next button - write values to main components."""
    # Update page selection components
    page_min_update = (
        gr.Number(value=page_min_val) if page_min_val is not None else gr.Number()
    )
    page_max_update = (
        gr.Number(value=page_max_val) if page_max_val is not None else gr.Number()
    )

    # Update cost-related components (if SHOW_COSTS is True)
    textract_output_found_update = (
        gr.Checkbox(value=textract_output_found_val)
        if textract_output_found_val is not None
        else gr.Checkbox()
    )
    relevant_ocr_output_with_words_found_update = (
        gr.Checkbox(value=relevant_ocr_output_with_words_found_val)
        if relevant_ocr_output_with_words_found_val is not None
        else gr.Checkbox()
    )
    total_pdf_page_count_update = (
        gr.Number(value=total_pdf_page_count_val)
        if total_pdf_page_count_val is not None
        else gr.Number()
    )
    estimated_aws_costs_update = (
        gr.Number(value=estimated_aws_costs_val)
        if estimated_aws_costs_val is not None
        else gr.Number()
    )
    estimated_time_taken_update = (
        gr.Number(value=estimated_time_taken_val)
        if estimated_time_taken_val is not None
        else gr.Number()
    )

    # Update cost code components (if GET_COST_CODES or ENFORCE_COST_CODES is True)
    cost_code_dataframe_update = (
        gr.Dataframe(value=cost_code_dataframe_val)
        if cost_code_dataframe_val is not None
        else gr.Dataframe()
    )
    cost_code_choice_update = (
        gr.Dropdown(value=cost_code_choice_val)
        if cost_code_choice_val is not None
        else gr.Dropdown()
    )

    return (
        page_min_update,  # page_min
        page_max_update,  # page_max
        textract_output_found_update,  # textract_output_found_checkbox
        relevant_ocr_output_with_words_found_update,  # relevant_ocr_output_with_words_found_checkbox
        total_pdf_page_count_update,  # total_pdf_page_count
        estimated_aws_costs_update,  # estimated_aws_costs_number
        estimated_time_taken_update,  # estimated_time_taken_number
        cost_code_dataframe_update,  # cost_code_dataframe
        cost_code_choice_update,  # cost_code_choice_drop
        gr.Walkthrough(selected=5),  # walkthrough
    )


def sync_walkthrough_outputs_to_original(summary_text, output_file_value):
    """Sync walkthrough output components to original components.

    This function takes the outputs from the redaction process and duplicates
    them to both walkthrough and original output components.

    Args:
        summary_text: The output summary text
        output_file_value: The output file value

    Returns:
        Tuple of (walkthrough_summary, walkthrough_file, original_summary, original_file)
    """
    return (
        summary_text,  # walkthrough_redaction_output_summary_textbox
        output_file_value,  # walkthrough_output_file
        summary_text,  # redaction_output_summary_textbox (original)
        output_file_value,  # output_file (original)
    )


def sync_walkthrough_tabular_outputs_to_original(summary_text, output_file_value):
    """Sync walkthrough tabular output components to original components.

    This function takes the outputs from the tabular redaction process and duplicates
    them to both walkthrough and original output components.

    Args:
        summary_text: The output summary text
        output_file_value: The output file value

    Returns:
        Tuple of (walkthrough_summary, walkthrough_file, original_summary, original_file)
    """
    return (
        summary_text,  # walkthrough_text_output_summary
        output_file_value,  # walkthrough_text_output_file
        summary_text,  # text_output_summary (original)
        output_file_value,  # text_output_file (original)
    )


def update_step_3_tabular_visibility(is_data_file):
    """Update visibility of Step 3 tabular components based on file type.

    Args:
        is_data_file: Boolean indicating if uploaded file is a data file

    Returns:
        Tuple of visibility updates for tabular components
    """
    return (
        gr.update(
            visible=is_data_file
        ),  # walkthrough_pii_identification_method_drop_tabular
        gr.update(visible=is_data_file),  # walkthrough_anon_strategy
        gr.update(visible=is_data_file),  # walkthrough_do_initial_clean
    )


def update_step_4_visibility(is_data_file):
    """Update visibility of Step 4 components based on file type.

    Args:
        is_data_file: Boolean indicating if uploaded file is a data file

    Returns:
        Tuple of visibility updates for document and tabular components
    """
    # For Row components, we need to update visibility of children
    # Return updates for button and both output components in each row
    return (
        gr.update(visible=not is_data_file),  # step_4_next_document_redact_btn
        gr.update(visible=is_data_file),  # step_4_next_tabular_redact_btn
    )


def handle_main_text_extract_method_selection(text_extract_method):
    """Handle text extraction method selection for main components - show local OCR options only if Local OCR model is selected,
    and show AWS Textract settings only if AWS Textract is selected.

    Args:
        text_extract_method: Selected text extraction method

    Returns:
        Tuple of visibility updates for local OCR accordion, inference server accordion, and AWS Textract accordion
    """
    # Show local OCR method accordion only if "Local OCR model - PDFs without selectable text" is selected
    # When "AWS Bedrock VLM OCR" is selected, the local OCR method is automatically set to "bedrock-vlm" but the component is hidden
    show_local_ocr = text_extract_method == TESSERACT_TEXT_EXTRACT_OPTION
    # Show AWS Textract settings accordion only if "AWS Textract service - all PDF types" is selected
    show_aws_textract = (
        text_extract_method == TEXTRACT_TEXT_EXTRACT_OPTION
        and SHOW_AWS_TEXT_EXTRACTION_OPTIONS
    )
    # Show inference server VLM model accordion only if local OCR is selected (not Bedrock VLM) and the option is enabled
    show_inference_server = (
        text_extract_method == TESSERACT_TEXT_EXTRACT_OPTION
        and SHOW_INFERENCE_SERVER_VLM_MODEL_OPTIONS
    )

    return (
        gr.update(visible=show_local_ocr),  # local_ocr_method_accordion
        gr.update(
            visible=show_inference_server
        ),  # inference_server_vlm_model_accordion
        gr.update(visible=show_aws_textract),  # aws_textract_signature_accordion
    )


def handle_main_pii_method_selection(pii_method):
    """Handle PII method selection for main components - show appropriate entity dropdowns and hide all if No PII redaction is selected.

    Args:
        pii_method: Selected PII detection method

    Returns:
        Tuple of visibility updates for PII method dropdown, local entities accordion, comprehend entities accordion,
        LLM entities accordion, and LLM custom instructions accordion
    """
    # Check if "No PII redaction" is selected
    is_no_redaction = pii_method == NO_REDACTION_PII_OPTION

    # If no redaction, hide all PII-related components
    if is_no_redaction:
        return (
            gr.update(
                visible=True
            ),  # pii_identification_method_drop (keep visible so user can change selection)
            gr.update(visible=False),  # local_entities
            gr.update(visible=False),  # comprehend_entities
            gr.update(visible=False),  # llm_entities
            gr.update(visible=False),  # llm_custom_instructions
        )

    # Check if method is Local
    show_local_entities = pii_method == LOCAL_PII_OPTION
    # Check if method is AWS Comprehend
    show_comprehend_entities = pii_method == AWS_PII_OPTION
    # Check if method is an LLM option
    is_llm_method = (
        pii_method == LOCAL_TRANSFORMERS_LLM_PII_OPTION
        or pii_method == INFERENCE_SERVER_PII_OPTION
        or pii_method == AWS_LLM_PII_OPTION
    )

    return (
        gr.update(visible=True),  # pii_identification_method_drop
        gr.update(visible=show_local_entities),  # local_entities_accordion
        gr.update(visible=show_comprehend_entities),  # comprehend_entities_accordion
        gr.update(visible=is_llm_method),  # llm_entities_accordion
        gr.update(visible=is_llm_method),  # llm_custom_instructions_accordion
    )
