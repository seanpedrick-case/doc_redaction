import re
import secrets
import base64
import time
import pandas as pd

from faker import Faker
from gradio import Progress
from typing import List, Dict, Any

from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, DictAnalyzerResult, RecognizerResult
from presidio_anonymizer import AnonymizerEngine, BatchAnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig, ConflictResolutionStrategy

from tools.helper_functions import output_folder, get_file_path_end, read_file, detect_file_type
from tools.load_spacy_model_custom_recognisers import nlp_analyser, score_threshold

# Use custom version of analyze_dict to be able to track progress
from tools.presidio_analyzer_custom import analyze_dict


fake = Faker("en_UK")
def fake_first_name(x):
    return fake.first_name()

def process_recognizer_result(result, recognizer_result, data_row, dictionary_key, df_dict, keys_to_keep):
        output = []

        if hasattr(result, 'value'):
            text = result.value[data_row]
        else:
            text = ""        

        if isinstance(recognizer_result, list):
            for sub_result in recognizer_result:
                if isinstance(text, str):
                    found_text = text[sub_result.start:sub_result.end]
                else:
                    found_text = ''
                analysis_explanation = {key: sub_result.__dict__[key] for key in keys_to_keep}
                analysis_explanation.update({
                    'data_row': str(data_row),
                    'column': list(df_dict.keys())[dictionary_key],
                    'entity': found_text
                })
                output.append(str(analysis_explanation))
        
        return output

# Writing decision making process to file
def generate_decision_process_output(analyzer_results: List[DictAnalyzerResult], df_dict: Dict[str, List[Any]]) -> str:
    """
    Generate a detailed output of the decision process for entity recognition.

    This function takes the results from the analyzer and the original data dictionary,
    and produces a string output detailing the decision process for each recognized entity.
    It includes information such as entity type, position, confidence score, and the context
    in which the entity was found.

    Args:
        analyzer_results (List[DictAnalyzerResult]): The results from the entity analyzer.
        df_dict (Dict[str, List[Any]]): The original data in dictionary format.

    Returns:
        str: A string containing the detailed decision process output.
    """
    decision_process_output = []
    keys_to_keep = ['entity_type', 'start', 'end']

    # Run through each column to analyse for PII
    for i, result in enumerate(analyzer_results):

        # If a single result
        if isinstance(result, RecognizerResult):
            decision_process_output.extend(process_recognizer_result(result, result, 0, i, df_dict, keys_to_keep))

        # If a list of results
        elif isinstance(result, list) or isinstance(result, DictAnalyzerResult):
            for x, recognizer_result in enumerate(result.recognizer_results):
                decision_process_output.extend(process_recognizer_result(result, recognizer_result, x, i, df_dict, keys_to_keep))

        else:
            try:
                decision_process_output.extend(process_recognizer_result(result, result, 0, i, df_dict, keys_to_keep))
            except Exception as e:
                print(e)

    decision_process_output_str = '\n'.join(decision_process_output)

    print("decision_process_output_str:\n\n", decision_process_output_str)
    

    return decision_process_output_str

def anon_consistent_names(df):
    # ## Pick out common names and replace them with the same person value
    df_dict = df.to_dict(orient="list")

    analyzer = AnalyzerEngine()
    batch_analyzer = BatchAnalyzerEngine(analyzer_engine=analyzer)

    analyzer_results = batch_analyzer.analyze_dict(df_dict, language="en")
    analyzer_results = list(analyzer_results)

    # + tags=[]
    text = analyzer_results[3].value

    # + tags=[]
    recognizer_result = str(analyzer_results[3].recognizer_results)

    # + tags=[]
    recognizer_result

    # + tags=[]
    data_str = recognizer_result  # abbreviated for brevity

    # Adjusting the parse_dict function to handle trailing ']'
    # Splitting the main data string into individual list strings
    list_strs = data_str[1:-1].split('], [')

    def parse_dict(s):
        s = s.strip('[]')  # Removing any surrounding brackets
        items = s.split(', ')
        d = {}
        for item in items:
            key, value = item.split(': ')
            if key == 'score':
                d[key] = float(value)
            elif key in ['start', 'end']:
                d[key] = int(value)
            else:
                d[key] = value
        return d

    # Re-running the improved processing code

    result = []

    for lst_str in list_strs:
        # Splitting each list string into individual dictionary strings
        dict_strs = lst_str.split(', type: ')
        dict_strs = [dict_strs[0]] + ['type: ' + s for s in dict_strs[1:]]  # Prepending "type: " back to the split strings
        
        # Parsing each dictionary string
        dicts = [parse_dict(d) for d in dict_strs]
        result.append(dicts)

    #result

    # + tags=[]
    names = []

    for idx, paragraph in enumerate(text):
        paragraph_texts = []
        for dictionary in result[idx]:
            if dictionary['type'] == 'PERSON':
                paragraph_texts.append(paragraph[dictionary['start']:dictionary['end']])
        names.append(paragraph_texts)

    # + tags=[]
    # Flatten the list of lists and extract unique names
    unique_names = list(set(name for sublist in names for name in sublist))
    
    # + tags=[]
    fake_names = pd.Series(unique_names).apply(fake_first_name)

    # + tags=[]
    mapping_df = pd.DataFrame(data={"Unique names":unique_names,
                    "Fake names": fake_names})

    # + tags=[]
    # Convert mapping dataframe to dictionary
    # Convert mapping dataframe to dictionary, adding word boundaries for full-word match
    name_map = {r'\b' + k + r'\b': v for k, v in zip(mapping_df['Unique names'], mapping_df['Fake names'])}

    # + tags=[]
    name_map

    # + tags=[]
    scrubbed_df_consistent_names = df.replace(name_map, regex = True)

    # + tags=[]
    scrubbed_df_consistent_names

    return scrubbed_df_consistent_names

def anonymise_script(df, anon_strat, language:str, chosen_redact_entities:List[str], in_allow_list:List[str]=[], progress=Progress(track_tqdm=False)):

    print("Identifying personal information")
    analyse_tic = time.perf_counter()

    key_string = ""

    # DataFrame to dict
    df_dict = df.to_dict(orient="list")

    if in_allow_list:
        in_allow_list_flat = in_allow_list #[item for sublist in in_allow_list for item in sublist]
    else:
        in_allow_list_flat = []

    #analyzer = nlp_analyser #AnalyzerEngine()
    batch_analyzer = BatchAnalyzerEngine(analyzer_engine=nlp_analyser)

    anonymizer = AnonymizerEngine()#conflict_resolution=ConflictResolutionStrategy.MERGE_SIMILAR_OR_CONTAINED)

    batch_anonymizer = BatchAnonymizerEngine(anonymizer_engine = anonymizer)

    #print("Allow list:", in_allow_list)
    #print("Input data keys:", df_dict.keys())

    # Use custom analyzer to be able to track progress with Gradio
    analyzer_results = analyze_dict(batch_analyzer, df_dict, language=language, 
                                                            entities=chosen_redact_entities,
                                                            score_threshold=score_threshold,
                                                            return_decision_process=True,
                                                            allow_list=in_allow_list_flat)
    
    analyzer_results = list(analyzer_results)

    # Usage in the main function:
    decision_process_output_str = generate_decision_process_output(analyzer_results, df_dict)

    #print("decision_process_output_str:\n\n", decision_process_output_str)

    analyse_toc = time.perf_counter()
    analyse_time_out = f"Analysing the text took {analyse_toc - analyse_tic:0.1f} seconds."
    print(analyse_time_out)

    # Create faker function (note that it has to receive a value)
    fake = Faker("en_UK")

    def fake_first_name(x):
        return fake.first_name()

    # Set up the anonymization configuration WITHOUT DATE_TIME
    simple_replace_config = eval('{"DEFAULT": OperatorConfig("replace", {"new_value": "REDACTED"})}')
    replace_config = eval('{"DEFAULT": OperatorConfig("replace")}')
    redact_config = eval('{"DEFAULT": OperatorConfig("redact")}')
    hash_config = eval('{"DEFAULT": OperatorConfig("hash")}')
    mask_config = eval('{"DEFAULT": OperatorConfig("mask", {"masking_char":"*", "chars_to_mask":100, "from_end":True})}')
    people_encrypt_config = eval('{"PERSON": OperatorConfig("encrypt", {"key": key_string})}') # The encryption is using AES cypher in CBC mode and requires a cryptographic key as an input for both the encryption and the decryption.
    fake_first_name_config = eval('{"PERSON": OperatorConfig("custom", {"lambda": fake_first_name})}')

    if anon_strat == "replace with <REDACTED>": chosen_mask_config = simple_replace_config
    if anon_strat == "replace with <ENTITY_NAME>": chosen_mask_config = replace_config
    if anon_strat == "redact": chosen_mask_config = redact_config
    if anon_strat == "hash": chosen_mask_config = hash_config
    if anon_strat == "mask": chosen_mask_config = mask_config
    if anon_strat == "encrypt": 
        chosen_mask_config = people_encrypt_config
        # Generate a 128-bit AES key. Then encode the key using base64 to get a string representation
        key = secrets.token_bytes(16)  # 128 bits = 16 bytes 
        key_string = base64.b64encode(key).decode('utf-8')
    elif anon_strat == "fake_first_name": chosen_mask_config = fake_first_name_config

    # I think in general people will want to keep date / times
    keep_date_config = eval('{"DATE_TIME": OperatorConfig("keep")}')

    combined_config = {**chosen_mask_config, **keep_date_config}
    combined_config

    anonymizer_results = batch_anonymizer.anonymize_dict(analyzer_results, operators=combined_config)

    scrubbed_df = pd.DataFrame(anonymizer_results)
    
    return scrubbed_df, key_string, decision_process_output_str

def anon_wrapper_func(anon_file, anon_df, chosen_cols, out_file_paths, out_file_part, out_message, excel_sheet_name, anon_strat, language, chosen_redact_entities, in_allow_list, file_type, anon_xlsx_export_file_name, log_files_output_paths):

    def check_lists(list1, list2):
            return any(string in list2 for string in list1)
        
    def get_common_strings(list1, list2):
        """
        Finds the common strings between two lists.

        Args:
            list1: The first list of strings.
            list2: The second list of strings.

        Returns:
            A list containing the common strings.
        """
        common_strings = []
        for string in list1:
            if string in list2:
                common_strings.append(string)
        return common_strings

    # Check for chosen col, skip file if not found
    all_cols_original_order = list(anon_df.columns)

    any_cols_found = check_lists(chosen_cols, all_cols_original_order)

    if any_cols_found == False:
        out_message = "No chosen columns found in dataframe: " + out_file_part
        print(out_message)
    else:
        chosen_cols_in_anon_df = get_common_strings(chosen_cols, all_cols_original_order)

    # Split dataframe to keep only selected columns
    print("Remaining columns to redact:", chosen_cols_in_anon_df)
    
    anon_df_part = anon_df[chosen_cols_in_anon_df]
    anon_df_remain = anon_df.drop(chosen_cols_in_anon_df, axis = 1)
    
    # Anonymise the selected columns
    anon_df_part_out, key_string, decision_process_output_str = anonymise_script(anon_df_part, anon_strat, language, chosen_redact_entities, in_allow_list)
        
    # Rejoin the dataframe together
    anon_df_out = pd.concat([anon_df_part_out, anon_df_remain], axis = 1)
    anon_df_out = anon_df_out[all_cols_original_order]
    
    # Export file

    #  Rename anonymisation strategy for file path naming
    if anon_strat == "replace with <REDACTED>": anon_strat_txt = "redact_simple"
    elif anon_strat == "replace with <ENTITY_NAME>": anon_strat_txt = "redact_entity_type"
    else: anon_strat_txt = anon_strat

    # If the file is an xlsx, add a new sheet to the existing xlsx. Otherwise, write to csv
    if file_type == 'xlsx':

        anon_export_file_name = anon_xlsx_export_file_name

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        with pd.ExcelWriter(anon_xlsx_export_file_name, engine='openpyxl', mode='a') as writer:
            # Write each DataFrame to a different worksheet.
            anon_df_out.to_excel(writer, sheet_name=excel_sheet_name, index=None)

        decision_process_log_output_file = anon_xlsx_export_file_name + "_" + excel_sheet_name + "_decision_process_output.txt"
        with open(decision_process_log_output_file, "w") as f:
            f.write(decision_process_output_str)

    else:
        anon_export_file_name = output_folder + out_file_part + "_anon_" + anon_strat_txt + ".csv"
        anon_df_out.to_csv(anon_export_file_name, index = None)

        decision_process_log_output_file = anon_export_file_name + "_decision_process_output.txt"
        with open(decision_process_log_output_file, "w") as f:
            f.write(decision_process_output_str)

    out_file_paths.append(anon_export_file_name)
    log_files_output_paths.append(decision_process_log_output_file)

    # As files are created in a loop, there is a risk of duplicate file names being output. Use set to keep uniques.
    out_file_paths = list(set(out_file_paths))

    # Print result text to output text box if just anonymising open text
    if anon_file=='open_text':
        out_message = [anon_df_out['text'][0]]

    return out_file_paths, out_message, key_string, log_files_output_paths
       
def anonymise_data_files(file_paths:List[str], in_text:str, anon_strat:str, chosen_cols:List[str], language:str, chosen_redact_entities:List[str], in_allow_list:List[str]=None, latest_file_completed:int=0, out_message:list=[], out_file_paths:list = [], log_files_output_paths:list = [], in_excel_sheets:list=[], first_loop_state:bool=False, progress=Progress(track_tqdm=True)):
    
    tic = time.perf_counter()

    # If this is the first time around, set variables to 0/blank
    if first_loop_state==True:
        latest_file_completed = 0
        out_message = []
        out_file_paths = []

    # Load file
    # If out message or out_file_paths are blank, change to a list so it can be appended to
    if isinstance(out_message, str):
        out_message = [out_message]

    if not out_file_paths:
        out_file_paths = []
    

    if in_allow_list:
        in_allow_list_flat = in_allow_list #[item for sublist in in_allow_list for item in sublist]
    else:
        in_allow_list_flat = []
    
    anon_df = pd.DataFrame()
    #out_file_paths = []
    
    # Check if files and text exist
    if not file_paths:
        if in_text:
            file_paths=['open_text']
        else:
            out_message = "Please enter text or a file to redact."
            return out_message, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths
        
    # If we have already redacted the last file, return the input out_message and file list to the relevant components
    if latest_file_completed >= len(file_paths):
        print("Last file reached, returning files:", str(latest_file_completed))
        # Set to a very high number so as not to mess with subsequent file processing by the user
        latest_file_completed = 99
        final_out_message = '\n'.join(out_message)
        return final_out_message, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths
    
    file_path_loop = [file_paths[int(latest_file_completed)]]
        
    for anon_file in progress.tqdm(file_path_loop, desc="Anonymising files", unit = "file"):

        if anon_file=='open_text':
            anon_df = pd.DataFrame(data={'text':[in_text]})
            chosen_cols=['text']
            sheet_name = ""
            file_type = ""
            out_file_part = anon_file

            out_file_paths, out_message, key_string, log_files_output_paths = anon_wrapper_func(anon_file, anon_df, chosen_cols, out_file_paths, out_file_part, out_message, sheet_name, anon_strat, language, chosen_redact_entities, in_allow_list, file_type, "", log_files_output_paths)
        else:
            # If file is an xlsx, we are going to run through all the Excel sheets to anonymise them separately.
            file_type = detect_file_type(anon_file)
            print("File type is:", file_type)

            out_file_part = get_file_path_end(anon_file.name)
    
            if file_type == 'xlsx':
                print("Running through all xlsx sheets")
                #anon_xlsx = pd.ExcelFile(anon_file)
                if not in_excel_sheets:
                    out_message.append("No Excel sheets selected. Please select at least one to anonymise.")
                    continue

                anon_xlsx = pd.ExcelFile(anon_file)                

                # Create xlsx file:
                anon_xlsx_export_file_name = output_folder + out_file_part + "_redacted.xlsx"

                from openpyxl import Workbook

                wb = Workbook()
                wb.save(anon_xlsx_export_file_name)


                # Iterate through the sheet names
                for sheet_name in in_excel_sheets:
                    # Read each sheet into a DataFrame
                    if sheet_name not in anon_xlsx.sheet_names:
                        continue

                    anon_df = pd.read_excel(anon_file, sheet_name=sheet_name)

                    # Process the DataFrame (e.g., print its contents)
                    print(f"Sheet Name: {sheet_name}")
                    print(anon_df.head())  # Print the first few rows

                    
                    out_file_paths, out_message, key_string, log_files_output_paths  = anon_wrapper_func(anon_file, anon_df, chosen_cols, out_file_paths, out_file_part, out_message, sheet_name, anon_strat, language, chosen_redact_entities, in_allow_list, file_type,  anon_xlsx_export_file_name, log_files_output_paths)
                    
            else:
                sheet_name = ""
                anon_df = read_file(anon_file)
                out_file_part = get_file_path_end(anon_file.name)
                out_file_paths, out_message, key_string, log_files_output_paths = anon_wrapper_func(anon_file, anon_df, chosen_cols, out_file_paths, out_file_part, out_message, sheet_name, anon_strat, language, chosen_redact_entities, in_allow_list, file_type, "", log_files_output_paths)

        # Increase latest file completed count unless we are at the last file
        if latest_file_completed != len(file_paths):
            print("Completed file number:", str(latest_file_completed))
            latest_file_completed += 1 

        toc = time.perf_counter()
        out_time = f"in {toc - tic:0.1f} seconds."
        print(out_time)    
        
        if anon_strat == "encrypt":
            out_message.append(". Your decryption key is " + key_string + ".")

        out_message.append("Anonymisation of file '" + out_file_part + "' successfully completed in")

        out_message_out = '\n'.join(out_message)
        out_message_out = out_message_out + " " + out_time

        out_message_out = out_message_out + "\n\nGo to to the Redaction settings tab to see redaction logs. Please give feedback on the results below to help improve this app."
    
    return out_message_out, out_file_paths, out_file_paths, latest_file_completed, log_files_output_paths, log_files_output_paths
