import re
import secrets
import base64
import time
import pandas as pd

from faker import Faker

from gradio import Progress
from typing import List

from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine
from presidio_anonymizer import AnonymizerEngine, BatchAnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from tools.helper_functions import output_folder, get_file_path_end, read_file
from tools.load_spacy_model_custom_recognisers import nlp_analyser, score_threshold

# Use custom version of analyze_dict to be able to track progress
from tools.presidio_analyzer_custom import analyze_dict


fake = Faker("en_UK")
def fake_first_name(x):
    return fake.first_name()

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

def anonymise_script(df, anon_strat, language:str, chosen_redact_entities:List[str], allow_list:List[str]=[], progress=Progress(track_tqdm=False)):
    # DataFrame to dict
    df_dict = df.to_dict(orient="list")

    if allow_list:
        allow_list_flat = [item for sublist in allow_list for item in sublist]

    #analyzer = nlp_analyser #AnalyzerEngine()
    batch_analyzer = BatchAnalyzerEngine(analyzer_engine=nlp_analyser)

    anonymizer = AnonymizerEngine()

    batch_anonymizer = BatchAnonymizerEngine(anonymizer_engine = anonymizer)

    # analyzer_results = batch_analyzer.analyze_dict(df_dict, language=language, 
    #                                                         entities=chosen_redact_entities,
    #                                                         score_threshold=score_threshold,
    #                                                         return_decision_process=False,
    #                                                         allow_list=allow_list_flat)

    print("Identifying personal information")
    analyse_tic = time.perf_counter()

    print("Allow list:", allow_list)

    # Use custom analyzer to be able to track progress with Gradio
    analyzer_results = analyze_dict(batch_analyzer, df_dict, language=language, 
                                                            entities=chosen_redact_entities,
                                                            score_threshold=score_threshold,
                                                            return_decision_process=False,
                                                            allow_list=allow_list_flat)
    analyzer_results = list(analyzer_results)
    #analyzer_results

    analyse_toc = time.perf_counter()
    analyse_time_out = f"Analysing the text took {analyse_toc - analyse_tic:0.1f} seconds."
    print(analyse_time_out)

    # Generate a 128-bit AES key. Then encode the key using base64 to get a string representation
    key = secrets.token_bytes(16)  # 128 bits = 16 bytes 
    key_string = base64.b64encode(key).decode('utf-8')

    # Create faker function (note that it has to receive a value)
    
    fake = Faker("en_UK")

    def fake_first_name(x):
        return fake.first_name()

    # Set up the anonymization configuration WITHOUT DATE_TIME
    replace_config = eval('{"DEFAULT": OperatorConfig("replace")}')
    redact_config = eval('{"DEFAULT": OperatorConfig("redact")}')
    hash_config = eval('{"DEFAULT": OperatorConfig("hash")}')
    mask_config = eval('{"DEFAULT": OperatorConfig("mask", {"masking_char":"*", "chars_to_mask":100, "from_end":True})}')
    people_encrypt_config = eval('{"PERSON": OperatorConfig("encrypt", {"key": key_string})}') # The encryption is using AES cypher in CBC mode and requires a cryptographic key as an input for both the encryption and the decryption.
    fake_first_name_config = eval('{"PERSON": OperatorConfig("custom", {"lambda": fake_first_name})}')


    if anon_strat == "replace": chosen_mask_config = replace_config
    if anon_strat == "redact": chosen_mask_config = redact_config
    if anon_strat == "hash": chosen_mask_config = hash_config
    if anon_strat == "mask": chosen_mask_config = mask_config
    if anon_strat == "encrypt": chosen_mask_config = people_encrypt_config
    elif anon_strat == "fake_first_name": chosen_mask_config = fake_first_name_config

    # I think in general people will want to keep date / times
    keep_date_config = eval('{"DATE_TIME": OperatorConfig("keep")}')

    combined_config = {**chosen_mask_config, **keep_date_config}
    combined_config

    anonymizer_results = batch_anonymizer.anonymize_dict(analyzer_results, operators=combined_config)

    scrubbed_df = pd.DataFrame(anonymizer_results)

    # Create reporting message
    out_message = "Successfully anonymised"
    
    if anon_strat == "encrypt":
        out_message = out_message + ". Your decryption key is " + key_string + "."
    
    return scrubbed_df, out_message

def do_anonymise(in_file, in_text:str, anon_strat:str, chosen_cols:List[str], language:str, chosen_redact_entities:List[str], allow_list:List[str]=None, progress=Progress(track_tqdm=True)):
    
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

    # Load file
    
    anon_df = pd.DataFrame()
    out_files_list = []
    
    # Check if files and text exist
    if not in_file:
        if in_text:
            in_file=['open_text']
        else:
            out_message = "Please enter text or a file to redact."
            return out_message, None
    
    for match_file in progress.tqdm(in_file, desc="Anonymising files", unit = "file"):

        if match_file=='open_text':
            anon_df = pd.DataFrame(data={'text':[in_text]})
            chosen_cols=['text']
            out_file_part = match_file
        else:
            anon_df = read_file(match_file)
            out_file_part = get_file_path_end(match_file.name)

        

        # Check for chosen col, skip file if not found
        all_cols_original_order = list(anon_df.columns)

        any_cols_found = check_lists(chosen_cols, all_cols_original_order)

        if any_cols_found == False:
            out_message = "No chosen columns found in dataframe: " + out_file_part
            print(out_message)
            continue
        else:
            chosen_cols_in_anon_df = get_common_strings(chosen_cols, all_cols_original_order)

        # Split dataframe to keep only selected columns
        print("Remaining columns to redact:", chosen_cols_in_anon_df)
        
        anon_df_part = anon_df[chosen_cols_in_anon_df]
        anon_df_remain = anon_df.drop(chosen_cols_in_anon_df, axis = 1)
        
        # Anonymise the selected columns
        anon_df_part_out, out_message = anonymise_script(anon_df_part, anon_strat, language, chosen_redact_entities, allow_list)
            
        # Rejoin the dataframe together
        anon_df_out = pd.concat([anon_df_part_out, anon_df_remain], axis = 1)
        anon_df_out = anon_df_out[all_cols_original_order]
        
        # Export file
        
        
        # out_file_part = re.sub(r'\.csv', '', match_file.name)

        anon_export_file_name = output_folder + out_file_part + "_anon_" + anon_strat + ".csv"
        
        anon_df_out.to_csv(anon_export_file_name, index = None)

        out_files_list.append(anon_export_file_name)

        # Print result text to output text box if just anonymising open text
        if match_file=='open_text':
            out_message = anon_df_out['text'][0]
    
    return out_message, out_files_list
