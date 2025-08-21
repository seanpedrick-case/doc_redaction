from typing import List
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, EntityRecognizer, Pattern, RecognizerResult
from presidio_analyzer.nlp_engine import SpacyNlpEngine, NlpArtifacts, NerModelConfiguration
import spacy
from spacy.matcher import Matcher
from spaczz.matcher import FuzzyMatcher
spacy.prefer_gpu()
from spacy.cli.download import download
import Levenshtein
import re
import os
import requests
import gradio as gr
from tools.config import DEFAULT_LANGUAGE, TESSERACT_FOLDER

score_threshold = 0.001
custom_entities = ["TITLES", "UKPOSTCODE", "STREETNAME", "CUSTOM"]

# Create a class inheriting from SpacyNlpEngine
class LoadedSpacyNlpEngine(SpacyNlpEngine):
    def __init__(self, loaded_spacy_model, language_code: str):
        super().__init__(ner_model_configuration=NerModelConfiguration(labels_to_ignore=["CARDINAL", "ORDINAL"])) # Ignore non-relevant labels
        self.nlp = {language_code: loaded_spacy_model}

def _base_language_code(language: str) -> str:
    lang = _normalize_language_input(language)
    if "_" in lang:
        return lang.split("_")[0]
    return lang

def load_spacy_model(language: str = DEFAULT_LANGUAGE):
    """
    Load a spaCy model for the requested language and return it as `nlp`.

    Accepts common inputs like: "en", "en_lg", "en_sm", "de", "fr", "es", "it", "nl", "pt", "zh", "ja", "xx".
    Falls back through sensible candidates and will download if missing.
    """

    synonyms = {
        "english": "en",
        "catalan": "ca",
        "danish": "da",
        "german": "de",
        "french": "fr",
        "greek": "el",
        "finnish": "fi",
        "croatian": "hr",
        "lithuanian": "lt",
        "macedonian": "mk",
        "norwegian_bokmaal": "nb",
        "polish": "pl",
        "russian": "ru",
        "slovenian": "sl",
        "swedish": "sv",
        "dutch": "nl",
        "portuguese": "pt",
        "chinese": "zh",
        "japanese": "ja",
        "multilingual": "xx",
    }

    lang_norm = _normalize_language_input(language)
    lang_norm = synonyms.get(lang_norm, lang_norm)
    base_lang = _base_language_code(lang_norm)

    candidates_by_lang = {
        # English
        "en": [
            "en_core_web_lg",
            "en_core_web_trf",
            "en_core_web_md",
            "en_core_web_sm",
        ],
        "en_lg": ["en_core_web_lg"],
        "en_trf": ["en_core_web_trf"],
        "en_md": ["en_core_web_md"],
        "en_sm": ["en_core_web_sm"],

        # Major languages (news pipelines)
        "ca": ["ca_core_news_lg", "ca_core_news_md", "ca_core_news_sm"], # Catalan
        "da": ["da_core_news_lg", "da_core_news_md", "da_core_news_sm"], # Danish
        "de": ["de_core_news_lg", "de_core_news_md", "de_core_news_sm"], # German
        "el": ["el_core_news_lg", "el_core_news_md", "el_core_news_sm"], # Greek
        "es": ["es_core_news_lg", "es_core_news_md", "es_core_news_sm"], # Spanish
        "fi": ["fi_core_news_lg", "fi_core_news_md", "fi_core_news_sm"], # Finnish
        "fr": ["fr_core_news_lg", "fr_core_news_md", "fr_core_news_sm"], # French
        "hr": ["hr_core_news_lg", "hr_core_news_md", "hr_core_news_sm"], # Croatian
        "it": ["it_core_news_lg", "it_core_news_md", "it_core_news_sm"], # Italian
        "ja": ["ja_core_news_lg", "ja_core_news_md", "ja_core_news_sm"], # Japanese
        "ko": ["ko_core_news_lg", "ko_core_news_md", "ko_core_news_sm"], # Korean
        "lt": ["lt_core_news_lg", "lt_core_news_md", "lt_core_news_sm"], # Lithuanian
        "mk": ["mk_core_news_lg", "mk_core_news_md", "mk_core_news_sm"], # Macedonian
        "nb": ["nb_core_news_lg", "nb_core_news_md", "nb_core_news_sm"], # Norwegian Bokmål
        "nl": ["nl_core_news_lg", "nl_core_news_md", "nl_core_news_sm"], # Dutch
        "pl": ["pl_core_news_lg", "pl_core_news_md", "pl_core_news_sm"], # Polish
        "pt": ["pt_core_news_lg", "pt_core_news_md", "pt_core_news_sm"], # Portuguese
        "ro": ["ro_core_news_lg", "ro_core_news_md", "ro_core_news_sm"], # Romanian
        "ru": ["ru_core_news_lg", "ru_core_news_md", "ru_core_news_sm"], # Russian
        "sl": ["sl_core_news_lg", "sl_core_news_md", "sl_core_news_sm"], # Slovenian
        "sv": ["sv_core_news_lg", "sv_core_news_md", "sv_core_news_sm"], # Swedish 
        "uk": ["uk_core_news_lg", "uk_core_news_md", "uk_core_news_sm"], # Ukrainian
        "zh": ["zh_core_web_lg", "zh_core_web_mod", "zh_core_web_sm", "zh_core_web_trf"], # Chinese

        # Multilingual NER
        "xx": ["xx_ent_wiki_sm"],
    }

    if lang_norm in candidates_by_lang:
        candidates = candidates_by_lang[lang_norm]
    elif base_lang in candidates_by_lang:
        candidates = candidates_by_lang[base_lang]
    else:
        # Fallback to multilingual if unknown
        candidates = candidates_by_lang["xx"]

    last_error = None
    for candidate in candidates:
        # Try importable package first (fast-path when installed as a package)
        try:
            module = __import__(candidate)
            print(f"Successfully imported spaCy model: {candidate}")
            return module.load()
        except Exception as e:
            last_error = e

        # Try spacy.load if package is linked/installed
        try:
            nlp = spacy.load(candidate)
            print(f"Successfully loaded spaCy model via spacy.load: {candidate}")
            return nlp
        except Exception as e:
            last_error = e

        # Check if model is already downloaded before attempting to download
        try:
            # Try to load the model to see if it's already available
            nlp = spacy.load(candidate)
            print(f"Model {candidate} is already available, skipping download")
            return nlp
        except OSError:
            # Model not found, proceed with download
            pass
        except Exception as e:
            last_error = e
            continue

        # Attempt to download then load
        try:
            print(f"Downloading spaCy model: {candidate}")
            download(candidate)
            nlp = spacy.load(candidate)
            print(f"Successfully downloaded and loaded spaCy model: {candidate}")
            return nlp
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Failed to load spaCy model for language '{language}'. Last error: {last_error}")

# Language-aware spaCy model loader
def _normalize_language_input(language: str) -> str:
    return language.strip().lower().replace("-", "_")

# Update the global variables to use the new function
ACTIVE_LANGUAGE_CODE = _base_language_code(DEFAULT_LANGUAGE)
nlp = None # Placeholder, will be loaded in the create_nlp_analyser function below #load_spacy_model(DEFAULT_LANGUAGE)

def get_tesseract_lang_code(short_code:str):
    """
    Maps a two-letter language code to the corresponding Tesseract OCR code.

    Args:
        short_code (str): The two-letter language code (e.g., "en", "de").

    Returns:
        str or None: The Tesseract language code (e.g., "eng", "deu"),
                     or None if no mapping is found.
    """
    # Mapping from 2-letter codes to Tesseract 3-letter codes
    # Based on ISO 639-2/T codes.
    lang_map = {
        "en": "eng",
        "de": "deu",
        "fr": "fra",
        "es": "spa",
        "it": "ita",
        "nl": "nld",
        "pt": "por",
        "zh": "chi_sim",  # Mapping to Simplified Chinese by default
        "ja": "jpn",
        "ko": "kor",
        "lt": "lit",
        "mk": "mkd",
        "nb": "nor",
        "pl": "pol",
        "ro": "ron",
        "ru": "rus",
        "sl": "slv",
        "sv": "swe",
        "uk": "ukr"
    }

    return lang_map.get(short_code)

def download_tesseract_lang_pack(short_lang_code:str, tessdata_dir=TESSERACT_FOLDER + "/tessdata"):
    """
    Downloads a Tesseract language pack to a local directory.

    Args:
        lang_code (str): The short code for the language (e.g., "eng", "fra").
        tessdata_dir (str, optional): The directory to save the language pack.
                                     Defaults to "tessdata".
    """
    
    # Create the directory if it doesn't exist
    if not os.path.exists(tessdata_dir):
        os.makedirs(tessdata_dir)

    # Get the Tesseract language code
    lang_code = get_tesseract_lang_code(short_lang_code)

    if lang_code is None:
        raise ValueError(f"Language code {short_lang_code} not found in Tesseract language map")
    
    # Set the local file path
    file_path = os.path.join(tessdata_dir, f"{lang_code}.traineddata")
    
    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"Language pack {lang_code}.traineddata already exists at {file_path}")
        return file_path
    
    # Construct the URL for the language pack
    url = f"https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/{lang_code}.traineddata"

    # Download the file
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Successfully downloaded {lang_code}.traineddata to {file_path}")
        return file_path

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {lang_code}.traineddata: {e}")
        return None

#### Custom recognisers
def custom_word_list_recogniser(custom_list:List[str]=[]):
    # Create regex pattern, handling quotes carefully

    quote_str = '"'
    replace_str = '(?:"|"|")'

    custom_regex = '|'.join(
        rf'(?<!\w){re.escape(term.strip()).replace(quote_str, replace_str)}(?!\w)'
        for term in custom_list
    )
    #print(custom_regex)

    custom_pattern = Pattern(name="custom_pattern", regex=custom_regex, score = 1)
    
    custom_recogniser = PatternRecognizer(supported_entity="CUSTOM", name="CUSTOM", patterns = [custom_pattern], 
        global_regex_flags=re.DOTALL | re.MULTILINE | re.IGNORECASE)

    return custom_recogniser

# Initialise custom recogniser that will be overwritten later
custom_recogniser = custom_word_list_recogniser()

# Custom title recogniser
titles_list = ["Sir", "Ma'am", "Madam", "Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Ms.", "Miss", "Dr", "Dr.", "Professor"]
titles_regex = '\\b' + '\\b|\\b'.join(rf"{re.escape(title)}" for title in titles_list) + '\\b'
titles_pattern = Pattern(name="titles_pattern",regex=titles_regex, score = 1)
titles_recogniser = PatternRecognizer(supported_entity="TITLES", name="TITLES", patterns = [titles_pattern], 
    global_regex_flags=re.DOTALL | re.MULTILINE)

# %%
# Custom postcode recogniser

# Define the regex pattern in a Presidio `Pattern` object:
ukpostcode_pattern = Pattern(
    name="ukpostcode_pattern",
    regex=r"\b([A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}|GIR ?0AA)\b",
    score=1
)

# Define the recognizer with one or more patterns
ukpostcode_recogniser = PatternRecognizer(supported_entity="UKPOSTCODE", name = "UKPOSTCODE", patterns = [ukpostcode_pattern])

### Street name

def extract_street_name(text:str) -> str:
    """
    Extracts the street name and preceding word (that should contain at least one number) from the given text.

    """    
   
    street_types = [
    'Street', 'St', 'Boulevard', 'Blvd', 'Highway', 'Hwy', 'Broadway', 'Freeway',
    'Causeway', 'Cswy', 'Expressway', 'Way', 'Walk', 'Lane', 'Ln', 'Road', 'Rd',
    'Avenue', 'Ave', 'Circle', 'Cir', 'Cove', 'Cv', 'Drive', 'Dr', 'Parkway', 'Pkwy',
    'Park', 'Court', 'Ct', 'Square', 'Sq', 'Loop', 'Place', 'Pl', 'Parade', 'Estate',
    'Alley', 'Arcade', 'Avenue', 'Ave', 'Bay', 'Bend', 'Brae', 'Byway', 'Close', 'Corner', 'Cove',
    'Crescent', 'Cres', 'Cul-de-sac', 'Dell', 'Drive', 'Dr', 'Esplanade', 'Glen', 'Green', 'Grove', 'Heights', 'Hts',
    'Mews', 'Parade', 'Path', 'Piazza', 'Promenade', 'Quay', 'Ridge', 'Row', 'Terrace', 'Ter', 'Track', 'Trail', 'View', 'Villas',
    'Marsh', 'Embankment', 'Cut', 'Hill', 'Passage', 'Rise', 'Vale', 'Side'
    ]

    # Dynamically construct the regex pattern with all possible street types
    street_types_pattern = '|'.join(rf"{re.escape(street_type)}" for street_type in street_types)

    # The overall regex pattern to capture the street name and preceding word(s)

    pattern = rf'(?P<preceding_word>\w*\d\w*)\s*'
    pattern += rf'(?P<street_name>\w+\s*\b(?:{street_types_pattern})\b)'

    # Find all matches in text
    matches = re.finditer(pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE)

    start_positions = []
    end_positions = []

    for match in matches:
        preceding_word = match.group('preceding_word').strip()
        street_name = match.group('street_name').strip()
        start_pos = match.start()
        end_pos = match.end()
        #print(f"Start: {start_pos}, End: {end_pos}")
        #print(f"Preceding words: {preceding_word}")
        #print(f"Street name: {street_name}")

        start_positions.append(start_pos)
        end_positions.append(end_pos)

    return start_positions, end_positions

class StreetNameRecognizer(EntityRecognizer):

    def load(self) -> None:
        """No loading is required."""
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts) -> List[RecognizerResult]:
        """
        Logic for detecting a specific PII
        """

        start_pos, end_pos = extract_street_name(text)

        results = []

        for i in range(0, len(start_pos)):

            result = RecognizerResult(
                        entity_type="STREETNAME",
                        start = start_pos[i],
                        end = end_pos[i],
                        score= 1
                    )
        
            results.append(result)
        
        return results
    
street_recogniser = StreetNameRecognizer(supported_entities=["STREETNAME"])

## Custom fuzzy match recogniser for list of strings
def custom_fuzzy_word_list_regex(text:str, custom_list:List[str]=[]):
    # Create regex pattern, handling quotes carefully

    quote_str = '"'
    replace_str = '(?:"|"|")'

    custom_regex_pattern = '|'.join(
        rf'(?<!\w){re.escape(term.strip()).replace(quote_str, replace_str)}(?!\w)'
        for term in custom_list
    )

    # Find all matches in text
    matches = re.finditer(custom_regex_pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE)

    start_positions = []
    end_positions = []

    for match in matches:
        start_pos = match.start()
        end_pos = match.end()

        start_positions.append(start_pos)
        end_positions.append(end_pos)

    return start_positions, end_positions

def spacy_fuzzy_search(text: str, custom_query_list:List[str]=[], spelling_mistakes_max:int = 1, search_whole_phrase:bool=True, nlp=nlp, progress=gr.Progress(track_tqdm=True)):
    ''' Conduct fuzzy match on a list of text data.'''

    all_matches = []
    all_start_positions = []
    all_end_positions = []
    all_ratios = []

    #print("custom_query_list:", custom_query_list)

    if not text:
        out_message = "No text data found. Skipping page."
        print(out_message)
        return all_start_positions, all_end_positions

    for string_query in custom_query_list:

        #print("text:", text)
        #print("string_query:", string_query)

        query = nlp(string_query)

        if search_whole_phrase == False:
            # Keep only words that are not stop words
            token_query = [token.text for token in query if not token.is_space and not token.is_stop and not token.is_punct]

            spelling_mistakes_fuzzy_pattern = "FUZZY" + str(spelling_mistakes_max)

            #print("token_query:", token_query)

            if len(token_query) > 1:
                #pattern_lemma = [{"LEMMA": {"IN": query}}]
                pattern_fuzz = [{"TEXT": {spelling_mistakes_fuzzy_pattern: {"IN": token_query}}}]
            else:
                #pattern_lemma = [{"LEMMA": query[0]}]
                pattern_fuzz = [{"TEXT": {spelling_mistakes_fuzzy_pattern: token_query[0]}}]

            matcher = Matcher(nlp.vocab)        
            matcher.add(string_query, [pattern_fuzz])
            #matcher.add(string_query, [pattern_lemma])
        
        else:
            # If matching a whole phrase, use Spacy PhraseMatcher, then consider similarity after using Levenshtein distance.
            #tokenised_query = [string_query.lower()]
            # If you want to match the whole phrase, use phrase matcher
            matcher = FuzzyMatcher(nlp.vocab)
            patterns = [nlp.make_doc(string_query)]  # Convert query into a Doc object
            matcher.add("PHRASE", patterns, [{"ignore_case": True}])

        batch_size = 256
        docs = nlp.pipe([text], batch_size=batch_size)

        # Get number of matches per doc
        for doc in docs: #progress.tqdm(docs, desc = "Searching text", unit = "rows"):
            matches = matcher(doc)
            match_count = len(matches)

            # If considering each sub term individually, append match. If considering together, consider weight of the relevance to that of the whole phrase.
            if search_whole_phrase==False:
                all_matches.append(match_count)

                for match_id, start, end in matches:
                    span = str(doc[start:end]).strip()
                    query_search = str(query).strip()
                    #print("doc:", doc)
                    #print("span:", span)
                    #print("query_search:", query_search)
                    
                    # Convert word positions to character positions
                    start_char = doc[start].idx  # Start character position
                    end_char = doc[end - 1].idx + len(doc[end - 1])  # End character position

                    # The positions here are word position, not character position
                    all_matches.append(match_count)
                    all_start_positions.append(start_char)
                    all_end_positions.append(end_char)
                
            else:
                for match_id, start, end, ratio, pattern in matches:
                    span = str(doc[start:end]).strip()
                    query_search = str(query).strip()
                    #print("doc:", doc)
                    #print("span:", span)
                    #print("query_search:", query_search)
                    
                    # Calculate Levenshtein distance. Only keep matches with less than specified number of spelling mistakes
                    distance = Levenshtein.distance(query_search.lower(), span.lower())

                    #print("Levenshtein distance:", distance)
                    
                    if distance > spelling_mistakes_max:                                       
                        match_count = match_count - 1
                    else:
                        # Convert word positions to character positions
                        start_char = doc[start].idx  # Start character position
                        end_char = doc[end - 1].idx + len(doc[end - 1])  # End character position

                        #print("start_char:", start_char)
                        #print("end_char:", end_char)

                        all_matches.append(match_count)
                        all_start_positions.append(start_char)
                        all_end_positions.append(end_char)
                        all_ratios.append(ratio)                        


    return all_start_positions, all_end_positions

class CustomWordFuzzyRecognizer(EntityRecognizer):
    def __init__(self, supported_entities: List[str], custom_list: List[str] = [], spelling_mistakes_max: int = 1, search_whole_phrase: bool = True):
        super().__init__(supported_entities=supported_entities)
        self.custom_list = custom_list  # Store the custom_list as an instance attribute
        self.spelling_mistakes_max = spelling_mistakes_max  # Store the max spelling mistakes
        self.search_whole_phrase = search_whole_phrase  # Store the search whole phrase flag

    def load(self) -> None:
        """No loading is required."""
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts) -> List[RecognizerResult]:
        """
        Logic for detecting a specific PII
        """
        start_pos, end_pos = spacy_fuzzy_search(text, self.custom_list, self.spelling_mistakes_max, self.search_whole_phrase)  # Pass new parameters

        results = []

        for i in range(0, len(start_pos)):
            result = RecognizerResult(
                entity_type="CUSTOM_FUZZY",
                start=start_pos[i],
                end=end_pos[i],
                score=1
            )
            results.append(result)

        return results
    
custom_list_default = []
custom_word_fuzzy_recognizer = CustomWordFuzzyRecognizer(supported_entities=["CUSTOM_FUZZY"], custom_list=custom_list_default)


# Pass the loaded model to the new LoadedSpacyNlpEngine
loaded_nlp_engine = LoadedSpacyNlpEngine(loaded_spacy_model = nlp, language_code = ACTIVE_LANGUAGE_CODE)


def create_nlp_analyser(language: str = DEFAULT_LANGUAGE, custom_list: List[str] = None, 
                       spelling_mistakes_max: int = 1, search_whole_phrase: bool = True, existing_nlp_analyser: AnalyzerEngine = None):
    """
    Create an nlp_analyser object based on the specified language input.
    
    Args:
        language (str): Language code (e.g., "en", "de", "fr", "es", etc.)
        custom_list (List[str], optional): List of custom words to recognize. Defaults to None.
        spelling_mistakes_max (int, optional): Maximum number of spelling mistakes for fuzzy matching. Defaults to 1.
        search_whole_phrase (bool, optional): Whether to search for whole phrases or individual words. Defaults to True.
    
    Returns:
        AnalyzerEngine: Configured nlp_analyser object with custom recognizers
    """

    if existing_nlp_analyser is None:   
        pass
    else:
        if existing_nlp_analyser.supported_languages[0] == language:
            nlp_analyser = existing_nlp_analyser
            print(f"Using existing nlp_analyser for {language}")
            return nlp_analyser

    # Load spaCy model for the specified language
    nlp_model = load_spacy_model(language)
    
    # Get base language code
    base_lang_code = _base_language_code(language)
    
    # Create custom recognizers
    if custom_list is None:
        custom_list = []
    
    custom_recogniser = custom_word_list_recogniser(custom_list)
    custom_word_fuzzy_recognizer = CustomWordFuzzyRecognizer(
        supported_entities=["CUSTOM_FUZZY"], 
        custom_list=custom_list,
        spelling_mistakes_max=spelling_mistakes_max,
        search_whole_phrase=search_whole_phrase
    )
    
    # Create NLP engine with loaded model
    loaded_nlp_engine = LoadedSpacyNlpEngine(
        loaded_spacy_model=nlp_model, 
        language_code=base_lang_code
    )
    
    # Create analyzer engine
    nlp_analyser = AnalyzerEngine(
        nlp_engine=loaded_nlp_engine,
        default_score_threshold=score_threshold,
        supported_languages=[base_lang_code],
        log_decision_process=False,
    )
    
    # Add custom recognizers to nlp_analyser
    nlp_analyser.registry.add_recognizer(custom_recogniser)
    nlp_analyser.registry.add_recognizer(custom_word_fuzzy_recognizer)
    
    # Add language-specific recognizers for English
    if base_lang_code == "en":
        nlp_analyser.registry.add_recognizer(street_recogniser)
        nlp_analyser.registry.add_recognizer(ukpostcode_recogniser)
        nlp_analyser.registry.add_recognizer(titles_recogniser)
    
    return nlp_analyser

# Create the default nlp_analyser using the new function
nlp_analyser = create_nlp_analyser(DEFAULT_LANGUAGE)


