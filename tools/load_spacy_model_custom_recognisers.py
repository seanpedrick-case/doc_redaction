# %%
from typing import List
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, EntityRecognizer, Pattern, RecognizerResult
from presidio_analyzer.nlp_engine import SpacyNlpEngine, NlpArtifacts
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spaczz.matcher import FuzzyMatcher
spacy.prefer_gpu()
from spacy.cli.download import download
import Levenshtein
import re
import gradio as gr

model_name = "en_core_web_sm" #"en_core_web_trf"
score_threshold = 0.001
custom_entities = ["TITLES", "UKPOSTCODE", "STREETNAME", "CUSTOM"]

#Load spacy model
try:
	import en_core_web_sm
	nlp = en_core_web_sm.load()
	print("Successfully imported spaCy model")

except:
	download(model_name)
	nlp = spacy.load(model_name)
	print("Successfully downloaded and imported spaCy model", model_name)

# #### Custom recognisers
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
        out_message = "Prepared data not found. Have you clicked 'Load data' above to prepare a search index?"
        print(out_message)
        return out_message, None  

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
                    print("doc:", doc)
                    print("span:", span)
                    print("query_search:", query_search)
                    
                    # Calculate Levenshtein distance. Only keep matches with less than specified number of spelling mistakes
                    distance = Levenshtein.distance(query_search.lower(), span.lower())

                    print("Levenshtein distance:", distance)
                    
                    if distance > spelling_mistakes_max:                                       
                        match_count = match_count - 1
                    else:
                        # Convert word positions to character positions
                        start_char = doc[start].idx  # Start character position
                        end_char = doc[end - 1].idx + len(doc[end - 1])  # End character position

                        print("start_char:", start_char)
                        print("end_char:", end_char)

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

# Create a class inheriting from SpacyNlpEngine
class LoadedSpacyNlpEngine(SpacyNlpEngine):
    def __init__(self, loaded_spacy_model):
        super().__init__()
        self.nlp = {"en": loaded_spacy_model}

# Pass the loaded model to the new LoadedSpacyNlpEngine
loaded_nlp_engine = LoadedSpacyNlpEngine(loaded_spacy_model = nlp)


nlp_analyser = AnalyzerEngine(nlp_engine=loaded_nlp_engine,
                default_score_threshold=score_threshold,
                supported_languages=["en"],
                log_decision_process=False,
                )

# Add custom recognisers to nlp_analyser
nlp_analyser.registry.add_recognizer(street_recogniser)
nlp_analyser.registry.add_recognizer(ukpostcode_recogniser)
nlp_analyser.registry.add_recognizer(titles_recogniser)
nlp_analyser.registry.add_recognizer(custom_recogniser)
nlp_analyser.registry.add_recognizer(custom_word_fuzzy_recognizer)

