# %%
from typing import List
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, EntityRecognizer, Pattern, RecognizerResult
from presidio_analyzer.nlp_engine import SpacyNlpEngine, NlpArtifacts
import spacy
spacy.prefer_gpu()
from spacy.cli.download import download
import re

# %%
model_name = "en_core_web_lg" #"en_core_web_trf"
score_threshold = 0.001

# %% [markdown]
# #### Custom recognisers

# %%
# Custom title recogniser
titles_list = ["Sir", "Ma'am", "Madam", "Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Ms.", "Miss", "Dr", "Dr.", "Professor"]
titles_regex = '\\b' + '\\b|\\b'.join(rf"{re.escape(title)}" for title in titles_list) + '\\b'
titles_pattern = Pattern(name="titles_pattern",regex=titles_regex, score = 1)
titles_recogniser = PatternRecognizer(supported_entity="TITLES", patterns = [titles_pattern],
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
ukpostcode_recogniser = PatternRecognizer(supported_entity="UKPOSTCODE", patterns = [ukpostcode_pattern])

# %%
# Examples for testing

#text = "I live in 510 Broad st SE5 9NG ."

#numbers_result = ukpostcode_recogniser.analyze(text=text, entities=["UKPOSTCODE"])
#print("Result:")
#print(numbers_result)

# %%
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
    matches = re.finditer(pattern, text, re.IGNORECASE)

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


# %%
# Some examples for testing

#text = "1234 Main Street, 5678 Oak Rd, 9ABC Elm Blvd, 42 Eagle st."
#text = "Roberto lives in Five 10 Broad st in Oregon"
#text = "Roberto lives in 55 Oregon Square"
#text = "There is 51a no way I will do that"
#text = "I am writing to apply for"

#extract_street_name(text)

# %%
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

# %%
# Create a class inheriting from SpacyNlpEngine
class LoadedSpacyNlpEngine(SpacyNlpEngine):
    def __init__(self, loaded_spacy_model):
        super().__init__()
        self.nlp = {"en": loaded_spacy_model}

# %%
# Load spacy model
try:
	import en_core_web_lg
	nlp = en_core_web_lg.load()
	print("Successfully imported spaCy model")

except:
	download("en_core_web_lg")
	nlp = spacy.load("en_core_web_lg")
	print("Successfully downloaded and imported spaCy model")

# Pass the loaded model to the new LoadedSpacyNlpEngine
loaded_nlp_engine = LoadedSpacyNlpEngine(loaded_spacy_model = nlp)



# %%
nlp_analyser = AnalyzerEngine(nlp_engine=loaded_nlp_engine,
                default_score_threshold=score_threshold,
                supported_languages=["en"],
                log_decision_process=False,
                )

# %%
nlp_analyser.registry.add_recognizer(street_recogniser)
nlp_analyser.registry.add_recognizer(ukpostcode_recogniser)
nlp_analyser.registry.add_recognizer(titles_recogniser)

