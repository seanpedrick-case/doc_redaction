import pandas as pd
#import argparse
#import glob
import os
import re
from tools.helper_functions import OUTPUT_FOLDER
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
#import spacy
import numpy as np
import random
import string
from typing import List
from gradio import Progress

import en_core_web_lg #en_core_web_sm
nlp = en_core_web_lg.load()
#from tqdm import tqdm

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')

similarity_threshold = 0.9


def combine_ocr_output_text(input_files:List[str], output_folder:str=OUTPUT_FOLDER):
    """
    Combines text from multiple CSV files containing page and text columns.
    Groups text by file and page number, concatenating text within these groups.
    
    Args:
        input_files (list): List of paths to CSV files
    
    Returns:
        pd.DataFrame: Combined dataframe with columns [file, page, text]
    """
    all_data = []
    output_files = []

    if isinstance(input_files, str):
        file_paths_list = [input_files]
    else:
        file_paths_list = input_files
    
    for file in file_paths_list:

        if isinstance(file, str):
            file_path = file
        else:
            file_path = file.name

        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        if 'page' not in df.columns or 'text' not in df.columns:
            print(f"Warning: Skipping {file_path} - missing required columns 'page' and 'text'")
            continue

        df['text'] = df['text'].fillna('').astype(str)
        
        # Group by page and concatenate text
        grouped = df.groupby('page')['text'].apply(' '.join).reset_index()
        
        # Add filename column
        grouped['file'] = os.path.basename(file_path)
        
        all_data.append(grouped)
    
    if not all_data:
        raise ValueError("No valid CSV files were processed")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns
    combined_df = combined_df[['file', 'page', 'text']]

    output_combined_file_path = output_folder + "combined_ocr_output_files.csv"
    combined_df.to_csv(output_combined_file_path, index=None)

    output_files.append(output_combined_file_path)
    
    return combined_df, output_files

def process_data(df:pd.DataFrame, column:str):
    '''
    Clean and stem text columns in a data frame
    '''
    
    def _clean_text(raw_text):
        # Remove HTML tags
        clean = re.sub(r'<.*?>', '', raw_text)
        # clean = re.sub(r'&nbsp;', ' ', clean)
        # clean = re.sub(r'\r\n', ' ', clean)
        # clean = re.sub(r'&lt;', ' ', clean)
        # clean = re.sub(r'&gt;', ' ', clean)
        # clean = re.sub(r'<strong>', ' ', clean)
        # clean = re.sub(r'</strong>', ' ', clean)

        # Replace non-breaking space \xa0 with a space
        # clean = clean.replace(u'\xa0', u' ')
        # Remove extra whitespace
        clean = ' '.join(clean.split())

        # # Tokenize the text
        # words = word_tokenize(clean.lower())

        # # Remove punctuation and numbers
        # words = [word for word in words if word.isalpha()]

        # # Remove stopwords
        # words = [word for word in words if word not in stop_words]

        # Join the cleaned words back into a string
        return clean

    # Function to apply lemmatization and remove stopwords
    def _apply_lemmatization(text):
        doc = nlp(text)
        # Keep only alphabetic tokens and remove stopwords
        lemmatized_words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        return ' '.join(lemmatized_words)
    
    df['text_clean'] = df[column].apply(_clean_text)

    df['text_clean'] = df['text_clean'].apply(_apply_lemmatization)
    
    return df

def identify_similar_pages(input_files: List[str], similarity_threshold: float = 0.9, output_folder:str=OUTPUT_FOLDER, progress=Progress(track_tqdm=True)):
    output_paths = []
    
    progress(0.1, desc="Cleaning input texts")

    # Load and clean data
    df, output_files = combine_ocr_output_text(input_files)
    output_paths.extend(output_files)
    df = process_data(df, 'text')  # Assume this returns 'text_clean', 'file', and 'page' columns

    # Vectorize text
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['text_clean'])

    progress(0.3, desc="Calculating text similarity")

    # Compute sparse cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=False)  # Keep sparse format

    # Extract indices of similar pages above threshold
    coo_matrix = similarity_matrix.tocoo()
    similar_pages = np.array([(i, j, v) for i, j, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data) if v > similarity_threshold])

    if similar_pages.size == 0:
        return pd.DataFrame(), output_paths  # Return empty if no matches
    
    

    # Create a DataFrame for similar pairs
    similarity_df = pd.DataFrame(similar_pages, columns=['Page1_Index', 'Page2_Index', 'Similarity_Score'])
    
    # Remove duplicate pairs (keep one direction)
    similarity_df = similarity_df[similarity_df['Page1_Index'] < similarity_df['Page2_Index']]

    progress(0.8, desc="Mapping back results")
    # Map indices to metadata
    # index_map = df[['file', 'page', 'text']].to_dict(orient='index')
    # similarity_df['Page1_File'] = similarity_df['Page1_Index'].map(lambda x: index_map[x]['file'])
    # similarity_df['Page2_File'] = similarity_df['Page2_Index'].map(lambda x: index_map[x]['file'])
    # similarity_df['Page1_Page'] = similarity_df['Page1_Index'].map(lambda x: index_map[x]['page'])
    # similarity_df['Page2_Page'] = similarity_df['Page2_Index'].map(lambda x: index_map[x]['page'])
    # similarity_df['Page1_Text'] = similarity_df['Page1_Index'].map(lambda x: index_map[x]['text'][0:200])
    # similarity_df['Page2_Text'] = similarity_df['Page2_Index'].map(lambda x: index_map[x]['text'][0:200])

    # Create a DataFrame with the metadata
    metadata_df = df[['file', 'page', 'text']].reset_index()

    # Merge to get the metadata for Page1
    similarity_df = similarity_df.merge(metadata_df, left_on='Page1_Index', right_on='index', suffixes=('', '_Page1'))
    similarity_df = similarity_df.rename(columns={'file': 'Page1_File', 'page': 'Page1_Page', 'text': 'Page1_Text'})

    # Merge to get the metadata for Page2
    similarity_df = similarity_df.merge(metadata_df, left_on='Page2_Index', right_on='index', suffixes=('', '_Page2'))
    similarity_df = similarity_df.rename(columns={'file': 'Page2_File', 'page': 'Page2_Page', 'text': 'Page2_Text'})

    # Optionally, drop the index columns if not needed
    #similarity_df = similarity_df.drop(columns=['index_Page1', 'index_Page2'])


    similarity_df["Similarity_Score"] = similarity_df["Similarity_Score"].round(3)

    # Sort results
    similarity_df_out = similarity_df[['Page1_File', 'Page1_Page', 'Page2_File', 'Page2_Page', 'Similarity_Score', 'Page1_Text', 'Page2_Text']]
    similarity_df_out = similarity_df_out.sort_values(["Page1_File", "Page1_Page", "Page2_File", "Page2_Page", "Similarity_Score"], ascending=[True, True, True, True, False])

    similarity_df_out['Page1_Text'] = similarity_df_out['Page1_Text'][0:100]
    similarity_df_out['Page2_Text'] = similarity_df_out['Page2_Text'][0:100]

    progress(0.8, desc="Saving output files")

    # Save results
    similarity_file_output_path = output_folder + 'page_similarity_results.csv'
    similarity_df_out.to_csv(similarity_file_output_path, index=False)
    output_paths.append(similarity_file_output_path)

    # Save per-file redaction lists
    for redact_file in similarity_df_out['Page2_File'].unique():
        output_file_name = output_folder + redact_file + "_whole_page.csv"
        whole_pages_to_redact_df = similarity_df_out.loc[similarity_df_out['Page2_File'] == redact_file, ['Page2_Page']].drop_duplicates(['Page2_Page']).sort_values('Page2_Page')
        whole_pages_to_redact_df.to_csv(output_file_name, header=False, index=False)
        output_paths.append(output_file_name)

    return similarity_df_out, output_paths

# Perturb text
# Apply the perturbation function with a 10% error probability
def perturb_text_with_errors(series:pd.Series):

    def _perturb_text(text, error_probability=0.1):
        words = text.split()  # Split text into words
        perturbed_words = []
        
        for word in words:
            if random.random() < error_probability:  # Add a random error
                perturbation_type = random.choice(['char_error', 'extra_space', 'extra_punctuation'])
                
                if perturbation_type == 'char_error':  # Introduce a character error
                    idx = random.randint(0, len(word) - 1)
                    char = random.choice(string.ascii_lowercase)  # Add a random letter
                    word = word[:idx] + char + word[idx:]
                
                elif perturbation_type == 'extra_space':  # Add extra space around a word
                    word = ' ' + word + ' '
                
                elif perturbation_type == 'extra_punctuation':  # Add punctuation to the word
                    punctuation = random.choice(string.punctuation)
                    idx = random.randint(0, len(word))  # Insert punctuation randomly
                    word = word[:idx] + punctuation + word[idx:]
            
            perturbed_words.append(word)
        
        return ' '.join(perturbed_words)

    series = series.apply(lambda x: _perturb_text(x, error_probability=0.1))

    return series
