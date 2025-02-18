import pandas as pd
import argparse
import glob
import os
import re
from tools.helper_functions import output_folder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import random
import string
from typing import List

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

similarity_threshold = 0.9

stop_words = set(stopwords.words('english'))
# List of words to remove from the stopword set
#words_to_remove = ['no', 'nor', 'not', 'don', 'don't', 'wasn', 'wasn't', 'weren', 'weren't', "don't", "wasn't", "weren't"]

# Remove the specified words from the stopwords set
#for word in words_to_remove:
#    stop_words.discard(word.lower())
    
stemmer = PorterStemmer()
vectorizer = TfidfVectorizer()

def combine_ocr_output_text(input_files):
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

def process_data(df, column:str):
    '''
    Clean and stem text columns in a data frame
    '''
    
    def _clean_text(raw_text):
        # Remove HTML tags
        clean = re.sub(r'<.*?>', '', raw_text)
        clean = re.sub(r'&nbsp;', ' ', clean)
        clean = re.sub(r'\r\n', ' ', clean)
        clean = re.sub(r'&lt;', ' ', clean)
        clean = re.sub(r'&gt;', ' ', clean)
        clean = re.sub(r'<strong>', ' ', clean)
        clean = re.sub(r'</strong>', ' ', clean)

        # Replace non-breaking space \xa0 with a space
        clean = clean.replace(u'\xa0', u' ')
        # Remove extra whitespace
        clean = ' '.join(clean.split())

        # Tokenize the text
        words = word_tokenize(clean.lower())

        # Remove punctuation and numbers
        words = [word for word in words if word.isalpha()]

        # Remove stopwords
        words = [word for word in words if word not in stop_words]

        # Join the cleaned words back into a string
        return ' '.join(words)

    # Function to apply stemming
    def _apply_stemming(text):
        # Tokenize the text
        words = word_tokenize(text.lower())
        
        # Apply stemming to each word
        stemmed_words = [stemmer.stem(word) for word in words]
        
        # Join the stemmed words back into a single string
        return ' '.join(stemmed_words)




    df['text_clean'] = df[column].apply(_clean_text)
    df['text_clean'] = df['text_clean'].apply(_apply_stemming)
    
    return df

def identify_similar_pages(input_files:List[str]):

    output_paths = []

    df, output_files = combine_ocr_output_text(input_files)

    output_paths.extend(output_files)

    # Clean text
    df = process_data(df, 'text')

    # Vectorise text
    tfidf_matrix = vectorizer.fit_transform(df['text_clean'])

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Find the indices of the most similar pages
    np.fill_diagonal(similarity_matrix, 0)  # Ignore self-comparisons
    similar_pages = np.argwhere(similarity_matrix > similarity_threshold)  # Threshold of similarity

    #print(similar_pages)

    # Create a DataFrame for similar pairs and their scores
    similarity_df = pd.DataFrame({
        'Page1_Index': similar_pages[:, 0],
        'Page2_Index': similar_pages[:, 1],
        'Page1_File': similar_pages[:, 0],
        'Page2_File': similar_pages[:, 1],
        'Similarity_Score': similarity_matrix[similar_pages[:, 0], similar_pages[:, 1]]
    })

    # Filter out duplicate pairs (keep only one direction)
    similarity_df = similarity_df[similarity_df['Page1_Index'] < similarity_df['Page2_Index']]

    # Map the indices to their corresponding text and metadata
    similarity_df['Page1_File'] = similarity_df['Page1_File'].map(df['file'])
    similarity_df['Page2_File'] = similarity_df['Page2_File'].map(df['file'])

    similarity_df['Page1_Page'] = similarity_df['Page1_Index'].map(df['page'])
    similarity_df['Page2_Page'] = similarity_df['Page2_Index'].map(df['page'])

    similarity_df['Page1_Text'] = similarity_df['Page1_Index'].map(df['text'])
    similarity_df['Page2_Text'] = similarity_df['Page2_Index'].map(df['text'])

    similarity_df_out = similarity_df[['Page1_File', 'Page1_Page', 'Page2_File', 'Page2_Page', 'Similarity_Score', 'Page1_Text', 'Page2_Text']]
    similarity_df_out = similarity_df_out.sort_values(["Page1_File", "Page1_Page", "Page2_File", "Page2_Page", "Similarity_Score"], ascending=[True, True, True, True, False])

    # Save detailed results to a CSV file
    similarity_file_output_path = output_folder + 'page_similarity_results.csv'
    similarity_df_out.to_csv(similarity_file_output_path, index=False)

    output_paths.append(similarity_file_output_path)

    if not similarity_df_out.empty:
        unique_files = similarity_df_out['Page2_File'].unique()
        for redact_file in unique_files:
            output_file_name = output_folder + redact_file + "_whole_page.csv"
            whole_pages_to_redact_df = similarity_df_out.loc[similarity_df_out['Page2_File']==redact_file,:][['Page2_Page']]
            whole_pages_to_redact_df.to_csv(output_file_name, header=None, index=None)

            output_paths.append(output_file_name)            


    return similarity_df_out, output_paths

# Perturb text
# Apply the perturbation function with a 10% error probability
def perturb_text_with_errors(series):

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

# Run through command line
# def main():
#     parser = argparse.ArgumentParser(description='Combine text from multiple CSV files by page')
#     parser.add_argument('input_pattern', help='Input file pattern (e.g., "input/*.csv")')
#     parser.add_argument('--output', '-o', default='combined_text.csv', 
#                        help='Output CSV file path (default: combined_text.csv)')

#     args = parser.parse_args()
    
#     # Get list of input files
#     input_files = glob.glob(args.input_pattern)
    
#     if not input_files:
#         print(f"No files found matching pattern: {args.input_pattern}")
#         return
    
#     print(f"Processing {len(input_files)} files...")
    
#     try:
#         # Combine the text from all files
#         combined_df = combine_ocr_output_text(input_files)
        
#         # Save to CSV
#         combined_df.to_csv(args.output, index=False)
#         print(f"Successfully created combined output: {args.output}")
#         print(f"Total pages processed: {len(combined_df)}")
        
#     except Exception as e:
#         print(f"Error processing files: {str(e)}")

# if __name__ == "__main__":
#     main()
