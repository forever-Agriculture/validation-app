"""
Citation Validation Script - Project "Unicorn Nest Candidate Test"

Overview:
This script is designed to validate AI-generated citations for data entries
against a provided source text. The primary goal is to determine the likely
veracity of a citation by measuring its semantic similarity to relevant parts
of the source material. This helps prioritize manual data review efforts by
flagging citations that are likely true, likely untrue, or ambiguous.

Configuration:
Key parameters for this script are managed via a `config.json` file expected
in the same directory as the script (`app/config.json`). This includes
model name, similarity thresholds, and input/output file names.
Defaults are used if the config file is missing or a parameter is not specified.

Methodology:
1.  Data Loading: Loads records from a JSON file (`data.json`). Each record
    contains data fields, some of which have 'value' and 'citation' pairs,
    along with a 'source_text' field.
2.  Text Preprocessing (`chunk_source_text`):
    a.  The `source_text` often concatenates content from multiple "files",
        delimited by "Content from ". The text is first split by this delimiter.
    b.  Each resulting segment is treated as Markdown content. It's converted
        to HTML using `markdown-it-py` and then to plain text using `BeautifulSoup`.
        This handles Markdown structures and extracts readable text.
    c.  The plain text is then tokenized into sentences using NLTK's `sent_tokenize`.
    d.  Sentences are normalized (lowercase, whitespace stripping) and filtered
        to remove very short or empty strings, forming a list of "chunks".
    e.  This "Ultra-Simple Structure-Aware Chunker (V9)" approach was adopted
        after experimentation, as more complex header stripping provided
        diminishing returns for the chosen embedding model.
3.  Semantic Similarity Model:
    a.  Utilizes a pre-trained sentence transformer model (`all-MiniLM-L6-v2`
        from the `sentence-transformers` library). This model is chosen for its
        balance of performance and efficiency in generating dense vector
        embeddings for text.
4.  Validation Logic (`validate_citation_item`):
    a.  Both the citation text and each source text chunk are normalized.
    b.  If a citation is too short or no source text chunks are available,
        it's marked appropriately.
    c.  Embeddings are generated for the normalized citation and all source
        text chunks.
    d.  Cosine similarity is calculated between the citation embedding and
        each chunk embedding. The maximum similarity score is taken.
    e.  This maximum score is converted to a confidence percentage (0-100%).
    f.  Based on predefined thresholds, a granular validation status
        (e.g., "very likely true", "likely untrue", "ambiguous / review recommended")
        is assigned.
5.  Output: The original records are updated with 'validation_status' and
    'confidence_score_percent' fields for each processed citation. The
    results are saved to `output_validated.json`.

Key Libraries:
- `nltk`: For sentence tokenization.
- `sentence-transformers`: For loading the language model and generating embeddings.
- `numpy`: For numerical operations (though minimally used directly in the final version).
- `markdown-it-py`: For robust Markdown to HTML conversion.
- `beautifulsoup4`: For parsing HTML and extracting text content.

Usage:
Ensure all dependencies in `requirements.txt` are installed. Run the script
from the command line: `python app/validator.py`
"""

import json
import nltk
import re
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer, util
import numpy as np
from markdown_it import MarkdownIt
from bs4 import BeautifulSoup

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE_PATH = SCRIPT_DIR / "config.json"

# Default configuration values
DEFAULT_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.6,
    "min_citation_length_words": 2,
    "min_chunk_length_chars": 10,
    "data_file_name": "data.json",
    "output_file_name": "output_validated.json"
}

def load_config(config_path: Path) -> dict:
    """
    Loads configuration from a JSON file.
    If the file doesn't exist or is invalid, returns default settings.
    """
    if config_path.exists() and config_path.is_file():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # Merge user_config with defaults, user_config takes precedence
                config = {**DEFAULT_CONFIG, **user_config} 
                print(f"Loaded configuration from {config_path}")
                return config
        except json.JSONDecodeError:
            print(f"Warning: Error decoding {config_path}. Using default settings.")
        except Exception as e:
            print(f"Warning: Could not load {config_path} due to {e}. Using default settings.")
    else:
        print(f"Warning: Config file {config_path} not found. Using default settings.")
    return DEFAULT_CONFIG.copy() # Return a copy of defaults

# Load configuration
config = load_config(CONFIG_FILE_PATH)

# File Paths (derived from config or defaults)
DATA_FILE_PATH = SCRIPT_DIR / config.get("data_file_name", DEFAULT_CONFIG["data_file_name"])
OUTPUT_FILE_PATH = SCRIPT_DIR / config.get("output_file_name", DEFAULT_CONFIG["output_file_name"])

# Model Configuration
MODEL_NAME = config.get("model_name", DEFAULT_CONFIG["model_name"])

# Validation Logic Parameters
SIMILARITY_THRESHOLD = config.get("similarity_threshold", DEFAULT_CONFIG["similarity_threshold"])
MIN_CITATION_LENGTH_WORDS = config.get("min_citation_length_words", DEFAULT_CONFIG["min_citation_length_words"])
MIN_CHUNK_LENGTH_CHARS = config.get("min_chunk_length_chars", DEFAULT_CONFIG["min_chunk_length_chars"])

# Ensure types are correct after loading from JSON (JSON loads numbers as float/int)
SIMILARITY_THRESHOLD = float(SIMILARITY_THRESHOLD)
MIN_CITATION_LENGTH_WORDS = int(MIN_CITATION_LENGTH_WORDS)
MIN_CHUNK_LENGTH_CHARS = int(MIN_CHUNK_LENGTH_CHARS)

def load_data(file_path: str) -> list:
    """
    Loads the JSON data from the specified file.

    Args:
        file_path (str): The path to the JSON data file.

    Returns:
        list: A list of records from the JSON file.
              Returns an empty list if the file is not found or
              if there's an error during JSON decoding.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return []

def normalize_text(text: str) -> str:
    """
    Performs basic text normalization.
    Converts to lowercase, replaces newlines with spaces, and strips leading/trailing whitespace.
    Also handles potential output from BeautifulSoup which might be None.

    Args:
        text (str): The input text.

    Returns:
        str: The normalized text. Returns an empty string if input is None.
    """
    if text is None:
        return ""
    # Replace newlines with spaces, then lowercase and strip.
    return text.replace('\n', ' ').lower().strip()

def chunk_source_text(text: str, min_chunk_length: int = MIN_CHUNK_LENGTH_CHARS) -> list[str]:
    """
    Ultra-Simple Structure-Aware Chunker (V9).
    Splits text by "Content from ", then converts each segment from Markdown
    to plain text, sentence tokenizes, and normalizes.
    No explicit header stripping is performed beyond Markdown conversion.
    """
    if not text:
        return []

    all_valid_chunks = []
    md_parser = MarkdownIt() # Initialize once

    # Split the entire source text by the "Content from " delimiter
    # This delimiter often precedes a new file's content.
    raw_segments = text.split("Content from ")

    for raw_segment_text in raw_segments:
        current_content = raw_segment_text.strip()
        if not current_content:
            continue
        
        # Directly convert current segment (which might contain headers) to plain text.
        # The sentence embedding model is relied upon to discern meaning even with
        # potential header text mixed in.
        html_content = md_parser.render(current_content)
        soup = BeautifulSoup(html_content, "html.parser")
        plain_text_from_markdown = soup.get_text(separator=" ") # Use space as separator

        # Normalize whitespace (e.g., multiple spaces to one)
        processed_text = re.sub(r'\s+', ' ', plain_text_from_markdown).strip()

        if not processed_text:
            continue

        # Tokenize the cleaned text into sentences
        sentences = nltk.sent_tokenize(processed_text)

        for sentence in sentences:
            normalized_sentence = normalize_text(sentence) # Lowercase, strip
            # Filter out very short sentences/chunks.
            if len(normalized_sentence.split()) >= MIN_CITATION_LENGTH_WORDS and \
               len(normalized_sentence) >= min_chunk_length:
                all_valid_chunks.append(normalized_sentence)
    
    return all_valid_chunks

def validate_citation_item(
    citation_text: str, 
    source_text_chunks: list[str], 
    model: SentenceTransformer
) -> tuple[str, float]:
    """
    Validates a single citation against chunks of source text.

    Args:
        citation_text (str): The citation text.
        source_text_chunks (list[str]): A list of pre-processed source text chunks.
        model (SentenceTransformer): The loaded sentence transformer model.

    Returns:
        tuple[str, float]: A tuple containing the granular validation status 
                           and a confidence score (0.0 to 1.0).
    """
    normalized_citation = normalize_text(citation_text)

    if not normalized_citation or len(normalized_citation.split()) < MIN_CITATION_LENGTH_WORDS:
        return "untrue (too short)", 0.0 

    if not source_text_chunks:
        return "untrue (no source text)", 0.0

    try:
        # THIS IS THE CORE OF THE SCRIPT!
        citation_embedding = model.encode(normalized_citation, convert_to_tensor=True) # creating vectors with the "meaning" of the citation
        chunk_embeddings = model.encode(source_text_chunks, convert_to_tensor=True) # creating vectors with the "meaning" of the source text chunks

        cos_scores = util.cos_sim(citation_embedding, chunk_embeddings)[0]
        
        cos_scores_np = cos_scores.cpu().numpy()
        if cos_scores_np.size == 0:
             max_score = 0.0
        else:
            max_score = float(np.max(cos_scores_np))

    except Exception as e:
        print(f"Error during embedding/similarity calculation for citation '{normalized_citation[:50]}...': {e}")
        return "error processing", 0.0 

    confidence = max(0.0, min(1.0, max_score)) 

    if confidence >= 0.80:
        status = "very likely true"
    elif confidence >= SIMILARITY_THRESHOLD: 
        status = "likely true"
    elif confidence >= 0.40: 
        status = "ambiguous / review recommended"
    elif confidence >= 0.20: 
        status = "likely untrue"
    else: 
        status = "very likely untrue"
        
    return status, confidence

def save_validated_data(records: list, file_path: Path):
    """
    Saves the validated records to a JSON file.

    Args:
        records (list): The list of records with validation information.
        file_path (Path): The path to save the output JSON file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"\nValidated data successfully saved to {file_path}")
    except Exception as e:
        print(f"\nError saving validated data to {file_path}: {e}")

def main():
    """
    Main function to orchestrate the citation validation process.
    Loads data, preprocesses text, validates citations, and saves results.
    Uses parameters loaded from config.json or defaults.
    """
    print("Starting citation validation process...")
    print(f"Using Model: {MODEL_NAME}")
    print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"Min Citation Words: {MIN_CITATION_LENGTH_WORDS}")
    print(f"Min Chunk Chars: {MIN_CHUNK_LENGTH_CHARS}")
    print(f"Data File: {DATA_FILE_PATH}")
    print(f"Output File: {OUTPUT_FILE_PATH}")

    # Initialize the Sentence Transformer model
    print(f"\nLoading sentence transformer model: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading sentence transformer model: {e}")
        print("Please ensure 'sentence-transformers' is installed and the model name is correct.")
        return

    # Load the data
    records = load_data(DATA_FILE_PATH)

    if not records:
        print("No data loaded. Exiting.")
        return

    print(f"Successfully loaded {len(records)} records.")

    # --- Core Validation Loop ---
    print("\nStarting core validation loop...")
    validated_records_count = 0
    citations_processed_count = 0

    for i, record in enumerate(records):
        # Simplified the print statement for less verbose output during processing.
        # print(f"Processing record {i+1}/{len(records)}: {record.get('name', {}).get('value', 'N/A')}...")
        source_text = record.get("source_text", "")
        source_text_chunks = chunk_source_text(source_text)

        # Fields that contain citations
        # Note: 'name' is an object, others are lists of objects
        citation_fields = ["name", "investor_type", "fund_main_location", "investment_stages_and_rounds"]

        for field_name in citation_fields:
            field_data = record.get(field_name)
            if not field_data:
                continue

            items_to_validate = []
            if isinstance(field_data, dict): # Handles 'name' field
                items_to_validate.append(field_data)
            elif isinstance(field_data, list): # Handles other fields like 'investor_type'
                items_to_validate.extend(field_data)
            
            for item in items_to_validate:
                if isinstance(item, dict) and "citation" in item and "value" in item:
                    citation_text = item.get("citation")
                    if citation_text:
                        # Removed the specific debug block for "Conor VC - Espoo Citation"
                        status, confidence = validate_citation_item(citation_text, source_text_chunks, model)
                        item["validation_status"] = status
                        item["confidence_score_percent"] = round(confidence * 100, 2) # Store as percentage
                        citations_processed_count += 1
                    else: # Handle cases where citation key exists but value is None or empty
                        item["validation_status"] = "no citation provided"
                        item["confidence_score_percent"] = 0.0 # Explicitly 0 for no citation
        
        validated_records_count +=1
        if (i + 1) % 10 == 0 or (i + 1) == len(records): # Print progress every 10 records and for the last one
            print(f"Processed {i+1}/{len(records)} records...")

    
    print(f"\nCore validation loop complete.")
    print(f"Processed {validated_records_count} records.")
    print(f"Processed {citations_processed_count} individual citations.")

    # Save the updated records
    if records:
        save_validated_data(records, OUTPUT_FILE_PATH)

if __name__ == "__main__":
    # Download necessary NLTK resources if not already present.
    nltk_resources = ["punkt", "punkt_tab"]
    all_downloads_successful_or_already_present = True # Renamed for clarity

    for resource_name in nltk_resources:
        resource_path_token = f'tokenizers/{resource_name}'
        if resource_name == "punkt_tab":
            # For punkt_tab, NLTK usually looks for a language-specific pickle
            resource_path_token = f'tokenizers/{resource_name}/english.pickle'
        
        try:
            nltk.data.find(resource_path_token)
            print(f"NLTK resource '{resource_name}' (checking for {resource_path_token}) already available.")
        except LookupError:
            print(f"NLTK resource '{resource_name}' not found. Downloading...")
            try:
                nltk.download(resource_name, quiet=True)
                print(f"'{resource_name}' resource downloaded successfully.")
            except Exception as e:
                all_downloads_successful_or_already_present = False # Mark failure
                print(f"Failed to download '{resource_name}'. Error: {e}")
                print(f"Please ensure you have an internet connection and try again.")
                print(f"The script might not function correctly without '{resource_name}'.")
        except Exception as e:
            all_downloads_successful_or_already_present = False # Mark failure
            print(f"An unexpected error occurred during NLTK setup for '{resource_name}': {e}")

    if not all_downloads_successful_or_already_present:
        print("\nWarning: Some NLTK resources could not be verified or downloaded successfully.")
        print("The script will attempt to proceed, but tokenization errors may occur.")
        # import sys
        # sys.exit("Exiting due to missing NLTK resources.")
    
    main() 