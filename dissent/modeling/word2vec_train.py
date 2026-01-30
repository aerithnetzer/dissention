# Standard library
import logging
import os
from multiprocessing import Pool
from pathlib import Path

# Third-party libraries
import pandas as pd
from gensim.models import Word2Vec
from loguru import logger
from tqdm import tqdm
import typer

# NLTK
import nltk
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize

# Set up project directory paths

PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = "models"

# Configure logging settings

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)

OUTPUT_DIR = MODELS_DIR / "iteration_0003"  # Directory to save trained models
_ = os.makedirs(OUTPUT_DIR, exist_ok=True)  # WHAT DOES THIS DO?
WORKERS = 32  # Number of CPU cores to use for parallel processing
app = typer.Typer() # Sets up CLI

# Download required NLTK data files

nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("words")

# Clean and tokenize text by removing stopwords and non-English words

def preprocess_text(text):
    stop_words = set(stopwords.words("english"))  # Load English stopwords
    english_words = set(words.words())  # Load valid English words dictionary
    tokens = word_tokenize(text.lower())  # Convert to lowercase and split into tokens
    tokens = [  # Filter tokens to keep only valid alphabetic English words
        word
        for word in tokens
        if word.isalpha() and word not in stop_words and word in english_words
    ]
    return tokens  # Return cleaned token list


@app.command()
def main():
    # Process each parquet file in the processed data directory
    for i, file in tqdm(enumerate(PROCESSED_DATA_DIR.rglob("part*"))):
        docs = []  # Store opinion texts from current file
        df = pd.read_parquet(file)  # Load parquet file
        df = df[:500]  # Limit to first 500 rows for testing
        
        # Extract opinion texts from dataframe
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
            opinions = row.get("opinions")  # Get opinions field from row
            if opinions is not None:  # Check if opinions exist
                for o in opinions:  # Loop through each opinion
                    opinion_text = o.get("opinion_text")  # Extract opinion text
                    if opinion_text is not None:  # Check if text exists
                        docs.append(opinion_text)  # Add to documents list
                    else:
                        continue  # Skip if no text
            else:
                continue  # Skip if no opinions

        # Preprocess all documents in parallel using multiprocessing
        # Uses 32 worker processes to speed up text preprocessing
        with Pool(WORKERS) as p:
            preprocessed_sentences = list(tqdm(p.imap(preprocess_text, docs), total=len(docs)))

        # Initialize Word2Vec model with hyperparameters
        # vector_size=100: each word represented as 100-dimensional vector
        # window=5: considers 5 words before and after target word for context
        # min_count=100: I FORGET WHICH DIRECTION THIS GOES IN
        # workers=32: uses 32 CPU cores for training
        # sg=1: uses skip-gram algorithm (predicts context from word)
        model = Word2Vec(
            vector_size=100,
            window=5,
            min_count=10,
            workers=WORKERS,
            sg=1,
        )

        # Build the model's vocabulary from the documents
        # Creates a dictionary mapping each unique word to an index number
        # Also counts word frequencies to filter out rare words (based on min_count parameter)
        model.build_vocab(preprocessed_sentences)

        # Check if the vocabulary is empty (no word-to-index mappings)
        if not model.wv.key_to_index:
            raise ValueError("Vocabulary is empty after build_vocab")

        # Train the Word2Vec model on the documents for 5 epochs
        # An epoch is one complete pass through the entire dataset
        # Multiple epochs help the model learn better word representations
        model.train(
            preprocessed_sentences,
            total_examples=len(preprocessed_sentences),
            epochs=5,
        )

        # Save the trained model with a checkpoint number
        # Each file gets its own model saved with a unique index (i)
        model.save(
            str(OUTPUT_DIR + f"word2vec_checkpoint_{i:05d}.model")
        )
        logger.success("Features generation complete.")
        
        # Test the model by finding similar words to 'opinion'
        try:
            similar_words_cat = model.wv.most_similar("opinion", topn=5)  # Find 5 most similar words
            print("Most similar words to 'opinion':", similar_words_cat)
        except KeyError as e:  # Handle case where 'opinion' is not in vocabulary
            print(f"KeyError: {e}")


if __name__ == "__main__":
    main()  # Run the CLI application