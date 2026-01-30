# Standard library
import logging
import os

# Third-party libraries
import typer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local application imports
from dissent.config import MODELS_DIR, DATA_DIR, PROCESSED_DATA_DIR

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)
app = typer.Typer()

@app.command()
def main():
    model_path = MODELS_DIR / "word2vec.model"
    model = Word2Vec.load(str(model_path))

    # Load all words from txt files using rglob
    evidence_words = []
    intuition_words = []
    
    txt_files = DATA_DIR.rglob("*.txt")
    
    for file_path in txt_files:  # Changed 'f' to 'file_path'
        with open(file_path, "r") as f:  # Now 'file_path' is correct
            words_to_test = f.readlines()
            for word in words_to_test:
                words_cleaned = word.replace("\n","")
                if words_cleaned:
                    # Categorize based on filename
                    if "evidence" in file_path.name:  # Changed to 'file_path.name'
                        evidence_words.append(words_cleaned)
                    elif "intuition" in file_path.name:  # Changed to 'file_path.name'
                        intuition_words.append(words_cleaned)

    # Filter words that exist in the model's vocabulary
    valid_evidence = [w for w in evidence_words if w in model.wv]
    valid_intuition = [w for w in intuition_words if w in model.wv]
    
    if not valid_evidence or not valid_intuition:
        print("Error: Need valid evidence and intuition words")
        return
    
    # Get word vectors
    evidence_vectors = [model.wv[w] for w in valid_evidence]
    intuition_vectors = [model.wv[w] for w in valid_intuition]
    
    # Compute cosine similarity matrix (evidence vs intuition)
    similarity_matrix = cosine_similarity(evidence_vectors, intuition_vectors)
    
    # Print matrix
    print(similarity_matrix)

if __name__ == "__main__":
    main()