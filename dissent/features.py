# Standard library
import logging
import os
from multiprocessing import Pool
from pathlib import Path

# Third-party libraries
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

# NLTK
import nltk
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize

# Local application imports
from dissent.config import PROCESSED_DATA_DIR, MODELS_DIR

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)
OUTPUT_DIR = MODELS_DIR / "iteration_0005"
_ = os.makedirs(OUTPUT_DIR, exist_ok=True)
WORKERS = 64
app = typer.Typer()


def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    english_words = set(words.words())
    tokens = word_tokenize(text.lower())
    tokens = [
        word
        for word in tokens
        if word.isalpha() and word not in stop_words and word in english_words
    ]
    return tokens

@app.command()
def main():
    model = Word2Vec(
        vector_size=100,
        window=5,
        min_count=10,
        workers=WORKERS,
        sg=1,
    )
    for i, file in tqdm(enumerate(PROCESSED_DATA_DIR.rglob("part*"))):
        docs = []
        df = pd.read_parquet(file)
        print(f"Length before filtering: {len(df)}")
        df = df[df["court_type"] == "S"]
        print(f"Length after filtering: {len(df)}")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
            opinions = row.get("opinions")
            if opinions is not None:
                for o in opinions:
                    opinion_text = o.get("opinion_text")
                    if opinion_text is not None:
                        docs.append(opinion_text)
                    else:
                        continue
            else:
                continue

        with Pool(WORKERS) as p:
            preprocessed_sentences = list(tqdm(p.imap(preprocess_text, docs), total=len(docs)))

        if i == 0:
            model.build_vocab(preprocessed_sentences)
        else:
            model.build_vocab(preprocessed_sentences, update=True)

        if not model.wv.key_to_index:
            raise ValueError("Vocabulary is empty after build_vocab")

        model.train(
            preprocessed_sentences,
            total_examples=len(preprocessed_sentences),
            epochs=10,
        )

        model.save(
            str(OUTPUT_DIR / f"word2vec_checkpoint_{i:05d}.model")
        )
        logger.success("Features generation complete.")
        try:
            similar_words_dissent = model.wv.most_similar("dissent", topn=5)
            print("Most similar words to 'dissent':", similar_words_dissent)
        except KeyError as e:
            print(f"KeyError: {e}")


if __name__ == "__main__":
    main()
