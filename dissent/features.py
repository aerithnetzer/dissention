from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer
from sentence_transformers import SentenceTransformer
from dissent.config import PROCESSED_DATA_DIR, MODELS_DIR
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
import os

import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)
OUTPUT_DIR = MODELS_DIR / "iteration_0002"
_ = os.makedirs(OUTPUT_DIR, exist_ok=True)
WORKERS = 32
app = typer.Typer()
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("words")


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
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    for i, file in tqdm(enumerate(PROCESSED_DATA_DIR.rglob("part*"))):
        docs = []
        df = pd.read_parquet(file)
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

        model = Word2Vec(
            vector_size=100,
            window=5,
            min_count=1,
            workers=WORKERS,
            sg=1,
        )

        model.build_vocab(preprocessed_sentences)

        if not model.wv.key_to_index:
            raise ValueError("Vocabulary is empty after build_vocab")

        model.train(
            preprocessed_sentences,
            total_examples=len(preprocessed_sentences),
            epochs=5,
        )

        model.save(
            str(OUTPUT_DIR / f"word2vec_checkpoint_{i:05d}.model")
        )  # Most similar words to 'cat'
        logger.success("Features generation complete.")
        try:
            similar_words_cat = model.wv.most_similar("opinion", topn=5)
            print("Most similar words to 'opinion':", similar_words_cat)
        except KeyError as e:
            print(f"KeyError: {e}")


if __name__ == "__main__":
    main()
