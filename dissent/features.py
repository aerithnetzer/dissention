from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer
from dissent.config import PROCESSED_DATA_DIR, MODELS_DIR, RAW_DATA_DIR
from gensim.models import Word2Vec
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool
import os

import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)
OUTPUT_DIR = MODELS_DIR / "iteration_0007"
_ = os.makedirs(OUTPUT_DIR, exist_ok=True)
WORKERS = 64
app = typer.Typer()


def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    english_words = set(words.words())
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in stop_words and word in english_words
    ]
    return tokens


@app.command()
def main():
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    model = Word2Vec(
        vector_size=300,
        window=5,
        min_count=20,
        workers=WORKERS,
        sg=1,
    )
    for i, file in tqdm(enumerate(RAW_DATA_DIR.rglob("part*"))):
        docs = []
        df = pd.read_parquet(file)
        print(f"Length before filtering: {len(df)}")
        df = df[df["court_type"] == "S"]
        df["year"] = df["date_filed"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
        df.to_parquet(PROCESSED_DATA_DIR / f"shard_{i:05d}.parquet")
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
        )  # Most similar words to 'cat'
        logger.success("Features generation complete.")
        try:
            similar_words_cat = model.wv.most_similar("opinion", topn=5)
            print("Most similar words to 'opinion':", similar_words_cat)
        except KeyError as e:
            print(f"KeyError: {e}")


if __name__ == "__main__":
    main()
