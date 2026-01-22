from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer
from sentence_transformers import SentenceTransformer
from dissent.config import PROCESSED_DATA_DIR
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import glob

app = typer.Typer()

nltk.download("punkt_tab")
nltk.download("stopwords")


def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "test.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    print(str())
    for i, file in tqdm(enumerate(PROCESSED_DATA_DIR.rglob("part*"))):
        print(file)
        docs = []
        df = pd.read_parquet(file)
        for _, row in tqdm(df.iterrows(), total=len(df)):
            opinions = row.get("opinions")
            if opinions is not None:
                for o in opinions:
                    opinion_text = o.get("opinion_text")
                    docs.append(opinion_text)
                    print(type(opinion_text))
            else:
                continue

        logger.success("Features generation complete.")
        # -----------------------------------------
        # # Preprocess sentences
        preprocessed_sentences = [preprocess_text(sentence) for sentence in tqdm(docs)]

        model = Word2Vec(
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
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

        model.save(f"word2vec_shard_{i:05d}.model")  # Most similar words to 'cat'
        try:
            similar_words_cat = model.wv.most_similar("opinion", topn=5)
            print("Most similar words to 'opinion':", similar_words_cat)
        except KeyError as e:
            print(f"KeyError: {e}")


if __name__ == "__main__":
    main()
