import typer
from dissent.config import MODELS_DIR
from gensim.models import Word2Vec
import os

import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)
app = typer.Typer()

@app.command()
def main():
    model_path = MODELS_DIR / "word2vec.model"
    model = Word2Vec.load(str(model_path)) #type casting 1 + int("1")

    words_to_test = [
        "undemocratic",
        "concur",
        "murder",
        "constitution",
        "precedent",
        "abortion",
        "gender",
        "sex",
        "education",
        "religion",
        "homosexual",
        "woman",
        "female",
        "male",
        "freedom",
        "poverty",
        "wealth",
        "voting",
        "trump",
        "literalism",
        "riot",
        "insurrection",
        "equality",
        "strongly",
    ]
    for w in words_to_test:
        sims = model.wv.most_similar(w)
        print(f"\n\nWords most similar to {w}:\n", str(sims))


if __name__ == "__main__":
    main()
