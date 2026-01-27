import typer
from dissent.config import MODELS_DIR
from gensim.models import Word2Vec
import os

import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)
OUTPUT_DIR = MODELS_DIR / "iteration_0005"
_ = os.makedirs(OUTPUT_DIR, exist_ok=True)
WORKERS = 32
app = typer.Typer()


@app.command()
def main():
    model_files = sorted(list(OUTPUT_DIR.glob("*.model")))
    model_path = str(model_files[-1])
    model = Word2Vec.load(model_path)

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
