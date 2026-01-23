import typer
from dissent.config import MODELS_DIR
from gensim.models import Word2Vec
import os

import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)
OUTPUT_DIR = MODELS_DIR / "iteration_0003"
_ = os.makedirs(OUTPUT_DIR, exist_ok=True)
WORKERS = 32
app = typer.Typer()


@app.command()
def main():
    model_files = OUTPUT_DIR.rglob("*.model")
    model_path = model_files[-1]
    model = Word2Vec.load(model_path)
    sims = model.wv.most_similar("dissent")
    print(sims)
    sims = model.wv.most_similar("concur")
    print(sims)


if __name__ == "__main__":
    main()
