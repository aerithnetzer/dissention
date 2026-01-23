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
    model_files = sorted(list(OUTPUT_DIR.glob("*.model")))
    model_path = str(model_files[-1])
    model = Word2Vec.load(model_path)
    sims = model.wv.most_similar("dissent")
    print(f"{sims}\n\n")
    sims = model.wv.most_similar("concur")
    print(f"{sims}\n\n")
    sims = model.wv.most_similar("murder")
    print(f"{sims}\n\n")
    sims = model.wv.most_similar("constitution")
    print(f"{sims}\n\n")
    sims = model.wv.most_similar("constitution")
    print(f"{sims}\n\n")
    sims = model.wv.most_similar("precedent")
    print(f"{sims}\n\n")
    sims = model.wv.most_similar("precedent")
    print(f"{sims}\n\n")


if __name__ == "__main__":
    main()
