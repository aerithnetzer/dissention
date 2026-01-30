# Standard library
from pathlib import Path

# Third-party libraries
from loguru import logger
from tqdm import tqdm
import typer
from gensim.models import Word2Vec

# Local application imports
from dissent.config import FIGURES_DIR, PROCESSED_DATA_DIR, DATA_DIR, MODELS_DIR

app = typer.Typer()

model_path = MODELS_DIR / "word2vec.model"
model = Word2Vec.load(str(model_path)) #type casting 1 + int("1")

@app.command()
def main():

    seed_word_files = [DATA_DIR / "evidence_seed_words.txt", DATA_DIR / "intuition_seed_words.txt"]
    print(seed_word_files)
    for f in seed_word_files:
        output_file = f.name
        output_file = output_file.replace("_seed_words.txt","")
        expanded_key_words = []
        with open(f, "r") as data:
            seed_words = data.readlines()
            with open(DATA_DIR / output_file, "w") as output_data: #data is an interface, textio wrapper
                for w in seed_words:
                    output_data.write(w)
            print(type(data))
        print(seed_words)
        for word in seed_words:
            cleaned_word = word.replace("\n","")
            similar = model.wv.most_similar(cleaned_word)
            for s in similar:
                if s[1] > 0.75:
                    expanded_key_words.append(s[0])
        with open(DATA_DIR / output_file, "a") as data: #data is an interface, textio wrapper
            for w in expanded_key_words:
                data.write(w + "\n")
            print(type(data))
   
   # with open(for f in seed_word_files)
   # words_to_test = 
    
   # for w in words_to_test:
    #    sims = model.wv.most_similar(w)

if __name__ == "__main__":
    app()
