# Standard library
from pathlib import Path

# Third-party libraries
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

# Local application imports
from dissent.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def load_harvard_dataset():
    from datasets import load_dataset

    ds = load_dataset(
        "harvard-lil/cold-cases",
    )

    ds.save_to_disk(PROCESSED_DATA_DIR)

    return None


def load_parquet_files():
    "../data/processed/shard_00000.parquet"


def main():
    """
    Memory-efficient merge of opinions with opinion clusters and dockets.

    Output guarantees:
    - One row per opinion
    - court_id is NULL if unavailable
    - date_filed is NULL if unavailable
    """
        # Filter by date
    df = pd.concat([pd.read_parquet(f) for f in tqdm(PROCESSED_DATA_DIR.rglob("*.parquet"))])
    df["year"] = df["date_filed"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
    df = df[df["year"] > 2018]
    df = df[df["court_type"] == "S"]
    df2 = df[
    df["court_jurisdiction"].isin(["North Carolina, NC", "Alabama, AL"])
    ].copy()
    print(df2.head())
    df2.to_parquet(PROCESSED_DATA_DIR / "dataset.parquet")

#        opinions = row.get("opinions")
#        if opinions is not None:   
#            for o in opinions:
#                opinion_text = o.get("opinion_text")

if __name__ == "__main__":
    load_harvard_dataset()
    main()
