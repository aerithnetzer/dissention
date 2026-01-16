from pathlib import Path
from loguru import logger
import pandas as pd
import typer
from tqdm import tqdm
from dissent.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def process(dockets: pd.DataFrame, opinion_clusters: pd.DataFrame):
    return True


@app.command()
def main():
    """
    This function takes as input the opinions csv and opinion clusters csv.
    Merges on cluster IDs, and then filters on relevant courts.
    """

    dockets: pd.DataFrame = pd.read_csv(RAW_DATA_DIR / "dockets-2024-12-31.csv.bz2", quotechar="`")
    opinions_clusters: pd.DataFrame = pd.read_csv(
        RAW_DATA_DIR / "opinion-clusters-2024-12-31.csv.bz2",
        quotechar="`",
        compression="bz2",
    )
    opinions: pd.DataFrame = pd.read_csv(
        RAW_DATA_DIR / "opinions-2024-12-31.csv.bz2",
        quotechar="`",
        compression="bz2",
    )

    df = pd.merge(dockets, opinions, how="inner", on="id")
    df = pd.merge(df, opinions_clusters, "inner", on="id")

    print("Opinion columns: \n", opinions.columns)
    print("Docket columns: \n", dockets.columns)
    print("Clusters columns: \n", opinions_clusters.columns)
    print("Moiged columns: \n", df.columns)
    print("Moiged \n", df.head())

    df.to_parquet(PROCESSED_DATA_DIR / "dataset.parquet", compression="gzip")


if __name__ == "__main__":
    app()
