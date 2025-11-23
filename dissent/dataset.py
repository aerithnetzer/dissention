from pathlib import Path
from loguru import logger
import pandas as pd
import typer

from dissent.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main():
    """
    This function takes as input the opinions csv and opinion clusters csv.
    Merges on cluster IDs, and then filters on relevant courts.
    """

    logger.info("Processing dataset")
    logger.info("Now loading dataframes")

    opinions = pd.read_csv(
        RAW_DATA_DIR / "opinions-2024-12-31.csv.bz2",
        usecols=["id", "cluster_id", "type"],
    )

    opinion_clusters = pd.read_csv(
        RAW_DATA_DIR / "opinion-clusters-2024-12-31.csv.bz2",
        usecols=["id", "date_filed", "docket_id"],
        quotechar="`",
    )

    print(opinion_clusters.columns)
    dockets: pd.DataFrame = pd.read_csv(
        RAW_DATA_DIR / "dockets-2024-12-31.csv.bz2", nrows=100_000, quotechar="`"
    )
    print(dockets.columns)
    logger.info("Finished reading dataframes")
    # First join: opinions with opinion_clusters
    df = opinions.set_index("cluster_id").join(
        opinion_clusters.set_index("id"), how="inner", rsuffix="_cluster"
    )

    logger.info("Finished first join")
    df = df.join(dockets.set_index("id"), on="docket_id", how="inner", rsuffix="_docket")

    logger.info("Finished second join")
    df.to_csv(INTERIM_DATA_DIR / "dataset.csv.bz2", quotechar="`")

    logger.info("Finished writing dataset")


if __name__ == "__main__":
    app()
