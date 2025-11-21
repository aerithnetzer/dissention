from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from dissent.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def merge_dataframes(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the dataframes. Assumes that `opinions` is left table and `opinions_clusters` is right label.
    """
    df = pd.merge(left, right, how="inner", left_on="cluster_id", right_on="id")
    return df


def load_dataframe(path: Path, nrows: int | None) -> pd.DataFrame:
    if isinstance(nrows, int):
        df = pd.read_csv(path, compression="bz2", quotechar="`", nrows=nrows)
    else:
        df = pd.read_csv(path, compression="bz2", quotechar="`")
    return df


@app.command()
def main():
    """
    This function takes as input the opinions csv and opinion clusters csv.
    Merges on cluster IDs, and then filters on relevant courts.
    """
    logger.info("Processing dataset")
    logger.info("Now loading dataframes")

    opinions = load_dataframe(RAW_DATA_DIR / "opinions-2024-12-31.csv.bz2", nrows=None)
    opinion_clusters = load_dataframe(
        path=RAW_DATA_DIR / "opinion-clusters-2024-12-31.csv.bz2", nrows=None
    )
    dockets: pd.DataFrame = load_dataframe(
        path=RAW_DATA_DIR / "dockets-2024-12-31.csv.bz2", nrows=None
    )

    df = pd.merge(
        left=opinions, right=opinion_clusters, how="inner", left_on="cluster_id", right_on="id"
    )
    df = pd.merge(df, dockets, how="inner", left_on="docket_id", right_on="id")
    df.to_csv(INTERIM_DATA_DIR / "dataset.csv.bz2", quotechar="`")


if __name__ == "__main__":
    app()
