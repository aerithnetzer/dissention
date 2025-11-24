from pathlib import Path
from loguru import logger
import pandas as pd
import typer
from tqdm import tqdm
from dissent.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def process(dockets: pd.DataFrame, opinion_clusters: pd.DataFrame):
    chunk_size = 10**5
    df = pd.DataFrame()
    for i, chunk in tqdm(
        enumerate(
            pd.read_csv(
                RAW_DATA_DIR / "opinions-2024-12-31.csv.bz2",
                chunksize=chunk_size,
                quotechar="`",
                usecols=["id", "cluster_id", "type"],
            )
        )
    ):
        print(chunk.columns)
        print(opinion_clusters.columns)
        print(dockets.columns)
        # First join: opinions with opinion_clusters

        df_chunk = chunk.set_index("cluster_id").join(
            opinion_clusters.set_index("id"), how="inner", rsuffix="_cluster"
        )

        df_chunk = df_chunk.join(
            other=dockets.set_index("id"), on="docket_id", how="inner", rsuffix="_docket"
        )
        if len(df_chunk) > 0:
            df_chunk.to_parquet(PROCESSED_DATA_DIR / f"dataset_shard_{i:010d}.parquet")
            logger.info(f"Saved shard {i}, of length: {len(df_chunk)}")
        else:
            logger.info("This chunk has no relevant data.")

    return True


@app.command()
def main():
    """
    This function takes as input the opinions csv and opinion clusters csv.
    Merges on cluster IDs, and then filters on relevant courts.
    """

    opinion_clusters = pd.read_csv(
        RAW_DATA_DIR / "opinion-clusters-2024-12-31.csv.bz2",
        usecols=["id", "date_filed", "docket_id"],
        quotechar="`",
    )

    dockets: pd.DataFrame = pd.read_csv(RAW_DATA_DIR / "dockets-2024-12-31.csv.bz2", quotechar="`")
    success = process(dockets=dockets, opinion_clusters=opinion_clusters)


if __name__ == "__main__":
    app()
