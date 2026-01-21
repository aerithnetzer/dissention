from pathlib import Path
from loguru import logger
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from tqdm import tqdm
from dissent.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def process(dockets: pd.DataFrame, opinion_clusters: pd.DataFrame):
    return True


def main():
    """
    Memory-efficient merge of dockets, opinion clusters, and opinions.
    """

    # ---- 1. Load SMALL tables (column-pruned) ----
    dockets = pd.read_csv(
        RAW_DATA_DIR / "dockets-2024-12-31.csv.bz2",
        usecols=["id", "date_created", "court_id"],
        quotechar="`",
        compression="bz2",
    ).rename(columns={"id": "docket_id"})

    logger.info("Dockets loaded")

    opinion_clusters = pd.read_csv(
        RAW_DATA_DIR / "opinion-clusters-2024-12-31.csv.bz2",
        usecols=["id", "date_filed", "docket_id"],
        quotechar="`",
        compression="bz2",
    ).rename(columns={"id": "cluster_id"})

    logger.info("Opinion clusters loaded")

    # ---- 2. Merge dockets ↔ clusters ----
    base = opinion_clusters.merge(
        dockets,
        on="docket_id",
        how="inner",
    )

    # Enforce dtypes
    base["cluster_id"] = base["cluster_id"].astype("int64")
    base["court_id"] = base["court_id"].astype("str")

    logger.debug(f"Base columns: {base.columns}")
    # ['cluster_id', 'date_filed', 'docket_id', 'date_created', 'court_id']

    # ---- 3. Stream opinions in chunks ----
    i = 0
    for chunk in tqdm(
        pd.read_csv(
            RAW_DATA_DIR / "opinions-2024-12-31.csv.bz2",
            usecols=["id", "cluster_id", "plain_text", "author_id"],
            quotechar="`",
            compression="bz2",
            chunksize=100_000,
        )
    ):
        if chunk.empty:
            continue

        chunk["cluster_id"] = chunk["cluster_id"].astype("int64")

        # ---- 4. Merge opinions ↔ base ----
        merged = chunk.merge(
            base[["cluster_id", "court_id", "date_filed"]],
            on="cluster_id",
            how="inner",
        )

        if merged.empty:
            logger.info("No matching opinions in this chunk")
            continue

        merged = merged[["cluster_id", "court_id", "date_filed", "plain_text"]]

        merged.to_parquet(
            PROCESSED_DATA_DIR / f"shard_{i:05d}.parquet",
            index=False,
        )

        logger.info(f"Wrote {len(merged)} opinions")
        i += 1


main()
