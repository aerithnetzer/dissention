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
    Memory-efficient merge of dockets, opinions, and opinion clusters.
    """

    # ---- 1. Load SMALLER tables first (column-pruned) ----
    dockets = pd.read_csv(
        RAW_DATA_DIR / "dockets-2024-12-31.csv.bz2",
        usecols=["id", "date_created", "court_id"],
        quotechar="`",
        compression="bz2",
    )
    print("Dockets Loaded")
    logger.debug(
        f"Docket columns: {dockets.columns}",
    )
    opinion_clusters = pd.read_csv(
        RAW_DATA_DIR / "opinion-clusters-2024-12-31.csv.bz2",
        # usecols=["id", "date_filed", "docket_id"],
        quotechar="`",
        compression="bz2",
    )
    print("Clusters Loaded")
    logger.debug(
        f"Opinions cluster columns: {opinion_clusters.columns}",
    )
    # ---- 2. Merge small tables first ----
    base = dockets.merge(opinion_clusters, left_on="id", right_on="docket_id", how="inner")
    base["id_x"] = pd.to_numeric(base["id_x"], errors="raise").astype("int64")
    base = base.rename(columns={"id_x": "cluster_id"})

    base["cluster_id"] = base["cluster_id"].astype("int64")

    base = base.drop(columns=["id_y"], errors="ignore")

    logger.debug(f"Base columns: {base.columns}")
    # Optional: filter courts EARLY
    # base = base[base["court_id"].isin(RELEVANT_COURTS)]

    # ---- 4. Prepare Parquet writer ----

    # ---- 5. Stream opinions in chunks ----
    i = 0
    for chunk in tqdm(
        pd.read_csv(
            RAW_DATA_DIR / "opinions-2024-12-31.csv.bz2",
            usecols=[
                "id",
                "cluster_id",
                "plain_text",
                "author_id",
            ],
            quotechar="`",
            compression="bz2",
            chunksize=1_000_000,  # tune based on RAM
        )
    ):
        # Downcast aggressively
        chunk["id"] = chunk["id"].astype("int64")
        chunk["cluster_id"] = pd.to_numeric(chunk["cluster_id"], errors="raise").astype("int64")

        logger.debug(f"Opinions columns: {chunk.columns}")
        # Filter BEFORE merge
        merged = base.merge(
            chunk,
            on="cluster_id",
            how="inner",
        )
        print(merged.columns)
        merged = merged[["cluster_id", "court_id", "plain_text", "date_filed"]]
        print(merged.head(n=5))
        merged.to_parquet(PROCESSED_DATA_DIR / f"shard_{i:05d}.parquet")
        if chunk.empty:
            logger.info(f"{len(chunk)} valid ids found. Skipping writing.")
            continue

        logger.info(f"{len(merged)} valid ids found.")

        i += 1


if __name__ == "__main__":
    main()
