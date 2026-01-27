from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

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

    # ------------------------------------------------------------------
    # 1. Load SMALL tables (column-pruned)
    # ------------------------------------------------------------------

    dockets = pd.read_csv(
        RAW_DATA_DIR / "dockets-2024-12-31.csv.bz2",
        usecols=["id", "court_id"],
        quotechar="`",
        compression="bz2",
    ).rename(columns={"id": "docket_id"})

    dockets["court_id"] = dockets["court_id"].astype("string")

    logger.info(f"Loaded {len(dockets):,} dockets")

    opinion_clusters = pd.read_csv(
        RAW_DATA_DIR / "opinion-clusters-2024-12-31.csv.bz2",
        usecols=["id", "date_filed", "docket_id"],
        quotechar="`",
        compression="bz2",
    ).rename(columns={"id": "cluster_id"})

    opinion_clusters["cluster_id"] = opinion_clusters["cluster_id"].astype("int64")
    opinion_clusters["date_filed"] = pd.to_datetime(
        opinion_clusters["date_filed"], errors="coerce"
    )

    logger.info(f"Loaded {len(opinion_clusters):,} opinion clusters")

    # ------------------------------------------------------------------
    # 2. Merge clusters ↔ dockets
    # ------------------------------------------------------------------

    base = opinion_clusters.merge(
        dockets,
        on="docket_id",
        how="left",
    )

    base = base[["cluster_id", "court_id", "date_filed"]]

    logger.info(
        f"Base table ready: {len(base):,} clusters "
        f"({base['court_id'].isna().mean():.1%} missing court_id)"
    )

    # ------------------------------------------------------------------
    # 3. Stream opinions and LEFT JOIN metadata
    # ------------------------------------------------------------------

    shard_idx = 0

    for chunk in tqdm(
        pd.read_csv(
            RAW_DATA_DIR / "opinions-2024-12-31.csv.bz2",
            usecols=["id", "cluster_id", "plain_text"],
            quotechar="`",
            compression="bz2",
            chunksize=500_000,
        ),
        desc="Processing opinion chunks",
    ):
        if chunk.empty:
            continue

        chunk["cluster_id"] = pd.to_numeric(chunk["cluster_id"], errors="coerce").astype("Int64")

        merged = chunk.merge(
            base,
            on="cluster_id",
            how="left",
        )

        merged = merged[["id", "cluster_id", "court_id", "date_filed", "plain_text"]]

        merged["court_id"] = merged["court_id"].astype("string")

        out_path = PROCESSED_DATA_DIR / f"shard_{shard_idx:05d}.parquet"
        merged.to_parquet(out_path, index=False)

        logger.info(
            f"Shard {shard_idx:05d}: {len(merged):,} opinions "
            f"({merged['court_id'].isna().mean():.1%} missing court_id)"
        )

        shard_idx += 1


if __name__ == "__main__":
    load_harvard_dataset()
