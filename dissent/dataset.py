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
        nrows=1_000,
        usecols=["id", "date_created", "court_id"],
        quotechar="`",
        compression="bz2",
    )
    logger.debug(
        f"Docket columns: {dockets.columns}",
    )
    opinion_clusters = pd.read_csv(
        RAW_DATA_DIR / "opinion-clusters-2024-12-31.csv.bz2",
        # usecols=["id", "date_filed", "docket_id"],
        quotechar="`",
        nrows=1_000,
        compression="bz2",
    )
    logger.debug(
        f"Opinions cluster columns: {opinion_clusters.columns}",
    )
    # ---- 2. Merge small tables first ----
    base = dockets.merge(opinion_clusters, left_on="id", right_on="docket_id", how="inner")
    logger.debug(f"Base columns: {base.columns}")
    # Optional: filter courts EARLY
    # base = base[base["court_id"].isin(RELEVANT_COURTS)]

    # ---- 3. Build ID whitelist ----
    valid_ids = set(base["id_x"].astype("int64"))

    # ---- 4. Prepare Parquet writer ----
    output_path = PROCESSED_DATA_DIR / "dataset.parquet"
    parquet_writer = None

    # ---- 5. Stream opinions in chunks ----
    for chunk in tqdm(
        pd.read_csv(
            RAW_DATA_DIR / "opinions-2024-12-31.csv.bz2",
            usecols=[
                "id",
                "plain_text",
                "author_id",
                # add ONLY needed columns
            ],
            quotechar="`",
            compression="bz2",
            chunksize=10_000,  # tune based on RAM
        )
    ):
        # Downcast aggressively
        chunk["id"] = chunk["id"].astype("int64")

        logger.debug(f"Opinions columns: {chunk.columns}")
        # Filter BEFORE merge
        chunk = chunk[chunk["id"].isin(valid_ids)]
        if chunk.empty:
            logger.info(f"{len(chunk)} valid ids found. Skipping writing.")
            continue

        logger.info(f"{len(chunk)} valid ids found.")
        merged = base.merge(chunk, on="id", how="inner")

        table = pa.Table.from_pandas(merged, preserve_index=False)

        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(
                output_path,
                table.schema,
                compression="gzip",
            )

        parquet_writer.write_table(table)

        # Explicit cleanup
        del chunk, merged, table

    if parquet_writer:
        parquet_writer.close()


if __name__ == "__main__":
    main()
