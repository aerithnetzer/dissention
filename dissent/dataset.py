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
    Memory-efficient merge of dockets, opinions, and opinion clusters.
    """

    # ---- 1. Load SMALLER tables first (column-pruned) ----
    dockets = pd.read_csv(
        RAW_DATA_DIR / "dockets-2024-12-31.csv.bz2",
        usecols=[
            "id",
            "court_id",
            "date_filed",
            # add ONLY what you actually need
        ],
        nrows=50,
        quotechar="`",
        compression="bz2",
    )

    opinion_clusters = pd.read_csv(
        RAW_DATA_DIR / "opinion-clusters-2024-12-31.csv.bz2",
        usecols=[
            "id",
            "precedential_status",
            "cluster_type",
        ],
        quotechar="`",
        nrows=50,
        compression="bz2",
    )

    # ---- 2. Merge small tables first ----
    base = dockets.merge(opinion_clusters, on="id", how="inner")

    # Optional: filter courts EARLY
    # base = base[base["court_id"].isin(RELEVANT_COURTS)]

    # ---- 3. Build ID whitelist ----
    valid_ids = set(base["id"].astype("int64"))

    # ---- 4. Prepare Parquet writer ----
    output_path = PROCESSED_DATA_DIR / "dataset.parquet"
    parquet_writer = None

    # ---- 5. Stream opinions in chunks ----
    for chunk in pd.read_csv(
        RAW_DATA_DIR / "opinions-2024-12-31.csv.bz2",
        usecols=[
            "id",
            "opinion_text",
            "author_id",
            # add ONLY needed columns
        ],
        quotechar="`",
        nrows=50,
        compression="bz2",
        chunksize=250_000,  # tune based on RAM
    ):
        # Downcast aggressively
        chunk["id"] = chunk["id"].astype("int64")

        # Filter BEFORE merge
        chunk = chunk[chunk["id"].isin(valid_ids)]
        if chunk.empty:
            continue

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
    app()
