from loguru import logger
import pandas as pd

from dissent.config import INTERIM_DATA_DIR

# Need a number of dissents every month for every state court


def main():
    df = pd.read_csv(
        INTERIM_DATA_DIR / "dataset.csv.bz2", usecols=["date_filed", "type", "court_id"]
    )
    print(len(df))
    # Filter for dissents
    df = df[df["type"].isin(["nc"])]

    logger.info(df.head())


if __name__ == "__main__":
    main()
