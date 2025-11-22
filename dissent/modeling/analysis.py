import pandas as pd

from dissent.config import INTERIM_DATA_DIR

# Need a number of dissents every month for every state court


def main():
    df = pd.read_csv(INTERIM_DATA_DIR / "dataset.csv.bz2")
    print(df.columns)
    print(df[["type"]].head())


if __name__ == "__main__":
    main()
