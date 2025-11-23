# Getting started

## System dependencies

1. `uv`

## Installation

1. Clone the repository: `git clone github.com/aerithnetzer/dissention.git`

2. `cd` into repository

3. Run `uv sync`

4. Run `make environment`

## Dataset

Run `make dataset`

This will:

1. Download all necessary `.csv.bz2` files from CourtListener
2. Join them together into a single dataset using `cudf`
3. Save the dataset in `data/interim/dataset.csv.bz2`
