import os
import pandas as pd
from pathlib import Path


###################################################################
# This script splits parsed CSVs into labeled and unlabeled files.
###################################################################

DATA_DIR = os.path.dirname(os.path.realpath(__file__))
CSV_DIR = Path(DATA_DIR + r"\sharpshooter_parsed")
LABELED_DIR = DATA_DIR + r"\sharpshooter_labeled"
UNLABELED_DIR = DATA_DIR + r"\sharpshooter_unlabeled"

# Loop through all .csv files in the folder
for csv_path in CSV_DIR.glob("*.csv"):
    print(f"Processing {csv_path.name}...")

    df = pd.read_csv(csv_path, index_col=0, low_memory=False)

    label_col = "labels"

    label_series = df[label_col]
    first_unlabeled_idx = label_series.isna() | (label_series == "")
    cutoff = first_unlabeled_idx.idxmax() if first_unlabeled_idx.any() else len(df)

    labeled = df.iloc[:cutoff]
    unlabeled = df.iloc[cutoff:]

    unlabeled = unlabeled.drop(columns=[label_col])

    base_name = csv_path.stem.replace("_raw", "")
    labeled_path = LABELED_DIR + rf"\{base_name}_labeled.csv"
    unlabeled_path = UNLABELED_DIR + rf"\{base_name}_unlabeled.csv"
    print(f"Parsed {csv_path.stem}")

    labeled.to_csv(labeled_path)
    print(f"✔️ Wrote {len(labeled)} labeled rows → {labeled_path}")

    unlabeled.to_csv(unlabeled_path)
    print(f"✔️ Wrote {len(unlabeled)} unlabeled rows → {unlabeled_path}")