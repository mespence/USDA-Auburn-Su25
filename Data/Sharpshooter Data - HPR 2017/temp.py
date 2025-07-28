import pandas as pd
from pathlib import Path

def find_unlabeled_rows_in_file(csv_path, label_col="labels"):
    try:
        df = pd.read_csv(csv_path, index_col=0, low_memory=False)

        unlabeled_mask = df[label_col].isna() | (df[label_col].astype(str).str.strip() == "")
        unlabeled_rows = df[unlabeled_mask]

        if unlabeled_rows.empty:
            print(f"[OK] {csv_path.name}: All rows labeled ✅")
        else:
            print(f"[WARN] {csv_path.name}: {len(unlabeled_rows)} unlabeled rows:")
            for idx, row in unlabeled_rows.iterrows():
                print(f"   - Row index {idx}, time = {row.get('time', '??')}, label = {row[label_col]!r}")
    except Exception as e:
        print(f"[ERROR] {csv_path.name}: Failed to read/process file ({e})")

def find_unlabeled_rows(path, label_col="labels"):
    path = Path(path)
    if path.is_file():
        find_unlabeled_rows_in_file(path, label_col)
    elif path.is_dir():
        csv_files = list(path.glob("*.csv"))
        print(f"Scanning directory: {path} ({len(csv_files)} CSV files found)")
        for csv_file in csv_files:
            find_unlabeled_rows_in_file(csv_file, label_col)
    else:
        print(f"[ERROR] Path not found: {path}")

# --- Example usage ---
# Replace with your directory or file path
find_unlabeled_rows(r"D:\USDA-Auburn\CS-Repository\Data\Sharpshooter Data - HPR 2017\sharpshooter_labeled")
