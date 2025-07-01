import pandas as pd
from pathlib import Path

# Set your CSV path
csv_path = Path(r"C:\EPG-Project\Summer\CS-Repository\Data\Sharpshooter Data - HPR 2017\Backus BGSS 2017 HPR data - for Mudd CS.csv")  # <-- change this

# Load the CSV
df = pd.read_csv(csv_path)

# Determine the label column
label_col = 'waveform'

# Drop NaNs and empty strings, then get unique values
unique_labels = df[label_col].dropna()
unique_labels = unique_labels[unique_labels != ""].unique()

# Sort and print
print(f"Unique labels in '{csv_path.name}':")
for label in sorted(unique_labels):
    print(f" - {label}")
