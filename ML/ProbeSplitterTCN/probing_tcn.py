

import os
import json
import pandas as pd


NON_PROBING_LABELS = ["N", "Z"]


def load_label_map(save_path: str) -> tuple[dict, dict]:
    """
    Load a label map from disk.

    Parameters:
        save_path (str): Path to load label_map.json

    Returns:
        label_map (Dict[str, int]): the label map
        inv_label_map (Dict[int, str]): the inverse label map
    """
    label_map_path = os.path.join(save_path, "label_map.json")

    if not os.path.exists(label_map_path):
        return None
    
    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    label_map = {k: int(v) for k, v in label_map.items()}
    inv_label_map = {v: k for k, v in label_map.items()}
    return label_map, inv_label_map

def build_probing_label_map(dataframes: list[pd.DataFrame], label_column = "labels") -> tuple[dict, dict]:
    """
    Build probing label map from a list of DataFrames and writes to `label_map.json`.
    "N" and "Z" are non-probing (0), all other labels are probing (1).    

    Parameters:
        dataframes: List of pandas DataFrames with a column containing labels.
        label_column (str): Column containing label strings (e.g. "Z", "D", "F1")

    Returns:
        label_map (Dict[str, int]): mapping of each original label to 0 or 1
        inv_label_map (Dict[int, list[str]]): reverse mapping from 0/1 to original labels
    """
    all_labels = set()
    for df in dataframes:
        labels = df[label_column].dropna().unique()
        all_labels.update(label.upper() for label in labels)

    label_map = {
        label: 0 if label in NON_PROBING_LABELS else 1
        for label in sorted(all_labels)
    }
    
    inv_label_map = {0: [], 1: []}
    for label, val in label_map.items():
        inv_label_map[val].append(label)

    output_path = os.path.join(os.path.dirname(__file__), 'label_map.json')
    with open(output_path, "w") as f:
        json.dump(label_map, f, indent=4)

    return label_map, inv_label_map




if __name__ == "__main__":
    DATA_DIR = r"..\data"

    excluded = {
        "a01", "a02", "a03", "a10", "a15",
        "b01", "b02", "b04", "b07", "b12", "b188", "b202", "b206", "b208",
        "c046", "c07", "c09", "c10",
        "d01", "d03", "d056", "d058", "d12",
    }

    def read_file(file_path):
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, engine="pyarrow")
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path, columns=["time", "pre_rect", "labels"], engine="pyarrow")
        else:
            return None
        return df.rename(columns={"pre_rect": "voltage"})
    
    data_dfs = []

    for filename in os.listdir(DATA_DIR):
        if any(bug_id in filename for bug_id in excluded):
            continue

        print(f"Reading {filename}...")
        file_path = os.path.join(DATA_DIR, filename)

        df = read_file(file_path)
        if df is None:
            continue

        # 🔍 Optional debug check: find where label == 2
        if "labels" in df.columns:
            label_matches = df.index[df["labels"] == "U"].tolist()
            if label_matches:
                print(f"[!] Label U found in '{filename}'")# on row(s): {label_matches}")

        data_dfs.append(df)

    build_probing_label_map(data_dfs)






# class ProbeSplitterTCN:
#     def __init__(self):
#         self.label_map = 

