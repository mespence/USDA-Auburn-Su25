import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def read_file(file_path):
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, engine="pyarrow")
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path, columns=["time", "pre_rect", "labels"], engine="pyarrow")
        else:
            return None
        return df.rename(columns={"pre_rect": "voltage"})
    except Exception as e:
        print(f"[!] Error reading {file_path}: {e}")
        return None


def load_dfs_from_dir(data_dir, excluded = [], max_workers=None) -> list[pd.DataFrame]:
    """
    Loads .csv and .parquet files in parallel, excluding files with IDs in the excluded set.
    
    Args:
        data_dir (str): Directory containing the files.
        excluded (set[str]): Set of substrings to exclude based on filename.
        max_workers (int): Number of threads to use.
    
    Returns:
        List[pd.DataFrame]: List of successfully loaded DataFrames.
    """
    all_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith((".csv", ".parquet")) and not any(bug_id in f for bug_id in excluded)
    ]

    data_dfs = []
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_file, file_path): file_path for file_path in all_files}
        progress_bar = tqdm(as_completed(futures), total=len(futures), desc="Loading files")
        for future in progress_bar:
            df = future.result()
            file_path = futures[future]
            if df is not None:
                data_dfs.append(df)
                total_bytes += os.path.getsize(file_path)

    elapsed_time = progress_bar.format_dict["elapsed"]
    total_gb = total_bytes / (1024 * 1024 * 1024)

    print(f"Loaded {len(data_dfs)} files ({total_gb:.2f} GB) in {elapsed_time:.2f} seconds.")

    return data_dfs
