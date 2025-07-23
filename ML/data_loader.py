import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Union, Optional
from collections.abc import Iterable


def import_data(
    data_path: Union[str, Path],
    filetype: str,
    exclude: Optional[Iterable[str]] = None,
    include: Optional[Iterable[str]] = None
):
    """
    Loads and processes EPG datasets into a list of DataFrames, each with metadata in attrs["file"].

    Parameters:
    -----------
    data_path : str or Path
        Directory containing the data files.
    filetype : str
        File extension to match (".csv" or ".parquet").
    exclude : iterable of str, optional
        Substrings to exclude from filenames.
    include : iterable of str, optional
        Substrings to include in filenames (must match at least one to be included).

    Returns:
    --------
    list[pd.DataFrame]
        A list of DataFrames with the source filename stored in .attrs["file"].
    """
    exclude = set(s.lower() for s in exclude) if exclude else set()
    include = set(s.lower() for s in include) if include else None

    # collect files to be read
    filepaths = []
    for f in Path(data_path).expanduser().glob(f"*{filetype}"):
        fname = f.name.lower()
        if exclude and any(s in fname for s in exclude):
            continue
        if include and not any(s in fname for s in include):
            continue
        filepaths.append(str(f))

    def read_file(filepath):
        """Helper function to read and clean a single file based on extension."""
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath, index_col=0, engine="pyarrow")
            df.drop(columns=["post_rect"], errors="ignore", inplace=True)
        elif filepath.endswith(".parquet"):
            df = pd.read_parquet(filepath, columns=["time", "pre_rect", "labels"], engine="pyarrow")
            df.reset_index(drop=True, inplace=True)
        else:
            return None
        df.rename(columns={"pre_rect": "voltage"}, inplace=True)
        df.attrs["file"] = filepath
        return df

    # multi-threaded reading
    dataframes = []
    total_bytes = 0
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(read_file, path): path for path in filepaths}
        progress_bar = tqdm(
            as_completed(futures),total=len(futures), 
            desc=f"Reading {len(futures)} {filetype} files"
        )
        for future in progress_bar:
            df = future.result()
            if df is not None:
                dataframes.append(df)
                total_bytes += os.path.getsize(futures[future])
    elapsed_time = progress_bar.format_dict["elapsed"]
    total_gb = total_bytes / (1024 * 1024 * 1024)

    tqdm.write(f"Loaded {len(dataframes)} files ({total_gb:.2f} GB) in {elapsed_time:.2f} seconds.")

    return dataframes