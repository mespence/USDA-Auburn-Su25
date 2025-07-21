

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import rf


class DataImport:
    def __init__(self, data_path, filetype: str = ".csv", exclude = [], folds = 5):
        self.df_dict = self.import_data(data_path, filetype, exclude)
        self.df_list = list(self.df_dict.values())
        self.random_state = 42
        kf = KFold(n_splits = folds, random_state = self.random_state, shuffle = True)
        self.cross_val_iter = list(kf.split(self.df_list))

    def import_data(self, data_path, filetype, exclude):
        """
        import_data takes in a path to cleaned data (.csv or .parquet) and 
        returns it as a list of dataframes, excluding filenames with the 
        substrings in `exclude`
        """
        # collect files to be read
        filepaths = [
            str(f) for f in Path(data_path).expanduser().glob(f"*{filetype}") 
            if not any(s.lower() in f.name.lower() for s in exclude)
        ]

        def read_file(filepath):
            if filepath.endswith(".csv"):
                df = pd.read_csv(filepath, index_col = 0, engine="pyarrow")
            elif filepath.endswith(".parquet"):
                df = pd.read_parquet(filepath, columns=["time", "pre_rect", "labels"], engine="pyarrow")
                df = df.set_index(df.columns[0])
            else:
                return None, None
            df = df.rename(columns={"pre_rect": "voltage"}).drop(columns=["post_rect"], errors="ignore") 
            return filepath, df

        # multi-threaded reading
        data_dict = {}
        try:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(read_file, path) for path in filepaths]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Reading {len(futures)} {filetype} files", leave = False):
                    file, df = future.result()
                    if file and df is not None:
                        data_dict[file] = df
        except KeyboardInterrupt:
            print("Shutting down threads...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise  

        return data_dict



def run_kfold_crossval(probes, build_model_fn, k=5, random_seed=42, verbose=True):

    """
    Run K-Fold cross-validation using a list of DataFrames (probes).

    Parameters:
    - probes: list of pd.DataFrame, each with a "labels" column.
    - build_model_fn: function that takes `train_probes` and returns a model with a `predict(test_probes)` method.
    - k: number of folds.
    - random_seed: for reproducibility.
    - verbose: print evaluation results.

    Returns:
    - List of classification reports and accuracies for each fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(probes)):
        train_probes = [probes[i] for i in train_idx]
        test_probes = [probes[i] for i in test_idx]

        if verbose:
            print(f"\n=== Fold {fold_idx + 1}/{k} ===")
            print(f"Train set: {len(train_probes)} probes")
            print(f"Test set: {len(test_probes)} probes")

        # Train model
        model = build_model_fn(train_probes)
        model.save()

        # Evaluate
        all_true, all_pred = [], []
        for probe in test_probes:
            true_labels = probe["labels"].values
            pred_labels = model.predict([probe])[0]
            all_true.extend(true_labels)
            all_pred.extend(pred_labels)

        acc = accuracy_score(all_true, all_pred)
        report = classification_report(all_true, all_pred, output_dict=True)

        if verbose:
            print(f"Accuracy: {acc:.4f}")

        results.append({
            "fold": fold_idx + 1,
            "accuracy": acc,
            "report": report
        })

    return results

def build_model(train_probes):
    model = rf.Model()
    model.train(train_probes)
    return model


if __name__ == "__main__":
    EXCLUDE = {
        "a01", "a02", "a03", "a10", "a15",
        "b01", "b02", "b04", "b07", "b12", "b188", "b202", "b206", "b208",
        "c046", "c07", "c09", "c10",
        "d01", "d03", "d056", "d058", "d12",
        "_b", "_c", "_d"
    }

    data_dir = r"data/sharpshooter_labeled"

    data = DataImport(data_dir, exclude=EXCLUDE, folds = 5)
    print(data.df_dict)


    #run_kfold_crossval(data, build_model)

    


