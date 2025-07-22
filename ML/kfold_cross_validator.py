

import os
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors 
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix, 
    ConfusionMatrixDisplay, accuracy_score, f1_score
)

import rf


class DataImport:
    """
    A class for importing and organizing labeled time-series datasets from CSV or Parquet files.

    Attributes:
    -----------
    df_list : list[pd.DataFrame]
        A list of preprocessed DataFrames, each storing its source filename in df.attrs["file"].
    random_state : int
        Random seed used for reproducibility in KFold shuffling.
    cross_val_iter : list[tuple]
        A list of (train_index, test_index) tuples for K-fold cross-validation splits.
    """
    def __init__(self, data_path, filetype: str, exclude=[], folds=5):
        """
        Initializes the DataImport class.

        Parameters:
        -----------
        data_path : str or Path
            Directory containing the data files (.csv or .parquet).
        filetype : str
            File extension to filter files by (e.g., ".csv" or ".parquet").
        exclude : list[str], optional
            Substrings; any file whose name contains one will be excluded.
        folds : int, optional
            Number of folds to use for K-fold cross-validation (default is 5).
        """
        self.df_list = self.import_data(data_path, filetype, exclude)
        self.random_state = 42
        kf = KFold(n_splits=folds, random_state=self.random_state, shuffle=True)
        self.cross_val_iter = list(kf.split(self.df_list))

    def import_data(self, data_path, filetype, exclude):
        """
        Loads and processes EPG datasets into a list of DataFrames, each with metadata in attrs["file"].

        Parameters:
        -----------
        data_path : str or Path
            Directory containing the data files.
        filetype : str
            File extension to match (".csv" or ".parquet").
        exclude : list[str]
            Substrings to exclude from filenames.

        Returns:
        --------
        list[pd.DataFrame]
            A list of DataFrames with the source filename stored in .attrs["file"].
        """
        # collect files to be read
        filepaths = [
            str(f) for f in Path(data_path).expanduser().glob(f"*{filetype}") 
            if not any(s.lower() in f.name.lower() for s in exclude)
        ]

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
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(read_file, path) for path in filepaths]
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"Reading {len(futures)} {filetype} files"):
                df = future.result()
                if df is not None:
                    dataframes.append(df)

        return dataframes
    
    def get_probes(self, dfs):
        """
        Input: a list of dataframes containing
        """

        all_probes = []
        all_probe_names = []

        for df in dfs:
            probe_indices = self.leak_probe_finder(df["labels"].values)
            filename_base = Path(df.attrs["file"]).stem # remove extension

            probes = [
                df.iloc[start:end].reset_index(drop=True).copy()
                for start, end in probe_indices
            ]

            for i, probe in enumerate(probes):
                probe.attrs["file"] = df.attrs["file"]
                probe_name = f"{filename_base}_{i}"
                all_probes.append(probe)
                all_probe_names.append(probe_name)

        return all_probes, all_probe_names

    
    def leak_probe_finder(self, labels):
            """
            Returns a list of (start, end) index tuples for contiguous probe segments
            where labels are only the probing labels (i.e. not in the non-probing list).
            """
            NON_PROBING_LABELS = ["N", "Z"]

            probe_mask = ~pd.Series(labels).isin(NON_PROBING_LABELS)
            probe_indices = np.where(probe_mask)[0]

            if len(probe_indices) == 0:
                return []

            breaks = np.where(np.diff(probe_indices) > 1)[0]
            segment_starts = np.insert(probe_indices[breaks + 1], 0, probe_indices[0])
            segment_ends = np.append(probe_indices[breaks], probe_indices[-1])
            return list(zip(segment_starts, segment_ends))


def build_model(train_probes):
    model = rf.Model()
    model.train(train_probes)
    return model

def dynamic_importer(path):
    spec = importlib.util.spec_from_file_location("model", path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    return model

def optuna_objective(data, trial):
    labels_true = []
    labels_pred = []
    for fold, (train_index, test_index) in enumerate(data.cross_val_iter):
        train_data = [data.df_list[i] for i in train_index]
        test_data = [data.df_list[i] for i in test_index]
        train_data, _ = data.get_probes(train_data)
        test_data, _ = data.get_probes(test_data)
        
        model = rf.Model(trial = trial)
        model.train(train_data)
            
        predicted_labels = model.predict(test_data)

        # Flatten everything
        for df, preds in zip(test_data, predicted_labels):
            labels_true.extend(df["labels"].values)
            labels_pred.extend(preds)

    f1 = f1_score(labels_true, labels_pred, average="macro")
    print(f1_score(labels_true, labels_pred, average=None))
    with open(f"RF_optuna.txt", "a") as f:
        print(trial.datetime_start, trial.number, trial.params, f1, file=f)
        return f1




def plot_labels(time, voltage, true_labels, pred_labels, probs = None):
    """
    plot_labels produced a matplotlib figure containing three subplots
        that visualize a waveform along with the true and predicted labels
    Input:
        time: a series of time values
        voltage: a time series of voltage values from the waveform
        true_labels: a time series of the true label for each time point
        pred_labels: a time series of the predicted labels for each time point
    Output:
        (fig, axs): a tuple
    """


    def generate_label_colors(labels):
        labels = sorted(set(labels))
        cmap = plt.get_cmap("tab10")  # or "tab20", "nipy_spectral", etc.
        colors = [mcolors.to_hex(cmap(i / len(labels))) for i in range(len(labels))]
        return dict(zip(labels, colors))
    
    unique_labels = ['B', 'B2', 'B4', 'C', 'CG', 'D', 'DG', 'F', 'F1', 'F2', 'F3', 'F4', 'FB', 'G', 'N', 'P', 'Z']
    label_to_color = generate_label_colors(unique_labels)

    fig, axs = plt.subplots(3, 1, sharex = True)
    recording = 1
    fill_min, fill_max = voltage.min(), voltage.max()
    
    # First plot will be the true labels
    axs[0].plot(time, voltage, color = "black")
    for label, color in label_to_color.items():
        fill = axs[0].fill_between(time, fill_min, fill_max, 
                where = (true_labels == label), color=color, alpha = 0.5)
        fill.set_label(label)
    axs[0].legend(bbox_to_anchor=(0.5, 1), 
                  bbox_transform=fig.transFigure, loc="upper center", ncol=9)
    axs[0].set_title("True Labels")
    # Second plot will be the predicted labels
    axs[1].plot(time, voltage, color = "black")
    for label, color in label_to_color.items():
        axs[1].fill_between(time, fill_min, fill_max, 
                where = (pred_labels == label), color=color, alpha = 0.5)
    axs[1].set_title("Predicted Labels")
    # Third plot will be marked where there is a difference between the two
    axs[2].plot(time, voltage, color = "black")
    axs[2].fill_between(time, fill_min, fill_max, 
            where = (pred_labels != true_labels), color = "gray", alpha = 0.5)
    axs[2].set_title("Incorrect Labels")
    # Axes titles and such
    fig.supxlabel("Time (s)")
    fig.supylabel("Volts")
    fig.tight_layout()
    return fig

def generate_report(test_data, predicted_labels, test_names, save_path, model_name, fold):
    # Flatten everything
    labels_true = []
    labels_pred = []
    for df, preds in zip(test_data, predicted_labels):
        labels_true.extend(df["labels"].values)
        labels_pred.extend(preds)

    # Make sure we have a place to save everything
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    print('precision')
    # precision et. al
    labels = sorted(np.unique(labels_true))
    precision, recall, fscore, _ = precision_recall_fscore_support(
        labels_true, labels_pred, labels=labels, average=None, zero_division=0
    )
    metrics = {}
    for label, p, r, f in zip(labels, precision, recall, fscore):
        metrics[f"{label}_precision"] = p
        metrics[f"{label}_recall"] = r
        metrics[f"{label}_fscore"] = f

    out_dataframe = pd.DataFrame([metrics])

    print("accuracy")

    # accuracy
    accuracy = accuracy_score(labels_true, labels_pred)
    out_dataframe["accuracy"] = accuracy

    print("confusion matrix")
    # confusion matrix
    ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred, \
                                            normalize = 'true')
    plt.savefig(rf"{save_path}/{model_name}_ConfusionMatrix_Fold{fold}.png")

    print("difference plots")
    # difference plots
    for i, (df, preds, name) in enumerate(zip(test_data, predicted_labels, test_names)):
        fig = plot_labels(df["time"], df["voltage"], df["labels"].values, np.array(preds))
        fig.savefig(fr"{save_path}/{model_name}_{os.path.split(name)[1]}_Fold{fold}.png")
        plt.close(fig)

    print(f"Fold {fold} Overall Accuracy: {accuracy}")
    return labels_true, labels_pred, out_dataframe



def main():
    STUDY = False
    EXCLUDE = {
        "a01", "a02", "a03", "a10", "a15",
        "b01", "b02", "b04", "b07", "b12", "b188", "b202", "b206", "b208",
        "c046", "c07", "c09", "c10",
        "d01", "d03", "d056", "d058", "d12",
    }

    data_dir = r"data"

    data = DataImport(data_dir, filetype = ".parquet", exclude=EXCLUDE, folds = 5)

    if STUDY:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda x : optuna_objective(data, x), n_trials = 100, show_progress_bar=True)
        print(study.best_params)
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(f"RF_hyper.png")
        return
    
    summary_data = []
    labels_true = []
    labels_pred = []
    for fold, (train_index, test_index) in enumerate(data.cross_val_iter):
        print(f"=== Evaluating Fold {fold} ===")
        train_data = [data.df_list[i] for i in train_index]
        test_data = [data.df_list[i] for i in test_index]
        train_data, _ = data.get_probes(train_data)
        test_data, test_names = data.get_probes(test_data)

        model = rf.Model()
        print("Training Model...")
        model.train(train_data)

        print("Evaluating Model...")
        predicted_labels = model.predict(test_data)

        print("Generating Report...")
        true, pred, stats = generate_report(test_data, predicted_labels, test_names, "out", "RF", fold)
        print("Report generated.")
        summary_data.append(stats)
        labels_true.extend(true)
        labels_pred.extend(pred)
        
    out_summary_data = pd.concat(summary_data)

    # Calculate statistics across every dataset
    labels = sorted(np.unique(labels_true))
    all_precision, all_recall, all_fscore, _ = precision_recall_fscore_support(labels_true, labels_pred, 
                                                            labels=labels, average = None, zero_division=0)
    temp_dict = {"precision" : all_precision, 
                 "recall" : all_recall, 
                 "fscore" : all_fscore}
    out_dataframe = pd.DataFrame(temp_dict, index=labels).stack()
    out_dataframe.index = out_dataframe.index.map('{0[1]}_{0[0]}'.format)
    out_dataframe = out_dataframe.to_frame().T
    out_dataframe["accuracy"] = accuracy_score(labels_true, labels_pred)

    out_summary_data = pd.concat([out_summary_data, out_dataframe])
    out_summary_data.to_csv(f"out/RF_SummaryStats.csv")

    overall = ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred, \
                                            normalize = 'true')
    overall.plot().figure_.savefig(rf"out/RF_OverallConfusionMatrix.png")

    all_data = pd.DataFrame({'labels_true': labels_true,
                             'labels_pred': labels_pred})
    all_data.to_csv(f"out/RF_allpredictions.csv")


if __name__ == "__main__":
    main()