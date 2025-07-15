import os
import glob
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import importlib.util
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import optuna

from old.data_augmentation import build_augmented_dataset
from old.postprocessing import PostProcessor

class DataImport:
    def __init__(self, data_path, folds):
        self.raw_dfs, _ = self.import_data(data_path)
        self.random_state = 42
        kf = KFold(n_splits=folds, random_state=self.random_state, shuffle=True)
        self.cross_val_iter = list(kf.split(self.raw_dfs))

    def import_data(self, data_path):
        filenames = glob.glob(os.path.expanduser(f"{data_path}/*.csv"))
        dataframes = [pd.read_csv(f) for f in filenames]
        for file, df in zip(filenames, dataframes):
            df["file"] = file
            df['labels'] = df['labels'].replace('Z', 'W')
        return dataframes, filenames

    def get_train_test_split(self):
        return train_test_split(self.raw_dfs, test_size=0.2, random_state=self.random_state)

def dynamic_importer(path):
    spec = importlib.util.spec_from_file_location("model", path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model

def plot_labels(time, voltage, true_labels, pred_labels):
    all_labels = sorted(set(np.unique(true_labels)) | set(np.unique(pred_labels)))
    cmap = cm.get_cmap("tab20", len(all_labels))
    label_to_color = {label: cmap(i) for i, label in enumerate(all_labels)}

    fig, axs = plt.subplots(3, 1, sharex=True)
    fill_min, fill_max = voltage.min(), voltage.max()

    axs[0].plot(time, voltage, color="black")
    for label, color in label_to_color.items():
        axs[0].fill_between(time, fill_min, fill_max, where=(true_labels == label), color=color, alpha=0.5)
    axs[0].set_title("True Labels")

    axs[1].plot(time, voltage, color="black")
    for label, color in label_to_color.items():
        axs[1].fill_between(time, fill_min, fill_max, where=(pred_labels == label), color=color, alpha=0.5)
    axs[1].set_title("Predicted Labels")

    axs[2].plot(time, voltage, color="black")
    axs[2].fill_between(time, fill_min, fill_max, where=(pred_labels != true_labels), color="gray", alpha=0.5)
    axs[2].set_title("Incorrect Labels")

    fig.supxlabel("Time (s)")
    fig.supylabel("Volts")
    fig.tight_layout()
    return fig

def generate_report(test_data, predicted_labels, test_names, save_path, model_name, fold):
    labels_true, labels_pred = [], []
    for df, preds in zip(test_data, predicted_labels):
        labels_true.extend(df["labels"].values)
        labels_pred.extend(preds)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    labels = sorted(np.unique(labels_true))
    precision, recall, fscore, _ = precision_recall_fscore_support(labels_true, labels_pred, labels=labels, average=None, zero_division=0)
    stats_df = pd.DataFrame({"precision": precision, "recall": recall, "fscore": fscore}, index=labels).stack()
    stats_df.index = stats_df.index.map('{0[1]}_{0[0]}'.format)
    stats_df = stats_df.to_frame().T
    stats_df["accuracy"] = accuracy_score(labels_true, labels_pred)

    ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred, normalize='true')
    plt.savefig(rf"{save_path}/{model_name}_ConfusionMatrix_Fold{fold}.png")

    for df, preds, name in zip(test_data, predicted_labels, test_names):
        fig = plot_labels(df["time"], df["pre_rect"], df["labels"].values, np.array(preds))
        fig.savefig(fr"{save_path}/{model_name}_{os.path.basename(name)}_Fold{fold}.png")
        plt.close(fig)

    print(f"Fold {fold} Overall Accuracy: {stats_df['accuracy'].values[0]}")
    return labels_true, labels_pred, stats_df

def optuna_objective(data, args, trial, **kwargs):
    labels_true, labels_pred = [], []
    for fold, (train_index, test_index) in enumerate(data.cross_val_iter):
        train_data = [data.raw_dfs[i] for i in train_index]
        test_data = [data.raw_dfs[i] for i in test_index]

        if args.augment:
            train_data = build_augmented_dataset(train_data, size=len(train_data))

        model = dynamic_importer(args.model_path).Model(trial=trial, **kwargs)
        model.train(train_data, test_data, fold)
        preds = model.predict(test_data)

        for df, p in zip(test_data, preds):
            labels_true.extend(df["labels"].values)
            labels_pred.extend(p)

    f1 = f1_score(labels_true, labels_pred, average="macro")
    with open(f"{args.model_name}_optuna.txt", "a") as f:
        print(trial.datetime_start, trial.number, trial.params, f1, file=f)
    return f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--post_process")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--optuna", action="store_true")
    parser.add_argument("--attention", action="store_true")
    args = parser.parse_args()

    print("Loading Data...")
    data = DataImport(args.data_path, 5)

    if args.optuna:
        study = optuna.create_study(direction='maximize')
        kwargs = {}
        study.optimize(lambda x: optuna_objective(data, args, x, **kwargs), n_trials=100, show_progress_bar=True)
        print(study.best_params)
        return

    summary_data, labels_true, labels_pred = [], [], []
    for fold, (train_index, test_index) in enumerate(data.cross_val_iter):
        train_data = [data.raw_dfs[i] for i in train_index]
        test_data = [data.raw_dfs[i] for i in test_index]
        test_names = [df["file"].iloc[0] for df in test_data]

        if args.augment:
            train_data = build_augmented_dataset(train_data, size=len(train_data))

        model = dynamic_importer(args.model_path).Model(save_path=args.save_path)
        model.train(train_data, test_data, fold)

        if args.post_process is None:
            predicted_labels = model.predict(test_data)
        else:
            _, logits = model.predict(test_data, return_logits=True)
            logits = [l.squeeze(0).detach().numpy() for l in logits]
            post_process = PostProcessor(train_data, model.inv_label_map)
            if args.post_process.lower() in {"viterbi", "v"}:
                predicted_labels = [post_process.postprocess_viterbi(l) for l in logits]
            elif args.post_process.lower() in {"smooth", "s"}:
                predicted_labels = [post_process.postprocess_smooth(l) for l in logits]
            else:
                raise ValueError("Invalid postprocess argument")

        true, pred, stats = generate_report(test_data, predicted_labels, test_names, args.save_path, args.model_name, fold)
        summary_data.append(stats)
        labels_true.extend(true)
        labels_pred.extend(pred)

    out_summary_data = pd.concat(summary_data)
    labels = sorted(np.unique(labels_true))
    precision, recall, fscore, _ = precision_recall_fscore_support(labels_true, labels_pred, labels=labels, average=None, zero_division=0)
    final_df = pd.DataFrame({"precision": precision, "recall": recall, "fscore": fscore}, index=labels).stack()
    final_df.index = final_df.index.map('{0[1]}_{0[0]}'.format)
    final_df = final_df.to_frame().T
    final_df["accuracy"] = accuracy_score(labels_true, labels_pred)

    full_stats = pd.concat([out_summary_data, final_df])
    full_stats.to_csv(f"{args.save_path}/{args.model_name}_SummaryStats.csv")

    ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred, normalize='true')
    plt.savefig(f"{args.save_path}/{args.model_name}_OverallConfusionMatrix.png")

    pd.DataFrame({'labels_true': labels_true, 'labels_pred': labels_pred}).to_csv(f"{args.save_path}/{args.model_name}_allpredictions.csv")

if __name__ == "__main__":
    main()
