import numpy as np
import pandas as pd
import os
import sys
from rf import Model
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import optuna

from sklearn.metrics import precision_recall_fscore_support, \
                            confusion_matrix, \
                            ConfusionMatrixDisplay, \
                            accuracy_score, \
                            f1_score

excluded = {
    "a01", "a02", "a03", "a10", "a15",
    "b01", "b02", "b04", "b07", "b12", "b188", "b202", "b206", "b208",
    "c046", "c07", "c09", "c10",
    "d01", "d03", "d056", "d058", "d12",
    "b11", # TEST FILE
}

        
def read_file(filepath):   
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



if __name__ == "__main__":
    data_dir = r"D:\USDA-Auburn\CS-Repository\Data\Sharpshooter Data - HPR 2017\sharpshooter_parquet"
    probes = []

    # Collect valid file paths
    file_paths = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if not any(bug_id in filename for bug_id in excluded) and filename.endswith((".csv", ".parquet"))
    ]
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(read_file, path) for path in file_paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading files"):
            df = future.result()
            if df is not None:
                probes.append(df)
                
    def optuna_objective(data, trial):
        labels_true = []
        labels_pred = []
        for fold, (train_index, test_index) in enumerate(data.cross_val_iter):
            train_data = [data.raw_dfs[i] for i in train_index]
            test_data = [data.raw_dfs[i] for i in test_index]
            #train_data, _ = data.get_probes(train_data)
            #test_data, test_names = data.get_probes(test_data)

            
            model = Model(trial = trial)
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

    #study = optuna.create_study(direction='maximize')
    #study.optimize(lambda x : optuna_objective(probes, x), n_trials = 100, show_progress_bar=True, )


    
    rf_model = Model()
    rf_model.train(probes)


    print("Running Model")
    test_df = pd.read_csv(r"C:\Users\Clinic\Desktop\USDA-Auburn-Su25\Data\Sharpshooter Data - HPR 2017\sharpshooter_b11_labeled.csv")
    test_df.rename(columns={"pre_rect": "voltage"}, inplace = True)
    predictions = rf_model.predict([test_df])[0]
    print("Model run.")
    print()
    print(predictions)

    print("Saving output...")
    rf_model.save()
    test_df["labels"] = predictions
    test_df.to_csv("out.csv")
    print("Output saved.")













