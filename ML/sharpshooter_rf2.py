import numpy as np
import pandas as pd
import os
import sys
import optuna
from sklearn.model_selection import KFold
from rf import Model
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import precision_recall_fscore_support, \
                            confusion_matrix, \
                            classification_report, \
                            ConfusionMatrixDisplay, \
                            accuracy_score, \
                            f1_score
from tqdm import tqdm

excluded = {
    # "a01", "a02", "a03", "a10", "a15",
    # "b01", "b02", "b04", "b07", "b12", "b188", "b202", "b206", "b208",
    # "c046", "c07", "c09", "c10",
    # "d01", "d03", "d056", "d058", "d12",
    "b11", # TEST FILE
}

NON_PROBING_LABELS = ["N", "Z", "n", "z"]

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

def get_probing_sections(df) -> list[pd.DataFrame]:
    """
    Extract continuous probing sections from df
    """
    probing_sections_list = []

    labels = df['labels'].astype(str) # Ensure labels are strings for comparison

    # Search labels for indices with probing labels
    probe_mask = ~labels.isin(NON_PROBING_LABELS)
    probe_indices = np.where(probe_mask)[0]

    if len(probe_indices) == 0:
        # No probing labels found in this DataFrame
        return []

    # A break bt probing occurs when the difference between consecutive probe_indices is greater than 1
    breaks = np.where(np.diff(probe_indices) > 1)[0]
    
    # Determine start and end indices of each segment
    segment_starts = np.insert(probe_indices[breaks + 1], 0, probe_indices[0])
    segment_ends = np.append(probe_indices[breaks], probe_indices[-1])
    
    # Iterate through each identified segment and append to probes
    for start, end in zip(segment_starts, segment_ends):
        # Slice the original DataFrame to get the segment
        segment_df = df.iloc[start : end + 1].copy() # +1 bc slicing is exclusive of end index
        probing_sections_list.append(segment_df)
    
    return probing_sections_list


if __name__ == "__main__":
    data_dir = r"C:\Users\Clinic\Desktop\USDA-Auburn-Su25\Data\Sharpshooter Data - HPR 2017\Data"
    # r"C:\Users\Clinic\Desktop\USDA-Auburn-Su25\Data\Sharpshooter Data - HPR 2017"
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
                probes.extend(get_probing_sections(df))

    def optuna_objective(data, trial, cross_val_iter):
        labels_true = []
        labels_pred = []
        for fold, (train_index, test_index) in enumerate(cross_val_iter):
            train_data = [data[i] for i in train_index]
            test_data = [data[i] for i in test_index]
            #train_data, _ = data.get_probes(train_data)
            #test_data, test_names = data.get_probes(test_data)

            model = Model(trial = trial)
            model.train(train_data)
                
            predicted_labels = model.predict(test_data)

            # Flatten everything
            for df, preds in zip(test_data, predicted_labels):
                labels_true.extend(df["labels"].values)
                labels_pred.extend(preds)

        f1_weighted = f1_score(labels_true, labels_pred, average='weighted')
        f1 = f1_score(labels_true, labels_pred, average="macro")
        print("f1 weighted: ", f1_weighted)
        with open(f"RF_optuna.txt", "a") as f:
            print(trial.datetime_start, trial.number, trial.params, f1_weighted, file=f)
        return f1_weighted
    
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    cross_val_iter = list(kf.split(probes))

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda x : optuna_objective(probes, x, cross_val_iter), n_trials = 100, show_progress_bar=True, )
                
    rf_model = Model()

    if probes: # Only train if there's data for probing sections
        rf_model.train(probes)
    else:
        print("No valid probing data found for training the model. Exiting.")
        sys.exit(1) # Exit if no training data

    print("Running Model")
    test_df = pd.read_csv(r"C:\Users\Clinic\Desktop\USDA-Auburn-Su25\Data\Sharpshooter Data - HPR 2017\Data\sharpshooter_b11_labeled.csv")
    test_df.rename(columns={"pre_rect": "voltage"}, inplace = True)

    # --- Start of new prediction logic for 'P' labels ---
    
    # Create a copy to work with for prediction, ensuring 'labels' are strings
    test_df_for_prediction = test_df.copy()
    test_df_for_prediction['labels'] = test_df_for_prediction['labels'].astype(str)

    test_probing_segments = get_probing_sections(test_df_for_prediction)

    # Initialize a new column for predictions in the original DataFrame
    test_df["predictions"] = test_df["labels"] # Default to original labels

    if not test_probing_segments:
        print("No probing sections ('P' labels) found in the test data to predict on.")
    else:
        # Predict on list of probing segments
        # pass only time and voltage features
        segments_for_prediction = [seg[['time', 'voltage']] for seg in test_probing_segments]
        
        predictions_on_probing_sections_list = rf_model.predict(segments_for_prediction)

        # Assign predictions back to the original df based on their original indices
        current_prediction_idx = 0
        for i, segment_df in enumerate(test_probing_segments):
            # The indices of the segment_df are its original indices from test_df
            original_indices = segment_df.index 
            test_df.loc[original_indices, "predictions"] = predictions_on_probing_sections_list[i]
            # Ensure the length matches, if not, there's an issue with prediction output or slicing
            assert len(predictions_on_probing_sections_list[i]) == len(original_indices)

        # --- Evaluation Section ---
        # Get the true labels and predicted labels for only the probing sections
        
        # We need the original true probing labels for comparison
        y_true = test_df_for_prediction[~test_df_for_prediction['labels'].isin(NON_PROBING_LABELS)]['labels']
        y_pred = test_df.loc[y_true.index, "predictions"] # Get predictions for the same indices

        all_labels = sorted(list(set(y_true.unique()).union(set(y_pred.unique()))))


        print("\n--- Model Evaluation (on Probing Sections Only) ---")
        
        # 1. Classification Report (Precision, Recall, F1-score for each class)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0)) # zero_division=0 to avoid warnings for classes with no true or predicted instances

        # 2. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        print("\nConfusion Matrix:")
        print(cm)
        
        # 3. Overall Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nOverall Accuracy (on Probing Sections): {accuracy:.4f}")

        # 4. F1-score (Macro and Weighted average)
        # Macro: Calculates metric for each label, and finds their unweighted mean.
        # Weighted: Calculates metric for each label, and finds their average, weighted by support.
        f1_macro = f1_score(y_true, y_pred, average='macro', labels=all_labels, zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', labels=all_labels, zero_division=0)
        print(f"Macro F1-score (on Probing Sections): {f1_macro:.4f}")
        print(f"Weighted F1-score (on Probing Sections): {f1_weighted:.4f}")

    print("Model run.")
    print()
    print(test_df["predictions"].value_counts()) # Show predicted label distribution

    print("Saving output...")
    # It's better to save the original test_df with a new "predictions" column
    test_df.to_csv("out.csv", index=False) # index=False prevents writing DataFrame index
    print("Output saved.")
    


    # predictions = rf_model.predict([test_df])[0]
    # print("Model run.")
    # print()
    # print(predictions)

    # print("Saving output...")
    # rf_model.save()
    # test_df["labels"] = predictions
    # test_df.to_csv("out.csv")
    # print("Output saved.")













