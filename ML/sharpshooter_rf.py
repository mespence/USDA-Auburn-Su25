import numpy as np
import pandas as pd
import os
import sys
from rf import Model
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

excluded = {
    # "a01", "a02", "a03", "a10", "a15",
    # "b01", "b02", "b04", "b07", "b12", "b188", "b202", "b206", "b208",
    # "c046", "c07", "c09", "c10",
    # "d01", "d03", "d056", "d058", "d12",
    "b11", # TEST FILE
}

NON_PROBING_LABELS = ["N", "Z", "n", "z"]

def read_file(file_path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, engine="pyarrow")
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path, columns=["time", "pre_rect", "labels"], engine="pyarrow")
    else:
        return None
    return df.rename(columns={"pre_rect": "voltage"})

if __name__ == "__main__":
    data_dir = "/Users/ashleykim/Desktop/USDA/USDA-Auburn-Su25/Data/Sharpshooter Data - HPR 2017"
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
                if 'labels' not in df.columns:
                    print(f"Warning: 'labels' column not found in {future.source_path if hasattr(future, 'source_path') else 'a file'}. Skipping.")
                    continue

                labels = df['labels'].astype(str) # Ensure labels are strings for comparison

                # Apply the logic to find probing segments
                probe_mask = ~labels.isin(NON_PROBING_LABELS)
                probe_indices = np.where(probe_mask)[0]

                if len(probe_indices) == 0:
                    # No probing labels found in this DataFrame
                    continue

                # Find breaks in contiguous segments
                # A break occurs when the difference between consecutive probe_indices is greater than 1
                breaks = np.where(np.diff(probe_indices) > 1)[0]
                
                # Determine start and end indices of each segment
                segment_starts = np.insert(probe_indices[breaks + 1], 0, probe_indices[0])
                segment_ends = np.append(probe_indices[breaks], probe_indices[-1])
                
                # Iterate through each identified segment and append to probes
                for start, end in zip(segment_starts, segment_ends):
                    # Slice the original DataFrame to get the segment
                    segment_df = df.iloc[start : end + 1] # +1 because slicing is exclusive of end index
                    probes.append(segment_df)
                
    rf_model = Model()

    if probes: # Only train if there's data in probes
        rf_model.train(probes)
    else:
        print("No valid probing data found for training the model. Exiting.")
        sys.exit(1) # Exit if no training data


    print("Running Model")
    test_df = pd.read_csv("/Users/ashleykim/Desktop/USDA/USDA-Auburn-Su25/Data/Sharpshooter Data - HPR 2017/sharpshooter_b11_labeled.csv")
    test_df.rename(columns={"pre_rect": "voltage"}, inplace = True)

    # --- Start of new prediction logic for 'P' labels ---
    
    # Create a copy to work with for prediction, ensuring 'labels' are strings
    test_df_for_prediction = test_df.copy()
    test_df_for_prediction['labels'] = test_df_for_prediction['labels'].astype(str)

    # Filter test_df to include only 'P' or other valid labels (not N or Z)
    # This creates the subset on which the model will actually predict
    probing_mask_test = ~test_df_for_prediction['labels'].isin(NON_PROBING_LABELS)
    test_df_probing_sections = test_df_for_prediction[probing_mask_test]

    if test_df_probing_sections.empty:
        print("No probing sections ('P' labels) found in the test data to predict on.")
        # If no probing sections, the predictions column will remain as 'original labels' or 'NaN'
        # based on how you initialize it.
        test_df["predictions"] = test_df["labels"] # Or np.nan, or a placeholder
    else:
        # Predict only on the 'P' sections
        # Note: The rf_model.predict expects a list of DataFrames.
        # If test_df_probing_sections might have multiple actual "P" segments that were originally
        # separated by N/Z in the test_df, you might want to split it further,
        # but often the predict function can take a single contiguous DF if features are ready.
        # For simplicity, we'll pass the filtered DF as a single item in a list.
        
        # Make sure the features used for prediction match what the model was trained on.
        # Your model trains on ['time', 'voltage', 'labels'] but 'labels' isn't a feature for prediction
        # It's the target. So, select only 'time' and 'voltage' for prediction.
        predictions_on_probing_sections = rf_model.predict([test_df_probing_sections[['time', 'voltage']]])[0] # [0] because predict returns a list

        # Initialize a new column for predictions in the original DataFrame
        # You can choose how to handle non-probing areas: keep original label, set to NaN, etc.
        test_df["predictions"] = test_df["labels"] # Default to original labels

        # Assign predictions back to the original DataFrame based on the mask
        # We need the index of the filtered DataFrame to align with the original DataFrame's index
        test_df.loc[probing_mask_test, "predictions"] = predictions_on_probing_sections


        # --- Evaluation Section ---
        # Get the true labels and predicted labels for only the probing sections
        y_true = test_df.loc[probing_mask_test, "labels"]
        y_pred = test_df.loc[probing_mask_test, "predictions"]

        # Ensure that y_true and y_pred contain the same set of labels,
        # otherwise classification_report might have issues.
        # This is especially important if 'predictions_on_probing_sections'
        # only outputs 'G' but 'y_true' has 'P' and 'G'.
        
        # Get all unique labels present in the true and predicted sets
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













