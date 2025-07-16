import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score, classification_report

# Best Parameters:
#     window: 961
#     threshold: 0.09406124969163045
#     min_probe_length: 867
#     np_pad: 390

# --- Evaluation with Best Parameters ---
# Accuracy with best parameters: 0.9793
#               precision    recall  f1-score   support

#           NP       0.98      0.99      0.98   2379522
#            P       0.97      0.97      0.97   1220479

#     accuracy                           0.98   3600001
#    macro avg       0.98      0.98      0.98   3600001
# weighted avg       0.98      0.98      0.98   3600001

class ProbeSplitter:
    def simple_probe_finder(recording, window = 950, threshold = 0.1,
                        min_probe_length = 850, np_pad = 80):
        """
        Input: recording: A pre-rectified mosquito recording as a 1-D nupmy 
                    array. Pre-rectified recordings are necessary as baseline is 
                    not 0 in post-rectified recordings.
                window: Before NP regions can be identified, a rolling
                    average filter is applied to remove noise in the NP regions.
                    window is the size of this filter in samples.
                threshold: The maximum value of an NP sample.
                min_probe_length: The minimum acceptable length of a probe in
                    samples.
                np_pad: the number of NP samples before and after each probe to
                    include. Note that high values might result in overlapping with
                    the next probe.
        Output: A list of (start sample, end sample) tuples for each probe. By 
                default contains about 5 seconds of NPs at the beginning and end            
                of each probe. We say "about" because this splitting is done
                in an unsupervised manner, although it is largely pretty good.
        """
        
        smoothed = np.convolve(recording, np.ones(window), "same")/window
        is_NP = smoothed < threshold # NP is where the signal is close to 0
        
        # Find starts and ends, combine into tuple
        find_sequence = lambda l, seq : [i for i in range(len(l)) if l[i:i+len(seq)] == seq]
        is_NP_list = list(is_NP)
        probe_starts = find_sequence(is_NP_list, [True, False])
        probe_ends = find_sequence(is_NP_list, [False, True])
        probes = zip(probe_starts, probe_ends)
        
        # Remove probes that are too short and pad
        probes = [(max(0, start - np_pad), end + np_pad) for start, end in probes if end - start > min_probe_length]
        
        return probes

sharpshooter_files = [
    "/Users/ashleykim/Desktop/USDA/USDA-Auburn-Su25/Data/Sharpshooter Data - HPR 2017/sharpshooter_d198_labeled.csv",
    "/Users/ashleykim/Desktop/USDA/USDA-Auburn-Su25/Data/Sharpshooter Data - HPR 2017/sharpshooter_b11_labeled.csv",
    "/Users/ashleykim/Desktop/USDA/USDA-Auburn-Su25/Data/Sharpshooter Data - HPR 2017/sharpshooter_d07_labeled.csv",
    "/Users/ashleykim/Desktop/USDA/USDA-Auburn-Su25/Data/Sharpshooter Data - HPR 2017/sharpshooter_c182_labeled.csv",
    "/Users/ashleykim/Desktop/USDA/USDA-Auburn-Su25/Data/Sharpshooter Data - HPR 2017/sharpshooter_a12_labeled.csv",
    "/Users/ashleykim/Desktop/USDA/USDA-Auburn-Su25/Data/Sharpshooter Data - HPR 2017/sharpshooter_c02_labeled.csv",
]


# file_path = "/Users/ashleykim/Desktop/USDA/USDA-Auburn-Su25/Data/Sharpshooter Data - HPR 2017/sharpshooter_d198_labeled.csv"
# data = pd.read_csv(file_path)
# print(f"Successfully loaded data from: {file_path}")

# pre_rect = data["pre_rect"].values # Use 'pre_rect' column
# ground_truth_labels = data["labels"].astype(str).str.lower()
# true_is_probe = ~ground_truth_labels.isin(["n", "z"]) # n and z are both np

# --- Optuna Objective Function ---
def objective(trial, file_paths):
    """
    Objective function for Optuna to optimize the simple_probe_finder parameters.
    """
    # 1. Suggest parameters
    # window: Integer, typically related to sample rate. Range from 50 to 500.
    window = trial.suggest_int("window", 500, 1200)
    # threshold: Float, critical for distinguishing NP from P. Range from 0.01 to 0.1.
    threshold = trial.suggest_float("threshold", 0.005, 0.2)
    # min_probe_length: Integer, minimum length of a valid probe. Range from 500 to 2000.
    min_probe_length = trial.suggest_int("min_probe_length", 500, 1000)
    # np_pad: Integer, padding around probes. Range from 50 to 500.
    np_pad = trial.suggest_int("np_pad", 50, 500)

    accuracies = []

    for file_path in file_paths:
        try:
            # Load data for the current file
            data = pd.read_csv(file_path)
            # print(f"Processing file: {os.path.basename(file_path)}") # Optional: for debugging progress
        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}. Skipping this file.")
            continue # Skip to the next file if one is not found
        except Exception as e:
            print(f"Error loading {file_path}: {e}. Skipping this file.")
            continue

        # Prepare data for optimization for the current file
        pre_rect = data["pre_rect"].values
        ground_truth_labels = data["labels"].astype(str).str.lower()
        true_is_probe = ~ground_truth_labels.isin(["n", "z"]) # True for probe (P), False for NP ('n' or 'z')

        # Run simple_probe_finder with suggested parameters on current file's data
        probes = ProbeSplitter.simple_probe_finder(
            pre_rect,
            window=window,
            threshold=threshold,
            min_probe_length=min_probe_length,
            np_pad=np_pad
        )

        # Simulate the evaluation part
        predicted_is_probe = np.zeros_like(pre_rect, dtype=bool)
        for start, end in probes:
            start = max(0, start)
            end = min(len(predicted_is_probe), end)
            predicted_is_probe[start:end] = True

        # Calculate accuracy score for the current file
        accuracies.append(accuracy_score(true_is_probe, predicted_is_probe))

    # If no files were processed successfully, return a very low value so Optuna avoids these parameters
    if not accuracies:
        return 0.0 # Or -np.inf if you prefer

    # Return the average accuracy across all processed files
    return np.mean(accuracies)

# --- Run Optuna Optimization ---
if __name__ == "__main__":
    print("Starting Optuna optimization across multiple files...")
    # Create an Optuna study. We want to maximize the average accuracy.
    study = optuna.create_study(direction="maximize")
    # Use a lambda function to pass file_paths to the objective
    study.optimize(lambda trial: objective(trial, sharpshooter_files), n_trials=50) # Adjust n_trials as needed

    print("\nOptimization finished.")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Average Accuracy): {trial.value:.4f}")
    print("  Best Parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Optional: Evaluate with the best parameters and print classification report
    # This part will now evaluate on the *first* file in your list for demonstration
    # For a full multi-file report, you'd need to loop through all files again.
    print("\n--- Evaluation with Best Parameters on the FIRST file (for detailed report) ---")
    try:
        first_file_data = pd.read_csv(sharpshooter_files[0])
        pre_rect_first_file = first_file_data["pre_rect"].values
        ground_truth_labels_first_file = first_file_data["labels"].astype(str).str.lower()
        true_is_probe_first_file = ~ground_truth_labels_first_file.isin(["n", "z"])

        best_probes = ProbeSplitter.simple_probe_finder(
            pre_rect_first_file,
            window=trial.params["window"],
            threshold=trial.params["threshold"],
            min_probe_length=trial.params["min_probe_length"],
            np_pad=trial.params["np_pad"]
        )

        best_predicted_is_probe = np.zeros_like(pre_rect_first_file, dtype=bool)
        for start, end in best_probes:
            start = max(0, start)
            end = min(len(best_predicted_is_probe), end)
            best_predicted_is_probe[start:end] = True

        print(f"Accuracy on first file with best parameters: {accuracy_score(true_is_probe_first_file, best_predicted_is_probe):.4f}")
        print(classification_report(true_is_probe_first_file, best_predicted_is_probe, target_names=["NP", "P"]))
    except IndexError:
        print("No files provided in 'sharpshooter_files' list for detailed evaluation.")
    except Exception as e:
        print(f"Error during detailed evaluation on first file: {e}")