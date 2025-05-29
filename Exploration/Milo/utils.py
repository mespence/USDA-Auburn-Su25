import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import pandas as pd
from itertools import groupby

def plot_labels(time, voltage, true_labels, pred_labels, label_to_color):
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
    axs[2].fill_between(time[:len(pred_labels)], fill_min, fill_max, 
            where = (pred_labels != true_labels[:len(pred_labels)]), color = "gray", alpha = 0.5)
    axs[2].set_title("Incorrect Labels")
        # Axes titles and such
    fig.supxlabel("Time (s)")
    fig.supylabel("Volts")
    fig.tight_layout()
    return fig

def make_plots(test_dataset, model, label_map, save_dir, device, label_to_color, save=True):
    inv_label_map = {k:v for v,k in label_map.items()}
    if save:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(len(test_dataset)):
        batch = test_dataset[i]
        fname = test_dataset.names[i]
        test_voltage, test_true_label, _ = batch
        masked_preds, masked_labels = model.predict_batch(batch, device=device)
        assert (np.array(masked_labels) == test_true_label.numpy()).all()
        renamed_masked_labels = np.array([inv_label_map[p] for p in masked_preds])
        renamed_true_labels = np.array([inv_label_map[p] for p in test_true_label.tolist()])
        fig = plot_labels(np.arange(len(test_voltage)) / 100, test_voltage.numpy(), renamed_true_labels, renamed_masked_labels, label_to_color=label_to_color)
        
        save_location = os.path.join(save_dir, fname+".png")
        if save:
            fig.savefig(save_location, dpi=300, bbox_inches='tight')

    assert len(os.listdir(save_dir)) == len(test_dataset)

# subprocess.run(["powershell", "-Command", f"ii '{save_dir}'"], shell=True)


def evaluate(model, test_dataloader, device):
    all_labels, all_preds = model.predict(test_dataloader, device)
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')

def generate_confusion(all_labels, all_preds, label_map, sort=True):
    class_names = {v: k for k, v in label_map.items()}
    if sort:
        raw_labels = sorted([i for i in range(len(class_names))], key=lambda i: class_names[i])
    else:
        raw_labels = [i for i in range(len(label_map))]

    labels = [class_names[i] for i in raw_labels]
    
    cm = confusion_matrix(all_labels, all_preds, labels=raw_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix with Class Names')
    plt.show()
    return all_labels, all_preds


def detailed_f1_stats(all_labels, all_preds, label_map):
    # Calculate precision, recall, and F1 scores for each class
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average=None)

    # Calculate overall (macro) precision, recall, and F1 scores
    overall_precision = precision_score(all_labels, all_preds, average='macro')
    overall_recall = recall_score(all_labels, all_preds, average='macro')
    overall_f1 = f1_score(all_labels, all_preds, average='macro')

    # Display per-class results
    class_names = {v: k for k, v in label_map.items()}
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print(f"Class {class_names[i]}: Precision = {p:.2f}, Recall = {r:.2f}, F1 Score = {f:.2f}")

    # Display overall statistics
    print(f"\nOverall (Macro) Statistics:")
    print(f"Precision = {overall_precision:.2f}, Recall = {overall_recall:.2f}, F1 Score = {overall_f1:.2f}")


def load_files(files, root):
    dfs = []
    if files is None:
        files = os.listdir(root)
    
    for file in files:
        df = pd.read_csv(os.path.join(root,file)).drop(columns=["Unnamed: 0"])
        df["labels"] = df['labels'].str.upper().replace("Z","W")
        df['previous_label'] = df['labels'].shift(1)
        df["file"] = file
        dfs.append(df.dropna().reset_index())

    big_df = pd.concat(dfs, axis=0)
    return big_df, dfs

def probe_finder(recording, window = 500, threshold = 0.1,
                 min_probe_length = 1500, np_pad = 500):
    """
    Input: recording: A pre-rectified mosquito recording as an 1-D nupmy 
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

def split_into_probes_simple(dfs, window = 500, threshold = 0.1,
                 min_probe_length = 1500, np_pad = 500):
    
    all_probes = []
    all_probe_names = []

    for df in dfs:
        probes = probe_finder(df['pre_rect'].values, window=window, threshold=threshold, min_probe_length=min_probe_length, np_pad=np_pad)
        split_probes = [df.iloc[start:end].reset_index(drop=True).copy() for start, end in probes]
        split_probe_names = [df["file"][0][:-4]+"_"+str(i) for i,df in enumerate(split_probes)]

        all_probe_names.extend(split_probe_names)
        all_probes.extend(split_probes)

    return all_probes, all_probe_names

def get_start_end_indices(indices):
    groups = []
    for _, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
        group = [i[1] for i in g]
        groups.append((group[0], group[-1]))  # Store start and end indices
    return groups


def split_into_probes_data_leak(dfs):
    all_probes = []
    all_probe_names = []

    for df in dfs:
        non_np_indices = df.index[df['labels'] != 'NP'].tolist()
        probes = get_start_end_indices(non_np_indices)
        split_probes = [df.iloc[start:end].reset_index(drop=True).copy() for start, end in probes]
        split_probe_names = [df["file"][0][:-4]+"_"+str(i) for i,df in enumerate(split_probes)]

        all_probe_names.extend(split_probe_names)
        all_probes.extend(split_probes)

    return all_probes, all_probe_names