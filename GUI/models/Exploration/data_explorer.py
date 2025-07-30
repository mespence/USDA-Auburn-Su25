import pandas as pd
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
from data_loader import import_data

DATA_FOLDER = r"C:\Users\Clinic\Desktop\USDA-Auburn-Su25\Data\Sharpshooter Data - HPR 2017\sharpshooter_parquet"
EXCLUDE = {
        "a01", "a02", "a03", "a10", "a15",
        "b01", "b02", "b04", "b07", "b12", "b188", "b202", "b206", "b208",
        "c046", "c07", "c09", "c10",
        "d01", "d03", "d056", "d058", "d12",
    }

dfs = import_data(data_path=DATA_FOLDER, filetype="parquet", exclude=EXCLUDE)

def get_min_segment_lengths_across_dfs(dfs, label_col="labels"):
    segment_lengths = {}

    for df in dfs:
        labels = df[label_col].values
        current_label = labels[0]
        current_length = 1
        for i in range(1, len(labels)):
            if labels[i] == current_label:
                current_length += 1
            else:
                segment_lengths.setdefault(current_label, []).append(current_length)
                current_label = labels[i]
                current_length = 1
        segment_lengths.setdefault(current_label, []).append(current_length)

    min_lengths = {label: min(lengths) for label, lengths in segment_lengths.items()}
    return min_lengths, segment_lengths

def get_label_segment_stats(dfs, label_col="labels"):
    lengths = {}
    for df in dfs:
        labels = df[label_col].values
        for label, group in groupby(labels):
            seg_len = len(list(group))
            lengths.setdefault(label, []).append(seg_len)
    stats = {}
    for label, seg_lens in lengths.items():
        stats[label] = {
            "mean": float(np.mean(seg_lens)),
            "median": float(np.median(seg_lens)),
            "count": len(seg_lens)
        }
    return stats


def plot_label_segment_histograms(dfs, label_col="labels", bins=50):
    lengths = {}
    for df in dfs:
        labels = df[label_col].values
        for label, group in groupby(labels):
            seg_len = len(list(group))
            lengths.setdefault(label, []).append(seg_len)
    # Plot
    for label, seg_lens in lengths.items():
        plt.figure()
        plt.hist(seg_lens, bins=bins, alpha=0.7)
        plt.title(f"Segment Lengths for Label '{label}'")
        plt.xlabel("Segment Length")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()

min_lengths, all_lengths = get_min_segment_lengths_across_dfs(dfs)
print(min_lengths)

stats = get_label_segment_stats(dfs)
print(stats)

plot_label_segment_histograms(dfs)

# -- OUTPUT --

# -- Min Segment Lengths --
# { 'Z': 59, 
#   'P': 31, 
#   'B2': 90, 
#   'C': 9, 
#   'D': 37, 
#   'F2': 24, 
#   'B': 9, 
#   'G': 70, 
#   'DG': 98, 
#   'CG': 111, 
#   'F3': 46, 
#   'F4': 71, 
#   'F1': 4}

# -- Label Segment Stats --
# {'Z': {'mean': 385739.58585858584, 'median': 26051.0, 'count': 396}, 
#  'P': {'mean': 3322.868131868132, 'median': 2170.0, 'count': 728}, 
#  'B2': {'mean': 385.68036529680364, 'median': 375.0, 'count': 219}, 
#  'C': {'mean': 524.8724226804123, 'median': 223.0, 'count': 1552}, 
#  'D': {'mean': 11019.389766741911, 'median': 3040.0, 'count': 1329}, 
#  'F2': {'mean': 307.30151338766007, 'median': 234.0, 'count': 859}, 
#  'B': {'mean': 528.6805359661495, 'median': 313.5, 'count': 1418}, 
#  'G': {'mean': 66748.33037694014, 'median': 11753.0, 'count': 451}, 
#  'DG': {'mean': 15338.464454976303, 'median': 6758.5, 'count': 422}, 
#  'CG': {'mean': 4123.158620689655, 'median': 684.0, 'count': 145}, 
#  'F3': {'mean': 666.4285714285714, 'median': 638.0, 'count': 49}, 
#  'F4': {'mean': 322.2, 'median': 289.0, 'count': 100}, 
#  'F1': {'mean': 260.0, 'median': 136.5, 'count': 86}}

