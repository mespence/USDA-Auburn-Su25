# %%
from sklearn.metrics import precision_score, recall_score, f1_score
label_map = 

all_labels, all_preds = r
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

# %%
label_to_color = {
    "NP": "red",
    "J": "blue",
    "K": "green",
    "L": "purple",
    "M": "pink",
    "N": "cyan",
    "Z": "orange"
}

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
    fig, axs = plt.subplots(4 if len(probs) > 0 else 3, 1, sharex = True)
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
        axs[1].fill_between(time[:len(pred_labels)], fill_min, fill_max, 
                where = (pred_labels == label), color=color, alpha = 0.5)
    axs[1].set_title("Predicted Labels")
    # Third plot will be marked where there is a difference between the two
    axs[2].plot(time, voltage, color = "black")
    axs[2].fill_between(time[:len(pred_labels)], fill_min, fill_max, 
            where = (pred_labels != true_labels[:len(pred_labels)]), color = "gray", alpha = 0.5)
    axs[2].set_title("Incorrect Labels")
    if len(probs) > 0:
        # Fourth plot with confidences
        axs[3].plot(time[:len(probs)], probs)
        axs[3].set_title("Confidence")
        axs[3].set_ylim([0, 1])
        # Axes titles and such
    fig.supxlabel("Time (s)")
    fig.supylabel("Volts")
    fig.tight_layout()
    
for i in range(12):
    recording = i
    df = dataframes[recording]
    transformed_df = test_data[recording]
    pred_labels = model.predict(transformed_df.drop(["label"], axis=1)).repeat(sample_rate * chunk_seconds)
    probs = model.predict_proba(transformed_df.drop(["label"], axis=1)).max(axis=1).repeat(sample_rate * chunk_seconds)
    plot_labels(df["time"], df["pre_rect"], df["labels"], pred_labels, probs)