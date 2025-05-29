# %%
import pandas as pd
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy
import random
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.signal import stft

random.seed(42)



SAMPLING_RATE = 100
ROOT = "c:\\Users\\Milo Knell\\tarsalis_data_clean"

def clip_extra_nps(df, keep=100):
    cs = (df['labels'] == "NP").cumsum() - 1

    num_starting_np = (cs == cs.index).idxmin()

    cs_inv = (df[::-1]['labels'] == "NP").cumsum() - 1
    num_ending_np = len(df) - (cs_inv == cs_inv.index[::-1]).idxmin()

    assert num_starting_np >= keep and num_ending_np >= keep
    new_df = df[num_starting_np-keep:len(df)-num_ending_np+keep]
    assert len(df) - len(new_df) == num_starting_np + num_ending_np - 2*keep
    return df[num_starting_np-keep:len(df)-num_ending_np+keep]

def apply_stft_to_df(df, target_column, fs=1.0, nperseg=SAMPLING_RATE):
    # Perform STFT on the target column data
    f, t, Zxx = stft(df[target_column].values, fs=fs, nperseg=nperseg, noverlap=nperseg - 1)
    
    # Create a DataFrame for the STFT magnitudes with frequency columns
    magnitude_df = pd.DataFrame(np.abs(Zxx).T, columns=[f"Frequency_{freq:.2f}" for freq in f])
    magnitude_df = magnitude_df.drop(magnitude_df.index[-1])
    # Concatenate the original DataFrame with the STFT DataFrame
    transformed_df = pd.concat([df.reset_index(drop=True), magnitude_df], axis=1)
    return transformed_df


def mode(s):
    mode = scipy.stats.mode(s).mode
    return mode

def load_files(files, root=ROOT):
    dfs = []
    if files is None:
        files = os.listdir(root)
    
    for file in files:
        df = pd.read_csv(os.path.join(root,file)).drop(columns=["Unnamed: 0"])
        df["labels"] = df['labels'].str.upper().replace("Z","W")
        df['previous_label'] = df['labels'].shift(1)
        df["file"] = file
        df_clipped = clip_extra_nps(df).dropna().reset_index()
        df_transformed = apply_stft_to_df(df_clipped, "post_rect")
        dfs.append(df_transformed)

    big_df = pd.concat(dfs, axis=0)
    return dfs, big_df

class TimeSeriesDataset(Dataset):
    def __init__(self, df, label_map, ticks_before, ticks_during, ticks_after, skip_num):
        freq_cols = [c for c in df.columns if "Frequency_" in c]
        self.X = df[["pre_rect"]+freq_cols].values
        self.y = df["labels"].map(label_map).values

        self.ticks_before = ticks_before
        self.ticks_during = ticks_during
        self.ticks_after = ticks_after
        self.skip_num = skip_num
        self.total_length = self.X.shape[0]

        # Start and end index calculation now includes ticks_during
        self.start_index = self.ticks_before
        self.end_index = self.total_length - self.ticks_after - self.ticks_during + 1
        self.step = self.skip_num + 1

        self.num_samples = max(0, (self.end_index - self.start_index + self.step - 1) // self.step)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Map the dataset index to the actual index in the data
        i = self.start_index + idx * self.step

        if i >= self.end_index:
            raise IndexError(f"Index {idx} out of range.")

        # Define the range for x values (before, during, and after the current index)
        start = i - self.ticks_before
        end = i + self.ticks_during + self.ticks_after  # Includes 'during' ticks

        # Slice the x values without copying the entire array
        X_values = self.X[start:end, :]

        # Define the range for y values (for the 'during' period)
        y_start = i
        y_end = i + self.ticks_during
        y_values = np.median(self.y[y_start:y_end])
    

        # Convert to torch tensors
        X_values = torch.tensor(X_values, dtype=torch.float32)
        y_values = torch.tensor([y_values], dtype=torch.int64)

        return X_values, y_values

batch_size = 64
ticks_before = int(SAMPLING_RATE*3)
ticks_during = int(SAMPLING_RATE*0.5)
ticks_after = int(SAMPLING_RATE*3)
skip_num = int(SAMPLING_RATE*0.5)

files = os.listdir(ROOT)
tr_size = int(len(files)*0.8)
test_size = len(files) - tr_size

tr_files = set(random.sample(files, tr_size))
test_files = set(files)-set(tr_files)

tr_dfs, tr_df = load_files(tr_files)
test_dfs, test_df = load_files(test_files)

labels = set(tr_df["labels"].unique()) | set(test_df["labels"].unique())
label_map = {label:i for label,i in zip(labels, range(len(labels)))}

tr_dataset = TimeSeriesDataset(tr_df, label_map, ticks_before, ticks_during, ticks_after, skip_num)
tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TimeSeriesDataset(test_df, label_map, ticks_before, ticks_during, ticks_after, skip_num)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# %%

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, is_last=False):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2  # For 'same' padding

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.utils.weight_norm(self.conv2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Adjust dimensions if the input and output channels are different
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None
        self.is_last = is_last
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        if not self.is_last:
            out = self.relu2(out)
        out = self.dropout2(out)

        # Apply skip connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dilation_base=2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            is_last = (i==num_levels-1)
            dilation = dilation_base ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [ResidualBlock(in_channels, out_channels, kernel_size, dilation, dropout, is_last)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=2, dropout=0.2, dilation_base=2):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout, dilation_base)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling over the sequence length
        self.fc = nn.Linear(num_channels[-1], num_classes)  # Fully connected layer for classification

    def forward(self, x):
        # Input shape: (batch_size, num_inputs, seq_length)
        y = self.tcn(x)
        
        # Global average pooling across the sequence length
        y = self.global_avg_pool(y)  # Shape: (batch_size, num_channels[-1], 1)
        y = y.squeeze(-1)  # Remove the last dimension, shape: (batch_size, num_channels[-1])
        
        # Pass through fully connected layer for classification
        y = self.fc(y)  # Shape: (batch_size, num_classes)
        
        return y


num_inputs = 52          # Number of input channels (e.g., features per time step)
num_classes = len(label_map)        # Number of classes for segmentation
num_channels = [64, 96, 144, 216, 324] #[64, 128, 192, 256, 320, 384, 448, 512]  # Number of channels in each residual block
kernel_size = 5         # Kernel size for convolutions
dropout = 0.2           # Dropout rate
dilation_base = 2       # Base for exponential dilation
epochs = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Instantiate the model
model = TCN(num_inputs, num_channels, num_classes, kernel_size, dropout, dilation_base)
model.to(device)

def predict(model, test_dataloader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            batch_X = batch_X.permute(0, 2, 1).to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)

            all_preds.extend(outputs.argmax(dim=1).reshape(-1).cpu().tolist())
            all_labels.extend(batch_y.reshape(-1).cpu().tolist())

    return all_labels, all_preds

def evaluate(model, test_dataloader):
    all_labels, all_preds = predict(model, test_dataloader)
    return accuracy_score(all_labels, all_preds)

def generate_confusion(model, test_dataloader):
    all_labels, all_preds = predict(model, test_dataloader)

    class_names = {v: k for k, v in label_map.items()}
    cm = confusion_matrix(all_labels, all_preds)
    labels = [class_names[i] for i in range(len(label_map))]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix with Class Names')
    plt.show()
    return all_labels, all_preds


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
accs = []


for epoch in range(epochs):
    print(epoch)
    tot_loss = 0
    for batch_X, batch_y in tr_dataloader:
        batch_X = batch_X.permute(0, 2, 1).to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_X)
        loss = criterion(outputs.reshape(-1, 7), batch_y.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
    

    acc = evaluate(model, test_dataloader)
    accs.append(acc)
    losses.append(tot_loss)
    print(losses[-1], acc)

r = generate_confusion(model, test_dataloader)

# %%
from sklearn.metrics import precision_score, recall_score, f1_score
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
    pred_labels = model.predict(transformed_df.drop(["label"], axis=1)).repeat(SAMPLING_RATE * chunk_seconds)
    probs = model.predict_proba(transformed_df.drop(["label"], axis=1)).max(axis=1).repeat(SAMPLING_RATE * chunk_seconds)
    plot_labels(df["time"], df["pre_rect"], df["labels"], pred_labels, probs)