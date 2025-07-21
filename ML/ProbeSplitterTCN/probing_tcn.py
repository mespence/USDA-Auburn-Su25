

import os
import json
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import stft
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader import load_dfs_from_dir


NON_PROBING_LABELS = ["N", "Z"]

def load_label_map(save_path: str) -> tuple[dict, dict]:
    """
    Load a label map from disk.

    Parameters:
        save_path (str): Path to load label_map.json

    Returns:
        label_map (Dict[str, int]): the label map
        inv_label_map (Dict[int, str]): the inverse label map
    """
    if not os.path.exists(save_path):
        return None
    
    with open(save_path, "r") as f:
        label_map = json.load(f)

    label_map = {k: int(v) for k, v in label_map.items()}

    inv_label_map = defaultdict(list)
    for label, value in label_map.items():
        inv_label_map[value].append(label)
    inv_label_map = dict(inv_label_map)

    return label_map, inv_label_map

def build_probing_label_map(dataframes: list[pd.DataFrame], label_column = "labels") -> tuple[dict, dict]:
    """
    Build probing label map from a list of DataFrames and writes to `label_map.json`.
    "N" and "Z" are non-probing (0), all other labels are probing (1).    

    Parameters:
        dataframes: List of pandas DataFrames with a column containing labels.
        label_column (str): Column containing label strings (e.g. "Z", "D", "F1")

    Returns:
        label_map (Dict[str, int]): mapping of each original label to 0 or 1
        inv_label_map (Dict[int, list[str]]): reverse mapping from 0/1 to original labels
    """
    all_labels = set()
    for df in dataframes:
        labels = df[label_column].dropna().unique()
        all_labels.update(label.upper() for label in labels)

    label_map = {
        label: 0 if label in NON_PROBING_LABELS else 1
        for label in sorted(all_labels)
    }
    
    inv_label_map = {0: [], 1: []}
    for label, val in label_map.items():
        inv_label_map[val].append(label)

    output_path = os.path.join(os.path.dirname(__file__), 'label_map.json')
    with open(output_path, "w") as f:
        json.dump(label_map, f, indent=4)

    return label_map, inv_label_map


class ProbeSplitterTCN:
    def __init__(self, epochs = 4, trial = None):
        random.seed(42)

        self.SAMPLING_RATE = 100
        map_path = os.path.join(os.path.dirname(__file__), "label_map.json") 
        label_maps = load_label_map(map_path) # assumes .json in same directory as this file
        if label_maps is not None:
            self.label_map = label_maps[0]
        else:
            raise ValueError("label_map.json not found. Run build_probing_label_map() first.")
        self.inv_label_map = {0: "NP", 1: "P"}


        self.num_inputs = 52 # Number of input channels (features per time step): 1 voltage, 51 STFT freq bins ?
        self.num_classes = 2 # Number of classes for segmentation: 0 (non-probing) or 1 (probing)
        self.num_channels = [64, 96, 144, 216, 324] # Number of channels in each residual block
        self.kernel_size = 5         # Kernel size for convolutions
        self.dropout = 0.2           # Dropout rate
        self.dilation_base = 2       # Base for exponential dilation
        self.epochs = epochs

        self.batch_size = 64  # JH: tweaks this for GPU based on RAM limits
        self.ticks_before = int(self.SAMPLING_RATE*3)
        self.ticks_during = int(self.SAMPLING_RATE*0.5)
        self.ticks_after = int(self.SAMPLING_RATE*3)
        self.skip_num = int(self.SAMPLING_RATE*0.5)

        if trial:
            self.num_channels = self.num_channels[:trial.suggest_int("layers", 1, 5)]
            self.epochs = trial.suggest_categorical("epochs", [2, 4, 8])
            self.dropout = trial.suggest_categorical("dropout", [0.0001, 0.001, 0.01, 0.1])
        else:
            self.epochs = 8
            self.dropout = 0.001


        self.num_channels = [8, 16]
        self.epochs = 1
        self.batch_size = 4
        self.ticks_before = 20
        self.ticks_during = 10
        self.ticks_after = 20
        self.skip_num = 30
        

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = TCN(self.num_inputs, self.num_channels, 
                         self.num_classes, self.kernel_size, 
                         self.dropout, self.dilation_base)
        self.model.to(self.device)

        dirname = os.path.dirname(__file__)
        self.save_path = os.path.join(dirname, 'tcn_weights')

    def train(self, probes, test_probes = None, fold = None):
        tr_dfs, tr_df = self.load_probes(probes)
        tr_dataset = TimeSeriesDataset(tr_df, self.label_map, self.ticks_before, 
                                       self.ticks_during, self.ticks_after, 
                                       self.skip_num)
        tr_dataloader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True)

        """
        test_dfs, test_df = self.load_probes(test_probe)
        train_dataset = TimeSeriesDataset(test_dfs, self.label_map, self.ticks_before, 
                                       self.ticks_during, self.ticks_after, 
                                       self.skip_num)
        tr_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        """

        criterion = nn.BCEWithLogitsLoss()
        #criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        train_losses = []
        test_losses = []
        print(f"Starting training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            tot_loss = 0
            self.model.train()
            for batch_X, batch_y in tqdm(tr_dataloader):
                batch_X = batch_X.permute(0, 2, 1).to(self.device)
                batch_y = batch_y.float().to(self.device)
                #batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                #loss = criterion(outputs, batch_y.reshape(-1))
                
                optimizer.zero_grad()
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()
            avg_loss = tot_loss / len(tr_dataloader)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")
            train_losses.append(avg_loss)
            """
            # Get the test loss
            with torch.no_grad():
                running_loss = 0
                self.model.eval()
                for batch in test_dataloader:
                    x, y, weights = batch
                    x, y, weights = x.to(self.device), y.to(self.device), \
                                    weights.to(self.device)
                    outputs = self.model(x.permute(0,2,1))
                    loss = criterion(outputs, y)
                    
                    running_loss += loss.item()
                test_loss = running_loss / len(test_dataloader)
                test_losses.append(test_loss)
        plt.plot(train_losses, label = "Train")
        plt.plot(test_losses, label = "Test")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(f"{self.save_path}/loss_curve_fold{fold}.png")
            """


    def train_streamed(self, probe_list, num_epochs=4):
        print(f"Training on {len(probe_list)} probes.")
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for epoch in range(self.epochs):
            print(f"\n=== Epoch {epoch+1}/{self.epochs} ===")
            random.shuffle(probe_list)  # optional: reshuffle probes per epoch
            for i, probe_df in enumerate(probe_list):
                print(f"\n[Probe {i+1}/{len(probe_list)}] Training on probe...")

                _, df = self.load_probes([probe_df])
                dataset = TimeSeriesDataset(df, self.label_map,
                                            self.ticks_before, self.ticks_during,
                                            self.ticks_after, self.skip_num)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

                self.model.train()
                tot_loss = 0

                for batch_X, batch_y in tqdm(dataloader, desc="Training batches", leave=False):
                    batch_X = batch_X.permute(0, 2, 1).to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    tot_loss += loss.item()

                avg_loss = tot_loss / len(dataloader)
                print(f"Avg loss for probe: {avg_loss:.4f}")

    def predict(self, probes):
        all_predictions = []
        for probe in tqdm(probes):
            test_dfs, test_df = self.load_probes([probe])
            test_dataset = TimeSeriesDataset(test_df, self.label_map, self.ticks_before, 
                                             self.ticks_during, self.ticks_after, 
                                             self.skip_num)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            all_preds = []
            with torch.no_grad():
                for batch_X, _ in test_dataloader:
                    batch_X = batch_X.permute(0, 2, 1).to(self.device)
                    outputs = self.model(batch_X)
                    probs = torch.sigmoid(outputs)
                    binary_preds = (probs > 0.5).int().squeeze(1)
                    output_list = [self.inv_label_map[x] for x in binary_preds.cpu().tolist()]
                    all_preds.extend(output_list)

                    # output_list = outputs.argmax(dim=1).reshape(-1).cpu().tolist()
                    # output_labels = [self.inv_label_map[x] for x in output_list]
                    # all_preds.extend(output_labels)
            # Expand the data to be the size of the original input
            all_preds = np.repeat(all_preds, self.skip_num)
            all_preds = np.pad(all_preds, (0, len(probe) - len(all_preds)), 'edge')
            all_predictions.append(all_preds)
        return all_predictions

    def apply_stft_to_df(self, df, target_column):
        # Perform STFT on the target column data
        signal = df[target_column].values
        fs = self.SAMPLING_RATE

        f, t, Zxx = stft(signal, fs=fs, nperseg=fs, noverlap=fs - 1)

        
        # Create a DataFrame for the STFT magnitudes with frequency columns
        mag = np.abs(Zxx).astype(np.float32).T  # Shape: (num_windows, num_freqs)
        freq_cols = [f"Frequency_{freq}" for freq in f]
        magnitude_df = pd.DataFrame(mag, columns=freq_cols)
        trimmed_df = df.iloc[:len(magnitude_df)].copy()

        # Concatenate the original DataFrame with the STFT DataFrame
        transformed_df = pd.concat([trimmed_df.reset_index(drop=True), magnitude_df], axis=1)
        return transformed_df
    
    def load_probes(self, probes):
        import time
        dfs = []
        print("Loading probes...")
        for i, df in enumerate(probes):
            c1 = time.perf_counter()
            df["labels"] = df["labels"].str.upper()
            df['previous_label'] = df['labels'].shift(1)
            df_transformed = self.apply_stft_to_df(df, "voltage")
            dfs.append(df_transformed)
            c2 = time.perf_counter()
            print(f"Iter {i+1} ({c2 - c1 : .4f} s): {(i+1) / len(probes) * 100:.2f}%")
            
        big_df = pd.concat(dfs, axis=0, ignore_index=True)
        print("Probes loaded.")
        return dfs, big_df

    def save(self):
        torch.save(self.model.state_dict(), self.save_path)

    def load(self, path = None):
        self.model = TCN(self.num_inputs, self.num_channels, 
                         self.num_classes, self.kernel_size, 
                         self.dropout, self.dilation_base)
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location = self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

class TimeSeriesDataset(Dataset):
    def __init__(self, df, label_map, ticks_before, ticks_during, ticks_after, skip_num):
        freq_cols = [c for c in df.columns if "Frequency_" in c]
        self.X = df[["voltage"]+freq_cols].values
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
        y_values = np.nanmedian(self.y[y_start:y_end])
    

        # Convert to torch tensors
        X_values = torch.tensor(X_values, dtype = torch.float32)
        y_values = torch.tensor([y_values], dtype = torch.float32)

        return X_values, y_values



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, is_last=False):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2  # For 'same' padding

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.conv1 = nn.utils.parametrizations.weight_norm(self.conv1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.utils.parametrizations.weight_norm(self.conv2)
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
        self.fc = nn.Linear(num_channels[-1], 1)  # Fully connected layer for classification, 1 class since binary TCN

    def forward(self, x):
        # Input shape: (batch_size, num_inputs, seq_length)
        y = self.tcn(x)
        
        # Global average pooling across the sequence length
        y = self.global_avg_pool(y)  # Shape: (batch_size, num_channels[-1], 1)
        y = y.squeeze(-1)  # Remove the last dimension, shape: (batch_size, num_channels[-1])
        
        # Pass through fully connected layer for classification
        y = self.fc(y)  # Shape: (batch_size, num_classes)
        
        return y


if __name__ == "__main__":
    tcn = ProbeSplitterTCN()

    DATA_DIR = r"..\data"

    excluded = {
        "a01", "a02", "a03", "a10", "a15",
        "b01", "b02", "b04", "b07", "b12", "b188", "b202", "b206", "b208",
        "c046", "c07", "c09", "c10",
        "d01", "d03", "d056", "d058", "d12",
    }

    data_dfs = load_dfs_from_dir(DATA_DIR, excluded=excluded)

    tcn.train_streamed(data_dfs)
    tcn.save()
    tcn.load(r"D:\USDA-Auburn\CS-Repository\ML\ProbeSplitterTCN\tcn_weights")

    output = tcn.predict([data_dfs[1]])

    data_dfs[1]['pred'] = output[0]
    data_dfs[1].to_csv("out.csv")
    print(output)


    


    #build_probing_label_map(data_dfs)


