# %%
import pandas as pd
import numpy as np
import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.nn.utils.rnn import pad_sequence
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import subprocess

from utils import make_plots, evaluate, generate_confusion, detailed_f1_stats, split_into_probes_simple, split_into_probes_data_leak, load_files
from data_augmentation import DataAugmentor

random.seed(42)

ROOT = "c:\\Users\\milok\\clasp_transitions"

label_to_color = {
    "NP": "red",
    "J": "blue",
    "K": "green",
    "L": "purple",
    "M": "pink",
    "N": "cyan",
    "W": "orange"
}

class TimeSeriesDataset(Dataset):
    def __init__(self, dfs, label_map, data_columns, class_column, probe_splitter, transform=None, weight=False, split_by_probes=True):
        """
        dfs: List of DataFrames, each containing 'x' (time series) and 'labels' columns.
        label_map: Dictionary mapping each letter to a number for prediction (should be a function that can be applied to a tensor).
        transform: Optional transform to be applied on the data.
        weight: Optional bool whether to weight by class or not.
        split_by_probes: Optional bool whether to split by probes or not.
        """
        self.data_columns = data_columns
        self.class_column = class_column
        self.transform = transform

        # Process each dataframe in the list of dfs
        self.x = []
        self.y = []
        self.weights = []

        # process splitting by probes and provide names for each file
        if split_by_probes:
            dfs, self.names = probe_splitter(dfs)
        else:
            self.names = [df["file"] for df in dfs]

        for df in dfs:
            # Extract time series data and labels for each df
            x_tensor = torch.tensor(df[self.data_columns].values, dtype=torch.float32)
            y_tensor = torch.tensor(df[self.class_column].map(label_map).values, dtype=torch.long)
            self.x.append(x_tensor)
            self.y.append(y_tensor)

            # Handle weights if requested
            if weight:
                weights_tensor = torch.tensor(self.calculate_weights(df).tolist(), dtype=torch.float32)
            else:
                weights_tensor = torch.ones(x_tensor.shape, dtype=torch.float32)

            self.weights.append(weights_tensor)

        # Pad sequences to the length of the longest time series in all datasets
        self.x = self.x
        self.y = self.y
        self.weights = self.weights

    def calculate_weights(self, df):
        '''
        calculate weights based on class makeup
        '''
        class_counts = df[self.class_column].value_counts().to_dict()
        total_samples = len(df)
        weights_by_class = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
        return df[self.class_column].map(weights_by_class)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        weight = self.weights[idx]
        if self.transform:
            x = self.transform(x)
        return x, y, weight

files = os.listdir(ROOT)
tr_size = int(len(files)*0.8)
test_size = len(files) - tr_size

tr_files = set(random.sample(files, tr_size))
test_files = list(set(files)-set(tr_files))

big_tr_df, tr_dfs = load_files(tr_files, root=ROOT)
big_test_df, test_dfs = load_files(test_files, root=ROOT)

labels = set(big_tr_df["labels"].unique()) | set(big_test_df["labels"].unique())
label_map = {label:i for label,i in zip(labels, range(len(labels)))}

batch_size = 1


data_columns = ["pre_rect", "profile"] #["pre_rect"]
probe_splitter=split_into_probes_data_leak #split_into_probes_simple

tr_dataset = TimeSeriesDataset(tr_dfs, label_map, data_columns=data_columns, class_column = "labels", probe_splitter=probe_splitter)
tr_dataloader = tr_dataset #DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TimeSeriesDataset(test_dfs, label_map, data_columns=data_columns, class_column = "labels", probe_splitter=probe_splitter)
test_dataloader = test_dataset #DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# %%
def crop_after_upsampling(upsampled, downsampled, pad_left, pad_right):
    # Calculate the size difference and crop accordingly
    upsampled_size = upsampled.size(-1)
    downsampled_size = downsampled.size(-1)
    
    # Crop the tensor to match the size of the downsampled feature map
    if upsampled_size > downsampled_size:
        crop_left = pad_left
        crop_right = upsampled_size - downsampled_size - pad_right
        upsampled = upsampled[:, :, crop_left:upsampled_size - crop_right]
    
    return upsampled


def add_padding_for_downsampling(x, kernel_size, stride):
    # Calculate the necessary padding to keep the size even after downsampling
    pad_total = (stride - (x.size(-1) % stride)) % stride
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    # Apply padding
    x_padded = F.pad(x, (pad_left, pad_right))
    return x_padded, (pad_left, pad_right)  # Return padding values to track for cropping


def pad_or_crop(tensor, dim, target_size):
    # Calculate the difference in size
    current_size = tensor.size(dim)
    diff = target_size - current_size

    if diff > 0:
        # Padding is needed: apply symmetric padding on both sides along the target dimension
        pad = [0] * (2 * tensor.dim())
        pad[-(2 * dim + 1)] = diff // 2       # pad before
        pad[-(2 * dim + 2)] = diff - diff // 2 # pad after
        tensor = F.pad(tensor, pad)
        
    elif diff < 0:
        # Cropping is needed: calculate the crop indices for symmetric cropping
        crop_start = (-diff) // 2
        crop_end = crop_start + target_size
        tensor = tensor.narrow(dim, crop_start, target_size)
        
    # If diff == 0, no operation is needed; the size is already correct
    return tensor



class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3, init_features=8):
        super(UNet1D, self).__init__()
        self.num_layers = num_layers
        features = init_features
        
        # Encoding layers
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(num_layers):
            encoder = EncoderBlock(in_channels if i == 0 else features, 2*features)
            self.encoders.append(encoder)
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            features *= 2  # Increase feature size

        # Bottleneck
        self.bottleneck = EncoderBlock(features, features*2)
        features *= 2

        # Decoding layers
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            upconv = nn.ConvTranspose1d(features, features // 2, kernel_size=2, stride=2)
            self.upconvs.append(upconv)
            decoder = DecoderBlock(features, features // 2)
            self.decoders.append(decoder)
            features //= 2  # Decrease feature size

        # Final output layer
        self.conv = nn.Conv1d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        encodings = []
        paddings = []
        for i in range(self.num_layers):
            #x, pad_l_r = add_padding_for_downsampling(x, kernel_size=2, stride=2)
            #paddings.append(pad_l_r)
            x = self.encoders[i](x)
            encodings.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoding path
        for i in range(self.num_layers):
            x = self.upconvs[i](x)


            #pad_left, pad_right = paddings[-(i+1)]
            prev_encoding = encodings[-(i+1)]
            #x = crop_after_upsampling(x, prev_encoding, pad_left, pad_right)
            x = pad_or_crop(x, dim=2, target_size=prev_encoding.shape[2])

            
            x = torch.cat((x, prev_encoding), dim=1)  # Skip connection
            x = self.decoders[i](x)

        return self.conv(x)
    
    
    def predict_batch(self, batch, device):
        x, y, weights = batch
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        outputs = self.forward(x.permute(0,2,1))

        masked_preds = outputs.argmax(dim=1).view(-1).cpu().tolist()
        masked_labels = y.cpu().view(-1).tolist()
        return masked_preds, masked_labels
    
    def predict(self, test_dataloader, device):
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_dataloader:
                masked_preds, masked_labels = model.predict_batch(batch, device=device)

                all_preds.extend(masked_preds)
                all_labels.extend(masked_labels)

        return all_labels, all_preds

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Defines a single encoding block consisting of 2x: Conv1d, BatchNorm1d, and ReLU.
        """
        super(EncoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

    def __repr__(self):
        return f"EncoderBlock({self.in_channels}, {self.out_channels})" + super(EncoderBlock, self).__repr__()[12:]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Defines a single decoding block consisting of 2X: Conv1d, BatchNorm1d, and ReLU.
        """
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
class MaxPool1dWithOddInputHandling(nn.Module):
    def __init__(self, kernel_size=2, stride=None, padding=0, dilation=1, ceil_mode=False):
        super(MaxPool1dWithOddInputHandling, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)

    def forward(self, x):
        # Check if the sequence length (dimension 2) is odd
        if x.size(2) % 2 != 0:
            # If odd, pad by 1 on the right side
            x = F.pad(x, (0, 1))
        # Apply the MaxPool1d operation
        return self.pool(x)

epochs=50
lr=5e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(label_map)
if probe_splitter == split_into_probes_data_leak:
    num_classes -= 1
    

model = UNet1D(in_channels=len(tr_dataset.data_columns), out_channels=num_classes) 
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, capturable=False)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in tr_dataloader:
        x, y, weights = batch
        x, y, weights = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device), weights.unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        outputs = model(x.permute(0,2,1))
        
        loss = criterion(outputs, y)

        weighted_loss = (loss) * weights
        weighted_loss = weighted_loss.mean()
        weighted_loss.backward()
        optimizer.step()
        
        running_loss += weighted_loss.item()
    train_loss = running_loss / len(tr_dataloader)

    acc, f1 = evaluate(model, test_dataloader, device=device)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}')

all_labels, all_preds = model.predict(test_dataloader, device)
generate_confusion(all_labels, all_preds, label_map)
detailed_f1_stats(all_labels, all_preds, label_map)

save_dir = r"c:\Users\milok\USDA-Auburn\Exploration\Milo\unet_probe_split_plots"
PLOT = False
if PLOT:
    make_plots(test_dataset=test_dataset, model=model, label_map=label_map, label_to_color=label_to_color, save_dir=save_dir, device=device, save=True)