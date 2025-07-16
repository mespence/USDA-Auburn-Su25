import pandas as pd
import numpy as np
import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pad_sequence
import subprocess
import tqdm
from matplotlib import pyplot as plt
from positional_encodings.torch_encodings import PositionalEncoding1D

class Model():
    def __init__(self, epochs=64, lr=5e-4, num_layers=8, growth_factor=1, features=32, n_conv_steps_per_block=2, block_kernel_size=3, up_down_sample_kernel_size=2, block_padding=1, weight_decay=1e-6, dropout_rate=1e-5, bottleneck_type="block", ignore_N=None, transformer_window_size=None, embed_dim=None, transformer_layers=None, transformer_nhead=None, save_path=None, trial = None):
        random.seed(42)  
        # Going to have to make this explicit for the time being...
        self.label_map = {
            "J"  : 0,
            "K"  : 1,
            "L"  : 2,
            "M"  : 3,
            "N"  : 4,
            "W"  : 5
        }
        self.inv_label_map = {i:label for label, i in self.label_map.items()}

        self.data_columns = ["pre_rect"]
        
        self.batch_size = 1
        self.epochs=epochs
        self.lr=lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.block_kernel_size = block_kernel_size
        self.up_down_sample_kernel_size = up_down_sample_kernel_size
        self.block_padding = block_padding
        self.n_conv_steps_per_block = n_conv_steps_per_block
        self.num_layers = num_layers
        self.growth_factor = growth_factor
        self.features = features
        self.bottleneck_type = bottleneck_type
        self.transformer_window_size = transformer_window_size
        self.ignore_N = ignore_N
        
        
        if embed_dim is None:
            if self.bottleneck_type == "windowed_attention":
                self.embed_dim = self.features * (self.growth_factor**self.num_layers)
            elif self.bottleneck_type == "attention":
                self.embed_dim = self.features * (self.growth_factor**self.num_layers)
            else:
                self.embed_dim = None
        else:
            self.embed_dim = embed_dim

        self.transformer_layers = transformer_layers

        if transformer_nhead is None:
            if self.bottleneck_type == "windowed_attention":
                self.transformer_nhead = self.embed_dim // 32
            elif self.bottleneck_type == "attention":
                self.transformer_nhead = self.transformer_window_size // 32
            else:
                self.transformer_nhead = None
        else:
            self.transformer_nhead = transformer_nhead

        if trial:
            self.epochs = trial.suggest_categorical("epochs", [8, 16, 32, 64])
            self.lr = trial.suggest_categorical("lr", [5e-3, 5e-4, 5e-5, 5e-6])
            self.dropout_rate = trial.suggest_categorical("dropout", [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
            self.weight_decay = trial.suggest_categorical("weight_decay", [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
            self.num_layers = trial.suggest_categorical("num_layers", [2, 4, 6, 8])
            self.features = trial.suggest_categorical("features", [8, 16, 32, 64])

            # special params for attention
            if self.bottleneck_type == "windowed_attention":                
                self.transformer_window_size = trial.suggest_categorical("transformer_window_size", [50, 100, 150, 200, 250])
                self.embed_dim = self.features # force to be same size as features
                self.transformer_layers = trial.suggest_categorical("transformer_layers", [2,4,6])
                self.transformer_nhead = max(self.features // trial.suggest_categorical("heads_per_channel", [16, 32]), 1)
            else:
                self.bottleneck_type = "block"
            """
            self.growth_factor = trial.suggest_categorical("growth_factor", [1, 2])
            self.bottleneck_type = trial.suggest_categorical("bottleneck_type", ["windowed_attention", "attention", "block"])
            self.transformer_window_size = trial.suggest_categorical("transformer_window_size", [100, 150])
            self.transformer_layers = trial.suggest_categorical("transformer_layers", [1, 2, 4])
            self.transformer_nhead = trial.suggest_categorical("transformer_nhead", [1, 2, 4])
            self.features = trial.suggest_categorical("features", [16, 32])
            if self.bottleneck_type == "attention" or self.bottleneck_type == "windowed_attention":
                self.embed_dim = self.features * (self.growth_factor**self.num_layers)
            else:
                self.embed_dim = trial.suggest_categorical("embed_dim", [16, 32])
            """
        # if not a block, i.e actually used, make sure divisible 
        if self.bottleneck_type != "block":
            assert self.embed_dim % self.transformer_nhead == 0


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_classes = len(self.label_map)
            
        self.model = UNet1D(input_size=len(self.data_columns), 
                            output_size=self.num_classes,
                            growth_factor=self.growth_factor,
                            features=self.features,
                            num_layers=self.num_layers, 
                            n_conv_steps_per_block=self.n_conv_steps_per_block, 
                            dropout_rate=self.dropout_rate, 
                            block_kernel_size=self.block_kernel_size,
                            up_down_sample_kernel_size=self.up_down_sample_kernel_size,
                            block_padding=self.block_padding,
                            bottleneck_type=self.bottleneck_type, 
                            transformer_window_size=self.transformer_window_size, 
                            embed_dim=self.embed_dim, 
                            transformer_layers=self.transformer_layers, 
                            transformer_nhead=self.transformer_nhead) 

        dirname = os.path.dirname(__file__)
        self.save_path = save_path

    def train(self, probes, test_probes, fold = None, save_train_curve=False, show_train_curve=False):
        self.model = self.model.to(self.device)

        tr_dfs, tr_df = self.load_probes(probes)
        tr_dataset = TimeSeriesDataset(tr_dfs, self.label_map, data_columns=self.data_columns, 
                                       class_column = "labels", ignore_N=self.ignore_N)
        tr_dataloader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True)

        if test_probes:
            test_dfs, test_df = self.load_probes(test_probes)
            test_dataset = TimeSeriesDataset(test_dfs, self.label_map, data_columns=self.data_columns,
                                             class_column = "labels",ignore_N=self.ignore_N)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, capturable=False)

        train_losses = []
        test_losses = []
        for epoch in tqdm.tqdm(range(self.epochs)):
            self.model.train()
            running_loss = 0.0
            for batch in tr_dataloader:
                x, y, weights = batch
                x, y, weights = x.to(self.device), y.to(self.device), \
                                weights.to(self.device)

                optimizer.zero_grad()
                #print(x.shape)
                outputs = self.model(x.permute(0,2,1))
                
                loss = criterion(outputs, y)

                weighted_loss = (loss) * weights
                weighted_loss = weighted_loss.mean()
                weighted_loss.backward()
                optimizer.step()
                
                running_loss += weighted_loss.item()
            train_loss = running_loss / len(tr_dataloader)
            train_losses.append(train_loss)

            # Get the test loss
            if test_probes:
                self.model.eval()
                with torch.no_grad():
                    running_loss = 0
                    for batch in test_dataloader:
                        x, y, weights = batch
                        x, y, weights = x.to(self.device), y.to(self.device), \
                                        weights.to(self.device)
                        outputs = self.model(x.permute(0,2,1))
                        loss = criterion(outputs, y)
                        
                        weighted_loss = loss * weights
                        weighted_loss = weighted_loss.mean()
                        running_loss += weighted_loss.item()
                    test_loss = running_loss / len(test_dataloader)
                    test_losses.append(test_loss)
        if save_train_curve:
            plt.plot(train_losses, label = "Train")
            plt.plot(test_losses, label = "Test")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.savefig(f"{self.save_path}/loss_curve_fold{fold}.png")
        if show_train_curve:
            plt.plot(train_losses, label = "Train")
            plt.plot(test_losses, label = "Test")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.show()

    def predict(self, probes, preprocess = False, return_logits=False):
        test_dataset = TimeSeriesDataset(probes, self.label_map, data_columns=self.data_columns,
                                         class_column = "labels", ignore_N=self.ignore_N)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        all_predictions = []
        all_logits = []
        self.model.eval()
        with torch.no_grad():
            for probe in test_dataloader:
                x, _, _ = probe
                x = x.to(self.device)

                outputs = self.model.forward(x.permute(0,2,1))
                if return_logits:
                    all_logits.append(outputs.cpu())
                outputs = outputs.argmax(dim=1).view(-1).cpu().tolist()
                output_labels = [self.inv_label_map[x] for x in outputs]
                all_predictions.append(output_labels)
            if return_logits:
                return all_predictions, all_logits
            else:
                return all_predictions

    def load_probes(self, probes):
        big_probe = pd.concat(probes, axis=0)
        return probes, big_probe

    def save(self):
        torch.save(self.model.state_dict(), "unet_weights")

    def load(self, path = None):
        self.model = UNet1D(input_size=len(self.data_columns), 
                            output_size=self.num_classes,
                            growth_factor=self.growth_factor,
                            features=self.features,
                            num_layers=self.num_layers, 
                            n_conv_steps_per_block=self.n_conv_steps_per_block, 
                            dropout_rate=self.dropout_rate, 
                            block_kernel_size=self.block_kernel_size,
                            up_down_sample_kernel_size=self.up_down_sample_kernel_size,
                            block_padding=self.block_padding,
                            bottleneck_type=self.bottleneck_type, 
                            transformer_window_size=self.transformer_window_size, 
                            embed_dim=self.embed_dim, 
                            transformer_layers=self.transformer_layers, 
                            transformer_nhead=self.transformer_nhead) 
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location = self.device))
        self.model = self.model.to(self.device)

class TimeSeriesDataset(Dataset):
    def __init__(self, dfs, label_map, data_columns, class_column, transform=None, weight=False, ignore_N=False):
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

            if ignore_N:
                mask = (df[self.class_column] == "N").values
                weights_tensor[mask] = 0.0

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
    def __init__(self, input_size, output_size, growth_factor, features, num_layers, n_conv_steps_per_block, dropout_rate, block_kernel_size, up_down_sample_kernel_size, block_padding, bottleneck_type, transformer_window_size, embed_dim, transformer_layers, transformer_nhead):
        super(UNet1D, self).__init__()
        self.num_layers = num_layers
        self.features = features
        self.growth_factor = growth_factor
        self.bottleneck_type = bottleneck_type
        self.transformer_window_size = transformer_window_size
        self.embed_dim = embed_dim
        self.transformer_layers = transformer_layers
        self.transformer_nhead = transformer_nhead


        # input layer
        self.in_conv = nn.Conv1d(in_channels=input_size, out_channels=features, kernel_size=1)

        # Encoding layers
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(num_layers):
            encoder = EncoderBlock(features, growth_factor*features, n_conv_steps_per_block=n_conv_steps_per_block, dropout_rate=dropout_rate, block_kernel_size=block_kernel_size, block_padding=block_padding, growth_factor=growth_factor)
            self.encoders.append(encoder)
            self.pools.append(nn.MaxPool1d(kernel_size=up_down_sample_kernel_size, stride=2))
            features *= growth_factor  # Increase feature size

        # Bottleneck
        if self.bottleneck_type == "block":
            # in the bottleneck, we don't do any growth and just map features-->features
            self.bottleneck = EncoderBlock(features, features, n_conv_steps_per_block=n_conv_steps_per_block, dropout_rate=dropout_rate, block_kernel_size=block_kernel_size, block_padding=block_padding, growth_factor=1)
        elif self.bottleneck_type == "windowed_attention":
            self.bottleneck = WindowedTransformerBotleneck(features, self.transformer_window_size, self.embed_dim, self.transformer_layers, self.transformer_nhead)
        elif self.bottleneck_type == "attention":
            self.bottleneck = TransformerBotleneck(self.embed_dim, self.transformer_layers, self.transformer_nhead)
        else:
            assert False
        

        # Decoding layers
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            upconv = nn.ConvTranspose1d(features, features, kernel_size=up_down_sample_kernel_size, stride=up_down_sample_kernel_size)
            self.upconvs.append(upconv)
            decoder = DecoderBlock(features, features // growth_factor, n_conv_steps_per_block=n_conv_steps_per_block, dropout_rate=dropout_rate, block_kernel_size=block_kernel_size, block_padding=block_padding, growth_factor=growth_factor)
            self.decoders.append(decoder)
            features //= growth_factor  # Decrease feature size

        # output layer
        self.out_conv = nn.Conv1d(in_channels=features, out_channels=output_size, kernel_size=1)

    def forward(self, x):
        # Encoding path
        encodings = []
        paddings = []
        x = self.in_conv(x)

        initial_size = x.shape[2]

        for i in range(self.num_layers):
            x = self.encoders[i](x)
            # if growth factor is not 1, then downconv
            x = self.pools[i](x)
            encodings.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoding path
        for i in range(self.num_layers):
            prev_encoding = encodings[-(i+1)]
            x = pad_or_crop(x, dim=2, target_size=prev_encoding.shape[2])
            x = torch.add(x, prev_encoding)
            x = self.upconvs[i](x)
            x = self.decoders[i](x)

        # pad up to the inital size
        x = pad_or_crop(x, dim=2, target_size=initial_size)
        x = self.out_conv(x)
        return x
    
    
    def predict_batch(self, batch, device):
        self.model.eval()
        with torch.no_grad():
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
        self.model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                masked_preds, masked_labels = self.predict_batch(batch, device=device)

                all_preds.extend(masked_preds)
                all_labels.extend(masked_labels)

        return all_labels, all_preds

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv_steps_per_block, dropout_rate, block_kernel_size, block_padding, growth_factor):
        """
        Defines a single encoding block consisting of nX: Conv1d, InstanceNorm1d, GELU, dropout.
        """
        super(EncoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.growth_factor = growth_factor
        self.block = nn.Sequential()

        self.block.append(nn.Conv1d(in_channels, out_channels, kernel_size=block_kernel_size, padding=block_padding))
        self.block.append(nn.InstanceNorm1d(out_channels))
        self.block.append(nn.GELU())
        self.block.append(nn.Dropout(p=dropout_rate))
        
        for _ in range(n_conv_steps_per_block-1):
            self.block.append(nn.Conv1d(out_channels, out_channels, kernel_size=block_kernel_size, padding=block_padding))
            self.block.append(nn.InstanceNorm1d(out_channels))
            self.block.append(nn.GELU())
            self.block.append(nn.Dropout(p=dropout_rate))

    def forward(self, x):
        if self.growth_factor == 1:
            return self.block(x) + x # add skip connection
        elif self.growth_factor == 2:
            return self.block(x)

    def __repr__(self):
        return f"EncoderBlock({self.in_channels}, {self.out_channels})" + super(EncoderBlock, self).__repr__()[12:]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_conv_steps_per_block, dropout_rate, block_kernel_size, block_padding, growth_factor):
        """
        Defines a single decoding block consisting of nX: Conv1d, InstanceNorm1d, and GELU.
        """
        super(DecoderBlock, self).__init__()
        self.growth_factor = growth_factor
        self.block = nn.Sequential()

        self.block.append(nn.Conv1d(in_channels, out_channels, kernel_size=block_kernel_size, padding=block_padding))
        self.block.append(nn.InstanceNorm1d(out_channels))
        self.block.append(nn.GELU())
        for _ in range(n_conv_steps_per_block-1):
            self.block.append(nn.Conv1d(out_channels, out_channels, kernel_size=block_kernel_size, padding=block_padding))
            self.block.append(nn.InstanceNorm1d(out_channels))
            self.block.append(nn.GELU())
            
    def forward(self, x):
        if self.growth_factor == 1:
            return self.block(x) + x # add skip connection
        elif self.growth_factor == 2:
            return self.block(x)

import torch.nn.functional as F

class WindowedTransformerBotleneck(nn.Module):
    def __init__(self, in_channels, window_size, embed_dim, transformer_layers, nhead):
        super(WindowedTransformerBotleneck, self).__init__()
        self.stride = window_size  # replace with stride for overlapping...
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        if self.in_channels != self.embed_dim:
            print("WARNING: USING EMBEDDING LAYERS IN TRANSFORMER")
            self.linear_embed = nn.Linear(self.in_channels, self.embed_dim)
            self.linear_decode = nn.Linear(self.embed_dim, self.in_channels)

        self.positional_encoder = PositionalEncoding1D(self.embed_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=transformer_layers)
    
    def forward(self, x):
        batch, channels, seq_len = x.shape

        # --- Padding ---
        # Calculate the amount of padding needed to make seq_len a multiple of window_size
        pad_len = (self.window_size - (seq_len % self.window_size)) % self.window_size
        if pad_len > 0:
            # Pad on the right side of the sequence dimension
            x = F.pad(x, (0, pad_len))
        new_seq_len = x.shape[-1]  # = seq_len + pad_len

        # --- Windowing ---
        # x has shape: (batch, channels, new_seq_len)
        # Use unfold on the last dimension to create windows of size window_size
        # The resulting shape is (batch, channels, num_windows, window_size)
        x_windowed = x.unfold(dimension=2, size=self.window_size, step=self.stride)
        batch, channels, num_windows, window_size = x_windowed.shape
        
        # Rearrange dimensions to prepare for transformer encoding:
        # We want shape: (window_size, batch*num_windows, channels)
        x_windowed = x_windowed.permute(3, 0, 2, 1).reshape(window_size, batch * num_windows, channels)
        
        if self.in_channels != self.embed_dim:
            x_windowed = torch.vmap(self.linear_embed)(x_windowed)

        # --- Positional Encoding ---
        x_windowed = x_windowed + self.positional_encoder(x_windowed)
        
        # --- Transformer Encoding ---
        #print("TRANSFORMER", x_windowed.shape)
        encoded = self.transformer_encoder(src=x_windowed)
 
        if self.in_channels != self.embed_dim:
            encoded = torch.vmap(self.linear_decode)(encoded)
        # --- Reassemble ---
        encoded = encoded.reshape(window_size, batch, num_windows, channels).permute(1, 3, 2, 0)
        # Merge windows along the sequence dimension: (batch, channels, num_windows * window_size)
        output = encoded.reshape(batch, channels, -1)
        
        # Remove extra padded positions to recover the original sequence length
        if pad_len > 0:
            output = output[:, :, :seq_len]
        return output

    
class TransformerBotleneck(nn.Module):
    def __init__(self, embed_dim, transformer_layers, nhead):
        super(TransformerBotleneck, self).__init__()
        self.embed_dim = embed_dim

        self.positional_encoder = PositionalEncoding1D(self.embed_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model = self.embed_dim, nhead = nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers = transformer_layers)
    
    def forward(self, x):
        # Apply a positional embedding
        x = x.permute(2, 0, 1) # seq_len, batch, channels
        x_positional_encoded = x + self.positional_encoder(x)
        # Transformer Encoding
        encoded = self.transformer_encoder(src = x_positional_encoded)
        encoded = encoded.permute(1, 2, 0)
        return encoded
    
# NOTE: no longer in use
class MaxPool1dWithOddInputHandling(nn.Module):
    def __init__(self, pool_kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        super(MaxPool1dWithOddInputHandling, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode)

    def forward(self, x):
        # Check if the sequence length (dimension 2) is odd
        if x.size(2) % 2 != 0:
            # If odd, pad by 1 on the right side
            x = F.pad(x, (0, 1))
        # Apply the MaxPool1d operation
        return self.pool(x)
