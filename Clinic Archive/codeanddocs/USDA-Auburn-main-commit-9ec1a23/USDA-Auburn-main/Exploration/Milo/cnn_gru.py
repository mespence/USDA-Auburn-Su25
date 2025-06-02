# %%
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy
import random

def mode(s):
    mode = scipy.stats.mode(s).mode
    return mode

window = 3
pad = window / 10
sampling_rate = 100
batch_size = 64

def data_splitter(df, window, step):
    return [df.iloc[i:i + window] for i in range(0, len(df) - window + 1, step)]

label_encoder = LabelEncoder()
int_labels = label_encoder.fit(big_df["labels"])



def build_dataloader(subset_files):
    df = big_df[big_df["file"].isin(subset_files)]

    split = data_splitter(df, int(100*window), int(100*pad))
    X = [split[i]["pre_rect"] for i in range(len(split))]
    y = [set_window_class(int_labels.transform(split[i]["labels"])) for i in range(len(split))]

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

tr_size = int(len(dfs)*0.8)
test_size = len(dfs) - tr_size

random.seed(42)
tr_files = set(random.sample(files, tr_size))
test_files = set(files)-set(tr_files)
assert test_size == len(test_files)

tr_dataloader = build_dataloader(tr_files)
test_dataloader = build_dataloader(test_files)

# %%

class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=2, pool_stride=2, activation='relu'):
        super(Conv1DBlock, self).__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation function")
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

class CNN_GRU_Model(nn.Module):
    def __init__(self, input_channels, channels, kernel_sizes, pool_sizes, padding_sizes, gru_hidden_size, num_classes, ):
        super(CNN_GRU_Model, self).__init__()
        
        self.conv_blocks = []
        for in_channel, out_channel, kernel_size, pool_size, padding in zip([input_channels] + channels, channels, kernel_sizes, pool_sizes, padding_sizes):
            self.conv_blocks.append(Conv1DBlock(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, pool_size=pool_size, padding=padding))

        self.gru = nn.GRU(input_size=channels[-1], hidden_size=gru_hidden_size, batch_first=True)
        self.fc = nn.Linear(gru_hidden_size, num_classes)

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = x.permute(0, 2, 1)
        _, h_n = self.gru(x)  # h_n is the final hidden state
        out = self.fc(h_n.squeeze(0))  # Output shape: (batch_size, num_classes)
        return out
        


model = CNN_GRU_Model(
    input_channels=1,
    channels=[16, 32, 64],
    kernel_sizes=[3, 3, 3],
    pool_sizes=[2, 2, 2],
    padding_sizes=[1, 1, 1],
    gru_hidden_size=64,
    num_classes=7
)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
accs = []

epochs = 10
for epoch in range(epochs):
    print(epoch)
    tot_loss = 0
    for batch_X, batch_y in tr_dataloader:
        batch_X = batch_X.unsqueeze(1)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
    

    acc = evaluate(model, test_dataloader)
    accs.append(acc)
    losses.append(tot_loss)
    print(losses[-1], acc)