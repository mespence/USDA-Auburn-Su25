# %%
import pandas as pd
import numpy as np
import os


dfs = []
files = []
root = "tarsalis_data_clean"
for file in os.listdir(root):
    df = pd.read_csv(os.path.join(root,file)).drop(columns=["Unnamed: 0"])
    df["labels"] = df['labels'].str.upper().replace("Z","W")
    df['previous_label'] = df['labels'].shift(1)
    df["file"] = file
    files.append(file)
    dfs.append(df.dropna().reset_index())

big_df = pd.concat(dfs, axis=0)

# %%
s = pd.concat([df.query("previous_label!='NP' & labels=='NP'").dropna() for df in dfs],  axis=0)
# %%
df = big_df.copy()
df["labels"] = df["labels"].str.upper()

import matplotlib.pyplot as plt
import networkx as nx

# J->L
# L->NP

combined_transition_counts = pd.DataFrame(columns=['previous_label', 'labels', 'count'])
for df in dfs:
    # Step 1: Calculate transition counts for each dataframe
    transitions = df[df['labels'] != df['previous_label']]
    transition_counts = transitions.groupby(['previous_label', 'labels']).size().reset_index(name='count')
    
    # Step 2: Append the transition counts to the combined dataframe
    combined_transition_counts = pd.concat([combined_transition_counts, transition_counts])

# Step 3: Group by 'previous_label' and 'labels' to sum the counts across all dataframes
combined_transition_counts = combined_transition_counts.groupby(['previous_label', 'labels']).sum().reset_index()

# Step 4: Normalize the combined transition matrix to get probabilities
transition_sums = combined_transition_counts.groupby('previous_label')['count'].transform('sum')
combined_transition_counts['probability'] = combined_transition_counts['count'] / transition_sums

transition_counts = combined_transition_counts

# Create a directed graph
G = nx.DiGraph()

# Add edges to the graph
for _, row in transition_counts.iterrows():
    G.add_edge(row['previous_label'], row['labels'], weight=row['probability'])

# Set node positions for visualization
pos = nx.spring_layout(G)

# Draw the graph
plt.figure(figsize=(8, 6))

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, width=2, edge_color='gray')
edge_labels = {(row['previous_label'], row['labels']): round(row['probability']*100,1) for _, row in transition_counts.iterrows()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

plt.title("Label Transitions Graph")
plt.axis('off')  # Hide axis
plt.show()

print(transition_counts)
print(big_df['labels'].value_counts())


# %%
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_tcn import TCN
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
#s = pd.Series([set_window_class(int_labels.transform(split[i]["labels"])) for i in range(len(split))])
#{label_encoder.inverse_transform([k])[0]:v/len(s) for k,v in s.value_counts().items()}

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