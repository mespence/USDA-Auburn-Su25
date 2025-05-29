import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

def plot_labeled_time_series_for_column(data, column):
    unique_labels = data['labels'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))  # Use a colormap for consistent coloring
    label_color_map = dict(zip(unique_labels, colors))

    # Identify segments where labels are consistent
    data['segment'] = (data['labels'] != data['labels'].shift()).cumsum()

    # Plotting
    plt.figure(figsize=(12, 6))

    for segment, segment_data in data.groupby('segment'):
        state = segment_data['labels'].iloc[0]
        color = label_color_map[state]
        plt.plot(segment_data['time'], segment_data[column], label=state if segment not in plt.gca().get_legend_handles_labels()[1] else None, color=color, linewidth=2)

    plt.legend()
    plt.show()

def plot_timeseries_from_labeled_data(data, column):
    time = data['time']

    label_encoder = LabelEncoder()
    labels = np.concatenate([data['labels'].to_numpy().reshape(-1, 1)])
    labels_encoded = label_encoder.fit_transform(labels)
    n_labels = len(set(labels_encoded))
    print("Number of labels: ", n_labels)

    labels_encoded = labels_encoded.reshape(-1,1)

    fig, ax = plt.subplot()

    ax.plot(time ,labels_encoded)
    ax.plot(data['time'], data[column])
    fig.show()

def plot_states_with_value(states, value, value_label, values=0):
    fig, axs = plt.subplots(2, 1, figsize=(10,6))
    ax1 = axs[0]
    ax2 = axs[1]
    
    ax1.set_ylabel('hidden state')
    if (values>0):
        ax1.plot(states[0:values], label="state", color='tab:green')
    else:
        ax1.plot(states, label='states', color='tab:green')

    ax2.set_ylabel(value_label)
    if (values>0):
        ax2.plot(value[value_label][0:values], label=value_label, color='tab:red')
    else:
        ax2.plot(value[value_label], label=value_label, color='tab:red')
    fig.tight_layout()
    fig.legend()
    plt.show()

def plot_states_with_labels(states, data, column, values = 0):
    fig, axs = plt.subplots(2, 1, figsize=(10,6))
    ax1 = axs[0]
    ax2 = axs[1]
    
    ax1.set_ylabel('hidden state')
    if (values>0):
        ax1.plot(states[0:values], label="state", color='tab:green')
    else:
        ax1.plot(states, label='states', color='tab:green')
    
    ## Plots the labels
    
    unique_labels = data['labels'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))  # Use a colormap for consistent coloring
    label_color_map = dict(zip(unique_labels, colors))

    # Identify segments where labels are consistent
    data['segment'] = (data['labels'] != data['labels'].shift()).cumsum()

    for segment, segment_data in data.groupby('segment'):
        state = segment_data['labels'].iloc[0]
        color = label_color_map[state]
        ax2.plot(segment_data['time'], segment_data[column], label=state if segment not in plt.gca().get_legend_handles_labels()[1] else None, color=color, linewidth=2)

    plt.legend()
    plt.show()

label_to_color = {
    "NP": "red",
    "J": "blue",
    "K": "green",
    "L": "purple",
    "M": "pink",
    "N": "cyan",
    "W": "orange"
}

def plot_labels(data, true_labels, pred_labels):
    """
    plot_labels produced a matplotlib figure containing three subplots
        that visualize a waveform along with the true and predicted labels
    Input:
        time: a series of time values
        post_rect: a time series of post_rect values from the waveform
        true_labels: a time series of the true label for each time point
        pred_labels: a time series of the predicted labels for each time point
    Output:
        (fig, axs): a tuple
    """
    fig, axs = plt.subplots(3, 1, sharex = True)
    recording = 1
    fill_min, fill_max = min(data), max(data)
    
    # First plot will be the true labels
    axs[0].plot(data, color = "black")
    for label, color in label_to_color.items():
        fill = axs[0].fill_between(np.arange(len(data)), fill_min, fill_max, 
                where = (true_labels == label), color=color, alpha = 0.5)
        fill.set_label(label)
    axs[0].legend(bbox_to_anchor=(0.5, 1), 
                  bbox_transform=fig.transFigure, loc="upper center", ncol=9)
    axs[0].set_title("True Labels")
    # Second plot will be the predicted labels
    axs[1].plot(data, color = "black")
    for label, color in label_to_color.items():
        axs[1].fill_between(range(len(data)), fill_min, fill_max, 
                where = (pred_labels == label), color=color, alpha = 0.5)
    axs[1].set_title("Predicted Labels")
    # Third plot will be marked where there is a difference between the two
    axs[2].plot(data, color = "black")
    axs[2].fill_between(range(len(data)), fill_min, fill_max, 
            where = (pred_labels != true_labels[:len(pred_labels)]), color = "gray", alpha = 0.5)
    axs[2].set_title("Incorrect Labels")
        # Axes titles and such
    fig.supylabel("Post rect")
    fig.tight_layout()
    return fig
