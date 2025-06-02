import pandas as pd
import numpy as np
import os
import glob
from typing import List
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from sklearn.utils import check_random_state

def load_files(files, path):
    dfs = []
    data_dir = os.path.abspath(path)
    
    for file in files:
        df = pd.read_csv(os.path.join(data_dir,file)).drop(columns=["Unnamed: 0"])
        df["labels"] = df['labels'].str.upper().replace("Z","W")
        df['previous_label'] = df['labels'].shift(1)
        df["file"] = file
        dfs.append(df.dropna().reset_index())

    big_df = pd.concat(dfs, axis=0)
    return big_df, dfs

def get_files(path: str):
    data_dir = os.path.abspath(path)
    filenames = glob.glob(data_dir + '/*.csv')
    return filenames

def preprocess_timeseries(dataframes, feature_list):
    concatted_dataset = pd.concat(dataframes, ignore_index=True)
    features = concatted_dataset[feature_list]

    scaler = StandardScaler()

    scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=feature_list)

    lengths = [len(df) for df in dataframes]

    return scaled_features, lengths, concatted_dataset

# Takes training data and an estimation of states, returns a trained model
def train_gaussian_HMM(n_states: int, scaled_features, lengths, covariance_type, n_iter, random_state):
    model = hmm.GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=n_iter, random_state=random_state)
    model.fit(scaled_features, lengths)
    return model

def predict_with_model(model, features):
    hidden_states = model.predict(features)
    state_probs = model.predict_proba(features)
    return hidden_states, state_probs

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

def split_into_probes(dfs, window = 500, threshold = 0.1,
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

class HMMTimeSeriesData:
    def __init__(self, dataframes, labels=['resistance', 'voltage', 'pre_rect', 'post_rect'], split_by_probes=False):
        if not split_by_probes:
            self.data = dataframes
        else:
            self.data, self.probe_names = split_into_probes(dataframes)
        self.scaled_features, self.lengths, self.concatted_dataset = preprocess_timeseries(self.data, labels)

    def GetData(self):
        return self.data
    
    def GetLengths(self):
        return self.lengths
"""
class Model:
    model: GaussianHMM
    type: str
    covariance_type: str

    def __init__(self, type, covariance):
        self.type = "Gaussian"
        self.covariance_type = covariance

    def train(self, data, lengths, components, n_iter):
        self.model = train_gaussian_HMM(components, data, lengths, self.covariance_type, n_iter, check_random_state(None))

    def getModel(self) -> GaussianHMM:
        return self.model
    
    def converged(self) -> bool:
        self.model
        """