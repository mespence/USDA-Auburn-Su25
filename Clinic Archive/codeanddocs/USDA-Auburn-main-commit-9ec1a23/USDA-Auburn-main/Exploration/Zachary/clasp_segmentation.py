import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import glob
import os
from collections import defaultdict
import argparse
from tqdm import tqdm

from claspy.segmentation import BinaryClaSPSegmentation
from claspy.clasp import ClaSP
from claspy.window_size import suss

def import_data(data_path):
    """
    import_data takes in a path to cleaned data and returns it
    as a list of dataframes.
    """
    filenames = glob.glob(os.path.expanduser(f"{data_path}/*.csv"))
    dataframes = [pd.read_csv(f) for f in filenames]
    return filenames, dataframes

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

def find_transitions(filenames, dataframes):
    """
    find_transitions takes in the data, uses probe_finder to perform a 
    segmentation into probes, and then uses ClaSP to locate likely 
    transition points within each probe. It returns the dataframe with 
    likely transition points denoted in the column 'transition'.
    """
    out_dataframes = []
    for file, df in tqdm(zip(filenames, dataframes)):
        name = file[file.rfind("\\") + 1:]
        probes = probe_finder(df["pre_rect"].values)
        out_dataframes.append(df)
        out_dataframes[-1]["profile"] = 0
        for start, end in probes:
            ts = df["post_rect"][start:end].values
            clasp = ClaSP(window_size = suss(ts), n_jobs = 1)
            # Here I assume that profile is end-padded which I did based
            # off of how the clasp plot function works.
            profile = clasp.fit_transform(ts)
            profile = np.clip(profile, a_min = 0)
            out_dataframes[-1]["profile"][start:start + len(profile)] = profile
        out_dataframes[-1].to_csv(f"clasp_transitions/{name}")
    return out_dataframes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path",
                        type = str)
    args = parser.parse_args()
    
    print("Importing Data")
    filenames, data = import_data(args.data_path)
    print("Segmenting...")
    data_transitions = find_transitions(filenames, data)

if __name__ == "__main__":
    main()
